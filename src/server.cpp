// Neural Proof-of-Work Verification Server - OpenCL Only
// All crypto/neural ops done on GPU - no CPU implementations

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "httplib.h"
#include "json.hpp"
#include "kernel_embedded.h"
#include "config.h"

using json = nlohmann::json;

// ============================================================================
// Tokenomics
// ============================================================================
//
// Hardware benchmark: 4x RTX 5090 @ ~27 MH/s for ~$1.15/hr (vast.ai)
// Observed: 32 bits in ~220s, 34 bits in ~362s
//
// Difficulty scaling (each +1 bit = 2x harder):
//   32 bits: 2^32 / 27M = ~160 sec → ~22 proofs/hr → ~$0.05/proof
//   40 bits: 2^40 / 27M = ~11 hrs  → ~$13/proof
//   48 bits: 2^48 / 27M = ~121 days → ~$3,300/proof
//
// Reward formula: 1.5^(bits - MIN_BITS)
//   32 bits = 1.0 tokens      (baseline, ~$0.05 cost)
//   40 bits = 25.6 tokens     (~$0.51/token cost)
//   48 bits = 656.8 tokens    (~$5.03/token cost)
//
// Using 1.5x per bit (not 2x) so rewards grow slower than difficulty,
// making higher bits progressively more valuable per token.
//
constexpr int MIN_BITS = 32;
constexpr int SERVER_PORT = 8080;
constexpr const char* PROOFS_FILE = "proofs.jsonl";
constexpr const char* WEIGHT_EPOCH = "epoch0";

// ============================================================================
// Utility Functions
// ============================================================================

std::string bytes_to_hex(const uint8_t* data, size_t len) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; i++) {
        ss << std::setw(2) << static_cast<int>(data[i]);
    }
    return ss.str();
}

int count_leading_zero_bits(const uint8_t* digest, size_t len) {
    int bits = 0;
    for (size_t i = 0; i < len; i++) {
        uint8_t b = digest[i];
        if (b == 0) { bits += 8; continue; }
        while ((b & 0x80u) == 0) { bits++; b <<= 1; }
        break;
    }
    return bits;
}

// Reward grows 1.5x per bit (slower than 2x difficulty growth)
// See tokenomics comment above for cost/reward analysis
double compute_reward(int bits) {
    if (bits < MIN_BITS) return 0.0;
    return std::pow(1.5, bits - MIN_BITS);
}

// ============================================================================
// OpenCL Context
// ============================================================================

struct OpenCLContext {
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel verify_kernel;
    cl_mem weights_buf;
    std::vector<int8_t> weights;
};

bool init_opencl(OpenCLContext& cl) {
    cl_int err;

    // Find GPU
    cl_uint num_platforms;
    if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
        std::cerr << "No OpenCL platforms found" << std::endl;
        return false;
    }
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);

    cl.device = nullptr;
    for (cl_platform_id platform : platforms) {
        cl_uint num_devices;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &cl.device, &num_devices) == CL_SUCCESS && num_devices > 0) {
            break;
        }
    }
    if (!cl.device) {
        std::cerr << "No GPU found" << std::endl;
        return false;
    }

    char name[256];
    clGetDeviceInfo(cl.device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    std::cout << "Using GPU: " << name << std::endl;

    cl.context = clCreateContext(nullptr, 1, &cl.device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) return false;

#ifdef CL_VERSION_2_0
    cl.queue = clCreateCommandQueueWithProperties(cl.context, cl.device, nullptr, &err);
#else
    cl.queue = clCreateCommandQueue(cl.context, cl.device, 0, &err);
#endif
    if (err != CL_SUCCESS) return false;

    const char* src = reinterpret_cast<const char*>(kernel_cl);
    size_t srcLen = kernel_cl_len;
    cl.program = clCreateProgramWithSource(cl.context, 1, &src, &srcLen, &err);
    if (err != CL_SUCCESS) return false;

    // HASHES_PER_THREAD needed for mining kernel (even though server doesn't use it)
    err = clBuildProgram(cl.program, 1, &cl.device, "-D HASHES_PER_THREAD=1 -cl-fp32-correctly-rounded-divide-sqrt", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char log[16384];
        clGetProgramBuildInfo(cl.program, cl.device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::cerr << "Build error: " << log << std::endl;
        return false;
    }

    cl.verify_kernel = clCreateKernel(cl.program, "neural_pow_verify", &err);
    if (err != CL_SUCCESS) return false;

    return true;
}

bool generate_weights_gpu(OpenCLContext& cl) {
    cl_int err;

    cl_kernel gen_kernel = clCreateKernel(cl.program, "generate_weights_kernel", &err);
    if (err != CL_SUCCESS) return false;

    std::string epoch = WEIGHT_EPOCH;
    cl_mem epoch_buf = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       epoch.size(), (void*)epoch.data(), &err);
    cl_mem weights_buf = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, TOTAL_WEIGHTS, nullptr, &err);
    if (err != CL_SUCCESS) {
        clReleaseKernel(gen_kernel);
        return false;
    }

    cl_uint epoch_len = static_cast<cl_uint>(epoch.size());
    clSetKernelArg(gen_kernel, 0, sizeof(cl_mem), &epoch_buf);
    clSetKernelArg(gen_kernel, 1, sizeof(cl_uint), &epoch_len);
    clSetKernelArg(gen_kernel, 2, sizeof(cl_mem), &weights_buf);

    size_t global_size = (TOTAL_WEIGHTS + 31) / 32;
    err = clEnqueueNDRangeKernel(cl.queue, gen_kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(epoch_buf);
        clReleaseMemObject(weights_buf);
        clReleaseKernel(gen_kernel);
        return false;
    }

    cl.weights.resize(TOTAL_WEIGHTS);
    err = clEnqueueReadBuffer(cl.queue, weights_buf, CL_TRUE, 0, TOTAL_WEIGHTS, cl.weights.data(), 0, nullptr, nullptr);

    clReleaseMemObject(epoch_buf);
    clReleaseMemObject(weights_buf);
    clReleaseKernel(gen_kernel);

    if (err != CL_SUCCESS) return false;

    // Create persistent weights buffer for verification
    cl.weights_buf = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     cl.weights.size(), cl.weights.data(), &err);
    return err == CL_SUCCESS;
}

struct VerifyResult {
    uint64_t score;
    std::vector<uint8_t> digest;
    int bits;
};

bool verify_nonce_gpu(OpenCLContext& cl, const std::string& wallet, const std::vector<uint8_t>& nonce, VerifyResult& result) {
    cl_int err;

    cl_mem wallet_buf = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                        wallet.size(), (void*)wallet.data(), &err);
    cl_mem nonce_buf = clCreateBuffer(cl.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       nonce.size(), (void*)nonce.data(), &err);
    cl_mem score_buf = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, sizeof(cl_ulong), nullptr, &err);
    cl_mem digest_buf = clCreateBuffer(cl.context, CL_MEM_WRITE_ONLY, DIGEST_BYTES, nullptr, &err);
    if (err != CL_SUCCESS) return false;

    // Set args: weights, wallet, wallet_len, nonce, nonce_len, score, digest
    cl_uint wallet_len = static_cast<cl_uint>(wallet.size());
    cl_uint nonce_len = static_cast<cl_uint>(nonce.size());
    clSetKernelArg(cl.verify_kernel, 0, sizeof(cl_mem), &cl.weights_buf);
    clSetKernelArg(cl.verify_kernel, 1, sizeof(cl_mem), &wallet_buf);
    clSetKernelArg(cl.verify_kernel, 2, sizeof(cl_uint), &wallet_len);
    clSetKernelArg(cl.verify_kernel, 3, sizeof(cl_mem), &nonce_buf);
    clSetKernelArg(cl.verify_kernel, 4, sizeof(cl_uint), &nonce_len);
    clSetKernelArg(cl.verify_kernel, 5, sizeof(cl_mem), &score_buf);
    clSetKernelArg(cl.verify_kernel, 6, sizeof(cl_mem), &digest_buf);

    size_t global_size = 1;
    err = clEnqueueNDRangeKernel(cl.queue, cl.verify_kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(wallet_buf);
        clReleaseMemObject(nonce_buf);
        clReleaseMemObject(score_buf);
        clReleaseMemObject(digest_buf);
        return false;
    }

    result.digest.resize(DIGEST_BYTES);
    clEnqueueReadBuffer(cl.queue, score_buf, CL_TRUE, 0, sizeof(cl_ulong), &result.score, 0, nullptr, nullptr);
    clEnqueueReadBuffer(cl.queue, digest_buf, CL_TRUE, 0, DIGEST_BYTES, result.digest.data(), 0, nullptr, nullptr);
    result.bits = count_leading_zero_bits(result.digest.data(), DIGEST_BYTES);

    clReleaseMemObject(wallet_buf);
    clReleaseMemObject(nonce_buf);
    clReleaseMemObject(score_buf);
    clReleaseMemObject(digest_buf);

    return true;
}

// ============================================================================
// Server State
// ============================================================================

struct ProofRecord {
    int64_t timestamp;
    std::string wallet;
    std::string nonce;
    int bits;
    double reward;
    std::string digest;

    json to_json() const {
        return {
            {"timestamp", timestamp},
            {"wallet", wallet},
            {"nonce", nonce},
            {"bits", bits},
            {"reward", reward},
            {"digest", digest}
        };
    }

    static ProofRecord from_json(const json& j) {
        return {
            j["timestamp"].get<int64_t>(),
            j["wallet"].get<std::string>(),
            j["nonce"].get<std::string>(),
            j["bits"].get<int>(),
            j["reward"].get<double>(),
            j["digest"].get<std::string>()
        };
    }
};

struct ServerState {
    OpenCLContext cl;
    std::unordered_set<std::string> seen_nonces;  // O(1) duplicate check (derived from proofs)
    std::vector<ProofRecord> proofs;
    std::mutex mutex;
    std::ofstream proof_log;
    uint64_t total_rejected = 0;
};

// ============================================================================
// Signal Handling
// ============================================================================

httplib::Server* g_server = nullptr;

void signal_handler(int) {
    if (g_server) {
        g_server->stop();
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    int port = SERVER_PORT;
    if (argc > 1) {
        port = std::atoi(argv[1]);
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    ServerState state;

    std::cout << "Initializing OpenCL..." << std::endl;
    if (!init_opencl(state.cl)) {
        std::cerr << "Failed to initialize OpenCL" << std::endl;
        return 1;
    }

    std::cout << "Generating weights on GPU from epoch: " << WEIGHT_EPOCH << std::endl;
    if (!generate_weights_gpu(state.cl)) {
        std::cerr << "Failed to generate weights" << std::endl;
        return 1;
    }
    std::cout << "Generated " << state.cl.weights.size() << " weight bytes" << std::endl;

    // Load existing proofs from file
    {
        std::ifstream in(PROOFS_FILE);
        if (in.is_open()) {
            std::string line;
            while (std::getline(in, line)) {
                if (line.empty()) continue;
                try {
                    ProofRecord p = ProofRecord::from_json(json::parse(line));
                    state.seen_nonces.insert(p.nonce);
                    state.proofs.push_back(p);
                } catch (...) {
                    // Skip malformed lines
                }
            }
            std::cout << "Loaded " << state.proofs.size() << " existing proofs from " << PROOFS_FILE << std::endl;
        }
    }

    state.proof_log.open(PROOFS_FILE, std::ios::app);
    if (!state.proof_log.is_open()) {
        std::cerr << "Failed to open " << PROOFS_FILE << std::endl;
        return 1;
    }

    httplib::Server svr;

    // Health check
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    // Stats
    svr.Get("/stats", [&state](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(state.mutex);
        json j;
        j["total_proofs"] = state.proofs.size();
        j["total_rejected"] = state.total_rejected;
        j["min_bits"] = MIN_BITS;
        res.set_content(j.dump(), "application/json");
    });

    // Proofs (sorted by bits descending)
    svr.Get("/proofs", [&state](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(state.mutex);

        // Copy and sort by bits descending
        std::vector<ProofRecord> sorted = state.proofs;
        std::sort(sorted.begin(), sorted.end(), [](const ProofRecord& a, const ProofRecord& b) {
            return a.bits > b.bits;
        });

        json arr = json::array();
        for (const auto& p : sorted) {
            arr.push_back(p.to_json());
        }
        res.set_content(arr.dump(), "application/json");
    });

    // Submit proof
    svr.Post("/submit", [&state](const httplib::Request& req, httplib::Response& res) {
        json response;

        try {
            json body = json::parse(req.body);

            if (!body.contains("wallet") || !body.contains("nonce")) {
                response["error"] = "missing wallet or nonce";
                res.status = 400;
                res.set_content(response.dump(), "application/json");
                return;
            }

            std::string wallet = body["wallet"];
            std::string nonce_str = body["nonce"];
            std::vector<uint8_t> nonce(nonce_str.begin(), nonce_str.end());

            // Hold lock for entire verification + storage to prevent TOCTOU race
            std::lock_guard<std::mutex> lock(state.mutex);

            // Check for duplicate
            if (state.seen_nonces.count(nonce_str)) {
                response["error"] = "duplicate nonce";
                state.total_rejected++;
                res.status = 400;
                res.set_content(response.dump(), "application/json");
                return;
            }

            // Verify proof on GPU
            VerifyResult vr;
            if (!verify_nonce_gpu(state.cl, wallet, nonce, vr)) {
                response["error"] = "verification failed";
                state.total_rejected++;
                res.status = 500;
                res.set_content(response.dump(), "application/json");
                return;
            }

            if (vr.bits < MIN_BITS) {
                response["error"] = "difficulty too low";
                response["bits"] = vr.bits;
                response["min_bits"] = MIN_BITS;
                state.total_rejected++;
                res.status = 400;
                res.set_content(response.dump(), "application/json");
                return;
            }

            // Accept proof
            auto now = std::chrono::system_clock::now();
            ProofRecord proof = {
                std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count(),
                wallet,
                nonce_str,
                vr.bits,
                compute_reward(vr.bits),
                bytes_to_hex(vr.digest.data(), DIGEST_BYTES)
            };

            state.seen_nonces.insert(proof.nonce);
            state.proofs.push_back(proof);
            state.proof_log << proof.to_json().dump() << std::endl;
            state.proof_log.flush();

            std::cout << "[Proof] " << proof.bits << " bits | wallet=" << proof.wallet << " | reward=" << proof.reward << std::endl;

            response["success"] = true;
            response["bits"] = proof.bits;
            response["reward"] = proof.reward;
            response["digest"] = proof.digest;
            res.set_content(response.dump(), "application/json");

        } catch (const std::exception& e) {
            response["error"] = e.what();
            res.status = 400;
            res.set_content(response.dump(), "application/json");
        }
    });

    std::cout << "Server starting on port " << port << std::endl;
    std::cout << "MIN_BITS: " << MIN_BITS << std::endl;
    std::cout << "Endpoints:" << std::endl;
    std::cout << "  GET  /health - Health check" << std::endl;
    std::cout << "  GET  /stats  - Server statistics" << std::endl;
    std::cout << "  GET  /proofs - All proofs (sorted by bits)" << std::endl;
    std::cout << "  POST /submit - Submit proof {wallet, nonce}" << std::endl;
    std::cout << "Press Ctrl+C to stop." << std::endl;

    g_server = &svr;
    svr.listen("0.0.0.0", port);

    state.proof_log.close();
    std::cout << "\nServer stopped." << std::endl;

    return 0;
}
