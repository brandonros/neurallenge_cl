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
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <sqlite3.h>

#include "httplib.h"
#include "json.hpp"
#include "kernel_embedded.h"
#include "config.h"

using json = nlohmann::json;

// ============================================================================
// Tokenomics (corrected for continuous rewards)
// ============================================================================
//
// Hardware benchmark:
//   4x RTX 5090 @ ~27 MH/s for ~$1.15/hr (vast.ai)
//
// Observed performance:
//   32 bits ≈ 160–220s
//   34 bits ≈ 360s
//
// Difficulty scaling (each +1 bit = 2× harder):
//   32 bits: 2^32 / 27M ≈ 160s  → ~22 proofs/hr
//   40 bits: ≈ 11 hours
//   48 bits: ≈ 121 days
//
// Reward formula:
//   reward(bits) = 1.5^(bits - MIN_BITS)
//
// Base rewards:
//   32 bits = 1.0 token
//   40 bits = 25.6 tokens
//   48 bits = 656.8 tokens
//
// --------------------------------------------------------------------------
// IMPORTANT CORRECTION:
// Mining for high-bit proofs earns ALL lower-bit proofs along the way.
// A 48-bit search is NOT exclusive — it continuously yields 32+, 33+, … proofs.
//
// Expected proofs per hour at 27 MH/s:
//   32 bits: ~11 proofs × 1.0   = 11.00 tokens
//   33 bits: ~5.5  proofs × 1.5 =  8.25 tokens
//   34 bits: ~2.75 proofs × 2.25=  6.19 tokens
//   35 bits: ~1.38 proofs × 3.375= 4.64 tokens
//   ...
//
// This forms a geometric series with ratio 0.75:
//
//   Total tokens/hr
//   = 11 × (1 + 0.75 + 0.75² + …)
//   = 11 × (1 / (1 - 0.75))
//   ≈ 44 tokens/hour
//
// Effective cost:
//   $1.15 / 44 ≈ $0.026 per token
//
// --------------------------------------------------------------------------
// Long-run implication:
//
// 48-bit expected time: ~121 days
// Total cost: ~$3,300
// Tokens earned along the way: ~131,000
// Marginal 48-bit bonus: ~657 tokens (~0.5% extra)
//
// Conclusion:
//   • High-bit proofs are prestige + anti-spam signals
//   • The real economic yield comes from cumulative lower-bit rewards
//   • Token issuance is stable and predictable
//   • Incentive still favors continuous honest mining, not grinding resets
//
// --------------------------------------------------------------------------
// Token Supply & Weekly Mining Estimates
//
// Total supply: 1,000,000,000 (1 billion) tokens
//
// Per 4x RTX 5090 rig (~27 MH/s):
//   Tokens per hour:  ~44
//   Tokens per day:   ~1,056
//   Tokens per week:  ~7,400
//   Cost per week:    ~$193 (at $1.15/hr on-demand)
//
// Weekly earnings as % of total supply:
//   1 rig:    7,400 / 1B = 0.00074%
//   10 rigs:  0.0074%
//   100 rigs: 0.074%
//
// Time to mine entire supply (solo, 1 rig):
//   1B / 44 tokens/hr = ~2,600 years
//
// This ensures:
//   • Slow, predictable token distribution
//   • No single miner can dominate supply quickly
//   • Long-term sustainability of mining incentives
//
// --------------------------------------------------------------------------
// Profitability Analysis
//
// Break-even token price:
//   $193 / 7,400 tokens = ~$0.026/token
//
// Weekly profit at various token prices:
//   $0.010/token: 7,400 × $0.010 = $74  → -$119 loss
//   $0.015/token: 7,400 × $0.015 = $111 → -$82 loss
//   $0.020/token: 7,400 × $0.020 = $148 → -$45 loss
//   $0.025/token: 7,400 × $0.025 = $185 → -$8 loss
//   $0.026/token: break-even
//   $0.030/token: 7,400 × $0.030 = $222 → +$29 profit
//
// Market cap at various prices (1B supply):
//   $0.010/token: $10M
//   $0.015/token: $15M
//   $0.020/token: $20M
//   $0.025/token: $25M
//   $0.026/token: $26M (break-even floor)
//   $0.030/token: $30M
//
// ============================================================================

constexpr int MIN_BITS = 20;
constexpr int SERVER_PORT = 8080;
constexpr const char* DB_FILE = "proofs.db";
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
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &cl.device, &num_devices) == CL_SUCCESS && num_devices > 0) {
            break;
        }
    }
    if (!cl.device) {
        std::cerr << "No OpenCL CPU device found" << std::endl;
        return false;
    }

    char name[256];
    clGetDeviceInfo(cl.device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    std::cout << "Using CPU: " << name << std::endl;

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
};

struct ServerState {
    OpenCLContext cl;
    sqlite3* db = nullptr;
    std::mutex mutex;
    uint64_t total_rejected = 0;  // In-memory counter (not persisted)
};

// ============================================================================
// SQLite Database
// ============================================================================

bool init_database(sqlite3* db) {
    const char* schema = R"(
        CREATE TABLE IF NOT EXISTS proofs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            wallet TEXT NOT NULL,
            nonce TEXT UNIQUE NOT NULL,
            bits INTEGER NOT NULL,
            reward REAL NOT NULL,
            digest TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_proofs_wallet ON proofs(wallet);
        CREATE INDEX IF NOT EXISTS idx_proofs_bits ON proofs(bits DESC);
        CREATE INDEX IF NOT EXISTS idx_proofs_nonce ON proofs(nonce);
    )";

    char* err_msg = nullptr;
    if (sqlite3_exec(db, schema, nullptr, nullptr, &err_msg) != SQLITE_OK) {
        std::cerr << "Failed to create schema: " << err_msg << std::endl;
        sqlite3_free(err_msg);
        return false;
    }
    return true;
}

bool insert_proof(sqlite3* db, const ProofRecord& proof) {
    sqlite3_stmt* stmt;
    const char* sql = "INSERT INTO proofs (timestamp, wallet, nonce, bits, reward, digest) VALUES (?, ?, ?, ?, ?, ?)";

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return false;
    }

    sqlite3_bind_int64(stmt, 1, proof.timestamp);
    sqlite3_bind_text(stmt, 2, proof.wallet.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_text(stmt, 3, proof.nonce.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_int(stmt, 4, proof.bits);
    sqlite3_bind_double(stmt, 5, proof.reward);
    sqlite3_bind_text(stmt, 6, proof.digest.c_str(), -1, SQLITE_TRANSIENT);

    bool success = (sqlite3_step(stmt) == SQLITE_DONE);
    sqlite3_finalize(stmt);
    return success;
}

bool nonce_exists(sqlite3* db, const std::string& nonce) {
    sqlite3_stmt* stmt;
    const char* sql = "SELECT 1 FROM proofs WHERE nonce = ? LIMIT 1";

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return true;  // Fail safe: assume exists on error
    }

    sqlite3_bind_text(stmt, 1, nonce.c_str(), -1, SQLITE_TRANSIENT);
    bool exists = (sqlite3_step(stmt) == SQLITE_ROW);
    sqlite3_finalize(stmt);
    return exists;
}

int64_t get_proof_count(sqlite3* db) {
    sqlite3_stmt* stmt;
    if (sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM proofs", -1, &stmt, nullptr) != SQLITE_OK) {
        return 0;
    }
    int64_t count = 0;
    if (sqlite3_step(stmt) == SQLITE_ROW) {
        count = sqlite3_column_int64(stmt, 0);
    }
    sqlite3_finalize(stmt);
    return count;
}

json get_proofs_paginated(sqlite3* db, int limit, int offset) {
    json arr = json::array();

    sqlite3_stmt* stmt;
    const char* sql = "SELECT timestamp, wallet, nonce, bits, reward, digest FROM proofs ORDER BY digest ASC LIMIT ? OFFSET ?";

    if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        return arr;
    }

    sqlite3_bind_int(stmt, 1, limit);
    sqlite3_bind_int(stmt, 2, offset);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        json proof;
        proof["timestamp"] = sqlite3_column_int64(stmt, 0);
        proof["wallet"] = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        proof["nonce"] = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        proof["bits"] = sqlite3_column_int(stmt, 3);
        proof["reward"] = sqlite3_column_double(stmt, 4);
        proof["digest"] = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        arr.push_back(proof);
    }

    sqlite3_finalize(stmt);
    return arr;
}

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

    // Initialize SQLite database
    std::cout << "Opening database: " << DB_FILE << std::endl;
    if (sqlite3_open(DB_FILE, &state.db) != SQLITE_OK) {
        std::cerr << "Failed to open database: " << sqlite3_errmsg(state.db) << std::endl;
        return 1;
    }

    // Enable WAL mode for better concurrent read/write performance
    sqlite3_exec(state.db, "PRAGMA journal_mode=WAL", nullptr, nullptr, nullptr);

    if (!init_database(state.db)) {
        std::cerr << "Failed to initialize database schema" << std::endl;
        sqlite3_close(state.db);
        return 1;
    }

    int64_t proof_count = get_proof_count(state.db);
    std::cout << "Database contains " << proof_count << " proofs" << std::endl;

    httplib::Server svr;

    // Health check
    svr.Get("/health", [](const httplib::Request&, httplib::Response& res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });

    // Stats
    svr.Get("/stats", [&state](const httplib::Request&, httplib::Response& res) {
        std::lock_guard<std::mutex> lock(state.mutex);
        json j;
        j["total_proofs"] = get_proof_count(state.db);
        j["total_rejected"] = state.total_rejected;
        j["min_bits"] = MIN_BITS;
        res.set_content(j.dump(), "application/json");
    });

    // Proofs (sorted by bits descending, paginated)
    svr.Get("/proofs", [&state](const httplib::Request& req, httplib::Response& res) {
        int limit = 100;
        int offset = 0;

        if (req.has_param("limit")) {
            limit = std::max(1, std::min(1000, std::atoi(req.get_param_value("limit").c_str())));
        }
        if (req.has_param("offset")) {
            offset = std::max(0, std::atoi(req.get_param_value("offset").c_str()));
        }

        std::lock_guard<std::mutex> lock(state.mutex);
        json response;
        response["total"] = get_proof_count(state.db);
        response["limit"] = limit;
        response["offset"] = offset;
        response["proofs"] = get_proofs_paginated(state.db, limit, offset);
        res.set_content(response.dump(), "application/json");
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

            // Check for duplicate (indexed lookup in SQLite)
            if (nonce_exists(state.db, nonce_str)) {
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

            if (!insert_proof(state.db, proof)) {
                // UNIQUE constraint violation = duplicate nonce (race condition)
                response["error"] = "duplicate nonce";
                state.total_rejected++;
                res.status = 400;
                res.set_content(response.dump(), "application/json");
                return;
            }

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
    std::cout << "  GET  /proofs - Proofs (sorted by bits, ?limit=100&offset=0)" << std::endl;
    std::cout << "  POST /submit - Submit proof {wallet, nonce}" << std::endl;
    std::cout << "Press Ctrl+C to stop." << std::endl;

    g_server = &svr;
    svr.listen("0.0.0.0", port);

    sqlite3_close(state.db);
    std::cout << "\nServer stopped." << std::endl;

    return 0;
}
