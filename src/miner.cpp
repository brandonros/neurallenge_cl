// Neural Proof-of-Work Miner - OpenCL Only
// All crypto/neural ops done on GPU - no CPU implementations

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <condition_variable>

#include "httplib.h"
#include "json.hpp"

using json = nlohmann::json;

#ifndef DEFAULT_USERNAME
#define DEFAULT_USERNAME "anonymous"
#endif

#include "kernel_embedded.h"
#include "config.h"

#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)

constexpr const char* DEFAULT_EPOCH = "epoch0";

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

int score_to_lz_bits(uint64_t score) {
    if (score == 0) return 64;
    int bits = 0;
    while ((score & (1ULL << 63)) == 0) { bits++; score <<= 1; }
    return bits;
}

std::string format_digest_with_marker(const std::string& hex, int lz_bits) {
    size_t nibble_pos = static_cast<size_t>(std::min<int>(lz_bits / 4, static_cast<int>(hex.size())));
    std::string marked = hex;
    marked.insert(nibble_pos, "|");
    return marked;
}

// ============================================================================
// Shared State
// ============================================================================

// Submission queue entry
struct SubmitEntry {
    std::string wallet;
    std::string nonce;
    int bits;
    std::string digest_hex;
};

struct SharedState {
    std::mutex best_mutex;
    uint64_t best_score;
    uint64_t fixed_threshold;  // Fixed threshold for "collect all" mode
    std::vector<uint8_t> best_nonce;
    std::vector<uint8_t> best_digest;
    std::atomic<bool> running{true};
    std::string username;
    bool verbose = false;
    std::chrono::steady_clock::time_point start_time;

    // Submission queue
    std::string server_url;
    std::string epoch;
    std::mutex submit_mutex;
    std::condition_variable submit_cv;
    std::queue<SubmitEntry> submit_queue;
    std::atomic<uint64_t> submitted{0};
    std::atomic<uint64_t> accepted{0};
    std::atomic<uint64_t> rejected{0};
};

// ============================================================================
// GPU Context
// ============================================================================

struct ResultBuffer {
    cl_mem found_count_buf;
    cl_mem found_nonces_buf;
    cl_mem found_digests_buf;
    cl_event kernel_done;
    bool pending;
};

struct GPUContext {
    int device_index;
    std::string device_name;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel mine_kernel;
    cl_mem weights_buf;
    cl_mem wallet_buf;
    cl_uint wallet_len;

    ResultBuffer buffers[2];
    int current_buffer;

    std::atomic<uint64_t> evals_computed{0};
    std::atomic<uint64_t> matches_found{0};
    std::atomic<uint64_t> launch_counter{0};
};

// ============================================================================
// OpenCL Helpers
// ============================================================================

std::vector<cl_device_id> discover_all_gpus() {
    std::vector<cl_device_id> all_devices;
    cl_uint num_platforms;
    if (clGetPlatformIDs(0, nullptr, &num_platforms) != CL_SUCCESS || num_platforms == 0) {
        return all_devices;
    }
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    for (cl_platform_id platform : platforms) {
        cl_uint num_devices;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices) == CL_SUCCESS && num_devices > 0) {
            std::vector<cl_device_id> devices(num_devices);
            clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
            all_devices.insert(all_devices.end(), devices.begin(), devices.end());
        }
    }
    return all_devices;
}

// Generate weights on GPU using generate_weights_kernel
bool generate_weights_gpu(GPUContext& ctx, const std::string& epoch, std::vector<int8_t>& weights) {
    cl_int err;

    // Create generate_weights kernel
    cl_kernel gen_kernel = clCreateKernel(ctx.program, "generate_weights_kernel", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create generate_weights_kernel: " << err << std::endl;
        return false;
    }

    // Create buffers
    cl_mem epoch_buf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       epoch.size(), (void*)epoch.data(), &err);
    cl_mem weights_buf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, TOTAL_WEIGHTS, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create buffers for weight generation" << std::endl;
        clReleaseKernel(gen_kernel);
        return false;
    }

    // Set args
    cl_uint epoch_len = static_cast<cl_uint>(epoch.size());
    clSetKernelArg(gen_kernel, 0, sizeof(cl_mem), &epoch_buf);
    clSetKernelArg(gen_kernel, 1, sizeof(cl_uint), &epoch_len);
    clSetKernelArg(gen_kernel, 2, sizeof(cl_mem), &weights_buf);

    // Launch - one work-item per 32 bytes
    size_t global_size = (TOTAL_WEIGHTS + 31) / 32;
    err = clEnqueueNDRangeKernel(ctx.queue, gen_kernel, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to launch generate_weights_kernel: " << err << std::endl;
        clReleaseMemObject(epoch_buf);
        clReleaseMemObject(weights_buf);
        clReleaseKernel(gen_kernel);
        return false;
    }

    // Read back weights
    weights.resize(TOTAL_WEIGHTS);
    err = clEnqueueReadBuffer(ctx.queue, weights_buf, CL_TRUE, 0, TOTAL_WEIGHTS, weights.data(), 0, nullptr, nullptr);

    clReleaseMemObject(epoch_buf);
    clReleaseMemObject(weights_buf);
    clReleaseKernel(gen_kernel);

    return err == CL_SUCCESS;
}

bool create_gpu_context(cl_device_id device, int device_index, GPUContext& ctx) {
    ctx.device_index = device_index;
    ctx.device = device;

    char name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, nullptr);
    ctx.device_name = name;

    cl_int err;

    ctx.context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create context: " << err << std::endl;
        return false;
    }

#ifdef CL_VERSION_2_0
    ctx.queue = clCreateCommandQueueWithProperties(ctx.context, device, nullptr, &err);
#else
    ctx.queue = clCreateCommandQueue(ctx.context, device, 0, &err);
#endif
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create command queue: " << err << std::endl;
        return false;
    }

    const char* src = reinterpret_cast<const char*>(kernel_cl);
    size_t srcLen = kernel_cl_len;

    ctx.program = clCreateProgramWithSource(ctx.context, 1, &src, &srcLen, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create program: " << err << std::endl;
        return false;
    }

    const char* buildOpts = "-D HASHES_PER_THREAD=" STRINGIFY(HASHES_PER_THREAD)
                            " -cl-fp32-correctly-rounded-divide-sqrt";
    err = clBuildProgram(ctx.program, 1, &device, buildOpts, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char log[16384];
        clGetProgramBuildInfo(ctx.program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
        std::cerr << "[GPU " << device_index << "] Build error: " << log << std::endl;
        return false;
    }

    ctx.mine_kernel = clCreateKernel(ctx.program, "neural_pow_mine", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create mine kernel: " << err << std::endl;
        return false;
    }

    // Double-buffered result buffers
    for (int i = 0; i < 2; i++) {
        ctx.buffers[i].found_count_buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
        ctx.buffers[i].found_nonces_buf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, MAX_RESULTS * NONCE_BYTES, nullptr, &err);
        ctx.buffers[i].found_digests_buf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, MAX_RESULTS * DIGEST_BYTES, nullptr, &err);
        ctx.buffers[i].kernel_done = nullptr;
        ctx.buffers[i].pending = false;
    }
    ctx.current_buffer = 0;

    return true;
}

void setup_weights_buffer(GPUContext& ctx, const std::vector<int8_t>& weights) {
    cl_int err;
    ctx.weights_buf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      weights.size(), (void*)weights.data(), &err);
}

void setup_mining_args(GPUContext& ctx, const std::string& wallet) {
    cl_int err;

    // Wallet buffer (for proof binding)
    ctx.wallet_len = static_cast<cl_uint>(std::min(wallet.size(), static_cast<size_t>(WALLET_MAX_BYTES)));
    ctx.wallet_buf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     ctx.wallet_len, (void*)wallet.data(), &err);

    // Set static kernel args (weights, wallet, wallet_len)
    clSetKernelArg(ctx.mine_kernel, 0, sizeof(cl_mem), &ctx.weights_buf);
    clSetKernelArg(ctx.mine_kernel, 1, sizeof(cl_mem), &ctx.wallet_buf);
    clSetKernelArg(ctx.mine_kernel, 2, sizeof(cl_uint), &ctx.wallet_len);
}

// ============================================================================
// GPU Worker Thread
// ============================================================================

// Extract score from first 8 bytes of digest (big-endian)
uint64_t digest_to_score(const uint8_t* digest) {
    uint64_t score = 0;
    for (int i = 0; i < 8; i++) {
        score = (score << 8) | digest[i];
    }
    return score;
}

void process_kernel_results(GPUContext& ctx, SharedState& shared, ResultBuffer& buf) {
    ctx.evals_computed.fetch_add(static_cast<uint64_t>(GLOBAL_SIZE) * HASHES_PER_THREAD);

    cl_uint found_count;
    clEnqueueReadBuffer(ctx.queue, buf.found_count_buf, CL_TRUE, 0, sizeof(cl_uint), &found_count, 0, nullptr, nullptr);

    if (found_count > 0) {
        size_t results_to_read = std::min(static_cast<size_t>(found_count), static_cast<size_t>(MAX_RESULTS));

        std::vector<uint8_t> all_nonces(results_to_read * NONCE_BYTES);
        std::vector<uint8_t> all_digests(results_to_read * DIGEST_BYTES);

        clEnqueueReadBuffer(ctx.queue, buf.found_nonces_buf, CL_TRUE, 0,
                           results_to_read * NONCE_BYTES, all_nonces.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(ctx.queue, buf.found_digests_buf, CL_TRUE, 0,
                           results_to_read * DIGEST_BYTES, all_digests.data(), 0, nullptr, nullptr);

        // Output ALL results that meet threshold and queue for submission
        {
            std::lock_guard<std::mutex> lock(shared.best_mutex);
            for (size_t i = 0; i < results_to_read; i++) {
                uint8_t* nonce = all_nonces.data() + i * NONCE_BYTES;
                uint8_t* digest = all_digests.data() + i * DIGEST_BYTES;
                uint64_t score = digest_to_score(digest);

                int bits = count_leading_zero_bits(digest, DIGEST_BYTES);
                std::string digest_hex = bytes_to_hex(digest, DIGEST_BYTES);
                std::string nonce_str(reinterpret_cast<char*>(nonce), NONCE_BYTES);

                ctx.matches_found.fetch_add(1);

                // Track best for final summary
                if (score < shared.best_score) {
                    shared.best_score = score;
                    shared.best_nonce.assign(nonce, nonce + NONCE_BYTES);
                    shared.best_digest.assign(digest, digest + DIGEST_BYTES);
                }

                // Output this result
                std::cout << "\n[GPU " << ctx.device_index << "] FOUND: "
                          << bits << " bits | " << shared.username << "/" << nonce_str
                          << " | " << format_digest_with_marker(digest_hex, bits) << std::endl;

                // Queue for submission
                {
                    std::lock_guard<std::mutex> submit_lock(shared.submit_mutex);
                    shared.submit_queue.push({shared.username, nonce_str, bits, digest_hex});
                    shared.submit_cv.notify_one();
                }
            }
        }

        if (found_count > MAX_RESULTS) {
            std::cout << "  (batch had " << found_count << " results, showing first " << MAX_RESULTS << ")" << std::endl;
        }
    }
}

// ============================================================================
// Submission Worker Thread
// ============================================================================

void submit_worker_thread(SharedState& shared) {
    std::cout << "[Submit] Submitting proofs to " << shared.server_url << std::endl;

    httplib::Client cli(shared.server_url);
    cli.set_connection_timeout(5);
    cli.set_read_timeout(10);

    while (shared.running.load()) {
        SubmitEntry entry;

        // Wait for work
        {
            std::unique_lock<std::mutex> lock(shared.submit_mutex);
            shared.submit_cv.wait_for(lock, std::chrono::milliseconds(100), [&] {
                return !shared.submit_queue.empty() || !shared.running.load();
            });

            if (!shared.running.load() && shared.submit_queue.empty()) break;
            if (shared.submit_queue.empty()) continue;

            entry = std::move(shared.submit_queue.front());
            shared.submit_queue.pop();
        }

        // Submit to server
        json payload;
        payload["wallet"] = entry.wallet;
        payload["nonce"] = entry.nonce;
        payload["epoch"] = shared.epoch;

        shared.submitted.fetch_add(1);

        auto res = cli.Post("/submit", payload.dump(), "application/json");

        if (res && res->status == 200) {
            try {
                json resp = json::parse(res->body);
                if (resp.contains("success") && resp["success"].get<bool>()) {
                    shared.accepted.fetch_add(1);
                    double reward = resp.value("reward", 0.0);
                    std::cout << "\n[Submit] ACCEPTED: " << entry.bits << " bits | reward="
                              << std::fixed << std::setprecision(2) << reward << std::endl;
                } else {
                    shared.rejected.fetch_add(1);
                    std::string err = resp.value("error", "unknown");
                    std::cout << "\n[Submit] REJECTED: " << entry.bits << " bits | " << err << std::endl;
                }
            } catch (...) {
                shared.rejected.fetch_add(1);
                std::cout << "\n[Submit] REJECTED: " << entry.bits << " bits | parse error" << std::endl;
            }
        } else {
            shared.rejected.fetch_add(1);
            std::string err = res ? std::to_string(res->status) : "connection failed";
            std::cout << "\n[Submit] FAILED: " << entry.bits << " bits | " << err << std::endl;
        }
    }

    // Drain remaining queue
    std::lock_guard<std::mutex> lock(shared.submit_mutex);
    while (!shared.submit_queue.empty()) {
        auto& entry = shared.submit_queue.front();
        json payload;
        payload["wallet"] = entry.wallet;
        payload["nonce"] = entry.nonce;
        payload["epoch"] = shared.epoch;

        auto res = cli.Post("/submit", payload.dump(), "application/json");
        if (res && res->status == 200) {
            shared.accepted.fetch_add(1);
        } else {
            shared.rejected.fetch_add(1);
        }
        shared.submit_queue.pop();
    }

    std::cout << "[Submit] Worker stopped" << std::endl;
}

void gpu_worker_thread(GPUContext& ctx, SharedState& shared) {
    std::cout << "[GPU " << ctx.device_index << "] Started mining on " << ctx.device_name << std::endl;

    std::random_device rd;
    std::mt19937 rng(rd());

    size_t global_size = GLOBAL_SIZE;
    size_t local_size = LOCAL_SIZE;

    while (shared.running.load()) {
        ResultBuffer& buf = ctx.buffers[ctx.current_buffer];

        if (buf.pending) {
            clWaitForEvents(1, &buf.kernel_done);
            clReleaseEvent(buf.kernel_done);
            buf.kernel_done = nullptr;
            buf.pending = false;
            process_kernel_results(ctx, shared, buf);
        }

        if (!shared.running.load()) break;

        // Use FIXED threshold - don't tighten over time
        // This gives us ALL results >= target bits
        cl_ulong target_score = shared.fixed_threshold;

        ctx.launch_counter.fetch_add(1);
        cl_uint seed_lo = rng();
        cl_uint seed_hi = rng();

        cl_uint zero = 0;
        clEnqueueWriteBuffer(ctx.queue, buf.found_count_buf, CL_FALSE, 0, sizeof(cl_uint), &zero, 0, nullptr, nullptr);

        // Args 0-2 (weights, wallet, wallet_len) set in setup_static_buffers
        clSetKernelArg(ctx.mine_kernel, 3, sizeof(cl_ulong), &target_score);
        clSetKernelArg(ctx.mine_kernel, 4, sizeof(cl_uint), &seed_lo);
        clSetKernelArg(ctx.mine_kernel, 5, sizeof(cl_uint), &seed_hi);
        clSetKernelArg(ctx.mine_kernel, 6, sizeof(cl_mem), &buf.found_count_buf);
        clSetKernelArg(ctx.mine_kernel, 7, sizeof(cl_mem), &buf.found_nonces_buf);
        clSetKernelArg(ctx.mine_kernel, 8, sizeof(cl_mem), &buf.found_digests_buf);

        cl_int err = clEnqueueNDRangeKernel(ctx.queue, ctx.mine_kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, &buf.kernel_done);
        if (err != CL_SUCCESS) {
            std::cerr << "[GPU " << ctx.device_index << "] Kernel error: " << err << std::endl;
            break;
        }

        buf.pending = true;
        ctx.current_buffer = 1 - ctx.current_buffer;
    }

    for (int i = 0; i < 2; i++) {
        if (ctx.buffers[i].pending && ctx.buffers[i].kernel_done) {
            clWaitForEvents(1, &ctx.buffers[i].kernel_done);
            clReleaseEvent(ctx.buffers[i].kernel_done);
        }
    }

    std::cout << "[GPU " << ctx.device_index << "] Stopped" << std::endl;
}

// ============================================================================
// Signal Handling
// ============================================================================

volatile std::sig_atomic_t g_shutdown_requested = 0;

void signal_handler(int) {
    g_shutdown_requested = 1;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::string username = DEFAULT_USERNAME;
    std::string server_url = "http://localhost:8080";
    std::string epoch = DEFAULT_EPOCH;
    int target_zero_bits = 16;
    bool verbose = false;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]\n"
                      << "Options:\n"
                      << "  -u, --user STRING   Username (default: " << DEFAULT_USERNAME << ")\n"
                      << "  -s, --server URL    Server URL for proof submission (default: http://localhost:8080)\n"
                      << "  -e, --epoch STRING  Weight epoch (default: " << DEFAULT_EPOCH << ")\n"
                      << "  -b, --bits N        Target leading zero bits (default: 16)\n"
                      << "  -x, --hex N         Target leading zero hex digits (N*4 bits)\n"
                      << "  -v, --verbose       Show network outputs\n"
                      << "  -h, --help          Show this help\n";
            return 0;
        } else if ((arg == "--user" || arg == "-u") && i + 1 < argc) {
            username = argv[++i];
        } else if (arg.rfind("--user=", 0) == 0) {
            username = arg.substr(7);
        } else if ((arg == "--server" || arg == "-s") && i + 1 < argc) {
            server_url = argv[++i];
        } else if (arg.rfind("--server=", 0) == 0) {
            server_url = arg.substr(9);
        } else if ((arg == "--epoch" || arg == "-e") && i + 1 < argc) {
            epoch = argv[++i];
        } else if (arg.rfind("--epoch=", 0) == 0) {
            epoch = arg.substr(8);
        } else if ((arg == "--bits" || arg == "-b") && i + 1 < argc) {
            target_zero_bits = std::max(std::stoi(argv[++i]), 1);
        } else if (arg.rfind("--bits=", 0) == 0) {
            target_zero_bits = std::max(std::stoi(arg.substr(7)), 1);
        } else if ((arg == "--hex" || arg == "-x") && i + 1 < argc) {
            target_zero_bits = std::max(std::stoi(argv[++i]) * 4, 1);
        } else if (arg.rfind("--hex=", 0) == 0) {
            target_zero_bits = std::max(std::stoi(arg.substr(6)) * 4, 1);
        } else if (arg == "--verbose" || arg == "-v") {
            verbose = true;
        } else {
            std::cerr << "Unknown option: " << arg << "\nUse --help for usage.\n";
            return 1;
        }
    }

    // Discover GPUs
    std::vector<cl_device_id> devices = discover_all_gpus();
    if (devices.empty()) {
        std::cerr << "No GPUs found!" << std::endl;
        return 1;
    }

    // Initialize first GPU for weight generation
    std::vector<std::unique_ptr<GPUContext>> gpus;
    gpus.push_back(std::make_unique<GPUContext>());
    if (!create_gpu_context(devices[0], 0, *gpus[0])) {
        std::cerr << "Failed to initialize GPU" << std::endl;
        return 1;
    }

    // Generate weights on GPU
    std::cout << "Generating weights on GPU from epoch: " << epoch << std::endl;
    std::vector<int8_t> weights;
    if (!generate_weights_gpu(*gpus[0], epoch.c_str(), weights)) {
        std::cerr << "Failed to generate weights" << std::endl;
        return 1;
    }
    std::cout << "Generated " << weights.size() << " weight bytes" << std::endl;

    // Setup weights buffer for first GPU
    setup_weights_buffer(*gpus[0], weights);

    // Mining mode - fixed threshold to collect ALL results >= target bits
    uint64_t fixed_threshold = (target_zero_bits >= 64) ? 0 : (1ULL << (64 - target_zero_bits));

    SharedState shared;
    shared.username = username;
    shared.server_url = server_url;
    shared.epoch = epoch;
    shared.verbose = verbose;
    shared.fixed_threshold = fixed_threshold;  // Never changes - collects all >= target
    shared.best_score = UINT64_MAX;            // For tracking best found
    shared.start_time = std::chrono::steady_clock::now();

    std::cout << "Neural Proof-of-Work Miner (OpenCL) - COLLECT ALL MODE" << std::endl;
    std::cout << "Username: " << shared.username << std::endl;
    std::cout << "Epoch: " << epoch << std::endl;
    std::cout << "Collecting ALL nonces with " << target_zero_bits << "+ leading zero bits" << std::endl;
    std::cout << "Server: " << server_url << std::endl;
    std::cout << "Network: " << INPUT_DIM << " -> " << HIDDEN_DIM << " -> " << HIDDEN_DIM << " -> " << HIDDEN_DIM << " -> " << OUTPUT_DIM << std::endl;
    std::cout << "Found " << devices.size() << " GPU(s)" << std::endl;

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // Initialize remaining GPUs and set mining args for all
    for (size_t i = 1; i < devices.size(); i++) {
        gpus.push_back(std::make_unique<GPUContext>());
        if (!create_gpu_context(devices[i], static_cast<int>(i), *gpus[i])) {
            std::cerr << "Failed to initialize GPU " << i << std::endl;
            return 1;
        }
        setup_weights_buffer(*gpus[i], weights);
    }

    // Setup mining args for all GPUs (wallet for proof binding)
    for (auto& gpu : gpus) {
        setup_mining_args(*gpu, username);
    }

    for (auto& gpu : gpus) {
        std::cout << "Initialized GPU " << gpu->device_index << ": " << gpu->device_name << std::endl;
    }

    std::cout << "\nMining started...\n" << std::endl;

    // Start submission worker thread
    std::thread submit_thread(submit_worker_thread, std::ref(shared));

    std::vector<std::thread> worker_threads;
    for (auto& gpu : gpus) {
        worker_threads.emplace_back(gpu_worker_thread, std::ref(*gpu), std::ref(shared));
    }

    // Stats loop
    uint64_t last_total = 0;
    auto last_time = std::chrono::steady_clock::now();
    double smoothed_rate = 0.0;
    constexpr double EMA_ALPHA = 0.1;

    while (shared.running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        if (g_shutdown_requested) {
            std::cout << "\nShutting down..." << std::endl;
            shared.running.store(false);
            break;
        }

        uint64_t total_evals = 0;
        uint64_t total_matches = 0;
        for (const auto& gpu : gpus) {
            total_evals += gpu->evals_computed.load();
            total_matches += gpu->matches_found.load();
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsed_total = std::chrono::duration_cast<std::chrono::seconds>(now - shared.start_time).count();
        auto interval_us = std::chrono::duration_cast<std::chrono::microseconds>(now - last_time).count();

        double instant_rate = 0.0;
        if (interval_us > 0 && total_evals > last_total) {
            instant_rate = static_cast<double>(total_evals - last_total) / (interval_us / 1e6);
        }

        if (instant_rate > 0) {
            smoothed_rate = (smoothed_rate == 0.0) ? instant_rate : (EMA_ALPHA * instant_rate + (1.0 - EMA_ALPHA) * smoothed_rate);
        }

        last_total = total_evals;
        last_time = now;

        int current_bits;
        {
            std::lock_guard<std::mutex> lock(shared.best_mutex);
            current_bits = score_to_lz_bits(shared.best_score);
        }

        if (total_evals > 0) {
            std::cout << "\r[Stats] " << std::fixed << std::setprecision(2)
                      << (smoothed_rate / 1e6) << " M/s"
                      << " | " << std::setprecision(3) << (total_evals / 1e9) << "B total"
                      << " | best=" << current_bits << " bits"
                      << " | " << total_matches << " found"
                      << " | " << shared.accepted.load() << "/" << shared.submitted.load() << " accepted"
                      << " | " << elapsed_total << "s"
                      << "        " << std::flush;
        }
    }

    // Notify submit thread to finish
    shared.submit_cv.notify_all();

    for (auto& t : worker_threads) {
        t.join();
    }

    submit_thread.join();

    // Final results
    std::cout << "\n\n========================================" << std::endl;
    std::cout << "Final Results" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Username: " << shared.username << std::endl;

    std::cout << "\nSubmissions:" << std::endl;
    std::cout << "  Submitted: " << shared.submitted.load() << std::endl;
    std::cout << "  Accepted:  " << shared.accepted.load() << std::endl;
    std::cout << "  Rejected:  " << shared.rejected.load() << std::endl;

    if (!shared.best_nonce.empty()) {
        std::string digest_hex = bytes_to_hex(shared.best_digest.data(), shared.best_digest.size());
        int lz_bits = count_leading_zero_bits(shared.best_digest.data(), shared.best_digest.size());
        std::string nonce_str(reinterpret_cast<char*>(shared.best_nonce.data()), shared.best_nonce.size());

        std::cout << "\nBest Result: " << lz_bits << " bits" << std::endl;
        std::cout << "Digest: " << format_digest_with_marker(digest_hex, lz_bits) << std::endl;
        std::cout << "Proof: " << shared.username << "/" << nonce_str << std::endl;
    } else {
        std::cout << "\nNo valid result found." << std::endl;
    }

    return 0;
}
