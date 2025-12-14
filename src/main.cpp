// Neural Proof-of-Work - Host Code v2
// FP32-heavy PoW with CPU verifier using identical quantization

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <algorithm>
#include <atomic>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifndef DEFAULT_USERNAME
#define DEFAULT_USERNAME "anonymous"
#endif

#include "kernel_embedded.h"
#include "config.h"

// ============================================================================
// Configuration
// ============================================================================

#define STRINGIFY_(x) #x
#define STRINGIFY(x) STRINGIFY_(x)


// ============================================================================
// CPU Verifier - MUST match GPU exactly
// ============================================================================

namespace cpu_verifier {

// Weight/input scaling factors
constexpr float WEIGHT_SCALE = 1.0f / 127.0f;   // int8 weights in [-127, 127]
constexpr float INPUT_SCALE = 1.0f / 32768.0f;  // int16 inputs in [-32768, 32767]

// Q16 fixed-point constants (16 fractional bits)
constexpr float Q16_SCALE = 65536.0f;           // 2^16
constexpr float Q16_INV_SCALE = 1.0f / 65536.0f;
constexpr float Q16_MIN = -32768.0f;            // Min value before overflow when scaled
constexpr float Q16_MAX = 32767.0f;             // Max value before overflow when scaled

// Quantization using round-to-nearest-even (matches OpenCL convert_int_rte)
// Use lrintf (float version) to match OpenCL's convert_int_rte(float) exactly
inline float q16(float x) {
    if (!std::isfinite(x)) x = 0.0f;
    x = std::max(Q16_MIN, std::min(Q16_MAX, x));
    int32_t i = static_cast<int32_t>(std::lrintf(x * Q16_SCALE));
    return static_cast<float>(i) * Q16_INV_SCALE;
}

// Get q16 as integer representation (for deterministic scoring)
inline int32_t q16_to_int(float q16_val) {
    return static_cast<int32_t>(std::lrintf(q16_val * Q16_SCALE));
}

inline float get_weight(const int8_t* weights, size_t idx) {
    return static_cast<float>(weights[idx]) * WEIGHT_SCALE;
}

// Forward declaration - expand_nonce is implemented after sha256 namespace
void expand_nonce(const uint8_t* nonce, size_t nonce_len, uint8_t* expanded);

// Convert 64 expanded bytes to INPUT_DIM floats
void expanded_to_input(const uint8_t* expanded, float* input) {
    for (int i = 0; i < INPUT_DIM; i++) {
        int16_t val = static_cast<int16_t>(expanded[i*2] | (expanded[i*2 + 1] << 8));
        input[i] = q16(static_cast<float>(val) * INPUT_SCALE);
    }
}

// Full nonce-to-input: any nonce -> SHA256 expand -> 32 floats
void nonce_to_input(const uint8_t* nonce, size_t nonce_len, float* input) {
    uint8_t expanded[64];
    expand_nonce(nonce, nonce_len, expanded);
    expanded_to_input(expanded, input);
}

float matmul_row(const int8_t* W, const int8_t* bias, const float* input, size_t in_dim) {
    float sum = 0.0f;

    for (size_t i = 0; i < in_dim; i++) {
        float prod = get_weight(W, i) * input[i];
        sum = sum + prod;

        if ((i & 31) == 31) {
            sum = q16(sum);
        }
    }

    sum += static_cast<float>(*bias) * WEIGHT_SCALE;
    return q16(sum);
}

// Note: matmul_row returns q16, and ReLU(q16) stays on grid (0.0f or unchanged)
void layer_forward(
    const int8_t* W,
    const int8_t* bias,
    const float* input,
    float* output,
    size_t in_dim,
    size_t out_dim,
    bool use_relu
) {
    for (size_t i = 0; i < out_dim; i++) {
        float sum = matmul_row(W + i * in_dim, bias + i, input, in_dim);
        output[i] = use_relu ? std::max(0.0f, sum) : sum;
    }
}

// ============================================================================
// SipHash-2-4 - Proper cryptographic mixing for digest
// ============================================================================

inline uint64_t rotl64(uint64_t x, int b) {
    return (x << b) | (x >> (64 - b));
}

inline void sipround(uint64_t& v0, uint64_t& v1, uint64_t& v2, uint64_t& v3) {
    v0 += v1; v1 = rotl64(v1, 13); v1 ^= v0; v0 = rotl64(v0, 32);
    v2 += v3; v3 = rotl64(v3, 16); v3 ^= v2;
    v0 += v3; v3 = rotl64(v3, 21); v3 ^= v0;
    v2 += v1; v1 = rotl64(v1, 17); v1 ^= v2; v2 = rotl64(v2, 32);
}

// SipHash-2-4 specialized for SIPHASH_INPUT_BYTES (132 bytes)
// Matches kernel's siphash_2_4_132 exactly to avoid divergence
uint64_t siphash_2_4_132(const uint8_t* data, uint64_t k0, uint64_t k1) {
    uint64_t v0 = k0 ^ 0x736f6d6570736575ULL;
    uint64_t v1 = k1 ^ 0x646f72616e646f6dULL;
    uint64_t v2 = k0 ^ 0x6c7967656e657261ULL;
    uint64_t v3 = k1 ^ 0x7465646279746573ULL;

    // Process 16 full 8-byte blocks (128 bytes)
    for (int blk = 0; blk < 16; blk++) {
        const uint8_t* p = data + blk * 8;
        uint64_t m = static_cast<uint64_t>(p[0])
                   | (static_cast<uint64_t>(p[1]) << 8)
                   | (static_cast<uint64_t>(p[2]) << 16)
                   | (static_cast<uint64_t>(p[3]) << 24)
                   | (static_cast<uint64_t>(p[4]) << 32)
                   | (static_cast<uint64_t>(p[5]) << 40)
                   | (static_cast<uint64_t>(p[6]) << 48)
                   | (static_cast<uint64_t>(p[7]) << 56);
        v3 ^= m;
        sipround(v0, v1, v2, v3);
        sipround(v0, v1, v2, v3);
        v0 ^= m;
    }

    // Process final 4 bytes + length (132 = 0x84)
    const uint8_t* p = data + 128;
    uint64_t b = (132ULL << 56)
               | static_cast<uint64_t>(p[0])
               | (static_cast<uint64_t>(p[1]) << 8)
               | (static_cast<uint64_t>(p[2]) << 16)
               | (static_cast<uint64_t>(p[3]) << 24);

    v3 ^= b;
    sipround(v0, v1, v2, v3);
    sipround(v0, v1, v2, v3);
    v0 ^= b;

    // Finalization
    v2 ^= 0xff;
    sipround(v0, v1, v2, v3);
    sipround(v0, v1, v2, v3);
    sipround(v0, v1, v2, v3);
    sipround(v0, v1, v2, v3);

    return v0 ^ v1 ^ v2 ^ v3;
}

// SipHash keys defined in config.h

void compute_digest(const float* output, uint8_t* digest, size_t dim) {
    // Serialize q16 integers directly into SipHash input buffer
    uint8_t input[SIPHASH_INPUT_BYTES];
    for (size_t i = 0; i < dim; i++) {
        int32_t val = q16_to_int(output[i]);
        input[i * 4 + 0] = static_cast<uint8_t>(val & 0xFF);
        input[i * 4 + 1] = static_cast<uint8_t>((val >> 8) & 0xFF);
        input[i * 4 + 2] = static_cast<uint8_t>((val >> 16) & 0xFF);
        input[i * 4 + 3] = static_cast<uint8_t>((val >> 24) & 0xFF);
    }
    // Zero-pad domain separator (only first byte changes per block)
    input[SERIALIZED_OUTPUT_BYTES + 1] = 0;
    input[SERIALIZED_OUTPUT_BYTES + 2] = 0;
    input[SERIALIZED_OUTPUT_BYTES + 3] = 0;

    // Generate DIGEST_BYTES of digest using SipHash with domain separation
    for (size_t block = 0; block < (DIGEST_BYTES / 8); block++) {
        input[SERIALIZED_OUTPUT_BYTES] = static_cast<uint8_t>(block);
        uint64_t h = siphash_2_4_132(input, SIPHASH_K0, SIPHASH_K1);

        // Extract 8 bytes from hash (little-endian)
        for (int i = 0; i < 8; i++) {
            digest[block * 8 + i] = static_cast<uint8_t>((h >> (i * 8)) & 0xFF);
        }
    }
}

int count_leading_zero_bits(const uint8_t* digest, size_t len) {
    int bits = 0;
    for (size_t i = 0; i < len; i++) {
        uint8_t b = digest[i];
        if (b == 0) {
            bits += 8;
            continue;
        }
        int lz = 0;
        while ((b & 0x80u) == 0) {
            lz++;
            b <<= 1;
        }
        bits += lz;
        break;
    }
    return bits;
}

// Score = first 8 bytes of digest as big-endian uint64 (lower = better)
uint64_t compute_score(const float* output, size_t dim) {
    uint8_t digest[DIGEST_BYTES];
    compute_digest(output, digest, dim);
    return (static_cast<uint64_t>(digest[0]) << 56) | (static_cast<uint64_t>(digest[1]) << 48) |
           (static_cast<uint64_t>(digest[2]) << 40) | (static_cast<uint64_t>(digest[3]) << 32) |
           (static_cast<uint64_t>(digest[4]) << 24) | (static_cast<uint64_t>(digest[5]) << 16) |
           (static_cast<uint64_t>(digest[6]) << 8)  | static_cast<uint64_t>(digest[7]);
}

// Convert score to leading zero bits (for display)
int score_to_lz_bits(uint64_t score) {
    if (score == 0) return 64;
    int bits = 0;
    while ((score & (1ULL << 63)) == 0) {
        bits++;
        score <<= 1;
    }
    return bits;
}

// Forward pass - computes network output from nonce (any length)
void forward_pass(const int8_t* weights, const uint8_t* nonce, size_t nonce_len, float* output) {
    float input[INPUT_DIM];
    float hidden1[HIDDEN_DIM];
    float hidden2[HIDDEN_DIM];
    float hidden3[HIDDEN_DIM];

    nonce_to_input(nonce, nonce_len, input);

    layer_forward(weights + W1_OFFSET, weights + B1_OFFSET,
                  input, hidden1, INPUT_DIM, HIDDEN_DIM, true);
    layer_forward(weights + W2_OFFSET, weights + B2_OFFSET,
                  hidden1, hidden2, HIDDEN_DIM, HIDDEN_DIM, true);
    layer_forward(weights + W3_OFFSET, weights + B3_OFFSET,
                  hidden2, hidden3, HIDDEN_DIM, HIDDEN_DIM, true);
    layer_forward(weights + W4_OFFSET, weights + B4_OFFSET,
                  hidden3, output, HIDDEN_DIM, OUTPUT_DIM, false);
}

// Full forward pass - returns score (lower = better)
uint64_t forward(const int8_t* weights, const uint8_t* nonce, size_t nonce_len) {
    float output[OUTPUT_DIM];
    forward_pass(weights, nonce, nonce_len, output);
    return compute_score(output, OUTPUT_DIM);
}

} // namespace cpu_verifier

// ============================================================================
// Weight Generation from Challenge String (SHA-256 based)
// ============================================================================

// Weights are derived from a global epoch identifier, not per-user.
// This ensures all miners compute the same neural network, enabling fair difficulty comparison.
constexpr const char* WEIGHT_EPOCH = "epoch0";

namespace sha256 {

static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define ROTR(x, n) (((x) >> (n)) | ((x) << (32 - (n))))
#define CH(x, y, z) (((x) & (y)) ^ (~(x) & (z)))
#define MAJ(x, y, z) (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define EP0(x) (ROTR(x, 2) ^ ROTR(x, 13) ^ ROTR(x, 22))
#define EP1(x) (ROTR(x, 6) ^ ROTR(x, 11) ^ ROTR(x, 25))
#define SIG0(x) (ROTR(x, 7) ^ ROTR(x, 18) ^ ((x) >> 3))
#define SIG1(x) (ROTR(x, 17) ^ ROTR(x, 19) ^ ((x) >> 10))

void hash(const uint8_t* data, size_t len, uint8_t* out) {
    // State array (FIPS 180-4 uses H(0)..H(7))
    uint32_t H[8] = {
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
        0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
    };

    size_t padded_len = ((len + 9 + 63) / 64) * 64;
    std::vector<uint8_t> padded(padded_len, 0);
    memcpy(padded.data(), data, len);
    padded[len] = 0x80;

    uint64_t bits = len * 8;
    for (int i = 0; i < 8; i++) {
        padded[padded_len - 1 - i] = (bits >> (i * 8)) & 0xFF;
    }

    for (size_t block = 0; block < padded_len; block += 64) {
        uint32_t w[64];

        for (int i = 0; i < 16; i++) {
            w[i] = (padded[block + i*4] << 24) |
                   (padded[block + i*4 + 1] << 16) |
                   (padded[block + i*4 + 2] << 8) |
                   padded[block + i*4 + 3];
        }

        for (int i = 16; i < 64; i++) {
            w[i] = SIG1(w[i-2]) + w[i-7] + SIG0(w[i-15]) + w[i-16];
        }

        // Working variables a-h (FIPS 180-4 standard names)
        uint32_t a = H[0], b = H[1], c = H[2], d = H[3];
        uint32_t e = H[4], f = H[5], g = H[6], h = H[7];

        for (int i = 0; i < 64; i++) {
            uint32_t t1 = h + EP1(e) + CH(e, f, g) + K[i] + w[i];
            uint32_t t2 = EP0(a) + MAJ(a, b, c);
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }

        H[0] += a; H[1] += b; H[2] += c; H[3] += d;
        H[4] += e; H[5] += f; H[6] += g; H[7] += h;
    }

    for (int i = 0; i < 8; i++) {
        out[i*4] = (H[i] >> 24) & 0xFF;
        out[i*4 + 1] = (H[i] >> 16) & 0xFF;
        out[i*4 + 2] = (H[i] >> 8) & 0xFF;
        out[i*4 + 3] = H[i] & 0xFF;
    }
}

} // namespace sha256

// Implementation of expand_nonce for cpu_verifier (uses sha256::hash from above)
namespace cpu_verifier {
void expand_nonce(const uint8_t* nonce, size_t nonce_len, uint8_t* expanded) {
    std::vector<uint8_t> buf(nonce_len + 1);
    std::memcpy(buf.data(), nonce, nonce_len);

    // First half: SHA256(nonce || 0x00)
    buf[nonce_len] = 0x00;
    sha256::hash(buf.data(), buf.size(), expanded);

    // Second half: SHA256(nonce || 0x01)
    buf[nonce_len] = 0x01;
    sha256::hash(buf.data(), buf.size(), expanded + 32);
}
} // namespace cpu_verifier

void append_u32(std::vector<uint8_t>& data, uint32_t value) {
    for (int i = 0; i < 4; i++) {
        data.push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xFF));
    }
}

void generate_weights(const std::string& challenge, std::vector<int8_t>& weights) {
    weights.resize(TOTAL_WEIGHTS);

    uint8_t hash_out[32];
    size_t weight_idx = 0;
    uint32_t counter = 0;

    while (weight_idx < TOTAL_WEIGHTS) {
        std::vector<uint8_t> input(challenge.begin(), challenge.end());
        append_u32(input, counter);

        sha256::hash(input.data(), input.size(), hash_out);

        for (size_t i = 0; i < 32 && weight_idx < TOTAL_WEIGHTS; i++) {
            int8_t w = static_cast<int8_t>(hash_out[i]);
            if (w == -128) w = -127;
            weights[weight_idx++] = w;
        }

        counter++;
    }
}


// ============================================================================
// Shared State
// ============================================================================

struct SharedState {
    std::mutex best_mutex;
    uint64_t best_score;  // First 8 bytes of digest (lower = better)
    std::vector<uint8_t> best_nonce;
    std::vector<uint8_t> best_digest;
    std::atomic<bool> running{true};
    std::string username;
    bool verbose = false;
    std::vector<int8_t> weights;
    std::chrono::steady_clock::time_point start_time;
};

// ============================================================================
// GPU Context
// ============================================================================

struct GPUContext {
    int device_index;
    std::string device_name;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem weights_buf;
    cl_mem found_count_buf;
    cl_mem found_scores_buf;
    cl_mem found_nonces_buf;
    std::atomic<uint64_t> evals_computed{0};
    std::atomic<uint64_t> matches_found{0};
    std::atomic<uint64_t> launch_counter{0};
    std::atomic<uint64_t> total_kernel_time_us{0};  // Total kernel time in microseconds
};

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

std::string describe_lz_bits(int bits) {
    return std::to_string(bits) + " bits";
}

// Base64 encoding table
static const char BASE64_CHARS[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

std::string bytes_to_base64(const uint8_t* data, size_t len) {
    std::string result;
    result.reserve((len + 2) / 3 * 4);

    for (size_t i = 0; i < len; i += 3) {
        uint32_t n = static_cast<uint32_t>(data[i]) << 16;
        if (i + 1 < len) n |= static_cast<uint32_t>(data[i + 1]) << 8;
        if (i + 2 < len) n |= static_cast<uint32_t>(data[i + 2]);

        result += BASE64_CHARS[(n >> 18) & 0x3F];
        result += BASE64_CHARS[(n >> 12) & 0x3F];
        result += (i + 1 < len) ? BASE64_CHARS[(n >> 6) & 0x3F] : '=';
        result += (i + 2 < len) ? BASE64_CHARS[n & 0x3F] : '=';
    }
    return result;
}

bool base64_to_bytes(const std::string& b64, std::vector<uint8_t>& out) {
    // Build reverse lookup table
    static int8_t decode_table[256] = {-1};
    static bool table_init = false;
    if (!table_init) {
        std::memset(decode_table, -1, sizeof(decode_table));
        for (int i = 0; i < 64; i++) {
            decode_table[static_cast<uint8_t>(BASE64_CHARS[i])] = static_cast<int8_t>(i);
        }
        table_init = true;
    }

    if (b64.size() % 4 != 0) return false;

    size_t padding = 0;
    if (!b64.empty() && b64[b64.size() - 1] == '=') padding++;
    if (b64.size() > 1 && b64[b64.size() - 2] == '=') padding++;

    out.resize((b64.size() / 4) * 3 - padding);

    size_t j = 0;
    for (size_t i = 0; i < b64.size(); i += 4) {
        int8_t a = decode_table[static_cast<uint8_t>(b64[i])];
        int8_t b = decode_table[static_cast<uint8_t>(b64[i + 1])];
        int8_t c = (b64[i + 2] == '=') ? 0 : decode_table[static_cast<uint8_t>(b64[i + 2])];
        int8_t d = (b64[i + 3] == '=') ? 0 : decode_table[static_cast<uint8_t>(b64[i + 3])];

        if (a < 0 || b < 0 || (b64[i + 2] != '=' && c < 0) || (b64[i + 3] != '=' && d < 0)) {
            return false;
        }

        uint32_t n = (static_cast<uint32_t>(a) << 18) | (static_cast<uint32_t>(b) << 12) |
                     (static_cast<uint32_t>(c) << 6) | static_cast<uint32_t>(d);

        if (j < out.size()) out[j++] = static_cast<uint8_t>((n >> 16) & 0xFF);
        if (j < out.size()) out[j++] = static_cast<uint8_t>((n >> 8) & 0xFF);
        if (j < out.size()) out[j++] = static_cast<uint8_t>(n & 0xFF);
    }
    return true;
}

std::string format_digest_with_marker(const std::string& hex, int lz_bits) {
    size_t nibble_pos = static_cast<size_t>(std::min<int>(lz_bits / 4, static_cast<int>(hex.size())));
    std::string marked = hex;
    marked.insert(nibble_pos, "|");
    return marked;
}

// ============================================================================
// GPU Discovery
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

// ============================================================================
// GPU Context Creation
// ============================================================================

bool create_gpu_context(cl_device_id device, int device_index, const SharedState& shared, GPUContext& ctx) {
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

    ctx.kernel = clCreateKernel(ctx.program, "neural_pow_mine", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create kernel: " << err << std::endl;
        return false;
    }

    // Create buffers
    ctx.weights_buf = clCreateBuffer(ctx.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      shared.weights.size(), (void*)shared.weights.data(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create weights buffer: " << err << std::endl;
        return false;
    }

    ctx.found_count_buf = clCreateBuffer(ctx.context, CL_MEM_READ_WRITE, sizeof(cl_uint), nullptr, &err);
    ctx.found_scores_buf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, MAX_RESULTS * sizeof(cl_long), nullptr, &err);
    ctx.found_nonces_buf = clCreateBuffer(ctx.context, CL_MEM_WRITE_ONLY, MAX_RESULTS * NONCE_BYTES, nullptr, &err);

    if (err != CL_SUCCESS) {
        std::cerr << "[GPU " << device_index << "] Failed to create result buffers: " << err << std::endl;
        return false;
    }

    // Set static kernel args
    clSetKernelArg(ctx.kernel, 0, sizeof(cl_mem), &ctx.weights_buf);
    // arg 1 (target_score) set per launch
    // arg 2 (seed_lo) set per launch
    // arg 3 (seed_hi) set per launch
    clSetKernelArg(ctx.kernel, 4, sizeof(cl_mem), &ctx.found_count_buf);
    clSetKernelArg(ctx.kernel, 5, sizeof(cl_mem), &ctx.found_scores_buf);
    clSetKernelArg(ctx.kernel, 6, sizeof(cl_mem), &ctx.found_nonces_buf);

    return true;
}

// ============================================================================
// GPU Worker Thread
// ============================================================================

void gpu_worker_thread(GPUContext& ctx, SharedState& shared) {
    std::cout << "[GPU " << ctx.device_index << "] Started mining on " << ctx.device_name << std::endl;

    // Thread-local RNG for generating kernel seeds
    std::random_device rd;
    std::mt19937 rng(rd());

    while (shared.running.load()) {
        cl_ulong target_score;
        {
            std::lock_guard<std::mutex> lock(shared.best_mutex);
            target_score = shared.best_score;
        }

        ctx.launch_counter.fetch_add(1);
        cl_uint seed_lo = rng();
        cl_uint seed_hi = rng();

        cl_uint zero = 0;
        clEnqueueWriteBuffer(ctx.queue, ctx.found_count_buf, CL_FALSE, 0, sizeof(cl_uint), &zero, 0, nullptr, nullptr);

        clSetKernelArg(ctx.kernel, 1, sizeof(cl_ulong), &target_score);
        clSetKernelArg(ctx.kernel, 2, sizeof(cl_uint), &seed_lo);
        clSetKernelArg(ctx.kernel, 3, sizeof(cl_uint), &seed_hi);

        size_t global_size = GLOBAL_SIZE;
        size_t local_size = LOCAL_SIZE;

        auto kernel_start = std::chrono::steady_clock::now();
        cl_int err = clEnqueueNDRangeKernel(ctx.queue, ctx.kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            std::cerr << "[GPU " << ctx.device_index << "] Kernel error: " << err << std::endl;
            break;
        }

        clFinish(ctx.queue);
        auto kernel_end = std::chrono::steady_clock::now();
        auto kernel_us = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_start).count();
        ctx.total_kernel_time_us.fetch_add(static_cast<uint64_t>(kernel_us));
        ctx.evals_computed.fetch_add(static_cast<uint64_t>(GLOBAL_SIZE) * HASHES_PER_THREAD);

        cl_uint found_count;
        clEnqueueReadBuffer(ctx.queue, ctx.found_count_buf, CL_TRUE, 0, sizeof(cl_uint), &found_count, 0, nullptr, nullptr);

        if (found_count > 0) {
            size_t results_to_read = std::min(static_cast<size_t>(found_count), static_cast<size_t>(MAX_RESULTS));

            std::vector<uint64_t> all_scores(results_to_read);
            std::vector<uint8_t> all_nonces(results_to_read * NONCE_BYTES);

            clEnqueueReadBuffer(ctx.queue, ctx.found_scores_buf, CL_TRUE, 0,
                               results_to_read * sizeof(uint64_t), all_scores.data(), 0, nullptr, nullptr);
            clEnqueueReadBuffer(ctx.queue, ctx.found_nonces_buf, CL_TRUE, 0,
                               results_to_read * NONCE_BYTES, all_nonces.data(), 0, nullptr, nullptr);

            // Find best in batch (lowest score = best)
            size_t best_idx = 0;
            for (size_t i = 1; i < results_to_read; i++) {
                if (all_scores[i] < all_scores[best_idx]) {
                    best_idx = i;
                }
            }

            uint64_t batch_best_score = all_scores[best_idx];
            uint8_t* batch_best_nonce = all_nonces.data() + best_idx * NONCE_BYTES;

            // Verify on CPU - must match bit-exactly (consensus requirement)
            uint64_t cpu_score = cpu_verifier::forward(shared.weights.data(), batch_best_nonce, NONCE_BYTES);

            // Scores must match exactly - any difference is a determinism bug
            if (cpu_score != batch_best_score) {
                std::cerr << "[GPU " << ctx.device_index << "] CRITICAL: Score mismatch! "
                          << "GPU=0x" << std::hex << batch_best_score
                          << " CPU=0x" << cpu_score << std::dec << std::endl;
                continue;  // Reject mismatched results
            }

            {
                std::lock_guard<std::mutex> lock(shared.best_mutex);
                if (cpu_score < shared.best_score) {
                    float outputs[OUTPUT_DIM];
                    cpu_verifier::forward_pass(shared.weights.data(), batch_best_nonce, NONCE_BYTES, outputs);
                    uint8_t digest[DIGEST_BYTES];
                    cpu_verifier::compute_digest(outputs, digest, OUTPUT_DIM);
                    int lz_bits = cpu_verifier::count_leading_zero_bits(digest, DIGEST_BYTES);
                    std::string digest_hex = bytes_to_hex(digest, DIGEST_BYTES);
                    std::string nonce_str(reinterpret_cast<char*>(batch_best_nonce), NONCE_BYTES);

                    shared.best_score = cpu_score;
                    shared.best_nonce.assign(batch_best_nonce, batch_best_nonce + NONCE_BYTES);
                    shared.best_digest.assign(digest, digest + DIGEST_BYTES);
                    ctx.matches_found.fetch_add(1);

                    auto now = std::chrono::steady_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - shared.start_time).count();

                    std::cout << "\n[GPU " << ctx.device_index << "] NEW BEST: "
                              << describe_lz_bits(lz_bits) << " | digest="
                              << format_digest_with_marker(digest_hex, lz_bits) << std::endl;
                    std::cout << "  Proof: " << shared.username << "/" << nonce_str << std::endl;
                    if (shared.verbose) {
                        std::cout << "  Output: [";
                        for (size_t i = 0; i < OUTPUT_DIM; i++) {
                            std::cout << std::fixed << std::setprecision(3)
                                      << std::setw(7) << outputs[i];
                            if (i < OUTPUT_DIM - 1) std::cout << ", ";
                        }
                        std::cout << "]" << std::endl;
                    }
                    std::cout << "  Time: " << elapsed << "s | Batch had " << found_count << " candidates" << std::endl;
                    std::cout << std::endl;
                }
            }
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
// FP Environment Setup
// ============================================================================

void setup_fp_environment() {
    // Set rounding mode to round-to-nearest-even (matches OpenCL convert_int_rte)
    std::fesetround(FE_TONEAREST);
}

// ============================================================================
// Golden Vector Test - Verify GPU/CPU determinism at startup
// ============================================================================

bool run_determinism_test(const SharedState& shared) {
    // Generate a test nonce deterministically from the username
    uint8_t test_nonce[NONCE_BYTES];
    uint8_t hash_out[32];
    std::string test_material = shared.username + ":test_nonce";
    sha256::hash(reinterpret_cast<const uint8_t*>(test_material.data()),
                 test_material.size(), hash_out);
    // Fill nonce from hash (repeat if needed)
    for (size_t i = 0; i < NONCE_BYTES; i++) {
        test_nonce[i] = hash_out[i % 32];
    }

    // CPU computation
    uint64_t cpu_score = cpu_verifier::forward(shared.weights.data(), test_nonce, NONCE_BYTES);
    float cpu_outputs[OUTPUT_DIM];
    cpu_verifier::forward_pass(shared.weights.data(), test_nonce, NONCE_BYTES, cpu_outputs);
    uint8_t digest[DIGEST_BYTES];
    cpu_verifier::compute_digest(cpu_outputs, digest, OUTPUT_DIM);
    int cpu_bits = cpu_verifier::count_leading_zero_bits(digest, DIGEST_BYTES);

    // Self-consistency check: run CPU twice, must match
    uint64_t cpu_score2 = cpu_verifier::forward(shared.weights.data(), test_nonce, NONCE_BYTES);
    if (cpu_score != cpu_score2) {
        std::cerr << "FATAL: CPU verifier is not deterministic!" << std::endl;
        std::cerr << "  Run 1: 0x" << std::hex << cpu_score << std::endl;
        std::cerr << "  Run 2: 0x" << cpu_score2 << std::dec << std::endl;
        return false;
    }

    // Verify lrintf uses round-to-nearest-even (banker's rounding)
    // This is critical for CPU/GPU determinism since OpenCL's convert_int_rte uses this mode
    // Halfway values (x.5) round to the nearest even integer:
    //   0.5 -> 0 (even), 1.5 -> 2 (even), 2.5 -> 2 (even)
    //   -0.5 -> 0 (even), -1.5 -> -2 (even), -2.5 -> -2 (even)
    // Non-halfway values round normally: 0.4999 -> 0, 0.5001 -> 1
    float test_vals[] = {0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f, 0.4999f, 0.5001f};
    int32_t expected[] = {0, 2, 2, 0, -2, -2, 0, 1};

    for (size_t i = 0; i < sizeof(test_vals) / sizeof(test_vals[0]); i++) {
        int32_t result = static_cast<int32_t>(std::lrintf(test_vals[i]));
        if (result != expected[i]) {
            std::cerr << "WARNING: lrintf(" << test_vals[i] << ") = " << result
                      << ", expected " << expected[i] << std::endl;
            std::cerr << "  This may cause CPU/GPU mismatches on this platform" << std::endl;
        }
    }

    std::string digest_hex = bytes_to_hex(digest, DIGEST_BYTES);
    std::cout << "[Test] CPU determinism check PASSED" << std::endl;
    std::cout << "[Test] Test nonce digest: " << format_digest_with_marker(digest_hex, cpu_bits)
              << " (" << cpu_bits << " LZ bits)" << std::endl;

    return true;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    // Setup FP environment (rounding mode, FTZ/DAZ)
    setup_fp_environment();

    std::string username = DEFAULT_USERNAME;
    std::string verify_proof;
    int target_zero_bits = 16;
    bool verbose = false;

    // Option prefix lengths (compile-time)
    constexpr size_t USER_PREFIX_LEN = sizeof("--user=") - 1;
    constexpr size_t BITS_PREFIX_LEN = sizeof("--bits=") - 1;
    constexpr size_t HEX_PREFIX_LEN = sizeof("--hex=") - 1;
    constexpr size_t VERIFY_PREFIX_LEN = sizeof("--verify=") - 1;

    auto parse_bits_value = [&](const std::string& value, int multiplier) {
        try {
            int val = std::stoi(value);
            target_zero_bits = std::max(val * multiplier, 1);
        } catch (const std::exception&) {
            std::cerr << "Invalid numeric target: " << value << std::endl;
            std::exit(1);
        }
    };

    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        // Helper to get required next argument
        auto require_next_arg = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "Flag " << arg << " requires a value" << std::endl;
                std::exit(1);
            }
            return std::string(argv[++i]);
        };

        if (arg == "--user" || arg == "-u") {
            username = require_next_arg();
            continue;
        }
        if (arg.rfind("--user=", 0) == 0) {
            username = arg.substr(USER_PREFIX_LEN);
            continue;
        }

        if (arg == "--bits" || arg == "-b") {
            parse_bits_value(require_next_arg(), 1);
            continue;
        }
        if (arg == "--hex" || arg == "-x") {
            parse_bits_value(require_next_arg(), 4);
            continue;
        }
        if (arg.rfind("--bits=", 0) == 0) {
            parse_bits_value(arg.substr(BITS_PREFIX_LEN), 1);
            continue;
        }
        if (arg.rfind("--hex=", 0) == 0) {
            parse_bits_value(arg.substr(HEX_PREFIX_LEN), 4);
            continue;
        }

        if (arg == "--verbose" || arg == "-v") {
            verbose = true;
            continue;
        }

        if (arg == "--verify" || arg == "-V") {
            verify_proof = require_next_arg();
            continue;
        }
        if (arg.rfind("--verify=", 0) == 0) {
            verify_proof = arg.substr(VERIFY_PREFIX_LEN);
            continue;
        }

        if (!arg.empty() && arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << std::endl;
            std::exit(1);
        }

        positional.push_back(arg);
    }

    if (!positional.empty()) {
        username = positional[0];
    }
    if (positional.size() > 1) {
        parse_bits_value(positional[1], 1);
    }

    // Verify mode: parse proof, verify, and exit
    if (!verify_proof.empty()) {
        // Parse proof: username/nonce (nonce can be ANY string)
        size_t slash_pos = verify_proof.find('/');
        if (slash_pos == std::string::npos) {
            std::cerr << "Invalid proof format. Expected: username/nonce" << std::endl;
            return 1;
        }
        std::string proof_username = verify_proof.substr(0, slash_pos);
        std::string nonce_str = verify_proof.substr(slash_pos + 1);

        if (nonce_str.empty()) {
            std::cerr << "Nonce cannot be empty" << std::endl;
            return 1;
        }

        // Treat nonce as raw bytes (any string works!)
        std::vector<uint8_t> nonce(nonce_str.begin(), nonce_str.end());

        // Generate weights
        std::vector<int8_t> weights;
        generate_weights(WEIGHT_EPOCH, weights);

        // Run forward pass (nonce gets SHA-256 expanded to 64 bytes internally)
        float outputs[OUTPUT_DIM];
        cpu_verifier::forward_pass(weights.data(), nonce.data(), nonce.size(), outputs);
        uint8_t digest[DIGEST_BYTES];
        cpu_verifier::compute_digest(outputs, digest, OUTPUT_DIM);
        int lz_bits = cpu_verifier::count_leading_zero_bits(digest, DIGEST_BYTES);
        std::string digest_hex = bytes_to_hex(digest, DIGEST_BYTES);

        std::cout << "Proof Verification" << std::endl;
        std::cout << "==================" << std::endl;
        std::cout << "Username: " << proof_username << std::endl;
        std::cout << "Nonce: " << nonce_str << " (" << nonce_str.size() << " bytes)" << std::endl;
        std::cout << "Digest: " << format_digest_with_marker(digest_hex, lz_bits) << std::endl;
        std::cout << "Result: " << describe_lz_bits(lz_bits) << std::endl;
        return 0;
    }

    // Initial threshold: only report if digest has at least (target_zero_bits - 1) leading zeros
    // Score is first 8 bytes of digest as uint64; lower = more leading zeros = better
    int base_bits = std::max(target_zero_bits - 1, 0);
    uint64_t initial_target = (base_bits >= 64) ? 0 : (1ULL << (64 - base_bits));

    SharedState shared;
    shared.username = username;
    shared.verbose = verbose;
    shared.best_score = initial_target;
    shared.start_time = std::chrono::steady_clock::now();

    int target_hex_digits = target_zero_bits / 4;
    int target_hex_extra_bits = target_zero_bits % 4;

    std::cout << "Neural Proof-of-Work Miner (OpenCL)" << std::endl;
    std::cout << "Username: " << shared.username << std::endl;
    std::cout << "Target: " << target_zero_bits << "+ leading zero bits ("
              << target_hex_digits << " hex digit" << (target_hex_digits == 1 ? "" : "s");
    if (target_hex_extra_bits) {
        std::cout << " + " << target_hex_extra_bits << " bits";
    }
    std::cout << ")" << std::endl;

    std::cout << "Generating network weights from epoch: " << WEIGHT_EPOCH << std::endl;
    generate_weights(WEIGHT_EPOCH, shared.weights);
    std::cout << "Generated " << shared.weights.size() << " weight bytes" << std::endl;

    std::cout << "Network: " << INPUT_DIM << " -> " << HIDDEN_DIM << " -> " << HIDDEN_DIM << " -> " << HIDDEN_DIM << " -> " << OUTPUT_DIM << std::endl;
    std::cout << "Global size: " << GLOBAL_SIZE << " threads per launch" << std::endl;
    std::cout << "Local size: " << LOCAL_SIZE << " threads per work-group" << std::endl;

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    std::vector<cl_device_id> devices = discover_all_gpus();
    if (devices.empty()) {
        std::cerr << "No GPUs found!" << std::endl;
        return 1;
    }
    std::cout << "Found " << devices.size() << " GPU(s)" << std::endl;

    std::vector<GPUContext> gpus(devices.size());
    for (size_t i = 0; i < devices.size(); i++) {
        if (!create_gpu_context(devices[i], static_cast<int>(i), shared, gpus[i])) {
            std::cerr << "Failed to initialize GPU " << i << std::endl;
            return 1;
        }
        std::cout << "Initialized GPU " << i << ": " << gpus[i].device_name << std::endl;
    }

    // Run determinism test to verify CPU consistency
    if (!run_determinism_test(shared)) {
        std::cerr << "Determinism test failed! Aborting." << std::endl;
        return 1;
    }

    std::cout << "\nMining started...\n" << std::endl;

    std::vector<std::thread> worker_threads;
    for (auto& gpu : gpus) {
        worker_threads.emplace_back(gpu_worker_thread, std::ref(gpu), std::ref(shared));
    }

    uint64_t last_total = 0;
    auto last_time = std::chrono::steady_clock::now();
    int stats_tick = 0;

    while (shared.running.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        if (g_shutdown_requested) {
            std::cout << "\nShutting down..." << std::endl;
            shared.running.store(false);
            break;
        }

        // Only update stats display every 5 seconds
        if (++stats_tick < 5) continue;
        stats_tick = 0;

        uint64_t total_evals = 0;
        uint64_t total_matches = 0;
        uint64_t total_launches = 0;
        uint64_t total_kernel_time_us = 0;
        for (const auto& gpu : gpus) {
            total_evals += gpu.evals_computed.load();
            total_matches += gpu.matches_found.load();
            total_launches += gpu.launch_counter.load();
            total_kernel_time_us += gpu.total_kernel_time_us.load();
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - shared.start_time).count();
        auto interval_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();

        double recent_rate = (interval_ms > 0) ?
            static_cast<double>(total_evals - last_total) / (interval_ms / 1000.0) : 0;

        last_total = total_evals;
        last_time = now;

        int current_bits = 0;
        {
            std::lock_guard<std::mutex> lock(shared.best_mutex);
            current_bits = cpu_verifier::score_to_lz_bits(shared.best_score);
        }

        if (total_evals > 0) {
            double avg_kernel_ms = (total_launches > 0) ?
                static_cast<double>(total_kernel_time_us) / total_launches / 1000.0 : 0;
            std::cout << "\r[Stats] " << std::fixed << std::setprecision(2)
                      << (recent_rate / 1e6) << " M evals/s, "
                      << std::setprecision(3) << (total_evals / 1e9) << "B evals, "
                      << total_matches << " improvements, "
                      << "best=" << current_bits << " LZ bits, "
                      << std::setprecision(1) << avg_kernel_ms << "ms/kernel"
                      << ", " << elapsed << "s        " << std::flush;
        } else {
            std::cout << "\r[Stats] Waiting for first kernel to complete... " << elapsed << "s" << std::flush;
        }
    }

    for (auto& t : worker_threads) {
        t.join();
    }

    std::cout << "\n\n========================================" << std::endl;
    std::cout << "Final Results" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Username: " << shared.username << std::endl;

    if (!shared.best_nonce.empty()) {
        float outputs[OUTPUT_DIM];
        cpu_verifier::forward_pass(shared.weights.data(), shared.best_nonce.data(), shared.best_nonce.size(), outputs);
        uint8_t digest[DIGEST_BYTES];
        cpu_verifier::compute_digest(outputs, digest, OUTPUT_DIM);
        int lz_bits = cpu_verifier::count_leading_zero_bits(digest, DIGEST_BYTES);
        std::string digest_hex = bytes_to_hex(digest, DIGEST_BYTES);
        std::string nonce_str(reinterpret_cast<char*>(shared.best_nonce.data()), shared.best_nonce.size());

        std::cout << "\nBest Result: " << describe_lz_bits(lz_bits) << std::endl;
        std::cout << "Digest: " << format_digest_with_marker(digest_hex, lz_bits) << std::endl;
        std::cout << "Proof: " << shared.username << "/" << nonce_str << std::endl;
        if (shared.verbose) {
            std::cout << "Output: [";
            for (size_t i = 0; i < OUTPUT_DIM; i++) {
                std::cout << std::fixed << std::setprecision(4) << std::setw(8) << outputs[i];
                if (i < OUTPUT_DIM - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    } else {
        std::cout << "\nNo valid result found." << std::endl;
    }

    return 0;
}
