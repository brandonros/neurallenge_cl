// Neural Proof-of-Work - Verification Server
// Accepts proof submissions via HTTP, verifies PoW, logs valid proofs

#pragma STDC FENV_ACCESS ON
#pragma STDC FP_CONTRACT OFF

#include <algorithm>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
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
#include "config.h"

using json = nlohmann::json;

// ============================================================================
// Configuration
// ============================================================================

constexpr int MIN_BITS = 28;  // Minimum difficulty to accept
constexpr int SERVER_PORT = 8080;
constexpr const char* PROOFS_FILE = "proofs.jsonl";
constexpr const char* WEIGHT_EPOCH = "epoch0";

// ============================================================================
// CPU Verifier - Same as miner, ensures bit-exact verification
// ============================================================================

namespace cpu_verifier {

constexpr float WEIGHT_SCALE = 1.0f / 127.0f;
constexpr float INPUT_SCALE = 1.0f / 32768.0f;
constexpr float Q16_SCALE = 65536.0f;
constexpr float Q16_INV_SCALE = 1.0f / 65536.0f;
constexpr float Q16_MIN = -32768.0f;
constexpr float Q16_MAX = 32767.0f;

inline float q16(float x) {
    if (!std::isfinite(x)) x = 0.0f;
    x = std::max(Q16_MIN, std::min(Q16_MAX, x));
    int32_t i = static_cast<int32_t>(std::lrintf(x * Q16_SCALE));
    return static_cast<float>(i) * Q16_INV_SCALE;
}

inline int32_t q16_to_int(float q16_val) {
    return static_cast<int32_t>(std::lrintf(q16_val * Q16_SCALE));
}

inline float get_weight(const int8_t* weights, size_t idx) {
    return static_cast<float>(weights[idx]) * WEIGHT_SCALE;
}

void expand_nonce(const uint8_t* nonce, size_t nonce_len, uint8_t* expanded);

void expanded_to_input(const uint8_t* expanded, float* input) {
    for (int i = 0; i < INPUT_DIM; i++) {
        int16_t val = static_cast<int16_t>(expanded[i*2] | (expanded[i*2 + 1] << 8));
        input[i] = q16(static_cast<float>(val) * INPUT_SCALE);
    }
}

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

void layer_forward(
    const int8_t* W, const int8_t* bias,
    const float* input, float* output,
    size_t in_dim, size_t out_dim, bool use_relu
) {
    for (size_t i = 0; i < out_dim; i++) {
        float sum = matmul_row(W + i * in_dim, bias + i, input, in_dim);
        output[i] = use_relu ? std::max(0.0f, sum) : sum;
    }
}

// SipHash-2-4
inline uint64_t rotl64(uint64_t x, int b) {
    return (x << b) | (x >> (64 - b));
}

inline void sipround(uint64_t& v0, uint64_t& v1, uint64_t& v2, uint64_t& v3) {
    v0 += v1; v1 = rotl64(v1, 13); v1 ^= v0; v0 = rotl64(v0, 32);
    v2 += v3; v3 = rotl64(v3, 16); v3 ^= v2;
    v0 += v3; v3 = rotl64(v3, 21); v3 ^= v0;
    v2 += v1; v1 = rotl64(v1, 17); v1 ^= v2; v2 = rotl64(v2, 32);
}

uint64_t siphash_2_4_132(const uint8_t* data, uint64_t k0, uint64_t k1) {
    uint64_t v0 = k0 ^ 0x736f6d6570736575ULL;
    uint64_t v1 = k1 ^ 0x646f72616e646f6dULL;
    uint64_t v2 = k0 ^ 0x6c7967656e657261ULL;
    uint64_t v3 = k1 ^ 0x7465646279746573ULL;

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

    v2 ^= 0xff;
    sipround(v0, v1, v2, v3);
    sipround(v0, v1, v2, v3);
    sipround(v0, v1, v2, v3);
    sipround(v0, v1, v2, v3);

    return v0 ^ v1 ^ v2 ^ v3;
}

void compute_digest(const float* output, uint8_t* digest, size_t dim) {
    uint8_t input[SIPHASH_INPUT_BYTES];
    for (size_t i = 0; i < dim; i++) {
        int32_t val = q16_to_int(output[i]);
        input[i * 4 + 0] = static_cast<uint8_t>(val & 0xFF);
        input[i * 4 + 1] = static_cast<uint8_t>((val >> 8) & 0xFF);
        input[i * 4 + 2] = static_cast<uint8_t>((val >> 16) & 0xFF);
        input[i * 4 + 3] = static_cast<uint8_t>((val >> 24) & 0xFF);
    }
    input[SERIALIZED_OUTPUT_BYTES + 1] = 0;
    input[SERIALIZED_OUTPUT_BYTES + 2] = 0;
    input[SERIALIZED_OUTPUT_BYTES + 3] = 0;

    for (size_t block = 0; block < (DIGEST_BYTES / 8); block++) {
        input[SERIALIZED_OUTPUT_BYTES] = static_cast<uint8_t>(block);
        uint64_t h = siphash_2_4_132(input, SIPHASH_K0, SIPHASH_K1);
        for (int i = 0; i < 8; i++) {
            digest[block * 8 + i] = static_cast<uint8_t>((h >> (i * 8)) & 0xFF);
        }
    }
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

void forward_pass(const int8_t* weights, const uint8_t* nonce, size_t nonce_len, float* output) {
    float input[INPUT_DIM];
    float hidden1[HIDDEN_DIM];
    float hidden2[HIDDEN_DIM];
    float hidden3[HIDDEN_DIM];

    nonce_to_input(nonce, nonce_len, input);

    layer_forward(weights + W1_OFFSET, weights + B1_OFFSET, input, hidden1, INPUT_DIM, HIDDEN_DIM, true);
    layer_forward(weights + W2_OFFSET, weights + B2_OFFSET, hidden1, hidden2, HIDDEN_DIM, HIDDEN_DIM, true);
    layer_forward(weights + W3_OFFSET, weights + B3_OFFSET, hidden2, hidden3, HIDDEN_DIM, HIDDEN_DIM, true);
    layer_forward(weights + W4_OFFSET, weights + B4_OFFSET, hidden3, output, HIDDEN_DIM, OUTPUT_DIM, false);
}

} // namespace cpu_verifier

// ============================================================================
// SHA-256 for weight generation and nonce expansion
// ============================================================================

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
            w[i] = (padded[block + i*4] << 24) | (padded[block + i*4 + 1] << 16) |
                   (padded[block + i*4 + 2] << 8) | padded[block + i*4 + 3];
        }
        for (int i = 16; i < 64; i++) {
            w[i] = SIG1(w[i-2]) + w[i-7] + SIG0(w[i-15]) + w[i-16];
        }

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

namespace cpu_verifier {
void expand_nonce(const uint8_t* nonce, size_t nonce_len, uint8_t* expanded) {
    std::vector<uint8_t> buf(nonce_len + 1);
    std::memcpy(buf.data(), nonce, nonce_len);
    buf[nonce_len] = 0x00;
    sha256::hash(buf.data(), buf.size(), expanded);
    buf[nonce_len] = 0x01;
    sha256::hash(buf.data(), buf.size(), expanded + 32);
}
}

// ============================================================================
// Weight Generation
// ============================================================================

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
// Utilities
// ============================================================================

std::string bytes_to_hex(const uint8_t* data, size_t len) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; i++) {
        ss << std::setw(2) << static_cast<int>(data[i]);
    }
    return ss.str();
}

double compute_reward(int bits) {
    if (bits < MIN_BITS) return 0.0;
    return std::pow(1.5, bits - MIN_BITS);
}

// ============================================================================
// Server State
// ============================================================================

struct ServerState {
    std::vector<int8_t> weights;
    std::unordered_set<std::string> seen_nonces;
    std::mutex mutex;
    std::ofstream proof_log;
    uint64_t total_proofs = 0;
    uint64_t total_rejected = 0;
};

// ============================================================================
// Signal Handling
// ============================================================================

volatile std::sig_atomic_t g_shutdown = 0;

void signal_handler(int) {
    g_shutdown = 1;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    std::fesetround(FE_TONEAREST);

    int port = SERVER_PORT;
    if (argc > 1) {
        port = std::atoi(argv[1]);
    }

    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    ServerState state;

    std::cout << "Generating weights from epoch: " << WEIGHT_EPOCH << std::endl;
    generate_weights(WEIGHT_EPOCH, state.weights);
    std::cout << "Generated " << state.weights.size() << " weight bytes" << std::endl;

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
        j["total_proofs"] = state.total_proofs;
        j["total_rejected"] = state.total_rejected;
        j["seen_nonces"] = state.seen_nonces.size();
        j["min_bits"] = MIN_BITS;
        res.set_content(j.dump(), "application/json");
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

            // Check for duplicate
            {
                std::lock_guard<std::mutex> lock(state.mutex);
                if (state.seen_nonces.count(nonce_str)) {
                    response["error"] = "duplicate nonce";
                    state.total_rejected++;
                    res.status = 400;
                    res.set_content(response.dump(), "application/json");
                    return;
                }
            }

            // Verify proof
            std::vector<uint8_t> nonce(nonce_str.begin(), nonce_str.end());
            float outputs[OUTPUT_DIM];
            cpu_verifier::forward_pass(state.weights.data(), nonce.data(), nonce.size(), outputs);

            uint8_t digest[DIGEST_BYTES];
            cpu_verifier::compute_digest(outputs, digest, OUTPUT_DIM);
            int bits = cpu_verifier::count_leading_zero_bits(digest, DIGEST_BYTES);

            if (bits < MIN_BITS) {
                response["error"] = "difficulty too low";
                response["bits"] = bits;
                response["min_bits"] = MIN_BITS;
                std::lock_guard<std::mutex> lock(state.mutex);
                state.total_rejected++;
                res.status = 400;
                res.set_content(response.dump(), "application/json");
                return;
            }

            double reward = compute_reward(bits);
            std::string digest_hex = bytes_to_hex(digest, DIGEST_BYTES);
            auto now = std::chrono::system_clock::now();
            auto timestamp = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();

            // Log proof
            {
                std::lock_guard<std::mutex> lock(state.mutex);
                state.seen_nonces.insert(nonce_str);
                state.total_proofs++;

                json proof;
                proof["timestamp"] = timestamp;
                proof["wallet"] = wallet;
                proof["nonce"] = nonce_str;
                proof["bits"] = bits;
                proof["reward"] = reward;
                proof["digest"] = digest_hex;

                state.proof_log << proof.dump() << std::endl;
                state.proof_log.flush();

                std::cout << "[Proof] " << bits << " bits | wallet=" << wallet.substr(0, 8) << "... | reward=" << reward << std::endl;
            }

            response["success"] = true;
            response["bits"] = bits;
            response["reward"] = reward;
            response["digest"] = digest_hex;
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
    std::cout << "  POST /submit - Submit proof {wallet, nonce}" << std::endl;

    svr.listen("0.0.0.0", port);

    state.proof_log.close();
    std::cout << "\nServer stopped." << std::endl;

    return 0;
}
