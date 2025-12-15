// Neural Proof-of-Work - OpenCL Kernel v2
// FP32-heavy PoW with proper determinism via fixed-point semantics
//
// Acknowledgment: This is "float-flavored fixed-point" - the compute is FP32,
// but determinism comes from quantizing to a fixed grid after every operation.

#pragma OPENCL FP_CONTRACT OFF

// ============================================================================
// Atomic compatibility shim (OpenCL 1.2 vs 2.0+)
// ============================================================================
#if defined(__opencl_c_atomic_order_relaxed) && defined(__opencl_c_atomic_scope_device)
// OpenCL 2.0+ path
#define FOUND_COUNT_T __global atomic_uint*
inline uint reserve_slot(FOUND_COUNT_T p) {
    return atomic_fetch_add_explicit(p, 1u, memory_order_relaxed, memory_scope_device);
}
#else
// OpenCL 1.2 path (macOS)
#define FOUND_COUNT_T __global volatile uint*
inline uint reserve_slot(FOUND_COUNT_T p) {
    return atomic_inc(p);
}
#endif

// ============================================================================
// Quantization - Bit-exact fixed-point conversion
// ============================================================================

// Q16 fixed-point constants (16 fractional bits)
#define Q16_SCALE 65536.0f
#define Q16_INV_SCALE (1.0f / 65536.0f)
#define Q16_MIN -32768.0f
#define Q16_MAX 32767.0f

// Quantization using convert_int_rte (OpenCL spec-defined round-to-nearest-even)
// This is the most deterministic option the spec provides
inline float q16(float x) {
    if (!isfinite(x)) x = 0.0f;
    x = clamp(x, Q16_MIN, Q16_MAX);
    int i = convert_int_rte(x * Q16_SCALE);
    return (float)i * Q16_INV_SCALE;
}

// Get the integer representation of a q16 value (for deterministic scoring)
inline int q16_to_int(float q16_val) {
    return convert_int_rte(q16_val * Q16_SCALE);
}

// ============================================================================
// Weight Access - int8 weights scaled to ~[-1, 1]
// ============================================================================

#define WEIGHT_SCALE (1.0f / 127.0f)   // int8 weights in [-127, 127]
#define INPUT_SCALE (1.0f / 32768.0f)  // int16 inputs in [-32768, 32767]

inline float get_weight(__global const char* weights, uint idx) {
    return (float)weights[idx] * WEIGHT_SCALE;
}

// Vectorized weight conversion - convert 8 chars to 8 floats
inline void weights_to_floats8(char8 w, float* out) {
    out[0] = (float)w.s0 * WEIGHT_SCALE;
    out[1] = (float)w.s1 * WEIGHT_SCALE;
    out[2] = (float)w.s2 * WEIGHT_SCALE;
    out[3] = (float)w.s3 * WEIGHT_SCALE;
    out[4] = (float)w.s4 * WEIGHT_SCALE;
    out[5] = (float)w.s5 * WEIGHT_SCALE;
    out[6] = (float)w.s6 * WEIGHT_SCALE;
    out[7] = (float)w.s7 * WEIGHT_SCALE;
}

// ============================================================================
// Matrix-Vector Multiply - Vectorized for better memory throughput
// ============================================================================

// Computes one output element: dot(W[row,:], input) + bias
// Uses 8-wide vector loads for better memory bandwidth utilization
// Maintains exact sequential accumulation order for determinism
inline float matmul_row(
    __global const char* W,      // Row of weights [in_dim]
    __global const char* bias,   // Single bias value
    const float* input,          // [in_dim] input vector
    uint in_dim
) {
    float sum = 0.0f;
    float wf[8];

    // Process 32 elements at a time (4x vload8), then quantize
    // Maintains exact sequential accumulation order: sum = ((sum + a) + b) + c ...
    uint i = 0;
    for (; i + 32 <= in_dim; i += 32) {
        // Load weights 0-7 with single memory transaction
        char8 w0 = vload8(0, W + i);
        weights_to_floats8(w0, wf);
        sum = sum + wf[0] * input[i+0];
        sum = sum + wf[1] * input[i+1];
        sum = sum + wf[2] * input[i+2];
        sum = sum + wf[3] * input[i+3];
        sum = sum + wf[4] * input[i+4];
        sum = sum + wf[5] * input[i+5];
        sum = sum + wf[6] * input[i+6];
        sum = sum + wf[7] * input[i+7];

        // Load weights 8-15
        char8 w1 = vload8(0, W + i + 8);
        weights_to_floats8(w1, wf);
        sum = sum + wf[0] * input[i+8];
        sum = sum + wf[1] * input[i+9];
        sum = sum + wf[2] * input[i+10];
        sum = sum + wf[3] * input[i+11];
        sum = sum + wf[4] * input[i+12];
        sum = sum + wf[5] * input[i+13];
        sum = sum + wf[6] * input[i+14];
        sum = sum + wf[7] * input[i+15];

        // Load weights 16-23
        char8 w2 = vload8(0, W + i + 16);
        weights_to_floats8(w2, wf);
        sum = sum + wf[0] * input[i+16];
        sum = sum + wf[1] * input[i+17];
        sum = sum + wf[2] * input[i+18];
        sum = sum + wf[3] * input[i+19];
        sum = sum + wf[4] * input[i+20];
        sum = sum + wf[5] * input[i+21];
        sum = sum + wf[6] * input[i+22];
        sum = sum + wf[7] * input[i+23];

        // Load weights 24-31
        char8 w3 = vload8(0, W + i + 24);
        weights_to_floats8(w3, wf);
        sum = sum + wf[0] * input[i+24];
        sum = sum + wf[1] * input[i+25];
        sum = sum + wf[2] * input[i+26];
        sum = sum + wf[3] * input[i+27];
        sum = sum + wf[4] * input[i+28];
        sum = sum + wf[5] * input[i+29];
        sum = sum + wf[6] * input[i+30];
        sum = sum + wf[7] * input[i+31];

        // Quantize every 32 elements (same as original)
        sum = q16(sum);
    }

    // Handle remaining elements (for in_dim not divisible by 32, e.g., INPUT_DIM=32)
    // Process in chunks of 8 with sequential accumulation
    for (; i + 8 <= in_dim; i += 8) {
        char8 w = vload8(0, W + i);
        weights_to_floats8(w, wf);
        sum = sum + wf[0] * input[i+0];
        sum = sum + wf[1] * input[i+1];
        sum = sum + wf[2] * input[i+2];
        sum = sum + wf[3] * input[i+3];
        sum = sum + wf[4] * input[i+4];
        sum = sum + wf[5] * input[i+5];
        sum = sum + wf[6] * input[i+6];
        sum = sum + wf[7] * input[i+7];
    }
    // Remaining < 8 elements (scalar)
    for (; i < in_dim; i++) {
        sum = sum + get_weight(W, i) * input[i];
    }

    sum += (float)(*bias) * WEIGHT_SCALE;

    return q16(sum);
}

// Full layer: output = activation(W @ input + bias)
// Note: matmul_row returns q16, and ReLU(q16) stays on grid (0.0f or unchanged)
inline void layer_forward(
    __global const char* W,      // [out_dim x in_dim]
    __global const char* bias,   // [out_dim]
    const float* input,          // [in_dim]
    float* output,               // [out_dim]
    uint in_dim,
    uint out_dim,
    int use_relu                 // 1 for hidden layers, 0 for output
) {
    for (uint i = 0; i < out_dim; i++) {
        float sum = matmul_row(W + i * in_dim, bias + i, input, in_dim);
        output[i] = use_relu ? fmax(0.0f, sum) : sum;
    }
}

// ============================================================================
// SHA-256 for nonce expansion (any string -> 64 bytes)
// ============================================================================

__constant uint SHA256_K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

#define SHA_CH(x, y, z)    bitselect((z), (y), (x))
#define SHA_MAJ(x, y, z)   (((x) & (y)) ^ ((x) & (z)) ^ ((y) & (z)))
#define SHA_BSIG0(x)       (rotate((x), 30u) ^ rotate((x), 19u) ^ rotate((x), 10u))
#define SHA_BSIG1(x)       (rotate((x), 26u) ^ rotate((x), 21u) ^ rotate((x), 7u))
#define SHA_SSIG0(x)       (rotate((x), 25u) ^ rotate((x), 14u) ^ ((x) >> 3))
#define SHA_SSIG1(x)       (rotate((x), 15u) ^ rotate((x), 13u) ^ ((x) >> 10))

// SHA-256 for short inputs (up to 55 bytes, fits in one block with padding)
// Output: 32 bytes
inline void sha256_short(const uchar* data, uint len, uchar* out) {
    uint w[16];

    // Clear w
    for (int i = 0; i < 16; i++) w[i] = 0;

    // Copy data (big-endian)
    for (uint i = 0; i < len; i++) {
        w[i >> 2] |= ((uint)data[i]) << (24 - 8 * (i & 3));
    }

    // Append 1 bit
    w[len >> 2] |= 0x80u << (24 - 8 * (len & 3));

    // Append length in bits (big-endian, len < 56 so fits in w[15])
    w[15] = len * 8;

    // Initial hash values
    uint H[8] = {
        0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
        0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u
    };

    uint a = H[0], b = H[1], c = H[2], d = H[3];
    uint e = H[4], f = H[5], g = H[6], h = H[7];

    // Rounds 0-15
    for (int i = 0; i < 16; i++) {
        uint t1 = h + SHA_BSIG1(e) + SHA_CH(e, f, g) + SHA256_K[i] + w[i];
        uint t2 = SHA_BSIG0(a) + SHA_MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    // Rounds 16-63
    for (int i = 16; i < 64; i++) {
        int j = i & 0xF;
        w[j] = SHA_SSIG1(w[(j + 14) & 0xF]) + w[(j + 9) & 0xF] +
               SHA_SSIG0(w[(j + 1) & 0xF]) + w[j];
        uint t1 = h + SHA_BSIG1(e) + SHA_CH(e, f, g) + SHA256_K[i] + w[j];
        uint t2 = SHA_BSIG0(a) + SHA_MAJ(a, b, c);
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }

    // Final hash
    H[0] += a; H[1] += b; H[2] += c; H[3] += d;
    H[4] += e; H[5] += f; H[6] += g; H[7] += h;

    // Output (big-endian)
    for (int i = 0; i < 8; i++) {
        out[i*4]     = (H[i] >> 24) & 0xFF;
        out[i*4 + 1] = (H[i] >> 16) & 0xFF;
        out[i*4 + 2] = (H[i] >> 8) & 0xFF;
        out[i*4 + 3] = H[i] & 0xFF;
    }
}

// Expand nonce to 64 bytes using SHA-256
// expanded = SHA256(nonce || 0x00) || SHA256(nonce || 0x01)
inline void expand_nonce(const uchar* nonce, uint nonce_len, uchar* expanded) {
    uchar buf[NONCE_BYTES + 1];

    // Copy nonce and append suffix byte
    for (uint i = 0; i < nonce_len; i++) {
        buf[i] = nonce[i];
    }

    // First half: SHA256(nonce || 0x00)
    buf[nonce_len] = 0x00;
    sha256_short(buf, nonce_len + 1, expanded);

    // Second half: SHA256(nonce || 0x01)
    buf[nonce_len] = 0x01;
    sha256_short(buf, nonce_len + 1, expanded + 32);
}

// ============================================================================
// Nonce to Input Conversion (via SHA-256 expansion)
// ============================================================================

// Convert 64 expanded bytes to INPUT_DIM floats
inline void expanded_to_input(const uchar* expanded, float* input) {
    for (int i = 0; i < INPUT_DIM; i++) {
        // Little-endian int16
        short val = (short)(expanded[i*2] | (expanded[i*2 + 1] << 8));
        input[i] = q16((float)val * INPUT_SCALE);
    }
}

// Full nonce-to-input: any nonce -> SHA256 expand -> 32 floats
inline void nonce_to_input(const uchar* nonce, uint nonce_len, float* input) {
    uchar expanded[64];
    expand_nonce(nonce, nonce_len, expanded);
    expanded_to_input(expanded, input);
}

// ============================================================================
// RNG (xoroshiro64**)
// ============================================================================

#define ROTL32(x, k) (((x) << (k)) | ((x) >> (32 - (k))))

// Base64 character set for nonce generation
__constant uchar BASE64_CHARS[64] = {
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '/'
};

inline uint splitmix32(uint x) {
    x += 0x9e3779b9u;
    x = (x ^ (x >> 16)) * 0x85ebca6bu;
    x = (x ^ (x >> 13)) * 0xc2b2ae35u;
    return x ^ (x >> 16);
}

inline void init_rng(uint thread_idx, uint seed_lo, uint seed_hi, uint* s0, uint* s1) {
    *s0 = splitmix32(seed_lo ^ thread_idx);
    *s1 = splitmix32(seed_hi ^ (thread_idx * 0x9e3779b9u));
    if (*s0 == 0 && *s1 == 0) *s0 = 1;
}

inline uint xoroshiro64_next(uint* s0, uint* s1) {
    uint result = ROTL32(*s0 * 0x9E3779BBu, 5) * 5;
    uint t = *s1 ^ *s0;
    *s0 = ROTL32(*s0, 26) ^ t ^ (t << 9);
    *s1 = ROTL32(t, 13);
    return result;
}

// Generate base64 characters directly (like Shallenge)
// This ensures what's stored IS what's displayed - no encoding/decoding needed
inline void generate_nonce(uint* s0, uint* s1, uchar* nonce) {
    for (int i = 0; i < NONCE_BYTES; ) {
        uint bits = xoroshiro64_next(s0, s1);
        // Extract 5 base64 chars per 32-bit word (6 bits each, 30 bits used)
        for (int j = 0; j < 5 && i < NONCE_BYTES; j++, i++) {
            nonce[i] = BASE64_CHARS[bits & 63];
            bits >>= 6;
        }
    }
}

// Network architecture constants come from neurallenge_config.h (prepended by Makefile)

// ============================================================================
// SipHash-2-4 - Proper cryptographic mixing for digest
// ============================================================================

inline ulong rotl64(ulong x, int b) {
    return (x << b) | (x >> (64 - b));
}

inline void sipround(ulong* v0, ulong* v1, ulong* v2, ulong* v3) {
    *v0 += *v1; *v1 = rotl64(*v1, 13); *v1 ^= *v0; *v0 = rotl64(*v0, 32);
    *v2 += *v3; *v3 = rotl64(*v3, 16); *v3 ^= *v2;
    *v0 += *v3; *v3 = rotl64(*v3, 21); *v3 ^= *v0;
    *v2 += *v1; *v1 = rotl64(*v1, 17); *v1 ^= *v2; *v2 = rotl64(*v2, 32);
}

// SipHash-2-4 with 128-bit key, returns 64-bit hash
// Specialized for 132-byte input (128 bytes data + 4 bytes domain separator)
inline ulong siphash_2_4_132(const uchar* data, ulong k0, ulong k1) {
    ulong v0 = k0 ^ 0x736f6d6570736575UL;
    ulong v1 = k1 ^ 0x646f72616e646f6dUL;
    ulong v2 = k0 ^ 0x6c7967656e657261UL;
    ulong v3 = k1 ^ 0x7465646279746573UL;

    // Process 16 full 8-byte blocks (128 bytes)
    for (int blk = 0; blk < 16; blk++) {
        const uchar* p = data + blk * 8;
        ulong m = (ulong)p[0]
                | ((ulong)p[1] << 8)
                | ((ulong)p[2] << 16)
                | ((ulong)p[3] << 24)
                | ((ulong)p[4] << 32)
                | ((ulong)p[5] << 40)
                | ((ulong)p[6] << 48)
                | ((ulong)p[7] << 56);
        v3 ^= m;
        sipround(&v0, &v1, &v2, &v3);
        sipround(&v0, &v1, &v2, &v3);
        v0 ^= m;
    }

    // Process final 4 bytes + length (132 = 0x84)
    const uchar* p = data + 128;
    ulong b = (132UL << 56)
            | (ulong)p[0]
            | ((ulong)p[1] << 8)
            | ((ulong)p[2] << 16)
            | ((ulong)p[3] << 24);

    v3 ^= b;
    sipround(&v0, &v1, &v2, &v3);
    sipround(&v0, &v1, &v2, &v3);
    v0 ^= b;

    // Finalization
    v2 ^= 0xff;
    sipround(&v0, &v1, &v2, &v3);
    sipround(&v0, &v1, &v2, &v3);
    sipround(&v0, &v1, &v2, &v3);
    sipround(&v0, &v1, &v2, &v3);

    return v0 ^ v1 ^ v2 ^ v3;
}

// SipHash keys defined in neurallenge_config.h

// ============================================================================
// Score Computation - Neural digest leading zeros
// ============================================================================

inline void compute_digest(const float* output, uchar* digest, uint dim) {
    // Serialize q16 integers directly into SipHash input buffer
    uchar input[SIPHASH_INPUT_BYTES];
    for (uint i = 0; i < dim; i++) {
        int val = q16_to_int(output[i]);
        input[i * 4 + 0] = (uchar)(val & 0xFF);
        input[i * 4 + 1] = (uchar)((val >> 8) & 0xFF);
        input[i * 4 + 2] = (uchar)((val >> 16) & 0xFF);
        input[i * 4 + 3] = (uchar)((val >> 24) & 0xFF);
    }
    // Zero-pad domain separator (only first byte changes per block)
    input[SERIALIZED_OUTPUT_BYTES + 1] = 0;
    input[SERIALIZED_OUTPUT_BYTES + 2] = 0;
    input[SERIALIZED_OUTPUT_BYTES + 3] = 0;

    // Generate DIGEST_BYTES of digest using SipHash with domain separation
    for (uint block = 0; block < (DIGEST_BYTES / 8); block++) {
        input[SERIALIZED_OUTPUT_BYTES] = (uchar)block;
        ulong h = siphash_2_4_132(input, SIPHASH_K0, SIPHASH_K1);

        // Extract 8 bytes from hash (little-endian)
        for (int i = 0; i < 8; i++) {
            digest[block * 8 + i] = (uchar)((h >> (i * 8)) & 0xFF);
        }
    }
}

// Score = first 8 bytes of digest as big-endian uint64
// Lower value = more leading zeros = better
inline ulong compute_score(const float* output, uint dim) {
    uchar digest[DIGEST_BYTES];
    compute_digest(output, digest, dim);
    // Big-endian: first byte is most significant
    return ((ulong)digest[0] << 56) | ((ulong)digest[1] << 48) |
           ((ulong)digest[2] << 40) | ((ulong)digest[3] << 32) |
           ((ulong)digest[4] << 24) | ((ulong)digest[5] << 16) |
           ((ulong)digest[6] << 8)  | (ulong)digest[7];
}

// ============================================================================
// Main Kernel
// ============================================================================

__kernel void neural_pow_mine(
    __global const char* weights,
    ulong target_score,              // First 8 bytes of best digest (lower = better)
    uint seed_lo,
    uint seed_hi,
    FOUND_COUNT_T found_count,       // Atomic counter (CL1.2 or CL2.0+ via shim)
    __global ulong* found_scores,    // First 8 bytes of digest
    __global uchar* found_nonces
) {
    uint thread_idx = (uint)get_global_id(0);

    // Initialize RNG
    uint s0, s1;
    init_rng(thread_idx, seed_lo, seed_hi, &s0, &s1);

    // Activation buffers
    float input[INPUT_DIM];
    float hidden1[HIDDEN_DIM];
    float hidden2[HIDDEN_DIM];
    float hidden3[HIDDEN_DIM];
    float output[OUTPUT_DIM];
    uchar nonce[NONCE_BYTES];

    for (int iter = 0; iter < HASHES_PER_THREAD; iter++) {
        // Generate random nonce
        generate_nonce(&s0, &s1, nonce);

        // Convert nonce to input (hash expansion)
        nonce_to_input(nonce, NONCE_BYTES, input);

        // Forward pass
        // Layer 1: 32 -> 256, ReLU
        layer_forward(
            weights + W1_OFFSET, weights + B1_OFFSET,
            input, hidden1,
            INPUT_DIM, HIDDEN_DIM, 1
        );

        // Layer 2: 256 -> 256, ReLU
        layer_forward(
            weights + W2_OFFSET, weights + B2_OFFSET,
            hidden1, hidden2,
            HIDDEN_DIM, HIDDEN_DIM, 1
        );

        // Layer 3: 256 -> 256, ReLU
        layer_forward(
            weights + W3_OFFSET, weights + B3_OFFSET,
            hidden2, hidden3,
            HIDDEN_DIM, HIDDEN_DIM, 1
        );

        // Layer 4: 256 -> 32, linear
        layer_forward(
            weights + W4_OFFSET, weights + B4_OFFSET,
            hidden3, output,
            HIDDEN_DIM, OUTPUT_DIM, 0
        );

        // Compute score (first 8 bytes of digest as uint64)
        ulong score = compute_score(output, OUTPUT_DIM);

        // Check target (lower = better)
        if (score < target_score) {
            uint slot = reserve_slot(found_count);

            if (slot < MAX_RESULTS) {
                found_scores[slot] = score;

                __global uchar* nonce_out = found_nonces + slot * NONCE_BYTES;
                for (int i = 0; i < NONCE_BYTES; i++) {
                    nonce_out[i] = nonce[i];
                }
            }
        }
    }
}
