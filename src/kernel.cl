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

// ============================================================================
// Matrix-Vector Multiply (scalar for determinism)
// ============================================================================

// Computes one output element: dot(W[row,:], input) + bias
inline float matmul_row(
    __global const char* W,      // Row of weights [in_dim]
    __global const char* bias,   // Single bias value
    const float* input,          // [in_dim] input vector
    uint in_dim
) {
    float sum = 0.0f;

    for (uint i = 0; i < in_dim; i++) {
        float prod = get_weight(W, i) * input[i];
        sum = sum + prod;

        if ((i & 31u) == 31u) {
            sum = q16(sum);
        }
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
// Nonce to Input Conversion
// ============================================================================

inline void nonce_to_input(const uchar* nonce, float* input) {
    for (int i = 0; i < INPUT_DIM; i++) {
        // Little-endian int16
        short val = (short)(nonce[i*2] | (nonce[i*2 + 1] << 8));
        input[i] = q16((float)val * INPUT_SCALE);
    }
}

// ============================================================================
// RNG (xoroshiro64**)
// ============================================================================

#define ROTL32(x, k) (((x) << (k)) | ((x) >> (32 - (k))))

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

inline void generate_nonce(uint* s0, uint* s1, uchar* nonce) {
    // Generate NONCE_BYTES using 32-bit RNG outputs (4 bytes each)
    for (int i = 0; i < (NONCE_BYTES / 4); i++) {
        uint r = xoroshiro64_next(s0, s1);
        nonce[i*4 + 0] = (r >> 0) & 0xFF;
        nonce[i*4 + 1] = (r >> 8) & 0xFF;
        nonce[i*4 + 2] = (r >> 16) & 0xFF;
        nonce[i*4 + 3] = (r >> 24) & 0xFF;
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

        // Convert nonce to input
        nonce_to_input(nonce, input);

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
