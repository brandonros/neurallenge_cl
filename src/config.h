// neurallenge_config.h - Shared constants for OpenCL kernel and C++ host
// This file must be valid C (for OpenCL) and C++ (for host)

#ifndef NEURALLENGE_CONFIG_H
#define NEURALLENGE_CONFIG_H

// Network architecture: 32 -> 256 -> 256 -> 256 -> 32
#define INPUT_DIM 32
#define HIDDEN_DIM 256
#define OUTPUT_DIM 32

// Digest size
#define DIGEST_BYTES OUTPUT_DIM

// Nonce size for kernel mining (bytes) - gets SHA-256 expanded to 64 bytes
// 16 bytes = 22 base64 chars, provides plenty of entropy
#define NONCE_BYTES 16

// Serialized output size for hashing
#define SERIALIZED_OUTPUT_BYTES (OUTPUT_DIM * 4)

// SipHash input size (serialized output + 4-byte domain separator)
#define SIPHASH_INPUT_BYTES (SERIALIZED_OUTPUT_BYTES + 4)

// Weight sizes
#define W1_SIZE (HIDDEN_DIM * INPUT_DIM)
#define B1_SIZE (HIDDEN_DIM)
#define W2_SIZE (HIDDEN_DIM * HIDDEN_DIM)
#define B2_SIZE (HIDDEN_DIM)
#define W3_SIZE (HIDDEN_DIM * HIDDEN_DIM)
#define B3_SIZE (HIDDEN_DIM)
#define W4_SIZE (OUTPUT_DIM * HIDDEN_DIM)
#define B4_SIZE (OUTPUT_DIM)
#define TOTAL_WEIGHTS (W1_SIZE + B1_SIZE + W2_SIZE + B2_SIZE + W3_SIZE + B3_SIZE + W4_SIZE + B4_SIZE)

// Weight offsets
#define W1_OFFSET 0
#define B1_OFFSET (W1_OFFSET + W1_SIZE)
#define W2_OFFSET (B1_OFFSET + B1_SIZE)
#define B2_OFFSET (W2_OFFSET + W2_SIZE)
#define W3_OFFSET (B2_OFFSET + B2_SIZE)
#define B3_OFFSET (W3_OFFSET + W3_SIZE)
#define W4_OFFSET (B3_OFFSET + B3_SIZE)
#define B4_OFFSET (W4_OFFSET + W4_SIZE)

// Result buffer limits
#define MAX_RESULTS 16

// SipHash keys (derived from "NeuralPoW Digest" ASCII)
#define SIPHASH_K0 0x4e657572616c506fUL
#define SIPHASH_K1 0x5720446967657374UL

#endif // NEURALLENGE_CONFIG_H
