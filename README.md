# neurallenge

GPU proof-of-work using FP32 neural network inference instead of SHA-256.

## How it works

1. Username → SHA-256 → deterministic int8 weights (~148KB)
2. Random 64-byte nonce → network input (32 floats)
3. Forward pass: 32 → 256 → 256 → 256 → 32 MLP with ReLU
4. Output → SipHash-2-4 → 32-byte digest
5. Count leading zero bits in digest (more = better)

All FP32 ops are quantized to Q16.16 fixed-point grid for cross-platform determinism.

## vs Shallenge

| | Shallenge | Neurallenge |
|---|---|---|
| Core operation | SHA-256 (integer) | MLP inference (FP32) |
| ASIC-friendly | Yes | No - requires FP32 units |
| Ops per hash | ~1000 int ops | ~200K FP32 ops |
| Memory | Minimal | ~148KB weights + activations |

## Build

```bash
make
./output/neurallenge brandonros --bits 24
```

## Usage

```
./output/neurallenge [OPTIONS] [USERNAME] [BITS]

Options:
  -u, --user STRING        Username (default: from Makefile)
  -b, --bits N             Target leading zero bits
  -x, --hex N              Target leading zero hex digits (N*4 bits)
  -v, --verbose            Show network outputs
```

## Proof format

```
username/nonce → digest
```

Anyone can verify by deriving weights from the username, running the forward pass with the nonce, and counting leading zeros in the digest.
