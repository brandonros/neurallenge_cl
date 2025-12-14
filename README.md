# neurallenge

GPU proof-of-work using FP32 neural network inference.

## How it works

Like Shallenge, but with a neural network in the middle:

1. **Epoch string → SHA-256 → ~148KB of neural network weights**
   Everyone mines the same network for a fair leaderboard.

2. **Random nonce → neural network → 32 output values**
   The network has 4 layers (32→256→256→256→32) of FP32 multiply-adds.

3. **Output → SipHash-2-4 → 32-byte digest → count leading zero bits**
   More zeros = better proof. Format: `username/nonce`.

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
  -V, --verify PROOF       Verify a proof and exit
```

## Verifying a proof

```bash
./output/neurallenge --verify "brandonros/GAehbrFwu9v3s5u2b4n0L8C8m7E2g7B3x5C4D6F7H8J9K0M1N2P3Q4R5S6T7U8V9W0=="
```

Output shows the digest and leading zero bits.

Nonces are base64-encoded (86 characters for 64 bytes).
