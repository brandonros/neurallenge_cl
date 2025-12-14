# neurallenge

GPU proof-of-work using FP32 neural network inference.

## How it works

Like Shallenge, but with a neural network in the middle:

1. **Epoch string → SHA-256 → ~148KB of neural network weights**
   Everyone mines the same network for a fair leaderboard.

2. **Nonce → SHA-256 expand to 64 bytes → neural network → 32 output values**
   Any string works as a nonce! It gets SHA-256 expanded to 64 bytes before the network.
   The network has 4 layers (32→256→256→256→32) of FP32 multiply-adds.

3. **Output → SipHash-2-4 → 32-byte digest → count leading zero bits**
   More zeros = better proof. Format: `username/nonce` where nonce can be any string.

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
# Mined proofs use base64-encoded random bytes
./output/neurallenge --verify "brandonros/ABC123xyz789+/=="

# But ANY string works as a nonce - be creative!
./output/neurallenge --verify "brandonros/hello world"
./output/neurallenge --verify "alice/my clever message here"
```

Output shows the digest and leading zero bits.

Nonces can be any string - short, long, whatever you want. The miner outputs base64-encoded random bytes, but you can submit any string for verification.
