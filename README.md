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
./output/neurallenge --verify "brandonros/1807944a1b9bb9d33b349f179a776784feae5a1ee7ae7fddbc0832ff733c53a0b7b3271d8ad34d63d241308cd22b9ace361f5245f88ccf76262cee7745924416"
```

Output shows the digest and leading zero bits.
