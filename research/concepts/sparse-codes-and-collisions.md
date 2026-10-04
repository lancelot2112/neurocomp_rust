# Sparse codes and collisions

Every symbol (letter, word, boundary marker) is a fixed random sparse bit pattern:
*k* active bits out of *n* ([`examples/common/mod.rs`](../../examples/common/mod.rs), `Encoder`).

## Why it matters
Two random codes share on average `k²/n` bits. A single bit is shared by about `V·k/n`
of the *V* symbols. For **exact matching** (predictive kernels), small overlaps are
harmless: 32-of-512 letter codes overlap about 2 bits, far below a 16-bit sample's
threshold. For **Hebbian statistics** over context codes, each connected bit stands
for every symbol that uses it, so masks blur together.

Measured on part-of-speech induction ([05](../experiments/05-syntax.md)), 5,001 word codes:

| n bits / k active | words per bit (≈ V·k/n) | NN tag agreement |
|---|---|---|
| 1024 / 32 | 156 | 35.7% |
| 4096 / 16 | 20 | 51.6% |
| 8192 / 8 | 4.9 | 65.0% |
| 16384 / 4 | 1.2 | 72.9% |
| 32768 / 4 | 0.6 | 74.9% |

The curve flattens once a bit names about one word. Past that point, the mask's
small sample size is the limit, not collisions.

## Rule of thumb
- Codes that kernels **match against**: moderate sparsity is fine (512/32 for letters).
- Codes that are **accumulated as statistics**: aim for `V·k/n ≲ 1`.

## Related
Ahmad & Hawkins 2016 give the matching-error math for sparse distributed
representations. Kanerva's sparse distributed memory and hyperdimensional computing,
and Random Indexing, cover the accumulation case
([related work](../related-work.md#sparse-distributed-representations)).
