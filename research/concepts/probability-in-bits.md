# Probability in bits: what is bitwise now, and how to keep it that way

The core of this codebase is bit math: sparse codes, AND / OR / AND-NOT, popcount and
thresholds. Recent experiments added real-valued state alongside it. This page lists
where each kind lives and how probabilities and weights can be held in bits instead.

## Inventory

| Mechanism | Representation now | Bitwise? |
|---|---|---|
| Word codes, sentence bags, episodes | sparse bit vectors, OR | yes |
| Overlap, recall, kernel matching | AND + popcount ≥ threshold | yes |
| Novelty / readout (`novel`, cue removal) | AND-NOT with a mask | yes |
| Dentate gyrus | popcount drive (integers) + k-winner-take-all | yes (integer) |
| Kernel reliability | `hits`, `misses` (u32 counters) → (h+1)/(h+m+2) as f32 | counts are; ratio is float |
| Habituation statistics | per-bit episode counts (u32) | counts |
| Route scores, `RouteGate` precision | counts → float ratio | counts are |
| CA1 comparator ([14](../experiments/14-ca1-comparator.md)) | overlap/popcount (a ratio of popcounts) × kernel reliability | ratio of counts |
| CA3 weights ([12](../experiments/12-dentate-gyrus-ca3.md#in-bits)) | ~~f32 per synapse~~ → 7-plane bit-sliced counter rows, lazy plane-shift decay | **yes** (integer) |
| Basal-ganglia "go" weights ([15](../experiments/15-basal-ganglia-selector.md)) | ~~f32 per bit~~ → 4-plane bit-sliced counters, bit-mask eligibility, stochastic ±1 steps | **yes** (RPE still a float ratio) |

Most "probabilities" here are already **ratios of counts**, which compare without
floats by cross-multiplying (p ≥ ½ ⇔ 2·hits ≥ hits + misses). The genuine drift is in the
CA3 weights and the basal-ganglia weights; both have now been converted (below).

## Ways to hold probability in bits
1. **Counts and ratios of popcounts.** A probability is hits / trials, and the share of a
   prediction a word gets is |code ∧ prediction| / |prediction|. Keep integers, compare
   by cross-multiplication. This is what kernels already do.
2. **Bit-sliced (vertical) counters.** Store k bit vectors ("planes"), plane i holding
   bit i of every position's counter. Incrementing every bit under a mask is a ripple
   of half adders: `carry = mask; for i: (plane_i, carry) = (plane_i ^ carry, plane_i & carry)`.
   That is k word-wide XOR/AND passes for all positions at once. **Halving every counter
   is dropping the lowest plane**, so exponential decay (a palimpsest) is a shift.
   A weighted sum over a candidate is Σ 2^i · popcount(candidate ∧ plane_i). This is the
   bundling/majority "count" of hyperdimensional computing
   ([Kanerva 2009](../related-work.md#sparse-distributed-representations)).
3. **Rate / population codes (stochastic computing).** A value p is a bit vector with
   about p·n ones. Multiplying independent values is AND, a weighted mix is a
   multiplexer, reading out is popcount ([Gaines 1969; Alaghi & Hayes 2013](../related-work.md#learning-rules)).
   A superposed prediction (the OR or count of several candidate codes) *is* a
   distribution: each candidate's probability is how much of it is present.
4. **Stochastic binary synapses.** Each synapse is one bit; learning sets or clears it
   with a small probability. Memories fade as later learning overwrites them, giving a
   palimpsest with no stored decay constant ([Amit & Fusi 1994; Fusi, Drew & Abbott 2005](../related-work.md#learning-rules)).
   The learning rate becomes a flip probability, and the expected weight is a
   probability held across many bits.

## Done: `SlicedCounter`
[`src/bitvec/bitcounter.rs`](../../src/bitvec/bitcounter.rs) holds one saturating counter
per bit position as `planes` bit vectors. Operations, all word-wide:
- `increment` / `decrement` under a mask: ripple of XOR/AND (half adders / subtractors),
  then saturate (OR / AND-NOT with the final carry or borrow).
- `add_power(mask, p)`: add 2^p (start the ripple at plane p); any amount is a few of these.
- `halve` / `shift_down(s)`: drop the lowest plane(s), i.e. floor(v / 2^s): decay.
- `sum(pattern)`: Σ_p 2^p · popcount(pattern ∧ plane_p).

**Basal ganglia** ([15](../experiments/15-basal-ganglia-selector.md)): go values in 4 planes;
candidates compared by cross-multiplying integer sums; the eligibility trace is the
chosen item's bit mask; each eligible counter steps ±1 with probability
1.5 × |reward − value|. Result: 89.1% vs 84.5% for floats (5 seeds), and steadier.

**CA3** ([12](../experiments/12-dentate-gyrus-ca3.md#in-bits)): see that page. Two details
that generalize:
- **Lazy decay.** Each row stores the epoch it was last normalized; a read skips the
  planes that would have been shifted out (floor(v/2^s) = Σ_{p≥s} bit_p · 2^(p−s)), so
  decay costs nothing until a row is touched.
- **Recency within a halving period.** Halving only every few stores makes episodes
  stored in the same period equal, which broke "most recent wins" (a unit test caught
  it). Fix: the n-th store in a period adds 16 · 2^(n/half_life), i.e. growth instead
  of decay between halvings, which is the same ordering as continuous exponential decay.

## Mapping to the drifted parts (original plan)
- **Basal ganglia:** go weights → a 4-plane bit-sliced counter per input bit (16 levels);
  the eligibility trace → a bit mask (chosen item's bits) or a short shift register of
  masks; dopamine → increment or decrement the counters under the mask, applied with
  probability ∝ |RPE| (the learning rate as a flip probability); value → Σ 2^i ·
  popcount(item ∧ plane_i).
- **CA3:** f32 weights → stochastic binary synapses or a small bit-sliced counter per
  synapse, halved by plane shift every N stores instead of `decay^age`.
- **Comparator:** already a ratio of popcounts × a ratio of counts; compare
  overlap·hits·2 against |prediction|·(hits + misses + 2) in integers.
