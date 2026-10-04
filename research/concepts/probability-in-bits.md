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
| CA3 weights ([12](../experiments/12-dentate-gyrus-ca3.md)) | f32 per synapse, exponential decay | **no** |
| Basal-ganglia "go" weights ([15](../experiments/15-basal-ganglia-selector.md)) | f32 per bit, eligibility trace f32 | **no** |

Most "probabilities" here are already **ratios of counts**, which compare without
floats by cross-multiplying (p ≥ ½ ⇔ 2·hits ≥ hits + misses). The genuine drift is in the
CA3 weights and the basal-ganglia weights.

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

## Mapping to the drifted parts
- **Basal ganglia:** go weights → a 4-plane bit-sliced counter per input bit (16 levels);
  the eligibility trace → a bit mask (chosen item's bits) or a short shift register of
  masks; dopamine → increment or decrement the counters under the mask, applied with
  probability ∝ |RPE| (the learning rate as a flip probability); value → Σ 2^i ·
  popcount(item ∧ plane_i).
- **CA3:** f32 weights → stochastic binary synapses or a small bit-sliced counter per
  synapse, halved by plane shift every N stores instead of `decay^age`.
- **Comparator:** already a ratio of popcounts × a ratio of counts; compare
  overlap·hits·2 against |prediction|·(hits + misses + 2) in integers.
