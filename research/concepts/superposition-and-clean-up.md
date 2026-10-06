# Superposition and clean-up: the math of a Hebbian population

The hippocampal pathways, the `Associate` leaf ([52](../experiments/52-associate-and-hippocampus-genome.md))
and the predictive kernels all store many patterns in one set of weights and get one of
them back. This page writes out what that is, why it fails, and the cheapest operations
that fix it. Measurements: `cargo run --release --example superposition`.

## Store: a histogram of co-occurrences
Each event e pairs an input pattern x_e (bits) with a target pattern y_e (cells), written
with strength a_e. A pathway's weights are the sum of outer products:

  W_ij = Σ_e a_e · x_e,i · y_e,j

W_ij counts how often input bit i and cell j were active together, weighted by write
strength. So yes: it is a histogram, one row per input bit. Here the counters are 7-bit
and halve every `half_life` writes (decay).

## Recall: a superposition
A cue q (the active input bits) drives every cell j by the rows of its active bits:

  h_j = Σ_{i∈q} W_ij = Σ_e a_e · |q ∩ x_e| · y_e,j

That is a **superposition of every stored target**, each weighted by how much its input
overlaps the cue. Split it into the event the cue came from (e*) and the rest:

  h = a·|q ∩ x_e\*| · y_e\*  (signal)  +  Σ_{e≠e\*} a_e |q ∩ x_e| y_e  (crosstalk)

## Read-out: clean-up by threshold
The population keeps the k most driven cells (or every cell within a fraction of the
best). This nonlinearity is what turns the superposition back into one pattern. It works
when the signal cells out-vote every crosstalk cell.
- **Settling** (CA3's recurrent pathway) repeats it: activity ← top-k(W_rec·activity +
  h). Each pass sharpens the pattern toward a stored one, which is an attractor.
- **The quantum-measurement analogy is fair, up to a point.** A superposition of many
  stored states is collapsed by a nonlinear read-out to one of them, and the "interference"
  is the crosstalk term. There is no unitary evolution or true interference, though: with
  counts every crosstalk term is positive, and nothing cancels.

## Why it fails: crosstalk that does not cancel
For random sparse codes:
- the cue has s bits of N;
- a target has k cells of M;
- E events are stored;
- cell j has been a target u_j ≈ E·k/M times.

Then, per cell:
- **Signal:** a·s (every cue bit votes for the right cells).
- **Crosstalk mean:** μ_j ≈ a · s · (s/N) · u_j. It grows linearly with E.
- **Crosstalk spread:** roughly √μ_j.

Top-k ignores a constant added to every cell, so a uniform μ is harmless. Two things are
not harmless:
1. **Cells used more than others** collect more crosstalk (μ_j ∝ u_j) and win by bulk.
2. **Inputs shared by many events.** In language, "is", "a" and "the" occur in most
   events. All 16 bits of a common word were written together, so their crosstalk is
   *coherent*: it adds up across the word's bits instead of averaging out. Its spread
   grows with (bits per word) × √(events that used it). This is the attractor of common
   events that experiments [45](../experiments/45-superposed-thought-and-ca3.md) and
   [48](../experiments/48-hippocampus-on-its-own.md) fell into.

## The minimal fixes
Two integer operations address these two failures. Each costs about one multiply per
active row or cell.

1. **Divide an input's vote by its use (1/n).** Row i, written by n_i events, has its
   drive multiplied by 1/n_i (in `Q16`). A common input's vote is spread over every event
   that used it, so its coherent noise shrinks to about one event's worth. This is inverse
   document frequency, and in the brain presynaptic scaling or heterosynaptic depression.
   - The hippocampus used ⌊log2 n⌋ right shifts on each counter. That rounds single
     writes to zero once n is large, and recall then collapses on random codes.
     Multiplying the row's sum by 1/n avoids the rounding.
2. **Subtract the expected count (centering).** From each cell's drive, subtract what
   it would get if the stored events were unrelated to the cue:
   (Σ_{i∈q} weighted n_i) · u_j / E × the mean write amount.
   - What remains is crosstalk with mean zero, whose size grows like √E rather than E,
     and much-used cells no longer win by bulk.
   - This is the covariance rule of Hopfield networks (weights from deviations from mean
     activity), done at read time from counts the population already keeps.

Together they turn the co-occurrence histogram into "how much more often than chance"
input i and cell j occur together: count_ij compared with n_i·u_j/E. That is the quantity
behind pointwise mutual information (as in word embeddings) and behind homeostatic
synaptic scaling.

### Measured (`examples/superposition.rs`; 2,048 inputs, 2,048 cells, k = 32; right = ≥ 80% of the target)

| codes | events | counts | shift-scaled | 1/n-scaled | centered | shift + centered | **1/n + centered** |
|---|---|---|---|---|---|---|---|
| random | 2,000 | 100% | 90% | 100% | 100% | 100% | **100%** |
| random | 4,000 | 73% | 0% | 77% | 100% | 0% | **100%** |
| language-like | 250 | 16% | 99% | 99% | 22% | 98% | **99%** |
| language-like | 1,000 | 0% | 81% | 86% | 0% | 84% | **89%** |
| language-like | 2,000 | 2% | 33% | 43% | 0% | 29% | **56%** |
| language-like | 4,000 | 2% | 12% | 13% | 0% | 0% | **14%** |

"Language-like" events are two common words, drawn from a skewed list of 50, plus one rare
word from 5,000.
- **Centering** is what random codes need (it fixes the cell-use bias).
- **1/n** is what language-like codes need (it fixes the coherent noise of common
  words).
- **Both together** is best, or tied best, in every row.
- **At 4,000 language-like events** many rare words occur in more than one event, so
  part of the task is ambiguous. Common rows also saturate the 7-bit counters.

## Phases and complex numbers
**Phase codes give zero-mean crosstalk for free.** In holographic reduced representations
(Plate 1995) and their Fourier form (FHRR), each item is a vector of unit complex numbers
(phases):
- binding is phase addition;
- superposition is the complex sum;
- clean-up is the nearest stored item.

Unrelated items have random phases, so their crosstalk terms point in random directions
and cancel on average: mean zero, size √E. Centering gets the same property for counts.

**The binary system already uses two kinds of phase:**
- **XOR binding** is the two-phase case (0 or π; Kanerva's binary spatter codes).
- **Cyclic rotation**, the slot binding in this codebase, is binding by a discrete phase
  shift of the whole vector.

**The integer version of "dropping into phase space"** is a phase code over integers
mod K (as in sparse block codes and modular composite representations; Snaider & Franklin
2014; Frady, Kleyko & Sommer 2021):
- each active cell carries a phase in 0..K;
- binding adds phases mod K;
- superposition is a histogram of (cell, phase) votes;
- clean-up takes the winning phase per cell.

It stays integer-only. Votes with a different phase do not add to the winning one, which
is the cancellation. This is not built. It would be a variant of `Associate` whose
counters are indexed by phase.

**Where that leaves the choice:**
- **Centering plus 1/n** is the minimal step, and it is measured here to recover most of
  the benefit for counts.
- **Phase codes** are the step after that. They are worth it if binding many roles in
  one pattern (several slots superposed) becomes the limit.

## References
- Willshaw, Buneman & Longuet-Higgins 1969 (binary associative memory); Hopfield 1982
  and the covariance rule (Sejnowski 1977); Tsodyks & Feigel'man 1988 (sparse codes).
- Plate 1995 (holographic reduced representations); Kanerva 2009 (hyperdimensional
  computing).
- Snaider & Franklin 2014 (modular composite representations); Frady, Kleyko & Sommer
  2021 (variable binding for sparse distributed representations).
- Turrigiano 2008 (homeostatic synaptic scaling); Church & Hanks 1990 (pointwise mutual
  information).
