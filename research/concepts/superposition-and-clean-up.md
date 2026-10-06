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
active row or cell, and neither needs a division per step
([53](../experiments/53-phase-codes-and-centering.md)).

1. **Divide an input's vote by its use (1/n).** Row i, written by n_i events, has its
   drive multiplied by 1/n_i. A common input's vote is spread over every event that used
   it, so its coherent noise shrinks to about one event's worth. This is inverse
   document frequency, and in the brain presynaptic scaling or heterosynaptic depression.
   - **Without dividing:** 1/n comes from a reciprocal table (`fixed::recip32`). Below
     4,096 it is a table entry; above that, the entry for n's top 12 bits, shifted.
     The relative error is under 1/2,048.
   - The hippocampus used ⌊log2 n⌋ right shifts on each counter. That rounds single
     writes to zero once n is large, and recall then collapses on random codes.
     Multiplying the row's sum by 1/n avoids the rounding.
2. **Subtract the expected count (centering).** From each cell's drive, subtract what
   it would get if the stored events were unrelated to the cue:
   (Σ_{i∈q} weighted n_i) × r_j × the write amount, where r_j is cell j's write rate.
   - What remains is crosstalk with mean zero, whose size grows like √E rather than E,
     and much-used cells no longer win by bulk.
   - This is the covariance rule of Hopfield networks (weights from deviations from mean
     activity), done at read time from counts the population already keeps.
   - **Without dividing (`Center::Homeostatic`):** each cell keeps its write rate as a
     running mean, r_j += (written − r_j) × 1/min(writes, 2^s), with 1/n from the
     table. That is a threshold that rises with the cell's own use: a homeostatic,
     BCM-like sliding threshold. It needs 32 fractional bits, because a cell's rate is
     about 1/64 and each step's change is far below one `Q16` unit (with 16 bits,
     flooring biased it low and lost recall).
   - Measured, it recalls exactly as well as exact division at every load.

Together they turn the co-occurrence histogram into "how much more often than chance"
input i and cell j occur together: count_ij compared with n_i·r_j. That is the quantity
behind pointwise mutual information (as in word embeddings) and behind homeostatic
synaptic scaling.

## Phases: integer complex weights (`PhaseAssociate`)
**Phase codes give zero-mean crosstalk with nothing subtracted.** Each active cell
carries a phase p in 0..K. In bits that is a block code: cell j at phase p sets bit
j·K + p.
- **Weights** are integer complex numbers: (cos, sin) from a table scaled by 2^14.
- **Write:** w_ij += e^{i(θ_j − φ_i)}, for input i at phase φ_i and target cell j at
  phase θ_j.
- **Read:** h_j = Σ_{i∈q} w_ij · e^{iφ_i}. The k cells with the largest |h_j|² fire, each
  at the table phase nearest to arg h_j.

**Why it cancels:**
- **The stored event:** its terms all arrive at phase θ_j and add up, so |signal| = the
  cue size.
- **Other events:** their terms arrive at unrelated phases and largely cancel. Their sum
  grows like √n, not n.
- **The link to centering:** the complex sum is each cell's phase histogram projected
  onto its first harmonic, and that projection has no constant term. So the phase code
  does the centering itself.

**The condition, measured in [53](../experiments/53-phase-codes-and-centering.md): the
input must carry phase too.**
- **Phases only on the output:** the cue's overlap with another stored input is still a
  positive count, and that overlap is the coherent part of the crosstalk. Recall is then
  no better than plain counts.
- **Phases on the input as well:** the overlap itself becomes a sum of random phases and
  cancels.

### Measured (`examples/superposition.rs`; 2,048 inputs, 2,048 cells, k = 32, 16 phases; right = ≥ 80% of the target cells)

| codes | events | counts | 1/n | centered | **1/n + centered** (homeostatic) | phase in (random) + out | phase in (by slot) + out + 1/n |
|---|---|---|---|---|---|---|---|
| random | 4,000 | 83% | 86% | 100% | **100%** | 100% | 100% |
| random | 8,000 | 0% | 0% | 93% | 94% | **100%** | 33% |
| language-like | 1,000 | 0% | 83% | 0% | 86% | **98%** | 93% |
| language-like | 2,000 | 1% | 42% | 0% | 57% | 79% | **82%** |
| language-like | 4,000 | 1% | 12% | 0% | 13% | 43% | **50%** |

"Language-like" events are two common words, drawn from a skewed list of 50, plus one rare
word from 5,000. The phase columns score cells; scored on cells *and* phases they are
lower (random input phases: 93% at 4,000 random events, 33% at 2,000 language-like ones).

- **Counts:**
  - **Centering** is what random codes need (it fixes the cell-use bias).
  - **1/n** is what language-like codes need (it fixes the coherent noise of common
    words).
  - **Both together**, done without division, is the best count-based variant.
- **Phases:**
  - **Random input phases** (each event's input bits a fresh random phase) are an upper
    bound: they need the cue to carry the stored phases. With them, recall holds 100% at
    8,000 random events where centered counts hold 93%, and 43% vs 13% at 4,000
    language-like events.
  - **Phases from the slot** are the realistic case, and the brain's analogue would be
    a word's phase set by its role: the same word in the same slot gets the same phase,
    so shared words still overlap coherently, but less often.
    - With 1/n, this beats every count variant on language-like codes: 82% vs 57% at
      2,000 events, 50% vs 13% at 4,000.
    - It does nothing for random codes (33% at 8,000), which have no slots to separate.
- **The cost:** recovering the exact phase is harder than recovering the cell. Sixteen
  phases are 22.5° bins, and noise pushes many cells into a neighbour.
- **At 4,000+ language-like events** many rare words occur in more than one event, so
  part of the task is ambiguous.

**In the full system it did not hold up**
([54](../experiments/54-phase-hippocampus-backed-out.md)). A hippocampus built from phasor
populations, with words at slot phases, did no better than the counts circuit on the
experiment 49 task (67 / 42% vs 64 / 59% held out) and ran 3.5× slower, so it was backed
out. Exact 1/n on the circuit's perforant path also failed there; centering was neutral.
The isolated gains came from full-cue recall of stored targets, and the circuit's partial
cues, novelty, tags and replay did not share them.

**Binding by phase.** The binary system already uses two kinds of phase:
- **XOR binding** is the two-phase case (0 or π; Kanerva's binary spatter codes).
- **Cyclic rotation**, the slot binding in this codebase, is binding by a discrete phase
  shift of the whole vector.

With phase codes, binding is adding a role's phase to every active cell, mod K. That is
what the slot phases above do, and it is the first use where phase pays.

## References
- Willshaw, Buneman & Longuet-Higgins 1969 (binary associative memory); Hopfield 1982
  and the covariance rule (Sejnowski 1977); Tsodyks & Feigel'man 1988 (sparse codes).
- Plate 1995 (holographic reduced representations); Kanerva 2009 (hyperdimensional
  computing).
- Snaider & Franklin 2014 (modular composite representations); Frady, Kleyko & Sommer
  2021 (variable binding for sparse distributed representations).
- Turrigiano 2008 (homeostatic synaptic scaling); Church & Hanks 1990 (pointwise mutual
  information).
- Noest 1988 (phasor neural networks); Frady & Sommer 2019 (threshold phasor
  associative memory, spikes at phases of a rhythm); Bienenstock, Cooper & Munro 1982
  (the sliding threshold).
