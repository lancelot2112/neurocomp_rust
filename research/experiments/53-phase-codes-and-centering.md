# 53 · Integer phase codes, and centering without division

**Question.** [52](52-associate-and-hippocampus-genome.md) found two cheap fixes for a
Hebbian population's crosstalk: 1/n weighting of much-used inputs, and centering
(subtracting each cell's chance drive). Both were done with divisions. Can they be done
without division? And does a phase code, integer complex weights whose crosstalk cancels
on its own, do better still?

**Code.**
- [`src/fixed.rs`](../../src/fixed.rs): `recip32` and `recip`, 1/n from a table with no
  division per call; `phasors`, the integer (cos, sin) table.
- [`src/program/modules.rs`](../../src/program/modules.rs):
  - `Center` (`Off`, `Exact`, `Homeostatic`) and `Scale::Inverse` without division, on
    `Associate`;
  - the leaves `PhaseAssociate`, `Phase` (a plain pattern given event-specific phases by
    a hash) and `Cells` (drop the phases);
  - `phase_code`, `phase_cells`.
- [`examples/superposition.rs`](../../examples/superposition.rs): the measurements.
- The math is in [superposition and clean-up](../concepts/superposition-and-clean-up.md).
- Unit tests: `homeostatic_centering_matches_exact`,
  `phase_population_recalls_cells_and_phases`, `reciprocals_need_no_division`. The
  genome parity tests of 51 and 52 still pass.

## What was built
1. **1/n without division.** A table of 2^32/n for n < 4,096. For larger n, the table
   entry of n's top 12 bits, shifted; the relative error is under 1/2,048. The table is
   built once; no step divides.
2. **Homeostatic centering.** Each cell keeps its write rate r_j as a running mean,
   r_j += (written − r_j) × 1/min(writes, 2^s), with 1/n from the table. Before the
   readout, each cell's drive has (cue rows' weighted writes) × r_j × the write amount
   subtracted. This is a threshold that rises with the cell's own use, the sliding
   threshold of BCM theory.
   - The first version kept r_j in `Q16` and used a shift for the step (a moving
     average over 1,024 writes). Both lost recall: a cell's rate is about 1/64, each
     step's change is far below one `Q16` unit, and flooring biased the rate low.
   - With 32 fractional bits and the 1/n step it matches exact division.
3. **`PhaseAssociate`, a phasor memory in integers.**
   - **Codes:** active cells carry a phase in 0..K, held as a block code (bit j·K + p).
   - **Weights:** integer complex numbers, w_ij += e^{i(θ_j − φ_i)}.
   - **Read:** h_j = Σ w_ij · e^{iφ_i}. The k largest |h_j|² win, each at the phase
     nearest to arg h_j.
   - Everything is table lookups, integer multiply-adds and shifts.

## Results (`cargo run --release --example superposition`; 2,048 inputs, 2,048 cells, k = 32, 16 phases)
Right = at least 80% of the event's target cells recovered. The phase populations are
scored on cells, except where marked "+ phase".

| codes | events | counts | 1/n | centered, exact | centered, homeostatic | 1/n + centered, homeostatic | phase out only | phase in (random) + out | same, + phase | phase in (by slot) + out + 1/n |
|---|---|---|---|---|---|---|---|---|---|---|
| random | 2,000 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| random | 4,000 | 83% | 86% | 100% | **100%** | 100% | 100% | 100% | 93% | 100% |
| random | 8,000 | 0% | 0% | 93% | **93%** | 94% | 31% | **100%** | 15% | 33% |
| language-like | 500 | 3% | 95% | 6% | 6% | 96% | 18% | 100% | 96% | **99%** |
| language-like | 1,000 | 0% | 83% | 0% | 0% | 86% | 5% | 98% | 68% | **93%** |
| language-like | 2,000 | 1% | 42% | 0% | 0% | 57% | 0% | 79% | 33% | **82%** |
| language-like | 4,000 | 1% | 12% | 0% | 0% | 13% | 0% | 43% | 8% | **50%** |
| language-like | 8,000 | 0% | 0% | 0% | 0% | 0% | 0% | 13% | 0% | 2% |

What the columns mean:
- **Language-like:** two common words (a skewed list of 50) and one rare word (of 5,000).
- **Phase in (random):** each event's input bits get a fresh random phase, which the cue
  must carry at recall. This is an upper bound.
- **Phase in (by slot):** a bit's phase is set by its word's slot, so the same word in the
  same slot always has the same phases. This is phase binding of word to role.

## Findings
1. **Centering needs no division.** A per-cell running rate, updated with a table
   reciprocal, recalls exactly as well as exact division at every load. Combined with
   table-based 1/n, it is the best count-based variant.
   - **The biology is a homeostatic threshold:** cells that are written often need more
     drive to fire.
   - **It needs precision:** 32 fractional bits, because the rates are small.
2. **Phases cancel crosstalk only when the input carries phase.**
   - **Phases on the target alone** ("phase out only") are worse than centered counts:
     the cue's overlap with another event's input is still a positive count, and that
     overlap is the coherent part of the crosstalk.
   - **With phase-coded inputs** the overlap itself becomes a sum of random phases. Cell
     recall then beats every count variant: 100% at 8,000 random events vs 93%, and 43%
     vs 13% at 4,000 language-like events. Nothing is subtracted; the code centers
     itself.
3. **Role phases are the realistic route, and they pay on language-like codes.** Giving
   each word the phases of its slot, plus 1/n, recalls the most of any variant on
   language-like events: 82% vs 57% at 2,000 events, 50% vs 13% at 4,000.
   - Common words occupy different slots across events, so their rows are no longer
     coherent.
   - Random codes have no slots, and gain nothing.
4. **The phase itself is harder to recover than the cell.** Scored on cells and phases,
   the random-input-phase memory falls to 93% at 4,000 random events and 33% at 2,000
   language-like ones. Sixteen phases are 22.5° bins, and noise pushes many cells into a
   neighbour. When the phase is only a means of cancellation, score the cells. When it
   carries information (a role), fewer phases or a clean-up stage would be needed.

## What this means for the architecture
- **Centering is cheap and safe:** `Center::Homeostatic` with `Scale::Inverse`. It is the
  candidate for the hippocampus genome's pathways, replacing the shift scaling that
  collapses at high load. The episodic runs still use the shift for parity.
- **Phase codes are worth it where the input can carry role phases:** binding a word to
  its slot by phase instead of by rotation. That is the experiment 49 binding space, with
  phases standing in for the slot fields. It needs the cortex's output to carry those
  phases.
- **The leaves exist** (`PhaseAssociate`, `Phase`, `Cells`), so either can be tried in a
  genome without new machinery.

## Next
Both follow-ups were tried in the hippocampus on the experiment 49 task and did not pay:
the phase-bound circuit was worse and 3.5× slower, and 1/n broke a seed. See
[54](54-phase-hippocampus-backed-out.md).

## Biology
- **Phase coding:** hippocampal place cells fire at systematically shifting phases of
  theta (phase precession; O'Keefe & Recce 1993), and items held in working memory are
  proposed to occupy different gamma phases of a theta cycle (Lisman & Jensen 2013).
  Phasor associative memories are a model of computing with spike timing (Frady & Sommer
  2019).
- **Homeostatic thresholds:** the BCM sliding threshold (Bienenstock, Cooper & Munro
  1982); intrinsic plasticity and synaptic scaling (Turrigiano 2008).
