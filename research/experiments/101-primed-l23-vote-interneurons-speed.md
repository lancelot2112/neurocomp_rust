# 101 · Primed L2/3, a burst vote, interneurons, and vectorized counting

**Question.** [100](100-primed-layer5.md) left four leads:
- give L2/3 itself, the column's predictor, the primed two-compartment cells;
- let a bursting column vote with the burst's own strength, since the mix ignored the better
  column;
- replace the context threshold's gain rule with inhibitory interneurons;
- make the primed layers faster.

## Code
- **Primed L2/3** (`L23=primed`, [`examples/episodic.rs`](../../examples/episodic.rs)): a
  `PrimedLayer5` pool of 16,384 cells as the column's L2/3. Whenever a cell fires, spike or
  burst, its output is the column's prediction. The old kernels speak only when no cell
  fires, and still serve the circuits that read the column's expectation directly (role
  cells, saccades).
- **Burst vote** (`BURST_VOTE=1`): when the column bursts, its vote in the thalamic mix
  weighs −log2(1 − strength), the strength being the burst's priming, instead of the
  word-keyed reliability learned for the old column.
- **Interneurons** (`INTERNEURONS=1`, `PrimedLayer5::set_interneurons`):
  - *SST (Martinotti)* cells inhibit the tufts. Their activity follows the area's bursting,
    and each cell's own SST input grows with its own recent bursting;
  - *VIP* cells inhibit the SST cells. The matrix drives them with an error signal: they
    rise when the layer's prediction failed and decay when it was confirmed;
  - the tuft threshold is 0.3 + 0.6 × SST × (1 − VIP), plus a quarter of the cell's own
    burstiness, at most 0.95;
  - *PV* cells are the winner-take-all among the cells that fired, as before.
- **Speed:**
  - no per-synapse search: the index points straight at each synapse;
  - dense per-step counts, one evaluation per step shared by prediction and learning;
  - dead synapses kept at zero strength;
  - **vectorized counting** (`L5_VEC=1`): each row bit holds one bitset over cells per
    compartment (connected synapses only), and the active bits' bitsets are added into
    bit-sliced counters with AND/XOR over whole words, 64 cells per operation;
  - a test (`vectorized_counting_matches_event_driven`) checks that it gives identical
    predictions to the event-driven counting over 3,000 steps of learning.

## Results
`LEARNING=three SLOW_P=0.25 SLEEP_P=1 REPLAY_PREDICT=1 L23=primed`, five seeds,
held-out:

| Entry | Reference ([94](94-sleep-gated-consolidation.md)) | Primed L2/3 | + burst vote | + interneurons | + both |
|---|---|---|---|---|---|
| family consolidated | 47.7 | 44.8 | 34.0 | 44.8 | 34.0 |
| hippocampus teaches cortex | 51.5 | 45.2 | 37.0 | 45.2 | 36.2 |
| index hippocampus | 34.1 | **41.8** | 32.8 | **41.8** | 35.1 |
| engram store | 45.4 | 47.1 | 49.4 | 46.8 | 49.4 |
| semantic store | 38.6 | **47.4** | 34.8 | **47.4** | 34.8 |
| mean change | | **+1.8** | −5.9 | +1.7 | −5.6 |

On seed 0:
- the primed cells make 99.5% of the column's predictions (the old kernels about 300–600 of
  about 105,000 steps);
- about 11,600 cells are committed, and about 100–170 freed;
- the column alone is right at 40–65% of test answers.

**Timing** (hippocampus teaches cortex, seed 0, primed L2/3, alone on the machine):

| | Word loop | L2/3 prediction phase | Learning and after | Held-out |
|---|---|---|---|---|
| no primed layer (same entry and settings) | 159 s | 34% | 60% | 58.2 |
| primed, event-driven | 271 s | 57% | 38% | 54.4 |
| primed, vectorized | 242 s | 46% | 49% | 54.4 |
| primed, vectorized, native instructions | 220 s | 47% | 49% | 54.4 |
| no primed layer, native instructions | 150 s | 36% | 59% | 58.2 |

*Correction:* the first version of this page compared with 21 s, a different entry run with
default settings. On the same entry and settings the primed layer costs about 1.5× (+70 s
with native instructions), not 11×.

## Findings
1. **Primed L2/3 gains on average (+1.8).** It is the second learned replacement with a
   positive mean, after the AND layer 5 of [99](99-two-compartment-layer5.md) (+2.8). Two
   entries gain strongly (semantic store +8.8, index hippocampus +7.7). The two family entries
   still lose (−2.9 and −6.3).
2. **The burst vote hurts (−5.9).** Giving the bursting column its own evidence lets it
   outvote the other sources. Its bursts are common (about 60% of steps) and strong, but not
   reliable enough at the answers to overrule memory and the cerebellum. The word-keyed
   weights were protecting the answers.
3. **The interneurons change almost nothing in L2/3** (+1.7 vs +1.8, most figures identical).
   In L2/3 spikes predict as well as bursts, so the context threshold only orders bursts
   before spikes and decides commitment. Their place is layer 5, where only bursts count.
4. **Vectorized counting is exact but only 11% faster.** Counting is no longer the main
   cost. Per touched cell the layer still counts its connected synapses one by one, and
   learning re-syncs the bitsets.
5. **The build was not using hardware popcount.** Without a target CPU, Rust compiles
   `count_ones` to a software sequence for baseline x86-64. Building for the machine's own
   instructions (`.cargo/config.toml`: `target-cpu=native`, with POPCNT and AVX-512's vector
   popcount) gives identical results, 6% faster for the whole network and 9% for the primed
   layer.

Not a default yet.

## Next
- **Speed:** keep each cell's connected-synapse counts up to date as synapses change, rather
  than recounting them for every touched cell; sync only the synapses that crossed the
  threshold; run several seeds at once.
- **The answer path:** a burst vote that is learned, not taken at face value. The column's
  burst strength enters the mix as context for its reliability, replacing word pairs as the
  key.
- **Interneurons in layer 5**, where bursts alone speak (`L5=primed INTERNEURONS=1`).
- The family entries lose with every primed variant; find out why (which sources answer
  there, and what the column says).
