# 104 · Fully bitwise cells, an id-indexed sweep, and self-calibration

**Question.** The primed cells of [100](100-primed-layer5.md)–[102](102-burst-key-l5-interneurons-bit-synapses.md)
kept an 8-bit strength per synapse, or flags in a per-synapse loop. The aim from the start was
a cell that is only masks: popcount to decide, learning by adding, removing and moving bits.
Three questions:
1. Does a fully bitwise cell learn as well, and run faster?
2. Can the plasticity index come from the cell's id instead of a random draw?
3. Can the learning rate and threshold calibrate themselves from data instead of being set
   by hand?

## Code
`BitCells` ([`src/program/bitcells.rs`](../../src/program/bitcells.rs), four tests), option
`L5_SYN=bitwise`:
- **A cell is masks.** For each input word it touches, three 64-bit masks: *active* (AMPA),
  *silent* (NMDA only: counts only when the cell is primed), *sticky* (consolidated). A new
  cell's input synapses are silent.
- **Firing:** popcount(input & active), plus popcount(input & silent) when the context primes
  the cell, against a threshold. The tuft's popcount share is the priming. Bursts beat
  spikes. The SST/VIP interneurons or the gain rule set the context threshold.
- **One plasticity event, one random word.** Its bytes index the synapse changed among the
  candidates (the n-th set bit of a candidate mask); its high bits gate the rarer changes.
  - on a confirmed fire: promote one silent synapse; consolidate one active synapse (1/4);
    prune one unused non-sticky synapse (1/2); grow one synapse onto an active input (1/4);
  - a contradicted burst prunes one context synapse;
  - lateral inhibition removes one context synapse from each losing primed cell.
- **Counting:** per input bit, a bitset over cells for each synapse state, updated one bit at
  a time as the masks change (the axons' view of their contacts), summed by the bit-sliced
  adder. A test checks these counts against counts taken from the cells' masks after 2,000
  learning steps.
- **Id index** (`L5_IDX=id`): the event's index and gates from (cell id + step) instead of a
  random word. Deterministic, no generator.
- **Self-calibration** (`L5_META=1`):
  - *rate (metaplasticity):* a change needs k random bits (probability 2^-k). A confirmed fire
    raises the cell's k (it settles, at most 1 in 64); a contradicted fire lowers it (it
    learns faster, at least 1 in 2). While the area's error rate is above one half, every
    cell's k counts one less;
  - *threshold (intrinsic plasticity):* a wrong fire raises the share of input synapses the
    cell needs (at most 95%: more selective); a confirmed fire lowers it slightly (at least
    50%);
  - both start where the fixed values were (k = 2, 80%).

## Results
`LEARNING=three SLOW_P=0.25 SLEEP_P=1 REPLAY_PREDICT=1 L23=primed`, five seeds, held-out
(built before [103](103-primed-layer5-becomes-default.md), so no default layer 5):

| Entry | Reference | Primed L2/3, strengths (101) | Bitwise | Bitwise, id index | **Bitwise + self-calibration** |
|---|---|---|---|---|---|
| family consolidated | 47.7 | 44.8 | 42.0 | 32.6 | 39.5 |
| hippocampus teaches cortex | 51.5 | 45.2 | 44.2 | 46.2 | 45.6 |
| index hippocampus | 34.1 | 41.8 | 36.0 | 30.0 | **47.5** |
| engram store | 45.4 | 47.1 | 44.7 | 45.9 | **50.1** |
| semantic store | 38.6 | 47.4 | 42.2 | 39.0 | 43.3 |
| mean change | | +1.8 | −1.6 | −4.7 | **+1.7** |

**Speed** (hippocampus teaches cortex, seed 0, all side by side under the same load; extra
time over the network without a primed layer):

| | Extra time |
|---|---|
| bit synapses in a per-synapse loop ([102](102-burst-key-l5-interneurons-bit-synapses.md)) | +54 s |
| first bitwise version (each touched cell walks its masks) | +113 s |
| **bitwise, counting through the transposed tables** | **+33 s** |

Learning takes 1% of the run.

## Findings
1. **The fully bitwise cell is the fastest primed layer** (+33 s against +54 s), and its
   learning is nearly free.
2. **With fixed rates it learns a little less well than strengths** (−1.6 vs +1.8).
3. **Self-calibration closes the gap** (+1.7): bits only, no per-synapse count, no hand-set
   rate or threshold beyond their starting points. Cells that keep being right stop changing;
   cells that keep being wrong change faster and become more selective. Index hippocampus
   gains most (+13.4 over the reference).
4. **The id-indexed sweep hurts (−4.7).** A deterministic sweep lines up with the stories'
   regular structure (family consolidated −15). The index needs to be uncorrelated with the
   input; one random word per event costs almost nothing.
5. The family entries still lose with every bitwise variant (−5.7 to −15).

## Next
- The bitwise, self-calibrating cells as **layer 5** (where the default now runs primed cells
  with strengths): if they hold the suite, the default becomes fully bitwise.
- The family entries: what the column and recall say there.
- The engram losses of the default layer 5 ([103](103-primed-layer5-becomes-default.md)).
