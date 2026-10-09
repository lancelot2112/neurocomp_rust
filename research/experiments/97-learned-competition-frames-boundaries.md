# 97 · Learned competition, learned frames, and learned event boundaries

**Question.** Three parts of the network are still decided by hand
([audit](../concepts/hand-written-rules.md)):
- **the column's competition:** among matched kernels, the deepest wins, then the most
  reliable. In [94](94-sleep-gated-consolidation.md) (addendum 3), the right consolidated
  candidate matched at 92% of answers and lost on that ranking;
- **the frames:** the column's input row has a fixed layout (word, memory frames, top-down,
  previous);
- **the period trigger:** storage, resets, consolidation traces, word counts and read-back all
  fire on the "." token.

Can each be learned instead?

## Code

**Learned competition** ([`src/kernel/class.rs`](../../src/kernel/class.rs),
`COMPETE=evidence`, `KernelClass::set_evidence_competition`):
- every matched kernel votes for its output, weighted by its reliability times a gain for its
  depth;
- the output with the most evidence wins;
- each depth's gain is learned from how often kernels of that depth were right, ((hits + 1) /
  (n + 2)), from every matched kernel at feedback;
- no order between depths is set.

**Learned frames:** the existing learned routing (`ROUTE=1`, [87](87-three-learning-systems-and-routing.md)),
alone and with the learned competition.

**Learned event boundaries** ([`src/program/boundary.rs`](../../src/program/boundary.rs),
`EVENT_BOUNDARY=learned`). This is event segmentation theory (Zacks et al. 2007): an event
ends where what comes next stops being predictable.
- **The cell.** One boundary cell, with synapses from the current input's code. Its activity
  is its expectation of the column's surprise at the next input. It learns by an integer delta
  rule from the surprise that follows. Nothing names a token.
- **Firing.** It fires when its expectation is above its running mean by more than its running
  mean deviation, with a refractory span of 3 inputs and a capacity of 16 (an event longer
  than that is closed). The page's end closes the last event.
- **What it drives.** Every use of the "." in the reading loop now goes through one signal,
  `boundary_now`: hippocampal storage, sentence resets, `end_sentence` in the areas, word
  counts, consolidation steps, read-back. The current event's start replaces the "search back
  for the last '.'" in about 15 places.
- **Unchanged default.** With the option off, the output is identical to before (checked on
  family consolidated, seed 0).

## Results

**Competition and frames** (`LEARNING=three SLOW_P=0.25 SLEEP_P=1 REPLAY_PREDICT=1`, five
seeds, held-out). The reference is three systems + sleep + the replay fix ([94](94-sleep-gated-consolidation.md),
addendum 2):

| Entry | Reference | Learned competition | Routing | Both |
|---|---|---|---|---|
| family consolidated | 47.7 | 37.8 | 42.3 | 40.0 |
| hippocampus teaches cortex | 51.5 | 48.7 | 32.0 | 43.3 |
| index hippocampus | 34.1 | **40.7** | 34.8 | 32.7 |
| engram store | 45.4 | **48.4** | 40.8 | 44.4 |
| semantic store | 38.6 | 39.7 | 43.5 | 36.4 |

Per-seed spreads are wide: for example family consolidated under the learned competition
gives 28.2 / 28.4 / 48.4 / 48.2 / 36.0. The learned depth gains on seed 0 are
[0.50, 0.32, 0.11, 0.12, 0.50]: depths 2–3 (the kernels reading the frame layout's context)
are the least reliable.

**Learned event boundaries** (default settings otherwise, five seeds, held-out):

| Entry | Recorded | Learned boundaries | Per seed | Boundaries at "." | Periods closing an event |
|---|---|---|---|---|---|
| family consolidated | 52.0 | 49.4 | 55.2 / 42.6 / 65.4 / 37.6 / 46.4 | 89–92% | 98–100% |
| family stated | 38.0 | **45.2** | 54.4 / 33.0 / 61.6 / 41.8 / 35.0 | 89–91% | 98–99% |
| engram store | 67.4 | 60.9 | 68.0 / 56.0 / 59.4 / 63.8 / 57.4 | 90–91% | 98–99% |
| semantic store | 63.1 | 60.2 | 56.4 / 59.2 / 65.6 / 59.2 / 60.6 | 90–92% | 98–99% |
| story boundary | 94.8 | 76.0 | 90.6 / 64.0 / 95.6 / 41.2 / 88.8 | 59–92% | 44–87% |
| relations | 62.8 | 41.6 | 41.2 / 36.4 / 37.8 / 45.2 / 47.6 | 89–93% | 91–98% |

## Findings
1. **The network finds the sentence by itself.** Never told about the ".", the boundary cell
   learns that what follows it is unpredictable. About 90% of its boundaries fall on the "."
   and it closes 98% of sentences on four of six entries. The other 10% fall mid-sentence,
   where the next word is also open (a name or a place).
2. **What it costs depends on the entry.** It is within 3 points on family consolidated and
   semantic store, ahead on family stated (+7), and behind on engram store (−6), story
   boundary (−19) and relations (−21). Relations reads each sentence as one fact, so a
   boundary inside a fact breaks it. Story boundary has longer stories, where segmentation is
   worse on three seeds. It is the first version of the cell, and nothing about its threshold
   has been tuned.
3. **The learned competition is mixed, not a fix.** It gains where the column's general
   kernels hid consolidated ones (index hippocampus +6.6, engram store +3.0), and loses on
   family consolidated (−9.9). A single gain per depth is too coarse. Whether a kernel is
   right depends on the moment, not only on how much of the row it reads.
4. **Routing as built does not replace the layout.** It is worse on three of five entries,
   and with the learned competition on all five. Learning where each frame sits in one row
   is the wrong problem: the frames are of two different kinds.

Nothing becomes a default. All three remain options; the boundary cell is the first
replacement of the period trigger that works on most of the suite.

## Next: a burst gate on apical and basal frames
Inspired by layer 5 pyramidal cells (Larkum's coincidence detection):
- **Two streams.** Basal dendrites receive what is happening (the word, the previous word).
  The apical tuft in layer 1 receives context: feedback from higher areas, the thalamic matrix
  (including the basal ganglia's and the cerebellum's output through the motor thalamus), and
  memory through association cortex. The column's row will be split the same way.
- **A burst** is the two agreeing: the feedforward prediction and the context's prediction
  coincide, and the next input confirms it. A source's burst rate is a reliability of the
  moment, not a stored statistic.
- **A thalamic gate** passes sources whose burst rate is above a threshold. Its gain (the
  reticular nucleus, acetylcholine, norepinephrine) lowers the threshold while nothing passes
  and raises it when too much does. If nothing bursts, the cerebellum's prediction goes
  through.
- **The same signal decides the column's competition:** a kernel that just predicted right
  here wins over a general one. It replaces both the depth gains above and the mix's
  word-keyed reliability tables.
