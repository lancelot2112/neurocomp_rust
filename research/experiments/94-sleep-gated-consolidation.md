# 94 · Sleep-gated consolidation into the slow cortex

**Question.** Under three learning systems ([87](87-three-learning-systems-and-routing.md)),
the consolidation entries stay 10–35 points under the default even after recall was
repaired. The slow cortex stays small (about 66–95 kernels) and answers nothing alone at
test. Consolidation replay taught only the higher area. In the brain, replay during sleep
(low acetylcholine) is when cortex learns readily, and it reinstates the cortical state of
the experience ([index theory](../concepts/hippocampal-functions.md#an-index-to-the-cortex-not-a-store-of-vectors)).
Does replaying the column's own state of the moment to it during sleep, with its
plasticity open, move facts into the slow cortex?

**Code** ([`examples/episodic.rs`](../../examples/episodic.rs), `SLEEP_P`):
- Each consolidation trace also keeps the column's input row at the answer: the cortical
  state of that moment.
- At every sleep, the replayed trace teaches the column too. The column's growth probability
  is raised to `SLEEP_P` (1) during replay. Awake it stays at `SLOW_P` (0.25 here).
- The uncertainty growth gate is opened during replay and restored after (second run). In
  the first run it blocked growth 12,000–17,000 times.
- `KernelClass::growth_gate` returns the gate so it can be restored.

## Results

`LEARNING=three SLOW_P=0.25`, five seeds, held-out:

| Entry | Default | Three systems | + sleep-gated replay | + gate open while asleep |
|---|---|---|---|---|
| family consolidated | 52.0 | 40.6 | 39.6 | 41.3 |
| hippocampus teaches cortex | 60.9 | 44.8 | 45.4 | 45.4 |
| index hippocampus | 66.0 | 31.4 | 25.6 | 31.8 |
| engram store | 67.4 | 39.0 | 39.0 | 40.8 |
| semantic store | 63.1 | 36.4 | 35.4 | 35.7 |

The slow cortex on seed 0, by entry:

| | Gate closed | Gate open |
|---|---|---|
| live kernels | 87–95 | 148–171 |
| merged by the column's sleep pass | 94–110 | 183–230 |

Replays reaching the column: 1,224–2,748 per run.

## Findings
1. **With the gate closed, replay cannot reach the slow cortex.** The uncertainty gate reads
   the replayed contexts as noisy and refuses growth, so thousands of replays add about ten
   kernels.
2. **With the gate open, the slow cortex grows but answers barely move** (+1–2 points on
   three entries, within noise). The column's own sleep pass merges about as many kernels
   as replay adds. And the replayed row is the column's input *with* memory's frames, which
   are absent once the hippocampus is lesioned at test.
3. **The gap is not plasticity alone.** What is replayed must be what the cortex will have to
   answer from: without memory's frames, the cue alone. And the sleep pass must not undo it
   (merging new, rarely confirmed kernels).

Nothing becomes a default.

## Next
- Replay the trace without memory's frames, as the cortex must answer alone at test.
- Exempt newly consolidated kernels from the sleep pass's merging for a while.
- Prioritise replay by dopamine (novelty and reward) instead of the familiarity-band rule
  (`band < 4`): the dopamine–novelty loop, next.
