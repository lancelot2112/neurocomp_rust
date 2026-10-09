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

## Addendum: why the slow cortex never answers

A diagnostic (`OWNDIAG`) prints what the column itself predicts at test answers (hippocampus
teaches cortex, `LEARNING=three SLOW_P=0.25 SLEEP_P=1`, gate open, seed 0). At all 1,000
answers it predicts **"cat"**, the commonest word after "the" ("the cat slept"), whatever the
question:

> ["mary", "smith", "went", "to", "the"]: column says "cat", answer bedroom

The mix's report agrees across runs: "column alone 0.0% right". The mixed answer equals the
cerebellum's accuracy (49.0 vs 48.7, 65.9 vs 65.9). So under three learning systems every
answer comes from the cerebellum, and nothing consolidation adds to the slow column reaches
an answer.

The cause is the column's competition, not plasticity or where replay goes. The general
kernel "the → cat" has the most evidence and wins the winner-take-all at every "the". The
place kernels that replay grows are specific and young, and they never win.

The two leads above:
- **memory's frames:** moot. These entries run `POLICIES=nomemory`, so the replayed row has
  no memory frames and matches what the cortex has at test.
- **merging:** secondary. It only removes a kernel that a more general, at least as reliable
  kernel with the same output covers.

**Next.** Let consolidated, context-specific kernels compete fairly. Two candidates:
- **Specificity:** a matched kernel that reads more of the context outranks a general one
  once it is reliable (the existing `TRUST` rule, depth over reliability above a floor).
- **Replay as evidence:** each replay in which a kernel predicts the trace's answer counts as
  a hit, so consolidated kernels arrive with the evidence they need.

## Addendum 2: a replay bug, and consolidation reaching the slow cortex

**The bug.** Learning (`KernelClass::feedback`) credits hits and misses to the kernels that
matched at the *last prediction*, and decides growth from the last winner. Consolidation
replay called `learn` on the replayed input without predicting it first. Every replay was
therefore credited against whatever the network last predicted while awake. With the
surprise gate on, if that stale winner happened to predict the replayed answer, learning was
skipped entirely. This affects replay into the higher area in the default suite too, so the
recorded figures carry it. `REPLAY_PREDICT=1` predicts each replayed input before learning
it, at all six replay sites (higher area, association area, column).

**Results** (five seeds, held-out):

| Entry | Recorded | Default + fix | Three + sleep-gated (above) | Three + sleep-gated + fix |
|---|---|---|---|---|
| family consolidated | 52.0 | 52.3 | 41.3 | **47.7** |
| hippocampus teaches cortex | 60.9 | 57.4 | 45.4 | **51.5** |
| index hippocampus | 66.0 | 56.3 | 31.8 | 34.1 |
| engram store | 67.4 | 66.6 | 40.8 | **45.4** |
| semantic store | 63.1 | 66.1 | 35.7 | 38.6 |

Under three systems, hippocampus teaches cortex on seed 0: the column alone is right at
**21.2%** of test answers (0.0% before), the mix 69.5%, all sources agreeing 18.8% of the time
(93% right then).

1. **Consolidation now reaches the slow cortex.** Under three learning systems every entry
   rises by 2–6 points, and the slow column answers for the first time. The gap to the default
   narrows: family consolidated 47.7 vs 52.0, hippocampus teaches cortex 51.5 vs 60.9.
2. **In the default network the fix is mixed:** semantic store +3, index hippocampus −10,
   others within noise. There the higher area had been learning from replay with the stale
   credit, and the suite's figures were recorded that way.
3. The column's winner-take-all still favours the general "the → cat" kernel most of the
   time (the column alone right at 21%). The specificity and evidence leads above remain.

`REPLAY_PREDICT` stays an option: it is correct, but it changes recorded figures in both
directions, so it needs the full suite at five seeds before it becomes the default.

## Addendum 3: the right answer is among the candidates; the hand-set ranking discards it

With the replay fix (`REPLAY_PREDICT`), a diagnostic of the matched kernels at test answers
(`OWNDIAG`, `KernelClass::matched_kernels`; hippocampus teaches cortex, three systems, seed 0)
shows:
- the column now proposes places ("bedroom"), not "cat";
- about 9 kernels match per answer, 7 of them proposing a place;
- **some matched kernel proposes the right place at 925 of 1,000 answers.**

All of them read the same number of frames (depth 3), so reliability decides. The same
general "bedroom" kernel (hit rate 0.29) wins over the right one (0.13–0.29), and the column
alone is right at 21%.

**Specificity** (`SPECIFIC=1`, `KernelClass::set_specificity`): within a depth, more matched
input bits before reliability. It is worse. On seed 0 the mixed answer is 52.0 (69.5 without),
held-out 37.8 (54.8), and the right answer is among the matched kernels at 695 answers (925).
The rule also changes which kernels win and learn during training.

**What this shows.** The competition among the column's kernels is ranked by hand: depth
(frames in a hand-set layout), then reliability. The right, consolidated candidate exists but
the ranking cannot tell it from a general one of the same depth, and a second hand rule
(specificity) made it worse. A flexible fix replaces the hand parts:
- learned routing (`ROUTE`) instead of the fixed frame layout, so depth means something the
  network chose;
- a learned competition: candidates weighed by a learned, per-context reliability, as the
  thalamic mix already weighs sources, or chosen by the basal ganglia.

