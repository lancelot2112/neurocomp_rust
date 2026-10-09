# 93 · An acetylcholine mode that gates reinstatement, and a norepinephrine gain

**Question.** In [92](92-reinstatement.md), reinstating the cortical state was destructive
when other stories' states came back. A rule restricting it to "this story only" fixed that.
An acetylcholine-like mode driven by the episode's novelty fixed it without the rule, but also
removed the question task's small gain: its signal came from the sentence recall, whose best
match was usually an old story. Two builds
([neuromodulation](../concepts/hippocampal-functions.md#acetylcholine-and-norepinephrine-modes-of-the-hippocampus)):
1. **One recall, one decision.** The mode's signal comes from the same place-weighted
   retrieval that reinstates. If its events include one from this story, the context is
   familiar here, so reinstate (low acetylcholine). If not, encode (high).
2. **A norepinephrine-like gain** (`NE=1`):
   - *phasic, a salience tag:* a sentence that surprised the column at two or more words is
     stored twice, so its event is strengthened as by a repeat;
   - *tonic, adaptive exploration* (Aston-Jones & Cohen): a running error rate of training
     answers raises the basal ganglia's exploration while outcomes are poor (0.1 up to 0.5)
     and lowers it as they improve.

  The test: does the exploration revive the learned hold that recall-tagged credit
  extinguished ([91](91-learned-working-memory-hold.md))?

**Code**: [`examples/episodic.rs`](../../examples/episodic.rs) (`ACH` with `REINSTATE`, `NE`);
`row_here` on the engram store.

## Results

**Acetylcholine, signal from the reinstating retrieval** (question task, learned routing,
five seeds, held-out):

| | Per seed | Mean | ACh at test |
|---|---|---|---|
| routing alone | 69.2 / 66.6 / 69.0 / 68.6 / 69.6 | 68.6 | |
| reinstatement ([92](92-reinstatement.md)) | 72.0 / 72.6 / 66.8 / 69.2 / 69.2 | 70.0 | |
| + ACh, signal from the sentence recall (92) | 71.6 / 66.0 / 67.8 / 67.8 / 70.4 | 68.7 | 0.93–0.95 |
| **+ ACh, signal from the reinstating retrieval** | **72.2 / 76.2 / 71.0 / 70.6 / 73.6** | **72.7** | 0.46 |

On the suite (seeds 0–1, recorded → reinstatement with this mode): engram store 67.4 → 63.8,
inference read 45.6 → 46.5, relations 62.8 → 65.3. On inference read and relations the level
was not updated at test and stayed at its starting value, holding reinstatement back.

**Norepinephrine**:

| | Result |
|---|---|
| learned hold + attach with `NE` (question task, five seeds) | 54.0 / 61.2 / 62.6 / 44.6 / 58.0, mean 56.1 (recall-tagged credit alone: 59.2). Exploration averaged 0.36 in training, but nothing is held at test |
| salience tag alone on the suite (seeds 0–1) | engram store 64.6, inference read 42.5, relations 65.7 (recorded 67.4 / 45.6 / 62.8) |

## Findings
1. **The acetylcholine mode, given the right signal, turns reinstatement into a gain:**
   68.6 → 72.7, every seed above the baseline. The mode now really switches (mean 0.46).
   When this story's events are found, the cortex's earlier state for this story comes back.
   When they are not, nothing foreign is reinstated. There is no "this story only" rule: the
   switch is the hippocampus's own judgement of familiarity in this context.
2. **It keeps the suite safe**, within noise on the three entries tried, as the rule did.
3. **Raised exploration does not revive the learned hold.** The extinction in 91 is not
   only a matter of too little exploration. The chain a hold must complete to be credited
   (attach, store, recall for the answer) stays too rare, whatever the exploration rate.
4. **The salience tag gives nothing measurable** on these entries. Storing a surprising
   event twice strengthens an engram row that recall already finds.

The acetylcholine mode with reinstatement is the first combination of these experiments
that is positive on the question task and safe on the suite. It is not a default yet: the
suite's other engram entries and five seeds are needed first.

## Next
- Five seeds of `REINSTATE=1 ACH=1` on every engram entry of the suite, with `STORE_TEST`
  and without; make the mode's update run on every recall path (two entries skip it).
- Season errors still dominate (131 of 500 on seed 0). Reinstate the best event's own state
  rather than up to four, so a known person's context does not come along.
