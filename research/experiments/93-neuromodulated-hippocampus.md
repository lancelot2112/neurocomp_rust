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

## Addendum: all engram entries at five seeds, and the best event only

**The suite** (`REINSTATE=1 ACH=1`, test stories not stored, five seeds, held-out, recorded →
with reinstatement and the mode):

| Entry | Recorded | Reinstatement + ACh | Per seed |
|---|---|---|---|
| engram store | 67.4 | 65.9 | 63.6 / 64.0 / 66.6 / 69.4 / 66.0 |
| engram walk | 66.8 | 67.8 | 66.8 / 65.4 / 70.6 / 70.6 / 65.6 |
| engram walk only | 87.0 | **91.2** | 92.4 / 89.0 / 92.6 / 90.0 / 91.8 |
| inference replay | 59.4 | **65.0** | 60.4 / 62.0 / 67.0 / 69.4 / 66.0 |
| inference read (lesioned at test) | 45.6 | 48.2 | 43.0 / 50.0 / 54.6 / 45.4 / 47.8 |
| cooperate | 54.0 | 57.8 | 60.4 / 55.8 / 64.0 / 60.8 / 48.0 |
| relations (lesioned at test) | 62.8 | 65.8 | 63.0 / 67.6 / 71.2 / 64.0 / 63.4 |
| speak | 63.1 | 65.8 | 63.2 / 67.8 / 71.2 / 63.6 / 63.2 |
| speech motor | 62.9 | 65.8 | 63.0 / 67.6 / 71.4 / 63.6 / 63.4 |
| belief decides | 82.4 | 80.7 | 79.4 / 80.8 / 81.4 / 80.6 / 81.4 |

Eight of ten entries rise and two fall slightly. The mean change is +2.3 points: the engram
walk alone +4.2, inference replay +5.6, relations, speak and speech motor about +3. Each
difference alone is within the spread of five seeds; together they lean one way.

The two entries that lesion the hippocampus at test also gain. There, reinstatement acted
only in training, so what the cortex learned while its earlier states were reinstated
carried into reading without the hippocampus. The runs with test stories stored, and the
control for them, were stopped before they finished.

**The best event only** (`REINSTATE_TOP=1`, question task, five seeds):

| | Mean | Season errors (of 500) | Other-family errors |
|---|---|---|---|
| top 4 (above) | 72.7 | 131 on seed 0 | |
| top 1 | 71.2 (73.4 / 69.8 / 72.4 / 71.8 / 68.4) | 84–107 | 27–68 |

One event brings the season back more often, but the single best event is often the known
person's sentence, which brings the other family. The retrieval does not know which event is
about whom: the binding problem, which this task does not otherwise test. Four events
dilute it.

**Status.** Reinstatement with the acetylcholine mode gains on the question task (+4.1) and
leans positive on the suite (+2.3 on average, 8 of 10 entries). It is the strongest candidate
for a default from 88–93. The remaining check is the full suite (the other entries do not use
the engram store and are unaffected).

