# 96 · An association area as the hippocampus's cortical partner

**Question.** In the brain, hippocampal output leaves through the deep entorhinal layers to
association cortex (perirhinal, parahippocampal) for both awake recall and sleep replay. It
reaches primary areas only through association cortex's own feedback
([the entorhinal gateway](../concepts/hippocampal-functions.md#the-entorhinal-cortex-one-gateway-both-ways)).
In [94](94-sleep-gated-consolidation.md)–[95](95-circuit-place-and-dopamine.md), sending
hippocampal output into the column failed, and reinstating into the higher area worked. Does
a dedicated association area, built to hold and learn from what the hippocampus sends, carry
consolidation?

**Code** ([`examples/episodic.rs`](../../examples/episodic.rs), `ASSOC=1`). A second area of
the higher area's kind ([sentence | slow context] → next word):
- **learning:** slow while awake (growth probability `ASSOC_P`, 1/16). During sleep,
  consolidation replay goes to it instead of to the higher area, with growth probability 1
  and the uncertainty gate open;
- **the hippocampus's return:** reinstated states (`REINSTATE`) join its context, not the
  higher area's;
- **reaching reading:** through the thalamic mix, as a source of its own whose reliability
  per context is learned.

## Results

Five seeds, held-out:

| Entry | Recorded | + association area | Three systems | Three + association area |
|---|---|---|---|---|
| family consolidated | 52.0 | 41.1 | 40.6 | 37.8 |
| hippocampus teaches cortex | 60.9 | 43.9 | 44.8 | 44.3 |
| index hippocampus | 66.0 | 44.5 | 31.4 | **45.8** |
| engram store | 67.4 | 48.0 | 39.0 | 39.8 |
| semantic store | 63.1 | 50.4 | 36.4 | 36.9 |

Per-seed spreads are wide (31–70). The area grows to 1,650–2,230 kernels from 1,440–2,540
replays. At test it proposes a word at every answer and is right at 30–45%.

On the question task (`REINSTATE ACH`, learned routing), reinstatement into the association
area gives 68.4 (71.2 / 68.0 / 66.0 / 66.2 / 70.6). Into the higher area it gave 72.7
([93](93-neuromodulated-hippocampus.md)). The area reaches 4,900–5,600 kernels from about
28,000 replays and is right at 35–47% of test answers.

## Findings
1. **A separate area that reaches reading only through a vote is too weak a path.** Moving
   consolidation replay out of the higher area takes it away from the area the column
   already reads top-down. The vote of a 30–45%-right source cannot replace that, and the
   default consolidation entries lose 11–22 points.
2. **The higher area is already our association cortex.** It reads the sentence and a slow
   context, learns quickly, receives consolidation replay (`CONSOLIDATE`), and is the target
   where reinstatement worked. A second area beside it duplicates its role with a worse
   connection to reading.
3. **Under three learning systems, one entry gains** (index hippocampus 31.4 → 45.8, which
   lesions the hippocampus at test). There the area answers from consolidated knowledge
   alone. The others are unchanged.
4. Nothing becomes a default.

## Next
- Keep the higher area as the association cortex. Make its role explicit: the hippocampus's
  input is built from its state, and reinstatement and replay go to it, as they do now.
- If a second area is kept, its prediction must reach the column the way the higher area's
  does, as a top-down frame and not only a vote, and replay should reach both areas, as replay
  reaches association cortex broadly.
- Consolidation under three learning systems is still the open gap. It is not in where
  replay goes: what is replayed (with memory's frames) and the slow column's sleep merging
  ([94](94-sleep-gated-consolidation.md)) remain the leads.
