# 95 · Place cells in the index circuit, and a dopamine–novelty loop for replay

**Question.** Two steps from the [hippocampus's to-do list](../concepts/hippocampal-functions.md#an-index-to-the-cortex-not-a-store-of-vectors):
1. **The engram store as a circuit.** The index circuit (`IndexMemory`: one unit per event,
   rare inputs weighted more, the best match wins) works on bit patterns but has no place
   code. The engram store's place is a separate hash and bonus. Plausibly, place cells are
   just more input: each story's place bits are stored with every event and added to every
   cue. Inverse weighting then makes them decisive only where this story has events.
2. **The dopamine–novelty loop** (Lisman & Grace 2005). Hippocampal novelty drives the VTA
   (through the subiculum, accumbens and pallidum), and dopamine back in the hippocampus
   makes new memories persist and marks them for replay. Consolidation replay was chosen by a
   hand rule, familiarity band < 4.

**Code.**
- [`src/program/index_memory.rs`](../../src/program/index_memory.rs): `place_bits` (16 per
  story, from a pool of 4,096), with `row_here`, `recall_here_all`, `recall_soft`, `last_row`
  and a test (`place_cells_prefer_this_episode`). The option is `INDEX_PLACE=16`.
- [`examples/episodic.rs`](../../examples/episodic.rs), `DA=1`: each consolidation trace gets
  a dopamine level. That is the hippocampus's novelty for the answer sentence's event (halved),
  plus 1/4 if any novelty, plus 1/4 if the answer was right. Each sleep, a trace is replayed
  with probability equal to its level.

## Results

Five seeds, held-out:

| Entry | Recorded | `DA=1` |
|---|---|---|
| family consolidated | 52.0 | 51.4 |
| hippocampus teaches cortex | 60.9 | 48.0 |
| index hippocampus | 66.0 | 57.3 |
| engram store | 67.4 | 62.0 |
| semantic store | 63.1 | 61.4 |

Mean dopamine 0.17–0.19 on the engram entries, about 1,800 traces chosen over all sleeps (the
band rule replayed about 3,000). On one entry it was 0.83, with 8,441 chosen.

| Index hippocampus | Held-out | Per seed |
|---|---|---|
| recorded | 66.0 | |
| place cells (`INDEX_PLACE=16`) | 56.5 | 47.8 / 63.0 / 47.4 / 57.8 / 66.6 |
| place cells + reinstatement + ACh mode | 52.2 | 55.6 / 59.2 / 36.2 / 67.2 / 42.8 |

## Findings
1. **Dopamine tagging as built is not calibrated.** Novelty measured on the answer sentence
   ("tom went to the") is low for almost every story, so far fewer traces replay and three
   entries lose 5–13 points. On the entry using another circuit, novelty is high everywhere.
   It is the right mechanism, with the wrong measurement: the novelty that matters is that of
   the story's new bindings ("tom is a smith"), not of the question.
2. **Place cells as input make recall prefer the current story during training too**, and
   the index-hippocampus entry loses 10 points with a wide spread. This is the effect story
   context had in [90](90-holding-an-open-question.md): the suite's answers lean on recall of
   other stories. Reinstatement and the mode on top make it worse (52.2).
3. **Where hippocampal output goes matters.** In the brain, the return path (CA1 and
   subiculum → deep entorhinal layers V/VI → association cortex) serves both awake recall and
   sleep replay. Replay reaches the cortex in a receptive window (slow oscillation, spindle and
   ripple coupled). It reaches primary areas only through association cortex's own feedback.
   Ours sends recall into the column's row (`HC_EC`) and, in [94](94-sleep-gated-consolidation.md),
   replay into the column. Reinstating into the higher area ([92](92-reinstatement.md)–[93](93-neuromodulated-hippocampus.md))
   is the one path that has worked.

Nothing becomes a default.

## Next
- **Consolidation through association cortex only:** replay to the higher area with
  sleep-gated plasticity there; the column learns from it through the top-down frame it
  already reads.
- **Dopamine from the story's new bindings,** calibrated so the replay volume stays
  comparable, then with reward from the outcome of recall at the answer.
- **Place cells gated by the mode:** in the cue only in retrieval mode (low acetylcholine),
  never while encoding, as in Hasselmo's account of the entorhinal input.
