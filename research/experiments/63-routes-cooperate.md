# 63 · Routes that cooperate through the cortex's uncertainty

**Question.** In [62](62-replay-as-reading.md), replay taught the column "lucy" → "went",
and that silenced the semantic route. Rollout completion needs a *contradicted definite
expectation* to fetch "lucy is a jones", and after replay there is none. New names fell
from 60% to 46% with the hippocampus lesioned. Can the routes cooperate if memory is
consulted according to the cortex's uncertainty, instead of a hand-set trigger?

**Code.** [`examples/episodic.rs`](../../examples/episodic.rs): `COOPERATE=1`.
- At each step the column's own prediction (current + previous word, no top-down or memory
  frames) is peeked.
- Where it is missing or under half reliable, the semantic store is asked about the
  sentence's least familiar word.
- Its content ("lucy" → "is a jones") joins the higher areas' sentence context for that
  step only. The word stream is not touched.

Regression entry `cooperate`: cooperation + 50 readings of replay, rollout off,
hippocampus lesioned.

## Results (family stated once; seeds 0 / 1 / 2; semantic store + walk on)

| Rollout | Replay (62) | Cooperation | Hippocampus at test | Held out | Trained |
|---|---|---|---|---|---|
| on | none | none | lesioned | 60 / 64 / 56% | 60.5% |
| on | 50× | none | lesioned | 42 / 38 / 57% | 64.5% |
| on | none | on every surprising word | lesioned | 58 / 39 / 45% | 50.7% |
| on | none | **where the column is unsure** | lesioned | 58 / 61 / 60% | 59.7% |
| on | 50× | **where the column is unsure** | lesioned | **56 / 62 / 60%** | 58.1% |
| **off** | 50× | **where the column is unsure** | lesioned | **56 / 62 / 60%** | 58.1% |
| on | 50× | where the column is unsure | intact | 88 / 93 / 86% | 87.7% |
| (walk only, 60) | none | none | intact | 91 / 95 / 91% | 89.8% |

## Findings
1. **Gating memory by the cortex's uncertainty removes the conflict.**
   - With replay, new names go back from 46% to 59.5%, the level of the semantic route
     alone.
   - The column's confident "lucy → went", learned in replay, no longer shuts memory out:
     memory is asked where the column is unsure of what follows (at "… went to the"), not
     where a definite expectation is contradicted.
2. **Rollout completion is no longer needed.** With rollout off, the results are
   identical. The hand-set choice-point rule (definite expectation, contradicted by the
   page) is replaced by the column's own reliability.
3. **Consulting memory on every surprising word hurts.** New names fall to 47% and
   trained names to 51%. The semantic store's content for ordinary words ("dog" → "the ran
   away") floods the higher area's context. Uncertainty about what comes *next* is the
   right gate, not surprise at what just came.
4. **The routes do not add up yet.**
   - Semantic store alone: 60%. Replay alone: 48%. Both together: 59.5%.
   - Both carry the same knowledge (lucy's family) to the same place (the higher area's
     context at the question), so one covers what the other does.
5. **With the hippocampus on,** cooperation costs a little against the walk alone (89%
   vs 92%): the semantic content added at uncertain steps competes with the walk's
   recalled event in the mix.

## How this maps onto the hippocampus–cortex relation
- **The hippocampus already volunteers.**
  - It is cued automatically on every word: the current sentence's bindings and the
    story's context, about 70,000 recalls per run.
  - Whatever completes is posted back: as a memory frame to the column, and as a candidate
    to the source mix.
  - No request line. Thresholds (the minimum overlap, the walk's bridge rule) decide
    whether anything fires.
- **The gating is by cortical uncertainty.** The source mix weights the hippocampus's
  candidate by learned reliability, keyed on the column's confidence band, so recall
  counts where the column is unsure and little where it is sure. This experiment gives
  the semantic store the same footing: asked only where the column is unsure.
- **Usage fades as the cortex learns.** After replay the column is sure of "lucy went to
  the", and memory is no longer consulted at those steps.
- **Not modelled yet:**
  - prefrontal steering of the cue (a controller that edits the cue and the threshold);
  - novelty damping recall during encoding (only the store's strength uses novelty);
  - replay timed by cortical slow oscillations.

## Next
- **Graded instead of binary gating:** weight what each store posts by the column's
  prediction error, so both always answer and the weights decide. This is the source
  mix's job: give the semantic store's context a candidate in the mix, not only an
  input.
- **A controller that edits the cue:** when the first recall does not settle the answer,
  try the next least familiar word (the walk's bridge, chosen by the cortex).
