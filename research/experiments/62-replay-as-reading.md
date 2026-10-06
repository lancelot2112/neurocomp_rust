# 62 · Replay as reading: inferred stories read through the same steps

**Question.** In [61](61-inferred-replay.md) inferred events were taught to the higher
area as hand-built (input, target) pairs. Most of the failures there came from that: the
input format, the state, the learning rule and the growth masks all differed from
reading. If the hippocampus instead replays an inferred event as a short *story*, read by
the cortex exactly as it reads the page, does the cortex learn the composed answer, and
how far?

**Code.** [`examples/episodic.rs`](../../examples/episodic.rs):
- **The refactor:** the story loop reads, for each story index, the queued replay stories
  first and then the real one, through the same body. Replay stories are read as training
  (so learning is on even on the night before the test). They are flagged `replaying`.
  Nothing is stored in memory: not the engram store, list memory, CA3, the semantic
  buffer, answer traces or word counts. Nothing is counted in training or test
  statistics.
- **The store keeps each row's story prefix** (the words read before its sentence).
- **An inferred event becomes** [its source story's prefix] + [the inferred sentence],
  e.g. "spring came . later it rained . lucy went to the garden ." It is queued `reps`
  times, shuffled.
- `INFER_REPLAY=reps` now reads; `INFER_PAIRS=1` keeps the pair teaching of 61.
- Regression entry `infer-read` (walk only, 50 readings, hippocampus lesioned). Every
  earlier entry is unchanged.

## Results (family stated once; seeds 0 / 1 / 2)

| Setting | Replay | Hippocampus at test | Held out (new names) | Trained | Top-down held the answer |
|---|---|---|---|---|---|
| Walk only | none | lesioned | 23 / 17 / 25% | 64.9% | 37–42% |
| same | pairs (61) | lesioned | 32 / 33 / 27% | 64.8% | 46–47% |
| same | read, 1× | lesioned | 26 / 18 / 27% | 62.0% | 39–46% |
| same | read, 5× | lesioned | 30 / 29 / 33% | 60.7% | 46–52% |
| same | read, 20× | lesioned | 41 / 36 / 40% | 65.2% | 48–56% |
| **same** | **read, 50×** | **lesioned** | **50 / 43 / 50%** | **67.3%** | **54–60%** |
| same | read, 100× | lesioned | 51 / 41 / 51% | 67.1% | 51–58% |
| same | read, 50× | intact | 87 / 90 / 88% | 90.3% | |
| same | none | intact | 91 / 95 / 91% | 89.8% | |
| Semantic store + rollout | none | lesioned | 60 / 64 / 56% | 60.5% | 59–62% |
| same | read, 1× | lesioned | 23 / 21 / 24% | 59.7% | 36–44% |
| same | read, 50× | lesioned | 42 / 38 / 57% | 64.5% | 50–62% |

Cost: about 4,600 replayed words per reading of a seed's ~200 inferred events, against
about 74,600 words read in training.

## Findings
1. **Read as stories, the composed relation reaches the cortex.**
   - With the hippocampus lesioned, new names go from chance (22% mean) to 48%. Pair
     teaching reached 30%.
   - Trained names rise slightly (65 → 67%): the replayed stories are ordinary practice
     for the frame too.
   - No masks, schema heuristics or special learning rule were needed: the cortex learns
     from replay the way it learns from the page. The column learns the sentence; the
     higher area learns where the column was surprised, with its own state.
2. **Repetition matters, as in the brain:** 1× does little, 20× gives 40%, and 50×
   saturates at about 48%. Waking examples of a trained name number in the hundreds.
3. **It conflicts with the existing semantic route.**
   - With the semantic store and rollout completion, new names drop from 60% to 23% at
     1× and recover only to 46% at 50×.
   - Replay teaches the column "lucy" → "went" with confidence, so the rollout no longer
     meets a choice point at "lucy" and never fetches "lucy is a jones" from the semantic
     store.
   - The two routes do the same job, and the new one, while it is still weak, disables
     the old one.
4. **With the hippocampus intact,** replay costs a little (92 → 88%): the cortex's
   strengthened guesses compete with the walk's recall.
5. **The remaining gap** (48% vs 92% with the hippocampus) is the cortex's own limit
   with these inputs:
   - the season must still be in the higher area's surprise window when the question
     comes;
   - the answer has to win against the many "… went to the" kernels of trained names.

## Biology
- **Sleep replay reactivates the cortex in the format of waking experience,** as
  time-compressed sequences (Ji & Wilson 2007).
- **Replay can run sequences never experienced,** assembled from separately learned
  parts (Liu et al. 2019).
- **Consolidation takes many replays over many nights** (systems consolidation; McClelland
  et al. 1995). Here: about 50 readings per inferred event.

## Next
- **Replay across sleeps,** prioritised by the cortex's error: stop replaying events the
  cortex already answers. The store reports error for this, but the read path does not
  feed it back yet.
- **Make the routes cooperate:** let the column's uncertainty, not a hand-set choice
  point, decide when rollout or recall is used.
- **Keep the story's setting in the higher area's state,** so the season survives to the
  question.
