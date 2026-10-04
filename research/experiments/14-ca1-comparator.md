# 14 · A CA1-style comparator: store what wasn't predicted

**Question.** Experiments [11](11-episodic-memory.md)–[13](13-big-loop.md) decide what
to store, cue with and read out by *frequency* (habituation: drop bits that are in more
than 40% of episodes; cue with the rarest bits). Can the cortical predictor's own
**prediction error** do this instead, as CA1's comparison of expected and actual input
does ([Hasselmo 2005; Lisman & Grace 2005](../related-work.md#hippocampus-and-entorhinal-cortex))?
See [hippocampal functions](../concepts/hippocampal-functions.md#novelty-should-come-from-prediction-not-frequency).

**Code.** `NOVELTY=prediction` in [`examples/episodic.rs`](../../examples/episodic.rs).
Each word's predicted probability is compared with what arrived; the sentence's
*surprising* words are stored as the episode, used as the cue, and read out whole
(no habituation). `PREDICTED_SHARE` (0.5) is the threshold; `RARITY=all` cues with every
surprising word instead of the rarest stored item. The output adds **answer in
recall**: how often the recalled frames contain the answer, separating memory from
predictor failures.

## Comparator versions (1 seed, held-out pairs)

| Comparator | Cue | Varied 1–2 facts | Varied 1–3 facts | Answer in recall (1–2) |
|---|---|---|---|---|
| frequency habituation ([11](11-episodic-memory.md)) | rarest | 98.1% | 98.5% | – |
| bitwise mismatch (`code AND NOT prediction`) | all surprising | ~7% (short) | – | – |
| word-level: share of prediction < 0.5 | rarest stored | 63.8% | 38.4% | 74.6% |
| **word-level × kernel reliability < 0.5** | rarest stored | **93.2%** | **86.6%** (answer in recall 98.3%) | 93.4% |

## What each step showed
1. **Bitwise mismatch fails because predictions superimpose classes.** Early on, the
   prediction after "where is" overlapped every name, so the actual name counted as
   "predicted" and the cue was just "?". The comparison must be graded: how much of the
   prediction did *this* word get.
2. **Prediction removes function words without counting.** With the word-level
   comparator the stored frequencies are: "the" 0.00, "to" 0.01, "went" 0.03 of episodes,
   names ~0.14, places ~0.12, while "picked" is stored *most* (0.23): after a name the
   predictor can't tell "went" from "picked", so it is surprising. Novelty is relative to
   the predictor, not to the corpus.
3. **"Predicted" isn't "already known".** A binary hit test skipped facts whose place the
   predictor happened to guess ("sandra went to the office" with office predicted), so
   memory kept Sandra's *old* location. Weighting by the predicting kernel's reliability
   (a lucky guess from a 20%-reliable kernel is still a surprise) fixed it: 64% → 93%.
4. **What is still frequency-based:** choosing *which* stored item to cue with (the
   rarest among stored items, an IDF-like specificity). Cueing with every surprising
   word recalls old question episodes when "?" or "where" was unpredicted.
5. **Fillers and adverbs are genuinely unpredictable** ("then", "so", "quickly") and are
   now stored and recalled alongside the place; the predictor must learn to ignore them.

## Two-hop stories
TWOHOP_PROB

## Next
- Make the cue choice learned too (which surprising item to cue with) — a basal-ganglia
  style selector with delayed reward ([basal ganglia and cerebellum](../concepts/basal-ganglia-and-cerebellum.md)).
- A better predictor makes a better comparator: with only two words of context, "?"
  after "where is the ball" is unpredictable.
