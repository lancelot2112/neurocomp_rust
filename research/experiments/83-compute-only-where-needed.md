# 83 · Recall and the higher area only where the column needs help

**Question.** Hippocampal recall and the higher area's prediction run on every word and are
the largest costs. Can they run only where the column needs them: recall reusing CA3's
settled state within a sentence, the higher area idle while the column is sure?

**Code.** [`examples/episodic.rs`](../../examples/episodic.rs):
- `HC_SURPRISE=1`: recall again only on a surprising word, a sentence's first, or a column
  unsure of the next word (its own prediction, without memory or top-down frames, under
  half reliable); otherwise the last recall in the sentence (CA3's settled state) is reused.
- `HIER_SURPRISE=1`: the higher areas predict and learn only then too; otherwise they send an
  empty top-down frame.
- `GATED` report: steps reused and run.

## Results
**Gating on the last word's surprise alone** (motor speech, seed 0): the higher area's frame
was stale exactly where the answers are (they follow predictable words: "X went to the _"):
seen 0.6%, held-out 8.2%. Recall alone: 66.2 / 64.8% → 63.2 / 63.0%, 68% of recalls reused.

**Gating on "the column needs help"** (surprise, or unsure of the next word), the three-seed
suite, held-out (recorded → gated):

| Entry | Recall gated | Higher area gated |
|---|---|---|
| story boundary (26) | 94.1 → 94.1 | **94.1 → 0.0** |
| saccades (29) | 96.6 → 96.6 | **96.6 → 0.0** |
| slot memory (36) | 16.3 → 16.3 | **16.3 → 40.7** |
| family consolidated (41) | 53.9 → 53.9 | **53.9 → 43.1** |
| full hippocampus (46) | 41.4 → 41.4 | 41.4 → 35.4 |
| hippocampus teaches cortex (49) | 61.9 → 61.9 | **61.9 → 50.5** |
| index hippocampus (55) | **66.3 → 50.6** | 66.3 → 55.7 |
| engram walk only (60) | 84.0 → 83.8 | **84.0 → 90.2** |
| inference replay (61) | 53.4 → 54.3 | **53.4 → 65.8** |
| cooperate (63) | 56.8 → 61.3 | 56.8 → 62.4 |
| relations, speak, motor speech | 63.5 → 63.5–63.8 | **63.5 → 66.3** |

About 42% of recalls and 42% of the higher area's steps are skipped (seed 0, motor speech:
113 → 98 s with both).

## Findings
1. **When to consult the higher area decides a lot, both ways.** Withholding its frame where
   the column is sure helps where the frame is noise (slot memory +24, inference replay +12,
   engram walk +6, the relation entries +3) and is fatal where the frame carries the story's
   context the column cannot see (story boundary and saccades: 0%).
2. **Recall gating is close to neutral:** the same on most entries, better on cooperation,
   worse on the index hippocampus; it saves less than hoped, since the column is unsure
   more often than it is surprised.
3. **The trigger must be about the next word,** not the last: answers follow predictable
   words.
4. Neither becomes a default. A hand rule cannot say when top-down helps; the network should
   learn it per context.

## Next
- A learned gate on the top-down channel: a basal-ganglia go/no-go per context (the column's
  confidence band, the word), rewarded by whether consulting the area changed the prediction
  for the better, and charged for its compute, as the curiosity module charges questions.
