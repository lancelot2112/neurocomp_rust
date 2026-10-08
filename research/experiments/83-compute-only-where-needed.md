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

## Addendum: a learned gate

**Code.** `HIER_SURPRISE=learned`: whether to consult the higher area is a basal-ganglia
go/no-go per context (the column's own confidence band, whether the last word surprised
it, the current word; the code shares bits across words and bands). Skipping sends an empty
frame and leaves the area idle.
- **Reward, counterfactual:** consulting is scored against the column's own prediction with
  the top-down frame blanked (one extra look-up): right where it would have been wrong 1,
  wrong where it would have been right 0, no difference one half less `HIER_COST`. Skipping
  is worth one half.
- `HIER_LEARN_ALWAYS=1`: consult on every learning step and let the gate decide only when
  answering (off by default). `HIERGATE` report: share of steps consulted.

**Versions tried (seed 0, held-out; motor speech / story boundary / slot memory):**

| Version | Consulted at test | Motor speech | Story boundary | Slot memory |
|---|---|---|---|---|
| ungated (recorded, seed 0) | 100% | ≈ 61–65 | 95–98 | ≈ 16 |
| reward = next word right − cost 0.05 | ≈ 50% | 55.2 | 95.2 | 63.2 |
| same, cost 0.01 | 0% | 0 | – | – |
| counterfactual, cost 0.05 | 6–11% | 44.4 | 67.2 | 63.8 |
| counterfactual, consult always while learning | 0–58% | 69.0 | **0** | 28.0 |
| counterfactual, cost 0 (suite setting) | 11–38% | 48.6 | 95.8 | 64.0 |

**Three-seed suite, counterfactual, cost 0** (held-out, recorded → learned gate; the area
consulted at 5–58% of test steps, about 20% on most entries):

| Gains | | Losses | |
|---|---|---|---|
| slot memory | 16.3 → **52.0** | sleep generalisation | 74.5 → 27.2 |
| inference replay | 53.4 → **63.1** | belief decides | 81.7 → 59.8 |
| engram walk only | 84.0 → **92.1** | relations, speak, motor speech | 63.5 → 42.6–42.9 |
| engram walk | 66.4 → 67.6 | hippocampus teaches cortex | 61.9 → 43.1 |
| | | schema advantage | 65.7 → 43.6 |
| | | role transfer, cooperate | 60.6 → 45.7, 56.8 → 44.1 |
| | | story boundary, saccades | 94.1 → 83.1 (one seed 58.8), 96.6 → 86.1 |

Most other entries lose 1–7 points.

**Findings.**
1. **The gate learns to skip most steps (70–90%) and keeps the context tasks alive,**
   unlike the hand rule (story boundary 83 against 0), and it keeps the hand rule's gains
   where the frame is noise (slot memory +36, inference replay +10, engram walk only +8).
2. **But one step's counterfactual undervalues the area.** Blanking the frame for one look-up
   rarely flips the next word, so most consultations score "no difference" and the gate sits
   near indifference. What the frame is worth is spread over many steps: the area's own state
   across a story, and what the column learns from it. Skipping stops both, and the
   relation and hippocampus entries lose 10–47 points.
3. **A cost per consultation collapses the gate** (any cost above 0 makes "no difference"
   worse than skipping, which is most steps): the price of compute cannot be learned from a
   one-step reward when the benefit arrives later.
4. **Train and test must gate alike.** Consulting on every learning step and gating only at
   test leaves the column untrained on empty frames: story boundary 0%.
5. Not a default; the option stays off.

## Next
- Credit over time for the top-down gate: reward a consultation by the answers that follow
  (eligibility traces in the basal ganglia), not by the next word alone.
- Keep the area's state running cheaply when not consulted (step its context, skip its
  prediction), so skipping does not lose the story.
- Only then a cost for compute, as the curiosity module charges questions.
