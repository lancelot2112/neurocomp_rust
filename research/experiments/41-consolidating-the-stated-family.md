# 41 · Consolidating the stated family: the rule strengthens, the link does not move

**Question.** In [40](40-family-stated-once.md) the hippocampus supplies a new member's
family ("tom is a smith", stated once) and the cortex applies the family rule. In Tse et
al. 2007, schema-consistent pairs stopped needing the hippocampus within 48 hours. Can
sleep replay move "tom → smith" into the cortex, so the chain still runs with the
hippocampus switched off?

**Code.** In [`examples/episodic.rs`](../../examples/episodic.rs):
- `CONSOLIDATE=r` and `CONSOLIDATE_INTERLEAVE`, from [38](38-consolidation-of-one-shot-episodes.md);
- new: `CONSOLIDATE_STEPS` (`=1` or `=assoc`), `ROLLOUT_AREA`, and the `AREADIAG`
  diagnostic (under `COMPLETEDIAG`).

The settings are the `family-consolidated` entry of
[`scripts/regress.tsv`](../../scripts/regress.tsv), plus `BIND_LESION=1` for the lesion.

All runs here are exactly repeatable (`neurocomp::det`, see the
[roadmap](../roadmap.md#infrastructure-repeatable-runs-and-a-regression-suite-done)).
The no-consolidation rows therefore differ from page 40's: same settings, a different
draw.

## What was tried
- **Answer traces** (38's consolidation): every training story's answer leaves a trace
  (question sentence, the story's slot bindings, the answer). The novel ones are replayed
  at each sleep to the higher area, interleaved with as many familiar ones, 3 rounds.
  New names are never in a training question, so no trace holds "tom".
- **Step traces** (`CONSOLIDATE_STEPS=1`): every word the network failed to predict in a
  novel sentence leaves a trace too. For example, "smith" after "tom is a", with the
  sentence so far as input. This is replay of what surprised it.
- **Association traces** (`CONSOLIDATE_STEPS=assoc`): the same, but the input is only the
  sentence's rare words ("tom"). That is an association, "tom goes with smith", rather
  than a sequence.
- **The higher area as a rollout source** (`ROLLOUT_AREA=1`): a rollout step's word
  comes from the slot memory, else the higher area's prediction, else the column. Each
  must be of the kind the column expects. This lets a link consolidated into the higher
  area drive the chain.

## Results (seeds 0 / 1 / 2, schema group, stated once, rollout at test)

| | Held-out right | Family supplied correctly | Right given the right family | Trained names |
|---|---|---|---|---|
| Intact, no consolidation | 42 / 51 / 28% | 73 / 91 / 39% | 55 / 55 / 60% | 65% |
| **Intact, answer traces** | **67 / 53 / 42%** | 95 / 69 / 66% | **70 / 68 / 60%** | 66% |
| Intact, answer + step traces | 51 / 42 / 61% | 81 / 69 / 86% | 62 / 59 / 69% | 67% |
| Intact, answer + association traces | 27 / 42 / 26% | 49 / 62 / 39% | 49 / 63 / 63% | 68% |
| Lesioned, no consolidation | 29 / 42 / 29% | 54 / 68 / 38% | 50 / 56 / 63% | 62% |
| **Lesioned, answer traces** | **44 / 46 / 41%** | 64 / 68 / 65% | **67 / 68 / 59%** | 65% |
| Lesioned, answer + step traces | 37 / 43 / 30% | 60 / 71 / 38% | 60 / 59 / 64% | 65% |
| Lesioned, answer + association traces | 25 / 29 / 23% | 42 / 43 / 34% | 50 / 62 / 63% | 65% |
| No-schema group, lesioned, answer + step traces | 14 / 19 / 18% | 43 / 60 / 65% | 11 / 17 / 20% | 16% |

`ROLLOUT_AREA` changed nothing: every run with it printed exactly the same as without.

## Findings
1. **Consolidation strengthens the rule, not the link.** Replaying answer traces lifts
   held-out answers to 42–67% intact (mean 54%, from 40%) and 41–46% lesioned (mean 44%,
   from 33%).
   - **The gain is in applying the rule.** Given the right family, answers are right
     59–70% (from 50–63%).
   - **The family is not consolidated.** With the hippocampus off, the rollout calls
     every new member a smith on seeds 0 and 1. That is right for two of the three
     (64–68%) without knowing which is which.
2. **Neither step nor association traces moved the link.**
   - Step traces gave about what no consolidation did.
   - Association traces made things worse, even intact (26–42%): the replayed
     "rare word → next word" kernels compete with the area's normal predictions.
3. **Why: no cortical path carries "tom" to the surname step.** The `AREADIAG`
   diagnostic at the rollout's "tom is a ?" (lesioned, step traces):
   - **The column** sees the current word ("a") and the higher area's top-down frame.
     It expects {smith, jones} and guesses one.
   - **The higher area's prediction is still "is"** in most cases, otherwise "jones" for
     every name. It updates only on surprising words, so its top-down frame at "a" is
     the prediction it made at "tom". That prediction is the next word, "is", not the
     family.
   - The replayed sequence "{tom is a} → smith" is never asked in that form at test,
     and the association "tom → smith" competes with "tom → is" for the same next-word
     slot.
4. **What is missing is a cortical *concept* of tom, not a better replay.** In the brain,
   consolidated knowledge about an entity ("tom is a smith") is part of a semantic
   representation that stays active while the entity is in play. A next-word predictor
   holds no such thing.
   - The network has two kinds of store. Next-word predictors (the column, the higher
     area) learn sequences. The slot memory binds episodes. Neither is an associative,
     cortical store that, given "tom", activates "smith" and keeps it active.
   - That store is what consolidation should write into.

## Next
- **Bring back the semantic store of [17](17-consolidation.md).** That store is a
  separate predictive kernel class. It learns cue → content from hippocampal replay: the
  episode's rarest word ("tom") predicts the rest of its novel content ("smith"). It
  stands in when the hippocampus recalls nothing. It was built for the older memory-frame
  pipeline and is not used by the slot-memory, rollout or mixing machinery of 36–40.
  The plan:
  - feed it at sleep from the slot memory's episodes, interleaved, as in 38;
  - make it a rollout source after the slot memory (of the expected kind, as always);
  - make it one more source in the mix.
  - It is the anterior temporal "hub" of semantic memory. Consolidation would write
    "tom ~ smith" there, and the rollout would read it with the hippocampus off.
- **Then rerun this test:** the target is the right family with the hippocampus lesioned
  rising from about 57% (a constant guess) toward the intact network's level.

## Biology
- **Systems consolidation with a schema** (Tse et al. 2007, 2011): fast, and
  hippocampus-independent after 48 hours. Here, half of that appears: the schema's rule
  is strengthened by replay, but the new item's membership stays hippocampal.
- **Semantic memory's hub** (Patterson, Nestor & Rogers 2007; Lambon Ralph et al. 2017):
  the anterior temporal lobes bind an entity's features across modalities into one
  concept. Semantic dementia, which damages them, loses item knowledge while sequences
  and episodes can be spared.
- **Complementary learning systems** (McClelland, McNaughton & O'Reilly 1995): the
  hippocampus teaches the neocortex through interleaved replay. The neocortical learner
  must be able to represent the association being taught; here it cannot yet.

**Update:** done in [42](42-semantic-store.md). With the semantic store fed by
novelty-prioritised replay, the family is supplied correctly 95–100% of the time with the
hippocampus off.
