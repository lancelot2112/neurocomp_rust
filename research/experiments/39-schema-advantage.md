# 39 · A schema advantage: new members of a known family

**Question.** In [37](37-schema-supports-episode.md)–[38](38-consolidation-of-one-shot-episodes.md)
the schema group never beat the no-schema group, because every new pair was arbitrary
within its kind: there was no learned structure for it to fit. In Tse et al.'s design the
new pairs *fit* the schema. Here the new items follow the learned structure. Does a
schema now let them be answered at once, and is that knowledge in the cortex?

**Code.** In [`examples/episodic.rs`](../../examples/episodic.rs): `FAMILY`, `SURNAMES`,
`family_place`, `family_of`.

Run: `… TASK=season SEASON_LEN=4 POLICIES=nomemory HIER=1 HIER_LEVELS=1 MIX=1
HIER_DREAM=1 HIER_SLEEP_GEN=3 BIND=1 BIND_RARE=1 BIND_FAM=1 MIX_TEST_LEARN=1 CLASS_READ=1
FAMILY=1 SCHEMA_K=0 cargo run --release --example episodic`, with the full system of
[37](37-schema-supports-episode.md). The exact settings are the `schema-advantage` entry of
[`scripts/regress.tsv`](../../scripts/regress.tsv).

## Task: families
- **Two families, smith and jones.** Each has its own place per season, and every member
  goes there: mary, john and sandra smith; daniel, anna and peter jones. Questions read
  "john smith went to the ?" (season announced 0–3 filler stories before, as in 31).
- **New members at test:** tom smith, lucy jones and sam smith, never seen in training.
  Their places follow their family's rule.
- **Schema group:** trained on the family rule. **No-schema group**
  (`SEASON_RULE=random`): the same stories with random places for trained names. Its new
  members still follow the family rule, so there is something to learn, but no schema to
  learn it from.
- **Exposures:** `SCHEMA_K` = 0 (new members never shown before the test) or 1 (each new
  member shown once per season in the last 600 training stories).
- **Lesion:** `BIND_LESION` switches off the hippocampal slot memory at test.

## Results (seeds 0 / 1 / 2)

| | Trained names | **New members, never shown** | New members, shown once |
|---|---|---|---|
| **Schema group** | 61–82% | **70 / 75 / 55%** | 60 / 73 / 58% |
| No-schema group | 15–18% | 26 / 28 / 10% | 57 / 56 / 70% |
| **Schema group, hippocampus lesioned** | 62–74% | **54 / 72 / 70%** | – |
| No-schema group, hippocampus lesioned | 15–17% | 23 / 18 / 9% | 24 / 11 / 11% |

## Findings
1. **New items that fit the schema are answered with no exposure at all.** In the schema
   group, new family members score 55–75% the first time they appear, about what trained
   names score (61–82%). The no-schema group scores 10–28% on the same items.
   - This is the schema advantage that experiments 34–38 could not show.
   - The schema group learned the family rule over first names: sleep generalisation
     and the near-miss rule drop the first-name bits, since three names share each
     family's places ([31](31-role-cells-and-transfer.md), [33](33-generalisation-during-sleep.md)).
   - A new first name with a known surname then falls under the rule.
2. **That knowledge is in the cortex.** With the hippocampal slot memory switched off,
   the schema group still answers new members at 54–72%. The no-schema group's one-shot
   learning (56–70% after one exposure) disappears with the lesion (11–24%), because it
   lived only in the hippocampus.
   - This matches Tse's lesion results: knowledge that fits a schema is cortical, while
     arbitrary new pairs depend on the hippocampus.
3. **An exposure adds nothing for the schema group** (58–73% with one, 55–75% with none).
   The rule already gives the answer. For the no-schema group, the exposure is everything:
   its one-shot memory carries it.
4. **The two routes, side by side:**
   - **structure** (cortex, slow to learn, immediate for anything that fits);
   - **episodes** (hippocampus, one exposure, for what does not fit).

   This is the complementary-learning-systems picture, and the network now shows both
   halves and how they trade off.

## What this does not yet show
- **Tse's exact effect is one-trial learning of a pair that is arbitrary but fits a known
  layout.** Here the new item's answer is fully determined by its family, so no exposure
  is needed.
- **The intermediate case:** a new member whose family is *only stated once* ("tom is a
  smith"), with later questions naming only "tom". That needs memory to complete the
  missing family ("tom" → smith) and the cortex to apply the family rule: one-trial
  learning that fits a schema. It is the next test.

## Biology
- **Tse et al. 2007, 2011:** schema-consistent information is learned rapidly and becomes
  hippocampus-independent quickly; the medial prefrontal cortex holds the schema.
- **Complementary learning systems** (McClelland, McNaughton & O'Reilly 1995; Kumaran,
  Hassabis & McClelland 2016): neocortex learns structure slowly, hippocampus learns
  episodes fast. New information consistent with existing structure can be integrated
  quickly.
