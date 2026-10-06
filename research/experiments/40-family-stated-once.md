# 40 · "Tom is a smith": one statement, completed from memory, applied by the cortex

**Question.** In [39](39-schema-advantage.md) a new member's surname was in every
question ("tom smith went to the ?"), so no learning was needed. Here the family is
stated once in training ("tom is a smith"), and the test questions name only "tom". To
answer, the network has to chain two steps:
1. complete the missing family from one exposure (tom → smith);
2. apply the family rule (smith in winter → kitchen).

This is the one-trial, schema-consistent learning of Tse et al. 2007, and the first
thinking-in-steps problem of the [roadmap](../roadmap.md).

**Code.** In [`examples/episodic.rs`](../../examples/episodic.rs): `FAMILY_STATED`,
`COMPLETE`, and the `COMPLETE` report.

Run: `… TASK=season SEASON_LEN=4 POLICIES=nomemory HIER=1 HIER_LEVELS=1 MIX=1
HIER_DREAM=1 HIER_SLEEP_GEN=3 BIND=1 BIND_RARE=1 BIND_FAM=1 MIX_TEST_LEARN=1 CLASS_READ=1
FAMILY=1 FAMILY_STATED=1 SCHEMA_K=1 COMPLETE=rollout-test cargo run --release --example
episodic`. The exact settings are the `family-stated` entry of
[`scripts/regress.tsv`](../../scripts/regress.tsv).

## Task
- **As in 39:** two families (smith, jones), and the family decides the place per season.
  Trained questions read "john smith went to the ?".
- **Statements:** a quarter of training stories carry one, "mary is a smith .", about a
  trained name.
- **New members:** with `SCHEMA_K=1`, tom, lucy and sam are each stated once
  ("tom is a smith .", "lucy is a jones .", "sam is a smith .") in the last 600 training
  stories. They are never in a question during training.
- **Test:** "tom went to the ?", with no surname.
- **Chance:** two of the three new names are smiths, so always guessing "smith" and
  applying its rule already scores well above 1/6. The surname each rollout supplies is
  therefore reported directly.

## Completion by rollout (`COMPLETE=rollout`)
**What the diagnostics showed first.** One reading of "tom is a smith" grew a cortical
kernel: after "tom" the column expects "is". At test the page says "went" instead. The
cortex holds the frame of the statement, but not reliably which family.

So the network now follows its expectation when the page skips it:
- **Start:** the next word contradicts a *definite* expectation, meaning the column
  expects exactly one word.
- **Each step:** the expected word is inserted as an internal step, heard from inside
  and not read from the page. The word is the slot memory's ([36](36-slot-binding-memory.md))
  if it is of the kind the column expects, otherwise the column's own. This is the same
  rule as class-filtered readout ([37](37-schema-supports-episode.md)): the schema gives
  the kind, memory which one.
- **Stop:** the rollout continues while the page does not match the expectation, up to 4
  steps per sentence.
- **Example:** "tom went to the" becomes "tom [is a smith] went to the". The column then
  sees "smith went to the" and the family rule applies.

The first version did not work:
- **Memory alone** (`COMPLETE=1`: fill the expected slot once): it never fired. The slot
  expected after "tom" is the "is" slot, which memory holds nothing for.
- **Without the definite gate:** rollouts fired at every sentence start, where the column
  only guesses ("the dog ran away"), and used up their budget.

## Results (seeds 0 / 1 / 2)

| Held-out questions ("tom went to the ?") | Held-out right | Surname the rollout supplied: right family |
|---|---|---|
| Family never stated (`SCHEMA_K=0`) | 24 / 35 / 49% | – |
| Stated once, no completion | 22 / 19 / 26% | – |
| **Stated once, rollout (schema group)** | **35 / 51 / 40%** | **75 / 91 / 74%** |
| Stated once, rollout, hippocampus lesioned | 21 / 42 / 39% | 34 / 68 / 70% |
| Stated once, rollout also during training | 38 / 53 / 40% | 66 / 98 / 68% of rollouts supplying a surname |
| No-schema group, stated once, rollout | 19 / 17 / 17% | 50 / 83 / 33% |

Trained names: 61–64% in the schema group, 15% in the no-schema group.

Held-out answers split by the family the rollout supplied:

| | Right family | Wrong family |
|---|---|---|
| Schema group | **45 / 55 / 51%** right | 3 / 11 / 8% right |
| Schema group, hippocampus lesioned | 54 / 56 / 51% | 4 / 11 / 11% |
| No-schema group | 17 / 16 / 16% | 22 / 21 / 18% |

## Findings
1. **The two steps chain, and each comes from the system that should supply it.**
   - **The family comes from the hippocampus.** With the slot memory, the rollout
     supplies the right family 74–91% of the time. With it lesioned, 34–70%: about what
     the cortex's bias toward "smith" gives, since lucy, the one jones, is mostly called
     a smith.
   - **The rule comes from the cortex.** Given the right family, the answer is right
     45–55% of the time, with or without the hippocampus (51–56% lesioned). Given the
     wrong family it is almost always wrong (3–11%). The answer follows the completed
     family.
2. **The schema is what makes the completed family useful.** The no-schema group
   completes families too (33–83% right), but a family gets it nothing: 16–17% with the
   right one, chance either way. One statement helps only where a rule already exists to
   apply it to. This is Tse's point: one-trial learning works when the new fact fits a
   schema.
3. **A one-shot fact can hurt without the chaining step.** Stated once but not completed,
   new members score 19–26%, below the never-stated 24–49%. The statement leaves the
   cortex expecting "is" after "tom", which spoils the question the rule needs. Following
   that expectation, instead of fighting it, turns the same fact into an advantage
   (35–51%).
4. **The headline gain is modest because of the smith bias.** Against never-stated
   (36% mean), rollout reaches 42% (44% when it also runs in training). The surname
   numbers are the cleaner measure:
   - right family 80% on average with memory, against 57% lesioned;
   - near the two-thirds a constant "smith" guess would give when lesioned.
5. **The rule is weaker in this form than in 39** (about 52% given the right family,
   against 58–73% with the surname in the question). The question the column sees is
   "is a smith went to the", not "tom smith went to the" as in training.
6. **Repeat runs at the same seed differ slightly** (by at most a point here: e.g. 38 and
   39% on seed 2 of the lesion runs), probably from hash-map iteration order.

## What this is, as thinking
- **A first internal step:** when the input contradicts a definite expectation, the
  network continues its own expectation inside, drawing the specifics from memory, until
  the input fits again.
- **Its length is not fixed:** here three steps ("is a smith"), set by when the page
  matched again, not by a hand-set chain as in two-hop ([13](13-big-loop.md)).
- **What it still lacks:** it is triggered by surprise, not chosen. The roadmap's basal
  ganglia choice of "answer / look again / give up", rewarded for being right, would
  decide when an internal step is worth taking.

## Next
- **Consolidation of the stated family:** replay the statement episode in sleep, so
  "tom → smith" moves into the cortex and survives the lesion (Tse's 48 hours). This is
  the test 38 could not pass with arbitrary pairs.
- **Learned stepping:** let the basal ganglia choose when to roll out, rewarded by the
  answer, instead of the definite-expectation trigger.
- **Exactly repeatable runs:** find the source of the small run-to-run differences
  (probably hash-map order) and fix it.

## Biology
- **Schema-consistent one-trial learning:** Tse et al. 2007, 2011. The new pair is
  learned in one trial because it fits a known layout, and its use depends on that
  layout (medial prefrontal cortex).
- **Pattern completion and reinstatement:** the hippocampus completes a partial cue
  ("tom") into the stored episode ("tom is a smith"), and that reinstates the cortical
  pattern (Teyler & DiScenna 1986; Marr 1971).
- **Inner speech / internal simulation:** prediction run forward without input, as in
  hippocampal sequence replay during deliberation (Johnson & Redish 2007; Pfeiffer &
  Foster 2013) and in "thinking as simulated action" (Hesslow 2002).
