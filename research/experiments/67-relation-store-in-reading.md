# 67 · The relation store in reading, and relations of relations

**Question.** The relation store of [66](66-typed-relations.md) answers name + relation →
filler in a world built for it. Wired into reading in place of the semantic store's bag,
does it parse the stories' facts by itself, and does the cortex do better with it? And
can it learn relations of relations (grandfather = father's father) from what it is
told?

**Code.**
- [`src/program/relations.rs`](../../src/program/relations.rs) (`RelationStore`):
  - **Frames from fact shapes, not word frequency.** A fact's neighbours are the facts of
    the same length that agree with it in all but at most two positions; the positions
    where more than half of them agree are its frame, the rest its fillers. (Indexed by
    buckets: each fact is filed under every way of leaving two positions out.)
  - **Relations of relations at sleep.** For each stated fact (x, r, z), every two-step
    path x → y → z' through the store is counted. A path that gives the stated filler
    for at least three quarters of the r-facts it applies to (and at least twice) becomes
    a rule r = s1 ∘ s2. The rule infers r for every entity that has the path and was never
    told r, and those facts are replayed into the store like stated ones.
  - Test `grandfather_is_learned_as_father_of_father`: grandfathers stated for 6 of 9
    grandchildren. The store learns grandfather = father then father, and answers the 3
    grandchildren it was never told about (seeds 0, 1, 2, 3, 5, 9).
- [`examples/episodic.rs`](../../examples/episodic.rs): `REL=reps`. Training sentences
  are read into the store (not replayed stories, not test stories); at each sleep, facts
  are parsed and replayed `reps` times. Wherever the semantic store was asked about the
  sentence's least familiar word, the relation store answers instead: the relation the
  sentence's words name, if the word has it, else everything it knows about the word, as
  plain word codes. The `REL` report lists relations, the new names' answers, and the
  rules.
- Regression entry `relations` (cooperation + 50 readings of replay + relation store,
  hippocampus lesioned).

## What it learns from the stories (every seed)
- **Three relations:** "X is a Y" (lucy → jones), "X jones went to the Y" and "X smith
  went to the Y" (a first name → a place).
- **About the new names:** lucy → jones, tom → smith, sam → smith; the family, nothing
  else.
- About 3,750 facts parsed per run, 22,500 replays, 66 kernels.
- The first version split frame from fillers by word frequency, and failed: family names
  are frequent everywhere, while "is a" occurs only in facts, so it parsed "jones" as the
  frame and "is", "a" as fillers.
- No relations of relations: the task has none to find (the family is in the frame of
  the "went to" relations, not an entity of its own).

## Results (family stated once; seeds 0 / 1 / 2; hippocampus lesioned unless stated)

| Store | Reader | Replay (62) | Held out | Trained |
|---|---|---|---|---|
| semantic bag (60) | rollout | none | 60 / 64 / 56% | 60.5% |
| roles (66) | rollout | none | 68 / 68 / 55% | 64.4% |
| **relations** | rollout | none | **64 / 59 / 67%** | 61.6% |
| semantic bag (63) | cooperation | none | 58 / 61 / 60% | 59.7% |
| **relations** | cooperation | none | **61 / 58 / 63%** | 62.6% |
| semantic bag (63) | cooperation | 50× | 56 / 62 / 60% | 58.1% |
| **relations** | **cooperation** | **50×** | **65 / 65 / 64%** | 62.5% |
| semantic bag (63) | cooperation | 50×, intact | 88 / 93 / 86% | 87.7% |
| relations | cooperation | 50×, intact | 86 / 93 / 91% | 88.7% |

## Findings
1. **The relation store parses the stories' facts by itself.** From sentence shape alone it
   finds "X is a Y" and answers each new name with its family. No relation, role or slot
   was given.
2. **It gives the best consolidated figure on this task:** 64.4% for new names with the
   hippocampus lesioned (cooperation + replay), even across seeds (65 / 65 / 64%). The bag
   gave 59.5%. The answer is the filler alone ("jones"), not the fact's frame words ("is",
   "a"), so less noise reaches the higher area.
3. **It adds to replay where the bag did not.** With the bag, cooperation and replay
   together were no better than the semantic route alone (63: 60 / 48 / 59.5%). With
   relations: 60.9% without replay, 64.4% with.
4. **Intact, it costs nothing** (89.8% against 89% with the bag, 92% for the walk alone).
5. **Relations of relations work where the world has them** (the family-tree test): a
   composition is accepted only when it reproduces stated facts, and the facts it infers
   are consolidated into the store, so the cortex answers what it was never told. This is
   the same move as inferred replay (61, 62), done by the cortex on relations instead of by
   the hippocampus on episodes.

## How it relates to the hippocampus
- **Both are heteroassociative:** a key recalls a different pattern.
- **The hippocampus (engram store)** learns in one shot, stores whole episodes (bindings +
  context), completes them from part, and chains episodes (the walk).
- **The relation store** learns slowly by replay, stores single relations (one key → one
  filler), keyed by a permutation of the entity's code by the relation, and chains
  relations (rules). It plays the neocortex of complementary learning systems.

## Next
- **Relations whose fillers are relations' frames:** "X is a jones" and "X jones went to
  the Y" share "jones" as a filler in one and frame in the other; a parse that lets a
  frame word also be an entity would let the store compose lucy → jones → (went to) place
  without the hippocampus.
- **Rules with longer paths and inverse steps** (uncle = father then brother; sibling =
  father then inverse father).
