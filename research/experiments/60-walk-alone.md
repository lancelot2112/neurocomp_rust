# 60 · The walk alone: 92% where the stated fact meets the rule, and none of it in cortex

**Question.** In [59](59-engram-walk.md) the walk added only about 1.5 points. Was the
composition already being done by something else? With the semantic store lesioned, and
then the cortex's rollout completion as well, what does the walk carry alone? And does
the relation reach the cortex?

**Settings.**
- The `engram-walk` entry, with:
  - no semantic store (`SEMANTIC` unset);
  - optionally no rollout completion (`COMPLETE` unset).
- Family stated once (`SCHEMA_K=1`). None of the 500 held-out stories states the name's
  family (checked on the dump).
- Seeds 0 / 1 / 2. "Lesioned" = the hippocampus switched off at test.
- Regression entry `engram-walk-only`: no semantic store, no rollout, walk.

## Results (held out = new names, family stated once in another story)

| Semantic store | Rollout completion (40) | Walk | Hippocampus at test | Held out | Trained | Recall changed the answer |
|---|---|---|---|---|---|---|
| yes | yes | no | intact | 61 / 64 / 57% | 89.5% | 21.0% |
| yes | yes | yes | intact | 64 / 63 / 59% | 89.5% | 25.6% |
| no | yes | no | intact | 61 / 63 / 63% | 89.6% | 15.4% |
| no | yes | yes | intact | 63 / 64 / 62% | 89.6% | 18.5% |
| no | yes | either | lesioned | 41 / 49 / 29% | 64.9% | 0.2% |
| yes | **no** | no | intact | 20 / 19 / 22% | 89.8% | 21.2% |
| yes | **no** | **yes** | intact | **91 / 94 / 90%** | 89.8% | 59.9% (89% right) |
| no | **no** | no | intact | 21 / 17 / 25% | 89.8% | 21.5% |
| **no** | **no** | **yes** | intact | **91 / 95 / 91%** | 89.8% | 54.2% (87% right) |
| no | no | yes | lesioned | 23 / 17 / 25% | 64.9% | 0.8% |

For comparison: the text graph walk (58) 66%; the counts circuit (49) 53–64%.

## Findings
1. **The walk alone composes the stated fact with the rule: 92% held out.**
   - It needs no semantic store and no rollout completion.
   - It is far above the earlier mechanisms (57–64%) and above the text-level graph walk
     (66%). The network answers with the walk's recalled event and the column's
     knowledge of the place slot, not just a lookup of a continuation.
2. **The rollout completion was doing the same hop, worse, and it masked the walk.**
   - Rollout completion (40) makes the cortex "think" the stated fact ("lucy is a jones")
     by feeding hippocampal completions back into the column; it reaches 57–64%.
   - With rollout on, the walk adds 1–2 points. With rollout off, it adds 70.
   - The cortex's internal steps compete with the walk's direct recall: the rollout fills
     the column's input with the statement, and the walk's answer arrives as a second,
     conflicting source.
3. **The semantic store contributes only through rollout.** With rollout off and no
   walk, it gives 20%: its knowledge ("lucy" → jones) enters only as rollout steps.
4. **The relation does not reach the cortex.**
   - With the hippocampus lesioned at test, held out falls to chance (17–25%) whether or
     not the walk was used.
   - The walk acts only at recall. Nothing it composes is replayed or learned. With the
     semantic store and rollout, the cortex alone holds 29–49%, from the store's
     consolidation of the stated fact.
5. **Trained names are untouched** (89.8%), so the walk does not disturb what the cortex
   and plain recall already do.

## Where this helps
The walk is the right tool wherever a fact stated once must be combined with a
regularity from other episodes:
- **Inheritance:** a new entity → its stated category → what that category does.
- **Aliases and coreference:** a new name → the one it was stated equal to.
- **Transitive relations:** A in B, B in C.
- **Transfer:** a known relation in a new story world, where content is new and the
  structure is not.

Its limit is one hop. Chains need repeated steps, with a stopping rule (the answer slot
filled).

## Getting the relation into the cortex
What the walk composes should become a training event. That is generative (inferred)
replay:
1. In sleep, take a tagged new fact ("lucy is a jones").
2. Walk from it with the cues the cortex will face: the question forms seen with the
   family ("… went to the") and each context (season).
3. Replay the composed events ("lucy … went to the hallway" in winter) to the cortex as
   if read.

In the brain, replay has been seen to produce sequences never experienced, joining
separately learned pieces (Liu et al. 2019; Barron et al. 2020). That would make the
cortex answer new names on its own, and make the walk the teacher rather than the only
route.

## Next
1. **Inferred replay** from walks into the column and higher area; then test with the
   hippocampus lesioned.
2. **Rollout and walk together:** let the walk's answer gate the rollout (or the
   reverse), instead of competing.
3. **Generic patterns:** tasks with two-hop chains and with a relation transferred to new
   content, to test the walk beyond family × season.
