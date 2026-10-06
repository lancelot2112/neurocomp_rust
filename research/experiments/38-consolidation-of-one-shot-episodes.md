# 38 · Consolidating one-shot episodes into the cortex (partly)

**Question.** In Tse et al. 2007, new pairs learned in one trial against a schema became
independent of the hippocampus within 48 hours: lesioning it then left them intact.
After [36](36-slot-binding-memory.md)–[37](37-schema-supports-episode.md) the network
learns new name–place pairs in its hippocampal slot memory. Can sleep replay move them
into the cortex, so they survive a hippocampal lesion?

**Code.** In [`examples/episodic.rs`](../../examples/episodic.rs): `CONSOLIDATE=r`,
`BIND_LESION`, and the `CONSOLIDATE` report.

Run: `… TASK=season SEASON_LEN=4 POLICIES=nomemory HIER=1 HIER_LEVELS=1 MIX=1
HIER_DREAM=1 HIER_SLEEP_GEN=3 BIND=1 BIND_RARE=1 BIND_FAM=1 MIX_TEST_LEARN=1 CLASS_READ=1
SCHEMA_K=4 CONSOLIDATE=3 BIND_LESION=1 cargo run --release --example episodic`.

## Systems consolidation
- **A trace per training story,** at its answer: the question sentence, the story's slot
  bindings, the answer, and its familiarity band. This is what hippocampal indexing
  would let the brain reinstate.
- **Replay** at every sleep (every 500 stories), and once more the night before the test.
  Each novel trace (familiarity band < 4, the same hand-set cut as `CLASS_VOTE`) is
  replayed r times to the higher area, which learns from it with its normal rule
  (`HigherArea::learn`).
- **The gist, not the moment.** The replayed slow state holds only the words of the
  episode's uncommon bindings (in at most 30% of stored episodes), such as the season and
  the name. The filler of that moment is left out, so a consolidated kernel is not keyed
  on it.
- **Lesion** (`BIND_LESION`): at test the slot memory is switched off, so only the cortex
  (column and higher area) can answer.

## Results (schema group, seeds 0 / 1 / 2; new names = held-out test answers)

| | Trained names | New names, shown once | New names, shown 4 times |
|---|---|---|---|
| Intact, no consolidation ([37](37-schema-supports-episode.md)) | 71–82% | 36 / 28 / 44% | 39 / 47 / 66% |
| Lesioned, no consolidation | 60–68% | 10 / 4 / 6% | 12 / 18 / 8% |
| **Lesioned, consolidated (r = 3)** | 60–68% | 8 / 10 / 14% | **31 / 22 / 13%** |
| Lesioned, consolidated (r = 10) | 58–72% | 14 / 10 / 11% | 16 / 21 / 14% |
| Intact, consolidated (r = 3) | 64–78% | 11 / 24 / 51% | 32 / 44 / 54% |
| Intact, consolidated (r = 10) | 67–86% | 22 / 32 / 69% | 36 / 43 / 49% |

Each run replayed 770–1,210 traces (r = 3).

## Findings
1. **Consolidation moves some new pairs into the cortex, after several exposures.** With
   four exposures and three replays per sleep, the lesioned cortex answers 13–31% of
   new-name questions (mean 22%), against 8–18% (mean 13%) without consolidation. After
   one exposure it barely moves (8–14% against 4–10%).
2. **More replay is not more consolidation.** Ten replays per sleep did no better on the
   lesioned test (11–21%). Replaying the same trace again adds no new evidence, and the
   area's learning rule grows on error: once the replayed trace is predicted, further
   replays change nothing.
3. **With the hippocampus intact, consolidation mostly helps what was already known.**
   Trained names rise to 67–86% with ten replays. New names are mixed: better on one seed
   (51–69%), worse on others (11–32%). The newly consolidated cortical kernels now compete
   with the memory's answer, as the cortex did in [36](36-slot-binding-memory.md) before
   the arbitration learned.
4. **Why not more.** A consolidated kernel is keyed on (question sentence, gist), and at
   test the higher area's input holds the same question words and the season. But its
   slow state also holds the test story's own surprising filler, and the gist sampled
   at replay may include the episode's rarer filler words. The kernel then needs words
   the test story does not have.
   - The cortex learns from replay as it learns from reading: one input, one target.
   - Complementary learning systems call for *interleaved* replay of many episodes, old
     and new. Generalisation across them (the offline rule of [33](33-generalisation-during-sleep.md))
     can then drop what varies. Here each episode is replayed alone, and generalisation
     runs on a separate buffer.

## Interleaved replay feeding generalisation (`CONSOLIDATE_INTERLEAVE=1`)
- **Interleaved.** Each sleep replays the novel traces mixed with as many randomly drawn
  familiar ones (trained names), shuffled anew in each of r = 3 rounds.
- **Feeding generalisation.** Every replayed (input, target) pair also enters the replay
  that sleep generalisation reads (`KernelClass::add_replay`), and generalisation from
  replay ([33](33-generalisation-during-sleep.md)) runs right after. Rules can then form
  across new and old episodes.

| Schema group, seeds 0 / 1 / 2 | Lesioned: trained | Lesioned: new, once | Lesioned: new, 4 times | Intact: trained | Intact: new, once | Intact: new, 4 times |
|---|---|---|---|---|---|---|
| No consolidation | 60–68% | 10 / 4 / 6% | 12 / 18 / 8% | 71–82% | 36 / 28 / 44% | 39 / 47 / 66% |
| Block replay (above) | 60–68% | 8 / 10 / 14% | 31 / 22 / 13% | 64–78% | 11 / 24 / 51% | 32 / 44 / 54% |
| **Interleaved + generalisation** | 62–73% | 15 / 10 / 9% | **27 / 23 / 22%** | 66–81% | 26 / 32 / 56% | 37 / 49 / 60% |

The generalisation pass formed 35–83 general rules per run.

5. **Interleaving makes consolidation steadier, not much larger.** After four exposures
   the lesioned cortex answers 22–27% on every seed (block replay 13–31%, none 8–18%).
   Trained names hold up slightly better under the lesion (62–73%). With the hippocampus
   intact, interleaving no longer costs new names as block replay did (26–56% and 37–60%,
   close to no consolidation).
6. **One-exposure consolidation is still out of reach.** After a single exposure the
   lesioned cortex stays at 9–15%. Generalisation needs at least three confirmed
   near-misses per kernel, and one episode replayed among others supplies one. In Tse's
   rats, the new pair fits a schema that already holds the structure, so consolidation
   only has to add one association. Here every new pair is arbitrary within its kind,
   which is the test-design point already noted in [37](37-schema-supports-episode.md).

## Next
- **A test where new items follow the learned structure** (a new name joining a family
  whose places follow a known rule). There, consolidation of one exposure should be
  possible, because the general rule already exists and only the new item's membership
  must be learned.
- ~~**Interleaved replay with generalisation:**~~ done (above).
- **Diagnose the consolidated kernels:** at a lesioned new-name test, which kernels
  match, and which of their inputs are missing?

## Biology
- **Systems consolidation:** memories move from hippocampus to neocortex over time
  (Squire & Alvarez 1995; McClelland, McNaughton & O'Reilly 1995). With a schema it is
  fast: 48 hours in Tse et al. 2007.
- **Interleaved replay** is what lets cortex absorb new items without catastrophic
  interference (McClelland et al. 1995). Replaying a new item alone is the regime they
  warn against.
- **Hippocampal indexing:** the hippocampus stores an index that reinstates the cortical
  pattern of an episode (Teyler & DiScenna 1986). The traces here are that index, used
  for replay.
