# 68 · Frame words as entities, and sparse gating of both stores

**Questions.**
1. In [67](67-relation-store-in-reading.md), "jones" is a filler of "X is a Y" but part of
   the frame of "X jones went to the Y", so the relation store cannot go lucy → jones →
   place. Can a frame word also be read as an entity?
2. Memory helped most when it spoke sparsely: only where the column is unsure, and only
   the filler ([64](64-graded-gating.md), [67](67-relation-store-in-reading.md)). Does the
   hippocampus's answer need to be sent at the other steps at all? And can the basal
   ganglia learn the gate instead of the fixed threshold?

**Code.**
- [`src/program/relations.rs`](../../src/program/relations.rs), `RelationStore::lift`:
  - every fact is also read a second time, kept beside the first;
  - in the second reading, a frame position becomes a filler if it is weak (under 3/4 of
    the fact's neighbours agree there) or holds a word that is an entity (a filler)
    elsewhere;
  - "john jones went to the hallway" is then also "X _ went to the Y" with fillers (john,
    jones, hallway).
  - Test `a_frame_word_is_also_an_entity`:
    - "lucy is a jones" → jones → {hall, yard}, the joneses' places, read with `ask_all`;
    - with four joneses against three smiths, "jones" is a (weak) frame word of "X is a
      jones": the same ambiguity as "grandfather" in "X 's grandfather is Y". Both
      readings are kept.
- [`examples/episodic.rs`](../../examples/episodic.rs):
  - `REL_LIFT=1`: the second reading.
  - `REL_HOPS=2`: the store's answer adds a second hop, forward steps only (lucy → jones →
    the joneses' places).
  - `SPARSE_HC=1`: the hippocampus's answer (the slot readout, the route it answers
    through here) enters the source mix only where the column's own prediction is under
    half reliable.
  - `GATE=learned`: the basal ganglia decide whether the cortex asks the relation store,
    per (the column's own confidence band × the cue word's familiarity band). Reward: the
    higher area's outcome on the next word, minus a cost of 0.05 for asking.
  - `GATE` report: how often the store was asked in each confidence band, and how many
    hippocampal answers were withheld.

## Results (family stated once; seeds 0 / 1 / 2; cooperation + 50 readings of replay + relation store, as the `relations` entry)

| Setting | Hippocampus at test | Held out | Trained |
|---|---|---|---|
| relation store (67) | lesioned | 65 / 65 / 64% | 62.5% |
| + lifted frames | lesioned | 65 / 69 / 61% | 62.0% |
| + lifted frames, two hops | lesioned | 68 / 67 / 61% | **66.7%** |
| fixed gate (67) | lesioned | **65 / 65 / 64%** | 62.5% |
| learned gate | lesioned | 45 / 37 / 57% | 64.7% |
| hippocampus always sent (67) | intact | 86 / 93 / 91% | 88.7% |
| **hippocampus sent only where the column is unsure** | intact | **86 / 93 / 90%** | 88.7% |

What lifting finds (seed 0): a fourth relation, "X _ went to the Y". jones → {garden,
hallway, bathroom, bedroom} and smith → {kitchen, office, hallway, bedroom}: each family's
places across the seasons. No relations of relations were found (none exist in this
world).

## Findings
1. **A frame word can be an entity, and the store finds it by itself.** The second reading
   links each family to its places. lucy → jones → places can now be followed in the
   cortex alone.
2. **For new names it changes little** (64.4 → 65.1%, 65.5% with two hops). The second hop
   gives all four of a family's places; the season, which picks one, is in the higher
   area's state, and the higher area already maps "jones" + season to the place.
3. **Two hops help trained names** (62.5 → 66.7%): for a trained name, the family's places
   are the right candidates, and the area picks among them.
4. **The hippocampus is sparse in time for free.** Sending its answer only where the
   column is unsure withholds about 13,500–14,000 answers per seed (about 60% of test
   steps) and loses nothing (89.7% against 89.8%). Where the column is sure, the
   hippocampus's answer was redundant.
5. **The learned gate fails** (46% against 64%).
   - On two seeds it learned never to ask in the lowest confidence band, exactly where
     asking is needed. On seed 2 it asked at the most confident band too.
   - Its reward, the higher area's outcome on the next word, is dense: thousands of steps
     where asking changes nothing outweigh the few questions where it decides the answer.
     This is the same failure as the cue controller of [65](65-cue-controller.md).
   - A learned gate needs credit at the answer (as learned stepping in
     [43](43-learned-stepping.md) had), or a signal of when asking changed the outcome.
6. **So both stores can be sparse in time with a fixed threshold on the column's own
   reliability.** It is a single, local quantity (the winning kernel's hit rate) and
   needs no training. The threshold being right first time is the open question: why
   one half?

## Next
- Credit the gate at the answer, or by counterfactual (would the area have been right
  without asking?), before trying to learn it again.
- Count the messages: bits sent from memory to the cortex per word, as a cost to report
  beside accuracy.
