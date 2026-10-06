# 36 · Slot ⊗ content memory: one-exposure learning works in the hippocampus, not yet in the answer

**Question.** [35](35-fading-state-and-entorhinal-codes.md) concluded that the schema
test needs a structural code, as in medial entorhinal cortex and the
Tolman-Eichenbaum Machine (TEM). Episodes should be bindings of slot and content
("setting = winter", "person = tom", "place = kitchen"), recalled by slot, not bags of
words. Built here, then rerun on the schema test of [34](34-schema-test.md).

**Code.**
- `EpisodicMemory::recall_rare` and `recall_rare_scored` in
  [`src/program/memory.rs`](../../src/program/memory.rs), with a unit test.
- In [`examples/episodic.rs`](../../examples/episodic.rs): `BIND`, `BIND_HAB`,
  `BIND_RARE`, `BINDDIAG`, and the `BIND` report. The slot cells are the `RoleArea` of
  [31](31-role-cells-and-transfer.md).

Run (best setting): `… TASK=season SEASON_LEN=4 POLICIES=nomemory HIER=1 HIER_LEVELS=1
MIX=1 HIER_DREAM=1 HIER_SLEEP_GEN=3 BIND=1 BIND_RARE=1 SCHEMA_K=1 cargo run --release
--example episodic`.

## The slot ⊗ content memory
1. **Slot cells that follow a sequence.** At each word, the role cells (competitive
   Hebbian, [31](31-role-cells-and-transfer.md)) see the column's expectation for this
   slot plus the previous slot's code, rotated into the other half of the vector. The
   winning cell is the word's slot.
2. **Binding.** A word the column found surprising is stored bound to its slot: its code
   rotated by the slot's offset. A training story's episode is the union of its
   bindings (sparse: only surprising words, as CA1 would pass on).
3. **Recall by slot.** At each step:
   - the story's bindings so far cue the store;
   - the recalled episode is unbound with the offset of the slot the column expects
     next (the cell its next-word expectation selects);
   - decoding it answers "what filled this slot in the matching episode?".
4. **Into the answer.** That answer is one more source in the precision-weighted mix
   (`SourceMix`, [24](24-cortical-hierarchy.md)), with its learned reliability.
5. **Rarity-weighted recall** (`recall_rare`). Each cue bit votes with an integer weight
   ≈ log2(stored episodes / episodes containing it), from bit lengths. A rare binding
   (a name seen once) then outweighs several common ones.

## What emerged (`BINDDIAG`, seed 0)
- **A setting slot formed by itself:** slot 21 held only seasons (autumn, winter,
  spring). This is the clean context code [34](34-schema-test.md) lacked, learned without
  labels.
- **Coarser elsewhere:**
  - the person slot (0) is shared with every sentence start ("it", "the", "later",
    names);
  - the place slot (10) is shared with "cat" and "dog".
- **Readout works:** when the right episode is recalled, unbinding the expected slot
  gives the right place (`tom` → garden, `sam` → bedroom).
- **Plain overlap recall** picked episodes with the right season but the wrong person.
  Several common bindings ("it@0", "the@0") outvoted the one rare name binding. Hence
  rarity weighting.

## Results (seeds 0 / 1 / 2)
The slot memory's own answer, before the mix:

| Slot memory right on new-name answers | Plain recall | Habituated cue (0.3) | **Rarity-weighted recall** |
|---|---|---|---|
| New names never shown | – | – | 0 / 3 / 2% |
| **Shown once** | – | – | **28 / 31–33 / 69–76%** |
| Shown 4 times | – | – | 66–70 / 43–54 / 63–68% |
| All test answers, shown once | 29–31% | 31–40% | 40–60% |

Final answers (after the mix):

| | Trained names | New names, shown once | New names, shown 4 times |
|---|---|---|---|
| Schema group, no slot memory (34) | 63–68% | 6–10% | 8–18% |
| Schema group, slot memory | 57–70% | 5–14% | 8–17% |
| No-schema group, slot memory | 12–18% | **15–44%** | – |

## Findings
1. **One-exposure learning works in the hippocampal store.** After a single exposure, the
   slot memory answers a new name's place correctly 28–76% of the time, against 0–3% for
   names never shown. A learned setting slot, slot ⊗ content binding and rarity-weighted
   recall give one-shot association of (person, setting) → place.
2. **But the final answer does not use it in the schema group.** There the mix keeps
   trusting the cortex, the higher area and the column, which are reliable on trained
   names. For a name they have never learned they answer confidently and wrongly. Keying
   the memory's reliability on its recall strength did not change that: strong recalls
   (a rare name) hardly occur in training, so there is no evidence to trust them.
3. **The no-schema group does use it** (up to 44% on new names). Its cortex is unreliable
   on every name, so the mix learned to listen to memory. That is the reverse of Tse's
   result, and it locates the gap exactly. The schema group knows *what* to do (its
   structure is learned, and its memory can bind one-shot) but not *when* to defer to
   memory.
4. **The missing piece is familiarity-gated arbitration.**
   - The hippocampus should answer when the cortex's answer is not backed by familiarity
     with the item, a CA1-comparator or familiarity signal
     ([hippocampal functions](../concepts/hippocampal-functions.md)).
   - The store already measures it: a name's bits are rare in it.
   - Keying every source's reliability on that familiarity band (familiar or novel item)
     would let the mix learn, from the few training exposures, that the cortex is wrong
     on novel names and memory is not.

## Familiarity-gated arbitration (`BIND_FAM=1`)
- **The signal.** For each binding in the current sentence (a surprising word in its
  slot), the store's own statistics give how many stored episodes contain it. The rarest
  one's count, in log2 bands 0–7, is the sentence's familiarity: a name seen once falls
  in a low band, a trained name in a high one.
- **The use.** The band is part of every source's reliability key in the mix, so the mix
  learns, separately for familiar and novel items, how far to trust the column, the
  higher area and the slot memory. No threshold is set by hand.

| Schema group, seeds 0 / 1 / 2 | Slot memory on new names | Final answer, new names, without familiarity | **With familiarity** |
|---|---|---|---|
| Never shown | 2–3% | 5–8% | 5% |
| Shown once | 28 / 38 / 58% | 5–14% | 13 / 18 / 11% |
| Shown twice | 56 / 50 / 67% | – | **24 / 19 / 20%** |
| Shown 4 times | 66 / 52 / 68% | 8–17% | 10 / 29 / 26% |
| No-schema group, shown once | 30–69% | 15–44% | 20–59% |

Trained-name accuracy is unchanged (56–72%).

5. **Familiarity moves the answer toward memory, but only partly.** With two or four
   exposures, new names reach 19–29% in the final answer (from 8–17%), still well below
   what the slot memory knows (50–68%).
   - The mix learns its reliabilities only during training, and the novel band occurs
     there only in the 12–48 exposure stories, so its trust in memory for novel items
     rests on very few cases.
   - The obvious remedies: let the reliability counters keep learning at test (the brain
     does not stop learning), or seed the novel band from the no-schema-like case
     (cortex unfamiliar, memory specific) as a prior.

## Next
- ~~**Familiarity-gated arbitration:**~~ partly done (above). Next: more evidence for the
  novel band (learning at test, or a prior).
- **Familiarity-gated arbitration (first idea):** the familiarity band of the current sentence's
  surprising words, from the store's own frequency statistics, in every source's
  reliability key. Then rerun the schema test.
- **Consolidation:** replay the one-shot episodes in the next sleeps, so the cortex
  learns them too, as in Tse's 48-hour consolidation.

## Biology
- **TEM** (Whittington et al. 2020): structural codes in medial EC, sensory codes in
  lateral EC, conjunctive binding in the hippocampus. The slot cells, rotations and
  episodes here are a bit-level version.
- **Familiarity and recollection** are separate signals: perirhinal familiarity, and
  hippocampal recollection (Eichenbaum, Yonelinas & Ranganath 2007). A novelty signal is
  what lets the brain switch to hippocampal encoding and retrieval (Hasselmo 2005).
