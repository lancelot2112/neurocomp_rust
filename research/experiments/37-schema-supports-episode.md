# 37 · The schema supports the episode: class-level predictions

**Question.** In [36](36-slot-binding-memory.md) the slot ⊗ content memory learned a new
name's place from one or two exposures, and with familiarity-gated arbitration that
keeps learning at test, 28–52% of those answers reached the final output. But the
schema group did no better than the no-schema group. Its cortex, reliable on trained
names, answered novel names with a confident guess and competed with memory. In Tse et
al.'s rats the schema made new pairs *easier*. What would let the learned structure
support the one-shot memory instead of competing with it?

**Code.** In [`examples/episodic.rs`](../../examples/episodic.rs): `CLASS_READ` and
`CLASS_VOTE`.

Run (best setting): `… TASK=season SEASON_LEN=4 POLICIES=nomemory HIER=1 HIER_LEVELS=1
MIX=1 HIER_DREAM=1 HIER_SLEEP_GEN=3 BIND=1 BIND_RARE=1 BIND_FAM=1 MIX_TEST_LEARN=1
CLASS_READ=1 SCHEMA_K=1 cargo run --release --example episodic`.

## Two class-level mechanisms
The column already computes a class-level prediction: the union of what all its
matching kernels predict next, its possible continuations superposed
(`KernelClass::peek_union`, [30](30-cortex-driven-saccades.md)). At "tom went to the ?"
that is every place. Two uses:
- **Class-filtered readout (`CLASS_READ`): the schema supplies the kind, memory which
  one.** Unbinding the expected slot from a recalled episode often returns a
  superposition, such as {cat, dog, bathroom} ([36](36-slot-binding-memory.md)). Among
  the words the unbound episode holds (best overlap first), the first one the column
  expects here wins.
- **Class votes for novel items (`CLASS_VOTE`).** When the sentence's familiarity band is
  low (below 4: the rarest binding in fewer than 8 episodes; hand-set), the column votes
  its whole class instead of one guessed word, and the higher area does not vote. The
  cortex's weight is then spread over the kind, instead of backing one wrong member.

## Results (seeds 0 / 1 / 2; all with rarity-weighted recall, familiarity bands, learning at test)

| | Trained names | New, shown once | New, shown twice | New, shown 4 times |
|---|---|---|---|---|
| Before (36) | 62–70% | 16 / 21 / 28% | 52 / 28 / 31% | 48 / 37 / 30% |
| Class votes for novel items | 66–73% | 16 / 21 / 28% | 52 / 21 / 25% | 34 / 29 / 34% |
| **Class-filtered readout** | **71–82%** | **36 / 28 / 44%** | 38 / 35 / 49% | **39 / 47 / 66%** |
| Both | 69–82% | 24 / 28 / 60% | 38 / 36 / 49% | 52 / 47 / 59% |
| No-schema group, class-filtered readout | 12–18% | 42 / 21 / 70% | – | – |

The slot memory's own accuracy on new names with the filtered readout: 36–61% (shown
once), 46–67% (twice), 59–77% (4 times).

## Findings
1. **The schema supports the episode when it supplies the kind.** Filtering the memory's
   readout by the cortex's expectation lifts new names after one exposure to 28–44% (from
   16–28%), and after four to 39–66%.
   - It also lifts the trained names, 71–82% (from 62–70%): the same filter cleans every
     recall, not only the novel ones.
   - The cortex knows "a place goes here"; memory knows "this person, in this season, went
     to the bathroom". Together they answer better than either.
2. **Class votes alone did nothing.** Spreading the cortex's vote over the class removes
   its wrong guess, but it adds the same weight to every member, so it cannot pick the
   right one. What helps is using the class *inside* memory's choice, where the episode
   supplies the specifics.
3. **The comparison with Tse is now even, not yet in the schema's favour.** After one
   exposure the schema group averages 36%, the no-schema group 44% (seeds vary: 21–70%).
   - The no-schema group's cortex also has an expectation class at "to the" (places come
     after it in every story, random or not), so the filter helps it too.
   - What differs in Tse's design is that the new pairs fit the schema's *specific*
     structure (a known layout), not just its kind. Here, every new pair is arbitrary
     within the class.
4. **So the network now has the full loop of the schema result:**
   - learned structure (slots, a setting slot, place-after-"to the" expectations);
   - one-shot binding of a new item into that structure;
   - recall by structure, filtered by the cortex's expectation;
   - arbitration that learns, online, when to trust memory over cortex.

   After one exposure it answers new names at 28–44%, against 0–8% for names never
   shown.


## Next
- **A schema advantage proper:** new items whose answers follow the learned structure,
  e.g. a new name joining a family whose places follow a known rule. Then the schema
  group should beat the no-schema group, as in Tse's design.
- **Consolidation:** replay the one-shot episodes in sleep so the cortex learns them, and
  the hippocampus is no longer needed for them (Tse's 48 hours).

## Biology
- **Schema-dependent memory** involves hippocampus and medial prefrontal cortex together:
  the prefrontal schema guides what the hippocampus encodes and recalls (Tse et al. 2007,
  2011; van Kesteren et al. 2012).
- **Top-down expectation shapes recall:** retrieval is biased toward schema-consistent
  content (Bartlett 1932's classic reconstruction results). Here that bias filters
  ambiguous recall to the expected kind.
