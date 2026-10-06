# 56 · An engram store: rows of binding ids and grid phases

**Question.** The index memory of [55](55-index-memory.md) works but carries far too
much per row: about 1,000 input bits of story context per event. The proposal tested
here:
- store **indices and phases, not bit fields**: a few binding ids (a word in a slot is
  one id) and one phase per grid module for "where";
- index rows by what (posting lists) and by where (a hash of phase tuples);
- de-duplicate within a theta cycle;
- let a ring buffer's write head do the forgetting;
- replay by strength × the cortex's error.

Is it as good, and how much faster?

**Code.**
- [`src/program/engram.rs`](../../src/program/engram.rs): `EngramStore` (an
  `EpisodicCircuit`), `EngramConfig`, `Dedup`.
- Trait hooks: `report_error`, `begin_sleep`, `end_sequence`, `mark_consolidated`.
- Harness: `HIPPO=engram`, with `ENGRAM_TAU`, `ENGRAM_CAPACITY`, `ENGRAM_BONUS`,
  `ENGRAM_DEDUP`, `ENGRAM_NO_CONTEXT` and `ENGRAM_SEED_PLACE`. Binding ids replace bit
  fields; the cortex's error is reported after each replay.
- Regression entry `engram-store`.

## The design (as built)

**Row:**
- `what`: sorted binding ids, slot × 4,096 + word, with context bindings in their own
  slot fields;
- `phase`: one u8 per grid module, (story clock / scale) mod 256, scales 1, 4, 16, 64;
- `out`: the compact pattern the cortex reads;
- `strength`, `last touch`, `time` (append order), and the cortex's last error.

**Indexes:**
- **What:** a posting list per id. Tombstones are dropped when a list is next written.
- **Where:** a hash of the phase tuple.

**Recall:**
- cue ids are processed rarest first, each walking its posting list with weight 1/n
  through a touched-list;
- recall exits early once the leader cannot be overtaken by the remaining weight;
- rows at the current place can get a bonus.

**Store, as a theta cycle:**
1. Recall with the event.
2. If a row holds exactly these ids, strengthen it and do not store a duplicate.
   - `Any`: wherever the row is.
   - `Move` (default): also move the row to the current place.
   - `Place`: only a row at the current place counts as the same event.
3. Otherwise append a new row and index it, which costs O(K).

**Forgetting:**
- effective strength = strength − (now − last touch) / τ;
- when the write head reaches a strong row, the row gets a second chance (copied
  forward, up to 8 times per write); a weak row is overwritten.

**Move:** a story boundary moves the clock by one, O(G).

**Replay:** at `begin_sleep`, rows are ordered by strength × the cortex's last error, and
replays pop in that order. The harness reports the semantic store's error after each one.

## Results (the experiment 49 task; seeds 0 / 1 / 2)

| Memory | Rows (seed 0) | Train µs/word | Held out, lesioned | Held out, intact | Trained names, intact | Answers changed by recall |
|---|---|---|---|---|---|---|
| Counts circuit (49) | – | 2,190–2,390 | 64 / 59 / 53% | 65 / 59 / 59% | 67% | 0.0% |
| Index memory, bit keys (55) | 16,667 | 1,530–1,750 | 62 / 64 / 65% (forgetting) | 65 / 67 / 60% (no forgetting) | **95%** | 15.4% |
| Engram, content ids only, place bonus 1 id, dedup `Any` | 79 | **220–260** | 60 / 58 / 60% | 61 / 58 / 60% | 60% | 0.0% |
| Engram, content ids, dedup `Move` | 2,329 | 230–270 | **66 / 60 / 62%** | 66 / 60 / 62% | 64% | 0.4% |
| Engram, content ids, dedup `Place` | 13,467 | 330–350 | 60 / 61 / 61% | 60 / 61 / 61% | 62% | – |
| Engram, content + context ids, place bonus 1 id | 12,208 | 710–740 | 64 / 61 / 61% | 64 / 61 / 61% | 63% | 1.5% |
| Same, place bonus 1/16 id | – | 700–720 | – | 67 / 61 / 64% | 63% | 0.1% |
| **Engram, content + context ids, no place bonus** (default) | ~12,000 | **380–440** | 60 / 64 / 56% | 61 / 64 / 57% | **89.5%** | **21.0%** (81% right) |

All engram variants have the stated family right at 97–100% on every seed (counts
circuit: 67–100%). The list memory of 42 / 45 trains at about 230 µs/word.

## Findings
1. **Ids and dedup make the store as cheap as the list memory.** With content ids only
   and dedup, the store holds 79–2,300 rows and trains at 220–270 µs per word. That is
   the list memory's speed, a tenth of the counts circuit's, and every seed's family is
   right.
2. **A place code cannot stand in for the story context, on this task.**
   - The answer depends on the season stated at the start of the story. The useful
     recall is a *past* event with the same content in a story that began the same way.
   - That is a conjunction of content with context *content*.
   - A place (a unique position in the stream of stories) does not generalise across
     stories. Content-only variants recall well but almost never change an answer at
     test: intact equals lesioned.
3. **Context as ids restores the hippocampus's use** at a quarter of the index memory's
   cost.
   - With context bindings stored as ids (about 35 per row against about 1,000 bits),
     recall changes 21% of test answers, 81% of them right.
   - Trained names reach 89.5% intact against 60.5% lesioned, at 380–440 µs per word
     (index memory: 95% at 1,600).
   - Held-out accuracy is not better: 56–64%.
4. **An additive place bonus is wrongly scaled.** With 1/n weights a common binding is
   worth almost nothing, so even 1/16 of an id from the current place outweighs the
   content match, and recall returns the current story's own rows. Place should break
   ties or gate candidates, not add to the score.
5. **Dedup changes what the store is.**
   - `Any` collapses repeated sentences into one row: a semantic store more than an
     episodic one.
   - `Move` (last seen here) was the best content-only variant.
   - With context ids, few events repeat exactly, and the store stays episodic.
6. **Replay by strength × cortex error is wired** (the semantic store's error is
   reported after each replay). Its effect here is not separable from the other changes.

## What this says about "where"
The grid phases here are hand-assigned: a story clock at four scales. On a reading task
the only movement is "next story", one dimension with no commuting actions, and nothing
for phases to express beyond position. What the task needed was "which kind of story",
a property of content.

The proposal for where phases should come from (learn each relation as an operator on
the state; phases appear where the operators commute and are periodic) needs an
environment with such relations. A 2D world with north / east moves is one; book page
or line structure might be another.

## Next
1. **Place as a tie-break or gate,** not an additive bonus; then retest where-cues
   (e.g. "earlier in this story").
2. **Learned where-states** (the operator proposal), on a task that has commuting
   actions:
   - a fresh state id per row;
   - an (state, action) → state table;
   - loop closure by union-find on recognised content, with clone states for
     look-alikes;
   - a commutation test, from which cycles become modules and phases.
3. **Speed:** context ids make the cue about 35 ids long, and common context ids have
   long posting lists. Cap them as the index memory does, or keep only the story's
   surprising bindings as context.

## Biology
- **Hippocampal indexing** (Teyler & DiScenna 1986).
- **Grid cells as periodic codes of position:** Hafting et al. 2005; relations as
  operators in the Tolman-Eichenbaum Machine (Whittington et al. 2020).
- **Grid-like codes for conceptual spaces:** Constantinescu, O'Reilly & Behrens 2016.
- **Clone-structured cognitive graphs:** George et al. 2021.
