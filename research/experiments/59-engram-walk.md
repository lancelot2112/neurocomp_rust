# 59 · The walk, built into the engram store

**Question.** [58](58-a-story-is-a-graph.md) showed that one hop of a walk over a
plain words-in-sentences-in-stories graph composes a fact stated once with a rule learned
from many stories: 66% held out against 17% for text lookup. Does the same walk, built
into the engram store as the hippocampus's recall, lift the network?

**Code.**
- [`src/program/engram.rs`](../../src/program/engram.rs):
  - `EngramStore::recall_walk`;
  - a word index (word → rows, any slot);
  - `EngramConfig::{walk, walk_rare, word_space, content_fields}`;
  - `EpisodicCircuit::report`.
- Harness: `ENGRAM_WALK=1`, `ENGRAM_WALK_RARE`, and a `CIRCUIT` report line.
- Regression entry `engram-walk`. Unit test `a_walk_composes_a_stated_fact_with_a_rule`.

## The walk
1. **Recall** as usual.
2. **Find a bridge.** The winning row must come from another episode (another story),
   and must have been reached through a rare content binding of the cue: rare *as a
   word*, in at most 2 rows in any slot. A name stated once in another story is the
   case.
3. **Recall again with a substituted cue:**
   - the cue's other content, as words in any slot: a candidate must contain at least
     half of them;
   - the cue's context bindings (slot-specific);
   - the first row's other words, as a bonus that chooses among the candidates.

   For example, "lucy went to the" (+ winter) → "lucy is a jones" → a jones event in
   winter, "… went to the hallway".
4. **Answer** with that row. With no bridge, or nothing else in the cue, recall is as
   before.

**Three corrections were needed on the way:**
- **Bridge rarity by word:** about 16,000 walks per run otherwise. Ids are word-in-slot
  and the learned slots vary, so ordinary words often sit in rare slot ids.
- **The first row's words choose; they do not match.** Otherwise early in the question
  (cue: just "lucy") the second step returns another statement ("anna is a jones"). This
  cost seed 2 twenty points.
- **Cue content as words:** a stated question has a different form ("lucy went to the")
  than the stories that answer it ("john jones went to the"), so its slot-specific ids do
  not match.

## Results (the experiment 49 task, family stated once; seeds 0 / 1 / 2; hippocampus intact at test)

| Hippocampus recall | Held out | Trained | Recall changed the answer | Walks per run |
|---|---|---|---|---|
| Engram store, no walk (56) | 61 / 64 / 57% | 89.5% | 21.0% (81% of those right) | – |
| First walk (bridge rare as an id) | 64 / 63 / **37%** | 89.7% | 25.0% | ~16,000 |
| **Walk, as above** | **64 / 63 / 59%** | 89.5% | 25.6% (77% right) | 5,500–11,500 |
| *Text graph walk (58), seed 0 dump* | *66%* | *85%* | | |

With the hippocampus lesioned at test, the walk makes no difference (60 / 64 / 56%), as
expected: it acts at recall.

## Findings
1. **The walk lifts held-out answers a little:** 61 / 64 / 57% → 64 / 63 / 59%, mean
   +1.4 points. The network now matches the text-level graph walk on seed 0 (64 vs 66%).
2. **Most of the composition was already there.**
   - The network without the walk was at 57–64%.
   - The semantic store, consolidated by tagged replay (49), already maps a new name to
     its family, and the cortex's sleep generalisation holds the family × season rule.
   - The walk adds a second, hippocampal route to the same answer. It does not open a
     new kind of answer.
3. **A walk needs word-level identity.** Slot-specific ids are right for storage, since
   they bind a word to its role. But crossing between sentence forms (a statement and a
   question) needs the word in any role. The store keeps both indexes.
4. **Rare-word bridges happen often in training** (5,000–11,000 walks per run, mostly
   early, when every word is rare). They do no visible harm, but they cost time: training
   went from 440 to about 580 µs per word.

## Next
- **Walks only on demand:** when the first recall leaves the answer slot empty, as
  the text baseline's candidate filter does. That should remove most training walks.
- **Store each event once** (no copied context), with the walk doing the context match
  (57).
- **A task where the hippocampal route is the only route:** lesion the semantic store and
  see what the walk alone carries.
