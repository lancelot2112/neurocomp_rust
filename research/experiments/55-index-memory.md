# 55 · An index memory: one row per episode, winner-take-all recall

**Question.** Hebbian populations store every episode in shared weights, and their
crosstalk grows with every one ([52](52-associate-and-hippocampus-genome.md)–[54](54-phase-hippocampus-backed-out.md)).

A modern Hopfield network instead gives each memory its own hidden unit that holds the
pattern explicitly. Recall is a softmax over those units; at zero temperature it is
winner-take-all. The number of patterns that stay separable then grows exponentially
with the width of the code, at a cost of one row per memory. This is also hippocampal
indexing theory: a sparse index unit per episode that points back at the full pattern.

Does this structure, made event-based and given forgetting, beat the counts circuit on
the experiment 49 task?

**Code.**
- [`src/program/index_memory.rs`](../../src/program/index_memory.rs): `IndexMemory`,
  which implements `EpisodicCircuit`.
- Harness: `HIPPO=index`, with `INDEX_PERIOD`, `INDEX_PLAIN`, `INDEX_CAP` and
  `INDEX_MIN`; consolidation marking in tagged replay (`NO_CONSOLIDATE_MARK`).
- [`examples/superposition.rs`](../../examples/superposition.rs): the capacity
  comparison.
- Regression entry `index-hippocampus`.

## The design
- **Rows:** one per stored event. A row holds:
  - its keys, the active input indices (content and context);
  - its output pattern (EC V);
  - a strength and a last-touch time;
  - a pointer to the next row (the successor).
- **Recall is event-based:**
  - An inverted index maps each input index to the rows that contain it.
  - Each cue index adds its weight, 1/n from the reciprocal table, to the counters of its
    rows. Indices in more than 4,096 rows are skipped.
  - The best row with at least 16 shared indices wins; ties go to the most recent row.
- **Sequences:** `recall_sequence` follows the successor pointers. The harness ends a
  sequence at each story boundary.
- **Forgetting is lazy:**
  - effective strength = strength − (now − last touch) / `period`;
  - on write the strength is 128 + 127 × novelty, and each recall adds 16;
  - a periodic sweep evicts rows at zero;
  - consolidated rows fade four times faster.
- **Replay:** rows are sampled by strength × a weight (the harness passes 1 for now),
  with their successor chains.
- **Consolidation:** in tagged replay, when the semantic store already gives back at
  least 80% of the event's content from its cue, the row is marked consolidated.

## Results

**Capacity** (`examples/superposition.rs`; 2,048 inputs, 2,048 cells, k = 32; right = at
least 80% of the target recovered):

| codes | events | counts, 1/n + centered | phase (random input phases) | **index memory** |
|---|---|---|---|---|
| random | 8,000 | 94% | 100% | **100%** |
| random | 16,000 | 0% | 44% | **100%** |
| language-like | 2,000 | 57% | 79% | **99%** |
| language-like | 8,000 | 0% | 13% | **97%** |
| language-like | 16,000 | 0% | 0% | **96%** |

The index memory's misses on language-like events are events whose rare word also
occurs in other events with the same common words. Those events cannot be told apart,
so 96–99% is the ceiling.

**The experiment 49 task** (seeds 0 / 1 / 2; settings of `hippocampus-teaches-cortex`):

| Hippocampus | At test | Held-out right | Trained names right | Family right | Train µs/word |
|---|---|---|---|---|---|
| Counts circuit (49) | lesioned | 64 / 59 / 53% | 67% | 99 / 67 / 87% | 2,190–2,390 |
| Index, no forgetting | lesioned | 65 / 63 / 59% | 62% | 100 / 99 / 87% | 1,530–1,610 |
| Index, plain overlap (no 1/n) | lesioned | 61 / 61 / 58% | 62% | 99 / 100 / 88% | 1,600–1,650 |
| **Index, forgetting (period 64)** | lesioned | **62 / 64 / 65%** | 64% | **99 / 100 / 97%** | 1,640–1,750 |
| Counts circuit (49) | intact | 65 / 59 / 59% | 67% | 100 / 67 / 100% | 2,230–2,330 |
| **Index, no forgetting** | intact | **65 / 67 / 60%** | **95%** | **100 / 100 / 99%** | 1,610–1,730 |

## Findings
1. **Explicit rows remove the capacity limit.**
   - In isolation the index memory recalls 100% of 16,000 random events, against 0% for
     the best Hebbian population, and sits at the ceiling on language-like ones.
   - In the system it is better on average and faster: held out 62–65% with the
     hippocampus lesioned (counts circuit 53–64%), 1.6 vs 2.3 ms per word.
2. **Seed 1's family failure is gone.** Seed 1 had answered the stated family wrong since
   experiment 49 (67%), and is right now (99–100%). It was crosstalk between overlapping
   events in shared weights, the failure explicit rows cannot have.
3. **With the hippocampus intact, trained names go from 67% to 95%.** Every stored fact is
   recalled exactly, and the cortex uses it.
4. **Forgetting helps the cortex learn on its own.** Fading unused rows raised held-out
   accuracy on seed 2 from 59% to 65%, and family from 87% to 97%, with the hippocampus
   lesioned. Replay then samples rehearsed and new events, not the long tail of old ones.
5. **Trained names, with the hippocampus lesioned, are lower: 62–64% vs 67%.** The
   cortex learns trained names less well from the index memory's replay than from the
   counts circuit's attractors, which favour common events. Priority replay weighted by
   the cortex's error is not wired yet.
6. **The memory is no longer the cost.** In the word loop, recall takes 7% and storage
   20%; the cortex's prediction takes 69%.
7. **Novelty is low (mean 0.01).** Each event's keys include the whole story so far, and
   the stories repeat, so almost every event matches an earlier row closely. The keys
   carry far more than the event needs, the motivation for the next design.

## Next: indices and phases (the next experiment)
- **Rows of about 12 binding ids**, not about 1,000 input bits.
- **A grid-phase code for "where"** (story and position), replacing the context field.
- **A where-index** (a hash of phase tuples), with a bonus for the current place.
- **Theta-cycle deduplication:** a row that fires is strengthened, not stored again.
- **A ring buffer whose write head evicts weak rows.**
- **Rarest-first recall with early exit.**
- **Replay priority = strength × the cortex's error.**

## Biology
- **Hippocampal indexing theory:** the hippocampus stores an index to the neocortical
  pattern of an episode, not the pattern itself (Teyler & DiScenna 1986; Teyler &
  Rudy 2007).
- **Modern Hopfield networks:** exponential capacity from one hidden unit per memory,
  with attention as the update rule (Krotov & Hopfield 2016; Ramsauer et al. 2020).
- **Forgetting by disuse, and rehearsal by reactivation:** reconsolidation and
  retrieval-induced strengthening (Nader et al. 2000; Roediger & Karpicke 2006).
