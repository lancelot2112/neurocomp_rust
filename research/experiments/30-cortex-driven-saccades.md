# 30 · Cortex-driven saccades, and a first transfer test

**Question.** In [29](29-saccades.md) the basal ganglia chose saccades in a context made
from the identities of the previous and current word, built by the example code. In the
brain, cortex proposes saccades and the basal ganglia gate them: frontal eye fields and
parietal cortex send their state to the striatum and the superior colliculus (Hikosaka
et al. 2000). If the context is the column's own state, does the policy carry over to a
wording it has never seen? That is the first test of how literal the network's knowledge
is, on the way to schemas.

**Code.**
- `KernelClass::peek_union` in [`src/kernel/class.rs`](../../src/kernel/class.rs).
- In [`examples/episodic.rs`](../../examples/episodic.rs): `SACCADE_CTX=cortex`,
  `SEASON_NEW_WORDING`, and the `WORDING` report.

Run: `… TASK=season POLICIES=nomemory HIER=1 HIER_LEVELS=1 MIX=1 SACCADE=index
SACCADE_CTX=cortex SEASON_NEW_WORDING=1 cargo run --release --example episodic` (other
settings as in 29).

## Cortex-driven context
- **The column's state.** Before choosing a saccade, the column peeks, without top-down
  input and without changing any state. It takes the union of what all its matching
  kernels predict next: its possible continuations, superposed.
  - Confident: that is one word's code.
  - Uncertain: it superposes a class. At "john went to the ?" it holds every place.
- **Plus a code for its confidence bucket.** Together these are the striatum's input,
  in place of the word-pair code.
- **Binding.** Each candidate (read on, or a landmark word of the page index) is the
  state rotated by the candidate's own amount. Similar states share bits, and so share
  learned values. The word-pair code gave no sharing at all.

## Transfer test
`SEASON_NEW_WORDING=1`: held-out test questions read "X walked into the". Neither "walked"
nor "into" appears in training. Two measures:
- **The saccade policy:** before a new-wording answer, did the reader look back at the
  season sentence?
- **The answer:** was it right?

## Results (season task, one higher area, page index; seeds 0 / 1 / 2)

| Context | Trained wording: right | Looked back before trained / new-wording answers | New wording: right | Regressions per story |
|---|---|---|---|---|
| Word pair (29) | 88 / 86 / 97% | 76–100% / 1–45% | 0 / 0 / 0% | 27–35 |
| **Cortex** | 95 / 86 / 99% | 100% / 68 / 0 / 51% | 0 / 0 / 0% | 18 / 1.7 / 39 |
| Cortex, cost 0.5 | 69 / 57 / 97% | 48–100% / 0 / 23 / 0% | 0 / 0 / 0% | 1.2 / 5.2 / 0.6 |
| Oracle look-back | 94 / 65 / 96% | 100% / 100% | **0 / 0 / 0%** | 1 |

## Findings
1. **The cortex can drive the saccades.** With the column's state as the context,
   trained-wording accuracy matches the word-pair context (86–99%). The reader looked
   back at the season before every trained-wording answer, and one seed found the
   efficient policy (1.7 regressions per story, 5.9% re-reads) at cost 0.1.
2. **The policy does not reliably transfer.** Before new-wording answers the reader
   looked back 0–68% of the time. The seeds that look back often everywhere also do so
   here, by habit rather than transfer. The efficient seeds, which look back only at
   the question, did not (0%).
   - The reason: at "into the" the column's continuations come only from its
     single-word kernels ("the" → cat, dog, places...). The two-word "to the" kernels,
     which superpose the places, do not match. The state overlaps the trained one only
     partly.
3. **The answer does not transfer at all, even with a perfect look-back (0%).** The
   season is in view, but the higher area's kernels are keyed on the trained question's
   sentence bag ("went", "to"). The column has no kernel that predicts a place after
   "into the". So the literal part is the knowledge itself, not only the saccade policy.
4. **This is the baseline for schemas.** Everything the network has learned is keyed on
   word identities. A schema would let a new wording, a new name or a new setting word
   slot into known structure: "a person's place depends on the setting; the question
   asks for that place". It needs role-level codes in the column and the areas, not
   only in the saccade context ([roadmap](../roadmap.md) stage 4).

## Biology
- **Cortex proposes, basal ganglia gate:** frontal eye field and LIP priority maps project
  to the caudate and the superior colliculus; the basal ganglia disinhibit the chosen
  saccade (Hikosaka et al. 2000).
- **Schemas need abstraction:** prefrontal and hippocampal representations that
  generalise across items (Tse et al. 2007; Whittington et al. 2020), which this test
  shows the network does not yet have.

## Next
- **Role-level codes:** classes induced from distribution (as in [05](05-syntax.md)) as
  part of every word's code. Then "walked into" shares bits with "went to", and places
  with places. Rerun this transfer test as the measure.
- **Cerebellar fine-tuning** of landing (whole sentence or just the landmark word) and
  timing (look back a word earlier), learned from the error after each regression.
