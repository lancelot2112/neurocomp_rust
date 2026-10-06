# 61 · Inferred replay: teaching the cortex what the walk composes

**Question.** The walk answers new names at 92% while the hippocampus is there, and at
chance without it ([60](60-walk-alone.md)): nothing it composes reaches the cortex. In
sleep, can the hippocampus compose the events the cortex never read ("lucy went to the
hallway", in winter) and teach them to the higher area, so the cortex answers with the
hippocampus lesioned?

**Code.**
- [`src/program/engram.rs`](../../src/program/engram.rs):
  - `EngramStore::infer_from`;
  - `EpisodicCircuit::infer` and `last_row`;
  - rows keep their content in reading order;
  - new facts are queued.
- Harness: `INFER_REPLAY=reps`, `INFER_ROWS`, `INFER_KEEP_ONLY`, `INFER_LEARN`,
  `INFER_CTX_STATE`, `INFERDIAG`.
- Unit test `a_new_fact_yields_inferred_events`.
- Regression entry `infer-replay`.

## How an inferred event is made (in the engram store)
1. **A new fact** is a stored row holding a word no other row holds: N = "lucy" in "lucy
   is a jones".
2. **Its schema:** the rows that share all but two of its words, in any slot ("X is a Y
   ."). The *frame* words ("is", "a", ".") are in nearly all of them. The fact's most
   variable word is the *filler* that links it, the partner F = "jones".
   - Choosing the rarest other word instead picked "is": it occurs only in statements,
     "jones" also in questions.
   - Comparing by slot id instead of word failed too: the learned slots vary.
3. **Sources:** rows of other stories holding F, outside the fact's schema ("john jones
   went to the hallway").
4. **Composition:** the source in reading order, with the word in N's slot replaced by N
   and F dropped ("lucy went to the hallway"). A second variant keeps F.

## How it is taught (in the harness, at each sleep including the last one before test)
1. **The state:** each row remembers the higher area's slow state when it was read, the
   cortical pattern the index points back to (`last_row` → state).
2. **The schema across instances:**
   - For each distinct inferred event, the state is what most of its sources share. That
     reduces it to the season, plus a few frequent filler words.
   - For each new word, targets present in nearly all of its events are frame ("went",
     "to", "the") and are not taught. Only the varying ones (the places) are.
3. **The update is grow-only:** a kernel for (sentence so far + shared state → the
   place), whose growth is masked to N, the word before the target, and the shared
   state. Nothing else is blamed.

## Results (family stated once; seeds 0 / 1 / 2; hippocampus lesioned at test unless stated)

| Setting | Teaching | Held out | Trained | Top-down held the answer |
|---|---|---|---|---|
| Walk only (no semantic store, no rollout) | none | 23 / 17 / 25% | 64.9% | 37–42% |
| same | `area.learn`, sources' context | 0 / 7 / 13% | 64.1% | – |
| same | grow-only, unmasked | 22 / 25 / 20% | 65.8% | 36–40% |
| same | grow-only, source's cortical state | 15 / 23 / 16% | 64.3% | 32–40% |
| **same** | **schema: shared state, varying targets, masked growth** | **32 / 33 / 27%** | 64.8% | **46–47%** |
| same, hippocampus intact | schema | 91 / 95 / 91% | 89.9% | |
| Semantic store + rollout (60) | none | 60 / 64 / 56% | 60.5% | 59–62% |
| same | `area.learn`, sources' context | 50 / 13 / 40% | 58.8% | 33–53% |
| **same** | **schema** | **64 / 63 / 61%** | 60.5% | **61–62%** |

## Findings
1. **The composed relation now reaches the cortex, partly.**
   - Walk only: with the hippocampus lesioned, new names go from chance (22% mean) to 30%.
   - With the semantic store and rollout: from 60% to 63%, the best consolidated figure on
     this task.
   - Trained names are untouched.
2. **What was wrong, in order, and each fix generic:**
   1. **The fact's link word:** the filler of its schema, not the rarest word.
   2. **The final sleep before test was skipped:** the stated facts come late in training.
   3. **`area.learn` blamed the kernels that fire,** and masked growth by the live window
      (whatever story was last read). Grow-only fixed it.
   4. **One-shot kernels keyed on the wrong bits:** "went to the" + winter, which fires
      for every name, or "lucy" alone, which predicts "went" at the answer. Masking growth
      to the new word + the previous word + the state fixed that.
   5. **Each source story's state is noisy:** its own names and filler. The state shared
      across sources is the season. This is schema extraction from instances: what is
      constant is frame, what varies with the answer is context.
3. **Why it stops at about 30%:**
   - The season is often not in the higher area's state at test: the surprise window
     loses it after a few sentences, and autumn rarely survives.
   - A single grown kernel per pair competes with the many trained kernels for "… went to
     the".
   - The waking learner corrects its kernels over hundreds of examples; the inferred ones
     are never corrected.
4. **The hippocampus still does it better at test** (92%). Consolidation transfers part of
   it. In the brain it is also partial and slow, over many nights.

## Next
- **Repeat inferred replay every sleep and track which events the cortex already
  answers,** using the cortex's error to set replay priority (the store supports it). Let
  correct kernels gain trust by being right in replay, as waking kernels gain it by being
  right on the page.
- **Keep the season in the state:** the slow state should hold the story's frame (the
  season), not only recent surprises. That is the "context" the hippocampus already
  keeps, made cortical.
- **Teach the column too,** not only the higher area: the inferred sentence read in order.
