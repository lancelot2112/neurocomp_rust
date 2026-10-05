# 24 · A cortical hierarchy: a higher area that predicts the column's errors

**Question.** Everything so far ran through one cortical column, which sees three frames:
the current word, the memory or relay frames, and the previous word. Can a second, higher
area, working on a slower timescale and feeding back down, supply context that the column
cannot see, as cortical hierarchies do?

**Code.**
- `HigherArea` in [`src/program/cortex.rs`](../../src/program/cortex.rs).
- In [`examples/episodic.rs`](../../examples/episodic.rs): `TASK=habit` and `HIER=1`, plus
  `HIER_SPAN`, `HIER_RESIDUAL`, `HIER_SEPARATE`, `HIER_FOCUS`, `HIER_GENERALIZE`,
  `HIER_GROW_TRUST` and `HIERDIAG`.

Run: `COPY_GROW=1 GROW_TRUST=0.5 STICKY=255 GENERALIZE=0.5 GENERALIZE_AFTER=1
TRUST_AT_TEST=0.5 GROW_GATE=3 SLEEP_EVERY=500 SURPRISE_GATE=1 MEMO=1 CANON=1 TASK=habit
POLICIES=nomemory HIER=1 cargo run --release --example episodic`.

## Architecture
- **Feedforward (L5 → higher-order thalamus → L4 above).** The higher area receives two
  frames:
  - the bag of the current sentence's words;
  - a **slow state**: the words the column found surprising over the last 4 sentences,
    i.e. the column's L5 surprise integrated over time. Predictable filler never enters
    it, so a fact stated sentences ago stays in view.
- **L2/3 of the higher area.** A predictive `CorticalColumn` of its own, with canonical
  kernels and surprise-gated learning.
- **Feedback (L6 / apical → the column).** Its prediction is one more frame of the column's
  L4. The frame order is current word, memory / relays, top-down, previous word. The
  column learns when to copy the top-down frame, as it learns to copy a recalled memory.
- **Learning: predictive coding.** The higher area is trained only on words the column
  **failed** to predict, i.e. on the column's residual.

## Task: habit
> in the morning . the dog ran away . it rained . john went to the **hallway**

- **The rule:** each name has a fixed habitual place in the morning and another at night.
  This is slow statistics, learned over training, not a one-shot fact.
- **Why the column alone can't answer:** at the answer it sees "the", "to" and its memory
  frame, not the name or the time of day.
- **Why episodic memory can't either:** recall brings back the name's last trip, whose
  time of day is random.

## Results (held-out, seeds 0 / 1 / 2)

| Habit | Held-out | Top-down frame held the answer |
|---|---|---|
| column alone | 0 / 0 / 0% | – |
| + episodic memory | 52 / 52 / 49% | – |
| + higher area, trained on every word | 76 / 71 / 65% | 65–75% |
| + higher area, trained on the column's errors, top-down frame before memory | 90 / 87 / 87% | 87–91% |
| **+ higher area, errors only, top-down after memory (final)** | **81 / 82 / 80%** | 77–81% |
| final + episodic memory | 70 / 79 / 80% | 75–78% |
| final + sleep in the higher area (`HIER_SLEEP=1`) | 77 / 70 / 77% | 67–78% |

The other tasks with the higher area on (seed 0; without it in brackets):

| Task | Top-down after memory | Top-down before memory |
|---|---|---|
| Topic | **100%** (100%) | 37% |
| Varied, both settings | 100% (100%) | 100% |
| Give | 99.6% (99.8%) | 99.8% |
| Two-hop | 94.4% (93.0%) | 90.4% |

## Findings
1. **A higher area supplies what the column cannot see.** The column alone scores 0% on
   habit and memory 49–52%. With a higher area holding the sentence and a slow state, the
   column copies its top-down prediction to 80–90%.
2. **The higher area should learn the column's errors, not every word.** Trained on every
   word, it predicts from an unordered bag of the sentence. The bag at "john went to the"
   is a subset of the bag one word later, so it often predicts "the", "." or "to" at the
   answer (65–76%). Trained only on the words the column failed to predict, it learns to
   supply exactly what is missing (87–90%). This is predictive coding (Rao & Ballard
   1999): higher levels explain the residual of lower ones.
3. **Frame order is arbitration.** Growth deepens one frame at a time and stops at an
   empty frame, and new kernels key on the earlier frames first.
   - **Top-down before memory:** the column builds on top-down context first. That's best
     for habit (90%), but it breaks topic (37%), where top-down is right only 42% of the
     time and memory always.
   - **After memory:** the column relies on its own frames and uses top-down only where
     they do not suffice. Topic is back to 100%, two-hop is slightly better, and habit is
     81%.
   - The no-memory policy's always-empty placeholder frame first hid the top-down frame
     completely (0%), so with `HIER` that policy uses the top-down frame in its place.
4. **Memory and top-down compete.** With both, the column sees two candidate places (1.5
   per answer on average): the recalled trip (random time of day) and the habit. It
   sometimes copies the wrong one (70–80%). Arbitrating between sources needs more than
   frame order; the reliability of each source per context is the natural signal.
5. **What did not help:**
   - **One slow-state frame per sentence** (`HIER_SEPARATE`): 13–27%. The fact's sentence
     sits at a varying lag, and deep kernels must reach it.
   - **Attention to one remembered word at growth** (`HIER_FOCUS`): 14–61%, unstable.
   - **Near-miss generalisation in the higher area** (`HIER_GENERALIZE`): 0–11%. It pruned
     the time-of-day bits along with the noise.
   - **A growth-trust floor** (`HIER_GROW_TRUST`): no gain.
   - **The uncertainty gate on the higher area's growth:** wrong by design. It stops
     growth when no input frame carries the target, and the higher area's target is never
     in its input.
6. **Cost.** The higher area runs at every word and grows 9,000–17,000 kernels on tasks
   whose residual it cannot explain (memory-dependent answers), making them 3–5× slower.
   Sleep halves that at a cost of a few points on habit. The natural fix is a truly slower
   timescale: run the higher area only when the column is surprised, or once per
   sentence, not at every word.

## Biology
- **Hierarchy and predictive coding:** each level predicts the activity of the level below
  and is driven by its prediction errors (Rao & Ballard 1999; Friston 2005). Here the
  higher area is driven by the column's surprises and learns only from them.
- **Feedforward through higher-order thalamus:** layer 5 projects to higher-order thalamic
  nuclei (pulvinar, mediodorsal), which drive the next cortical area (Sherman & Guillery).
  The slow state is the column's L5 surprise carried up.
- **Feedback through layer 6 and apical dendrites:** top-down predictions arrive in layer 1
  on apical dendrites and through L6, modulating what the lower area does. Here that is a
  frame the column learns to use or ignore.
- **Timescales:** higher areas integrate over longer windows (Hasson et al. 2008; Murray et
  al. 2014). The 4-sentence slow state is the crude version.
