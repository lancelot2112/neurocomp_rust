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

## A thalamic gate on the top-down frame
In the brain, higher-order signals reach an area through the thalamus (pulvinar), and the
reticular nucleus and L6 feedback set how much gets through, so an unreliable signal can
be turned down. Here the top-down frame went straight into the column, so three gates
were tried (`HIER_GATE`, `HIER_GATE_SIGNAL`, `HIER_GATE_CONF`), each with the frame late
(after memory) and early (`HIER_EARLY`, right after the current word):
- **Use** (the L6 gate of [20](20-l6-corticothalamic-gating.md) on one channel, keyed on
  the current word). It is strengthened when the column read the top-down frame and came
  true; 1,000-story warm-up, weakening × 0.25.
- **Reliability** (`HIER_GATE_SIGNAL=area`). Same gate, strengthened when the *higher
  area's own* prediction came true (its L5 outcome).
- **Confidence** (`HIER_GATE_CONF=0.7`). A prediction passes only if the higher area's
  winning kernel is at least 0.7 reliable.

| Held-out | Late, no gate | Use gate | Reliability gate | Confidence gate | Early, no gate | Early + use | Early + confidence |
|---|---|---|---|---|---|---|---|
| Habit, no memory (seeds 0 / 1 / 2) | **81 / 82 / 80%** | 79 / 73 / 17% | 15 / 18 / 18% | 65 / 71 / 84% | 81 / 82 / 80% | 79 / 73 / 17% | 65 / 71 / 84% |
| Habit + memory | 70 / 79 / 80% | 52 / 51 / 79% | 52 / 48 / 46% | 72 / 80 / 77% | 67 / 67 / 83% | 50 / 51 / 85% | 70 / 88 / 78% |
| Topic (seed 0) | **100%** | 97% | 100% | 100% | 37% | 59% | 51% |
| Two-hop (seed 0) | 94.4% | 82.6% | 91.2% | 94.4% | 90.4% | 92.4% | 90.0% |
| Varied / give (seed 0) | 100 / 99.6% | 100 / 100% | 100 / 100% | 100 / 99.6% | 100 / 99.8% | 100 / 100% | 100 / 99.8% |

1. **A gate keyed on the current word is all-or-nothing at the answer:** it passed top-down
   at 100% or 0% of answers in every run. "the" occurs in "the cat", "in the morning" and
   "went to the", so a per-word gain cannot single out the answer position.
2. **"Used" is the wrong signal.** With memory on, the column often answers from memory,
   so the top-down frame looks unused and the gate shuts it even where it would have been
   right (habit + memory falls to the memory-only 50%).
3. **"Right" per word is also too coarse.** The higher area is trained only where the
   column fails, so at most occurrences of "the" its prediction is wrong. The reliability
   gate closed the channel on every task, at 0% passed.
4. **Per-prediction confidence is selective** (passes at 64–80% of habit answers, 4–11% of
   topic answers, ~2% on varied). But it blocks some correct, less confident predictions,
   and does not beat late placement without a gate.
5. **The arbitration problem is in the column, not the thalamus.**
   - **Growth stops at an empty frame.** With top-down early and gated off, frame 1 is
     empty and the memory-copy kernels behind it can never grow, so topic stays broken
     (51–59%).
   - **The winner is ranked by depth before reliability.** A kernel reading a later frame
     beats a more reliable one reading an earlier frame, so frame order decides the
     arbitration.

   A thalamic gate becomes useful once L2/3 can grow past empty frames and choose between
   sources by reliability. That is the next step. Late placement without a gate stays
   the default.

## Reliability-first ranking in L2/3
Two options in `KernelClass` (`src/kernel/class.rs`), both off by default:
- **`set_reliability_first`** (`RANK=reliability`): the matching kernel with the highest hit
  rate wins, and depth only breaks ties. Every matching kernel is scored at feedback,
  winner or not, so a deep kernel can earn the record to win later. (Under the surprise
  gate, a correct guess credits only the winner. The other kernels are scored only when
  it misses.)
- **`set_skip_empty`** (`SKIP_EMPTY=1`): deepening growth skips frames that are empty now,
  so a kernel can reach the memory frame behind a gated-off top-down frame.

Held-out, same settings as above (habit seeds 0 / 1 / 2; the rest seed 0):

| | Depth-first, late (default) | Reliability-first, late | Reliability-first, early | + skip empty, early | + skip empty + confidence gate, early |
|---|---|---|---|---|---|
| Habit, no memory | **81 / 82 / 80%** | 75 / 73 / 82% | 75 / 73 / 82% | 73 / 75 / 82% | 72 / 79 / 72% (passed 72–82%) |
| Habit + memory | 70 / 79 / 80% | 80 / 73 / 83% | **81 / 80 / 86%** | 83 / 67 / 84% | 77 / 67 / 80% (passed 47–73%) |
| Topic | **100%** | 90.4% | 46% | 49% | 94% (passed 13%) |
| Two-hop | **94.4%** | 90.4% | 93.8% | 93.8% | 94.0% |
| Give | 99.6% | 92.4% | 99.0% | 100% | **100%** |
| Varied, both | 100% | 100% | 100% | 100% | 100% |
| Column kernels (habit / give) | 416 / 4,202 | 412 / 4,331 | 412 / 4,494 | 497 / 4,381 | **126 / 3,301** |

1. **Reliability decides memory against top-down better than frame order does.** With
   both sources, habit + memory rises to 81 / 80 / 86% (early) from 70 / 79 / 80%, the
   best result for that combination so far. The column now picks the habit when the
   recalled trip has proved unreliable for this context.
2. **Reliability alone over-trusts shallow kernels.** A proven shallow kernel (hit rate
   0.8 over many uses) beats a fresh deep one (1/2 until it has a record), and under the
   surprise gate the deep one gets credit only when the shallow one misses. Specific
   contexts are learned more slowly, and topic (90%), two-hop (90%) and give (92%) lose a
   few points with top-down late.
3. **Skipping empty frames does not rescue early top-down by itself** (topic 49%). Without
   a gate the top-down frame is never empty. The cost is that every kernel deeper than it
   must also match its (varying) contents, so memory-copy kernels fragment by top-down
   word. Empty frames only matter once a gate empties them.
4. **All three together (reliability-first, skip empty, confidence gate) make early
   placement work.** Topic is back to 94% from 37–59%; two-hop, give and varied are at or
   above the default; the column needs 3–6× fewer kernels on habit and 20% fewer on give,
   because an empty top-down frame no longer splits kernels. Habit is 67–80%, a few
   points below late placement, because the gate also blocks some correct, less
   confident predictions.
5. **No variant wins everywhere yet.** Late depth-first stays the default. The open
   problem is how a fresh, specific kernel should compete with an established, general
   one. Candidates:
   - rank by reliability only among kernels with enough evidence (a minimum count, as
     the trust floor does), and by depth otherwise;
   - score every matching kernel on expected steps too, so deep kernels earn their record
     as fast as shallow ones (at some cost in speed).

### Evidence-gated ranking (`RANK_MIN`): worse
`set_reliability_first(Some(min))`. A kernel with fewer than `min` scored predictions
(hits + misses) has no record yet and ranks as fully reliable (rate 1/1). Among new
kernels the deepest wins, as by default, and a fresh specific kernel gets to fire and
earn its record before it competes on its rate.

| | Reliability-first (min 0), late | min 8, late | min 16, late | min 8, early | min 8 + skip empty + confidence gate, early |
|---|---|---|---|---|---|
| Habit, no memory | 75 / 73 / 82% | 76 / 76 / 81% | 74 / 81 / 76% | 76 / 76 / 81% | 73 / 77 / 71% |
| Habit + memory | 80 / 73 / 83% | 73 / 67 / 79% | 68 / 78 / 46% | **84 / 86 / 81%** | 69 / 77 / 78% |
| Topic | 90.4% | 76.6% | 60.8% | 15.6% | 55.2% |
| Two-hop | 90.4% | 65.0% | 68.4% | 87.2% | 94.2% |
| Give | 92.4% | 61.4% | 93.2% | 89.8% | 93.8% |
| Varied (two settings) | 100 / 100% | 98.8 / 92.2% | 100 / 100% | 69 / 73% | 100 / 100% |
| Column kernels (habit + memory, seed 0) | 703 | 1,236 | 1,103 | 1,067 | 597 |

Habit + memory with top-down early reaches 84 / 86 / 81%, the best so far. Everything
else gets worse, often by 20–40 points.
1. **Optimism floods the ranking with new kernels.** Every surprise grows kernels, and
   each new one outranks every proven kernel it matches alongside until it has `min`
   records. Kernels that match rarely never reach `min` at all (and halving can push a
   record back below it), so they stay optimistic for good. The memory tasks, where a
   proven copy kernel must win at the answer, lose most: topic 15–77%, give 61–93%.
2. **The churn feeds itself.** Each wrongly preferred new kernel misses, which grows more
   new kernels. The column ends with about 1.5–2× the kernels of plain reliability-first
   (habit + memory 1,236 against 703).
3. **So neither extreme works.** No optimism (min 0) under-trusts specific kernels, and
   optimism until `min` over-trusts them. How a new kernel should compete is a question
   of how fast it is *scored*, not how it is ranked meanwhile. Under the surprise gate,
   a kernel that doesn't win is scored only on the winner's misses. The next test is
   to score every matching kernel on expected steps too.

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
