# 23 · Compaction: an event-based fast path, uncertainty-gated growth and sleep

**Question.** The column learns in one pass, but it was 100–500× slower per word than a
small transformer, and on the elimination task 17,273 kernels matched a single context.
The aim was three things:
- an event-based path that touches only the active bits (about 32 of 8,192 per word),
  with learning still inline at every word;
- a prior that stays compact, using mechanisms the brain uses for the same job;
- accuracy kept on every task.

**Code.**
- In [`src/kernel/simple.rs`](../../src/kernel/simple.rs): `SimpleKernel::input_bits`,
  `input_set` and `output_set`.
- In [`src/kernel/class.rs`](../../src/kernel/class.rs): `set_growth_gate`, `sleep`,
  `set_replay` and `live`.
- In [`examples/episodic.rs`](../../examples/episodic.rs): `GROW_GATE`, `UNC_MIN`,
  `SLEEP_EVERY`, `REPLAY_LEN`, and the `COST` and `PROF` report lines.

Run: add `GROW_GATE=3 SLEEP_EVERY=500` to any experiment's settings.

## 1. Measure first
Per-stage timers on the word loop (`PROF`) put the time almost entirely in L2/3: kernel
matching took 76–78% and learning 19–21% on varied stories. Recall, relays, gates and
evaluation together took 2% or less. Within L2/3, the cost came from dense operations on
sparse objects:
- **Near-miss check:** for every kernel the input touched without fully matching, it
  popcounted the kernel's whole input mask, 3 × 8,192 bits (384 words), just to learn its
  size.
- **Learning:** checking a prediction ANDed and popcounted the kernel's full 8,192-bit
  output. The copy-credit tags and pruning scanned each hit or near-miss kernel's dense
  input mask.
- **Fast inhibition:** tags were a hash map that was walked every step to count them
  down.

## 2. Event-based fast path (results unchanged)
- **Cached sizes and positions:** each kernel caches its input-mask size, its input bits
  as a position list and its output as a position list (~32 bits). They are resynced
  wherever pruning or the Hebbian moves change a mask.
- **Expiry array:** fast-inhibition tags became one step number per kernel.

Kernel counts and accuracy were identical. Per word, before and after:

| | Training | Answering |
|---|---|---|
| Varied stories | 2,421–2,998 → 274–378 µs (8–9×) | 3,489–4,134 → 183–208 µs (19–20×) |
| Elimination | 2,664 → 399 µs (6.7×) | 3,570 → 467 µs (7.6×) |

## 3. Compacting the prior
What remained was structural: a context fanned out to thousands of redundant kernels.
Two brain mechanisms handle exactly this.

**Awake: expected uncertainty gates growth (acetylcholine-like; Yu & Dayan 2005).**
- A miss normally grows a new kernel. It now grows nothing when two conditions hold:
  - the winning kernel's context is *known* to be unpredictable: at least `UNC_MIN` = 16
    observations, and a hit rate below 2^k/(2^k+1), tested in integers as
    `(misses+1) << k > hits+1` with k = 3;
  - no input frame carries the target, so a new kernel would have nothing to copy or key
    on.
- The second condition keeps memory-copy learning intact: in varied stories the answer
  *is* in the memory frame, so growth still happens there.
- In short, only *unexpected* surprise drives growth.

**Asleep: synaptic homeostasis and replay (Tononi & Cirelli 2003, 2014).** Every 500
training stories:
1. **Downscale:** every kernel's hit and miss counts shift right by one.
2. **Prune:** kernels that have only ever failed are removed.
3. **Merge by replay:** the column keeps its last 512 inputs as active-bit lists, and
   replays them. Kernels with the same output that fire on *exactly the same* replayed
   inputs carry the same information. The most reliable one is kept and the others are
   removed. Kernels that fire on nothing in the replay are kept, since they may be rare
   but useful, such as a copy kernel for a rare word.

Removed slots go to a free list that growth reuses, and the inverted index is rebuilt
once per sleep.

A first version merged only kernels whose inputs were a *subset* of a more general
kernel's. It found nothing (0 merges): each kernel samples random bits of each word, so
two duplicates read different bits of the same words. They are redundant in what they
respond to, not in what they store. Replay sees that; a static comparison does not.

## Results (seeds 0 / 1 / 2; "none" is seed 0 on the same build)

| Task | Live kernels: none → compacted | Held-out: none → compacted | Training µs/word | Answering µs/word |
|---|---|---|---|---|
| Elimination (fast inhibition) | 17,273 → **1,193** (14×) | 100% → **100%** | 423 → **39** | 482 → **19** |
| Varied, 1–2 facts | 16,237 → 2,968–3,061 (5.4×) | 100% → **100 / 100 / 100%** | 272 → ~95 | 161 → 67–75 |
| Varied, 1–3 facts | 19,292 → 3,301–3,425 (5.7×) | 100% → **100 / 100 / 100%** | 356 → ~96 | 180 → 70–79 |
| Topic (learned prefrontal gate) | 14,779 → 2,581–2,867 (5.4×) | 100% → **100 / 100 / 100%** | 323 → ~101 | 196 → 71–84 |
| Give (L6 gate) | 24,992 → 5,123–5,418 (4.7×) | 100% → **100 / 100 / 99.8%** | 658 → ~223 | 228 → ~110 |
| Two-hop (basal-ganglia selector) | 33,257 → 6,742–6,805 (4.9×) | 90.0% → 89.0 / 86.4 / 71.8% | 1,161 → ~238 | 466 → ~149 |

Kernel memory, compacted:
- **As stored now** (dense masks): 4.9–50 MB, against 60–230 MB before.
- **As sparse indices plus the inverted index:** 0.3–2.3 MB.

The two-hop selector's seeds without compaction are 90.0 / 86.8 / 77.2% (8-bit counters,
[probability in bits](../concepts/probability-in-bits.md)). Seed 2 is 5 points lower
compacted, and it is the seed that has always been weakest (70% in
[15](15-basal-ganglia-selector.md) with the same predictor settings).

## Findings
1. **Most of the prior was duplicates.** Replay merged 8,000–22,000 kernels per run. The
   column was storing the same context → continuation many times over, as random bit
   samples of the same words.
2. **Compaction did not cost function.** Every task keeps its accuracy (100% where it was
   100%), except possibly two-hop seed 2 (−5 points).
3. **The fast path is now competitive with a small transformer per word**, while learning
   inline in one pass. From this morning's code to now, answering on elimination went
   from about 3,600 µs per word to 19 µs (~190×), and on varied stories from ~3,500 µs to
   ~70 µs (~50×). See the [comparison](../concepts/brain-transformer-comparison.md).
4. **What's left:**
   - **Masks are still stored densely,** about 50× more memory than the sparse form needs.
   - **Learning now dominates** (60%): every matched kernel is still scored at every
     word, even when the prediction was right. Surprise-gated learning, where an expected
     word costs almost nothing, is the next step.

## 4. Surprise-gated learning
**Mechanism** (`SURPRISE_GATE=1`, `KernelClass::set_surprise_gate`):
- **Expected word** (the winner predicted it): only the winner is confirmed: its hit,
  its recency and its copy-credit tags. Fast-inhibition tags still run.
- **Surprise** (wrong winner, or none): the full update. Every matched kernel is scored,
  near misses are pruned and new kernels grown.

This is the predictive-coding reading of plasticity: an expected input carries no error.
Results are with compaction on (seed 0 timings, run together on a quiet machine):

| Task | Words expected in training | Held-out (compaction only → + surprise gate) | Training µs/word | Learning share of time |
|---|---|---|---|---|
| Elimination | 82% | 100% → 100 / 100 / 100% | 42 → 41–44 | 61 → 52% |
| Varied (1–2 and 1–3 facts) | 66–67% | 100% → 100% on every seed | 94–103 → 83–98 | 25 → 19–20% |
| Topic | 66–68% | 100% → 100 / 100 / 100% | 98 → 77–78 | 27 → 17% |
| Give | 61% | 100% → 100 / 100 / 100% | 220 → 198–209 | 38 → 33% |
| Two-hop | 61% | 89.0 / 86.4 / 71.8% → **88.6 / 89.8 / 81.6%** | 240 → 208–217 | 35 → 31% |

1. **Accuracy holds, and two-hop's weak seed improves** (71.8% to 81.6%). Not re-scoring
   the losing kernels on every expected word seems to help: alternatives that win in
   other contexts are no longer pushed down each time they lose here.
2. **The speed-up is modest (8–21%).** About a third of the words are surprises, and they
   carry most of learning's cost: growth, scoring every matched kernel and pruning near
   misses.
3. **On elimination the gate left more kernels** (2,909 against 1,193 live), so
   answering slowed from 20 to 43–53 µs per word. Not yet diagnosed.
4. **The bottleneck has moved to the hippocampus.** L4 assembly, which is mostly recall,
   is now 32–47% of the time on the memory tasks. Recall scans all 200 stored episodes as
   dense 8,192-bit vectors at every word. That is the next event-based target: an
   inverted index from active bits to the episodes that contain them, as the predictor
   already has.

## 5. Event-based hippocampal recall
Recall compared the cue with every stored episode at full width (200 × 128-word AND and
popcount), at every word and more than once for the big-loop hops and branches. The
episodic memory now keeps an inverted index from each bit to the episodes containing it,
maintained on store and on forgetting. A cue's active bits vote for the episodes that
contain them. Since cues are rare bits, that touches a few entries.
- **Exactly equivalent:** overlaps and the most-recent tie-break are the same, checked
  against a full scan over 200 stores with forgetting and exclusions. Every task's
  results were identical.
- **Speed:** with compaction and the surprise gate on, seed 0:

| Task | Training µs/word | Answering µs/word | L4 (mostly recall) share of time |
|---|---|---|---|
| Varied, 1–2 / 1–3 facts | 91 / 83 → **54 / 57** | 67 / 67 → **40 / 46** | 32–47% → 25–26% |
| Topic | 78 → **54** | 67 → **45** | → 13% |
| Give | 204 → **161** | 109 → **75** | → 23% |
| Two-hop | 214 → **147** | 142 → **107** | → 17% |

Over the whole day, answering on varied stories went from about 3,500 µs per word to
40 µs (~87×), and on elimination from 3,570 to 19 µs (~190×). Results are the same or
better.

## 6. Sparse kernel storage
Grown kernels were stored as dense masks over the whole input (3 × 8,192 bits) plus an
8,192-bit output, about 4 KB each, to hold roughly 40 input and 32 output connections.
They are now sorted lists of bit positions (`SimpleKernel::sparse`), about 300 bytes
each.
- **Everything on the predictive path reads the lists:** prediction writes, unpredicted
  counts, pruning, `peek`, credited inputs and L5's `winner_reads`.
- **Hand-built Hebbian kernels** in the other examples keep dense masks.
- **Growth** finds a frame's active bits word by word, not bit by bit.

Results are **identical** (same kernel counts, same accuracy on all five tasks, seed 0):

| Task | Kernel connections stored | Whole model (+ inverted index, + 200 KB episodes) |
|---|---|---|
| Elimination | 4.9 → **0.25 MB** | 0.65 MB |
| Varied | 11.2–12.1 → **0.64–0.71 MB** | 1.4–1.6 MB |
| Topic | 10.9 → **0.61 MB** | 1.4 MB |
| Give | 48.2 → **1.29 MB** | 2.7 MB |
| Two-hop | 33.3 → **1.51 MB** | 3.1 MB |

Time per word is unchanged within noise; the dense masks were no longer read on the hot
path after section 2. The model is now as small as the transformer baselines (0.1–3.2 MB).

## 7. Memoised interpretation (after Hashlife)
Hashlife (Gosper 1984) is fast because each distinct sub-pattern is stored once and its
future is memoised. The cheapest analog here (`MEMO=1`, `KernelClass::set_memo`):
- **Memo:** a hash of the L4 input maps to the winning kernel, stamped with a version of
  the prior.
- **Version bumps:** full-path learning (scoring, pruning, growth), sleep, trust-floor
  changes, and counter halving.
- **Why expected words don't invalidate:** with surprise gating, an expected word only
  strengthens the winner, which keeps it the winner.
- **On a hit:** matching is skipped. If the word then turns out to be a surprise,
  learning recounts the matched kernels from the same input.
- **Off when** fast inhibition or a top-down bias is active, since they change the winner
  step by step.
- **Exactness:** a test runs two predictors, with and without the memo, over 3,000 noisy
  steps and checks they make identical predictions and end with the same kernels.

Results are **identical** on every task (seed 0, with compaction and the surprise gate):

| Task | Served from memo | Training µs/word | Answering µs/word |
|---|---|---|---|
| Varied, 1–2 / 1–3 facts | 7 / 8% | 49 / 57 → 52 / 59 | 44 / 44 → **37 / 37** |
| Topic | 4% | 50 → 53 | 47 → 44 |
| Give | 21% | 141 → 143 | 77 → **49** |
| Two-hop | 12% | 145 → 151 | 104 → **76** |

1. **Whole inputs rarely repeat.** L4 is the current word plus the recalled memory plus
   the previous word, and that exact combination seldom recurs (4–21%).
2. **Training gets nothing.** Every surprise, about a third of the words, changes the
   prior and invalidates the whole memo, and hashing costs a little.
3. **Answering gains 6–37%,** because the prior is fixed at test, so repeats pay off.
4. **The truer Hashlife analog is per sub-pattern.** Hashlife memoises quadtree
   *sub*-nodes, which repeat far more than whole boards. Here the sub-patterns are frames:
   only ~40 distinct words fill the current-word frame. Memoising each frame's
   contribution to the match counts (frame content → kernels and counts), and combining
   the frames, would hit almost always. Invalidating per kernel rather than globally would
   keep entries valid through training. Not built yet.

## 8. Per-frame memo (Hashlife's sub-nodes): exact, but not a net win
Hashlife memoises *sub*-patterns, which repeat far more than whole boards. The analog
here (`FRAME_MEMO=1`, `KernelClass::set_frame_memo`):
- **Entries:** each frame of the L4 input (current word, relayed or recalled frames,
  previous word) has an entry keyed by (frame, content). It holds that content's
  contribution to every kernel's match count, and matching sums the cached lists.
- **Per-kernel invalidation:** every change to a kernel's connections (growth, pruning,
  removal in sleep) is logged. A stale entry is patched by recounting only the changed
  kernels against its stored bits, and rebuilt beyond 512 changes. This keeps entries
  usable through training, where the whole-input memo of section 7 was always
  invalidated.

**A bug it exposed: winner ties depended on visit order.** The first real-task run was
not equivalent. Kernel counts differed, and two-hop fell from 88.6% to 84.6%, although a
unit test had passed. When two kernels ranked exactly equal, the first one visited won,
and the memo visits kernels in a different order from the index fan-out. The toy test
had no exact ties; real tasks do. Ties now go to the older kernel (lower id). This changes
the baseline itself on seed 0:
- two-hop 88.6% → **91.6%**;
- topic: held-out still 100%, seen pairs 100% → 93.4%;
- elimination keeps 2,880 kernels instead of 1,193, so answering slows from 21 to 43 µs.
  The old order-dependent ties happened to favour merging there.

**With the fix, the frame memo is exact:** identical kernel counts and accuracy on all
five tasks, and a unit test checks identical outputs and kernel statistics through
growth, pruning and sleep. The speed (seed 0):

| Task | Lookups: hit / patched / rebuilt | Training µs/word | Answering µs/word |
|---|---|---|---|
| Elimination | 28 / 72 / 0% | 41 → **38** | 43 → **31** |
| Give | 8 / 88 / 5% | 156 → 188 | 53 → **42** |
| Topic | 46 / 49 / 5% | 55 → 75 | 45 → **38** |
| Two-hop | 40 / 51 / 9% | 149 → 198 | 66 → 76 |
| Varied, 1–2 / 1–3 facts | 35 / 54 / 11% | 53 / 59 → 84 / 85 | 46 / 44 → 54 / 58 |

1. **Training is 1.3–1.6× slower** on four tasks. Most lookups need patching, and a
   patch replays up to 512 logged changes, more work than the fan-out it replaces.
2. **Answering is mixed:** faster where frame contents recur (elimination, give, topic),
   slower where the recalled memory frame keeps changing (varied, two-hop).
3. **Why the saving is small:** the index fan-out was already event-based. The memo saves
   only the repeat increments when a kernel reads several bits of the same word, and pays
   for hashing, entry lookups and patching.
4. **What would make it pay:** eager updates (a kernel change updates only the entries
   containing its bits, through a bit → entries index) instead of replaying a log, and
   memoising only the frames that recur (current and previous word), not the memory frame.
   Left off by default.

## 9. Canonical kernels (Hashlife's hash-consed nodes)
Hashlife stores each distinct sub-pattern once. Here the duplicates came from sampling:
each growth drew a *random* subset of a word's bits, so the same context produced a new,
different-looking kernel every time. Sleep then had to find and merge thousands of them
by replay. With `CANON=1` (`KernelClass::set_canonical`):
- **Canonical sampling:** each frame's sample is its active bits with the smallest fixed
  hash rank (splitmix64), so the same context always yields the same kernel.
- **Hash-consing:** a table maps connection sets, (input positions, output positions), to
  their kernel. A growth that would duplicate a live kernel refreshes it instead. The
  table is kept current through recycling, pruning and sleep removals.
- **Test:** growing twice from the same context with different random states gives one
  kernel; a different output gives a second.

Results, added to compaction, the surprise gate and the memo:

| Task | Kernels, seed 0 (before → canonical) | Held-out, seeds 0 / 1 / 2 | Training / answering µs/word |
|---|---|---|---|
| Elimination | 2,880 → **24** (every seed) | 100 / 100 / 100% | 42 / 43 → **6–7 / 5–6** |
| Varied (1–2 and 1–3 facts) | 2,607–2,964 → 2,000–2,210 | 100% on every seed | 60 / 37–46 → 42–50 / 30–41 |
| Topic | 2,476 → 1,935–2,045 | 100 / 100 / 100% (seen pairs 93.4 → 100%) | 61 / 40 → 42–53 / 35–40 |
| Give | 5,251 → 2,848–2,959 | 99.8 / 99.8 / 100% | 157 / 46 → 106–114 / 44–53 |
| Two-hop | 6,450 → 5,187–5,397 | 93.0 / 83.6 / 80.4% | 149 / 77 → 104–122 / 51–62 |

Two-hop's earlier seeds were 88.6 / 89.8 / 81.6% (surprise gate, before the tie-break fix;
not a same-build comparison). The mean is about the same, within this task's usual spread.

1. **Elimination collapses to 24 kernels**, six places across a few contexts. With random
   sampling the evidence about "is it the → ?" was split across thousands of duplicates,
   so no kernel ever saw enough to be recognised as unpredictable, and growth went on.
   With canonical sampling it all lands on one kernel. The uncertainty gate recognises the
   context within a few observations and suppresses growth (10,665 times), and answering
   takes 5–6 µs per word.
2. **Fewer kernels and faster training everywhere,** with accuracy unchanged (topic's
   seen pairs even recover to 100%). 1,700–6,400 growths per run found their kernel
   already present.
3. **Sleep is still needed.** Without it, kernel counts grow to 9,000–25,000 (seed 0) and
   topic falls to 70%. Hash-consing stops *identical* duplicates; replay merging still
   removes kernels that differ in their sample but respond to the same inputs (e.g.
   kernels grown from different memory contents).
4. **The feared cost did not show up.** Canonical samples always use the same bits of a
   word, so a collision with another word's code would affect every use. No accuracy loss
   on these tasks.

## Biology
- **Expected versus unexpected uncertainty** (Yu & Dayan 2005): acetylcholine is thought
  to signal known, irreducible noise, and noradrenaline a change in the world. Only the
  second should drive structural learning.
- **Synaptic homeostasis** (Tononi & Cirelli): waking learning adds synapses; slow-wave
  sleep scales them down, and weak ones are pruned.
- **Replay** (sleep reactivation, as in [17](17-consolidation.md)): reactivated patterns
  expose which units carry the same information. Here that is used to merge them.
- **Pruning of unused synapses** by microglia (Stevens et al. 2007; Schafer et al. 2012)
  corresponds to the "only fails" prune and to least-recently-used recycling.
