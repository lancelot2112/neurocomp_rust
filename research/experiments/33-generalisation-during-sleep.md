# 33 · Generalisation during sleep: general rules formed offline from replay

**Question.** In [32](32-general-and-specific.md), spawning a general copy beside the
specific kernel passed all three tests:
- transfer to new names;
- no loss on a name-dependent rule;
- habit.

But it took 42,000 kernels and 8× the test time. Sleep compacted that but cost 5–15
points, and sleep itself could not form new rules. Can rules be formed *offline*, from
the replay, and tested on it before they are kept?

**Code.**
- `KernelClass::set_sleep_generalize`, `generalize_from_replay`, `slept_general`, and
  replay pairs with targets, in [`src/kernel/class.rs`](../../src/kernel/class.rs), with
  a unit test.
- In [`examples/episodic.rs`](../../examples/episodic.rs): `HIER_SLEEP_GEN`, `HIER_DREAM`,
  `HIER_REPLAY_LEN`, and the `SLEEPGEN` report.

Run (best setting): `… TASK=season SEASON_LEN=4 SEASON_RULE=season NEW_NAMES=1
POLICIES=nomemory HIER=1 HIER_LEVELS=1 MIX=1 HIER_DREAM=1 HIER_SLEEP_GEN=3 cargo run
--release --example episodic` (other settings as in 31). There is no waking
generalisation in the higher area.

## Generalising from replay
1. **Remember with targets.** The class keeps its last `replay_len` (input, target) pairs
   (512 here).
2. **Find confirmed near-misses.** At each sleep time (every 500 stories), every kernel
   is matched against every replayed input through the inverted index. For each kernel,
   count its confirmed near-misses: at least half its inputs present, not enough to fire,
   and the target was its output. Also count how often each of its inputs was absent in
   them.
3. **Form a candidate.** If a kernel has at least n = 3 confirmed near-misses, the inputs
   absent in at least 3 of them are dropped. The threshold again tolerates a frame's
   worth of noise.
4. **Test it on the whole replay.** The candidate is kept only if it matches more
   replayed inputs than its source and is at least as reliable on them, compared by
   cross-multiplying smoothed rates. Its replay record becomes its starting hit and miss
   counts.
5. **Nothing is removed.** Specific kernels stay. Where both match, the usual ranking
   decides.

Two ways to run it:
- **`HIER_SLEEP_GEN` with `HIER_SLEEP`:** as step 0 of sleep, followed by sleep's
  downscaling, pruning and merging.
- **`HIER_DREAM`:** generalisation from replay alone, with nothing downscaled, pruned or
  merged.

**Unit test** (`sleep_forms_general_rules_from_replay`), on the minimal case of 32 with no
waking generalisation:
- **Role rule:** after one sleep the unseen name is answered for both cues (without sleep
  generalisation, for neither).
- **Name rule:** all six known pairs stay right.

## Results (seeds 0 / 1 / 2)

| | Role rule: known / **new** names | Name rule | Habit | Higher-area kernels | Test µs/word |
|---|---|---|---|---|---|
| No generalisation, no sleep | 52 / 56 / 64% / **21 / 0 / 32%** | 66 / 57 / 73% | 81 / 82 / 80% | 3,000–4,300 | ~106 |
| Waking spawn after 3 (32) | 61 / 81 / 70% / **61 / 81 / 69%** | 69 / 73 / 71% | 85 / 84 / 82% | 22,000–42,000 | 865 |
| Sleep only (control) | 45 / 54 / 71% / **7 / 0 / 24%** | 62 / 56 / 64% | 77 / 70 / 77% | 1,700–2,700 | 88–101 |
| Sleep + generalisation (512 replayed) | 64 / 76 / 79% / **58 / 76 / 82%** | 46 / 59 / 55% | 75 / 70 / 74% | 900–2,000 | 67–109 |
| Sleep + generalisation (2,048 replayed) | 50 / 59 / 72% / **45 / 8 / 72%** | 60 / 53 / 61% | 71 / 57 / 76% | 1,400–2,700 | 76–93 |
| **Generalisation from replay only** | 66 / 78 / 76% / **63 / 76 / 78%** | **62 / 70 / 65%** | **85 / 82 / 81%** | 3,100–4,400 | 96–120 |

With replay only, each run formed 590–930 general rules and rejected 2,100–4,900
candidates that failed the replay test.

## Findings
1. **General rules can be formed offline and tested before they are kept.** From replay
   alone, new names score as well as known ones (63–78% against 66–78%). The name rule
   loses nothing (62–70% against 57–73%; 65.3 against 65.0 on average), and habit
   matches or beats the default (81–85%).
2. **At normal size and speed.** A few hundred rules per run do what tens of thousands of
   waking copies did: 3,100–4,400 kernels and 96–120 µs/word, the default model's size
   and cost (waking spawning: 22,000–42,000 kernels, 865 µs/word).
3. **The replay test does the selecting.** For every rule kept, two to eight candidates
   were rejected because they were not more general in fact, or less reliable, on the
   replay. Waking spawning had no such test, so it kept every copy and relied on the
   ranking to sort them out.
4. **Sleep's merge is what costs, not generalisation.** Sleep alone (no generalisation)
   already loses 1–12 points on habit and the name rule. With generalisation, its merge
   removes specific kernels whenever a general rule is at least as reliable over the
   replay, which the name rule cannot afford.
   - A longer replay (2,048) did not help: fewer candidates passed.
   - Merging needs evidence of where specifics matter that a short replay does not hold.
5. **So the split that works:** generalise offline, keep the specifics, let the waking
   ranking choose per case. That is closer to complementary learning systems than "sleep
   = compression": replay builds general structure in addition to, not instead of, the
   specific knowledge.

## Biology
- **Replay builds structure.** Hippocampal replay during sleep is thought to train the
  cortex on regularities across episodes (McClelland, McNaughton & O'Reilly 1995;
  systems consolidation). Sleep helps people discover hidden rules: more participants
  found a hidden shortcut in a number task after sleep than after waking (Wagner et al.
  2004), and sleep favours gist and generalisation over details (Lewis & Durrant 2011).
- **Not only downscaling:** synaptic homeostasis (Tononi & Cirelli 2014) is the pruning
  half. Here the generalising half helps on every test, and the merging half has to be
  careful with specifics.

## Next
- **The one-exposure schema test:** with general rules in place, can a new name's
  *specific* fact (under the name rule) be learned from a single story? The general
  rules supply the structure; one episode, perhaps replayed during the next sleep,
  supplies the fact.
- **Sleep for the column too,** and replay drawn from the hippocampal store
  ([17](17-consolidation.md)) instead of a recent buffer, so rules can draw on the whole
  past.
