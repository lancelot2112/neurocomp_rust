# 32 · General and specific kernels side by side

**Question.** In [31](31-role-cells-and-transfer.md), pruning inputs that did not matter
gave full transfer to names never seen in training, but it destroyed rules where the
name does matter (57–73% → 21–30%), and habit (80% → 42%). Pruning modifies the kernel
in place, so the specific knowledge is lost. What if generalisation spawned a general
copy and kept the specific original, with the usual ranking deciding between them?

**Code.**
- `KernelClass::set_generalize_spawn`, `spawned`, and `install` (kernel placement, now
  shared by growth and spawning) in [`src/kernel/class.rs`](../../src/kernel/class.rs),
  with a unit test.
- In [`examples/episodic.rs`](../../examples/episodic.rs): `HIER_GEN_SPAWN`,
  `HIER_TRUST_AT_TEST`, and the `SPAWN` report.

Run: `… TASK=season SEASON_LEN=4 SEASON_RULE=season NEW_NAMES=1 POLICIES=nomemory HIER=1
HIER_LEVELS=1 MIX=1 HIER_GENERALIZE=0.5 HIER_GEN_SPAWN=1 cargo run --release --example
episodic` (other settings as in 31).

## Spawning
- **The trigger** is unchanged: a near-miss whose prediction the target confirms, with
  inputs that were absent.
- **Instead of pruning those inputs from the kernel,** a new kernel is installed:
  - the kept inputs and the same output;
  - a threshold tolerating a frame's worth of noise, as at growth;
  - depth = how far back the kept inputs reach;
  - one hit recorded.
- **The original is untouched.** Only its counters for the dropped inputs are reset, so
  it does not spawn the same copy again. An identical copy is never installed twice.
- **Both compete as every kernel does:** depth, then reliability.
  - Where the dropped inputs never mattered, the general kernel matches cases the
    specific one cannot (a new name).
  - Where they matter, the specific kernel stays.

The unit test (`spawned_general_kernels_transfer_without_losing_specifics`) checks it on
a minimal case. Frame 0 is a name (the fourth never trained), frame 1 a cue:
- **Role rule** (target depends on the cue only): spawning answers the unseen name for
  both cues. With no generalisation, neither is answered.
- **Name rule** (target depends on both): spawning keeps all six known pairs right.

## Results (seeds 0 / 1 / 2)

**Role rule** (in a season everyone goes to the same place), new names at test:

| | Known names | New names | Higher-area kernels |
|---|---|---|---|
| No generalisation | 52 / 56 / 64% | 21 / 0 / 32% | 3,400–4,300 |
| Pruning in place (31) | 65 / 66 / 71% | 61 / 66 / 72% | 455–493 |
| **Spawning** | 66 / 68 / 80% | **62 / 72 / 81%** | 8,200–13,600 |
| Spawning + sleep in the higher area | 67 / 70 / 74% | 64 / 75 / 76% | 1,200–1,500 |

**Name rule** (each name has its own place per season), known names at test:

| | Accuracy | Higher-area kernels |
|---|---|---|
| No generalisation | 66 / 57 / 73% | 4,000–4,300 |
| Pruning in place | 23 / 30 / 21% | 480–520 |
| Spawning | 46 / 71 / 43% | 11,200–14,000 |
| Spawning + sleep | 56 / 51 / 43% | 1,300–1,600 |

**Habit** ([24](24-cortical-hierarchy.md): name and time of day decide the place):

| | Accuracy | Higher-area kernels |
|---|---|---|
| No generalisation | 81 / 82 / 80% | 2,900–3,500 |
| Pruning in place | 42 / 45 / 41% | 310–370 |
| **Spawning** | **86 / 87 / 89%** | 7,800–8,600 |
| Spawning + sleep | 91 / 83 / 78% | 830–960 |

A reliability floor at test in the higher area (`HIER_TRUST_AT_TEST=0.5`) changed none of
these results.

## Findings
1. **Spawning keeps the transfer.** New names score as well as known ones (62–81%
   against 66–80%), the best result on the role rule so far.
2. **Spawning repairs the over-generalisation on habit, and more.** Pruning in place had
   cut habit to 41–45%. Spawning gives 86–89%, above the 80–82% without generalisation.
   The general copies drop the noise in the higher area's window (filler words that
   vary from story to story). The specific kernels keep the name and the time of day.
3. **The name rule is mixed.** It is far better than pruning (43–71% against 21–30%),
   but below no generalisation on two of three seeds (43–46% against 66–73%).
   - The reliability floor changed nothing, so the general copies there are not
     unreliable. They have most likely dropped window noise rather than names, as on
     habit.
   - The likely cost is volume: 8,000–14,000 copies are spawned, each a slightly
     different subset, and only identical ones are merged. More, near-identical kernels
     compete for every answer.
4. **Sleep compacts the copies 8–10×** (replay merges kernels that answer the same
   inputs, [23](23-compaction.md)), with no loss of transfer on the role rule. It does not
   fix the name rule, and makes habit less stable across seeds (78–91%).
5. **So general and specific knowledge can live side by side,** learned from data alone.
   - Where a slot's filler never matters, the general kernel answers for new fillers.
   - Where it matters (habit), the specific ones win.
   - The remaining weakness is how copies are made: too many, too similar.

## Step 1: repeated evidence, and only reliable cover
Two changes to how copies are made:
- **Repeated evidence (`HIER_SPAWN_AFTER=n`, `set_spawn_after`).** An input is dropped
  from a copy only after it was absent in n confirmed near-misses of the kernel. A copy
  then stands for a regularity, not one coincidence.
- **Subsumption (always on with spawning).** No copy is spawned if a kernel with the same
  output already reads a subset of the kept inputs (it matches whenever the copy would)
  and is at least as reliable as the kernel the copy comes from.
  - The candidates are found through the covering kernel's lowest input bit, as in
    sleep's merge.
  - A first version omitted the reliability condition. It blocked 100,000–360,000 copies,
    and the unreliable over-general kernels that "covered" them then answered: habit
    fell to 62–78%. Requiring the cover to be at least as reliable, as sleep's merge does,
    fixed it.

| Seeds 0 / 1 / 2 | Role rule: known / **new** names | Name rule | Habit | Kernels (name rule) | Test µs/word (name rule) |
|---|---|---|---|---|---|
| No generalisation | 52 / 56 / 64% / **21 / 0 / 32%** | 66 / 57 / 73% | 81 / 82 / 80% | 4,000–4,300 | 106 |
| Spawning (above) | 66 / 68 / 80% / **62 / 72 / 81%** | 46 / 71 / 43% | 86 / 87 / 89% | 11,200–14,000 | – |
| Spawn after 1 + reliable cover | 68 / 67 / 74% / **71 / 67 / 80%** | 52 / 73 / 61% | 88 / 84 / 86% | 8,500–9,900 | – |
| **Spawn after 3 + reliable cover** | 61 / 81 / 70% / **61 / 81 / 69%** | **69 / 73 / 71%** | **85 / 84 / 82%** | 42,000 | 865 |
| … + sleep in the higher area | 66 / 65 / 62% / 62 / 56 / 68% | 66 / 67 / 56% | 67 / 69 / 82% | 2,000–2,400 | 69–83 |

1. **Spawning after three confirmations passes all three tests:**
   - new names as accurate as known ones (61–81%);
   - no loss on the name rule (69–73% against 57–73%, better on average);
   - habit above the default (82–85% against 80–82%).
2. **The price is size.** Copies made after repeated evidence are more varied (each keeps
   a different subset), so the higher area holds about 42,000 kernels, and test time
   rises to 865 µs/word, about 8×.
3. **Sleep compacts it 20×** (to 2,000–2,400 kernels, 69–83 µs/word) but gives back 5–15
   points on each test. Its merge removes specific kernels whenever a general one is at
   least as reliable over the replay, and 512 replayed inputs are too few to show where a
   specific kernel still matters.

## Does sleep prune, or consolidate into more general rules?
Both, but it does not invent rules. `KernelClass::sleep` does three things:
1. **Downscale.** Every kernel's hit and miss counts are halved, so old evidence weighs
   less (synaptic homeostasis, Tononi & Cirelli).
2. **Prune** kernels that have only failed.
3. **Merge:**
   - A kernel is removed when another with the same output reads a subset of its inputs
     and is at least as reliable. The general one then answers for both.
   - Kernels that match exactly the same replayed inputs collapse into the most reliable
     one.

So sleep chooses between general and specific versions that waking learning made. It
consolidates toward the general one when the evidence says the specifics add nothing.
With spawning, that is where specific kernels get absorbed into their general copies:
the complementary-learning-systems picture of slow consolidation into general
structure. What it cannot do yet is create a new, more general rule offline. Replay
could do that by applying the near-miss rule to replayed inputs (generalisation during
sleep), a natural next step.

## Next
- ~~Spawn less often~~, ~~merge copies by what they keep~~: done (step 1).
- **Keep it small without losing specifics:** sleep with a longer replay (so the merge
  sees where specifics matter), or a cap on copies per source kernel.
- **Generalise during sleep:** run the near-miss rule on replayed inputs, so general rules
  form offline.
- **Then the schema test proper:** a new name learned in one exposure for a
  name-dependent rule (Tse et al. 2007). The general kernels give the structure, and one
  episode supplies the specific fact.

## Biology
- **Complementary learning systems** keep general structure and specifics apart, and
  let the general support fast learning of new specifics (McClelland, McNaughton &
  O'Reilly 1995; McClelland 2013).
- **Generalisation with exceptions:** prototypes and exemplars together, in category
  learning. Spawning keeps both, and reliability picks per case.
