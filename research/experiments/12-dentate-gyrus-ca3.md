# 12 · Dentate gyrus expansion and a Hebbian CA3 store

**Question.** The list memory of [11](11-episodic-memory.md) is an idealization: one slot
per episode, exact recency. Can the classic hippocampal circuit do the same job:
dentate-gyrus pattern separation, Hebbian CA3 storage in shared weights, and recurrent
completion? And does pattern separation matter once memories are superimposed?

**Code.** [`src/program/hippocampus.rs`](../../src/program/hippocampus.rs) (`DentateGyrus`,
`Ca3Memory`); `Ca3` policies in [`examples/episodic.rs`](../../examples/episodic.rs).
Run: `TASK=varied SEEDS=3 POLICIES=ca3 cargo run --release --example episodic`
(env: `CA3_DECAY`, `CA3_READOUT`, `DG_FAN_IN`).

## The circuit (after Marr 1971; Treves & Rolls 1994)
- **Dentate gyrus:** a fixed random expansion. Each granule cell samples `fan_in`
  (300) random EC bits; the `k` most-driven cells win. The code is sparse and
  conjunctive (pattern separation).
- **CA3 storage:** Hebbian EC→CA3, CA3→CA3 (recurrent) and CA3→EC weights, linking the
  episode's EC bits to its dentate-gyrus code.
- **Recall:** the EC cue drives CA3 directly (perforant path), winner-take-all, then a
  few recurrent settling steps (pattern completion), then the EC readout keeps bits
  ≥ `readout_fraction` of the best.
- **Forgetting:** every stored episode multiplies all weights by `decay` (a palimpsest),
  so recent memories dominate; weak synapses are pruned.

## What it took (single seed, original stories, held-out pairs)

| Change | Held-out |
|---|---|
| store the whole sentence, decay 0.97 | ~17% (chance) |
| store only the **novel (habituated)** part, decay 0.97 | 0% |
| + decay 0.85 | 52% |
| + **decay 0.7** | **91%** |
| decay 0.85 + sparser granule fan-in (100) | 23% |
| decay 0.85 + sharp readout (0.8) | 0% |

What each step showed:
1. **Pattern separation fails if most of every episode is shared.** "went to the ."
   is in every fact sentence, dominates granule-cell drive, and every episode gets the
   *same* dentate-gyrus code. Recall then returned only function words (trace:
   "went to the ."). Encoding the **novel** part, i.e. habituated EC input (biologically,
   novelty-weighted encoding), fixed separation.
2. **Superposed memories blend, and recency must dominate.** Several episodes about the
   same person share their name's synapses. With slow decay the readout mixes their
   places ("anna → garden, bedroom"); held-out places lose to trained ones. Fast
   forgetting (0.7 per stored sentence) makes the latest episode win.
3. **A sharp readout is wrong for superpositions.** The cue word is part of *every*
   recalled episode, so it is the strongest component; an 80% cut keeps only the cue.
   Clean-up has to happen after the cue is removed.

## Results (varied stories, 3 seeds, held-out pairs)

| Memory | 1–2 facts | 1–3 facts |
|---|---|---|
| list memory ([11](11-episodic-memory.md)) | 99.0% | 97.5% |
| CA3, low separation (1,024 cells, 64 active), settle 2 | 97.7% | 93.5% (91/99/90) |
| CA3, high separation (16,384 cells, 32 active), settle 2 | 99.1% | 96.1% (89/100/100) |
| CA3, high separation, no settling | 99.0% | **99.3%** (98/100/100) |

## Findings
1. **A Hebbian, superimposed store matches the idealized list memory** (99% vs 99%;
   99.3% vs 97.5% with 1–3 facts), once it stores the novel part of each episode and
   forgets fast enough that the latest episode about a person dominates.
2. **Pattern separation matters as load grows.** With few dense cells (1,024 / 64 active)
   held-out accuracy drops from 97.7% to 93.5% when stories have up to 3 facts; with many
   sparse cells (16,384 / 32) it stays at 96–99%. Overlapping episodes blend in a small
   code.
3. **Recurrent settling (pattern completion) does not help here, and can hurt.** Without
   settling: 99.0% / 99.3%; with 2 steps: 99.1% / 96.1%, with one seed at 89%. The cue
   (a name) already selects the right cells; recurrent steps pull the code toward
   whatever attractor is strongest, sometimes an older or blended memory. Completion
   should pay off with *degraded* cues (missing or noisy words), which this task never
   presents. That needs its own test.
4. Forgetting is a fixed, global decay (0.7 per stored sentence). A store that keeps more
   than the last few facts per name would need decay tied to novelty or interference
   (the CA1 comparator of [14](14-ca1-comparator.md) is a step towards that).

## In bits
The store above used an f32 weight per synapse with exponential decay. It now uses
bit-sliced counters ([probability in bits](../concepts/probability-in-bits.md)):
- **Weights:** each pathway (EC→CA3, CA3→CA3, CA3→EC) has one row per source cell, a
  7-plane `SlicedCounter` (values 0–127) over its targets, allocated on first write.
- **Storing** adds an integer amount to the row's counters under the target mask
  (ripple XOR/AND from the planes of the amount's set bits).
- **Decay** is a plane shift every `half_life` stores (decay 0.7 → every 2 stores,
  ≈ 0.707 per store), applied lazily by skipping shifted-out planes when a row is read.
- **Recency between halvings:** the n-th store in a period adds 16 · 2^(n/half_life)
  (16 or 23 for half_life 2), so newer episodes outweigh older ones as with continuous
  decay. Without this, a unit test failed: same-period episodes tied.
- **Recall:** drive onto each target = Σ over active sources and planes of 2^p for each set
  bit; the same k-winner-take-all, settling and readout fraction as before.

| Memory (varied stories, 3 seeds, held-out) | 1–2 facts, floats | 1–2 facts, **bits** | 1–3 facts, floats | 1–3 facts, **bits** |
|---|---|---|---|---|
| list memory | 99.0% | 98.7% (same code; rerun) | 97.5% | 98.7% (same code; rerun) |
| CA3 1,024 / 64, settle 2 | 97.7% | 97.1% | 93.5% | 90.3% (98/78/95; answer in recall 99.1%) |
| CA3 16,384 / 32, settle 2 | 99.1% | 99.4% | 96.1% | 89.1% (92/79/96; answer in recall 99.9%) |
| CA3 16,384 / 32, no settling | 99.0% | 99.3% | 99.3% | 90.9% (96/78/100; answer in recall 99.8%) |

Same-seed check (short stories, seed 0, CA3 16,384 / 32, settle 2), floats vs bits:
1–2 facts **72.2% vs 73.6%** (answer in recall 94.4% vs 94.1%); 1–3 facts **100% vs 98.6%**
(answer in recall 100% in both). The bit-sliced store reproduces the float store; this seed's
low 1–2-fact score is the predictor under-using a correct recall, in both versions
(as with one seed of the list memory in [11](11-episodic-memory.md)).

**Findings, in bits** (corrected; see the seed-1 diagnostic below):
1. **With 1–2 facts, bits match floats** in every configuration (97.1–99.4% vs
   97.7–99.1%).
2. **With 1–3 facts, every bit version loses one seed** (seed 1 at 78–79%) while the
   floats column above had 99–100% on it. That comparison was **across builds**: the
   float column came from an earlier build of the example (the list memory, which does
   not use CA3, also moved, 97.5% → 98.7%). Run in the *same* build, the float store
   does no better on seed 1 (74.6%; below). The conversion to bits did not cause the drop.

## Decay as a shift: delay line and shift register
The counter store above halves every 2 stores and floors, so older episodes' weights
become equal and (we suspected) blend into the readout. Two alternatives store each
write in an **age plane** (one bit plane per recent store, in a ring indexed by store
number; decay = advancing the ring, expired planes cleared on reuse):
- **Delay line** (`Ca3Memory::new_delay_line`, `CA3_STORE=ring`): plane weight is an exact
  integer table round(1024·0.7^age), 11 planes. Same decay as floats, no rounding.
- **Shift register** (`Ca3Memory::new_shift_register`, `CA3_STORE=shift`): plane weights are
  powers of two (1024, 512, …, 1), so a synapse's planes *are* the binary digits of its
  weight, newest write = most significant bit. Writing is OR, decay is a shift
  (0.5 per store), no table, no rounding. Fully bit-native storage; summing over
  active inputs is still integer adds over set bits.

A new diagnostic, **places in recall**, counts distinct place words the memory hands the
predictor at the answer (1.00 = clean).

| Varied stories, CA3 16,384 / 32, 3 seeds, held-out (places in recall) | Counters | Delay line | Shift register |
|---|---|---|---|
| 1–2 facts, settle 2 | 99.4% (1.00) | 99.3% (1.00) | 97.2% (1.00) |
| 1–2 facts, no settling | 99.3% (1.00) | 99.5% (1.00) | 98.1% (1.00) |
| 1–3 facts, settle 2 | 89.1% (1.00; 92/79/96) | 92.7% (1.00; 94/84/100) | 92.1% (1.00; 97/79/100) |
| 1–3 facts, no settling | 90.9% (1.00; 96/78/100) | 90.0% (1.00; 90/81/99) | 90.3% (1.00; 98/75/98) |

**Seed-1 diagnostic** (same build, varied stories, 1–3 facts, CA3 16,384 / 32, settle 2;
`DIAG=1 SEED_START=1 SEEDS=1 CA3_STORE=float|ring|…`). For held-out questions, what the
predictor received, split by right and wrong answers:

| Store | Held-out | Seen pairs | Wrong: answer bits / recall bits / whole words | Right: answer bits / recall bits / whole words |
|---|---|---|---|---|
| float (`Ca3FloatMemory`) | 74.6% | 75.2% | 30.6/32, 149, 5.02 | 30.2/32, 138, 4.57 |
| counters | 79.4% | 76.6% | 30.9/32, 144, 4.84 | 30.2/32, 133, 4.41 |
| delay line | 84.0% | 80.6% | 30.1/32, 139, 4.70 | 30.4/32, 136, 4.49 |

**Findings:**
1. **The bit stores are as good as the float store**, here slightly better. Counters,
   delay line and shift register all recall exactly one place, with the answer present
   99.8–100% of the time, on every row of the table above.
2. **Partial place codes are not the cause.** Wrong answers received ~30.5 of the answer's
   32 bits, the same as right answers, in recalls of the same size and word count.
3. **The failure is the predictor's, and it is learned.** Seed-1 predictors are wrong
   on a clean, complete recall, and they are equally bad on *seen* pairs (75–81%). This
   is how training went on that seed (which kernels grew, and which input frames they
   key on), not what memory delivers at test. Next: look at what the seed-1 predictor's
   winning kernels read at the answer (memory frame vs the cue word), e.g. with credit
   readouts.
4. **Decay as a pure shift works.** The bit-native shift register (0.5 decay per store,
   writes as OR, decay as advancing a ring) performs like the 0.7 versions: 97.2/98.1%
   (1–2 facts) and 92.1/90.3% (1–3 facts).
**Which input the winning kernel reads** (delay line, settle 2, 1–3 facts; `DIAG=1`
prints `KDIAG`). Input frames are [current word "?" | recalled memory | previous word "now"];
each kernel samples 16 bits per frame it reaches.

| Seed, held-out | n | Winner reads memory | Mask bits current / memory / previous | Winner reliability |
|---|---|---|---|---|
| seed 0, right | 471 | 100% | 16.0 / 16.0 / 9.0 | 0.89 |
| seed 0, wrong | 29 | 100% | 16.0 / 16.0 / 16.0 | 0.53 |
| seed 1, right | 420 | 81% | 16.0 / 12.9 / 16.0 | 0.84 |
| **seed 1, wrong** | 80 | **29%** | 16.0 / **4.6** / 16.0 | **0.24** |

(With 1–2 facts both seeds look like seed 0: every winner reads all three frames,
reliability 0.99.)

6. **Seed 1's errors come from memory-blind kernels.** 71% of its wrong answers are made by
   a kernel that reads only "?" and "now", i.e. it guesses a place from context that
   carries no information (reliability 0.24). Such a kernel wins only when no
   memory-reading kernel of the same depth matches, so the memory-reading kernels
   failed to match a recall that *did* contain the answer.
7. **Likely reason:** a kernel samples its 16 memory bits at random from the whole recall,
   which holds ~4.5 words (the place plus adverbs, adjectives and fillers from the same
   sentence). Kernels that sampled incidental words match only when those words recur;
   on seed 1, with more facts and more varied recalls, they often don't, and the
   memory-blind fallback wins. Seed 0's wrong answers instead had unusually large
   recalls (198 bits, 6.7 words), which crowd the sample the same way.
8. So the remaining gap is **attention within the recall** (credit for *which* recalled
   bits predict the answer), not memory. Candidate fixes: gradual pruning of unused
   synapses (`generalize_after`, which took [09](09-thalamic-attention.md) to 82%);
   growing kernels that sample memory bits overlapping the target (the answer is in the
   recall); or a cleaner readout (the [CA1 comparator](14-ca1-comparator.md) or the
   [basal-ganglia selector](15-basal-ganglia-selector.md) choosing one item).

**Does gradual synapse pruning fix it?** (`GENERALIZE=0.5 GENERALIZE_AFTER=N`, the fix that
took [09](09-thalamic-attention.md) to 82%; seed 1, delay line, settle 2.)

| Seed 1 | 1–2 facts | 1–3 facts | Wrong answers (1–3): winner reads memory / memory bits / reliability |
|---|---|---|---|
| off | 99.2% | **84.0%** | 29% / 4.6 / 0.24 |
| prune after 3 misses | 99.2% | 77.4% | 56% / 2.2 / 0.34 |
| prune after 1 miss | 16.0% | 35.2% | 100% / 1.0 / 0.18 |

**No, it makes it worse.** Pruning removes inputs that were silent when a near-miss kernel
would have been right. The recalled place changes from story to story, so a kernel's
memory bits are exactly the inputs that look silent, and they go first: after one miss,
kernels lose them entirely (chance on 1–2 facts); after three, winners keep ~2–9 of 16
memory bits, too few to tell places apart. Pruning helps when the useful input is
*fixed* (a route's relayed word in 09) and hurts when it is *variable content to copy*.
What the predictor lacks is a way to grow kernels that read the right memory bits in the
first place: e.g. sample memory bits that overlap the target (the answer is in the
recall, so a "copy" kernel can be grown in one step), or hand it a one-item recall.

**Sticky synapses from credit** (`STICKY=f`, `KernelClass::set_sticky`): input bits that
earn *copy credit* (they carry the same bit, frame-relative, as the word the kernel
predicts) get a tag, and pruning a tagged bit takes f× as many silent confirmations
(synaptic tagging / metaplasticity). Tags are earned on hits, and in a second version also
at the kernel's birth (it is grown to predict this target from this input).

| Seed 1, held-out | 1–2 facts | 1–3 facts | 1–3 wrong answers: winner reads memory / memory bits / reliability |
|---|---|---|---|
| pruning off | 99.2% | **84.0%** | 29% / 4.6 / 0.24 |
| prune after 1 miss | 16.0% | 35.2% | 100% / 1.0 / 0.18 |
| + tags on hits (×4 or ×16) | 99.4% | 35.2% | (identical: tags never earned in time) |
| + tags at birth (×4) | 99.4% | 66.6% | **0% / 0.0 / 0.18** |
| prune after 3 misses | 99.2% | 77.4% | 56% / 2.2 / 0.34 |
| + tags on hits (×4) | 99.2% | 77.0% | 55% / 2.2 / 0.34 |
| + tags at birth (×4) | 99.2% | 76.4% | 53% / 2.1 / 0.32 |

- **Tags protect the right bits.** On 1–2 facts, tags turn aggressive pruning from chance
  (16%) into 99.4%: kernels are pruned down to their tagged place bits (4–8 of 16) and
  still match. Tags earned only on hits come too late with 1–3 facts (a new kernel's
  place bits are pruned at its first near miss, before any hit); tagging at birth fixes
  that.
- **With tags at birth the split is clean:** every right answer (333) comes from a
  memory-reading copy kernel with reliability 1.00, every wrong answer (167) from a
  memory-blind guesser (0 memory bits, reliability 0.18). Copy kernels are now perfect
  when they fire; the failures are questions where *no* copy kernel matches.
- **So the remaining problem is coverage, not pruning.** Likely cause (not yet confirmed):
  the guesser reads "?" and "now", so it spans the deepest frame; when it wins and is
  wrong, surprise-driven growth goes *deeper* than the winner, and there is no deeper
  frame, so no new copy kernel is grown for that case. A low-reliability kernel at
  maximum depth locks the case in. Next: grow a sibling at the same depth when a
  max-depth winner fails, or rank by reliability before depth for unreliable kernels.

**Coverage or ranking?** A third diagnostic (`CDIAG`) checks, at each wrong held-out answer,
whether a memory-reading kernel for the right place exists, how close it came to
matching, and how reliable it is. It turned the problem from "coverage" into three
separate predictor bugs, each fixed in turn (seed 1, varied, held-out):

| Fix (cumulative unless noted) | 1–2 facts | 1–3 facts | What the wrong answers showed |
|---|---|---|---|
| baseline (no pruning) | 99.2% | 84.0% | no copy kernel *for the "?" context*: the best one missed 13.8 current-word bits |
| prune after 1 miss + tags at birth | 99.4% | 66.6% | a copy kernel **fully matched** but lost to a deeper guesser (0.18) |
| + trust floor 0.5 (always) | 99.4% | 83.6% | the matching copy kernels were themselves over-pruned guessers (reliability 0.16–0.44 over 1–3k uses): 2–3 memory bits under a global tolerance of 3 |
| + frame floor (≥ tolerance+1 bits per frame) | 83.8% | – | worse: a 4-bit frame still matched with 1 bit present |
| **tolerance scaled to the smallest frame** (instead of the floor) | 99.4% | 81.8% | reliable copy kernel (0.99) fully matched, lost on **depth** to a guesser |
| + trust floor during training too | 82.4% | 100% | 1–2: copy kernels never grown for some places (training ranking changes growth) |
| **+ trust floor only when answering** (`TRUST_AT_TEST`) | **99.4%** | **100%** | none on 1–3; 3 on 1–2 (recall lacked the place bits) |

Control: the test-only trust floor **without** pruning stays at 99.2% / 84.0%. There the
right copy kernel does not exist for the "?" context (best one matches 0.22 of its
threshold), so ranking alone cannot help: pruning (which makes copy kernels general across
contexts) and the ranking fix are both needed.

The three bugs, all in how a predictive class picks and prunes kernels:
1. **Global tolerance on pruned kernels.** `threshold = bits − 3` lets a frame pruned to ≤ 4
   bits be absent and the kernel still fire. Fix: tolerance = ⌊smallest frame × (1 −
   match_fraction)⌋ after pruning.
2. **Depth before reliability.** The longest-context kernel wins even at reliability 0.18.
   Fix: depth only ranks among kernels at least 0.5 reliable, when answering.
3. **Pruning before credit.** Copy bits must be tagged when the kernel is born (it is
   grown to predict this target), not only after a hit.

Settings: `GENERALIZE=0.5 GENERALIZE_AFTER=1 STICKY=4 TRUST_AT_TEST=0.5`. Three-seed results (varied stories, held-out): list memory **100% / 100%** (1–2 / 1–3 facts, every seed; was 98.7% / 98.7%); CA3 FULL_FIX_CA3

11. Method lesson: compare variants **in the same build and run**. Numbers from different
   builds of an example are not comparable, even with the same seed.

See [hippocampal functions](../concepts/hippocampal-functions.md) for the wider map, and
[13](13-big-loop.md) for chaining recalls.
