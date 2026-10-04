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

**Findings, in bits:**
1. **With 1–2 facts, bits match floats** in every configuration (97.1–99.4% vs
   97.7–99.1%).
2. **With 1–3 facts, the memory still finds the answer** (answer in recall 99.1–99.9%),
   but held-out accuracy drops to 89–91% (floats: 93.5–99.3%). The drop is one seed:
   seed 1 scores 78–79% in *all three* CA3 bit configurations, where floats scored 99–100%.
   So it is systematic, not noise.
3. Likely cause: **less clean readouts.** "Answer in recall" checks that the answer is
   present, not that it is alone. Coarse integer weights (16–23 per write, halved every 2
   stores, floored) make older episodes' weights equal sooner, so more of them pass the
   50%-of-best readout and extra places reach the predictor, which then leans on trained
   pairs. Next: measure how many places each recall contains, and give the counters more
   resolution (more planes, larger writes) or a sharper readout after removing the cue.
4. The separation and settling effects seen with floats are not visible here; the seed-1
   drop dominates.

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
| 1–3 facts, settle 2 | CNT_S_3 | 92.7% (1.00; 94/84/100) | SHIFT_S_3 |
| 1–3 facts, no settling | CNT_NS_3 | 90.0% (1.00; 90/81/99) | SHIFT_NS_3 |

SHIFT_FINDINGS

See [hippocampal functions](../concepts/hippocampal-functions.md) for the wider map, and
[13](13-big-loop.md) for chaining recalls.
