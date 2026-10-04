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

See [hippocampal functions](../concepts/hippocampal-functions.md) for the wider map, and
[13](13-big-loop.md) for chaining recalls.
