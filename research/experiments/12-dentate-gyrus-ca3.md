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
| list memory ([11](11-episodic-memory.md)) | CA3_LIST_2 | CA3_LIST_3 |
| CA3, low separation (1,024 cells, 64 active), settle 2 | CA3_LOW_2 | CA3_LOW_3 |
| CA3, high separation (16,384 cells, 32 active), settle 2 | CA3_HIGH_2 | CA3_HIGH_3 |
| CA3, high separation, no settling | CA3_NOSETTLE_2 | CA3_NOSETTLE_3 |

## Findings
CA3_FINDINGS

See [hippocampal functions](../concepts/hippocampal-functions.md) for the wider map, and
[13](13-big-loop.md) for chaining recalls.
