# Top-down bias

A higher layer's expectation steering a lower layer's choice of output, without
changing what the lower layer learns. Implemented as
`KernelClass::process_predictive_biased(input, output, Some((bias, mode)))` in
[`src/kernel/class.rs`](../../src/kernel/class.rs).

## Modes
| Mode | Rule | When it helps |
|---|---|---|
| `SameDepth` | at the deepest matching depth, candidates agreeing with the bias win over hit rate | weak but informative top-down; never overrides longer bottom-up context |
| `Prefer` | any agreeing candidate wins | strong top-down (e.g. an oracle: 53% → 93%) |
| `Fallback` | output the bias when nothing matches | when bottom-up often has no answer |
| `PreferAndFallback` | both | |

## Evidence so far
- Word layer → character layer ([07](../experiments/07-top-down-bias.md)): gains are
  proportional to the higher layer's accuracy. +0.3 points with a real word layer
  (10% next-word accuracy), +39.5 with an oracle.
- Lexicon → segmentation ([04](../experiments/04-word-segmentation.md)): +2 F1 when
  restricted to known 3+ letter words. Letting 1–2 letter words trigger cuts **ran away**:
  more short cuts made short "words" more frequent, which made more short cuts.

## Design rules learned
1. **Weight top-down by its reliability** (precision weighting). Don't let a weak
   higher layer override a stronger lower one.
2. **Watch for positive feedback loops** when the lower layer's output trains the
   higher layer that biases it ([04](../experiments/04-word-segmentation.md)).
3. Top-down is a multiplier on the higher layer's quality, not a substitute for it.

## Related
Interactive activation (McClelland & Rumelhart 1981), predictive coding (Rao &
Ballard 1999; Friston 2005), HTM apical feedback: see
[related work](../related-work.md#top-down-feedback-and-predictive-coding).
