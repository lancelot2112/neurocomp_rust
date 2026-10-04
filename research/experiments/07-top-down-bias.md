# 07 · Top-down bias from a higher layer

**Question.** Does feedback from a higher layer, used as a *bias* on a lower layer's
prediction, help the network?

**Code.** `KernelClass::process_predictive_biased` and `BiasMode` in
[`src/kernel/class.rs`](../../src/kernel/class.rs); experiment
[`examples/topdown.rs`](../../examples/topdown.rs) (commit `13ab472`).
Run: `cargo run --release --example topdown [max_chars]` (~4 min).

## Setup
- First 60,000 characters of *Alice* (with spaces), read once, online.
- **Character layer:** predictive class over the last 6 characters.
- **Word layer:** predictive class over the last 2 words, run at each word end to
  predict the next word.
- **Top-down signal:** while a word is being read, if the predicted word still fits
  the letters so far, its next letter (or a space once complete) is the expected
  next character. After a space, the expected character is the predicted word's
  first letter.
- **Bias modes** (learning is identical in every condition; only the choice of
  output changes):
  - `SameDepth`: at the deepest matching depth, candidates agreeing with the bias win.
  - `Prefer`: any agreeing candidate wins.
  - `Fallback`: output the bias only when nothing matches.
  - `PreferAndFallback`: both.
- **Controls:** a confidence gate (bias only when the word layer's winning kernel has
  hit rate ≥ 0.5), and an **oracle** word layer that knows the true next word.

## Results

| Condition | Char accuracy | Word-initial | Inside word | Space/punct |
|---|---|---|---|---|
| no bias | 53.1% | 19.1% | 58.4% | 68.2% |
| SameDepth | **53.4%** | 20.2% | 58.6% | 68.0% |
| Prefer | 53.0% | 19.0% | 58.7% | 67.4% |
| Fallback | 53.1% | 19.1% | 58.4% | 68.2% |
| Prefer, gated on word confidence ≥ 0.5 | 53.2% | 19.4% | 58.7% | 67.6% |
| SameDepth, gated | 53.4% | 20.1% | 58.6% | 68.1% |
| oracle word layer + SameDepth | 73.1% | 46.5% | 82.3% | 73.1% |
| oracle word layer + Prefer | **92.6%** | 94.5% | 97.9% | 78.7% |

Diagnostics: the real word layer predicts the next word 10.5% of the time. Its
expected character exists at 24.5% of positions and is right 39.6% of those, below
the character layer's own 58% inside words.

## Findings
1. **The bias mechanism works, and its value is bounded by the higher layer's
   accuracy.** A perfect word layer lifts character prediction from 53% to 93%. A
   word layer that is right 10% of the time adds 0.3 points.
2. **Weak top-down must not override bottom-up.** `Prefer` slightly hurts with the
   real word layer. Gating on the higher layer's confidence removes the harm but adds
   no gain. This is precision weighting, as in predictive-coding accounts
   ([Rao & Ballard 1999; Friston 2005](../related-work.md#top-down-feedback-and-predictive-coding)).
3. `Fallback` changes nothing here: positions where no character kernel matches
   rarely have a word-layer expectation either.
4. The same lesson as the lexicon feedback in [04](04-word-segmentation.md):
   recurrence from above helps exactly as much as the higher layer knows, and can
   hurt (or run away) when it doesn't. The classic demonstration is the
   word-superiority effect in the interactive-activation model
   ([McClelland & Rumelhart 1981](../related-work.md#top-down-feedback-and-predictive-coding)).

## So, does recursion from higher layers help?
Not yet, because the higher layers are weak. Making top-down useful depends on a
better word layer. That needs generalization across similar contexts
([open questions](../open-questions.md) #2) and the binding mechanism
([variable binding](../concepts/variable-binding.md)). See
[top-down bias](../concepts/top-down-bias.md).
