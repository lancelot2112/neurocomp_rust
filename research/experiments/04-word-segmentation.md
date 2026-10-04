# 04 · Word segmentation and recognition without spaces

**Question.** Can the network discover words when it is never shown where they begin
and end, and recognize them once found?

**Code.** [`examples/segment_words.rs`](../../examples/segment_words.rs);
`KernelClass::target_probability()` in [`src/kernel/class.rs`](../../src/kernel/class.rs).
Run: `cargo run --release --example segment_words` (~3 min).

## Setup
- *Alice*, letters only (`alicewasbeginningtoget…`): 107,685 letters, 26,687 words.
  Truth boundaries come from the original spaces and punctuation, and are used only to score.
- An 8-letter-context predictive network reads the letters once, online.
- Scored on the second half of the book: boundary precision/recall/F1, and word-token
  F1 (a token counts only if both of its edges are right).
- Baseline: transitional probability P(next | current), the statistic infants appear
  to use ([Saffran et al. 1996](../related-work.md#word-segmentation)).

## Signals tried

| Cut rule | Boundary F1 | Token F1 |
|---|---|---|
| random cuts at the true word rate | 24.2 | 2.7 |
| transitional probability, local minima | 55.9 | 21.1 |
| network *confidence* (hit rate of its own guess), local minima | 52.2 | 19.8 |
| **network P(actual next letter), local minima** | **61.9** | **29.6** |
| + top-down: cut after a known 3+ letter word when P(next) < 0.5 | **63.9** | **30.8** |
| + top-down allowing 1–2 letter words | 48–56 | 6–16 |

*P(actual next letter)* is the hit rate of the longest-context matching kernel that
predicted the letter that actually came, or 0 if none did
(`KernelClass::target_probability`).

## Recognition (the lexicon)
Each cut chunk is presented to a word layer (a `KernelClass`) as its letters plus a
boundary marker. Either an existing word kernel fires (recognized) or a new one is
grown with a fresh word code. With the best rule:
- 6,017 lexicon entries, 12.8% exact real words; **41% of chunk tokens are real words**.
- 89% of second-half chunks are recognized as already-known entries.
- Most frequent entries: *the, th, and, he, to, ing, she, you, in, it, at, her, was, be, re, ed, said, of*.
  These are real words mixed with frequent sub-word pieces (*th, ing, ed, re*), the
  same pieces BPE-style tokenizers find.

## Findings
1. **The surprise of the actual next letter is a better boundary cue than
   transitional probability** (+6 F1), because it uses up to 8 letters of context, not 1.
   This matches the infant-statistics and Elman-style
   ([Elman 1990](../related-work.md#word-segmentation)) prediction-error accounts.
2. **The network's confidence in its own guess is not the same thing** and is a weaker
   cue. Surprise needs the outcome.
3. **Top-down feedback helps a little and can run away.** Letting recognized words
   trigger cuts snowballed into letter-by-letter segmentation when 1–2 letter chunks
   were allowed: frequent short "words" make more short cuts, which make them more frequent.
   Restricting to 3+ letters gives +2 F1. This is a positive-feedback hazard to remember
   for any self-training loop.
4. Segmentation is still far from adult level (Bayesian unigram/bigram models report
   token F1 of about 50–70 on child-directed speech:
   [Goldwater et al. 2009](../related-work.md#word-segmentation)). Later stages
   therefore use the true word tokens.
