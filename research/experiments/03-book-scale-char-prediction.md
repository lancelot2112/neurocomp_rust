# 03 · Book-scale character prediction

**Question.** Does the predictive network hold up on a real book, read once?

**Code.** [`examples/read_corpus.rs`](../../examples/read_corpus.rs) (commit `ac60b9e`); growth-rule
rework in [`src/kernel/class.rs`](../../src/kernel/class.rs).
Run: `./scripts/fetch_corpora.sh && cargo run --release --example read_corpus [path] [max_chars]`.

## Setup
- *Alice's Adventures in Wonderland* (Project Gutenberg via NLTK), lowercased and
  reduced to letters, space and basic punctuation: 142,205 characters, 36 symbols.
- Read **once**. Every prediction is made before the character is seen, then the
  model learns from it (prequential / online evaluation). Baselines use the same
  protocol.
- Baselines: fixed n-grams (1–8) and a back-off n-gram that uses the longest context
  seen so far and predicts its most frequent continuation.

## Changes needed to scale (and why)
1. **Inverted index** (input bit → kernels). Each step costs per active bit instead of
   per kernel. Without it the run was impractically slow at about 100K kernels.
2. **Frequency over recency.** Every matching kernel is scored each step. A wrong winner
   grows a *sibling* kernel at its depth that predicts the actual target, plus one a
   frame deeper. The winner is the longest matching context, then the best hit rate,
   so each context predicts its most frequent continuation.
   (Choosing by hit rate first was tried: 49.4% vs 49.7% on 30K characters, so rejected.)
3. **Per-frame match tolerance.** The threshold was 80% of *all* sampled bits, which
   let a 6-frame kernel fire when an entire frame was a different character. It is now
   sampled bits minus one frame's worth of noise. On the toy text, 6-frame accuracy
   went from 85.6% back to 89.4%.

## Results (full book)

| Model | Accuracy | Storage |
|---|---|---|
| 1-char n-gram | 29.3% | |
| 2-char n-gram | 43.6% | |
| 3-char n-gram | 52.3% | |
| 4-char n-gram | 53.0% (best fixed) | |
| 8-char n-gram | 27.4% | |
| back-off n-gram (≤8) | **59.0%** | 350K context→char entries |
| network, 4 frames | 55.9% | 36K kernels |
| network, 6 frames | **56.9%** | 74K kernels |
| network, 8 frames | 56.9% | 96K kernels, 227 s |
| network, 8 frames, 50K budget | 56.7% | 50K kernels (47K recycled) |
| network, 8 frames, 10K budget | 52.6% | 10K kernels (105K recycled) |

Learning curve (accuracy per tenth of the book, 6 frames):
`46.9 52.4 54.6 57.8 57.5 58.6 60.0 60.3 60.8 59.8`.

## Findings
1. The network beats every fixed-order model and trails the back-off n-gram by 2
   points, with about 1/5 of the storage.
2. Recycling now **degrades gracefully**: halving storage to 50K costs 0.2 points.
   Compare the collapse on the toy text in [02](02-predictive-growth.md).
3. The remaining gap is probably the first time a context is seen. The back-off model
   always has *some* order with counts. The network only predicts when a kernel's
   sampled bits match.

## Related
PPM ([Cleary & Witten 1984](../related-work.md#sequence-memory)) is the
classical version of "longest matching context wins". HTM temporal memory
([Hawkins & Ahmad 2016](../related-work.md#sequence-memory)) grows dendritic segments
on unpredicted input in the same spirit.
