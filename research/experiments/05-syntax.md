# 05 · Syntax: next-word prediction and part-of-speech induction

**Question.** Given words, does the network pick up syntax, both as sequential
prediction and as word classes?

**Code.** [`examples/syntax.rs`](../../examples/syntax.rs). Run:
`cargo run --release --example syntax [tokens_for_prediction]` (default 100K; `0` skips part a).

## Setup
- Brown corpus (1,161,192 tokens, NLTK), lowercased, punctuation kept as tokens.
  Vocabulary is the top 5,000 words plus `<unk>` (87.7% token coverage).
- Brown tags collapsed to 12 coarse parts of speech, **used only to score**.

## (a) Next-word prediction (100K tokens, online)

| Model | Accuracy |
|---|---|
| always the most frequent word | 9.8% |
| 2-word / 3-word n-gram (no back-off) | 7.0% / 3.0% |
| back-off n-gram (≤3) | 11.1% |
| network, 3-word context | 9.1% (129K kernels) |

Word-level prediction from 100K tokens is hard for every count-based model. The
network grows a kernel on almost every miss (129K kernels for 100K tokens), and it
has no fallback when no kernel matches. See [open questions](../open-questions.md).

## (b) Part-of-speech induction
One `SimpleKernel` per frequent word, with an input mask over
`[previous word code | next word code]`. Every occurrence fires it, and the repo's
existing Hebbian `strengthen` rule moves one connection onto the active context
([Hebbian mask rule](../concepts/hebbian-mask-rule.md)). Similarity between words is
mask overlap. Metric: does a word's nearest neighbor share its majority tag?
(1,000 most frequent words.)

| Context codes | Mask | NN tag agreement |
|---|---|---|
| chance (random neighbor) | | 24.6% |
| 1024 bits / 32 active | 32 | 35.7% |
| 4096 / 16 | 128 | 51.6% |
| 8192 / 8 | 256 | 65.0% |
| 16384 / 4 | 256 | 72.9% |
| **32768 / 4** | **512** | **74.9%** |
| count vectors + cosine (reference) | | 82.5% |

Per tag at the best setting: NOUN 94%, VERB 85%, PRON 75%, NUM 75%, PUNCT 64%,
ADV 49%, ADP 46%, CONJ 40%, DET 39%, ADJ 35%.

Neighborhoods:
`he → it, they, there, she, we, i` · `said → had, asked, did, found, used, made` ·
`was → is, had, were, must, would, did` · `three → two, other, many, these, one` ·
`very → too, so, more, most, such` · `the → a, an, in, his, this`.

## Findings
1. **The existing local Hebbian rule learns syntactic classes**, but only with sparse
   enough codes. With dense codes, each context bit is shared by about 150 words and
   the masks can't tell contexts apart. This is the biggest single lever
   ([sparse codes and collisions](../concepts/sparse-codes-and-collisions.md)).
2. Weighting shared bits by rarity ("hub suppression", like IDF/PPMI) made things
   *worse* at every size (e.g. 65.0% → 59.0%), so it was dropped.
3. The remaining gap to count vectors (75% vs 82.5%) is plausibly sampling. A mask is
   a small stochastic sample of a word's context distribution.

## Related
Distributional POS induction ([Schütze 1995; Brown et al. 1992; Christodoulopoulos
et al. 2010](../related-work.md#distributional-syntax-and-semantics)). Random
sparse context codes are essentially Random Indexing ([Sahlgren
2005](../related-work.md#distributional-syntax-and-semantics)).
