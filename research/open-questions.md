# Open questions and next steps

Ordered roughly by how much they matter for "reading like a transformer".

1. **Variable binding / content-addressed retrieval.** The 0% on held-out bAbI-style
   pairs ([06B](experiments/06-meaning.md)) is the main architectural gap. Try fast
   one-shot binding kernels, XOR binding, or copy kernels
   ([variable binding](concepts/variable-binding.md)). Success criterion: held-out-pair
   accuracy well above chance (17%).
2. **Generalization across similar symbols.** Predictive kernels match exact bits, so
   "the cat sat" teaches nothing about "the dog sat". Feed the learned syntactic and
   semantic codes ([05](experiments/05-syntax.md), [06A](experiments/06-meaning.md))
   into the predictor instead of random word codes. Expect next-word accuracy to beat
   the n-gram baselines, which have the same blind spot.
3. **Fallback when nothing matches.** The back-off n-gram always predicts; the network
   sometimes predicts nothing ([03](experiments/03-book-scale-char-prediction.md),
   [05a](experiments/05-syntax.md)). Options: a depth-0 "unigram" kernel per output, or
   lower-tolerance partial matches.
4. **Growth efficiency at word level.** About 1.3 kernels grown per token. Gate growth on
   repeated surprise in the same context (a counter), not every miss.
5. **Better segmentation.** Add a backward predictor (word starts are predictable
   right-to-left), or iterate segmentation → lexicon → re-segmentation over the book.
   Compare against Bayesian models ([Goldwater et al. 2009](related-work.md#word-segmentation)).
6. **One pipeline instead of separate examples.** Letters → words (stage 2) → syntax
   codes (3) → topical codes (4) → predictor, all online in one `RuntimeNetwork`.
7. **Closing the Hebbian vs count-vector gap** (75 vs 82.5% POS, about 58 vs 71% semantic):
   larger masks with more moves, or weighting moves by surprise.
