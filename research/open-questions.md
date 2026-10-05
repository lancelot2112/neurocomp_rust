# Open questions and next steps

Ordered roughly by how much they matter for "reading like a transformer".

## Current (after [23](experiments/23-compaction.md))
- **Remaining per-word cost.** After event-based recall
  ([23](experiments/23-compaction.md#5-event-based-hippocampal-recall)), L2/3 matching is
  25–50% and learning 24–38%; kernels are now stored sparsely
  ([23](experiments/23-compaction.md#6-sparse-kernel-storage)). Whole-input memoisation is built
  ([23 §7](experiments/23-compaction.md#7-memoised-interpretation-after-hashlife)): exact,
  but it hits only 4–21%. The per-frame memo is exact but not a net
  win ([23 §8](experiments/23-compaction.md#8-per-frame-memo-hashlifes-sub-nodes-exact-but-not-a-net-win));
  eager per-kernel updates might fix that. Canonical kernels (hash-consing) are built and
  a clear win ([23 §9](experiments/23-compaction.md#9-canonical-kernels-hashlifes-hash-consed-nodes)).
  Chunking is built ([23 §10](experiments/23-compaction.md#10-chunking-hashlifes-time-skipping)):
  a small speed-for-accuracy trade; habits-only chunks keep accuracy but lose the gain.
  Skipping recall and frame assembly within a chunk keeps accuracy and speeds answering
  up to 35%, limited by how much of the text is habit (1–24% here). Also the dense
  8,192-bit frame copies L4 still builds every word.
- **Why the surprise gate leaves more kernels on elimination** (2,909 against 1,193).
- **Sparse mask storage.** Kernel masks are stored at full input width: 5–50 MB where
  0.3–2.3 MB is needed.
- **Several columns, a hierarchy, and real text.** Everything runs through one column on
  templated stories. A second area reading the first's L5 output through a thalamic
  relay is the start of a hierarchy, and the step toward reading a book again with
  episodic memory, gating and compaction ([comparison](concepts/brain-transformer-comparison.md)).
- **Basal ganglia and L6 together** on the same relays (select by reward among what L6
  lets through).
- **Two-hop's weak seed.** The selector picks well (answer in recall 86–89%), but the
  predictor misses on seed 2 (72–77%).

## Earlier list (items 1, 4 and parts of 7 have since been addressed: held-out binding
[11](experiments/11-episodic-memory.md)–[21](experiments/21-several-routes.md), growth
gating [23](experiments/23-compaction.md))

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
7. **Causal credit for hidden layers.** In the long-gap task
   ([08](experiments/08-credit-assignment.md)), three-factor credit reaches 44% against
   100% for an oracle. Try ablation credit, surprise-timed eligibility, and
   credit-guided growth ([credit assignment](concepts/credit-assignment.md)).
8. **A better higher layer to make top-down pay off.** Top-down bias is worth up to
   +40 points with a perfect word layer and +0.3 with today's
   ([07](experiments/07-top-down-bias.md)). Items 1–3 above are what would improve
   the word layer.
9. **Closing the Hebbian vs count-vector gap** (75 vs 82.5% POS, about 58 vs 71% semantic):
   larger masks with more moves, or weighting moves by surprise.
