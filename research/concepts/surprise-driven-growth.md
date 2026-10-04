# Surprise-driven growth and recycling

The local learning rule that turned the network into a working sequence learner
([02](../experiments/02-predictive-growth.md), [03](../experiments/03-book-scale-char-prediction.md)).
Implemented in `KernelClass::predictive` / `process_predictive` / `feedback` / `grow` in
[`src/kernel/class.rs`](../../src/kernel/class.rs).

## The idea
A reader that isn't consuming information shows up as **surprise**: the next input
arrives and nothing predicted it. Growth is therefore tied to unpredicted input, not
to raw input activity. Growing on activity alone would memorize noise.

## Rules (per predictive kernel class)
1. **Predict.** Each kernel's input mask is a sample of active bits from the last *d*
   history frames (*d* = its context depth). Candidates are kernels whose matched bits
   reach threshold. The **winner** is the deepest candidate, then the best smoothed hit
   rate `(hits+1)/(fires+2)`, then the most matched bits. Only the winner writes its
   output (its prediction).
2. **Score.** When the actual next input arrives, every candidate (not only the
   winner) gets a hit if the target confirms at least half of its output bits,
   otherwise a miss. Each context therefore keeps a hit rate per continuation.
3. **Grow on surprise.** If the winner was wrong (or nothing matched) and more than
   half the target went unpredicted:
   - grow a **sibling** at the winner's depth that predicts the actual target (unless
     a matching kernel at that depth already does), so frequent continuations can win;
   - if a kernel fired wrongly, also grow one **a frame deeper**, so a longer context
     can override a shorter one that keeps being wrong.
4. **Recycle at budget.** Replace the least-recently-*useful* kernel (oldest last hit).
   An inverted index (input bit → kernels) is kept in sync.
5. **Tolerance.** Threshold = sampled bits − one frame's worth of noise
   (`floor(sample_bits × (1 − match_fraction))`). A threshold proportional to all
   sampled bits let deep kernels fire with a whole wrong frame
   ([03](../experiments/03-book-scale-char-prediction.md)).

## What it amounts to
A variable-order Markov model built on demand: PPM-style "longest matching context
wins" ([Cleary & Witten 1984](../related-work.md#sequence-memory)), with storage spent
only where shorter contexts failed. HTM temporal memory
([Hawkins & Ahmad 2016](../related-work.md#sequence-memory)) is the closest neural
relative. It grows distal dendritic segments from a sample of previously active cells
when a column bursts (is unpredicted).

## Side signals it exposes
- `confidence()`: the winner's hit rate, i.e. how sure the network is of its guess.
- `target_probability()`: the hit rate of the deepest candidate that predicted the
  *actual* target, i.e. how expected the outcome was. Its dips mark word boundaries
  ([04](../experiments/04-word-segmentation.md)). This is far better than `confidence()`
  for that job.

## Known weaknesses
- No prediction at all for a context with no matching kernel. N-gram back-off always
  has some order to fall back to ([03](../experiments/03-book-scale-char-prediction.md),
  [05a](../experiments/05-syntax.md)).
- Kernels match exact sampled bits, so nothing generalizes across *similar* symbols.
  Every context is learned separately.
- Grows on nearly every miss for high-entropy streams (word-level: about 1.3 kernels per token).
