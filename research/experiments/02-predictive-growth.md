# 02 · Predictive learning with surprise-driven growth

**Question.** If the output is trained to predict the next input, and kernels are grown
where nobody predicted the input, does the network learn to read?

**Code.** Mechanism in [`src/kernel/class.rs`](../../src/kernel/class.rs)
(`KernelClass::predictive`, `feedback`, `grow`) and
[`src/program/neurocomp.rs`](../../src/program/neurocomp.rs) (`input_frames`,
`RuntimeNetwork::learn`). Experiment: network D in
[`examples/read_text.rs`](../../examples/read_text.rs).
Commits `e89b76d` (first version) and `ac60b9e` (rework, see experiment 03).

## Setup
Same toy text and protocol as [01](01-stock-network-reading.md). Network D is
`input → output` where the edge reads the last *k* input frames concatenated. The
output is decoded by overlap with the character codes. No probe is needed.

## Results

| Model | Training text | Held-out sentence |
|---|---|---|
| 1-char / 3-char / 6-char n-gram | 61.6 / 85.2 / 92.0% | 63.9 / 79.2 / 65.3% |
| D, 1-frame window | 60.8% | 65.3% |
| D, 3-frame window | 84.0% | 81.9% |
| D, 6-frame window (154 kernels) | 89.4% | 81.9% |

(Numbers after the experiment-03 rework of the growth rule. The first version gave
89.4% / 81.9% for 6 frames but only 54% / 57% for 1 frame.)

## Findings
1. **Surprise-driven growth works.** With 6 frames of context, the network matches the
   6-gram on seen text and **beats every n-gram on the held-out sentence**, because it
   falls back to shorter contexts when a long one is unfamiliar. That is PPM-style
   back-off, discovered by a local rule ([related work: PPM](../related-work.md#sequence-memory)).
2. **Only contexts that were needed get stored.** 154 kernels at depths
   `[23, 40, 36, 19, 18, 18]`: deep kernels only exist where a shallower one was wrong.
3. **"Latest answer" is a poor rule.** The first version made a kernel switch to the
   most recent target after it missed. That cost a 1-frame window about 7 points
   versus a bigram. The fix, sibling kernels per continuation plus hit-rate
   tie-breaking, is described in [03](03-book-scale-char-prediction.md).
4. **Too small a budget thrashes**: at 64 kernels with LRU recycling, accuracy fell to
   about 55%.

## Mechanism summary
See [surprise-driven growth](../concepts/surprise-driven-growth.md) for the full rule set.
