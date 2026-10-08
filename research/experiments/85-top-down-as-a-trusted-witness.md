# 85 · The top-down frame as a trusted witness

**Question.** The higher area reaches the column by two paths. Its predicted word votes in
the thalamic mix (`SourceMix`), weighted by how often that source has been right in this
context: an implicit trust in the brain's own parts, unlike the Bayes module's trust in
external sources. Its top-down frame also goes straight into the column's L4 input, with
no check at all. Should that frame pass only as far as the thalamus trusts the area, with
the area itself still learning on every step, and the gate opening as it grows more
reliable?

**Code.** [`examples/episodic.rs`](../../examples/episodic.rs), `HIER_TRUST_GATE`:
- **The record.** On every training word the thalamus scores the area's proposed words
  right or wrong, per (previous word, current word, the area's confidence band): the mix's
  own measure, under a context known before the frame is placed (the mix's own context also
  holds a familiarity band computed later in the step). `HIER_TRUST_GATE=column` also keeps
  the column's own record in the same contexts.
- **The gate.** `=1`: the frame enters L4 where the area's record is at least one half
  (right more often than not). `=column`: where it is at least the column's own record. An
  area never scored in a context passes, so the record can form.
- The area still predicts, learns and votes in the mix every step. Only its frame into L4
  is gated. `TRUSTGATE` report: share of steps the frame passed.

## Results

The frame passes at 19–38% of steps (about 25% on most entries; 6% of test steps on slot
memory). The area is trained on the column's residual, so its word is usually wrong.

Seed 0, held-out (no gate / at least one half / at least the column's record): slot memory
19.4 / 70.4 / 64.6; story boundary 98.4 / 96.4 / 96.8; motor speech 64.8 / 61.6 / 60.2;
inference replay 62.2 / 59.6 / 59.2; cooperation 60.4 / 39.8 / 52.0; belief 83.4 / 78.4 /
80.6.

Three-seed suite, `HIER_TRUST_GATE=1` (held-out, recorded → gated):

| Gains | | Losses | |
|---|---|---|---|
| slot memory | 16.3 → **71.8** | sleep generalisation | 74.5 → 54.4 |
| engram walk only | 84.0 → **91.5** | hippocampus teaches cortex | 61.9 → 45.2 |
| inference replay | 53.4 → **60.0** | cooperate | 56.8 → 41.1 |
| | | index hippocampus | 66.3 → 51.1 |
| | | family consolidated | 53.9 → 40.8 |
| | | relations, speak, motor speech | 63.5 → 52.4–53.3 |
| | | full hippocampus | 41.4 → 31.6 |
| | | schema advantage | 65.7 → 56.3 |
| | | story boundary | 94.1 → 89.1 (seed 1: 73.2) |

Saccades, role transfer, inference read, engram walk and belief are within 4 points.
**Seen pairs rise** on the memory entries (about 63 → 78–83%) while held-out falls.

## Findings
1. **The thalamus does hold an implicit trust in each internal source,** and it adapts:
   each source's weight follows its record per context. Gating the frame by that record is
   cheap and needs no new learning machinery.
2. **But the frame is not a witness to the next word.** Judged by whether its word comes
   true, the area is mostly wrong (it learns only where the column fails), so the gate
   passes about a quarter of frames. What the column takes from the frame is context: a
   feature it learns to read, often useful where the area's own word is wrong. Withholding
   it costs most on held-out names (−10 to −20 points), and the column fits the training
   stories better without it (seen up 15–20 points): the frame was helping it generalise.
3. **This is experiment 24's finding with a finer context.** A per-word reliability gate
   closed the channel there; per (previous word, current word, confidence) it stays open a
   quarter of the time, which is still too little.
4. **Where the frame is noise, any gate helps:** slot memory +55, engram walk only +8,
   inference replay +7, as with the hand rule and the learned gate (83).
5. Not a default. The trust that should gate a frame is trust in the frame's use: whether
   the column's prediction with it beats the prediction without it, the counterfactual of
   [83](83-compute-only-where-needed.md), kept by the thalamus per context rather than
   learned by a go/no-go.

## Next
- Gate the frame by its counterfactual record (prediction with the frame against without
  it, one extra look-up) per context, keeping the area learning on every step.
- Scale rather than switch: pass part of the frame (a sampled share of its bits) in
  proportion to trust, so the column still sees a weaker frame where trust is low.
