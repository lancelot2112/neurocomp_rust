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

## Addendum: trust in the frame's use, and scaled frames

**Code.** `HIER_TRUST_GATE=cf`:
- **The record.** On every training step the column's prediction with the full frame and
  with none are compared (one extra look-up). Where they differ, the thalamus records a fix
  (right only with the frame) or a break (right only without), per (previous word, current
  word, the area's confidence band). Steps where the frame made no difference are not
  evidence.
- **Scaling, not switching.** The frame enters L4 as a fixed subset of its bits (by a hash
  of the bit position, so the column sees a consistent partial frame), the share
  2 × (fixes + 1) / (fixes + breaks + 2) up to all of it. With no record it passes whole, and
  it weakens only where breaks outnumber fixes. The record is always taken on the full
  frame, so a weakened frame can earn its way back.
- The area predicts, learns and votes every step, as before. `TRUSTGATE` report: mean share
  passed and steps scaled down. `HIER_CF_NOSCALE=1` keeps the look-ups and records but
  passes every frame whole (a control).

Also tried, `HIER_UP=surprise`: **only surprisal goes up.** The higher area's sentence frame
holds only the sentence's surprising words and the current word, not the words the column
predicted (its slow state already held only surprising words, and it already learned only
on the column's misses).

**Results.**

Only surprisal up (seed 0, held-out, against ungated): slot memory 19.4 → 78.6, every other
probe lower: motor speech 64.8 → 46.6, belief 83.4 → 62.4, inference replay 62.2 → 43.2,
cooperation 60.4 → 47.0, story boundary 98.4 → 86.4.

The counterfactual gate, three-seed suite (held-out, recorded → gated): the frame is
scaled down on few steps on most entries (mean share passed 99.9%; hippocampus teaches
cortex, seed 0: 175 training steps and 4 test steps), and heavily on the context tasks
(mean share 65–88%: story boundary, saccades, role transfer, sleep generalisation).

| Entry | Recorded | Gated |
|---|---|---|
| story boundary, saccades | 94.1, 96.6 | 94.7, 96.5 |
| role transfer, sleep generalisation | 60.6, 74.5 | 60.7, 73.8 |
| relations, speak, motor speech | 63.5 | **67.3** |
| slot memory | 16.3 | **28.3** |
| inference replay, inference read | 53.4, 44.2 | **59.5, 52.9** |
| family stated | 38.1 | **45.0** |
| belief decides, cooperate, engram walk only | 81.7, 56.8, 84.0 | 80.4, 58.9, 85.3 |
| hippocampus teaches cortex, index hippocampus | 61.9, 66.3 | **53.0, 57.5** |
| full hippocampus, schema advantage | 41.4, 65.7 | **35.5, 58.3** |

The control: with the same look-ups and records but nothing scaled
(`HIER_CF_NOSCALE=1`), hippocampus teaches cortex (seed 0) is identical to the ungated run
(60.6 / 67.4%); with 175 training steps scaled, it is 66.0 / 45.0%.

**Findings.**
1. **Trust in the frame's use is the right measure; trust in its word was not.** Judged by
   whether it changes the column's prediction for the better, the frame is rarely harmful:
   it passes whole almost everywhere, and the context tasks, where it is scaled to 65–88%,
   stay at parity (unlike 85's witness gate, which passed a quarter of frames).
2. **Small interventions move single runs a lot.** Scaling the frame at 0.2% of training
   steps moved one seed by 22 points: kernel growth is path-dependent, so any change early
   in training reshapes what is learned after it. The gated suite's gains (relation entries
   +4, slot memory +12, inference +6–9) and losses (hippocampus entries −6 to −9) are of
   the size this alone can produce on three seeds. The gate is roughly neutral; telling
   small real effects from this needs more seeds (or paired comparisons over many).
3. **The area needs the predicted words too.** Sending up only surprisal helps where the
   area's frame was noise (slot memory) and costs 12–21 points elsewhere. What the column
   predicted still says what the sentence is about; the area's slow state and its learning
   target are already surprisal-only, and that part stays.
4. Neither becomes a default.

## Next
- More seeds for decisions near the noise: the regression suite at 3 seeds cannot separate
  ±5-point effects on the memory entries.
- Learned input routing (roadmap, section 2): every source a relay bound to its source and
  scaled by this counterfactual record, instead of fixed slots.
