# 15 · A basal-ganglia selector: learning which recalled item to follow

**Question.** In [13](13-big-loop.md), two-hop questions failed when the loop followed
the *rarest* recalled item (5.5%) and worked when three branches were kept and the
predictor chose (80.8%). Can a reward-learned selector pick the one item to follow?

**Code.** [`src/program/basal_ganglia.rs`](../../src/program/basal_ganglia.rs)
(`BasalGanglia`); `Select` policy in [`examples/episodic.rs`](../../examples/episodic.rs).
Run: `TASK=twohop POLICIES=select cargo run --release --example episodic`
(`BG_TRACE` = eligibility-trace decay, default 0).

## Mechanism
See [basal ganglia and cerebellum](../concepts/basal-ganglia-and-cerebellum.md).
- **Candidates:** hop 1's recalled content split into word-like items (`items`).
- **Striatum:** a "go" weight per input bit (initially 0.5). A candidate's value is the
  mean weight over its bits.
- **Selection (disinhibition):** the best candidate is released; during training a random
  one with probability 0.1 (exploration). Only the released item cues hop 2.
- **Dopamine:** reward = 1 if hop 2's recall contains the next word, else 0. Weights on
  the chosen item's bits move by 0.1 × (reward − value) × eligibility (a three-factor
  rule). Learning is off at test, like the predictor's.
- The predictor gets two frames: hop 1 and the selected hop 2.

## Results (two-hop stories, held-out (object, place) answers)

| Policy | Held-out | Answer in recall |
|---|---|---|
| Loop(2): follow the rarest item ([13](13-big-loop.md)) | 5.5% (5 seeds) | – |
| Branch(3): 3 branches, predictor chooses ([13](13-big-loop.md)) | 80.8% (5 seeds) | – |
| **Select: basal-ganglia selector** | **84.5%** (5 seeds: 90/84/90/84/74) | 84.1% |
| Select, **bit-sliced counters** (4 planes, stochastic steps) | **89.1%** (5 seeds: 86/90/86/92/90) | 88.8% |
| Select, bit-sliced, **with the copy-credit predictor fixes** of [12](12-dentate-gyrus-ca3.md) | 81.5% (5 seeds: 90/79/70/81/87) | 87–91% |
| Select with the CA1 comparator ([14](14-ca1-comparator.md)) | 23.6% (1 seed) | 55.0% |

## Findings
1. **Learned values separate entities from verbs.** After training, names have values
   0.08–0.16 and "picked up" 0.00: following a name is what tends to recall the next
   word. Nothing told it what a name is; reward did.
2. **One learned choice beats several unlearned branches** (84.5% vs 80.8% over 5 seeds,
   though seeds vary from 74% to 90%): the predictor
   gets one clean frame instead of three to sort out. This is the basal-ganglia
   division of labour: select, then let cortex use the selection.
3. **Remaining error is in hop 2, not in the choice.** Following the right name recalls that
   person's most recent episode, which is sometimes another pick-up rather than their
   last move. Choosing among *episodes* (not just items) would be the next level.
4. **The selector does not rescue the CA1 comparator** (23.6%, vs 25.8% for Branch(3)
   with it). Under prediction-error storage the answer reaches the recalled frames only
   55% of the time, so the failure is upstream of the choice: what is stored and what
   cues hop 1, not which item hop 2 follows.
5. **The selector works in bits.** With go values in 4-plane bit-sliced counters
   (`SlicedCounter`), integer comparisons, bit-mask eligibility and stochastic ±1 steps
   (probability 1.5 × |error|), it scores **89.1%** over 5 seeds (86–92) vs 84.5% (74–90)
   for the float version: no worse, and steadier across seeds. Saturating 16-level
   counters bound how far any value can run away, and stochastic steps act like a
   small, noisy learning rate. See [probability in bits](../concepts/probability-in-bits.md).
6. **The copy-credit predictor fixes do not carry over to two hops.** With the settings that
   took one-hop CA3 to 99–100% (`COPY_GROW GROW_TRUST STICKY=255 GENERALIZE_AFTER=1
   TRUST_AT_TEST`), Select drops from 89.1% to 81.5% and Branch(3) from 80.8% to 65.9%
   ([13](13-big-loop.md)), with the answer still in the recalled frames 87–91% of the time.
   Likely cause (not yet diagnosed): the fixes assume one memory frame that carries the
   answer; with two hop frames a place appears in hop 1 *and* hop 2, so copy credit tags
   and protects kernels keyed on the wrong frame. Copy credit needs to know which frame
   the answer came from (e.g. tag only the frame the winning kernel read).
7. The reward is immediate (next word). Delayed choices (holding an item across words)
   need the eligibility trace (`BG_TRACE`), not yet tested.
