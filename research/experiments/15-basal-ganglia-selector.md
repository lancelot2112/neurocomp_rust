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
| **Select: basal-ganglia selector** | **90.4%** (1 seed); SELECT_5 (5 seeds) | 90.2% |
| Select with the CA1 comparator ([14](14-ca1-comparator.md)) | SELECT_CA1 | SELECT_CA1_RECALL |

## Findings
1. **Learned values separate entities from verbs.** After training, names have values
   0.08–0.16 and "picked up" 0.00: following a name is what tends to recall the next
   word. Nothing told it what a name is; reward did.
2. **One learned choice beats several unlearned branches** (90% vs 81%): the predictor
   gets one clean frame instead of three to sort out. This is the basal-ganglia
   division of labour: select, then let cortex use the selection.
3. **Remaining error is in hop 2, not in the choice.** Following the right name recalls that
   person's most recent episode, which is sometimes another pick-up rather than their
   last move. Choosing among *episodes* (not just items) would be the next level.
4. The reward is immediate (next word). Delayed choices (holding an item across words)
   need the eligibility trace (`BG_TRACE`), not yet tested.
