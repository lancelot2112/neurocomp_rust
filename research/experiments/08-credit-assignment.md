# 08 · Credit assignment for a hidden layer

**Question.** Do we need a credit-assignment operation, or are local rules enough?

**Code.** `credited_inputs`, `blamed_inputs` and `GrowthConfig::generalize` in
[`src/kernel/class.rs`](../../src/kernel/class.rs); experiment
[`examples/credit.rs`](../../examples/credit.rs) (commit `c024677`).
Run: `cargo run --release --example credit [episodes]` (~2.5 min at 6,000 episodes).

## Setup
- **Task (long gap):** `cue f f f … f ? answer`, cue ∈ {A,B,C,D}, 3–6 fillers from
  12, answer = f(cue). Scored on the answer, over the last 1,000 episodes, 5 seeds.
- **Predictor:** a predictive class that sees the current and previous symbol, plus a
  hidden frame. It cannot see the cue directly.
- **Hidden layer:** K = 4 persistent memory units. Each watches one symbol and, once
  it sees it, stays on for 8 ticks. There are 21 symbols and 4 units, so something
  must decide **what is worth remembering**. Every 100 episodes the worst unit is
  re-pointed according to a policy:
  - **fixed random:** never re-point;
  - **activity** (Hebbian-like): keep the most active units, re-point the least
    active to the most frequent unwatched symbol;
  - **credit:** a unit scores when a kernel the target confirmed used its bits
    (`credited_inputs`); re-point the least credited to a random symbol;
  - **three-factor:** a unit scores (reward − running baseline) on every tick it is
    active; reward = the prediction was right;
  - **oracle:** the units watch exactly the four cues.
- Also crossed with **synapse-level credit** inside the predictor: a kernel that
  nearly matched (≥ 0.5 or ≥ 0.3 of its connections) and would have been right drops
  its silent connections.

## Results (answer accuracy, chance 25%)

| Policy | 6K episodes | 20K episodes | + synapse rule 0.5 | + synapse rule 0.3 |
|---|---|---|---|---|
| oracle (units = cues) | **100%** | 100% | 100% | 100% |
| fixed random | 39.3% | 39.3% | 39.7% | 39.5% |
| activity (Hebbian-like) | **24.9%** | 25.6% | 25.4% | 25.5% |
| credit ("used by a correct kernel") | 36.2% | 34.3% | 29.3% | 30.4% |
| three-factor (reward − baseline) | **43.3%** | 44.1% | 40.9% | 40.1% |

Per-seed results are bimodal (≈25 or ≈50%): a run either keeps a cue unit or two, or none.

## What went wrong with naive credit (trace)
Credit per unit during a run, before reassignment:
`e2499: cueD=164 f0=216 f6=231 f2=70`. **Filler units earned more credit than the
cue unit.** A kernel grown at `?` samples *all* active hidden bits, so its mask
includes cue bits *and* whatever filler units happen to be on. When it is right,
every unit in its mask is credited. Frequent fillers co-occur with every cue's
episodes, so they soak up credit from all four. Being *present* when things went right
is not the same as *causing* them to go right.

## Findings
1. **Yes, a credit-assignment operation is needed.** With the right hidden units the
   task is solved (100%). Activity-driven allocation, the network's existing
   local logic, fills memory with frequent irrelevant symbols and stays at chance.
2. **Credit has to be causal, not correlational.** Crediting every input a correct
   kernel used is worse than random. The co-activation confound above is the textbook
   credit-assignment problem
   ([Lillicrap et al. 2020](../related-work.md#credit-assignment)).
3. **A three-factor rule (eligibility × reward − baseline) is the best local option
   tried** (+4 over random) but plateaus far from the oracle. The reward is diluted
   over every tick a unit is on, and random re-pointing explores slowly.
4. **Credit is needed at two levels, and they interact.** At the unit level: what to
   keep. At the synapse level: which inputs a kernel should require. A cue unit only
   earns reward once some kernel uses it *without* also requiring the incidental
   fillers. The simple synapse-level rule tried here did not trigger usefully.

## Next things to try
- **Counterfactual (ablation) credit:** re-run the winning prediction with one unit's
  bits removed. Credit the unit only if the prediction flips from right to wrong.
  This handles co-activation directly.
- **Eligibility traces timed to surprise:** reward only on ticks where the prediction
  beat a lower-depth kernel, not on every active tick
  ([Gerstner et al. 2018](../related-work.md#credit-assignment)).
- **Credit-guided growth:** when growing a kernel, sample hidden bits in proportion
  to unit utility instead of uniformly, so new kernels avoid incidental inputs.

See [credit assignment](../concepts/credit-assignment.md).
