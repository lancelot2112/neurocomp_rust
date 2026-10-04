# Credit assignment

Deciding which parts of the network deserve credit or blame for an outcome, so that
parts not directly connected to the error signal can still learn the right thing.

## What the network already does (and why it was enough until now)
- **Predictive kernels get exact, local credit.** Each one predicts a target directly
  and is scored hit/miss against it ([surprise-driven growth](surprise-driven-growth.md)).
- **Temporal credit by sampling.** A newly grown kernel samples all active bits across its
  context window, so whatever in the window mattered is included. It is crude: it also
  includes everything incidental.
- **Hebbian layers get no credit at all.** They learn activity statistics
  ([Hebbian mask rule](hebbian-mask-rule.md)). That is fine for word classes
  ([05](../experiments/05-syntax.md)) and fails when a layer must pick *task-relevant*
  features ([08](../experiments/08-credit-assignment.md): activity-driven memory stays at chance).

## What's available now
- `KernelClass::credited_inputs(bits)` / `blamed_inputs(bits)`: the input bits used by
  matching kernels that the last target confirmed or contradicted. A credit signal
  that flows backward along the connections actually used, readable by the layer
  below.
- `GrowthConfig::generalize`: synapse-level credit. A near-matching kernel that would
  have been right drops its silent connections.

## Results so far ([08](../experiments/08-credit-assignment.md))
| Signal | Long-gap task |
|---|---|
| none (fixed random hidden units) | 39% |
| activity (Hebbian-like) | 25% (chance) |
| "used by a correct kernel" (`credited_inputs`) | 30–36% |
| three-factor: eligibility × (reward − baseline) | 40–44% |
| oracle hidden units | 100% |

The gap between three-factor and oracle is the size of the problem still open.

## Why naive credit fails
**Co-activation.** Inputs that are merely present when a correct prediction is made
get the same credit as the input that caused it. Frequent, irrelevant inputs are
present most often, so they win. Fixes need some notion of *counterfactual*
contribution (what if this input had been absent?) or a baseline-corrected,
well-timed reward.

## Candidate operations, from local to global
1. **Three-factor rules** (eligibility trace × neuromodulatory reward − baseline): local,
   biologically plausible, already tried (+4 points).
2. **Counterfactual / ablation credit**: re-evaluate a prediction with one unit's bits
   removed. Local to one class, costs one extra evaluation per unit.
3. **Credit-guided growth**: sample new kernels' inputs according to unit utility.
4. **Feedback alignment / target propagation**: send error signals down through fixed
   random or learned feedback paths (Lillicrap et al. 2016; Bengio 2014). The nearest
   biological stand-ins for backprop.
5. **Backprop through time**: what transformers and LSTMs use. Exact, but non-local.

## Answer to "do we need a credit-assignment operation?"
For one predictive layer, no: its local hit/miss scoring is exact. As soon as a layer's
job is to supply *useful features* to another layer (memory, chunks, categories for
prediction), **yes**. The [08](../experiments/08-credit-assignment.md) experiment
shows local activity statistics choose the wrong features, and naive credit is fooled
by co-activation.

See [related work](../related-work.md#credit-assignment).
