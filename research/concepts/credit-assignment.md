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
- `KernelClass::peek(input)`: predict without changing any state. This is the
  building block for **ablation credit**: compare the prediction with and without a
  unit's bits.
- `KernelClass::set_growth_mask(mask)`: restrict which input bits new kernels may
  sample, for **credit-guided growth**.

## Results so far ([08](../experiments/08-credit-assignment.md))
| Signal | Long-gap task |
|---|---|
| none (fixed random hidden units) | 39% |
| activity (Hebbian-like) | 25% (chance) |
| "used by a correct kernel" (`credited_inputs`) | 30–36% |
| three-factor: eligibility × (reward − baseline) | 40–44% |
| **ablation** (would the answer change without this unit?) | 54.6% |
| **ablation + credit-guided growth** (each new kernel uses one unit, picked by credit) | **64.4%** |
| oracle hidden units | 100% |

Ablation credit plus credit-guided growth closes about half the gap between random and oracle.

## Why naive credit fails (and what fixed most of it)
**Co-activation.** Inputs that are merely present when a correct prediction is made
get the same credit as the input that caused it. Frequent, irrelevant inputs are
present most often, so they win. Counterfactual credit asks the right question ("what
if this input had been absent?"), but only once the kernels themselves don't depend
on incidental inputs. Hence credit-guided growth
([08 follow-up](../experiments/08-credit-assignment.md#follow-up-ablation-counterfactual-credit)).

## Candidate operations, from local to global
1. **Three-factor rules** (eligibility trace × neuromodulatory reward − baseline): local,
   biologically plausible, tried (+4 points over random).
2. **Counterfactual / ablation credit**: re-evaluate a prediction with one unit's bits
   removed. Costs one extra read-only evaluation per active unit. Tried: the best
   signal (+15 points over random).
3. **Credit-guided growth**: new kernels sample inputs according to unit credit. Tried
   with ablation: +10 more. It matters because a kernel that *requires* an incidental
   input makes that input genuinely necessary, and then even counterfactual credit
   rewards it. Credit and growth have to be designed together.
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
