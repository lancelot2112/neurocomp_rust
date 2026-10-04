# Basal ganglia and cerebellum: two more credit-assignment loops

A common division of labour ([Doya 2000](../related-work.md#credit-assignment)):
**cortex** learns representations without supervision, the **cerebellum** learns from
*errors* (supervised), and the **basal ganglia** learn from *reward* (reinforcement).
Each closes a loop with cortex through the thalamus. This page maps both onto what we
have and on which bottleneck each would help.

## Basal ganglia: selection, with delayed reward
- **Circuit:** cortex → striatum → GPi/SNr, which tonically *inhibits* thalamus;
  striatal "go" cells release one channel by disinhibition. Dopamine carries a reward
  prediction error; synapses that were eligible (an eligibility trace) when the error
  arrives are changed: a three-factor rule
  ([Frémaux & Gerstner 2016; Gerstner et al. 2018](../related-work.md#credit-assignment)).
- **We already have half of one.** The inhibition-gated route pool of
  [10](../experiments/10-route-pool-inhibition.md) *is* a basal-ganglia-like selector:
  per-route reliability ≈ striatal action values, winner-take-one release ≈
  disinhibition of one thalamic channel. What it lacks is **delay**: it is scored on the
  very next word.
- **Where that matters now:** choices whose payoff comes later. Which recalled item to
  follow ([13](../experiments/13-big-loop.md)), how many hops, whether to store an
  episode, which route to open several words before the answer. Those are exactly the
  "learned hops" of [hippocampal functions](hippocampal-functions.md#can-hops-be-learned).
  Needed: an eligibility trace per choice (decaying tag on the chosen route/item),
  and a reward-prediction error at the answer (or at any well-predicted word).
- **Gating memory, too:** in PBWM ([O'Reilly & Frank 2006](../related-work.md#credit-assignment))
  the basal ganglia learn *when to update* working memory, i.e. what gets stored or
  held. That is a learned version of the CA1 storage threshold in
  [14](../experiments/14-ca1-comparator.md).

## Cerebellum: error-driven prediction
- **Circuit:** mossy fibres → a huge granule-cell expansion (like the dentate gyrus) →
  Purkinje cells, whose parallel-fibre synapses are depressed when a climbing fibre
  signals an error ([Marr 1969; Albus 1971; Ito](../related-work.md#credit-assignment)).
  Each Purkinje cell gets *its own* error signal: vector, not scalar, credit.
- **Our predictor is already cerebellum-like.** A `KernelClass` is a sparse expansion of
  conjunctions trained by the per-output target (grow on surprise, feedback on hits and
  misses). It does not need backprop because each output has its own teacher.
- **Forward models and cancellation:** the cerebellum predicts the sensory consequences
  of actions and subtracts them (e.g. the electric fish's cerebellum-like
  electrosensory lobe, [Bell et al. 1997](../related-work.md#credit-assignment)). That
  is precisely the comparator we built for memory in [14](../experiments/14-ca1-comparator.md):
  store what was *not* predicted. A better forward model gives a better comparator.
- **Where it would help:** where errors exist for every output: the hidden-layer credit
  problem of [08](../experiments/08-credit-assignment.md) (climbing-fibre-style
  per-unit targets, i.e. target propagation), and fast, timed prediction (the
  thalamic routes as learned delays).

## Which one first?
The basal ganglia. Our open problems now are discrete choices with delayed outcomes
(which item to follow, how many hops, what to store), and the route pool is already the
skeleton: add eligibility traces and a reward-prediction error. The cerebellum would
mostly improve the predictor itself, which matters because the CA1 comparator is only
as good as the prediction it compares against.
