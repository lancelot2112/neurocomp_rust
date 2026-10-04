# The Hebbian mask rule

The repo's original local learning rule, in `SimpleKernel::strengthen`
([`src/kernel/simple.rs`](../../src/kernel/simple.rs)) via
`BitVector::mask_move_random_connected_and_not_set`
([`src/bitvec/bitmask.rs`](../../src/bitvec/bitmask.rs)):

> When a kernel fires, take one connection whose input bit is **silent** and move it to
> a random **active** bit it is not yet connected to.

The opposite move (`search_remap`, used when a kernel is inhibited) moves a connection
*away* from the active pattern.

## Behaviour
- The number of connections is fixed (the mask size). Repeated firing on a context
  distribution pulls the mask toward bits that are frequently active in that
  distribution. The stationary mask is a small **stochastic sample** of the context
  distribution, weighted toward frequent context.
- One move per firing is a learning rate. Words seen a few hundred times barely leave
  their random start with large masks. Several moves per occurrence fixed this for
  topical similarity (25% → 52–60%, [06](../experiments/06-meaning.md)).

## Where it worked
- **Part-of-speech induction**: one kernel per word over [prev | next] word codes →
  74.9% nearest-neighbor tag agreement ([05](../experiments/05-syntax.md)).
- **Topical semantic groups**: 52–60% vs 7% chance ([06](../experiments/06-meaning.md)).

## Where it didn't
- As the *only* learning signal for reading ([01](../experiments/01-stock-network-reading.md)):
  firing reinforces firing regardless of usefulness, and hidden states drift. Adding
  a target ([surprise-driven growth](surprise-driven-growth.md)) was necessary.
- Activity balancing (homeostasis) on Hebbian layers produced more distinct but
  drifting states, and probe accuracy dropped.

## Relation to other work
This is competitive Hebbian learning on binary synapses, close to HTM's spatial-pooler
permanence updates and to Random Indexing
([Sahlgren 2005](../related-work.md#distributional-syntax-and-semantics)) when the
context codes are random sparse vectors. Mask overlap is then roughly a sampled
dot-product of context-count vectors. Quality depends heavily on code sparsity:
[sparse codes and collisions](sparse-codes-and-collisions.md).
