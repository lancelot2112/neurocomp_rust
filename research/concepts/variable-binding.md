# Variable binding: the gap to transformers

**Observation** ([06B](../experiments/06-meaning.md)): after reading 3,000 stories like
`mary went to the kitchen . where is mary ? kitchen`, the network answers 100% of
questions about name/place pairs it has seen and **0%** about pairs it hasn't. It never
learned the *rule* "answer with the place that followed this name". It learned a table
of (name, place) combinations.

## Why the current network can't do it
A kernel's input mask is a sample of *specific* bits: "mary-bits in frame 9 AND
kitchen-bits in frame 5 → kitchen". Nothing in the kernel says "whatever was in frame
5". To answer about a new pair it would need to:
1. **bind** a role filler to a value while reading (mary ↦ kitchen), and
2. **retrieve by content** at question time (look up mary, return what is bound to it).

## How transformers do it
Attention is content-addressed retrieval. The question token's *query* matches the
*key* of the earlier "mary" position, and that position's *value* (carrying "kitchen",
via the previous-token mixing that induction heads rely on) is copied to the output
([Vaswani et al. 2017; Olsson et al. 2022](../related-work.md#binding-and-attention)).
Attention over a context is equivalent to one step of a modern Hopfield
associative memory ([Ramsauer et al. 2020](../related-work.md#binding-and-attention)).

## Candidate mechanisms for this network (not yet tried)
- **Fast one-shot binding memory.** A second predictive class whose kernels are grown
  *during reading* (not training) from "name code → place code" and recycled within
  the story. This is the fast-weights idea
  ([Ba et al. 2016](../related-work.md#binding-and-attention)). It needs a way to decide
  what to bind, e.g. syntax categories from [05](../experiments/05-syntax.md)
  (NOUN-like slot after "went to the").
- **Binding by superposition.** Hyperdimensional binding (XOR/permutation of codes,
  [Kanerva 2009; Plate 1995](../related-work.md#sparse-distributed-representations))
  stores `name ⊗ place` in one vector. Unbinding with `name` recovers `place`. XOR is
  already a `KernelOp`.
- **Copy kernels.** Kernels whose output is *taken from an input frame* instead of
  fixed bits: "output whatever was in frame k". That is a minimal pointer and the
  operation an induction head performs.

This is the most important open problem for "reading like a transformer"; see
[open questions](../open-questions.md).
