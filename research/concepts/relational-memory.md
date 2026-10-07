# Relational memory: a learned, directed relation graph, and how memory should talk to the cortex

Collected from experiments [63](../experiments/63-routes-cooperate.md)–[67](../experiments/67-relation-store-in-reading.md).

## The three stores, side by side

| | Hippocampus (engram store, [56](../experiments/56-engram-store.md)–[60](../experiments/60-walk-alone.md)) | Semantic store ([42](../experiments/42-semantic-store.md)) | Relation store ([66](../experiments/66-typed-relations.md), [67](../experiments/67-relation-store-in-reading.md)) |
|---|---|---|---|
| Kind | auto + heteroassociative | heteroassociative | heteroassociative |
| Learns | one shot, while reading | slowly, by sleep replay | slowly, by sleep replay |
| A key | an episode's bindings + context | one word | one word, permuted by (relation, direction) |
| An answer | a whole episode | a bag of the fact's other words | the filler alone (one word's code) |
| Chains | episodes (the walk) | – | relations (rules learned at sleep) |
| Plays | hippocampus | anterior temporal hub | neocortex of complementary learning systems |

All three are heteroassociative: a key recalls a different pattern. They differ in what a
key and an answer are, and in how fast they learn.

## The relation store: a generic, directed, labelled graph
[`src/program/relations.rs`](../../src/program/relations.rs), `RelationStore`.
- **Nodes** are words, any words; nothing in it is about families.
- **Edge labels (relations) are learned from sentence shape.** A fact's neighbours are
  the facts of the same length that agree with it in all but at most two positions. The
  positions where most of them agree are its frame ("_ 's father is _", "_ is a _"); the
  rest are its fillers.
  - Word frequency does not work: on the stories, family names are frequent everywhere
    and "is a" occurs only in facts, so frequency took "jones" as the frame.
- **Edges are directed.** A fact stores filler i → filler j for every ordered pair, each
  under its own key: the entity's code rotated by an offset hashed from (relation, i, j).
  Forward and inverse are different keys.
- **Reads:**
  - `ask` gives the strongest answer;
  - `ask_all` gives every answer (one-to-many: the union of every matching kernel);
  - `follow` walks a path;
  - `about` gives everything known about a word;
  - `relation_for` finds the relation a query's words name.
- **Relations of relations are learned at sleep.**
  - A two-step path that reproduces at least three quarters of a relation's stated facts
    (and at least two) becomes a rule (grandfather = father ∘ father).
  - Its inferred facts are replayed into the store like stated ones, so the cortex answers
    what it was never told.
  - **Depth grows one level per night:** great-grandfather needs the grandfathers to be
    consolidated first. Chains of `follow` are exact 7 deep in the tests.
- **Limits** ([67](../experiments/67-relation-store-in-reading.md#limits-of-any-relation)):
  - single-word fillers;
  - a relation is tied to its wording (no one-step "r1 = r2" rules yet);
  - a frame needs two other facts of its shape;
  - rules are two steps over the first two fillers;
  - transitive relations (r = r ∘ r) are not learned: under an open world a direct fact
    counts against them. They need a test by contradiction.

## How memory should talk to the cortex: sparse in content and in time
Every way of letting memory speak at every step diluted the higher area's context:

| How the stores' answers reach the cortex | Bag store | Relation store |
|---|---|---|
| on every surprising word (63) | 47% | – |
| graded: always, in the mix (64) | 47% | 48.5% |
| graded: always, leaking in with the column's doubt (64) | 38% | 49.8% |
| **binary: only where the column is unsure (63, 67)** | **59.5%** | **64.4%** |

(New names, hippocampus lesioned, with 50 readings of replay.)

- **Sparse in content:** a single filler (about 32 bits) instead of a bag of the fact's
  words. This alone lifted graded enrichment from 38 to 50%, and the binary gate from
  59.5 to 64.4%.
- **Sparse in time:** asked only where the column's own prediction is under half
  reliable. That is still about 15 points better than a little everywhere.
- **In the brain:** recall is automatic, but its *use* is gated. Hippocampal output to the
  cortex comes in sharp, sparse events (sharp-wave ripples offline; theta-phase-locked
  bursts online), and cortex-to-cortex messages are sparse spikes, not a constant blend.
- **Content must be readable by the receiver.** Role-bound (rotated) codes given to the
  higher area unread collapse it (59.5 → 27%, [66](../experiments/66-typed-relations.md)):
  rotated bits share nothing with the words it knows. Unbind first, then send words.

## Controlling the cue
- A learned cue controller (basal ganglia choosing as is / walk / content only / focus,
  rewarded by recalling the next word) did worse than the hand-set walk (78.5% vs 92%,
  [65](../experiments/65-cue-controller.md)).
- The next-word reward does not see the few steps where cueing matters. A prefrontal
  controller should react to the first recall (try, read the result, re-cue) and be
  rewarded at the answer.

## Open
- One-step rules that merge relations with different wording; transitive relations.
- A frame word that is also an entity ("jones" in "X is a jones" and "X jones went to the
  Y"), so the cortex can compose lucy → jones → place without the hippocampus.
- The relation store's parse and rules in the genome ([51](../experiments/51-networks-of-kernels.md)).
