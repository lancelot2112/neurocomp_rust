# neurocomp research wiki

A running lab notebook for the question **"can this network learn to read the way a
transformer does?"** Each experiment page records the setup, the numbers, what we
concluded, and what changed in the code because of it. Concept pages collect ideas
that cut across experiments. Add to it as you go: new experiment pages get the next
number, and every claim should point at the code or command that reproduces it.

## Where we are (headline results)

| Stage | Test | Network | Best baseline | Page |
|---|---|---|---|---|
| 0 | Next char, stock network, tiny text | bigram level at best (~61%) | 6-gram 92% | [01](experiments/01-stock-network-reading.md) |
| 1 | Next char, tiny text, surprise-driven growth | 89% seen / 82% held-out | 6-gram 92% / best n-gram 79% held-out | [02](experiments/02-predictive-growth.md) |
| 1 | Next char, *Alice* (142K chars, read once) | 56.9% | back-off n-gram 59.0%, best fixed n-gram 53.0% | [03](experiments/03-book-scale-char-prediction.md) |
| 2 | Word boundaries with no spaces | boundary F1 63.9, token F1 30.8 | transitional probability 55.9 / 21.1 | [04](experiments/04-word-segmentation.md) |
| 3 | Next word, Brown (100K tokens) | 9.1% | back-off n-gram 11.1% | [05](experiments/05-syntax.md) |
| 3 | Part-of-speech induction (1000 words) | 74.9% NN agreement | chance 24.6%, count vectors 82.5% | [05](experiments/05-syntax.md) |
| 4 | Semantic groups (92 words, 12 groups) | 52–60% | chance 7.4%, count vectors 70.7% | [06](experiments/06-meaning.md) |
| 4 | Fact binding, held-out name/place pairs | **0%** (100% on seen pairs) | chance 17% | [06](experiments/06-meaning.md) |
| – | Top-down bias: word layer → char layer (60K chars) | 53.4% (+0.3) | oracle word layer 92.6% | [07](experiments/07-top-down-bias.md) |
| – | Long-gap memory: what should hidden units hold? | ablation credit + guided growth 67% | oracle 100%, Hebbian-like 25% | [08](experiments/08-credit-assignment.md) |
| – | Binding via thalamic relay, held-out pairs | oracle route **100%**; learned (hindsight proposals + gradual generalization) **82%** | no relay 0% | [09](experiments/09-thalamic-attention.md) |
| – | Binding by inhibition over a route pool (1–2 facts) | context gate 100% (original), value gate 100% (long) | 4 channel slots 75% / 44.5% | [10](experiments/10-route-pool-inhibition.md) |
| – | Binding by one-shot episodic memory, variable sentence shapes | **98%** held-out | best fixed routes 33–34% | [11](experiments/11-episodic-memory.md) |
| – | Same, with dentate-gyrus expansion + Hebbian CA3 store | **99%** held-out (1–3 facts) | list memory 97.5% | [12](experiments/12-dentate-gyrus-ca3.md) |
| – | Two-hop questions ("where is the ball?") by big-loop recall | branching recall **80.8%** | single-cue chain 5.5%, one hop 0% | [13](experiments/13-big-loop.md) |
| – | Storing what the predictor didn't predict (CA1 comparator) | 93% / 87% held-out (1 seed) | frequency habituation 98% | [14](experiments/14-ca1-comparator.md) |
| – | Two-hop questions, learned choice of what to follow (basal ganglia) | **89.1%** held-out (5 seeds, bit-sliced counters) | branching recall 80.8% | [15](experiments/15-basal-ganglia-selector.md) |

**Short version.** Local growth rules driven by surprise turn the network into a
competent variable-order sequence memory (comparable to PPM-style n-gram back-off
with ~5x less storage), and the repo's Hebbian mask rule learns usable syntactic and
semantic word categories once the codes are sparse enough. What is missing for
"reading like a transformer" is **variable binding / content-addressed retrieval**:
the network memorizes combinations it has seen and cannot answer about new ones
([concept page](concepts/variable-binding.md)).

**Top-down feedback and credit assignment** ([07](experiments/07-top-down-bias.md),
[08](experiments/08-credit-assignment.md)). Bias from a higher layer works
mechanically: a perfect word layer lifts character prediction from 53% to 93%. But it
only helps as much as the higher layer knows. With today's word layer it adds 0.3
points. A credit-assignment operation *is* needed once a layer must supply useful
features to another. Activity-driven (Hebbian) choice fails, and naive credit is
fooled by co-active inputs. Ablation (counterfactual) credit plus credit-guided growth
gets 67% of the way where the oracle gets 100%.

**Attention via a thalamic relay** ([09](experiments/09-thalamic-attention.md)).
Routing "what followed the earlier occurrence of this word" into the predictor (an
induction head in thalamic form) takes held-out fact binding from 0% to 100%. Learning
*which* route to use is the bottleneck. Hindsight proposals ("which route would have
carried what I failed to predict?") find it, and gradual synapse-level credit ("drop
inputs that keep being irrelevant when you're right") stops the network memorizing
names: 82% on held-out pairs, all learned, with nothing task-specific put in by hand.

## Pages

Experiments (chronological)
1. [Stock network on a reading task](experiments/01-stock-network-reading.md)
2. [Predictive learning with surprise-driven growth](experiments/02-predictive-growth.md)
3. [Book-scale character prediction](experiments/03-book-scale-char-prediction.md)
4. [Word segmentation and recognition without spaces](experiments/04-word-segmentation.md)
5. [Syntax: next-word prediction and part-of-speech induction](experiments/05-syntax.md)
6. [Meaning: topical similarity and fact binding](experiments/06-meaning.md)
7. [Top-down bias from a higher layer](experiments/07-top-down-bias.md)
8. [Credit assignment for a hidden layer](experiments/08-credit-assignment.md)
9. [Thalamus-like relay as attention](experiments/09-thalamic-attention.md)
10. [Attention by inhibition: a route pool with learned gating](experiments/10-route-pool-inhibition.md)
11. [Episodic autoassociative memory](experiments/11-episodic-memory.md)
12. [Dentate gyrus expansion and a Hebbian CA3 store](experiments/12-dentate-gyrus-ca3.md)
13. [Big-loop recurrence: chaining recalls for two-hop questions](experiments/13-big-loop.md)
14. [A CA1-style comparator: store what wasn't predicted](experiments/14-ca1-comparator.md)
15. [A basal-ganglia selector: learning which recalled item to follow](experiments/15-basal-ganglia-selector.md)

Concepts
- [Surprise-driven growth and recycling](concepts/surprise-driven-growth.md)
- [The Hebbian mask rule](concepts/hebbian-mask-rule.md)
- [Sparse codes and collisions](concepts/sparse-codes-and-collisions.md)
- [Variable binding: the gap to transformers](concepts/variable-binding.md)
- [Top-down bias](concepts/top-down-bias.md)
- [Credit assignment](concepts/credit-assignment.md)
- [Hippocampal-formation functions: what we have and what's missing](concepts/hippocampal-functions.md)
- [Basal ganglia and cerebellum: two more credit-assignment loops](concepts/basal-ganglia-and-cerebellum.md)
- [Probability in bits: what is bitwise now, and how to keep it that way](concepts/probability-in-bits.md)

Reference
- [Bugs found and fixed](bugs-and-fixes.md)
- [Related work and reading list](related-work.md)
- [Open questions and next steps](open-questions.md)

## Reproducing

```sh
./scripts/fetch_corpora.sh                       # Alice, Bryant stories, tagged Brown -> data/
cargo test                                       # unit tests (kernels, growth, runtime)
cargo run --release --example read_text          # experiments 01-02 (seconds)
cargo run --release --example read_corpus        # experiment 03 (~12 min for all configs)
cargo run --release --example segment_words      # experiment 04 (~3 min)
cargo run --release --example syntax             # experiment 05 (~2 min)
cargo run --release --example meaning            # experiment 06 (~2 min)
cargo run --release --example topdown            # experiment 07 (~4 min)
cargo run --release --example credit             # experiment 08 (~15 min for all configs; pass 20000 for the long run)
cargo run --release --example thalamus           # experiments 09-10 (~15 min per policy set; see POLICIES)
cargo run --release --example episodic           # experiments 11-14 (~35 min; POLICIES=ca3 for 12; TASK=twohop POLICIES=loop for 13; NOVELTY=prediction for 14)
```

Learning uses `rand::thread_rng()` inside the kernels, so numbers move by a point or
two between runs; the pages quote single runs unless noted.
