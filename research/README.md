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
| – | Predictor fixes from diagnostics (copy credit, credit-guided growth, trust before depth) | list memory **100%**; CA3 **99.4–100%** on every seed and load | 98.7% / 90–99% before, seeds down to 75% | [12](experiments/12-dentate-gyrus-ca3.md) |
| – | Two-hop questions ("where is the ball?") by big-loop recall | branching recall **80.8%** | single-cue chain 5.5%, one hop 0% | [13](experiments/13-big-loop.md) |
| – | Storing what the predictor didn't predict (CA1 comparator) | 93% / 87% held-out (1 seed) | frequency habituation 98% | [14](experiments/14-ca1-comparator.md) |
| – | Two-hop questions, learned choice of what to follow (basal ganglia) | **89.1%** held-out (5 seeds, bit-sliced counters) | branching recall 80.8% | [15](experiments/15-basal-ganglia-selector.md) |
| – | Basal ganglia choose among thalamic channels (learned routes + memory recall) | **100%** every seed, memory released at every answer | routes only 36–41% | [16](experiments/16-thalamic-gate-memory-channel.md) |
| – | Facts the hippocampus has overwritten, recalled from cortex after replay (consolidation) | **100%** every seed | hippocampus only 0–37% (guessing) | [17](experiments/17-consolidation.md) |
| – | Small replay budget: replay chosen by questions (tagged or awake) | **100%** every seed | random replay 34–66% | [17](experiments/17-consolidation.md#prioritised-replay-questions-decide-what-is-consolidated) |
| – | Question names no one: recall cued by a working-memory slot, loaded by a learned basal-ganglia gate | **100%** every seed | cue at the question 15–17%; trace-credited gate 16–21% | [18](experiments/18-prefrontal-working-memory.md) |
| – | One shared reward from cortical layer 5 (the column's outcome, credited only if the prediction read the choice) for every selector | gates **100%** every seed; hop-2 selector 91 / 69 / 82% | each selector's own answer key: 100% / 86 / 83 / 86%; unattributed outcome: down to 18% / 32% | [19](experiments/19-l5-shared-reward.md) |
| – | Cortical layer 6 gates the thalamic relays (per context, learned from use, no reward) | **100%** every seed, 0.04–0.19 channels per word, 2.4× faster | all channels open: 100%, 3.5 channels per word | [20](experiments/20-l6-corticothalamic-gating.md) |
| – | Different questions need different relays: L6 opens several routes at once, per context (with a warm-up and slow weakening) | **100 / 100 / 98%** held-out, ~1 channel per word | all relays open 91 / 88 / 99%; basal ganglia (one channel) 55–60% | [21](experiments/21-several-routes.md) |
| – | Elimination ("which place hasn't been named?") by a fast-learning inhibitory loop in L2/3, gated by reliability in integers | **100%** every seed, varied stories still 100% | no inhibition 14–17% (chance); ungated: varied falls to 78–84% | [22](experiments/22-fast-inhibition.md) |
| – | Compaction: event-based fast path, uncertainty-gated growth, sleep (downscale, prune, merge by replay) | **5–14× fewer kernels**, accuracy kept on every task; answering 5–62 µs/word (from 160–3,600) with surprise-gated learning, event-based recall, sparse storage and canonical kernels | transformer 16–100 µs/word, but 25–77% on held-out binding where we get 72–100% | [23](experiments/23-compaction.md), [comparison](concepts/brain-transformer-comparison.md) |
| – | A cortical hierarchy: a higher area (sentence + slow state of past surprises) learns the column's errors and feeds back a top-down frame | **81 / 82 / 80%** on a task needing story-level context | column alone 0%, episodic memory 49–52% | [24](experiments/24-cortical-hierarchy.md) |
| – | A chain of areas (windows of 4, 16, 64 sentences), each voting in a precision-weighted mix: how far back a fact can be used | each area extends the reach (4–7 stories back: 5 → 18 → 23%), but accuracy stays low | as frames into the area below: worse (0%) | [25](experiments/25-area-chain.md) |
| – | Consolidation: novel episodes' gist replayed to the higher area in sleep; hippocampal lesion at test | after 4 exposures the lesioned cortex answers 13–31% of new names (8–18% without consolidation); after 1 exposure little | interleaved replay with generalisation makes it steadier (22–27% on every seed); one-exposure consolidation still out of reach | [38](experiments/38-consolidation-of-one-shot-episodes.md) |
| – | The schema supports the episode: the cortex's class expectation filters the memory's readout | **new names after one exposure 28–44% (from 16–28%), after four 39–66%; trained names up to 71–82%** | not yet better than the no-schema group (36% vs 44% mean after one exposure) | [37](experiments/37-schema-supports-episode.md) |
| – | Slot ⊗ content memory (after TEM): learned slot cells, words bound by slot rotation, rarity-weighted recall by slot | **one-exposure learning works in the store: new names 28–76% after one exposure** (0–3% unseen); a setting slot emerges unsupervised | with familiarity-gated arbitration that keeps learning at test, the answer reaches 28–52% after 2–4 exposures (memory alone 40–70%); the schema does not yet speed learning | [36](experiments/36-slot-binding-memory.md) |
| – | A fading state (drifting temporal context, as in lateral entorhinal cortex) for the higher areas | small gains (season 12–28% without boundaries; schema test new names 14–24%) | recency is not relevance: the code needed is structural (medial-EC / TEM) | [35](experiments/35-fading-state-and-entorhinal-codes.md) |
| – | The schema test (after Tse et al. 2007): new name–place pairs seen 1, 2 or 4 times, with and without a learned schema, with and without (context-bound) episodic memory | **not learned: new names 5–18%**, no better than never seen | one-shot kernels are keyed on incidental filler; recall is not context-specific | [34](experiments/34-schema-test.md) |
| – | Generalisation during sleep: general rules formed from replay and tested on it before they are kept; specifics kept | **new names 63–78% = known; name rule no loss; habit 81–85%, at the default size and speed** | sleep's merge on top costs 1–12 points | [33](experiments/33-generalisation-during-sleep.md) |
| – | General and specific kernels side by side: generalisation spawns a general copy, the specific kernel stays | spawning after 3 confirmations, covered only by as-reliable kernels: **new names = known names (61–81%), name rule 69–73% (no loss), habit 82–85%** | 42,000 kernels (8× slower); sleep compacts 20× but costs 5–15 points | [32](experiments/32-general-and-specific.md) |
| – | Learned roles: role cells from competitive Hebbian learning on the column's expectations; transfer to names never seen in training | role cells find "sentence start", "noun after the" and more, unsupervised. **Generalisation by pruning gives full transfer: new names 61–72% = known names 65–71%**, with 8× fewer kernels | the same pruning breaks name-dependent rules (21–30%) | [31](experiments/31-role-cells-and-transfer.md) |
| – | Cortex-driven saccades (the column's possible continuations as the basal ganglia's context) and a transfer test with an unseen question wording | 86–99% on the trained wording; new wording **0%**, even with a perfect look-back | the network's knowledge is keyed on word identities: the baseline for schemas | [30](experiments/30-cortex-driven-saccades.md) |
| – | Active reading: the basal ganglia choose saccades (read on, look back to the previous sentence or the page top); the page is external memory | **99.6 / 88 / 97%** with one higher area, no boundaries needed, half the cost of the three-area chain | reading straight through 3–11% | [29](experiments/29-saccades.md) |
| – | Reading with actions: "@open book" reinstates that book's saved context; three books read in interleaved sessions | 41–54% (reinstate + contradiction detection), chance 25–33% without | limited by crowded windows, not by the actions | [28](experiments/28-reading-with-actions.md) |
| – | Story boundaries detected by the network: a rare fact contradicted by a newer one of the same kind (kinds learned from neighbouring words) | **91 / 90 / 99%**, as good as being told (86 / 96 / 98%) | surprise spikes do not mark story starts here | [27](experiments/27-boundary-detection.md) |
| – | Same chain with a context boundary at each new story; self-supervised read-back (say back the fact you hold, hear it) | **86 / 96 / 98%**, and 87–100% with the fact 16+ stories back | no boundary 9–23%; read-back helps one area (3–11 → 27–33%), not the chain | [26](experiments/26-context-and-readback.md) |

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
16. [Memory recall as a thalamic channel, chosen by the basal ganglia](experiments/16-thalamic-gate-memory-channel.md)
17. [Consolidation: hippocampal replay into a cortical semantic store](experiments/17-consolidation.md)
18. [Prefrontal working memory: a gated slot that cues recall](experiments/18-prefrontal-working-memory.md)
19. [Layer 5 as the shared reward: one dopamine signal for every selector](experiments/19-l5-shared-reward.md)
20. [Layer 6 corticothalamic gating: cortex learns which relays to let through](experiments/20-l6-corticothalamic-gating.md)
21. [Several routes needed: L6 opens more than one relay where each is used](experiments/21-several-routes.md)
22. [A fast-learning inhibitory loop in L2/3](experiments/22-fast-inhibition.md)
23. [Compaction: an event-based fast path, uncertainty-gated growth and sleep](experiments/23-compaction.md)
24. [A cortical hierarchy: a higher area that predicts the column's errors](experiments/24-cortical-hierarchy.md)
25. [A chain of cortical areas: does each area reach further back in time?](experiments/25-area-chain.md)
26. [Context boundaries and read-back: keeping the right facts in reach](experiments/26-context-and-readback.md)
27. [Detecting context boundaries: when a fact is contradicted, a new story has begun](experiments/27-boundary-detection.md)
28. [Reading with actions: opening a book brings back its context](experiments/28-reading-with-actions.md)
29. [Active reading: look back instead of holding everything](experiments/29-saccades.md)
30. [Cortex-driven saccades, and a first transfer test](experiments/30-cortex-driven-saccades.md)
31. [Learned roles and transfer to new names](experiments/31-role-cells-and-transfer.md)
32. [General and specific kernels side by side](experiments/32-general-and-specific.md)
33. [Generalisation during sleep: general rules formed offline from replay](experiments/33-generalisation-during-sleep.md)
34. [The schema test: can a new fact be learned in one exposure? (not yet)](experiments/34-schema-test.md)
35. [A fading state, and what the entorhinal cortex would add](experiments/35-fading-state-and-entorhinal-codes.md)
36. [Slot ⊗ content memory: one-exposure learning works in the hippocampus, not yet in the answer](experiments/36-slot-binding-memory.md)
37. [The schema supports the episode: class-level predictions](experiments/37-schema-supports-episode.md)
38. [Consolidating one-shot episodes into the cortex (partly)](experiments/38-consolidation-of-one-shot-episodes.md)

Concepts
- [Brain, this network, and a transformer: function vs speed and memory](concepts/brain-transformer-comparison.md)
- [Surprise-driven growth and recycling](concepts/surprise-driven-growth.md)
- [The Hebbian mask rule](concepts/hebbian-mask-rule.md)
- [Sparse codes and collisions](concepts/sparse-codes-and-collisions.md)
- [Variable binding: the gap to transformers](concepts/variable-binding.md)
- [Top-down bias](concepts/top-down-bias.md)
- [Credit assignment](concepts/credit-assignment.md)
- [Hippocampal-formation functions: what we have and what's missing](concepts/hippocampal-functions.md)
- [Basal ganglia and cerebellum: two more credit-assignment loops](concepts/basal-ganglia-and-cerebellum.md)
- [Probability in bits: what is bitwise now, and how to keep it that way](concepts/probability-in-bits.md)
- [Architecture map: which brain systems we model, and how they connect](concepts/architecture-map.md)
- [Output and self-supervision: speaking, and hearing yourself speak](concepts/output-and-self-supervision.md)

Reference
- [Bugs found and fixed](bugs-and-fixes.md)
- [Related work and reading list](related-work.md)
- [Open questions and next steps](open-questions.md)
- [Roadmap: towards reliable higher-order thinking](roadmap.md)

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
