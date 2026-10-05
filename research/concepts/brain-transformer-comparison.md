# Brain, this network, and a transformer: function vs speed and memory

**One frame for all three:** a fast interpretation layer reads each input against a
**prior learned slowly from the statistics of past input**, and uses the mismatch to
decide what the input means and what to store. They differ in where the prior lives, what
the fast layer can hold, and what learning costs.

## Features

| | Brain | This network | Transformer (decoder-only) |
|---|---|---|---|
| **Slow prior** | Cortical synapses, learned over years | L2/3 kernels with integer hit / miss counts; basal-ganglia go counters; L6 relay gains | All weights, learned by backpropagation over many passes |
| **Fast interpretation** | Thalamocortical loops, ~10–100 ms per step | Each word: L4 assembles input, L2/3 kernels propose, a winner is chosen, surprise is measured, relays and memory are gated | One forward pass per token |
| **Activity** | Sparse: about 1–2% of cortical neurons active | Sparse: 32 of 8,192 bits (0.4%) per word; work scales with active bits | Dense: every weight used for every token |
| **Fast memory of *this* input** | Hippocampus, one-shot ([11](../experiments/11-episodic-memory.md)–[13](../experiments/13-big-loop.md)) | Episodic store, CA3-like; working-memory slot ([18](../experiments/18-prefrontal-working-memory.md)); fast inhibitory tags ([22](../experiments/22-fast-inhibition.md)) | The context window: every past token is attended |
| **Working memory** | Prefrontal stripes gated by the basal ganglia | A learned load / keep gate ([18](../experiments/18-prefrontal-working-memory.md)) | Implicit in attention |
| **Routing** | Thalamus, gated by the basal ganglia (select) and L6 (gain) | Both ([16](../experiments/16-thalamic-gate-memory-channel.md), [20](../experiments/20-l6-corticothalamic-gating.md)–[21](../experiments/21-several-routes.md)) | Attention heads |
| **Reward / credit** | Dopamine as reward-prediction error, plus local plasticity | Local three-factor rules; the L5 outcome as a shared reward ([19](../experiments/19-l5-shared-reward.md)); copy credit | Global gradient of one loss |
| **Learning schedule** | Inline, plus sleep (downscaling, replay, consolidation) | Inline at every word, plus sleep: replay into cortex ([17](../experiments/17-consolidation.md)), downscale / prune / merge ([23](../experiments/23-compaction.md)) | Offline, many passes, no consolidation |
| **Number format** | Spikes, synapses with a few stable states | Bits; 8-bit counters and integer compares, no floats in the comparisons ([probability in bits](probability-in-bits.md)) | 32-bit floats (or 8–16 bit after training) |

## Function: the same stories, held-out pairs (seed 0 for the transformer)
The transformer is a small GPT ([`research/baselines/transformer_baseline.py`](../baselines/transformer_baseline.py)).
It was trained by next-word prediction on exactly the stories our network sees (exported
with `DUMP_STORIES`) and scored the same way. Its column is the best held-out score over
three sizes (28k, 106k and 800k parameters) and 1, 3, 10, 30 or 100 passes.

| Task | Transformer, best held-out (its best on *seen* pairs) | This network, one pass |
|---|---|---|
| Elimination ([22](../experiments/22-fast-inhibition.md)) | **100%** (28k params, 10 passes; 800k, 1 pass) | **100%** |
| Topic: hold the subject ([18](../experiments/18-prefrontal-working-memory.md)) | **100%** (28k, 3 passes) | **100%** |
| Varied, 1–2 facts ([12](../experiments/12-dentate-gyrus-ca3.md)) | 77% (95%) | **100%** |
| Varied, 1–3 facts | 70% (76%) | **100%** |
| Two-hop ([15](../experiments/15-basal-ganglia-selector.md)) | 29% (52%) | **72–89%** |
| Give: two routes ([21](../experiments/21-several-routes.md)) | 25% (55%) | **98–100%** |

- **Where the transformer generalises:** elimination and topic, where attention solves
  the task by looking back over the story.
- **Where it doesn't:** on the four binding tasks, more passes made it *worse* on held-out
  pairs while seen pairs improved. The 800k model on two-hop went from 29% after 3 passes
  to 4% after 100. It memorises pairs instead of learning to copy them out of the story.
- **Caveats:** one seed, an untuned standard recipe and only 3,000 training stories. More
  data or tuning would likely help it, so the fair claim is "on this data".

## Cost
Both run single-threaded on the same machine, so ratios are rough. Our figures are with
compaction, surprise-gated learning and event-based recall
([23](../experiments/23-compaction.md); elimination without the surprise gate, which
costs it speed there).

| | Transformer (28k / 106k / 800k params) | This network |
|---|---|---|
| **Model memory** | 0.11 / 0.42 / 3.2 MB (fp32) | **0.65–3.1 MB** in all: kernel connections stored sparsely (0.25–1.5 MB), inverted index (0.2–1.4 MB), 200 KB episodic store |
| **Answering, per word** | 16–31 / 21–45 / 55–100 µs | 31–43 µs (elimination), 37–46 µs (varied, topic), 42–53 µs (give), 66–76 µs (two-hop), with the memos of [23 §7–8](../experiments/23-compaction.md#7-memoised-interpretation-after-hashlife) |
| **Training, per word per pass** | 9–16 / 25–42 / 90–160 µs | 39 µs (elimination), 54–57 µs (varied, topic), 147–161 µs (two-hop, give), learning inline |
| **Passes needed** | 3–100, or never (binding tasks) | 1 |
| **Total training to its best** | Elimination ~4–13 s; topic ~3 s; varied ~3 s (to 70–77%); give / two-hop: no amount of training reached ours | Elimination ~5 s; varied ~5 s; topic ~4 s; give ~12 s; two-hop ~18 s |

### So: faster, or less memory, for more function?
- **More function:** yes, on held-out binding, which is the part that needs one-shot
  memory of *this* story: 100% against 70–77% on varied, 72–89% against 29% on two-hop,
  and 98–100% against 25% on give. Equal (100%) where attention alone suffices.
- **Speed:** per word, now in the same range as the transformers when answering.
  - **Elimination:** 31–43 µs, between the 28k and 106k models and about 2× faster than
    the 800k model. It was 19 µs before winner ties were made order-independent
    ([23 §8](../experiments/23-compaction.md#8-per-frame-memo-hashlifes-sub-nodes-exact-but-not-a-net-win)).
  - **Varied and topic:** 37–44 µs, level with the 106k model (21–45 µs) and about 2×
    faster than the 800k model (74–102 µs).
  - **Give:** 42–53 µs, about 1.7× faster than the 800k model (79–90 µs).
  - **Two-hop:** 66–76 µs, level with the 800k model (60–70 µs).
  - **Training** reaches the result in one pass, so total training time is 4–18 seconds,
    against 3–13 seconds for the transformer on the tasks it can learn.
  - This is after a ~85–190× speed-up today, from making the path event-based, compacting
    the prior, gating learning on surprise and indexing recall. Before it, we were
    100–500× slower.
- **Memory:** comparable: 0.65–3.1 MB for the whole model against 0.1–3.2 MB, now that
  kernels are stored as bit-position lists
  ([23](../experiments/23-compaction.md#6-sparse-kernel-storage)).

## Against the brain
- **What we share:**
  - sparse codes, with event-driven cost;
  - one-shot episodic memory, and a fast loop against a slowly learned prior;
  - gating by the basal ganglia and by L6;
  - local learning with reward signals;
  - sleep that consolidates (replay) and compacts (downscale, prune, merge);
  - expected uncertainty gating structural learning.
- **What the brain does that we don't (yet):**
  - many columns and a hierarchy;
  - continuous time, oscillations and phase codes;
  - action and motor output (no red nucleus or cerebellar output);
  - learning only from surprise: we still score every matched kernel at every word;
  - and energy: about 20 W for ~86 billion neurons. That efficiency comes from doing
    almost nothing for an expected input, and the remaining speed-ups here point the same
    way (surprise-gated learning, sparse storage).
- **What a transformer has that the brain and we lack:** an exact, unlimited-recency
  context window. That is why it solves elimination and topic by lookup, and why it
  doesn't *need* a hippocampus to do so, within one context.
