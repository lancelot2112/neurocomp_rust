# Architecture map: which brain systems we model, and how they connect

A running inventory of the systems built so far, what each is in code, and which
connections between them exist. "Not modelled" is as important as what is.

## Systems

| System | Biology | Here (code) | Status |
|---|---|---|---|
| **Cortex** (column) | Columns that predict their input; layers 2/3 (cortico-cortical), 4 (thalamic input), 5 (output to subcortex), 6 (feedback to thalamus) | `CorticalColumn` (`program::cortex`): **L4** `assemble` (current input, thalamic / memory frames, previous input), **L2/3** the predictive `KernelClass` (`predict`, `learn`), **L5** `prediction` / `confidence` / `surprise`, **L6** `ContextBuffer` (context and match rules) | Built as one column; the episodic experiment runs through it (results identical to before). **L5 → basal ganglia** built: `outcome` / `outcome_via` is the shared reward for every selector ([19](../experiments/19-l5-shared-reward.md)). **L6 → thalamus** built: `CorticothalamicGate` opens relays per context, learned from use ([20](../experiments/20-l6-corticothalamic-gating.md)), several at once where each is needed ([21](../experiments/21-several-routes.md)). **L2/3 fast inhibitory loop**: one-shot, decaying tags on kernels, gated by reliability ([22](../experiments/22-fast-inhibition.md)). **Compaction**: uncertainty-gated growth while awake, and sleep that downscales, prunes and merges kernels by replay ([23](../experiments/23-compaction.md)). **Hierarchy**: a `HigherArea` above the column, driven by its L5 surprises over a slow window and feeding back a top-down frame, trained on the column's residual ([24](../experiments/24-cortical-hierarchy.md)). **Chain of areas**: windows of 4, 16, 64 sentences, each area a source in the precision-weighted mix ([25](../experiments/25-area-chain.md)). Not yet: an event-driven clock, several columns side by side, output (planned: [output and self-supervision](output-and-self-supervision.md)) |
| **Cerebellum** | Granule expansion + Purkinje cells, each corrected by its own climbing-fibre error | Not separate. The predictor is cerebellum-like: per-output teaching signal (`feedback`), growth on error, and now copy credit = input ∧ target ([12](../experiments/12-dentate-gyrus-ca3.md)) | Implicit |
| **Thalamus** | Relay nuclei gated by cortex and basal ganglia; pulvinar / MD route between cortical areas | Gates only (`program::thalamus`: `RouteGate`, `KernelGate`, the BG-driven gate of [16](../experiments/16-thalamic-gate-memory-channel.md), and the L6-driven `CorticothalamicGate` of [20](../experiments/20-l6-corticothalamic-gating.md)), and a pulvinar-like gain: `SourceMix` weights each source (column, memory, top-down) by its learned reliability and sums the evidence ([24](../experiments/24-cortical-hierarchy.md#precision-weighted-mixing-instead-of-switching)). The match rules over stored context moved to `program::cortex` (`ContextBuffer`, `RelayChannel`, `RouteScores`) | Built |
| **Basal ganglia** | Striatum selects one channel by disinhibiting thalamus; dopamine = reward-prediction error | `BasalGanglia`: per-bit go counters (bit-sliced), winner-take-one release, three-factor update ([15](../experiments/15-basal-ganglia-selector.md)) | Built; used for which recalled item to follow. The route gate of [10](../experiments/10-route-pool-inhibition.md) is BG-like but separate |
| **Entorhinal cortex** | Input/output of the hippocampus; lateral = content, medial = grid/structure | Sparse word codes as the EC pattern; habituation (`novel`) and the rarity cue (`rarest`) | Content, plus a drifting temporal context (the fading state, as in lateral EC: [35](../experiments/35-fading-state-and-entorhinal-codes.md)); **no grid / structure code yet**: the next step for schemas |
| **Dentate gyrus** | Sparse expansion, pattern separation | `DentateGyrus` (random projection + k-WTA) | Built ([12](../experiments/12-dentate-gyrus-ca3.md)) |
| **CA3** | Autoassociative recurrent store | `EpisodicMemory` (list) and `Ca3Memory` (Hebbian, bit-sliced / delay-line / shift-register weights; `Ca3FloatMemory` for comparison) | Built |
| **CA1** | Compares recall with current input (novelty) | `NOVELTY=prediction` comparator in the episodic example ([14](../experiments/14-ca1-comparator.md)) | Partial (works one-hop, hurts two-hop) |
| **Subiculum** | Hippocampal output hub, to thalamus / PFC | — | Not modelled |
| **Big loop** (EC→HC→EC) | Recalled content re-enters as the next cue | `recall_chain`, `recall_branches` ([13](../experiments/13-big-loop.md)) | Built |
| **Red nucleus** | Cerebellum → red nucleus → spinal cord (and → inferior olive) | — | Not modelled: **no motor or action output** at all |
| **Prefrontal / working memory** | Holds items; BG gates updates (PBWM); directs retrieval (via nucleus reuniens) | `WorkingMemory` slot + `PfcGate` (basal-ganglia load / keep, credit to the load whose content is held); the slot's content cues hippocampal recall ([18](../experiments/18-prefrontal-working-memory.md)). A question's cue also tags or replays episodes into cortex ([17](../experiments/17-consolidation.md#prioritised-replay-questions-decide-what-is-consolidated)) | Built: one slot, gate keyed on the word alone (no context yet) |
| **Neuromodulators** | Dopamine (reward), ACh (encode vs retrieve; expected uncertainty), NE (surprise; unexpected uncertainty) | Reward in `BasalGanglia` and the L5 outcome; surprise drives growth, except where uncertainty is expected (ACh-like growth gate, [23](../experiments/23-compaction.md)); novelty gates storage | Signals exist, no separate systems |

## Connections that exist

```
 word codes (EC) ──► predictor (cortex) ──► next-word prediction  (the only output)
      │                 ▲   ▲   ▲
      │                 │   │   └── teaching signal: the actual next word (cerebellum-like)
      │                 │   └────── thalamic relay frames (route chosen by RouteGate / KernelGate)
      │                 └────────── recalled frames from memory
      ▼
  DG ──► CA3 store ──► recall (cue: rarest / unpredicted part of the sentence)
                          │
                          ├─► hop 2 (big loop): cued by a recalled item
                          │       ▲
                          │       └── which item: BasalGanglia (reward = hop 2 recalled the next word)
                          └─► CA1 comparator (optional): store / cue only what the predictor missed
```

- Cortex → hippocampus: the sentence (and, with the comparator, what the predictor failed to
  predict) is what gets stored and what cues recall.
- Hippocampus → cortex: recalled content is an input frame of the predictor.
- Thalamus → cortex: relayed words are input frames; the gate is learned from whether the
  relay helped predict.
- Basal ganglia → hippocampal loop: chooses what the second hop follows.
- Cortex L6 → thalamus: per-context gain on each relay channel, Hebbian from use
  ([20](../experiments/20-l6-corticothalamic-gating.md)).
- Cortex L5 → basal ganglia: the column's outcome (attributed to the frame it read) is the
  shared reward ([19](../experiments/19-l5-shared-reward.md)).
- Basal ganglia → prefrontal → hippocampus: the gate loads a word into working memory; its
  content cues recall at the question ([18](../experiments/18-prefrontal-working-memory.md)).

## Connections that are missing
- ~~Basal ganglia → thalamus~~ and ~~hippocampus → thalamus~~: done in
  [16](../experiments/16-thalamic-gate-memory-channel.md) — `BasalGanglia` releases one of
  the learned relay routes or memory recall per word.
- ~~Cortex → basal ganglia~~ (layer 5): done in [19](../experiments/19-l5-shared-reward.md).
  With `REWARD=l5_used` every selector learns from the column's outcome, credited only if
  the prediction read the selector's frame. It is not yet the default; the hop-2 selector
  is still 4 points behind its own reward on average.
- ~~Cortex → thalamus~~ (layer 6 feedback): done in [20](../experiments/20-l6-corticothalamic-gating.md).
  Several routes at once tested in [21](../experiments/21-several-routes.md); not yet combined
  with the basal-ganglia gate.
- **Cerebellum → red nucleus / thalamus → output**: there are no actions; the system only
  reads and predicts.
- **Structure code** (medial EC grid / position) for word order.
