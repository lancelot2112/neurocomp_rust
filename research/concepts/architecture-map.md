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

## Inventory: built and still to build (after [36](../experiments/36-slot-binding-memory.md))

**Built** (each with the experiment that tests it):

| Brain system | What we have |
|---|---|
| Neocortical column | L4 input assembly, L2/3 predictive kernels (surprise-driven growth, integer reliability, fast inhibition, compaction), L5 prediction / confidence / outcome, L6 context and thalamic gating ([19](../experiments/19-l5-shared-reward.md)–[23](../experiments/23-compaction.md)) |
| Cortical hierarchy | Higher areas with slow states, residual (predictive-coding) learning, a chain of areas with longer windows ([24](../experiments/24-cortical-hierarchy.md), [25](../experiments/25-area-chain.md)) |
| Event boundaries | Story boundaries given or detected from contradicted facts ([26](../experiments/26-context-and-readback.md), [27](../experiments/27-boundary-detection.md)) |
| Thalamus | Relay gating (route gates, L6 corticothalamic gate), a pulvinar-like precision-weighted mix of sources ([20](../experiments/20-l6-corticothalamic-gating.md), [24](../experiments/24-cortical-hierarchy.md)) |
| Basal ganglia | Bit-sliced go counters with dopamine-like reward: recall, relays, working-memory gating, saccades ([15](../experiments/15-basal-ganglia-selector.md), [29](../experiments/29-saccades.md)) |
| Prefrontal working memory | A gated slot that cues recall ([18](../experiments/18-prefrontal-working-memory.md)) |
| Hippocampus | Episodic store, dentate gyrus expansion + CA3, big-loop recall, replay into a semantic store, rarity-weighted recall, slot ⊗ content episodes ([11](../experiments/11-episodic-memory.md)–[17](../experiments/17-consolidation.md), [36](../experiments/36-slot-binding-memory.md)) |
| Semantic store (anterior temporal hub) | A cortical cue → content store trained only by novelty-prioritised sleep replay; it carries a once-stated fact ("tom is a smith") after a hippocampal lesion ([17](../experiments/17-consolidation.md), [42](../experiments/42-semantic-store.md)) |
| Entorhinal cortex | Temporal context (fading state, lateral EC), learned slot cells (a coarse structure code, medial EC), a setting slot that emerged ([31](../experiments/31-role-cells-and-transfer.md), [35](../experiments/35-fading-state-and-entorhinal-codes.md), [36](../experiments/36-slot-binding-memory.md)) |
| Sleep | Downscaling, pruning, merging, generalisation from replay ([23](../experiments/23-compaction.md), [33](../experiments/33-generalisation-during-sleep.md)) |
| Eye movements | Basal-ganglia saccades with regressions, a page index of landmarks, cortex-driven context ([29](../experiments/29-saccades.md), [30](../experiments/30-cortex-driven-saccades.md)) |
| Actions as context | Efference-copy-like action tokens; context saved and reinstated per book ([28](../experiments/28-reading-with-actions.md)) |
| Association / role cells | Competitive Hebbian categories ([31](../experiments/31-role-cells-and-transfer.md)) |
| Generalisation | Synapse-level pruning, general copies beside specific kernels ([31](../experiments/31-role-cells-and-transfer.md), [32](../experiments/32-general-and-specific.md)) |
| Rehearsal | Read-back of held facts ([26](../experiments/26-context-and-readback.md)) |

**Partly built:**
- CA1 comparator: one-hop only ([14](../experiments/14-ca1-comparator.md)).
- Familiarity vs recollection arbitration ([36](../experiments/36-slot-binding-memory.md)).
- Calibration and abstention: measured, not acted on.
- Neuromodulation: dopamine-like reward and surprise; acetylcholine and noradrenaline
  only as fixed rules.

**Not built yet:**
- **Cerebellum** as a separate system: fine-tuning of saccades and timing.
- **Motor / speech output:** saying words, the efference copy of speech, the read-back
  loop of the [output plan](output-and-self-supervision.md).
- **Grid cells / metric structure codes;** path-integrated structure (TEM proper).
- **Subiculum** (hippocampal output hub).
- **Perirhinal cortex** (item familiarity as its own system).
- **Hippocampal–prefrontal schema circuits:** one-exposure learning now reaches the
  answer, filtered by the cortex's class expectation
  ([37](../experiments/37-schema-supports-episode.md)). Its consolidation into cortex is
  partial ([38](../experiments/38-consolidation-of-one-shot-episodes.md)), and a schema
  advantage now shows for items that fit the learned structure
  ([39](../experiments/39-schema-advantage.md)).
- **Several columns side by side** (breadth): other modalities and concepts.
- **Amygdala** (salience, value); **hypothalamus** (drives).
- **An event-driven clock** for the higher areas.
- **Learned versions of the remaining hand-set pieces:** sentence units, decoding to
  word identities in the example code, the 2% rarity threshold, the context store keyed
  by action.

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
