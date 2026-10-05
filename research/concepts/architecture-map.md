# Architecture map: which brain systems we model, and how they connect

A running inventory of the systems built so far, what each is in code, and which
connections between them exist. "Not modelled" is as important as what is.

## Systems

| System | Biology | Here (code) | Status |
|---|---|---|---|
| **Cortex** (column) | Columns that predict their input; layers 2/3 (cortico-cortical), 4 (thalamic input), 5 (output to subcortex), 6 (feedback to thalamus) | `CorticalColumn` (`program::cortex`): **L4** `assemble` (current input, thalamic / memory frames, previous input), **L2/3** the predictive `KernelClass` (`predict`, `learn`), **L5** `prediction` / `confidence` / `surprise`, **L6** `ContextBuffer` (context and match rules) | Built as one column; the episodic experiment runs through it (results identical to before). Not yet: several columns / areas, L5 → basal ganglia as the reward source, L6 → thalamus gating |
| **Cerebellum** | Granule expansion + Purkinje cells, each corrected by its own climbing-fibre error | Not separate. The predictor is cerebellum-like: per-output teaching signal (`feedback`), growth on error, and now copy credit = input ∧ target ([12](../experiments/12-dentate-gyrus-ca3.md)) | Implicit |
| **Thalamus** | Relay nuclei gated by cortex and basal ganglia; pulvinar / MD route between cortical areas | Gates only (`program::thalamus`: `RouteGate`, `KernelGate`, and the BG-driven gate of [16](../experiments/16-thalamic-gate-memory-channel.md)). The match rules over stored context moved to `program::cortex` (`ContextBuffer`, `RelayChannel`, `RouteScores`) | Built |
| **Basal ganglia** | Striatum selects one channel by disinhibiting thalamus; dopamine = reward-prediction error | `BasalGanglia`: per-bit go counters (bit-sliced), winner-take-one release, three-factor update ([15](../experiments/15-basal-ganglia-selector.md)) | Built; used for which recalled item to follow. The route gate of [10](../experiments/10-route-pool-inhibition.md) is BG-like but separate |
| **Entorhinal cortex** | Input/output of the hippocampus; lateral = content, medial = grid/structure | Sparse word codes as the EC pattern; habituation (`novel`) and the rarity cue (`rarest`) | Content only; **no grid / structure code** |
| **Dentate gyrus** | Sparse expansion, pattern separation | `DentateGyrus` (random projection + k-WTA) | Built ([12](../experiments/12-dentate-gyrus-ca3.md)) |
| **CA3** | Autoassociative recurrent store | `EpisodicMemory` (list) and `Ca3Memory` (Hebbian, bit-sliced / delay-line / shift-register weights; `Ca3FloatMemory` for comparison) | Built |
| **CA1** | Compares recall with current input (novelty) | `NOVELTY=prediction` comparator in the episodic example ([14](../experiments/14-ca1-comparator.md)) | Partial (works one-hop, hurts two-hop) |
| **Subiculum** | Hippocampal output hub, to thalamus / PFC | — | Not modelled |
| **Big loop** (EC→HC→EC) | Recalled content re-enters as the next cue | `recall_chain`, `recall_branches` ([13](../experiments/13-big-loop.md)) | Built |
| **Red nucleus** | Cerebellum → red nucleus → spinal cord (and → inferior olive) | — | Not modelled: **no motor or action output** at all |
| **Prefrontal / working memory** | Holds items; BG gates updates (PBWM); directs retrieval (via nucleus reuniens) | A question's recall cue, used to tag or immediately replay the recalled episode into cortex ([17](../experiments/17-consolidation.md#prioritised-replay-questions-decide-what-is-consolidated)) | Partial: retrieval-driven consolidation only |
| **Neuromodulators** | Dopamine (reward), ACh (encode vs retrieve), NE (surprise) | Reward in `BasalGanglia`; surprise drives growth; novelty gates storage | Signals exist, no separate systems |

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

## Connections that are missing
- ~~Basal ganglia → thalamus~~ and ~~hippocampus → thalamus~~: done in
  [16](../experiments/16-thalamic-gate-memory-channel.md) — `BasalGanglia` releases one of
  the learned relay routes or memory recall per word.
- **Cortex → basal ganglia** (layer 5) and **cortex → thalamus** (layer 6 feedback): the
  layers now exist in `CorticalColumn`, but L5's surprise/confidence does not yet drive the
  basal ganglia (each experiment computes its own reward), and L6 does not gate the thalamus.
- **Cerebellum → red nucleus / thalamus → output**: there are no actions; the system only
  reads and predicts.
- **Structure code** (medial EC grid / position) for word order.
