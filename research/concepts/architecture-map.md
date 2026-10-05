# Architecture map: which brain systems we model, and how they connect

A running inventory of the systems built so far, what each is in code, and which
connections between them exist. "Not modelled" is as important as what is.

## Systems

| System | Biology | Here (code) | Status |
|---|---|---|---|
| **Cortex** (predictive layer) | Columns that predict their input; layers 2/3 (cortico-cortical), 4 (thalamic input), 5 (output to subcortex), 6 (feedback to thalamus) | `KernelClass` predictive mode: sparse conjunction kernels grown on surprise, ranked by depth and reliability ([02](../experiments/02-predictive-growth.md)–[08](../experiments/08-credit-assignment.md)) | Built. **No layers**: one predictor per stage; a word layer feeding a char layer is the only stack ([07](../experiments/07-top-down-bias.md)) |
| **Cerebellum** | Granule expansion + Purkinje cells, each corrected by its own climbing-fibre error | Not separate. The predictor is cerebellum-like: per-output teaching signal (`feedback`), growth on error, and now copy credit = input ∧ target ([12](../experiments/12-dentate-gyrus-ca3.md)) | Implicit |
| **Thalamus** | Relay nuclei gated by cortex and basal ganglia; pulvinar / MD route between cortical areas | `Thalamus`: relay routes (query lag, value offset), discovery, `RouteScores`, `RouteGate` / `KernelGate` ([09](../experiments/09-thalamic-attention.md), [10](../experiments/10-route-pool-inhibition.md)) | Built (table gate works; bitwise gate fails) |
| **Basal ganglia** | Striatum selects one channel by disinhibiting thalamus; dopamine = reward-prediction error | `BasalGanglia`: per-bit go counters (bit-sliced), winner-take-one release, three-factor update ([15](../experiments/15-basal-ganglia-selector.md)) | Built; used for which recalled item to follow. The route gate of [10](../experiments/10-route-pool-inhibition.md) is BG-like but separate |
| **Entorhinal cortex** | Input/output of the hippocampus; lateral = content, medial = grid/structure | Sparse word codes as the EC pattern; habituation (`novel`) and the rarity cue (`rarest`) | Content only; **no grid / structure code** |
| **Dentate gyrus** | Sparse expansion, pattern separation | `DentateGyrus` (random projection + k-WTA) | Built ([12](../experiments/12-dentate-gyrus-ca3.md)) |
| **CA3** | Autoassociative recurrent store | `EpisodicMemory` (list) and `Ca3Memory` (Hebbian, bit-sliced / delay-line / shift-register weights; `Ca3FloatMemory` for comparison) | Built |
| **CA1** | Compares recall with current input (novelty) | `NOVELTY=prediction` comparator in the episodic example ([14](../experiments/14-ca1-comparator.md)) | Partial (works one-hop, hurts two-hop) |
| **Subiculum** | Hippocampal output hub, to thalamus / PFC | — | Not modelled |
| **Big loop** (EC→HC→EC) | Recalled content re-enters as the next cue | `recall_chain`, `recall_branches` ([13](../experiments/13-big-loop.md)) | Built |
| **Red nucleus** | Cerebellum → red nucleus → spinal cord (and → inferior olive) | — | Not modelled: **no motor or action output** at all |
| **Prefrontal / working memory** | Holds items; BG gates updates (PBWM) | — | Not modelled (closest: selector's choice held for one hop) |
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
- **Basal ganglia → thalamus**: the classical loop (BG disinhibits one thalamic channel) is
  done by a separate route gate, not by `BasalGanglia`.
- **Hippocampus → thalamus** (subiculum → anterior thalamus, nucleus reuniens): memory recall
  is not a route the gate can choose.
- **Cortex → thalamus** (layer 6 feedback) and **cortex → basal ganglia** (layer 5): no
  layers, so no separate feedback or action-selection outputs.
- **Cerebellum → red nucleus / thalamus → output**: there are no actions; the system only
  reads and predicts.
- **Structure code** (medial EC grid / position) for word order.
