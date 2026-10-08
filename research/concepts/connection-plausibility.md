# Connection audit: is each pathway one the brain has?

Every connection in the model, checked against the anatomy: what the model does, what the
brain has, and a verdict. **Plausible**: a pathway of this kind exists and carries this
kind of signal. **Approximate**: the pathway exists but the model takes a shortcut (a
lookup table, a global schedule). **Not plausible**: no mechanism of this kind is known,
so it is either replaced or kept only as a measurement. The functional modules that are
hand-written algorithms (the Bayes module, the relation store's rules, the curiosity
ranking) are not connections and are listed in the
[architecture map](architecture-map.md#learned-vs-hand-coded-after-44).

## Within the column

| Connection | Model | Brain | Verdict |
|---|---|---|---|
| First-order thalamus → L4 | The current word drives slot 0 of the column's input; every kernel reads it | Driver input from first-order relays to L4 (Sherman & Guillery 2006) | Plausible |
| L4 → L2/3 → L5 | L2/3 kernels predict the next word; L5 holds the prediction, its confidence and the surprise | The canonical microcircuit (Douglas & Martin 2004) | Plausible |
| L6 → L4 (the previous input) | The previous input is a frame the kernels can read | L6 keeps a short context and projects to L4 and to thalamus | Approximate (one step back, exact copy) |
| L2/3 fast inhibition | One-shot decaying tags on kernels ([22](../experiments/22-fast-inhibition.md)) | Local interneuron loops | Plausible |
| Growth on surprise | A new kernel where the column was wrong ([surprise-driven growth](surprise-driven-growth.md)) | Synaptogenesis and unsilencing driven by error and neuromodulators | Approximate |

## Between areas

| Connection | Model | Brain | Verdict |
|---|---|---|---|
| Column → higher area (feedforward) | The sentence's words and a slow state of its surprising words go up; the area learns only on the column's misses | Superficial-layer (L2/3) feedforward to L4 of the next area carries prediction error (Bastos et al. 2012); L5 also drives higher-order thalamus → next area (transthalamic) | Plausible |
| Predicted and surprising words both up (`HIER_UP=both`) | Surprising words are told apart by a rotated code | Two kinds of burst are reported: a minicolumn firing broadly on unpredicted input, and an L5 cell bursting when apical (top-down) and basal (bottom-up) input coincide (Larkum 2013), i.e. on a predicted, attended input; L5 bursts are what drive higher-order thalamus. Which group is tagged is only a label: either way both reach the area, distinguishable | Plausible as information; the tagged group should be the coincidence (predicted) one if it is meant as L5 bursts |
| Higher area → column (feedback) | The top-down frame is a slot of the column's input. Every kernel also reads the current word, so the frame only conditions a prediction; it never drives one alone | Feedback terminates mainly outside L4, on apical tufts in L1 and in L5/L6 (Rockland & Pandya 1979; Felleman & Van Essen 1991): modulatory, not driving | Functionally plausible (contextual, never driving); the "L4 slot" is a label to change to apical (L1) input |
| Areas vote in the thalamic mix (`SourceMix`) | Each source's proposed word weighed by its record per context | Pulvinar scales cortical streams by reliability (Saalmann et al. 2012); precision weighting | Approximate (a table of counts per context) |
| Areas grow by need ([84](../experiments/84-areas-grow-by-need.md)) | A bud learns in shadow; promoted on a sign test, else pruned | Adult cortex recruits uncommitted cortex, not new areas (Dehaene & Cohen 2007); silent synapses unsilenced by use | Approximate (recruitment, not creation) |

## Thalamic gating and routing

| Connection | Model | Brain | Verdict |
|---|---|---|---|
| L6 → thalamus gain ([20](../experiments/20-l6-corticothalamic-gating.md)) | Per-context gain on each relay, learned from use | Corticothalamic L6 feedback sets relay gain | Plausible |
| Basal ganglia → thalamus (go / no-go) | A selector releases a relay, a question or a consultation | BG disinhibit thalamic targets | Plausible |
| Channels bound to their source (`ROUTE_BIND=1`) | A fixed rotation per source | A pathway's fixed wiring onto its target cells | Plausible as wiring, but **harmful here**: a rotated word no longer matches the output code, so the column cannot copy it (new names fell 67 → 48%). Off: a channel keeps its code, as projections between areas keep topography, and its slot (fixed after development) says where it came from |
| A channel's share (`ROUTE`, `HIER_TRUST_GATE=cf`) | A fixed subset of the channel's bits passes, sized by its record | The reticular nucleus inhibits parts of a relay, turning its gain down | Plausible |
| The credit for a channel's share | **First version: the column asked twice, with and without the channel** (one extra look-up) | No mechanism re-runs a column on a different input | **Not plausible.** Replaced (below) |
| | **Now: from the same pass.** The kernels that do not read a slot are active alongside those that do; their best prediction is the prediction without it (`peek_shallow`) | A cell's basal drive alone against its apically supported firing; cells without a given input active in the same pass | Plausible |
| Which slot a channel takes | **First version: re-ranked every 250 stories, all run long** | Laminar targets are set in development and fixed after; gain stays plastic | **Not plausible.** Replaced: re-ranked only in a critical period (first 1,000 training stories), then fixed; shares keep adapting |

## Other credit signals

| Signal | Model | Verdict |
|---|---|---|
| The learned top-down gate's reward ([83](../experiments/83-compute-only-where-needed.md)), the counterfactual frame gate ([85](../experiments/85-top-down-as-a-trusted-witness.md)) | The column asked again with the frame blanked | Not plausible as built; both are options, not defaults. The same-pass record replaces it if they are kept |
| The bud's shadow vote ([84](../experiments/84-areas-grow-by-need.md)) | The mix's sum with and without one source's vote | Approximate: a comparison of two summed inputs, which a downstream cell can do |
| Eligibility traces in the basal ganglia | Recent choices' bits stay eligible for a later reward | Plausible (Frémaux & Gerstner 2016) |
| Teaching signal | Each kernel corrected by the actual next word | Cerebellum-like climbing-fibre error; plausible for a predictor |

## Rules kept from here on
1. **No second runs.** A credit signal must come from activity present in the same pass
   (shallower and deeper kernels, a source's vote against the sum) or from a later outcome
   (eligibility traces).
2. **Feedback modulates, never drives.** Top-down input conditions a prediction that the
   driver (the current word) already supports; no kernel may fire on top-down alone.
3. **Wiring is fixed after development; gains adapt.** Where an input lands is decided
   early; how much of it passes stays plastic.
4. **Codes stay topographic between areas.** A word keeps its code along a pathway, so it
   can be copied; where a binding is needed (the relation store's positions) it is fixed
   wiring, never computed per item.
