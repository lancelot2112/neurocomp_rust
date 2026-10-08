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

## The hippocampus

What the suite runs (the relation and speech entries, for example: `HIPPO=engram`,
`HIPPO_SELF`, `BIND`, `SPARSE_BIND`, `ENGRAM_WALK`, `INFER_REPLAY`, `CONSOLIDATE`). Note
that most suite entries lesion the hippocampus at test (`BIND_LESION`): its answers reach
the test only through what replay taught the cortex.

| Connection | Model | Brain | Verdict |
|---|---|---|---|
| Cortex → EC → DG / CA3 (encoding) | Each sentence's words bound to their slots (role cells, a structure code) and stored as one event, with the story's earlier bindings as context; DG and EC III → CA1 by a fixed hash projection | Lateral EC carries content, medial EC structure; DG and CA3 form conjunctive codes through broad, fixed projections (Whittington et al. 2020) | Plausible |
| Word ⊗ slot binding | Rotation by the slot's offset, unbound by the inverse rotation at readout | Conjunctive cells; CA1 decodes CA3's code back into EC's format | Plausible (fixed wiring both ways, so a recalled word comes back in its own code and can be copied) |
| CA3 recurrence, the big loop | Pattern completion; the engram walk steps through a rare binding to its partner ([59](../experiments/59-engram-walk.md)) | CA3 recurrent collaterals; EC → DG → CA3 → CA1 → EC re-entry | Plausible |
| When to store | One event per sentence ("." ends it); novel events tagged | Events cut at prediction-error peaks; novelty (CA1 mismatch, dopamine, acetylcholine) gates encoding | Approximate: the boundary is given, not found (experiment 27 found story boundaries from surprise, but sentences are still cut by hand) |
| Familiarity and rarity (`BIND_FAM`, `BIND_RARE`) | Episode counts per binding in log2 bands | A familiarity signal (perirhinal cortex) that falls with repetition | Approximate (a count table) |
| Hippocampus → cortex at reading | The slot readout's word votes in the thalamic mix, weighed by its record per familiarity band; only where the column is unsure (`SPARSE_HC`). Other settings put recalled content in a column input slot | CA1 → subiculum → deep EC → association cortex, landing much as feedback does (outside L4); also to thalamus (anterior nuclei, nucleus reuniens) | Approximate as the default. **`HC_ROUTE=1`** (with `ROUTE`): within one word's step, in theta order, the column's feedforward sweep gives the expectation, the hippocampus recalls, CA1 decodes the recall into word codes, and they return as an entorhinal feedback channel (context, never the driver) before the column predicts; its gain is learned like any channel's, and the vote is dropped. Plausible |
| Replay → cortex (consolidation) | At sleep, stored events replayed through the cortex, novelty-tagged first, then cue-free from random CA3 starts, interleaved with familiar ones | Sharp-wave ripple replay, coordinated with cortical spindles, trains neocortex slowly and interleaved (McClelland et al. 1995) | Plausible in kind; replay enters the column as if read (the driver slot), where real replay reaches cortex through EC |
| Replay priority | Novel events tagged at encoding are replayed first (`REPLAY_TAGGED`); then strength × the cortex's error on the event, measured while it was last replayed (the cortex's response against the replayed content) and returned to the hippocampus | Replay favours surprising and poorly learned experience (Mattar & Daw 2018); the cortex drives hippocampal replay and gets it back in a sleep dialogue (Sirota et al. 2003; Buzsáki 2015) | Plausible: one pass, the error carried back through EC |
| Generative replay (`INFER_REPLAY`) | The hippocampus composes events never experienced ("lucy is a jones" + a jones event → "lucy went to the hallway") and the higher area learns them | Replay of never-experienced sequences and structural inference in replay (Gupta et al. 2010; Liu et al. 2019) | Plausible |
| Cortex → cortex under uncertainty (`COOPERATE`) | Where the column is unsure, the semantic store's content for the least familiar word joins the higher area's context | Cortico-cortical retrieval of semantic knowledge, prefrontal-guided | Plausible (the store itself is cortical) |

No hippocampal pathway uses a second run of anything. The shortcuts left:
1. ~~Hippocampal output as EC feedback~~: `HC_ROUTE` (above).
2. Replay entering through the driver slot rather than EC: cortical reactivation is what
   replay causes, so the column re-reading the event is a fair abstraction of it.
3. ~~Replay priority~~: plausible as built (above; the first audit read it wrongly).
4. Sentence boundaries from surprise, as story boundaries already are.

## Coupling: who learns from which error

The rule the anatomy gives: **each learner learns from its own error, but what it tells
the rest of the network is the integrated prediction.** The first `LEARNING=three` runs
broke the second half (every downstream signal read the slow L2/3 alone, wrong almost
everywhere at first, so the higher area learned from nearly every word and stopped
supplying context); `CorticalColumn::set_output` restores it.

| Learner | Teacher | Compared with whose prediction | In the model |
|---|---|---|---|
| Cerebellum | Climbing fibres from the inferior olive; the olive is inhibited by the cerebellum's own deep nuclei, so the error is actual − the cerebellum's prediction | Its own | `Cerebellum::learn`: its own kernels' error |
| Cortex (slow) | Local prediction error in each area, and hippocampal replay at sleep | Its own, slowly | `column.learn`: L2/3's own winner, stochastic growth |
| Hippocampus | Novelty: CA1's mismatch between EC input and CA3 recall, dopamine and acetylcholine | What the cortex as a whole failed to predict, as it reaches EC | The column's output after integration (`set_output`) |
| Higher area | Prediction error from the lower area's superficial layers | The lower area's error after it integrated all its inputs (cerebellar and hippocampal ones arriving through the thalamus) | The same |
| Basal ganglia | Dopamine reward-prediction error | The outcome of what the network did or said | The column's L5 outcome, integrated |

**Within one word (~250 ms):**
1. **Feedforward sweep.** The word drives L4; a copy of the cortex's activity goes to the
   cerebellum (L5 → pontine nuclei → mossy fibres), whose prediction is ready within tens
   of milliseconds.
2. **Hippocampus.** Cued by the cortical representation through EC; recalls within a theta
   cycle; returns through CA1 → subiculum → EC as context.
3. **Thalamus.** The cerebellum's output (deep nuclei → motor and associative thalamus),
   the higher area's (pulvinar, and direct feedback onto apical dendrites) arrive as
   context, each with a gain set by the reticular nucleus and L6.
4. **Cortex predicts; L5 integrates.** L2/3 reads all of it; the L5 output is the
   network's prediction.
5. **The next word's surprise** is computed against that integrated output, and is what
   goes up to the higher area, to the hippocampus (novelty) and to dopamine (reward).

**Two consequences.**
- **Fast to slow transfer.** The cerebellum's prediction is an input the cortex reads, so
  the slow cortex can learn to use it and, over time, to predict without it: a fast
  process acquires, a slow one retains (two-rate models of adaptation, Smith et al. 2006).
  Hippocampal replay does the same for episodes.
- **A memory that answers in training removes the surprise that would teach the rest.**
  With the hippocampus routed into the column while reading (`HC_ROUTE`), answers it
  supplies are no longer surprising, so the higher area and the cerebellum never learn
  them, and with the hippocampus lesioned at test nothing knows them (motor speech 64.8 →
  19.6%). Replay, which trains the cortex whatever the waking surprise, is what should carry
  them over: the consolidation problem.

## Learning systems
| System | Model | Brain | Verdict |
|---|---|---|---|
| Fast, error-driven (cerebellum) | The column's kernels: one-shot growth on a miss, each output taught by the next word | Purkinje cells, each with its own climbing-fibre error; also predicts in language | Plausible as a cerebellum; mislabelled as cortex until now |
| Slow, statistical (cortex), `SLOW_CORTEX=1` | A second learner on the same input; a miss grows a kernel only with probability 1/16, with near-miss generalisation; learns from waking and replay; votes in the mix | Neocortex learns slowly and interleaved (McClelland et al. 1995); stochastic synapses with low transition probabilities (Amit & Fusi 1994) | Plausible |
| Fast, detailed (hippocampus) | The engram store | Hippocampus | Plausible |
| Reward (basal ganglia) | The go / no-go selectors with eligibility traces | Striatum, dopamine | Plausible |

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
