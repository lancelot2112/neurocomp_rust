# Start here: what the network is, what holds, what is open

A summary of the 85 experiment pages, by theme rather than by date. Each finding links to
the page that shows it; where a later page qualified or overturned an earlier one, the
later one is cited and the correction is listed in [errata](errata.md). The
[README](README.md) keeps the full results table and the chronological index.

**How to read the numbers.** Kernel growth is path-dependent: a change at 0.2% of training
steps moved one seed's held-out accuracy by 22 points ([85](experiments/85-top-down-as-a-trusted-witness.md#addendum-trust-in-the-frames-use-and-scaled-frames)).
Single-seed differences of 10–20 points, and three-seed mean differences of 5–10 points on
the memory tasks, can be noise. Findings below are those that hold beyond that, or are
marked *qualified*. Numbers on early pages come from builds that predate later fixes; the
current figures are the regression suite's (`scripts/regress.tsv`, three-seed means).

## The question
Can a network of integer, bitwise, locally learned kernels learn to read the way a
transformer does: predict, bind facts, use what it read stories ago, compose, and know
what it does not know? And can it do so with parts the brain has?

## The network today

| Part | What it is | Default in the suite | Pages |
|---|---|---|---|
| Predictive kernels (`KernelClass`) | Sparse conjunctions of input bits, grown in one shot where the prediction was wrong, each output taught by the next input. **This is a cerebellum's rule** ([connection audit](concepts/connection-plausibility.md#learning-systems)); the wiki called it cortex until page 85 | The column's L2/3 | [02](experiments/02-predictive-growth.md), [23](experiments/23-compaction.md) |
| Cortical column | L4 input row [current word, memory or relay slots, top-down, previous input], L2/3 kernels, L5 prediction, confidence and surprise, L6 context | Yes | [19](experiments/19-l5-shared-reward.md)–[23](experiments/23-compaction.md) |
| Higher areas | A slower area over the column, learning the column's residual over a window of sentences; a chain of areas with longer windows | One area (three on story boundary); growth by need is an option | [24](experiments/24-cortical-hierarchy.md)–[26](experiments/26-context-and-readback.md), [84](experiments/84-areas-grow-by-need.md) |
| Thalamus | Relay gates from L6 and the basal ganglia; a precision-weighted mix of every source's vote (`SourceMix`) | The mix (`MIX=1`) | [20](experiments/20-l6-corticothalamic-gating.md), [24](experiments/24-cortical-hierarchy.md#precision-weighted-mixing-instead-of-switching) |
| Hippocampus | Words bound to role slots, stored per event (engram store), recalled by a rare cue, walked one step, replayed at sleep (tagged first), composing new events (generative replay) | Engram store; lesioned at test on most entries | [36](experiments/36-slot-binding-memory.md), [49](experiments/49-sparse-binding-space.md), [56](experiments/56-engram-store.md), [59](experiments/59-engram-walk.md), [62](experiments/62-replay-as-reading.md) |
| Relation store | Typed facts ("X is a Y") parsed from sentence shape, rules composed at sleep | Yes on the relation entries | [66](experiments/66-typed-relations.md), [67](experiments/67-relation-store-in-reading.md) |
| Basal ganglia | Go / no-go selectors with eligibility traces: what to follow, when to step, when to answer, what to ask | Several | [15](experiments/15-basal-ganglia-selector.md), [43](experiments/43-learned-stepping.md), [80](experiments/80-learned-answer-or-unknown.md) |
| Bayes module | Facts as claims by sources; trust per source and topic learned from conflicts | On the belief entry | [75](experiments/75-bayes-module.md), [78](experiments/78-belief-decides-the-answer.md) |
| Output | Speech through a babbled motor area, efference copy, retelling | On the speech entries | [69](experiments/69-answering-by-speaking.md)–[71](experiments/71-speech-routing.md) |

Options built but not defaults: learned input routing (`ROUTE`), the hippocampus as an
entorhinal channel (`HC_ROUTE`), three learning systems (`LEARNING=three`: slow cortex,
fast `Cerebellum`), curiosity (`ASK`), gates on the top-down channel (83, 85), fact
completion (`REL_COMPLETE`).

## What holds, by theme

**Prediction and segmentation.** Surprise-driven growth predicts characters close to a
back-off n-gram on a book read once (56.9% against 59.0%, [03](experiments/03-book-scale-char-prediction.md))
and finds word boundaries from surprise (F1 61.9 against 55.9, [04](experiments/04-word-segmentation.md)).
At the word level it is below the frequency baseline (9.1% against 9.8%, [05](experiments/05-syntax.md)):
next-word prediction alone was never this network's strength.

**Binding needs memory, not more kernels.** The predictor binds seen pairs and fails every
held-out pair (0%, [06](experiments/06-meaning.md)); a one-shot episodic memory binds them
(95–99%, [11](experiments/11-episodic-memory.md)). Everything after builds on that.

**Gating and selection.** L6 learns which relays to pass without reward, at a fraction of
the cost (100%, 0.04–0.19 channels a word, [20](experiments/20-l6-corticothalamic-gating.md)),
and several at once where each is needed ([21](experiments/21-several-routes.md)). A fast
inhibitory loop handles "what just happened" (elimination 100% against 16%, [22](experiments/22-fast-inhibition.md)).
Credit at the answer works; credit on every next word does not ([43](experiments/43-learned-stepping.md),
[65](experiments/65-cue-controller.md), [83](experiments/83-compute-only-where-needed.md#addendum-2-credit-over-time)).

**Context across stories.** A higher area trained on the column's residual supplies
context the column cannot see (habit 0% → 81%, [24](experiments/24-cortical-hierarchy.md)).
With a story boundary, three areas hold a fact 16+ stories back (86–98%, [26](experiments/26-context-and-readback.md));
the network can find the boundary itself from contradicted facts ([27](experiments/27-boundary-detection.md))
and grow the three areas from one by need ([84](experiments/84-areas-grow-by-need.md)).
Looking back at the page (learned saccades) does it with one area (88–99.6%, [29](experiments/29-saccades.md)).
The top-down input helps the column as context, not as a witness to the next word
([85](experiments/85-top-down-as-a-trusted-witness.md)); every gate on it so far helps
some tasks and hurts others ([24](experiments/24-cortical-hierarchy.md), [83](experiments/83-compute-only-where-needed.md), [85](experiments/85-top-down-as-a-trusted-witness.md)).

**Generalising to new names.** Knowledge is keyed on word identities: new wordings fail
at 0% ([30](experiments/30-cortex-driven-saccades.md)). General rules formed offline from
replay transfer to new names at normal size (63–78%, [33](experiments/33-generalisation-during-sleep.md));
a learned schema makes never-shown members of a known family answerable at once, and from
the cortex alone (70/75/55% against 26/28/10%, [39](experiments/39-schema-advantage.md)).

**Consolidation.** Replay moves overwritten facts into a cortical store ([17](experiments/17-consolidation.md));
a semantic store fed by novelty-tagged replay carries a family with the hippocampus off
(95–100%, [42](experiments/42-semantic-store.md)). Rules from hippocampal replay fail
where the waking buffer works, for a format reason (sets versus left-to-right context,
[50](experiments/50-replay-for-rule-extraction.md)); replay read as short stories teaches composed
facts (lesioned 23 → 48%, [62](experiments/62-replay-as-reading.md)), but only when
unvalidated inferences are replayed ([74](experiments/74-proposals-and-premises.md)).
Source tags stop the network confusing its own retellings with the world (61 → 89.7%,
[73](experiments/73-source-memory.md)).

**Epistemics.** Trust learned from conflicts settles 1:1 disputes that a vote cannot
(family questions 98–100% under graded or posterior belief, against 29–68% trusting
everyone or voting, [78](experiments/78-belief-decides-the-answer.md)). The posterior says
"unknown" exactly where two equal sources disagree ([79](experiments/79-unknown-when-belief-is-split.md));
the choice to abstain can be learned from practice ([80](experiments/80-learned-answer-or-unknown.md)).
Trust has topics: a source honest about places can lie about families ([81](experiments/81-curiosity.md#addendum-asking-during-practice-learning-its-own-policy-and-paying-for-it)).
A learned asking policy finds the undecided question with practice ([81](experiments/81-curiosity.md)); a
cost per question works only when cheap.

**Output.** The network speaks its answers through a babbled motor area at no loss
(64.8% against 64.4%, [69](experiments/69-answering-by-speaking.md), [71](experiments/71-speech-routing.md)),
retells stories with the hippocampus planning (93–97.5%, [70](experiments/70-efference-copy-and-recitation.md)),
and abstains well when unsure (~60% answered at 84–91% right, [69](experiments/69-answering-by-speaking.md)).

**Infrastructure.** Every step is integer ([47](experiments/47-integer-only.md)); the
network is built from base kernels composed by a grammar ([51](experiments/51-networks-of-kernels.md));
caching, exact pruning and parallel runs make the suite affordable.

## What did not hold
Listed in [errata](errata.md), with the page that changed each. The main ones:
- The text search that "beat every memory" (67.6%) had stored the test stories; corrected,
  17% ([57](experiments/57-memory-vs-text.md) → [58](experiments/58-a-story-is-a-graph.md)).
- Many 3–5 point gains on three seeds are within noise (pages 15, 19, 40, 59, 61, 66, 67,
  68, 82, 83 among them).
- The column is not a slow cortex; claims that a kernel store "plays the neocortex"
  ([67](experiments/67-relation-store-in-reading.md)) describe a fast, cerebellum-like learner.
- Tagging a channel's source by rotating its code stops the column copying words (the
  routing and `HIER_UP=both` work after 85; [connection audit](concepts/connection-plausibility.md)).

## What is open
- **Three learning systems** (roadmap): a slow cortex beside the fast kernels, coupled so
  each learns from its own error and reports the integrated prediction. First runs are
  mixed; the slow cortex learns too little in 3,000 stories.
- **The consolidation problem**: a memory that answers while reading removes the surprise
  that would teach the rest; replay has to carry it over.
- **Learned input routing** with plausible credit (same-pass, critical period).
- **More seeds for decisions** near the noise.
- The [roadmap](roadmap.md) has the rest; [open questions](open-questions.md) the older list.
