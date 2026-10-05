# 16 · Memory recall as a thalamic channel, chosen by the basal ganglia

**Question.** Can the basal ganglia learn, from reward alone, which pathway to open into
cortex when one of the pathways is hippocampal memory recall and the others are relay
routes? And can the routes themselves be learned rather than hand-set? This closes two
of the missing connections in the [architecture map](../concepts/architecture-map.md):
basal ganglia → thalamus, and hippocampus → thalamus → cortex (subiculum → anterior
thalamus / nucleus reuniens).

**Code.** `Policy::ThalamicGate` and `Policy::LearnedGate` in
[`examples/episodic.rs`](../../examples/episodic.rs) (`POLICIES=gate`,
`POLICIES=learned_gate`); `BasalGanglia` ([15](15-basal-ganglia-selector.md));
`ContextBuffer`, `RelayChannel`, `RouteScores` in [`src/program/cortex.rs`](../../src/program/cortex.rs).

## Mechanism
- **Channels:** relay routes (q, +v) — "find the last time the word q steps back occurred,
  relay the word v after it" — plus **memory recall** (the recalled episode's novel content,
  cue removed).
  - *Fixed* (`ThalamicGate`): the hand-set routes (1,+4), (3,+4), (3,+8) of [09](09-thalamic-attention.md)–[11](11-episodic-memory.md).
  - *Learned* (`LearnedGate`): routes discovered from the predictor's surprises and ranked by
    consistency (`RouteScores`, as in [10](10-route-pool-inhibition.md)); the top 8, refreshed
    every 50 training stories.
- **Striatum:** each channel has a sparse identity code, bound to the current word by bit
  rotation, so the bit-sliced go counters hold a value per (channel, context).
- **Release:** one channel per word (winner-take-one, i.e. disinhibition); only its content
  reaches the predictor.
- **Dopamine:** reward 1 if the released content contained the next word; stochastic
  three-factor update.

## Results (varied stories, held-out pairs, copy-credit predictor of [12](12-dentate-gyrus-ca3.md))

| | Seed 0 | Seed 1 | Seed 2 |
|---|---|---|---|
| fixed routes only, 1–2 / 1–3 facts | 40.6 / 36.4% | 35.6 / 37.0% | 35.8 / 35.8% |
| memory only, 1–2 facts | 100% | 100% | 100% |
| **BG gate, fixed routes + memory**, 1–2 / 1–3 | **100 / 100%** | **100 / 100%** | **100 / 100%** |
| **BG gate, learned routes + memory**, 1–2 / 1–3 | **100 / 100%** | **100 / 100%** | **100 / 100%** |

In all 12 gated runs the gate released **memory at 100% of test answers** and never a route.
Learned route pools (1–2 facts): seed 0 (0,1) (1,2) (2,3) (3,4) (3,3) (2,4) (1,3) (2,1);
seed 1 (0,1) (1,2) (2,3) (3,4) (2,4) (3,3) (3,5) (2,1); seed 2 (0,1) (1,2) (2,3) (0,5) (3,4)
(2,4) (3,5) (3,3). Discovery finds (3,+4), one of the hand-set routes, on every seed.

## Findings
1. **The basal ganglia learn which pathway to open, from reward.** Nothing marks memory as
   the right source at "?"; the per-(channel, context) value does. Routes still get released
   elsewhere in the sentence (where short-range relays predict well), memory at the question.
2. **Nothing is hand-set any more:** routes are discovered, memory is one more channel, the
   choice is learned. Same score as memory alone, so the gate costs nothing.
3. **Where the pieces sit biologically** (see [cortex](../../src/program/cortex.rs) and
   [thalamus](../../src/program/thalamus.rs) module docs): a "route" is really two things.
   Storing context and matching by content is cortical (layer-6 / working memory) or
   hippocampal, an induction head; the thalamus does not store or search a history. What is
   thalamic is *choosing which pathway gets through* (higher-order nuclei, the reticular
   nucleus as an inhibitory gate), and that choice is driven by the basal ganglia. The code
   now follows this split: `ContextBuffer` and match rules live in `program::cortex`, the
   gates in `program::thalamus`.
4. On these stories the decision is easy (memory is right at every question). A harder test
   would mix questions that memory can't answer but a route can, so the gate has to switch.
