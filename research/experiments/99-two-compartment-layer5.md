# 99 · Two-compartment layer 5 cells: bursts from input and context together

**Question.** In [98](98-burst-gate.md), bursts used the column's ordinary kernels, which
have one synapse set and one threshold. A kernel was "coincident" only if its mask happened
to include bits of both the input and the context frames, so bursts decided 7% of
predictions. A layer 5 pyramidal cell has two compartments: basal dendrites for the input and
an apical tuft in layer 1 for context. It bursts when both are driven (Larkum 2013). Does a
layer of such cells, whose bursts override the column's habit, let the context-specific
answer win?

## Code
[`src/program/layer5.rs`](../../src/program/layer5.rs) (`Layer5`, with a test); option
`L5=two` in [`examples/episodic.rs`](../../examples/episodic.rs):
- **Two synapse sets per cell, each with its own threshold:**
  - *basal:* 16 bits sampled from the input frames, the row's first (current input) and last
    (previous input);
  - *apical:* 16 bits sampled from the context frames between them (top-down, memory);
  - each set needs 80% of its synapses active. An inverted index maps row bits to cells.
- **A burst** is both sets matching. Context alone does nothing. Input alone is the habit
  L2/3 already makes.
- **Competition.** Among bursting cells, the one with the highest burst trace wins (recent
  confirmations, rate 1/2), then the more reliable. A burst overrides L2/3's prediction and
  sets the column's output confidence to the trace.
- **Fallback.** With no burst, L2/3 speaks. Where it has nothing, the mix's other sources
  decide (the cerebellum under three systems).
- **Learning, gated by bursts:**
  - each bursting cell's trace rises if the next input confirms it, and falls otherwise;
  - after two misses with a low trace, the cell is recycled;
  - **recruitment:** where no burst predicted the next input and both streams are active, a
    new cell is grown on that coincidence (sampled basal and apical bits, the next input as
    its output);
  - consolidation replay (`SLEEP_P`) teaches the layer too.
- Test (`bursts_only_on_input_with_its_context`): the same input bursts toward one output in
  one context and toward another in a second context. It stays silent in a context never seen
  with it.

## Results
`LEARNING=three SLOW_P=0.25 SLEEP_P=1 REPLAY_PREDICT=1`, five seeds, held-out. The
reference is three systems + sleep + the replay fix ([94](94-sleep-gated-consolidation.md)):

| Entry | Reference | Column bursts ([98](98-burst-gate.md)) | **Two-compartment L5** | Per seed |
|---|---|---|---|---|
| family consolidated | 47.7 | 41.2 | 45.4 | 49.6 / 34.2 / 42.2 / 48.4 / 52.4 |
| hippocampus teaches cortex | 51.5 | 48.3 | 48.3 | 52.0 / 24.6 / 60.2 / 52.0 / 52.8 |
| index hippocampus | 34.1 | 36.6 | **45.9** | 35.6 / 43.6 / 55.8 / 65.0 / 29.4 |
| engram store | 45.4 | 48.3 | **47.9** | 39.0 / 49.2 / 59.6 / 39.8 / 51.8 |
| semantic store | 38.6 | 34.6 | **43.6** | 42.4 / 38.6 / 55.2 / 41.8 / 40.2 |

The mean change is +2.8 points.

On seed 0:
- bursts decide 94% of the column's predictions (7% in 98);
- about 9,800 cells are grown and 9,300 recycled, leaving about 510 alive;
- the column alone is right at 46–51% of test answers on three entries (it was 0% before the
  replay fix, and 21% after).

## Findings
1. **Two compartments make bursts the rule, not the exception.** Every cell reads input and
   context by construction, so a burst is available at almost every step.
2. **The context-specific answer now wins in the column.** On three entries the column alone
   answers about half the test questions right. That is what consolidation needed: what the
   slow cortex learned reaches the answer without the general habit hiding it.
3. **The first learned replacement that gains on average** (+2.8; index hippocampus +11.8,
   semantic store +5.0, engram store +2.5; family consolidated −2.3, hippocampus teaches
   cortex −3.2). Seed spreads are wide (one seed at 24.6).
4. **The layer churns.** Almost every grown cell is recycled, and about 500 survive. A cell
   is recycled after two misses with a low trace, which may discard context cells that are
   right only rarely but at the answers.
5. Under three systems the mixed answer does not yet gain as much as the column alone: the
   mix's weights were learned for the old column.

Not a default yet: it needs the default (one-system) suite and gentler recycling first.

## Next
- Gentler recycling: more misses before a cell is removed, or remove only under capacity
  pressure.
- The default suite with `L5=two` (does the burst override help the fast-learning column?).
- The mix: the column's burst as its vote's strength, replacing the word-keyed reliability
  for the column (the per-moment burst of [98](98-burst-gate.md)'s next step).
- Apical synapses that learn (burst-gated plasticity of single synapses), not only whole
  cells recruited and recycled.
