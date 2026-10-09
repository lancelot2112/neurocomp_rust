# 98 · A burst gate: apical–basal coincidence in the column, burst rates at the thalamus

**Question.** Reliability in the network is a stored statistic: each kernel's long-run hit
rate, and the thalamic mix's reliability per (source, word-pair context). In the cortex, a
layer 5 pyramidal cell bursts when its basal input (what is happening) and its apical input in
layer 1 (context: feedback from higher areas, the thalamic matrix, memory through association
cortex) coincide (Larkum 2013). Bursts are what drive the higher-order thalamus. Can
reliability be the burstiness of the moment instead of a stored table, with a thalamic gain
that lowers the bar when nothing bursts and the cerebellum taking over when nothing does?

## Code
**Column: burst competition** ([`src/kernel/class.rs`](../../src/kernel/class.rs),
`COMPETE=burst`, `KernelClass::set_burst_competition`):
- **Apical and basal streams.** The column's row is split: basal = the first frame (the input)
  and the last (the previous input); apical = the frames between (top-down, memory). A
  matched kernel that reads both is *coincident*: it can burst.
- **Burst trace.** Each kernel holds a trace (`KernelStats::burst`) of how often its recent
  coincident matches were confirmed by the next input, at rate 1/2: the moment, not the long run.
- **Winner.** The coincident matched kernel with the highest trace at or above a threshold wins.
- **Gain.** The threshold falls by 1/16 at each prediction where nothing passes, and rises by
  1/64 toward 1 when a burst wins. When nothing bursts, the usual ranking (depth, then
  reliability) chooses.

**Thalamus: burst gate** ([`examples/episodic.rs`](../../examples/episodic.rs), `GATE=burst`):
- each source of the mix (column, memory, higher area, cerebellum, …) has a burst rate. Where
  the sources disagreed, the rate moves a quarter of the way toward whether the source was
  right. This happens at test too, as activity rather than learning. No word-keyed table;
- only sources at or above a threshold pass. A passing source's vote weighs −log2(1 − rate)
  (`SourceMix::weight_of_rate`);
- the threshold falls while none passes, rises slowly while some do;
- if none passes, the cerebellum's prediction goes through;
- `GATE_ON=surprise`: rates counted only where the column's own prediction missed, so the
  everyday words ("went", "to", "the") don't inflate them.

## Results
`LEARNING=three SLOW_P=0.25 SLEEP_P=1 REPLAY_PREDICT=1`, five seeds, held-out. The
reference is three systems + sleep + the replay fix ([94](94-sleep-gated-consolidation.md)):

| Entry | Reference | Column bursts | Thalamic gate | Gate, surprise only |
|---|---|---|---|---|
| family consolidated | 47.7 | 41.2 | 32.2 | 36.6 |
| hippocampus teaches cortex | 51.5 | 48.3 | 39.6 | 40.4 |
| index hippocampus | 34.1 | **36.6** | 32.0 | 31.0 |
| engram store | 45.4 | **48.3** | 46.3 | 46.5 |
| semantic store | 38.6 | 34.6 | 39.1 | 39.6 |

On seed 0 of hippocampus teaches cortex (reference: column alone right 21%, mixed 69.5%):

| | Column alone | Mixed | Detail |
|---|---|---|---|
| column bursts | 27.1% | 66.6% | a burst won 7% of predictions; threshold fell to 0 |
| thalamic gate | 21.1% | 57.0% | burst rates: cerebellum 1.0, column 0.48, memory 0.71, top-down 0.19 |
| both | 27.1% | 30.0% | held-out 27.0 |
| gate, surprise only | 17.2% | 44.7% | burst rates: cerebellum 0.67, column 0.0 |

## Findings
1. **Column bursts:** they help the column alone (21 → 27% on seed 0) and two entries (index
   hippocampus +2.5, engram store +2.9). Three entries lose 3–7 points. Bursts decide only
   about 7% of predictions: few matched kernels read both streams, so the threshold
   falls to 0. Most predictions are still made by the ranking.
2. **A per-source burst rate cannot replace the per-context reliability.** At the thalamus, the
   cerebellum wins nearly every disagreement on the everyday words, reaches a burst rate of
   1.0, and its vote swamps the others at the answers (−12 to −16 points on two entries).
   Counting only where the column missed does not fix it, and it is flawed by construction:
   the column is wrong at every counted step, so its own rate goes to 0.
3. **What reliability needs is context.** A burst is a coincidence of input and context *at a
   moment*. A rate averaged over a source's moments loses exactly that. The word-keyed tables
   kept it, as a symbolic shortcut. The plausible version keeps the burst *per moment*: the
   thalamus weighs what bursts now, not what burst on average.
4. Nothing becomes a default.

## Next
- **Bursts as the vote, not a rate:** each source's vote at a step is its current coincidence
  (does its prediction agree with what its apical context predicts now), with no averaging.
- **More coincident kernels:** grow kernels that sample both streams (one basal and one apical
  frame at least), so bursts can decide more than 7% of predictions.
- **The gain on a slower clock:** a threshold that falls to 0 within a story has no effect;
  tie it to the outcome (reward), not to whether something passed.
