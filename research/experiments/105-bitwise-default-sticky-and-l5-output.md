# 105 · Bitwise layer 5 becomes the default; sticky synapses; layer 5 as the column's only output

**Question.** Three steps after [104](104-bitwise-cells-and-self-calibration.md):
1. Do the fully bitwise, self-calibrating cells hold the whole suite as the default layer 5,
   in place of the strength-based primed cells of [103](103-primed-layer5-becomes-default.md)?
2. With self-calibration, is the sticky mask still needed? Self-calibration works per cell
   (how much the cell changes); stickiness works per synapse (which synapses are protected).
3. **Wiring.** In the cortex, layer 2/3 does not project to the thalamus. It projects to other
   areas (layer 4 of the next area up) and down to layer 5. Layer 5 (thick-tufted cells) drives
   the higher-order thalamus, and layer 6 sets its gain. In our network layer 2/3's prediction
   goes straight to the thalamic mix. Does it work to let layer 2/3 feed layer 5 and only layer
   5 reach the thalamus?

## Code
- **Bitwise default:** the primed layer 5 is `BitCells` with self-calibration (`L5_SYN` defaults
  to bitwise, `L5_META` to on). `L5_SYN=strength` restores the strength-based cells of 103,
  `L5=off` the network without layer 5.
- **No sticky** (`L5_STICKY=0`): no synapse ever becomes sticky.
- **Layer 5 as the only output** (`L5_OUT=1`, [`examples/episodic.rs`](../../examples/episodic.rs)):
  - layer 2/3's prediction is appended to the row as a further input-side (basal) frame of
    layer 5 (`BitCells::set_basal_tail(2)`);
  - layer 5's output, spike or burst, is the column's output to the thalamus. When layer 5 is
    silent, the column sends nothing and the other sources (memory, the higher area, the
    cerebellum) decide;
  - sleep replay gives layer 5 layer 2/3's prediction for the replayed row too.
- **Build:** `.cargo/config.toml` targets x86-64-v3 (hardware POPCNT, AVX2). Building for the
  native CPU broke binaries when the session moved to a machine with a different CPU. Results
  are identical.

## Results
All 25 suite entries, five seeds, held-out. Each variant column is the bitwise default plus that
change:

| Entry | Default of 103 (strength cells) | **Bitwise default** | No sticky | Layer 5 as only output |
|---|---|---|---|---|
| story boundary | 97.2 | 95.3 | 95.2 | 77.6 |
| saccades | 97.8 | 98.2 | 97.2 | 68.2 |
| role transfer | 69.0 | 69.0 | 66.0 | 62.5 |
| sleep generalisation | 76.2 | 77.4 | 76.5 | 70.8 |
| slot memory | 14.1 | 11.5 | 14.1 | 12.8 |
| schema advantage | 72.6 | 73.2 | 74.0 | 66.3 |
| family stated | 43.1 | 45.6 | 50.3 | 41.4 |
| family consolidated | 55.8 | 47.8 | 59.2 | 46.4 |
| semantic store | 66.8 | 68.6 | 68.0 | 60.0 |
| learned stepping | 57.8 | 60.0 | 59.2 | 50.1 |
| closed loop | 58.0 | 60.9 | 60.6 | 50.4 |
| superposed evidence | 57.8 | 62.8 | 60.3 | 53.0 |
| full hippocampus | 45.6 | 38.4 | 46.7 | 37.5 |
| hippocampus teaches cortex | 57.1 | 61.6 | 55.5 | 49.6 |
| index hippocampus | 57.3 | 63.8 | 62.0 | 49.6 |
| engram store | 64.2 | 66.3 | 65.4 | 59.3 |
| engram walk | 64.1 | 65.6 | 66.0 | 60.6 |
| engram walk only | 78.5 | 85.5 | 85.4 | 72.0 |
| infer replay | 59.5 | 67.6 | 62.6 | 59.0 |
| infer read | 46.0 | 48.2 | 48.2 | 22.0 |
| cooperate | 57.6 | 61.8 | 61.0 | 47.7 |
| relations | 63.3 | 64.9 | 66.8 | 49.7 |
| speak | 63.6 | 65.0 | 66.4 | 50.6 |
| speech motor | 63.6 | 65.0 | 66.5 | 50.6 |
| belief decides | 80.3 | 83.0 | 82.1 | 79.9 |
| **mean change** | | **+1.6 over 103** (20 of 25 up) | **+0.3 over bitwise** (9 of 25 up) | **−10.4** (1 of 25 up) |

## Findings
1. **The bitwise, self-calibrating layer 5 is the default** (+1.6 over 103, 20 of 25 up). A
   cell is masks, learning moves bits, rates and thresholds calibrate from outcomes, and it is
   the fastest primed layer. Its losses are family consolidated (−8.0) and full hippocampus
   (−7.2).
2. **Sticky synapses are close to neutral overall** (+0.3 without them, but 15 of 25 entries
   down). They matter in opposite directions:
   - without them, the two entries the bitwise default lost recover (family consolidated
     +11.4, full hippocampus +8.3);
   - others need them (hippocampus teaches cortex −6.1, inference replay −5.0, role transfer
     −3.0).

   Per-synapse protection helps where a cell must keep a fact across interference, and hurts
   where the right synapse must move. Stickiness stays (no evidence to change the default).
   Its rate should calibrate like the others: consolidate more where confirmations repeat,
   less where the context changes.
3. **Layer 5 as the column's only output fails (−10.4).** Layer 5 cells must relearn what
   layer 2/3 already predicts, and a silent layer 5 cuts the cortex out of the answer. In the
   cortex, layer 5's thick-tufted cells also fire on input alone (regular spiking), so layer
   2/3's knowledge reaches the thalamus through them. Ours fire only once a cell has learned
   that pattern. Layer 2/3's direct vote stays.

## Next
- Self-calibrating stickiness (consolidation rate from the repeat rate of confirmations).
- A layer 5 that relays layer 2/3 by default and adds its bursts on top: a cell per layer 2/3
  output that fires on it (the regular-spiking path), so "layer 5 only" loses nothing.
- The family and full-hippocampus losses, and the engram entries.
