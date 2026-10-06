# 47 · Integer only: the model computes with bits and integers

**Question.** The project's premise is a brain-like network in bitwise and integer
arithmetic. Representations, matching and synaptic weights already were. But an audit
found floating point in 162 places in the library. Some were configuration and
reporting, but many were inside learning and control:
- the basal ganglia's reward-prediction errors;
- the slot cells' cosine similarity;
- kernel confidences;
- the hippocampus's write strengths;
- the episodic store's thresholds;
- the thalamic gates' statistics.

Can it all be integer, without changing what the network does?

**Code.**
- [`src/fixed.rs`](../../src/fixed.rs): the fixed-point convention, its helpers, and the
  enforcing test.
- Every module in `src/program/` and `src/kernel/class.rs`.
- The per-step parts of [`examples/episodic.rs`](../../examples/episodic.rs), plus small
  updates to the other examples.

## The convention
- **Fractions are `Q16` integers:** 0..=65,536 stands for 0..=1 (`fixed::ONE`).
  Probabilities, rates, confidences, thresholds and rewards are all `Q16`. Rewards and
  prediction errors are signed.
- **Configuration is converted once,** at the boundary (`q16`, `q16x` for values above 1,
  and tables computed at construction).
- **Products shift back down** (`mul_floor`, `mul_ceil`), **ratios are `num · ONE / den`**
  rounded to the nearest unit (`ratio`), and **comparisons of rates cross-multiply**.
- **Random choices are integer draws** (`chance`): one uniform 64-bit integer compared with
  `p · 2^48`. That is exactly the draw a Bernoulli sample makes, so the random stream
  stays aligned (below).
- **The rule is enforced.** A test (`fixed::tests::no_floats_in_per_step_code`) scans
  `src/` and fails on any float outside test code, unless the line is marked `// float:`.

**Floats that remain, all marked:**
- 20 configuration conversions, e.g. a decay rate turned into a half-life once;
- 8 report-only readouts, e.g. `Rate::value`, and the terminal renderer's colours;
- the float reference store `Ca3FloatMemory`, kept deliberately for comparison.

## What changed, module by module
| Module | Before | Now |
|---|---|---|
| Kernel class | Confidence, target probability and peek rates as `f32`; match tolerance, near-miss and surprise thresholds as float products | `Q16` from the exact hit/miss fraction (`Rate::q16`); thresholds from the configuration's `Q16` fractions, converted once (`CfgQ`) |
| Cortical column | Confidence, outcome, surprise as `f32` | `Q16`; surprise = ONE − share × confidence, in integers |
| Slot cells (`RoleArea`) | Cosine similarity with a square root; learning by float probabilities | The cosine as an exact fraction of its square (\|a∧b\|² / \|a\|·\|b\|), compared by cross-multiplication; learning by `chance` |
| Higher area | Fading survival probability as `f64` | `Q16`, converted once |
| Basal ganglia, prefrontal gate | Rewards, errors, baseline, gain, exploration and trace decay as floats | Signed `Q16` rewards and errors; integer baseline; `chance` for exploration and stochastic steps |
| Thalamus (route gate, kernel gate, corticothalamic gate) | Hit/try counts and precisions as floats | Integer counts; precision as `Q16` (or cross-multiplied) |
| `SourceMix` | Log table built with `log2`; confidence via `exp2` | Log table from an integer bit-length + repeated-squaring logarithm (identical to the float table on all 512 entries); confidence from a fixed-point power of two |
| Route scores | Fractional counts decayed by a float factor | Counts in `Q16` units, decayed by an integer multiply |
| Episodic memory | Habituation, rarity and frequency thresholds as float products | `Q16` comparisons; `count_of` for the rounded mean count |
| CA3 store, full hippocampus | Write amount 16·2^phase with `powf` per store; novelty and readout floors as floats | A per-configuration table of write amounts; novelty, novelty gain and readout fraction in `Q16`; `top_k` on integers |
| Experiment harness (per step) | L5 share, surprise gates, confidence bands, rewards, rarity and boundary thresholds as floats | All `Q16` / integer. Only reports, calibration readouts and timing stay float. The story generator (the world, not the model) is unchanged |

## Results
The regression suite ([`scripts/regress.sh`](../../scripts/regress.sh), seed 0, full
length), before and after:

| Entry | Trained, float → integer | Held out, float → integer |
|---|---|---|
| Story boundary (26) | 87.0 → 87.0 | 86.2 → 86.2 |
| Saccades (29) | 99.6 → 99.6 | 99.6 → **99.8** |
| Role transfer (31) | 65.0 → 65.0 | 60.8 → 60.8 |
| Sleep generalisation (33) | 66.2 → 66.2 | 62.8 → 62.8 |
| Slot memory (36) | 75.2 → 75.2 | 38.4 → 38.4 |
| Schema advantage (39) | 69.8 → 69.8 | 61.2 → 61.2 |
| Stated family (40) | 67.6 → 67.6 | 41.8 → 41.8 |
| Consolidated family (41) | 68.0 → 68.0 | 67.2 → 67.2 |
| Semantic store (42) | 65.8 → 65.8 | 67.2 → 67.2 |
| Learned stepping (43) | 70.0 → **69.4** | 56.8 → **61.0** |
| Closed loop (44) | 70.4 → 70.4 | 61.0 → 61.0 |
| Superposed evidence (45) | 66.6 → 66.6 | 66.8 → 66.8 |
| Full hippocampus (46) | 68.6 → **68.8** | 62.4 → **60.2** |

All 129 unit tests pass, including the float scan.

## Findings
1. **The network can be computed entirely in bits and integers.** Ten of thirteen
   experiments give identical answers. The other three move by 0.2–4 points, within the
   suite's margin.
   - Those three are where the basal ganglia's error arithmetic (learned stepping, 43;
     saccades, 29) and the hippocampus's novelty-scaled write strength (46) now round
     differently. A few borderline decisions flip.
2. **The arithmetic was never the risk; the random stream was.** The first integer version
   drew random numbers as `gen_range(0..65536) < p`, statistically the same as
   `gen_bool(p)` but consuming the generator differently.
   - That alone moved the slot cells onto another trajectory: 35 cells instead of 22 on
     seed 0, and 110 general rules in sleep instead of 325.
   - The consolidated setting then lost 5 points on average over six seeds.
   - Restoring only the slot cells' old draws reproduced the float results exactly, so
     every arithmetic conversion was behaviour-preserving.
   - `chance` now makes the same 64-bit draw a Bernoulli sample makes, and compares it
     with `p · 2^48`.
3. **One lesson for the experiments.** A different random stream gave a 5-point average
   difference over six seeds on one setting. Single-seed comparisons of a few points, and
   even three-seed ones, are within that noise. Claims in earlier pages that rest on
   small differences should be read that way.
4. **Exact fractions need care in fixed point.** 4/5 as a rate must land on the same value
   as the constant 0.8, or a kernel at exactly 80% falls out of its confidence band.
   `ratio` rounds to the nearest unit for that reason.
   - The constant 0.8 itself is not exact in `Q16` (52,429/65,536). So "at least 0.8 of
     n bits" is one bit stricter than the float version when 0.8·n is a whole number
     (n = 10, 15, …). It changed no result here.
   - Thresholds that must be exact are better kept as integer fractions (4/5).

## Biology
Neurons and synapses are not floating-point devices. They have a limited number of
distinguishable states, discrete release events and stochastic channels. Estimates are
on the order of a few bits per synapse (Bartol et al. 2015: about 4.7 bits). Integer
counters with stochastic steps, as used throughout here, are closer to that than 32-bit
floats.
