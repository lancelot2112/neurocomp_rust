# 103 · The default suite: primed layer 5 with interneurons becomes the default

**Question.** Two learned, plausible variants gained under three learning systems:
- primed L2/3, +1.8 ([101](101-primed-l23-vote-interneurons-speed.md));
- primed layer 5 with SST/VIP interneurons, +1.1 ([102](102-burst-key-l5-interneurons-bit-synapses.md)).

The default network uses one learning system, and the suite's 25 entries cover far more than
consolidation. Does either hold on the whole suite, so it can become the default?

**Runs.** Every entry of [`scripts/regress.tsv`](../../scripts/regress.tsv) with its recorded
settings plus the variant, seeds 0–4. The variants:
- `L23=primed L5_VEC=1`;
- `L5=primed INTERNEURONS=1 L5_VEC=1`.

## Results
Held-out, mean of five seeds; the reference is the recorded table:

| Entry | Recorded | Primed L2/3 | Primed L5 + interneurons |
|---|---|---|---|
| story boundary | 94.8 | 64.5 | **97.2** |
| saccades | 96.4 | 65.6 | **97.8** |
| role transfer | 67.2 | 38.9 | **69.0** |
| sleep generalisation | 74.7 | 61.3 | **76.2** |
| slot memory | 19.3 | **27.6** | 14.1 |
| schema advantage | 61.3 | 58.2 | **72.6** |
| family stated | 38.0 | 32.3 | **43.1** |
| family consolidated | 52.0 | 44.1 | **55.8** |
| semantic store | 63.1 | 51.4 | **66.8** |
| learned stepping | 54.5 | 44.0 | **57.8** |
| closed loop | 55.4 | 46.2 | **58.0** |
| superposed evidence | 57.2 | 48.9 | **57.8** |
| full hippocampus | 38.0 | 30.8 | **45.6** |
| hippocampus teaches cortex | 60.9 | 42.6 | 57.1 |
| index hippocampus | 66.0 | 47.0 | 57.3 |
| engram store | 67.4 | 52.1 | 64.2 |
| engram walk | 66.8 | 54.9 | 64.1 |
| engram walk only | 87.0 | 67.9 | 78.5 |
| inference replay | 59.4 | 48.4 | **59.5** |
| inference read | 45.6 | 20.6 | **46.0** |
| cooperate | 54.0 | 36.0 | **57.6** |
| relations | 62.8 | 40.0 | **63.3** |
| speak | 63.1 | 41.1 | **63.6** |
| speech motor | 62.9 | 40.8 | **63.6** |
| belief decides | 82.4 | 70.7 | 80.3 |
| **mean change** | | **−15.0** (1 of 25 up) | **+0.7** (17 of 25 up) |

## Findings
1. **Primed L2/3 fails the default suite (−15.0).** Under three learning systems the old L2/3
   is a slow learner, and the primed cells added what it lacked. In the default network L2/3
   is the fast, one-shot learner the suite was built around. Replacing 99.5% of its
   predictions loses exact one-shot recall: saccades −31, story boundary −30, role transfer
   −28. It stays an option.
2. **Primed layer 5 with interneurons holds (+0.7, 17 of 25 up).** It overrides L2/3 only
   when it bursts: when the input arrives on a cell its context has primed. The largest gains:
   - schema advantage +11.3;
   - full hippocampus +7.6;
   - family stated +5.1;
   - family consolidated +3.8;
   - semantic store +3.7.
3. **Its losses are the engram entries:** index hippocampus −8.7, engram walk only −8.5,
   slot memory −5.2, hippocampus teaches cortex −3.8, engram store −3.2. There, recall
   supplies the answer, and a confident burst from the column can override a memory that was
   right.

## Decision
**Primed layer 5 with SST/VIP interneurons is the default** (vectorized counting included,
which gives identical results). It is learned (Hebbian and anti-Hebbian plasticity, priming,
an inhibitory-circuit threshold) and replaces no rule by another rule: where nothing bursts,
the network is exactly as before.
- `L5=off` restores the old network; `INTERNEURONS=0` and `L5_VEC=0` turn off the parts.
- The regression table now records the five-seed means of this run (seen and held-out).
- The new default reproduces the suite run exactly (checked on family stated, seed 0).

Cost: about 15–25% more time per run.

## Next
- **The engram entries:** a burst that contradicts a confident recall should not win. The
  burst's reliability against memory, learned per burst state, or the hippocampus's novelty
  signal gating layer 5 (acetylcholine: familiar context, trust memory).
- **The bitwise cells and self-calibration** are running on the consolidation entries.
