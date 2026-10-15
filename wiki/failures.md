# What failed, and why

The root cause matters more than the symptom: most failures came from a few recurring causes.

## Recurring root causes

| Cause | Where it bit | What it looked like |
|---|---|---|
| Hand-written rules | rollout, question act, "definite" thresholds, familiarity bands, row layout, depth-first ranking (89, 94) | the network chose among options written for it; 68 → 91% came from the rule, not the network; the right candidate lost to the hand ranking at 925 of 1,000 answers |
| Oracles and leaks | text search stored the test stories (57); fixed story lengths let routes answer by position (09) | 67.6% held out was really 17% |
| Global statistics | word-frequency tables, rarity cues, per-source burst rates, novelty measured on the answer (13, 66, 95, 98) | fragile constants standing in for "informative"; a frequent name read as a frame |
| Dense codes | rotation-packed bindings (48), dense top-column output (108) | per-bit novelty unreadable; a cue erased downstream |
| Superposed Hebbian stores | common bits saturate (45, 48, 55) | every cue falls into one generic attractor |
| Unions of parts | "explained if any cell fires" on sequences ([kernel cells](kernel-cells-and-context.md)) | familiar parts explained a new conjunction; swaps went undetected |
| Credit that is not causal | co-activation (08), dense next-step reward (65, 68), long eligibility (108) | at chance, or worse than random |
| Slow learners as default | three learning systems (106) | −10.9; slow learners need more data |
| Dishonest world | untagged demonstrations, masked tones, sensed no-op acts (107, 108) | superstitions; results attributed to the wrong mechanism |
| Fixed-point slips | `reinforce_at` dropped the Q16 scale (108); shift scaling rounds single writes to zero (52) | a whole series of results invalid |

## Things tried and dropped

- **Phase codes** in the hippocampus: worse on one seed (−17) and 3.5× slower (54).
- **Graded mixing** of sources: lost to binary gating (64).
- **Learned competition, learned holds, learned cue controllers**: unstable or worse than the
  hand rule they replaced (65, 68, 91, 97).
  - Credit across about 20 decisions was too thin (91).
- **Rate codes at the thalamus**: averaging threw away the moment's context; one source's rate
  went to 1.0 (98).
- **Layer 5 as the only output** (−10.4) and **primed layer 2/3** replacing the one-shot layer
  (−15.0) (103, 105).
- **Fading (leaky) traces** as context: random leftovers; recency is not relevance (35, kernel
  cells).
- **Anti-Hebbian between whole context assemblies**: stopped any rival from forming ([kernel
  cells](kernel-cells-and-context.md)).
- **Fast fading of failed predictions**: wiped out predictions of things that happen half the
  time ([kernel cells](kernel-cells-and-context.md)).
- **Near-miss pruning in place**: destroyed kernels that copy variable content (84 → 35%, 12;
  31). Spawning a general copy instead kept both (32).

## Retracted or superseded

- 09: first route numbers (position shortcut).
- 12: "pattern separation matters as load grows", and the cross-build comparison.
- 24: "mixing nowhere hurts"; probation, reliability-first ranking and thalamic gates removed.
- 38–49: consolidation figures carry the stale-winner replay bug (94).
- 41: superseded by 42.
- 53: follow-ups backed out in 54.
- 57: text-search numbers corrected in 58.
- 81: a learned policy keeping its accuracy held only in first runs.
- 83: losses were 7–21 points, not 10–47.
- 89: Findings 1–2 withdrawn.
- 101: the primed layer costs about 1.5×, not 11×.
- 106: seed-0 recurrence gain did not hold over five seeds.
- 108: v8 transfer did not replicate; v24–v32 per-act credit results invalid (scale bug).
- Kernel stores called "cortex" learn like a cerebellum: fast, one-shot, conjunctive.

## Plumbing bugs not to repeat

- A history ring read at depth wrapped onto the frame being written.
- Inhibition set by output word but checked by input word.
- A default kernel group added a stray output bit 0.
- A kernel's depth was labelled by its window, not by the frames it actually reads.
- A fast loop ran inside slow learning, which was off at test.
- `HashMap` seeds were random per process (non-deterministic runs).
- A build without hardware popcount.
- A probe compared lags each counted from its own start, so near lags had more chances to match
  than far ones (a blind speaker scored echo 85; experiment 13).
- A curriculum's world settings did not reach the lifetimes the probes ran in (experiment 13).
