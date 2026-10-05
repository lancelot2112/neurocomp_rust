# 19 · Layer 5 as the shared reward: one dopamine signal for every selector

**Question.** Every basal-ganglia selector so far computed its own reward by checking its
own content against the target:
- the thalamic gate ([16](16-thalamic-gate-memory-channel.md)): did the released channel
  contain the next word?
- the hop-2 selector ([15](15-basal-ganglia-selector.md)): did hop 2's recall contain it?
- the prefrontal gate ([18](18-prefrontal-working-memory.md)): did the recall cued by
  working memory contain the answer?

In the brain, the selectors don't get a private answer key. Cortex projects from layer 5
to striatum, and dopamine reports how the outcome compared with the expectation. Can all
three selectors learn from one signal: how well the **column's own prediction** came
true?

**Code.**
- `CorticalColumn::outcome`, `winner_reads` and `outcome_via` in
  [`src/program/cortex.rs`](../../src/program/cortex.rs).
- `BasalGanglia::reward_candidate` with a baseline in
  [`src/program/basal_ganglia.rs`](../../src/program/basal_ganglia.rs).
- In [`examples/episodic.rs`](../../examples/episodic.rs): `REWARD=l5` (the plain outcome) or
  `REWARD=l5_used` (attributed); `BG_BASELINE=rate`; `POLICIES=thalamic_gate`.

## Mechanism
- **L5 outcome:** after each prediction, `outcome(actual)` = `actual`'s share of the
  prediction × the predicting kernel's reliability, in 0..1 (that is, 1 − surprise). One
  number for the whole column.
- **L5 attribution:** `winner_reads(frame)` says whether the winning kernel's input mask
  covers L4 frame `frame`.
  - L4 frames: frame 0 is the current word, then the frames the selectors filled, then the
    previous word.
  - `outcome_via(frame)` = the outcome if the prediction read that frame, else 0.
  - The frames the selectors fill: the released channel (frame 1) for the thalamic gate,
    hop 2's recall (frame 2) for the hop-2 selector, and the recall cued by working memory
    (frame 1) for the prefrontal gate.
- **Advantage credit** for the prefrontal gate, which credits one held load:
  - The error is reward − the running-average reward (rate 0.01), not reward − the load's
    own value.
  - With the old error, the gate failed under L5 reward on 2 of 3 seeds (finding 2).
  - With local rewards the gate still scores 100% on all three seeds under the change.

## Results (held-out pairs, seeds 0 / 1 / 2)

| Selector (task) | Its own reward | L5 outcome | **L5, attributed** |
|---|---|---|---|
| Thalamic gate, fixed routes (varied, 1–2 / 1–3 facts) | 100% every seed | 100% every seed | **100%** every seed |
| Thalamic gate, learned routes (varied, 1–2 / 1–3 facts) | 100% every seed | 100% every seed | **100%** every seed |
| Prefrontal gate (topic) | 100 / 100 / 100% | 18 / 18 / 100%; with advantage credit 82 / 100 / 100% | **100 / 100 / 100%** |
| Hop-2 selector (two-hop) | 86 / 83 / 86%¹ | 86 / 32 / 89%; with `BG_BASELINE` 87 / 42 / 89% | **91 / 69 / 82%** |

¹ Seed 1 rerun on the current build. In [15](15-basal-ganglia-selector.md) it was 90%.

**Answer in recall**, i.e. whether the selection was right:
- **Hop-2 selector:** 91–93% with attributed L5, 89–91% with its own reward, but 30–32%
  on seed 1 with the plain outcome.
- **Prefrontal gate:** 100% with attributed L5.

## Findings
1. **One cortical signal can replace three private answer keys.** With the attributed L5
   outcome, the thalamic gates and the prefrontal gate match their bespoke rewards
   exactly (100% on every seed). The hop-2 selector picks as well as before; the answer is
   in its recall 91–93% of the time. Its mean held-out score is 81% against 85%, with
   seed 1 the weak run (69%) because the predictor misses an answer the recall contains.
2. **A shared reward has to bootstrap.** A selector is rewarded only once the predictor
   has learned to use what it supplies, and the predictor learns to use it only once the
   selector supplies it.
   - The thalamic gates have no trouble: memory pays off at every answer.
   - The prefrontal gate with the plain L5 outcome stalled on 2 of 3 seeds. The mean L5
     reward at the question was 0.14. Each load's bandit estimate was pulled toward those
     small rewards, so "to" fell to 0.32–0.38, below keep's untrained 0.5. The gate stopped
     loading, and the predictor never learned to use the slot.
   - Crediting by advantage fixes it. The held load gains when holding it did better than
     usual, so "to" climbs to 1.00 while every reward is still small.
3. **A global reward is too noisy when the choice rarely matters.** The plain outcome
   rewards the hop-2 choice at every word. Most words ("went", "to", ".") are predicted
   without hop 2, so the reward barely depends on the choice. On seed 1 the selector
   locked onto the wrong item: the answer was in its recall only 30% of the time.
   - A running-average baseline doesn't fix this (42%), since the noise is in the reward,
     not in its offset.
   - Attribution does fix it. Reward only when the prediction read the choice's frame,
     and the answer is back in the recall 91% of the time.
   - This is credit assignment by *use*, the same idea as copy credit
     ([12](12-dentate-gyrus-ca3.md)), now passed from cortex to the basal ganglia.
4. **Attribution is local.** It needs only which input bits the winning kernel's mask
   covers, which the column already has. No selector sees the target.

## Biology
- **Layer 5 to striatum:** thick-tufted layer 5 neurons project to the striatum (and to
  thalamus and brainstem). Corticostriatal synapses are where dopamine-gated plasticity
  happens, so a selector's credit needs both a cortical "this was used" signal and a
  dopamine "this went well" signal: a three-factor rule. `outcome_via` is the product.
- **Dopamine as advantage:** dopamine neurons fire to outcomes better than expected, not
  to rewards as such ([Schultz, Dayan & Montague 1997](../related-work.md#credit-assignment)). The advantage credit, reward − running average, is that
  error.
- **What isn't modelled:** a separate critic (ventral striatum) that predicts the reward
  per context; here the expectation is one running average or the choice's own value.
