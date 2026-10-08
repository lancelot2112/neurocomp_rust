# 10 · Attention by inhibition: a route pool with learned gating

**Question.** Instead of a few channel *slots* that learning swaps routes in and
out of ([09](09-thalamic-attention.md)), can every discovered route stay available
and **inhibition** decide which ones get through? This is closer to the thalamic
reticular nucleus, an inhibitory shell that gates thalamic relay
([Crick 1984; Wimmer et al. 2015](../related-work.md#thalamus-and-attention)).

**Code.** `RouteGate` and `KernelGate` in
[`src/program/thalamus.rs`](../../src/program/thalamus.rs); policies `Gated`,
`GatedValue`, `GatedKernel` in [`examples/thalamus.rs`](../../examples/thalamus.rs).
Commits `c853838` (RouteGate), `8696e98` (growth fix), `11ebba9` (KernelGate).
Run: `JITTER=1 TASK=short|long POLICIES=gated cargo run --release --example thalamus`
(and `POLICIES=kernel_gate` for the kernel gate).

## Setup
- Stories as in 09, with random filler between stories (`JITTER=1`) so no fixed
  cross-story offsets exist, and two story sets:
  - **original:** "X went to the P . … where is X ?"; needs route (1,+4);
  - **long:** facts "X went to the P ." *or* "X went all the way over to the P .",
    questions "where is X right now ?"; needs **two** routes, (3,+4) and (3,+8),
    outside the old (q ≤ 2, v ≤ 6) menu.
- 3,000 training stories, 1,000 test stories with learning off, half on held-out
  name/place pairs. 5 seeds. Chance 1/6.
- **Pool:** every route found by open discovery
  ([`discover_routes` + `RouteScores`](09-thalamic-attention.md#follow-up-open-ended-route-discovery)),
  up to the 64 most consistent, refreshed every 50 stories. Every pool route computes
  its relay every tick.
- **Gate:** routes are inhibited by default. A route opens in the current context only
  once it has proven reliable there (≥ 3 observations, precision ≥ 0.5). Lateral
  inhibition then lets the single most reliable open route through (winner-take-one, a
  hard stand-in for softmax). Its value is the predictor's one relay frame:
  `[current word | gated relay | previous word]`.
- Three gates:
  - **context:** a table of hit/try counts keyed by (route, current word);
  - **context + value:** keyed by (route, current word, relayed word), so the gate can
    judge whether the *retrieved content* makes sense here;
  - **kernel gate:** no table. A predictive `KernelClass` reads
    `[route code | current word | relayed word]` and predicts RIGHT or WRONG, learned
    by the network's own surprise-driven growth and bit-pattern matching. Growth only
    adds a deeper frame when a shallower kernel keeps being wrong, so it can back off
    from route → route+context → route+context+value as each task needs.

## A growth bug found on the way
The first gated runs scored 12–19% even though a trace showed the gate passing exactly
the right route, carrying the right answer, at every `?`. The predictor never used it:
its kernel count froze at 161. Before the gate opened, the relay frame was empty, so
kernels grown at `?` connected only to `?` but were **labeled** with the full depth. One
such kernel per place meant "some kernel at this depth already predicts every answer",
so the sibling rule never grew anything new, and the depth-first winner rule let those
kernels outrank later relay-using ones.

**Fix (commit `8696e98`):** a kernel's depth is how far back its connections actually
reach, and "grow one frame deeper" is skipped when that frame is empty. A single-seed
check then gave 100% held-out. This changed the core growth rule, so the channel
baseline was re-run as well (below). See [bugs and fixes](../bugs-and-fixes.md).

## Results (held-out pairs, 5 seeds)

| Routing | Original, 1–2 facts | Original, 1–3 facts | Long, 1–2 facts | Long, 1–3 facts |
|---|---|---|---|---|
| hand-set routes (oracle) | 100% | 100% | 100% | 100% |
| 4 channel slots + open discovery | 75.0% (100/49/90/100/36) | 70.2% (100/56/92/100/2) | 44.5% (32/55/59/44/33) | 44.4% (27–60) |
| pool + **context** gate (table) | **100%** (all runs) | **100%** (all runs) | 54.4% (49–58) | 54.8% (52–59) |
| pool + **context + value** gate (table) | 31.5% (16–52) | 47.1% (34–52) | **100%** (all runs) | **100%** (all runs) |
| pool + **kernel gate** | 0% (1 seed) | not run | not run | not run |

## Findings
1. **Inhibition beats slots.** With the context gate, every discovered route stays
   available and only the one proven reliable at `?` gets through: 100% held-out on
   the original stories, every run, with no channel count to tune and no
   re-pointing churn.
2. **What the gate conditions on matters, and the two table gates are complementary.**
   - *Context only* can't separate (3,+4) from (3,+8) on the long stories: both are
     right about half the time at `?`, so it lets one through and gets half the answers.
   - *Context + value* can: "(3,+4) brought back a place → open; brought back *way* →
     close". That gives 100% on the long stories.
   - But on the original stories, an exact (route, word, relayed word) table over-fits:
     coincidental routes look reliable for specific place words, and held-out pairs suffer.
3. **The bitwise kernel gate (`KernelGate`, HD pattern matching instead of a table) fails
   so far** (1 seed, original stories, 1–2 facts): 100% on seen pairs, 0% held-out. It
   released local routes ((0,+1), (1,+2), (2,+9)) that let the predictor memorize
   (name, place) pairs, never a binding route such as (3,+4). Its RIGHT/WRONG kernels
   generalize over the route-code + context + value pattern, so a route that is
   "usually useful" for predicting *something* wins over the one that copies the answer.
   It is also very slow (hours per seed).

> **Correction (wiki pass, [errata](../errata.md)):** the remaining kernel-gate runs were never completed; the gate
> was abandoned for the episodic memory of [11](11-episodic-memory.md) and the gates of
> [16](16-thalamic-gate-memory-channel.md) and [20](20-l6-corticothalamic-gating.md).

## Next
- Experiment 11: an autoassociative (episodic) memory that binds facts in one shot and
  completes them from partial cues, with no fixed offsets
  ([open questions](../open-questions.md)).
- Soft passing (more than one winner, weighted) and multi-hop: feed the gated relay
  back as the next query.
