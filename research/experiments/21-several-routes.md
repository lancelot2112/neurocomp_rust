# 21 · Several routes needed: L6 opens more than one relay where each is used

**Question.** In [20](20-l6-corticothalamic-gating.md) the L6 gate only had to learn
"memory at the answer, no routes anywhere", because the routes were useless. Can it open
**different relays for different questions**, when no single channel answers them all?

**Code.** `TASK=give` and the options `L6_CONTEXT=pair`, `L6_WARMUP=n` and `L6_WEAKEN=w`
in [`examples/episodic.rs`](../../examples/episodic.rs); `CorticothalamicGate::weaken`
in [`src/program/thalamus.rs`](../../src/program/thalamus.rs).
Run: `COPY_GROW=1 GROW_TRUST=0.5 STICKY=255 GENERALIZE=0.5 GENERALIZE_AFTER=1
TRUST_AT_TEST=0.5 TASK=give POLICIES=l6_gated L6_WARMUP=1000 L6_WEAKEN=0.25 cargo run
--release --example episodic`.

## Task
> mary gave the ball to john . then sandra gave the key to anna .
> what did sandra give ? **key** — or — who got the ball ? **john**

- **Facts:** 2–3 per story, with all names distinct.
- **Questions:**
  - "what did X give ?" is answered by route **(2,3)**: the word two back is X, and
    three words after the earlier X is the object.
  - "who got the O ?" is answered by route **(1,2)**: the word one back is O, and two
    words after the earlier O is the receiver.
- **Memory alone is ambiguous:** recalling the fact sentence returns both the object and
  the receiver.
- **Channels:** the relay pool has these two routes, the three decoys of
  [16](16-thalamic-gate-memory-channel.md) and [20](20-l6-corticothalamic-gating.md), and
  memory, for six channels in all.
- **Held-out:** the question's (name, object) pair is never asked in training. There are
  3,000 training and 1,000 test stories; chance is 1/6.

## Results (held-out, seeds 0 / 1 / 2)

| Policy | Held-out | Channels per word | Open at test answers |
|---|---|---|---|
| memory only | 6 / 6 / 7% | – | – |
| basal-ganglia gate, one channel ([16](16-thalamic-gate-memory-channel.md)) | 58 / 55 / 60% | 1 | always (2,3) |
| all six channels open | 91 / 88 / 99% | 5.5 | all |
| L6 gate, context = current word | 5 / 9 / 8% | 0.00–0.02 | none |
| L6 gate, context = current + previous word | 48 / 48 / 9% | 0.00–0.02 | (2,3) for "give ?" only |
| **L6, current word, warm-up 1,000 + weakening × 0.25** | **100 / 100 / 98%** | **0.71–1.00** | **(2,3), (1,2) and memory at "?"; decoys never** |
| L6, word pair, warm-up + weakening × 0.25 | 82 / 80 / 92% | 0.57–0.77 | (2,3) + memory for "give ?", (1,2) for "O ?" |

## Findings
1. **No single channel answers both questions.** The basal-ganglia gate releases one
   channel per word and always picks (2,3), so it answers only "what did X give ?"
   (55–60%). Memory alone is worse than chance (6–7%): the recall holds both the object and
   the receiver, and the predictor copies the wrong one.
2. **The L6 gate first failed by closing channels too early.**
   - Every channel started at the open threshold, so one weakening step closed it.
   - This happened before L2/3 had learned to read the route, so a route never got the
     chance to prove useful.
   - Keyed on the current word alone, "?" is shared by both questions, so each route is
     useless half the time there and drifts closed.
   - Result: 5–9%, or 9–48% with the word-pair context.
3. **Two developmental fixes make it work.**
   - **Warm-up:** all relays stay open for the first 1,000 training stories while the gate
     learns. Cortex learns to use a relay before the feedback that can close it takes
     effect. Corticothalamic feedback also matures later than the feedforward pathway.
   - **Asymmetric plasticity:** weakening is 0.25 × the strengthening rate, so a relay
     stays open in a context if it is used in at least about 20% of its cases there.
4. **With the fixes, L6 opens several routes at once, and only where they are needed.**
   - Keyed on the current word, the gate opens (2,3), (1,2) and memory together at "?".
     The three decoys are never opened, and about one channel passes per word overall,
     against 5.5 when all are open.
   - The predictor chooses among the open relays, and scores 98–100%, *better* than with
     all six open (88–99%). Fewer irrelevant frames means fewer wrong copies.
   - This is the difference from the basal ganglia: winner-take-one cannot serve two
     question types that share a context; gain control can.
5. **A richer context is more selective but learns slower.**
   - With the previous word in the context, the gate opens one route per question type:
     (2,3) after "give ?", (1,2) after "O ?".
   - But "O ?" is six contexts (one per object), each seen a sixth as often, so it learns
     more slowly and scores 80–92%.
   - Selectivity is better left to L2/3, which already chooses among the open relays.
     L6's job is to remove what is never useful in a context.

## Biology
- **Gain control, not selection:** L6 feedback (with the reticular nucleus) raises or
  lowers the gain of many relays together. It suits "let these through here", while the
  basal ganglia suit "release exactly one".
- **Development:** corticothalamic projections arrive and mature after thalamocortical
  ones. A period when relays pass unfiltered, while cortex learns to use them, matches the
  warm-up.
