# 13 · Big-loop recurrence: chaining recalls for two-hop questions

**Question.** Can recalled content re-enter as the next cue (the EC → hippocampus → EC
"big loop") and answer questions that need two facts chained together?

**Code.** `EpisodicMemory::recall_chain`, `items`, `recall_branches` in
[`src/program/memory.rs`](../../src/program/memory.rs); `Loop` and `Branch` policies and
`TASK=twohop` in [`examples/episodic.rs`](../../examples/episodic.rs).
Run: `TASK=twohop POLICIES=loop cargo run --release --example episodic`.

## Task (bAbI task 2 style)
> then john went to the garden . mary picked up the ball . mary went to the kitchen .
> where is the ball ? **kitchen**

2–3 people, 1–2 moves each, 1–2 objects picked up, in random order; the answer is the
last place the object's holder went. Held-out: (object, place) answers never seen in
training, so the answer has to come from chaining ball → holder → holder's place.
Random fillers between stories; 3,000 training, 1,000 test stories; chance 1/6.

## Mechanisms
- **One hop** (as in 11): cue *ball* → "mary picked up the ball" → *mary*. Not the answer.
- **Loop(2)**, chained recall with inhibition of return: the rarest new item of hop 1
  becomes hop 2's cue, and the hop-1 episode can't be recalled again.
- **Branch(3)**, branching recall: hop 1's new content is split into word-like items
  (bits with the same stored count). Each of the 3 rarest items drives its own hop-2
  recall, like several attention heads. The predictor gets every branch as a frame and
  learns which one carries the answer.

## Results (held-out pairs)

| Policy | Single seed | 5 seeds |
|---|---|---|
| no memory | 17.0% | LOOP_NOMEM |
| one hop (list memory) | 0.0% | LOOP_EPISODIC |
| Loop(1) | 0.0% | LOOP_1 |
| Loop(2), single rarest-item cue | 1.4% | LOOP_2 |
| **Branch(3)** | **81.8%** | LOOP_BRANCH |

Check on one-hop varied stories (does branching hurt?): Branch(3) 97.3% held-out (3 seeds: 95/99/98) vs 99.0% for one hop, so extra branches cost little on one-hop questions.

## Findings
1. **Chaining recalls answers two-hop questions about unseen combinations**, but only
   when the loop doesn't have to guess which recalled item to follow.
2. **Following "the single rarest item" fails** (1.4%). In "then john picked up the
   box", the rarest new items are filler words and "picked up" (frequencies 0.24–0.26),
   about as rare as names (0.15–0.16), so the loop often follows the wrong one.
   Frequency alone can't tell which part of a memory is the entity to follow.
3. **Branching and letting learning choose works** (81.8%). This mirrors transformers:
   several heads retrieve different things, and later computation uses the right one.
   It is also the bit-vector version of the theta–gamma idea in
   [hippocampal functions](../concepts/hippocampal-functions.md#superposition-counts-and-phase):
   alternatives are kept apart and read out separately instead of being blended.
4. Still hand-set: the hop count, the branch count, and the "same count = same item"
   grouping (which relies on sparse random codes).
