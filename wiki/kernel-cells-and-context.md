# Kernel cells, context holding and prediction (the last stretch)

The final work on the earlier branch rebuilt context inference ("which phase of a reversal are
we in?") from kernel cells only. It is the closest thing to this branch's design, so its
lessons are the most directly reusable. Harness: `examples/latent_alone.rs` on that branch;
code: `src/program/kernel_cells.rs`.

## The task

Two lights, A and B. The paying light pays with probability 0.5 and swaps every 500 trials.
Per swap: was it detected (the held context let go), how fast, did it come back to the
context that phase had before, and how many false alarms.

## What the kernel cell became

- A cell is a binary mask over a high-dimensional input. It fires when at least 3/4 of its own
  mask bits are on: the threshold is a fraction of itself, so there is no hidden homeostasis
  variable.
- Inhibition is local: a fired cell holds down any cell whose mask overlaps its own by more
  than half of the smaller mask. The held-down cells that matched learn anti-Hebbian.
- Learning moves mask bits and keeps their number (density). Hebbian moves a bit from an off
  line to an on line; anti-Hebbian moves one from an on line to an off line. Without the
  density rule, cells shrank and fired on anything.
- Births: the part of an input that no winner covers (input AND NOT the winners' masks) gets
  new cells when it meets one cell's popcount (8 bits). Sibling cells born together are dealt
  disjoint shares of the lines, or they inhibit each other at once.
- Output: a fired cell writes its output mask with an op (OR, XOR, AND-NOT, ADD).

## Holding a context

- A context is an assembly: a set of kernel cells sharing a label. An input is explained when
  any of its cells fires.
- A young assembly (fewer than 16 explained inputs) learns whatever it does not explain. A
  settled one switches only when a rival explains inputs it does not, often enough (rival
  drive counts against a strength of bitlen(explained), a u8).
- Plain case (each light-outcome pair its own code, codes overlapping): **19-20/20 swaps
  detected and back, no false alarms**. With 50 distractor inputs: 14-18/20.

## Sequences broke it, and why

When the act is a sequence (light, then a click, then the outcome), the outcome step no longer
carries the light. Every fix that kept "explained = any cell fires" failed (0-4/10):
transition codes, conjunction cells over the history, a fading trace. The generic parts
("paid after the click") are explained in both contexts, so nothing contradicts.

Root cause: **a union of familiar parts looks familiar**. Context lives in the conjunction,
and OR throws the conjunction away.

## Prediction, error, and conjunctive births

1. **A lag chain beats a fading trace.** A trace (rotate, thin by a fixed mask, OR the new input)
   keeps random leftovers of earlier trials, so cells born on it rarely match again. A chain of
   lag stages (each step moves one stage along, rotated by a fixed amount per stage) holds exactly
   the last n steps, and the predictor learned the sequence. "B unpaid" was mispredicted 2 times
   in 105.
2. **The predictor cell.** It is a kernel cell reading the lag context, with an output mask fixed
   at birth: what came unpredicted. Predictor cells do not inhibit each other, so a context
   followed by either of two outcomes predicts both (their OR).
   - If its output came, it moves synapses toward the context.
   - If not, it moves them off, more slowly (chance 1/16 per synapse). Faster fading (1/2) wipes
     out a coin-flip prediction and collapses everything.
3. **Error = input XOR prediction.** It covers what came unpredicted and what was predicted but
   did not come. **The error alone carries no context**: "A paid, surprisingly" and "B paid,
   surprisingly" give identical error bits. The next layer needs the context together with the
   error.
4. **Gate by surprise.** Steps predicted within one cell's popcount are passed on as nothing.
   That cut false alarms from 30-60 to 0-6 per swap.
5. **Conjunctive births.** New cells for a new part are dealt lines from the whole input, so each
   spans context and error and fires only on both. Detection rose from 3-5/10 to 6-9/10 (random
   policy) and 6-7/10 (greedy), with about 10 false alarms per swap. The plain case was
   unchanged.

## Lessons

- Keep density when learning masks; let the threshold be a fraction of the mask.
- Inhibition between overlapping masks, plus anti-Hebbian in the inhibited, separates patterns.
- Represent conjunctions by birth on the whole input, not by OR of part detectors.
- What the next layer needs is the surprise **bound to its context**, gated by its own
  popcount.
- Use exact lags (rotated stages from the history ring), not leaky traces, as context.
- Make failure learning slower than success learning, so "fails sometimes" and "fails always"
  separate without a counter.
- Open: false alarms (~10 per swap), and the next layer cannot see that a step *was* fully
  predicted, which a higher area may need.
