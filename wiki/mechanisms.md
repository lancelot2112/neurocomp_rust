# Mechanisms worth rebuilding, in bit operations

Each mechanism that earned its place on the earlier branch, written as vectors, masks and one
write op, so that it can be rebuilt (or evolved) from this branch's kernels. `x` is the input
frame, `S` a kernel's mask, `|v|` the popcount.

## The kernel

- **Fire:** `|x & S| >= theta`, with `theta` a fraction of `|S|` (3/4 worked). Overlap is per
  frame when `x` spans several frames.
- **Inhibit:** a fired kernel holds down every kernel whose mask overlaps its own by more than
  half of the smaller mask. The better match wins; ties are broken by hash.
- **Write:** `dest <op>= out` with `<op>` one of `OR`, `AND`, `XOR`, `AND-NOT`, `NAND`, word
  `ADD`/`SUB`, word `ROTATE`.
- **Hebbian** (fired): move one bit of `S` from `S & !x` to `x & !S`.
- **Anti-Hebbian** (matched but inhibited, or its output failed): move one bit of `S` from
  `S & x` to `!x & !S`.
- **Birth** (coverage): `U = x & !(OR of winners' S)`. If `|U| >= one kernel's popcount`, deal
  `U`'s lines among k new kernels. For a conjunction, deal the whole `x` instead.
- **Self-calibration** (104): a change rate of 2^-k, k up after a confirmed fire and down after a
  contradicted one; the threshold share rises after a wrong fire. No hidden float.

## Composed circuits

| Circuit | Kernels and ops |
|---|---|
| Delay / lag | a kernel reading frame t-k of the history; or `ROTATE` by a fixed amount per stage |
| Exact lag chain (context) | stage j = input(t-j) rotated by j·r; context = OR of the stages |
| Prediction | kernels on the context `OR` their output masks into a prediction vector |
| Error | `XOR` of input and prediction (what came unpredicted, and what was predicted but did not come) |
| Novelty, onset | `present AND-NOT recalled`; `x_t AND-NOT x_(t-1)` |
| Surprise gate | pass the step only when `|error| >= one kernel's popcount` |
| Gating by the prior | `evidence AND expectation`; evidence from several sources is `OR`ed |
| Binding | `ROTATE` by a slot offset (unbind by the inverse) or `XOR` with a key; unbind before a matcher reads it |
| Counting in bits | bit-sliced counter: for plane i, `(p_i, c) = (p_i XOR c, p_i AND c)`; decay drops the low plane (a right shift) |
| Shift-register trace | write `OR`, decay by shift; matches a float trace (12) |
| Fast tag | one-shot `AND-NOT` of a just-confirmed cell's output, cleared at a boundary (22) |
| Two compartments | basal mask on the input, apical mask on the context; a primed (bursting) cell lets silent synapses count |
| Match and confirmation (nested codes) | on a match, a kernel ORs its small code into a match vector; when its prediction comes true, its big code (the small code plus extra lines, same word). Small only = surprise bound to context; big = expected |
| Go / no-go | two masks from the same random start, push-pull Hebbian, recruited up to an asymptote in popcounts |

## History

- `BitVecHistory` frames: current, t-1, t-2, and so on. A kernel's depth is how far back its
  synapses really reach, not the window it was given (10).
- Read frames that are complete. A history read at depth once wrapped onto the frame being
  written (bugs-and-fixes).
- Exact lags beat leaky traces as context. A decaying trace keeps random leftovers of earlier
  steps, so cells born on it rarely match again.

## Rarity without global statistics (open)

Superposed counts need each input scaled by how often it is used (lessons 23, 24). On the
earlier branch that was a 1/n or log(N/n) weight from global counts, which breaks this branch's
locality rule. Candidates that stay local:
- a per-line `u8` write counter that decays, kept by the line itself (presynaptic scaling);
- a homeostatic rate per cell, subtracted from its own drive;
- simply keeping codes sparse enough that no line is common.

## Evolved designs

Circuits evolution found on this branch that do a job in an unplanned way, recorded so they can be
reused, rebuilt or watched for.

### The onset reflex: task by stimulus timing (run 33, selection, generation 566)

**What it does.** In a mixed lifetime (blocks of "click at the lights" and blocks of "babble while
glyphs show") it clicks when the eye was **dark, then lit, then lit with the same code** (106 of 120
such steps in respond blocks, 2 of 1458 other steps) and talks when the eye has **stayed lit**
(45% of steps after four lit ones). A light comes out of darkness for three steps; a glyph stream
never lets the eye go dark. So the two tasks are told apart by the timing of what is seen, with no
use of the task cue, no memory of the task and no reading of the codes themselves (it scores the
same, about 630 to 690 against a goal of 600, cued or uncued, and with lights drawn like glyphs).
14 populations.

**In bit operations** (the parts the lesion map and the genome show; the word-level path that keeps
it from clicking in a continuous stream is not yet traced):
- eye-to-hand kernel populations (XOR, XOR, AND-NOT/clear), each with an **execution pause on the
  eye**: they run only while the eye had lines a frame ago, so they cannot act on a light's first
  step, which came after darkness;
- eye-to-hand SET wires, paused on other vectors, copying the eye's code into the hand;
- a voice path: channel C2 to the voice by XOR at lag 3, gated on touch and paused unless the hand
  had lines a frame ago, with a hand-to-C2 AND-NOT wire (the hand's act clears C2).

**Why it matters.** It uses execution pauses, a per-population gate on a vector's previous frame,
as a temporal-context detector: "was this lit a step ago" costs nothing to compute and decides
whether a population runs at all. Any rung whose tasks show the world with different timing can be
passed this way, which is why selection's held and three now show the same stimuli in every block
(experiment 17).

**Related: the time-share oscillator** (same island, generation 860, 73 populations). A wire from a
channel to itself writing NOR (`C9 = NOR(C9 a frame ago)`) flips every line every step, a period-2
clock; AND-NOT wires from it into the hand let clicks through only on alternate steps, and the voice
acts mostly on the others: two reflexes to a lit eye share the steps instead of being selected by
the task. Removing the self-NOR wire costs 2.3 goals.

Files: `runs/checks/selection-dive/` (genomes, lesion maps, a trace excerpt); experiment 17.

### The phase-reset oscillator: an onset detector (run 34, novelty island, generation 85)

A NOR wire from a channel into itself makes it alternate every step; kernels reading the eye write
into the same channel, and a token unlike the one before shifts the oscillation's phase. A NAND wire
from the channel into the hand turns one phase into a click, so the click lands 2 steps after the
change (17 of 21 breaks in a cycle1 stream; 3 of 106 quiet steps). It scores 669 on cycle1 and
about 0 on cycle2 and cycle3, where the token changes every step. Found by novelty search, whose
descriptor includes clicks after odd tokens. `runs/checks/novelty-dive/`, experiment 17.

### The wrap-around threshold (run 33, screen, generation 934)

Two wires from the eye into the hand writing `sub:3` (subtract in byte lanes) and the hand's own
decoder (a command is the move whose word holds the most lines, at least half a code's) make a
popcount threshold with no kernel: 0 - x in a byte lane wraps to a byte of high lines, so each seen
line becomes many lines in the hand; 8 seen lines (the cursor's mark) give too few to read as a
move, 23 (a glyph under the cursor) enough to read as the click. It passed screen's torch rung as a
click-when-seen reflex (experiment 17).

