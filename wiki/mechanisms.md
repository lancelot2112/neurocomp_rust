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

### The sticky hand: a move carried from step to step (run 41, duel, generation 171)

The duel island's best: 185 populations over 21 channels, the largest network a run has kept, but
62 of them matter (taking one out costs at least a tenth of a goal) and the rest ride along (12 do
better removed). Its core is a few wires that carry the act forward: two `sub:5` wires and an
`lsl:32` wire from the act into itself (outputs read) and a `set` wire from the senses into the
hand; each alone, taken out, costs 0.5 to 1.0 of a goal. So its next move is mostly its last: it
repeats its own move 72% of the time, LEFT a third of the rounds. A NOR wire from the ear (the
role tone) into the act, and the eye's view of the partner's move through two channels (C2, C5,
most of its populations) into the hand, bend that habit by role and partner.

Against the yardstick's drawn rules (`ISL_DUEL_PROBE=240`, 800 rounds a lifetime): a sticky move
beats rules that copy or react to it straight and loses to their flipped versions (as predictor
against an evader reacting to its last move 638, flipped -326; as evader against a copier of its
last move -318, flipped 551), but loses less than it wins, through its partial imitation of the
partner (its move is the partner's last 54% of the time as predictor), so it nets about 90 to 140
on the yardstick against a goal of 200. Against patterns, about 0. It scores better in the second
half of each turn than the first (against an evader reacting to its move two rounds back, 344 then
602), and the same with learning frozen: it adapts in activity, not by learning. About half its
populations write the voice, which the duel never scores but the ear hears: the voice is a loop
through the body (46 populations paused on it), a scratch channel. `runs/checks/duel41/`,
`wiki/other-agents.md`.

### The trial latch: one bit held for a whole trial (run 36, recall, generation 530)

The cue sets a state that keeps the hand clicking every step until the trial ends, and the trial's
end resets it; on a trial without the cue it never sets. It holds which kind of trial this is in
activity alone (the same score with learning frozen) and is blind to the delay, so it scored 1024
on lag16 (delays 16 to 32) one generation after lag8. It is a memory of one bit, set and reset by
events, not recall timed to a delay: the remember probe counted a click in the window on cue trials
against no-cue trials and charged nothing for clicking early. `runs/checks/recall36/`, experiment 17.

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

