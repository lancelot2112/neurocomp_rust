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
