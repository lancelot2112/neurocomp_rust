# neurocomp

Neural computing with high-dimensional bit vectors, popcount-masking kernels and local
Hebbian / anti-Hebbian learning. The question is what a network can build when it is
limited to a handful of cheap operations on bits and allowed to compose them.

## The idea

- **Everything is a bit vector.** Thousands of bits wide, sparse and distributed. Similarity is
  overlap (popcount of AND). Binding, comparison and clearing are bitwise.
- **Every vector keeps a short history** of its frames (current, t-1, t-2, ...). That ring is
  the barebones record of what fired; there is no other hidden state.
- **A kernel is a cell.** It has a binary mask over a source frame (its synapses) and fires
  when enough of them are on:

  ```
  fire  = popcount(source[t-k] AND in_mask) >= threshold      (a fraction of popcount(in_mask))
  write = dest  <op>=  out_mask                               (when it fires)
  ```

  where `<op>` is one word operation:

  | op | effect on the destination |
  |---|---|
  | `OR` | set the mask's bits |
  | `AND` | keep only the mask's bits |
  | `XOR` | toggle them (compare, bind, error) |
  | `AND-NOT` | clear them (inhibit, suppress) |
  | `NAND` | not-and |
  | `ADD` / `SUB` | integer add / subtract of the mask on a word |
  | `ROTATE` | rotate a 64-bit word (shift a code, make a lag) |

- **Learning is local and keeps synapse density.**
  - **Hebbian**: a kernel that fired moves one mask bit from a line that was off to a line that
    was on.
  - **Anti-Hebbian**: a kernel that matched but was inhibited moves one mask bit from a line
    that was on to one that was off. This is how patterns separate.
- **Inhibition is local.** Kernels whose masks overlap compete: the best-matched fires and the
  rest are held down.
- **Kernels are born as needed.** When part of an input meets one kernel's popcount and no
  kernel covers it, new kernels are wired to that part.
- **Composition is the program.** Kernels read the current frame (or an older one) of one
  vector and write the next. Chains of them make delays, traces, predictions, comparisons
  (`XOR` of a prediction with what came) and loops (a kernel writing into what it reads next).
  We do not add task-specific modules; we see what composes.

## Status

This branch restarts from the lean base. What exists:

- `src/bitvec`: the bit vector, its word ops (and, or, xor, not, rotate, shift, slice, mask,
  popcount, density) and `BitVecHistory`, the frame ring.
- `src/kernel`: `SimpleKernel` (input mask, threshold, output mask, op in `Or | And | Xor |
  Clear`; strengthen and search moves), kernel classes and groups.
- `src/program`: a graph of bit-vector nodes and kernel-group edges, built from a small
  instruction program, and a runtime network that steps it.
- `src/cli`: renderers for watching bit vectors.

Next: add `NAND`, `ADD`/`SUB` and `ROTATE` writes, overlap inhibition and births to the kernel,
then rebuild the basic experiments (sequence prediction, pattern separation, context holding)
on this base only.

## What earlier work taught us

An earlier branch (`ccr-8d772201-st7vqw`) grew a large brain-inspired system, with columns,
hippocampus, basal ganglia and a world. Its lessons are distilled in [`wiki/`](wiki/README.md):
what held up, what failed and why, and the mechanisms worth rebuilding here in bit-operation
terms.

## Build

```
cargo build --release
cargo test
```

Rust 2024 edition. Dependencies: `rand`, `petgraph`, `either`, `once_cell`.
