# CLAUDE.md

Guidance for working in this repository. Read `README.md` for the idea and `wiki/` for what
earlier attempts taught us before adding anything.

## The goal

Compute with high-dimensional bit vectors and nothing else. A network is a set of bit vectors,
each with a short history of its past frames, connected by **kernels**: popcount-masking cells
that read one vector and write the next with a single bit operation. Learning is local
Hebbian / anti-Hebbian moves of a kernel's mask bits, with inhibition between kernels whose
masks overlap (pattern separation). We let the network compose these few operations and see
what it can build. We do not add special-purpose modules to get a task done.

## The only primitives

- **Bit vector**: fixed width, stored as `u64` words (`src/bitvec`). Codes are sparse and
  distributed.
- **History**: `BitVecHistory` (a ring of frames: current, t-1, t-2, ...). This is the only
  memory of past firing. Delays, traces and sequences are built from it (a kernel reading
  frame t-k), not from new state.
- **Kernel** (`src/kernel`): an input mask over a source frame, a popcount threshold, an output
  mask, an operation, and a destination vector. When `popcount(input AND mask)` meets the
  threshold (and it is not inhibited), it writes its output mask into the destination with its
  op.
- **Write ops**: `OR` (set), `AND`, `XOR` (toggle / compare), `AND-NOT` (clear), `NAND`,
  integer `ADD` / `SUB` on a word, and a 64-bit word `ROTATE`. Any new op must be a single
  cheap word operation like these.
- **Learning** (density-preserving moves, the mask keeps its popcount):
  - Hebbian, on firing: move one mask bit from a line that was off to a line that was on.
  - Anti-Hebbian, when the kernel matched but was inhibited: move one mask bit from a line
    that was on to a line that was off.
- **Inhibition**: local, between kernels whose masks overlap (more than half of the smaller
  mask). The better-matched one fires; the others are held down and learn anti-Hebbian.
- **Births**: when part of an input is covered by no firing kernel and that part meets one
  kernel's popcount, new kernels are born on it (its lines dealt among them).

## Encoding and decoding

- **Input is raw.** Bytes are written as they are. The network finds its own codes.
- **Sliding windows.** An input vector holds the recent past: each step it shifts by one symbol
  (8 bits for a byte) and the new one is written at the front. A kernel at a fixed offset then
  always sees the same lag, and a wider kernel sees a longer context.
- **Kernels at several scales.** A kernel's window is 8, 16, 32, 64 bits, ... up to the whole
  vector, aligned to its width. Under a word it is a sub-word mask, so it costs one word.
- **Two-mask kernels for dense input.** A care mask and the pattern expected on it; the kernel
  counts cared lines that agree, ones and zeros alike. Sparse kernels (count lines that are on)
  are the special case. Learning moves care lines, setting their expected value.
- **The best match fires first**; kernels whose masks overlap are held down.
- **Decoding: the winner writes the slot.** A predicted symbol is written by the best-matched
  kernel into its slot; predicted XOR actual is the residue.
- **Match and confirmation share one vector, as nested codes.** A kernel that matches writes a
  small code (a few lines in one word) into a match vector. When its prediction comes true (a
  step later) it writes its big code: the same lines plus a few more. A reader sees nothing
  (not this context), the small code (this context, prediction failed: surprise bound to its
  context) or the big code (this context, as expected), and picks which by its threshold.
- **Decode from the winner only**, never from the OR of everything that fired.
- **Kernels write anywhere.** An edge's kernels write its destination vector; the program
  builds the map of vectors.

## Rules

- **No globals.** Every number that shapes behaviour belongs to a gene: a node (its width, its
  spontaneous activity) or an edge (its starting kernels, their synapses, threshold and output,
  its births, its learning rates, its gate's lag, its random stream). The grammar has starting
  values for a new gene; after that each gene carries its own copy, and evolution changes them
  one gene at a time. Nothing is network-wide or genome-wide.

- **Bits and integers only** in the network: no floats, no probabilities, no log-likelihoods.
  A small `u8` counter is allowed where a count is truly needed. A rate is a pattern of bits
  (a recursion, a history window), not a number.
- **Local rules only**. A kernel may use its own mask, its own match, its own firing, and the
  lines it reads and writes. No global statistics: no population means, no normalisation
  across the network, no "best of all cells" except through inhibition.
- **No hand-written task rules or oracles inside the network.** The harness (an example)
  supplies inputs and measures outputs; it never tells the network the answer, the context,
  the phase or the boundary. If a result needs such a rule, say so and treat it as a failure.
- **Thresholds are fractions of the kernel's own mask** (for example 3/4 of its bits), not
  global constants tuned per task. Homeostasis without hidden variables is preferred.
- **Prefer composition** of existing kernels and ops over a new mechanism. Before adding a
  structure, try to express it as kernels between bit vectors with one of the write ops.
- **Cheapest recursion first.** A loop is a kernel writing into a vector it (or its source)
  reads on the next step.

## The world and evolution

- Programs are tested in `src/world`. Only bit vectors cross: an act vector out, a senses
  vector in, each with fixed word-aligned channels.
- The world gives no rewards. Fitness comes from `world::observe`, which reads the event log.
  Never feed the observer's scores, the scene or the phase into the program.
- Keep the world honest: an act that does nothing is sensed as nothing.
- Keep it deterministic by seed. Evaluate on several seeds and compare genomes on the same ones.
- Genomes mutate locally: permanent gene ids, explicit loops, a random stream per gene, and
  graded wiring (see `wiki/evolution.md`).

## Workflow

- Experiments are `examples/<name>.rs`, configured by environment variables, printing a short
  table. Every option defaults to off so earlier results stay reproducible.
- Each experiment that teaches something gets a page in `wiki/experiments/` (or a line in an
  existing page): the question, the setup in bit-operation terms, the command, the numbers,
  and what it means. Negative results are recorded too.
- Report results as numbers against a baseline run in the same harness. Several seeds.
- Run `cargo build --release` and `cargo test` before committing.
- Keep `wiki/lessons.md` current: if a result changes a lesson, edit the lesson.
