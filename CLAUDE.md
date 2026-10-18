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
- **Write ops** (every population picks one; `d` is the destination word, `m` what is written):
  - combining: `OR`, `AND`, `XOR`, `AND-NOT`, `NAND`, `NOR`;
  - arithmetic: `ADD` / `SUB` in lanes of 8, 16, 32 or 64 bits, wrapping within a lane;
  - moving bits: shift left or right, and rotate left or right, by 0 to 63 bits. What is written
    is moved, then ORed in. A word carried from a vector's previous frame and shifted is a
    sliding window.
  - long shifts (`ISL_LONG_SHIFTS`): shift left or right by k bits across word boundaries, the
    bits leaving a word entering the next, then ORed in, so a pattern can travel a whole vector.
  - Within a step, writes into a vector land in the order their populations appear in the
    genome, whatever their ops; moving a population moves its writes. No op has a fixed place.
    Every edge reads past frames only (lag at least 1), so edges into different vectors are
    independent of each other.
  - Any new op must be a single cheap word operation like these.
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
- **The genome is the tag grammar** (`src/evolve/tags.rs`), the one we work on. It is a string
  of tokens:
  - `Channel { tag, width, hold }`: a bit vector of 2^width words (up to 2^11 words, 131,072
    bits). A channel that holds starts each step with its last frame, so only clearing writes
    (CLEAR, AND, NOR, ...) take lines away; otherwise it starts each step empty;
  - `Pop { from, to, ... }`: a population of kernels (or a wire) from one channel to another,
    with all its own properties inline: op, lag, span, scale, starting kernels, fan-in and
    fan-out, threshold, tonic activity, inhibition, births, gate, rates.
  - **Populations cover their whole source.** A population has a scale (tiles of 2^k bits,
    one every half tile, so neighbours overlap by half), not a location. Its kernels are born in
    the tile with the most input no firing kernel reads, their lines move only within their
    tile, and each writes at the same relative place in the destination (topographic). A wire
    copies the whole source. (The older placed field, `at`/`field`/`place`/`spread`, stays only
    for earlier seeds.)
  - Starting cells are drawn fresh each lifetime: the genome sets the kind of cell, not the
    cell.
  - **Execution gates.** A population may run only while a channel (by tag) had lines on, or
    none, a frame ago; paused, it reads, writes and learns nothing and costs nothing. A
    population whose input is all off is paused the same way. So a CLEAR population that empties
    a channel switches off what reads it.
  - **Work** is counted (kernels matched plus words wired), and evolution can prefer less of it
    (`PRB_THRIFT=1`: one lexicase case for less work per step).
  - **Kernel replication.** A population may copy its kernels that keep coming true: a
    predicting kernel (XOR or AND-NOT) is confirmed when every line it wrote was cancelled, a
    repeating kernel when the gate opened after its act; at 2^k confirmations (net, its own u8
    count) it is copied once, in place or to a neighbouring tile.
  - **Circuit copy and plasmids.** A mutation copies a whole circuit (a channel and every
    population touching it) with a private channel, sometimes reading another channel; and half
    of all crossovers move one whole circuit from the other genome instead of cutting the string.
  - **The self channel** (an input port, 256 bits, read-only) is what the network senses of itself,
    written by the runtime each step: last step's work and its kernel count as thermometer codes
    (popcount reads the order of magnitude), the step count in binary (a clock), and how many
    populations ran. It is the network's own state, not the observer's scores.
  Tags name the genome's own channels: a population's `from` and `to` resolve to the declared
  channel nearest in Hamming distance. The world's side is reached through **ports**, with a
  tag space of their own: `from_port` reads the input port nearest `from` (all senses, eye,
  ear, touch, affect, or the self channel); `to_port` writes the output port nearest `to` (the
  whole act, voice or hand). Any string of tokens builds a network.
- Mutation is local: insert, delete, copy or move a token, or change one property or flip one tag
  bit. Every population has a permanent id, which keys its own random stream.
- The edge-list (`genes.rs`) and stack genomes are kept for earlier results only.
- **Improvements come from evolution.** Hand-written genomes are only starting points. Do not
  tune a seed by hand to make a task work. Make the evolutionary system able to find it: enough
  lifetimes per genome for selection to beat noise, held-out lifetimes for every claim, and a
  grammar that can express what is needed.

## Workflow

- Experiments are `examples/<name>.rs`, configured by environment variables, printing a short
  table. Every option defaults to off so earlier results stay reproducible.
- Each experiment that teaches something gets a page in `wiki/experiments/` (or a line in an
  existing page): the question, the setup in bit-operation terms, the command, the numbers,
  and what it means. Negative results are recorded too.
- Report results as numbers against a baseline run in the same harness. Several seeds.
- Run `cargo build --release` and `cargo test` before committing.
- Keep `wiki/lessons.md` current: if a result changes a lesson, edit the lesson.
