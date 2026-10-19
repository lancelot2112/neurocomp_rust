# Looking inside an evolved network

How to find out what a run's networks actually do, and the order that works. Each tool is a
measure-only mode of `examples/islands.rs` (nothing evolves) run with the run's own settings
(the `set ISL_...` lines of `runs/runN.cmd`) and `ISL_RESUME` pointing at a checkpoint or a rung
log; or a check example built for one question.

1. **Score every genome, not the first** (`ISL_SIGNAL=n`): each resumed genome on n fresh worlds at
   its island's rung and the rung below, with the spread between genomes against the noise across
   worlds (icc, split-half r). A best genome far above the mean with split-half r near 0.9 is a
   real one; then cut the checkpoint into one-genome files and score each to find it. The genome
   a checkpoint lists first is the best by the run's ranking, which need not be the goal probe.
2. **Lesion the one that scores** (`ISL_LESION=n`): the genome scored, then with each population
   taken out in turn, the loss on each subtask as a share of its goal. Losses of both signs on
   every subtask mean a tangle; a few large losses on one subtask mean parts with jobs.
3. **Trace it** (`ISL_TRACE=<island>`, `ISL_TRACE_SEED`): one delayed-conditioning lifetime step by
   step (trial, step, cue, click, every vector's lines on, kernels fired per population). Average
   the vectors by step and trial kind: the vector that still differs once the stimulus is gone is
   the memory; `ISL_TRACE_WORDS=i` prints vector i's words in hex, to see how it is written (an
   add carries and drops lines, an OR or a set only adds them). For reading, `examples/tracecheck` does the same for one hand-made genome.
4. **Draw it** (`examples/netdump`): a checkpoint or rung log as vectors, populations and their
   modulatory inputs, for the dashboard's network view.
5. **Ask what it learns** (`examples/pairscheck`): any genome file on the paired-associate capacity
   curve against a lookup table and a blind control, with ablations of the world (`MIRROR`,
   `BODY`, `ONE_WORD`).

Worked examples: the generalist's tangle and recall's held-channel timer
([experiment 17](experiments/17-the-panel-and-the-learning-rule.md)); the evolved reader
([experiment 18](experiments/18-an-evolved-reader.md)).

## The op report: written and read by (`ISL_OPS_REPORT=1`)

Every 50 generations the log gives each island's elite's ops as `ops ... (written)` (the write op of
each population) and `inputs ... (read by)`: a wire, or the kernels' trigger (at-least, at-most,
band, tile; integer int>=, int<=, int~, int-learned, lanes>, lanes< with the lane width, `/64` the
whole word as a number), `+0` for two-mask kernels that expect zeros too. The same line for a
checkpoint, nothing evolving: `ISL_OPS_REPORT=1 ISL_RESUME=runs/islandsN/checkpoint.txt` with the
run's settings. Runs before 32 log only the written ops.

**The species log and the novelty side file** (from run 36's build). `ISL_SPECIES_LOG=<file>`
appends each novelty species' founder as it is founded: `species n island novelty generation g
useful <probe> score x behaviour <8 numbers>`, then its genome, so useful, behaviourally new
genomes are kept across runs (a library to seed a later run from, as that run's one change). Each
checkpoint now has `<checkpoint>.novelty` beside it, the novelty island's species and archive, and
a resume reads it back, so species and their numbering survive a restart (before, a resumed run
started with none: run 34 resumed numbered its species from 1 again).

**Branches in the diagrams.** `examples/netdump` writes each population's branches as rows of their
own (`kind` branch: the population's id, the vector it reads, the branch's target, its op and, for
a more-branch, its rotation and trigger), and the dashboard draws them as dotted arrows coloured by
op. Before, branches were never exported: the export asked the run's branch switch, which netdump
never sets. A block is labelled `passed <rung>` only when the file's name contains `rungs`.
