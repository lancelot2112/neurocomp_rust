# Wiki

A condensed record of what earlier work taught us, for the restart on bit vectors and
popcount-masking kernels. The full research wiki (108 experiments, about 1.6 MB) lives on
branch `ccr-8d772201-st7vqw` under `research/`. Experiment numbers here refer to its
`research/experiments/NN-*.md` pages.

| Page | What it holds |
|---|---|
| [lessons.md](lessons.md) | the durable lessons, each with the experiments behind it |
| [mechanisms.md](mechanisms.md) | the mechanisms worth rebuilding, written as vectors, masks and write ops |
| [failures.md](failures.md) | what failed and the root cause, retractions, and plumbing bugs |
| [evolution.md](evolution.md) | genomes and the world, for evolving kernel networks |
| [kernel-cells-and-context.md](kernel-cells-and-context.md) | the last stretch: kernel cells, context holding, prediction error |
| [open-questions.md](open-questions.md) | what is still unknown |

## In one paragraph

Sparse codes and thresholds set from what a kernel actually reads matter more than any
mechanism. Learning works when it is driven by prediction error, keeps mask density, and
separates patterns through inhibition between overlapping cells plus anti-Hebbian in the
inhibited. Context lives in conjunctions, and a union of familiar parts loses it. A prediction
error is only useful bound to what it followed. One-shot stores beat superposed Hebbian stores
on real (skewed) codes. Credit must be causal and short-range. Most apparent wins on the earlier
branch that later fell apart came from:
- hand-written rules;
- leaked test data;
- global statistics;
- seed noise.

This branch avoids all four by construction: bit operations only, local rules only, an honest
world, an outside observer, and several seeds.

## Adding to the wiki

When an experiment changes a lesson, edit the lesson in [lessons.md](lessons.md) and cite the
experiment. New experiments get pages in `experiments/` (question, setup in bit-operation terms,
command, numbers, meaning). Record negative results too.
