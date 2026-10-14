# 07. Back to basics: bare and naked genomes, a curriculum, and the search operators

**Setup.** `examples/curriculum.rs`.
- Population 32, the best 8 kept, 8 lifetimes a genome; the best checked on 16 held-out lifetimes
  of the stage's task and of the original reversal (swap every 100 trials of 400).
- Scores are discrimination, out of 1024.
- Stages: `instrumental` (click while the light is on), `reversal1` (one swap per lifetime),
  `reversal` (the original).

**Starting from the context circuit** (population 48, edge genome).
- Even the instrumental stage reached only 13, and evolution shrank the genome from 22 edges to
  7.
- The circuit makes a simple reflex hard to reach: its babblers have no click, and its act
  kernels read 4 lines spread over 8 words.

**Bare starts, instrumental, 30 generations.**

| Grammar, start | Best (training) | Held out |
|---|---|---|
| edge genome, act path | 838 | **665–717** (first run); 174–181 (second run, after the tie-break change) |
| tag grammar, one tonic population | 129 | 19–30 |
| stack, empty program | 0 | 0 |

**Ladder from the bare edge start** (instrumental 30, reversal1 30, reversal 30): instrumental
174 held out, then reversal1 15, reversal −16. The population loses the reflex once the light
swaps.

**Naked starts (empty genomes), instrumental, 40 generations, with the new operators.**

| Grammar | Best (training) | Held out | Size |
|---|---|---|---|
| **edges (no edges at all)** | 256 | **214–225** | 0 → 8–15 edges |
| tags (empty string) | 44 | −17 | 0 → 2–6 tokens |
| stack (empty program) | 0 | 0 | drifts 0–14 |

**What changed along the way.**
- **Ties favour offspring.** Ties had gone to the smaller genome, so from an empty start the
  empty genome always won and nothing drifted.
- **New search operators:**
  - crossover in all three grammars (a quarter of the children);
  - new edges and populations are born active (1–8 kernels; sometimes tonic, growing or gated);
  - stack: swap, copy a segment, delete a segment;
  - edges: copy a whole group onto a fresh vector, and tonic is now mutable.

**Meaning.**
- **The edge genome can grow a learning network from nothing.** With active new genes,
  recombination and drift on ties, an empty edge genome evolved the click reflex on held-out
  lifetimes (214–225). This is the first network evolution here built without a seed.
- **Reversal is still out of reach.** The populations that learn instrumental lose it at the
  first swap.
- **The stack grammar has a hole.** The edge into the act is added after the program runs, so no
  instruction can give it kernels. A program acts only through `WriteOutput` followed by
  `EdgeKernels`.
- **The tag grammar fits its training lifetimes but does not hold out.** Not yet diagnosed.
- Run-to-run variance is large: the same bare start gave 665–717 once and 174–181 the next time.
  Several runs per condition are needed before comparing grammars.
