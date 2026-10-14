# Evolving bit-vector programs

This branch evolves networks of kernels instead of designing them. Here is what the earlier
branch learned about genomes (51, 52, 86, `src/program/genes.rs` there) and about the world a
program is tested in (107, 108).

## The genome

- **Bit-vector ports compose exactly.** Modules with only bit-vector ports reproduced
  hand-written circuits bit for bit (51, 52). The column was `Delay` + `Concat` + `Predictor`;
  the hippocampus was a 29-instruction genome. Loops carry a one-tick delay.
- **A stack language is easy to read but a poor thing to mutate.** Inserting one instruction
  moves every signal below it, which rewires everything downstream. Mutate a gene list instead:
  - every gene has a **permanent id**, and its inputs name other genes by id and port, not by
    position;
  - a loop is **explicit** (read the source's value from the step before), and the tick order
    comes from a topological sort of the wiring;
  - each gene carries **its own parameters**, so one change changes one module;
  - each gene draws from **its own random stream**, seeded by its id, so adding a gene does not
    shift every other gene's draws (47: a changed draw pattern alone moved results 5 points).
- **Mutation effects (86):**
  - Adding or duplicating a gene was exactly neutral.
  - Rewiring or switching a gene off cost 35–87 points.
  - **Graded wiring** made rewiring smooth: 10 of 12 mutants within 5 points. A connection is a
    gain (the share of the source's bits that pass), which enters at 0 and grows in steps of
    1/16. In bit terms, a gain is a fixed sub-mask of the source.
  - Keep teaching and comparator inputs fixed: no new connections there.
  - Width changes on pass-through genes broke the layout. Keep widths word-aligned and fixed per
    port.
- **Areas can grow by need (84).** A shadow copy (a bud) is promoted by a sign test on how often
  it fixes rather than breaks a prediction. That took story boundaries from 8–17% to 85–96% from
  one starting area, and pruned the useless bud. This is selection inside a lifetime: the same
  test can serve as a fitness check.

## The world

`src/world` on this branch is a lean port of the earlier world, kept to what an evolution loop
needs.
- **Only bit vectors cross.** The program writes an act vector (voice and hand channels); the
  world returns a senses vector (eye, ear, touch, affect channels).
- **A body.** Acts drive motor primitives, random sparse codes given at birth. A command that
  matches none does nothing.
- **No reward in the senses.** Affective touch is a sense like any other. Valuing it is the
  program's business, and evolution's.
- **Events for an observer.** Scoring reads the events (`world::observe`), never the program's
  internals. That is the fitness function.
- **Honest.** A click on nothing is sensed as nothing; one's own voice is heard; the world is
  deterministic by seed.
- **Scenes:**
  - conditioning: Pavlovian, instrumental, delayed, and serial reversal;
  - sequences: tokens shown one per step, where saying the next one counts;
  - echo play: a caregiver answers sounds made aloud.

## Cautions for fitness

- **Seed noise is large (lesson 30).** Evaluate each genome on several world seeds, and compare
  genomes on the same seeds.
- **No shortcuts.** Vary scene lengths and orders so position cannot stand in for content (09).
- **Credit horizons are short.** Reward 25 or more steps away was not learnable within a
  lifetime (107). Evolution can bridge longer horizons than learning can, but the observer
  should still score behaviour where it happens.
