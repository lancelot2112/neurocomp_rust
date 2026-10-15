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

## On this branch

- The stack grammar could not express a re-entrant circuit (experiment 05), confirming the
  earlier lesson. `src/evolve/genes.rs` is the gene-list genome: permanent ids, edges naming
  their ends, local mutations. Both front ends compile to the same graph.
- Selecting on reversal needs many lifetimes per genome: with 4, gains on training lifetimes
  (about 180/1024) did not hold out (9–51).
- The tag grammar (`src/evolve/tags.rs`) is the genome from here on. Besides insert, delete,
  segment copy and move, and one-property changes, it has:
  - **split**: a population writes into a new channel one tag bit from its destination, as wide,
    and a wire carries that channel on to the old destination one frame later. The path keeps
    working (one frame later) and gains a stage evolution can then change.
  - **self-loop**: a population writes back into the channel it reads. Clocks (NOR), counters
    (ADD) and held state (OR) are self-loops (experiment 10), now one mutation away.
  - **sparse** as a population property (experiment 09).
  - **hold** as a channel property: the channel starts each step with what it held, so state
    needs no carry wire and lines go only when a population clears them. Off by default; new
    random channels hold a quarter of the time and mutation toggles it. Untested in evolution
    so far. The risk: OR writes pile up in a held channel until a clearing population evolves.
- **Write order is the genome's.** Writes into a vector land in the order their populations
  appear in the string, whatever their ops (before: set and shift, then arithmetic, then
  AND/NAND/NOR, then XOR/AND-NOT). Moving a population moves its writes, so the order is
  evolvable: an XOR can cancel before an OR lands, or an AND can mask what follows it. Every
  edge reads past frames only, so edges into different vectors are independent and could run
  in parallel. New random populations pick any of the twelve op families evenly (before, OR half
  the time). The residue circuit still cancels; 227 of 300 single mutants keep cancelling.
- **Full coverage, no location.** Populations read their whole source in half-overlapping
  tiles at one scale (a gene), are born where input is not yet read, and write topographically.
  Before, a population read a placed field (`at`, `field`), and evolution kept finding eye
  readers on one fixed word: the lights' codes move each lifetime, so those bets won on training
  lifetimes and lost held out (experiment 11 and the 24-lifetime run after it). Starting cells
  are also drawn per lifetime now; before, they were the same in every lifetime of a genome, so
  a lucky draw could be selected like a gene.
- **Islands and ladders** (experiment 13).
  - Islands are populations pressed on different probes. They trade one-population plasmids
    every 20 generations and are checkpointed so a change can be added to a running experiment.
  - A ladder gives an island stages, each with its own world (`Lesson`) and a goal its population
    must hold for 3 generations.
  - Reading climbs babble, then echo, then the teacher fading. Switching climbs respond, tell,
    win-stay, discriminate, then swaps. Recall climbs longer delays. Prediction is scored by
    surprise at a broken cycle. Shapes tell apart distorted lights.
