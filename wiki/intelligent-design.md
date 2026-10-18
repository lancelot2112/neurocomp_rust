# Concept: intelligent design to speed up evolutionary search

Status: tabled (2026-10-18). Not built. Recorded so it can be picked up later.

## The idea

Evolution in this project searches the tag grammar blindly: mutation, crossover and lexicase
selection over thousands of lifetimes. The audits of run 33 (experiment 17) found the search stuck
on rungs whose neighbourhood is flat (no single mutation changes the score) or only falls, and no
evolved genome that learns within its lifetime. A designer that can read the grammar, write a
genome, score it and look at its trace could jump across such plateaus in a few steps where
random mutation needs a rare combination.

So: a few agents (LLM subagents) each work in a loop on one target capability, writing genomes in
the grammar, scoring them, reading traces and lesions, and revising. Their designs then seed
evolution, which keeps, combines and improves them under selection.

## How it would run

1. **A design harness.** One tool an agent calls with a genome text file: the genome's scores on
   chosen rungs on held-out worlds, its learning battery (`ISL_LEARN_BATTERY`: the slope, late less
   early in a lifetime, and the gap, learning on less frozen), a lesion map and a trace. No new
   runtime code: a design uses the grammar's tokens only, as evolution must.
2. **Design loops.** About five agents, one target each, about ten rounds of design, score,
   inspect, revise. Candidate targets, all capabilities nothing has evolved:
   - win-stay learned from strokes (switching, shapes);
   - which light came, held over a delay (recall `two`);
   - a pairs reader for any mapping (the evolved reader reaches 90-97% of a fair lookup at N = 8);
   - order learned from context (order2 on varied shapes, `ISL_VARIED_ORDERS`);
   - selection gated by the task cue.

   A design counts only if it shows a learning gap on held-out worlds, not just a pass (the rungs
   leak; lesson 55).
3. **Seeding a run.** The designs go into the islands and the generalist as starting genomes;
   evolution continues from them. Every result is labelled designed seed against evolved, and the
   learning battery says whether anything learns.
4. **Grammar gaps.** What the designers cannot express in the grammar is written down: what the
   grammar needs for evolution to find these circuits on its own. This may be the most useful
   product.

## Why it is tabled

- It runs against two rules: CLAUDE.md's "hand-written genomes are only starting points; do not
  tune a seed by hand to make a task work" (a design loop is exactly that, done by agents), and the
  earlier decision that an LLM may be a teacher or a world but not the designer of the agents.
- Designed seeds would confound the question the runs ask (can evolution find general learners in
  bit math?) unless every claim is labelled and checked on tasks never designed for.
- It is costly: several agents looping over many design-and-score rounds.

## If it is picked up

- Decide first whether designed seeds may enter a run or stay analysis only (what the grammar can
  express).
- Keep the learning battery and held-out worlds as the only claims; a designed genome that passes a
  rung without a learning gap is a shortcut, as evolved ones were.
- A cheaper first form: designs as analysis only, to find grammar gaps, with no seeding.

Related: experiment 17 (the audits, the learning battery, step 2), lessons 55 and 56, the panel's
memos in [panel/](panel/).
