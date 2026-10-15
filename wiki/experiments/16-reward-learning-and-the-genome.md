# 16: Learning from reward, and lifetimes that write to the genome (island runs 9 to 11)

Runs 7 and 8 (experiment 15) stopped on two rungs: switching's and shapes' win-stay, and the
maze's turn. Both need learning from reward within a lifetime. This page is about telling a
rung's measure apart from what evolution has to build, and then about letting what a network
finds in a lifetime reach its genome.

## What a hand-written learner shows

Each check is a policy that sees only what a network senses, scored on a rung over 48 fresh
worlds. It shows whether the rung can be learned from the senses at all, and whether the probe
is noisy. The numbers are the mean, and the spread of a 12-world mean.

**Win-stay** (`examples/winstay_check`; serial reversal with no swaps, 200 trials):

| policy | win-stay before the fix | after |
|---|---|---|
| learner (clicks a light unless its last click went unstroked) | 1024 ±0 | 1024 ±0 |
| one light (always the first light it saw; ignores strokes) | 128 ±296 | 0 ±0 |
| random | -19 ±26 | -19 ±26 |

With no swaps a light always pays or never does, so a clicker that keeps to one light had only
stroked clicks or only unstroked ones. Each side with no clicks counted 0, so it scored +1024 or
-1024 by which light it kept to. That is a spread of about 300 in a 12-world mean, around a goal
of 80. Now a lifetime scores win-stay only when it has both. An oracle that never clicks the
wrong light also scores about 0 now, but a learner always has misses.

**Echo2** (`examples/echo_check`; 2 tokens, the teacher saying every one):

| policy | echo |
|---|---|
| echoer (learns each voice code's sound from hearing itself, then repeats what it heard) | 483 ±54 |
| stubborn (always one token) | -6 ±10 |
| random | 10 ±35 |

Echo2's probe is sound and learnable from the senses. Reading's stall there is in evolving the
circuit (a voice reflex, born while babbling, from a heard sound to the move that made it), not
in the measure.

## Run 9: the fixed win-stay probe

Run 8's settings with only the probe fixed (`runs/islands9`, 600 generations in about 2 hours).

Stage moves, each passed on held-out worlds:
- shapes: clean to shift64 by 84, and win-stay at 101;
- recall: its whole ladder, lag1 to lag4 between 97 and 186 (run 8: 210 to 420);
- switching: tell at 226, win-stay at 229;
- reading: echo2 at 20;
- prediction passed cycle1 at 415 and moved to cycle2, the first time in any run.

Screen's dark torch averaged 111 over generations 100-599 (runs 7 and 8: 45-48), best 204.

**Win-stay still averaged 2-3 on both islands** over 340-470 generations. With the probe fixed,
a genome that ignores strokes scores exactly 0, so this is now a clean reading: no genome
learned from reward. The probe hid the size of the step, but the step is real.

## A repeat rung below win-stay

Repeating what paid, with nothing else to learn:
- **The world:** a maze with no stem, the food on one side all lifetime (drawn by the world's
  seed; ladder key `fixed`), and no cue (`nocue`). Which turn pays is learned from the stroke
  alone.
- **The probe:** `repeat`, the 13th. It is how often the next turn repeats a stroked one minus an
  unstroked one, scoring 0 when a lifetime has only one kind.
- **Tests:** a learner policy (turn the other way after a miss, keep a turn that paid) scores
  above 500, and a turner that keeps to one side, or turns at random, about 0.

It is the first rung of the maze's ladder (`learn`) and comes before win-stay in switching's and
shapes'. The network's grammar is unchanged.

## The generalist at the frontier

In the plain world nine of the generalist's twelve probes scored about 0 for every genome (run
8's best: babble 845, tell 293, the rest 0-13), so lexicase found a gradient only in babble and
tell. With `ISL_FRONTIER=1` it is scored on each other island's current stage instead: that
island's probes, in its world, each probe from the first island that has it.

Run 10 has both the repeat rung and the frontier generalist (`runs/run10.cmd`).

## Epigenetic marks and exploration (run 11)

Learning steers evolution already: genomes whose networks learn well are selected. Nothing a
network finds in a lifetime reaches its genome, though. Two switches let some of it through,
without writing learned cells back (the genome sets the kind of cell, not the cell):

- **Exploration** (`ISL_EXPLORE=1`). A population whose kernels all go 64 steps without firing
  is stuck. It grows, once a lifetime, a trial connection: a predicting (XOR), replicating
  population that reads its destination and writes back into its source (a loop), or into the
  destination itself where the source is a port. The trial starts with no kernels. They are born
  where its input is uncovered and replicate when their predictions come true, so a trial that
  works multiplies.
- **Into the genome, tagged.** Each lifetime notes, per population, its kernels' predictions
  that came true, and those of any trial grown from it. The notes travel back through probing (a
  per-thread buffer, since each job runs on one thread) and are written into the genome before
  it breeds. A trial whose predictions came true 16 times or more over a generation is added
  after the population it grew from, tagged `explored`, once per genome.
- **Epigenetic marks** (`ISL_EPI=1`). A population's mark becomes half its old mark plus 16 for
  each doubling of its predictions that came true, and children inherit it. Mutation spares a
  marked population from deletion, and picks it for duplication more often, in proportion to its
  mark. New explored populations start marked.

All of it is local: a kernel's age, a population's own predictions, its own channels. Marks and
the tag are saved with the genome, and older checkpoints read back unmarked (tested). In a
20-generation smoke run, explored connections were already in two islands' elites (7 and 8
populations).

One limit: a population gains a mark only through predictions that come true, which are
defined for the predicting ops (XOR, AND-NOT) by the lines they cancel. Most populations use
other ops, so marks come mostly from explored and predicting circuits.

Run 11 is run 10 with both (`runs/run11.cmd`, started by `runs/after10.cmd` when run 10 ends).
Results to follow.
