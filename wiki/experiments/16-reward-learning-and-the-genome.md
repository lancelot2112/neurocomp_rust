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

**Result** (`runs/islands10`, stopped after generation 341, behind run 9 and with nothing new).
Switching reached repeat at 129 and shapes at 156, but the rung averaged about 0 on all three
islands (switching -12, shapes -4, the maze -2; switching's best was 318 once). Recall reached
lag1 at 268, against lag4 by 186 in run 9. Prediction's and screen's means over the last 100
generations were 23 and 51, against 102 and 107.

`examples/maze_check` shows what the genomes did on the repeat rung:

| island (top 4) | turned on | stroked | repeated after a stroke | after none |
|---|---|---|---|---|
| switching | 99-100% of trials | 44-46% | 76-96% | 80-96% |
| maze | 57-86% | 42-61% | 63-88% | 65-81% |
| shapes | 54-58% | 60-61% | 83-84% | 73-76% |

They choose, but the stroke changes nothing. Acting on a stroke needs two populations at once:
the affect sense copied into a channel, and a repeating population gated on that channel. A gate
names a declared channel, not a port.

## A feel rung below repeat (run 14)

The feel rung is one step smaller. Strokes come at random, whatever the program does
(`World::feel`), and the `feel` probe (a 15th) scores clicks in the 3 steps after a stroke minus
clicks at other steps. That is one circuit, from the affect port to the hand: the population
learning from reward builds on. A policy that clicks when touched scores above 250; clicking
always, at random or never scores about 0 (tested). With `ISL_FEEL=1` it goes before switching's
and shapes' repeat and the maze's learn.

## Task selection (run 13)

One lifetime with one network: blocks alternating a conditioning block (click at a light) and a
reading block (tokens shown), each block's task felt as a cue on the touch channel on the first
rung (`Task::Mixed`). The `select` probe (a 14th) is the mean of responding and babbling, each in
its own blocks, minus the voice in conditioning blocks and clicks in reading blocks. A selector
policy scores above 600, one doing both all the time below 0, a silent one 0 (tested). The
selection island (`ISL_SELECTION=1`) climbs cued, uncued (the task read from the scene), then
short blocks.

The queue: run 11 (marks and exploration) started at 19:09Z when run 10 was stopped; run 12
adds kernels that expect zeros (below); run 13 the selection island; run 14 the feel rung.

## Kernels that expect zeros (run 12)

Every evolved kernel counted only lines that are on. The two-mask kernel (a care mask and the
value expected on each cared line, counting agreement on ones and zeros alike) was in the kernel
from the start, but no population could build one. Absence was reachable only through the write
ops (NOR, NAND, AND-NOT into the destination) or by building an inverted channel (a tonic
population filling it, an AND-NOT one clearing what is on), two populations in order. Win-stay and
repeat turn on "my click went unstroked", the affect lines staying off, which a ones-counting
kernel cannot fire on.

With `ISL_ZEROS=1` a population has one more property, `expect` (drawn for one new population in
four, and flipped by mutation like any other):
- **Births.** A kernel is born on half its lines from the uncovered input that recruited it and
  half from lines of the same tile that are off, expecting each as it is now: "these on, and those
  off".
- **Starting kernels** expect every line off: they fire on an empty input, so a population reading
  the affect port can fire when no stroke came. (A population whose input is all off is not
  paused when its kernels act on nothing, which these do.)
- **Learning** moves cared lines and sets their expected value (Hebbian: expect what is there;
  anti-Hebbian: the opposite), as the kernel always did. A replicated kernel keeps its expectation
  on the moved lines.
- The match is still one word operation a word: `!(a ^ q) & m` where it was `a & m`.

Nothing else changes, and with the switch off, genomes draw, mutate and read back as before
(`expect=1` is written only when set; tested). In a 12-generation smoke run, 67 populations in the
checkpoint expected, up to 36% of an island elite's kernel populations.

Run 12 is run 11 with it (`runs/run12.cmd`).

**Result** (`runs/islands12`, stopped after generation 250, with nothing new on the repeat rung):

| | run 12 | run 11 |
|---|---|---|
| shapes | clean to repeat by 57 | clean until 230, repeat at 250 |
| recall | lag1 at 64, lag3 at 191 | lag1 at 78, lag3 at 102 |
| reading | echo2 at 13, not passed by 250 | echo2 passed at 337 |
| switching on repeat (mean) | -4 (best 103) | 48 (best 231) |
| shapes on repeat (mean) | 0 (best 194) | 0 |
| prediction cycle1 (mean) | 14 | 78 |

- **Expecting zeros sped up the simple rungs**: shapes climbed five rungs by 57, and recall
  reached lag1 earlier than in any run.
- **It did not help learning from reward.** Switching kept zero-expecting populations while it
  climbed (42% at 100), then lost most of them on the repeat rung (2-14%), and the rung
  averaged about 0. Sensing an absent stroke was not what was missing, or not all of it.
- Evolution kept them most where absence is the signal of a scene: 60% of the maze's kernel
  populations at 250, and 50% of screen's.
- Prediction fell from 78 to 14 on cycle1.

Next: run 13 adds freak accidents (below); run 14 the task-selection island; run 15 the feel
rung.

## Freak accidents (run 13)

Exploration grows a trial connection only from a population that is stuck, all its kernels 64
steps without firing. A population that works a little never explores, and busy populations are
often the useful ones. With `ISL_ACCIDENT=1` a population also grows its trial by accident:
- **The rate is a gene**, `accident`: each step, a chance of 1 in 2^k (0: never). A new population
  draws k from 12 to 18, about one lifetime in two down to under one in a hundred (lifetimes are
  1,000-2,000 steps), and mutation moves it a step at a time, so evolution sets how
  accident-prone each population is.
- **The draw** is a hash of the population's id and the step, so it is deterministic by seed and
  takes nothing from any random stream.
- **The trial is the same** as a stuck population's (an XOR, replicating population back into its
  source or onto its destination), still once a lifetime, and kept in the genome the same way:
  16 predictions come true in a generation, then tagged `explored`.

Tested: a tonic population that is never stuck grows no trial with k = 0 and one with k = 1, and
the rate survives the genome's text form. In a 12-generation smoke run, 823 populations carried
a rate.

Run 13 is run 12 with it (`runs/run13.cmd`). (A selection run started as run 13 was stopped
after 14 generations and started again as run 14.)

**Result** (`runs/islands13`, stopped after generation 250, with nothing new on the repeat rung
or echo2):

| | run 13 | run 12 | run 11 |
|---|---|---|---|
| shapes | repeat by 67 | repeat by 57 | repeat at 250 |
| switching | repeat at 144 | repeat at 110 | repeat at 67 |
| recall | lag1 at 138 | lag3 at 191 | lag3 at 102 |
| reading | echo2, not passed | echo2, not passed | echo2 passed at 337 |
| switching / shapes on repeat (mean) | -10 / -2 | -4 / 0 | 48 / 0 |
| prediction cycle1 (mean over the run) | **90** | 14 | 78 |

- **Accidents did not move the repeat rung or echo2.** Evolution kept them rare: the elites'
  mean k settled at 14 to 17, and a few populations mutated to none.
- **Prediction** held the highest cycle1 scores of any run (best 188), without passing.
- **The first plasmid to fix:** #175 (prediction to shapes) was in all 32 of shapes' networks,
  whole, 20 generations after it arrived. Shapes' repeat score did not change. Another, #142,
  gave shapes' repeat +543 on arrival, and was gone 20 generations later.

Next: run 14, the task-selection island.

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

Run 11 is run 10 with both (`runs/run11.cmd`, started by hand when run 10 was stopped).

**Result** (`runs/islands11`, 600 generations in 2 h 35 min):

| island | run 11 | run 10 (to 341) | run 9 |
|---|---|---|---|
| reading | **echo2 passed at 337; echo4 at 355, echo8 at 369; on fade6** | echo2, never passed | echo2, never passed |
| switching | repeat at 67; repeat mean 48 (best 231) | repeat at 129; mean -12 | win-stay at 229; mean about 0 |
| shapes | clean until 230, repeat at 250; mean 0 | repeat at 156; mean -4 | win-stay at 101 |
| recall | lag3 at 102, then stuck (mean 23) | lag1 at 268 | lag4 at 186 |
| prediction (cycle1 mean) | 78 (best 192) | 20 | passed at 415 |
| screen (glow mean) | 16 | 21 | 111 |

- **Reading passed echo2 for the first time in any run**, then two more echo rungs within 32
  generations. Its best then had 87 marked populations and 13 explored; by 580 it had grown to
  189 populations reading five ports. A circuit that echoes predicts its own voice, which is what
  marks count, so this is where marks were expected to help most.
- **Switching's repeat rung held a gradient** (mean 48 over 533 generations, against -12 in run
  10), but no island passed it.
- **The cost:** shapes sat on its first rung for 230 generations, recall stalled on lag3 and
  screen stayed near 0. The generalist bloated to 278 explored connections (69 of its 90
  populations on loops) before shedding most of them.
- Plasmids: 13% of 462 helped on arrival, many on the repeat rung (up to +207). One spread to
  28 of the maze's 32 networks, then died out.

Next: run 12 adds kernels that can expect zeros (below).
