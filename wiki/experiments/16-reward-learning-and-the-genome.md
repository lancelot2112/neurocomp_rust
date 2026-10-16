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

## Run 14: task selection

`runs/islands14`, stopped after generation 250. The fastest climbs of any run: shapes to repeat by
36, switching to repeat by 59, and recall's whole ladder by 139 (lag4; run 9 at 186). The
selection island's cued rung averaged 296 (goal 600, best 475): above 400 from 50 to 140 while
its best network was mostly loops (10 of 13 populations), then down to 146 as it lost them.
The repeat rung averaged -7 (switching) and 4 (shapes); echo2 -7.

## How far from learning from reward: hand edits (`examples/surgery`)

A measure, not a seed: hand edits to the evolved networks nearest the repeat rung, each scored
on fresh worlds of that rung (goal 300). The edits:
- **aff:** a channel copying the affect port (a wire). A gate names a channel, not a port, so
  this comes first.
- **learner:** a population from the eye to the hand that grows kernels repeating the act made
  `lag` steps back, learning only while the affect copy is on. One channel and two populations
  in all, with the wire.
- **gatehand:** the network's own populations into the hand made such learners. Three property
  changes each, no new population.
- **The control:** the grammar's hand-written act learner alone (`act_seed`) scores about 0 here
  (-33 and 1 on two sets of worlds).

The nearest network was **run 11's shapes, rank 6**: 3 populations, 107 on 32 worlds but -6 on
64 others (the first was luck).

| the network, 64 fresh worlds | repeat | after a stroke | after none |
|---|---|---|---|
| as evolved | -6 | 77% | 86% |
| + aff + gatehand (lags 2-5) | 4 to 24 | 76-82% | 86-88% |
| **+ aff + learner, lag 3** | **197** | **95%** | 90% |
| + aff + learner, lag 4 / 5 | 157 / 141 | 93% | 90% |
| + aff + learner + a babbler | 69 to 98 | 69-71% | 66% |

On the other candidates the learner graft did little:
- run 11's switching: 0;
- run 14's switching: up to 43;
- run 14's maze: about 50;
- run 14's shapes: 0;
- run 11's maze: about 0.

Networks that always make the same turn (100% repeat either way) give it nothing to learn from.

- **The distance is three tokens, needed together:** a channel, a wire from the affect port into
  it, and a gated repeating population with the right lag. Each alone scores about 0, so there is
  no gradient through them one mutation at a time. That is the same wall `maze_check` showed
  (affect into a channel, and a population gated on it).
- **Even then it is win-stay without lose-shift.** The edited network repeats a stroked turn 95%
  of the time, and an unstroked one 90%, because the evolved network's own habit of repeating
  remains. 197 of the goal's 300. A babbler breaks the habit but drowns the learning.
- **Kernel fading on the learner changed nothing** (identical scores), so extinction of unstroked
  repeats did not engage here.

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

## Run 15: the feel rung

`runs/islands15`, stopped after generation 250. The stroke reached the hand on two islands,
weakly: the maze's feel averaged 47 (63 over the last 50 generations, best 97) and switching's
rose from about 0 to 36 over the last 50 (54 at 250). Shapes' stayed at 0 over 218 generations.
None passed (goal 300). Recall climbed its whole ladder by 105, the fastest of any run. A
plasmid from the maze (#145) was in 23 of switching's 32 networks at 200, as switching's feel
score rose.

## Ports for gates, pauses and writes (run 16)

A population could read a sense (`from_port`) and write an output port (`to_port`), but its gate
(it learns only while X had lines on) and its pause (it runs only while X ...) could name only the
genome's own channels. So "learn only after a stroke" took three tokens lined up (a channel, a
wire copying the affect port into it, and a learner gated on that channel), each worth nothing
alone. With `ISL_PORTGATE=1`:
- **`gate_at`, `pause_at`:** a gate or pause may name a declared channel (0), an input port (1:
  all the senses, the eye, ear, touch, affect, or the self channel), or an output port (2: the act,
  voice or hand). A port that is part of a node is checked on its own words only. A new
  population's gate draws its kind a third each; mutation steps it.
- **`to_input`:** a population writing outward may write a sense port instead (the self channel
  stays read-only). What it writes is sensed, with the world's input, a step later; the world never
  reads it. A new population writing outward does so one time in eight.
- Off by default, and written to the genome only when set, so earlier runs draw and read back as
  before (tested).

`examples/surgery` on the same network as above (run 11's shapes, rank 6), the same 64 worlds:

| edit | added | repeat |
|---|---|---|
| affect copy + gated learner, lag 3 | 1 channel, 2 populations | 197 |
| **learner gated on the affect port, lag 3** | **1 population** | **273** |
| learner gated on the affect port, lag 4 | 1 population | -6 |

One population now does what three tokens did, and better: the copy channel cost a frame, so the
port gate lines up with the stroke at lag 3. The goal is 300, and the lose-shift half is still
missing (88% repeats after none).

In a 12-generation smoke run, 322 populations were gated on an input port, 264 on an output port,
62 paused on a port and 13 wrote a sense. Run 16 is run 15 with it (`runs/run16.cmd`).

**Result** (`runs/islands16`, 600 generations in 3 h 22 min):

| | run 16 | run 15 (to 250) |
|---|---|---|
| switching on feel (mean) | **67** (574 generations, best 99) | 11 |
| shapes on feel | **46** (73 over the last 100, best 96) | 0 |
| maze on feel | 36 (69 over the last 100) | 47 |
| screen's glow | **62** (best 163) | 2 |
| prediction | **cycle1 passed at 332** (run 9: 415) | mean 79 |
| recall | lag4 at 353 | lag4 at 105 |

- **Port gates made the stroke reach the hand on all three feel islands at once**, the first time;
  none reached the goal of 300. Screen's glow, about 0 in most runs, averaged 62.
- **The port features spread in turn:** gates first (100% of screen's elite by 50), pauses next
  without any help (selection 100% at 550, screen 50% at 500, recall and prediction 20-30% for a
  time), writing a sense last (5-23% on six islands from 300).
- **A feel circuit moved between islands:** #248 (switching to shapes) was in all of shapes' elite
  as its feel rose from 0, and the score kept rising after the plasmid died out.
- Recall and prediction paid early: recall reached lag1 at 33 but lag4 only at 353.

## Shadow explorers (run 17)

The explorer (run 11) grows a trial XOR loop that writes into the network, and it can grow only
predicting loops between the genome's own channels: no ports, no gates. With `ISL_SHADOW=1` it is
replaced by a **shadow predictor**, which looks for things its population's input can predict
without touching them:
- **What it reads:** what the stuck (or, with accidents, unlucky) population reads.
- **What it predicts**, one target after another from a place drawn from the population's id:
  the eye, ear, touch and affect senses, the self channel, the voice and hand, and every declared
  channel, less what it reads.
- **It writes nothing into the network.** Its kernels fire into a scratch vector of its own.
- **The check, each step, against the target a frame later:**
  - a kernel is right when the residue on the lines it wrote (its prediction XOR what came on)
    has a popcount under its threshold: at least `hit_quarters` quarters of its lines came on (a
    gene of the population, 3 to start, moved by mutation from 1 to 4). Not every line needs to;
  - a right kernel is reinforced (Hebbian: a synapse moves onto a line that was on in what it
    read then) and confirmed, so it can replicate;
  - one that missed is weakened (anti-Hebbian: a synapse moves off a line that was on) and moves
    its output toward lines that came on;
  - target lines no kernel predicted recruit a kernel, on the busiest word of what was read,
    writing those lines.
- **Moving on:** after 128 steps on a target, unless its predictions came true at least 16 more
  often than they failed, it clears its kernels and moves to the next target.
- **Into the genome, tagged:** over a generation's lifetimes, the best target whose net score
  reached 64 is kept, once a genome a generation. A channel of its own (as wide as the target),
  and a predictor population into it, reading what the parent reads and still learning against
  the target (`teacher`), tagged explored and marked. The channel's tag is chosen so that no
  existing tag resolves to it, so nothing else is rewired.

Like CRISPR, the explorer finds a sequence that works and splices a tagged copy into the genome.
This is an internalized prediction: the network gets "what X will be next step" in a channel,
which other populations can read, gate on (with port gates, on the target itself too) or
compare. A predictor's own kernels learn only by the check, not on firing.

Tested:
- a shadow from the eye learns to predict the touch sense when touch follows the eye, notes it,
  and writes nothing;
- the kept predictor arrives with a channel of its own, without moving the first population's
  wiring, and survives the text form;
- the kept predictor writes its prediction into its channel.

In a 16-generation smoke run, predictors were kept on seven of nine islands: of the self channel,
the hand and voice, other channels, the ear and touch. The generalist kept the most (189 in its
elite). Run 17 is run 16 with it (`runs/run17.cmd`, with recall's long ladder).

**Result** (`runs/islands17`, stopped after generation 250, behind run 16): **kept predictors
bloated one island after another, and each did worse while it carried them.** Recall carried 264 at
100 (147 of them predicting its own hand) and stayed on its first rung; it shed them (0 by 200) and
climbed lag1 to lag3 between 169 and 209. Selection went from 0 to 228 between 150 and 200 and
its score fell from 350 to 182; the generalist reached 605 by 250. The feel rung fell to about 0
on all three islands (run 16: 67, 46, 36). A genome keeps one predictor a generation, predictors of
steady targets score well, and their marks protect them, while each one's cost is small, so
selection removes them slowly. Shadow explorers need a limit (on how many a genome keeps, or on
self-targets such as the hand and the self channel) before they can be judged.

## Pauses drawn as gates are (run 18)

In run 16, a port gate on learning spread at once (100% of screen's elite by generation 50), but
port pauses did not: at generation 55, 237 of 4,180 populations had a pause and 4 named a port.
- **A new population never drew a pause.** It drew a learning gate one time in three.
- **A port pause took two mutations:** one setting a pause and another moving `pause_at` to a
  port, which did nothing without the first. 503 populations carried a port `pause_at` with no
  pause.
- **A wrong pause silences a whole population**, where a wrong gate only stops its learning.

With `ISL_PAUSES=1`, a new population is paused one time in four (on a channel, or with port
gates a channel, input port or output port, a third each), and a pause made by mutation draws
what it names in the same step. Tested; off, nothing is drawn. In a 6-generation smoke run, 118
of 752 populations were paused, 103 of them on a port. Run 18 is run 17 with it
(`runs/run18.cmd`).

**Result** (`runs/islands18`, stopped at 548 to start run 19): **port pauses spread at once**,
28-55% of the elite on four islands by 50 (run 16: 4 of 4,180 populations at 55), and screen's
glow averaged 105 over the run (best 201), the highest of any run, from networks of 1 to 4
populations, 67-92% paused on a port. The rest was held back by run 17's kept predictors, which
bloated islands in turn again (the generalist 410, selection 258, switching 155, recall 144):
switching's feel averaged 36 (run 16: 67), recall stalled on lag3, and shapes never left its
first rung.

## Recall's long ladder (runs 17 and 18)

Recall climbed its whole ladder (lag1 to lag4) in runs 9, 14 and 15, the fastest by generation
105. With `ISL_RECALL_LONG=1` its ladder goes on after lag4:

| rung | the delayed trial | what it asks |
|---|---|---|
| lag5 | the light at step 0, the window at 5-7 | hold 5 steps |
| lag8, lag16 | the window opens 8 or 16 steps after the light; the trial lengthened to fit | hold past what a frame of history carries: held state |
| distract | delay 8; on half the trials the other light flashes once during the delay and does not pay (on a no-cue trial it is all that shows) | remember which light came, not that one did |
| vary | each trial's delay drawn from 1 to 8 | hold until the window opens, without timing it |
| two | delay 8; the other light shows on no-cue trials | which light came (the last rung) |

- **The world:** a delayed trial's window starts at step max(5, delay), with the light `delay`
  steps before it, so up to a delay of 5 the trial is what it always was. The draws (a varied
  delay, a distractor) happen only when asked for.
- **The observer:** each event carries the world's window mark, and `remember` counts clicks in
  the window by that mark. Up to a delay of 5 the mark is steps 5-7, so scores are unchanged.
- **Tests:**
  - lag5, lag8 and lag16: a hand-written holder scores above 600, a blind clicker about 0;
  - vary: holding until the window scores, the fixed-timing holder does not;
  - distract: remembering light A scores, remembering any light much less;
  - two: remembering which light scores, remembering that one came about 0;
  - at a delay of 3 the window is unchanged.

It goes into the next runs (17 and 18), since it extends one island's rungs only and leaves the
other islands' comparisons as they were.

## Shadows as indicators (run 19)

Run 17 showed what keeping a shadow's find as a predictor costs: a genome gains one a generation,
predictors of steady targets keep their marks, and islands carried hundreds (the generalist 605)
while their scores fell. With `ISL_SHADOW_HINT=1` a shadow's find is only noted:
- **No predictor is built.** The best target a population's shadow predicted over a generation
  (net 64 true) is written on that population as a **hint**: a tag and what it names (a channel,
  an input port, an output port). It is an indicator of shared information between the
  population's input and that target, and costs nothing at run time.
- **The hint steers mutation:** when the population's gate or pause mutates, half the time it
  takes the hint ("learn only while X", "run only while X"), so the find is one step away for
  evolution to try, and selection decides.
- **Shadows start at random:** only by accident, at each population's own rate (one chance in
  2^k a step, k its gene, drawn 12 to 18), not when the population is stuck, so every population
  is sampled now and then.

Tested: a population's best target becomes its hint and no token is added, and the hint
survives the text form. In a 12-generation smoke run: no predictors, 124 hints (up to 28% of an
island's elite), one gate already on its hint. Run 19 is run 18 with it (`runs/run19.cmd`).

**So far** (generation 342; `wiki/island-runs.md` compares runs 16 to 19): no predictors and no
bloat; the maze's feel averaged 69 over 243-342 and shapes' 53, both their best of any run;
recall reached lag4 at 83, the earliest of any run; selection's cued 355. Hints sit on 11-49% of
populations, but only a few gates or pauses (2-7 an island) have taken one, since a hint acts
only when its population's gate or pause mutates. Switching's feel (49) and prediction (2) are
behind run 16.

**Over all 600 generations** (`runs/islands19/README.md`): recall's lag4 at 83 stayed the earliest
of any run, and it never passed lag4 (mean 40, best 101 of 150). The maze's feel averaged 55 (66
over the last 100, best 118), the best of any run; shapes' 52, switching's 40 (65 over the last
100). Screen's glow 58 (101 over the last 100), selection's cued 292, prediction's cycle1 22 (run
16 passed it). Plasmids: 41 of 522 helped (8%, run 16 4%). No rung was passed after 140. The hints
did not close loops: hinted populations sat on loops less than the rest, and every loop in the
best networks at 550 was a self-loop; run 20 builds edges on hints, toward feedback, and closes
loops directly.

## Edges built on hints (run 20)

In run 19 a hint steered only gates and pauses. That uses the hinted vector as a switch, not the
relationship the shadow measured (the population's input carries information about it). With
`ISL_HINT_USE=1` a hint makes an edge likelier on that path, of any kind:
- **The mutation:** one in sixteen, while any population in the genome carries a hint, adds a new
  population after a hinted one. Its kind (wire or kernels), op, gate, pause and the rest are
  drawn as any new population's; only its endpoints come from the hint.
- **The path:** from the hinted population's input to the hinted vector (to an output port: a
  reflex; to a sense: written, sensed next step beside what arrives; to a channel), or from the
  hinted vector to the hinted population's destination (the vector used as an input). The self
  channel cannot be written, so a hint there goes back only.
- **Output ports are read too** (`from_output`): a population may read the act, voice or hand as it
  was a step ago, so a hint at the hand can also go from the hand to the hinted population's
  destination, the network sensing its own last move. The runtime already kept the act's history;
  only the genome had no way to name it as a source. In run 20 it is also a general mutation
  (`ISL_READ_OUT=1`): a new population reading from outside reads an output port one time in four,
  and a property of its own switches a population between an input and an output port.
- **Hint bands** (`ISL_HINT_BAND=1`): full predictability means the target is derived from the
  source (a copy, or what the network computes from it), which is most of run 19's hints (357 at
  the self channel, 110 at the hand, of 790). A shadow now scores a target by its kernels that are
  partly right, more than `hint_low` quarters of their lines on (a gene, 1 to 3, starting at 1)
  and fewer than `hit_quarters`, less those at or under `hint_low`; fully right kernels count for
  nothing. Learning is unchanged, so a target the shadow learns completely stops scoring, and a
  correlated one keeps scoring. In a 12-generation smoke run: 103 hints, against 159 without it.
- **The self channel written like a sense** (`ISL_SELF_WRITE=1`): a population writing an input
  port may write the self channel, on top of what the runtime writes there each step.

- **Loops closed** (`ISL_LOOPS=1`): a mutation of its own, one in sixteen, picks a population,
  walks its path forward 0 to 3 more populations (each reading where the last wrote, ports
  included), and adds a new population of any kind from where the walk ends back to where it
  starts: a loop of 1 to 4 populations in one step, through channels, the senses (written) or the
  act (read). Until now only self-loops came in one mutation (`self_loop`, one in thirty-two), and
  a loop across channels needed two whose tags lined up.

**Results** (`runs/islands20/README.md`, 600 generations): recall passed lag1 to lag3 by 36, the
earliest of any run; screen's glow averaged 117 and selection's cued 388 (best 589 of 600), both
the best of any run; recall's lag4 averaged 67 (run 19: 40); shapes passed blur2 and shift8 in
the last 30 generations. The feel rung stayed short of run 16 on switching (51 against 67) and of
run 19 on the maze (48 against 55). Loops crossed channels from generation 50 (in run 19 never),
and networks grew and trimmed in cycles: the generalist's best 37, 72, 149, 78, 192, 77, 84, 47,
5 populations at 50-generation checks, at most 184 of 192 on loops through 17 channels; by 550
every island had trimmed to 1 to 8 populations. Reading the act spread (a quarter of populations
on several islands, up to 94% on screen). 54 of 521 plasmids helped (10%, the most of any run).

Run 20 so carries six changes on run 19 (edges on hints, the feedback bias, output reads, hint
bands, self writes, loops closed), at the user's choice; its results are for the set, not any one
of them.

Why so few loops (run 19's checkpoint at 445): 107 of 323 genomes declare no channel (no internal
loop is possible) and 32 one (only self-loops), and at 550 every loop in the nine best networks is a
self-loop. The tasks reward reflexes (nearly every plasmid that helped is "senses to act"); a
cross-channel loop needs two mutations whose tags line up, the first of no use alone; a loop runs
every step and costs work; OR loops saturate and XOR loops oscillate; and a reflex pays from the
first step where a learning loop needs time. And the grammar made self-loops in one mutation
(`self_loop`) but no other loop, which is why every loop seen is a self-loop; run 20's loop
mutation answers that.
- **Biased toward feedback:** where one way feeds back and the other does not, it goes the way that
  feeds back three times in four (else half and half). An edge feeds back when it closes a loop on
  the channel graph (its destination already reaches its source; every channel reaches the self
  channel, which the runtime writes from all the network's work) or runs from a higher level to a
  lower one (a level: the fewest populations from the senses), so a deeper vector is written back
  toward the input.

Why the bias: in run 19's elites at 445 (checkpoint), **hinted populations sat on loops less than
the others,** 45 of 790 (6%) against 1,051 of 4,390 (24%). Hints mostly name the self channel (357),
a channel (316) or the hand (110), and 562 of the 790 (71%) have a way that feeds back: the
generalist 266 of 331, the maze all 116, shapes 87 of 120, recall 65 of 107; screen's hints (81 at
the hand) almost none. Over the elites' best networks, run 19 had 33-35% of populations on a loop at
300-400, against 26-27% in run 16 and 8-20% in run 18; the hints, which only steered gates and
pauses, did not do it.
- It is a plain population: no teacher, no starting mark. It pays its carrying cost and selection
  keeps it or not. Gates and pauses still take hints when they mutate, as in run 19.

Tested: edges built on a hint at the affect sense run eye to affect (written) or affect to the
population's destination, about half each, keeping their own kind; a hint at the hand always runs
eye to hand. Run 20 is run 19 with it (`runs/run20.cmd`, started by `runs/after19.cmd`).

## Heat inputs (run 21)

Every kernel fires when the popcount of its input under its mask meets its threshold (a fraction
of its own mask, the `quarters` gene); wires have none. Nothing moved a threshold during a
lifetime: learning moves mask bits and keeps their count, and a gate or pause only switches a
population on or off. A kernel's firing check already lowered its threshold by a network-wide
"temperature", which the harness always passed as 0 (a global, against the rules).

With `ISL_HEAT=1` a population may carry a **heat input**, as it carries a pause: a tag naming a
channel, an input port or an output port, and whether it warms or cools. Each step, from how full
that vector was a frame ago (popcount `on` of its `bits`), every one of the population's kernels
moves its threshold by `threshold * on / bits`: down when warm (to one line when the vector is
full), up when cool (to twice). Bits and integers only, read like any edge reads, and a fraction
of the kernel's own threshold.

What it could make: a surprise or residue channel that makes a population eager just when its
prediction failed (arousal, attention), and a loop that cools itself as its channel fills (the
homeostasis that would stop an OR loop saturating). A new population has one one time in four;
two properties change it (warm or cool, its tag, or none) and what it names; with hints it takes
the hint half the time. Run 21 is run 20 with it (`runs/run21.cmd`).

## No plasmid exchange (run 22)

Every run since 3 moved plasmids between islands every 20 generations (two into each island), and
some carried a rung: in run 20, #2 (shapes to recall, generation 20, remember +74 on arrival) was
in recall's elite while recall passed lag1 to lag3 between 28 and 36, and #141 (prediction to
shapes, tell +46 at 160) came seven generations before shapes passed clean. But only 4-8% of
plasmids helped on arrival and most were gone within a few generations, so what the exchange adds
over each island's own evolution has not been measured. Run 22 is run 21 with the exchange off
(`ISL_EVERY=0`): the same settings, the islands evolving apart. Compared with run 21 island by
island, it measures the exchange: rungs, goal means, and whether loops and network sizes differ.

## Thrift off (run 23)

Three costs act on work (kernels matched plus words wired a step):
- **Thrift** (`ISL_THRIFT`, on since it was added): one lexicase case per world for less work.
  A rung scored on one goal probe has as many thrift cases as probe cases, so half of all cases
  reward the cheapest genome; the generalist, on all 15 probes, has one in sixteen. The islands
  with one goal probe kept best networks of 1 to 26 populations through run 20, the generalist
  37 to 192 (nearly all on loops); selection reached 510 of its 600 and fell to 273 as its best
  network shrank to one population.
- **Energy** (`ISL_RATION` 256 a step, food for scoring): a hard price on compute; past its
  ration a network must earn by scoring, or rest.
- **The carrying cost** (`ISL_CARRY` 4 a token): added to a genome's work per step (4 x 89 tokens
  = 356 a step for run 20's generalist at 350, against its 2,310 of work). It entered only the
  thrift case and the ranking, never the energy store, so with thrift off it would do nothing.

Since energy already prices compute, thrift is a second penalty, strongest on the islands still
on early rungs, and large networks may be what the higher rungs need (lesson 49). Run 23 is run 21
(exchange on) with thrift off, and the carrying cost made energy (`ISL_CARRY_ENERGY=1`): each step
a network pays 4 a token of its genome from the store its work comes out of, so carrying and
computing compete for the same energy, earned by scoring. It is compared with run 21.

