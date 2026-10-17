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

**Results** (`runs/islands21/README.md`, 600 generations): recall passed every delay rung, lag1
to lag16, by generation 88, and distract at 104; no earlier run had passed lag4 (runs 19 and 20
stuck on it for about 500 generations). Prediction's cycle1 averaged 58, the best of any run
(run 20: 20), and shapes climbed blur2 to feel in 13 generations late in the run. Heat inputs
spread on every island at once, warm on recall, prediction, selection and the generalist, cool on
the maze and reading. The feel rung went backward (switching 7, the maze 24, against run 20's 51
and 48), and selection's cued fell from 388 to 274. That a delay line gets long by heat (a
recent-activity vector making a population easier, or harder, to fire) is a guess the run cannot
test alone; runs 22 to 25 keep heat inputs and change other things.

## No plasmid exchange (run 22)

Every run since 3 moved plasmids between islands every 20 generations (two into each island), and
some carried a rung: in run 20, #2 (shapes to recall, generation 20, remember +74 on arrival) was
in recall's elite while recall passed lag1 to lag3 between 28 and 36, and #141 (prediction to
shapes, tell +46 at 160) came seven generations before shapes passed clean. But only 4-8% of
plasmids helped on arrival and most were gone within a few generations, so what the exchange adds
over each island's own evolution has not been measured. Run 22 is run 21 with the exchange off
(`ISL_EVERY=0`): the same settings, the islands evolving apart. Compared with run 21 island by
island, it measures the exchange: rungs, goal means, and whether loops and network sizes differ.

**Results** (`runs/islands22/README.md`): 15 rungs passed against run 21's 16. The exchange had
carried recall up its delay rungs (lag4 at 73 in run 21, at 400 here; lag16 at 88 against 582)
and shapes off its first rung (104 against 245), but without it most islands did as well or
better within a rung: prediction's cycle1 averaged 78 (run 21: 58, the best of any run),
switching's feel 35 (7) and the maze's 45 (24); screen and selection about level. Loops and heat
inputs came as in run 21 on every island. So the exchange buys speed on a ladder whose rungs a
neighbour's circuit can fill (a delay line, a shape's reflex), and costs the islands whose task
no neighbour solves; from run 23 on it stays off.

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

**Result** (stopped at 296; `runs/islands23/README.md`): worse than run 22 everywhere but
selection. Without thrift every island's networks grew at once (reading's best 277 populations at
generation 50, 684 at 200; selection's 320 at 250, against 2 to 31 for run 22's at 50), and the
growth bought nothing: 4 rungs passed by 296 against run 22's 8, the first reflex rungs 100 to 190
generations late (switching's tell at 123, recall's lag1 at 231), prediction's cycle1 at 0 the
whole run (run 22: mean 62). Paying for carrying out of energy did not hold size down. Lesson 49
stands (size is not a problem by itself), but size with nothing pruning it is: small networks pass
the first rungs, and with no pressure toward them there are none. Thrift stays on from run 24, and
runs 26 and 27 try gentler ways to keep big networks: size classes, and thrift relaxed while an
island is stuck.

## Is there a gradient on the hard rungs? (signal, after run 20)

The harder rungs stall for hundreds of generations (run 20: echo2 at -5 over 587, cycle1 at 20
over 600, feel at 48-51, lag4 at 67). `ISL_SIGNAL=48` with `ISL_RESUME=runs/islands20/checkpoint.txt`
(run 20's settings, measure only) scored each island's 32 genomes on 48 fresh worlds on its rung's
goal probe, and on the rung below. The icc is the share of one world's spread that is between
genomes (what one lexicase case sees); the split-half r sets genome means on odd worlds against
even ones (whether a mean ranks genomes).

| island, rung | mean | best | icc | split-half r |
|---|---|---|---|---|
| reading echo2 | -26 | 137 | 0.00 | -0.46 |
| prediction cycle1 | 5 | 47 | 0.00 | 0.14 |
| maze feel | 51 | 97 | 0.05 | 0.82 |
| shapes shift8 | 96 | 164 | 0.05 | 0.67 |
| switching feel | 48 | 115 | 0.16 | 0.94 |
| screen glow | 136 | 214 | 0.30 | 0.90 |
| recall lag4 | 117 | 261 (goal 150) | 0.52 | 0.97 |
| selection cued | 305 | 650 (goal 600) | 0.85 | 1.00 |
| reading babble (passed) | 546 | 1024 | 0.63 | 0.98 |

Three kinds of stall:
- **No gradient** (echo2, cycle1): the differences between genomes are noise, so selection is
  random there. They need graded probes or a smaller step.
- **A gradient lexicase does not see** (feel, shapes): means over many worlds rank genomes, but one
  world is mostly noise, and lexicase takes cases one world at a time (on the maze most genomes
  tie on most worlds: epsilon 0). Cases pooled over a few worlds would keep the signal.
- **No gradient problem** (lag4, cued): the best genomes pass on fresh worlds, but a rung's goal
  is checked against the mean of the whole population, mutated children included
  (`src/evolve/mod.rs:1047`), before the elite's held-out check; that mean stays near 117 and 305.

## Goals judged on the elite, and partial credit (run 24)

From the signal measurement above, two changes, each behind a switch:
- **The elite's pass** (`ISL_ELITE_PASS=1`): a rung's goal is checked on the mean of the island's
  elite (its best 8 by summed goal probes, on the training worlds), not of its whole population,
  before the elite's held-out check as before. The goals line still shows the population's mean.
- **Partial credit** (`ISL_PARTIAL=1`): in selection, echo, surprise and feel are replaced by
  graded versions; a rung's goal is still checked on the probe itself, so a rung is passed only
  for the ability.
  - `feel~`: clicks 1-3 steps after a stroke in full, 4-6 at half, 7-9 at a quarter, each against
    the click rate with no stroke in the 9 steps before (a clicker 5 steps late scores 150, where
    feel gives it about 0);
  - `echo~`: echo's matches with lags 5-8 at half (against 13-16), per taught token shown rather
    than per token said, so a network saying a few tokens is not scored on a handful;
  - `surprise~`: clicks 4-6 steps after a token at half as well.

With them, **passed rungs kept** (`ISL_KEEP_RUNGS=1`): a rung lists its own probes (most keep
the earlier probes, as respond and tell), but in its own world only, so lag5 never tests a delay of
2 again, and shift64 never a clean light. Each generation an island above its first rung is now
scored on one of the rungs it passed as well, drawn at random, in that rung's world, and those
rows join its lexicase cases, so a climb keeps what the rungs below asked and each rung builds on
the ones before. The rung's goal is still judged on the current rung's world alone. One more
probing an island a generation.

Run 24 is run 23 with the three (`runs/run24.cmd`), after run 23.

**Result of run 24** (stopped at 409 to start run 25; `runs/islands24/README.md`): 11 rungs passed
against run 22's 14 by 409. Judging on the elite moved islands up as soon as their best passed
(recall's lag1 to lag3 at 17 to 19, shapes' blur2 to feel at 229 to 272, the first rungs sooner
than in any run), but within a rung they then scored far below run 22 (switching's feel mean 5
against 31, the maze's 15 against 37, recall's lag3 25 against 84) and recall never passed lag4:
moving on before the population has the rung leaves little of it to build on. The networks stayed
small (most islands' best 1 to 15 populations at 400). Three islands spent the second half on feel,
whose goal turned out to be out of reach (below).

## Inhibit inputs (run 25)

Inhibition among a population's own kernels was already a gene (`share` turns it off): a kernel
whose mask lies more than half on lines a better-matched kernel took is held down and learns
anti-Hebbian. Nothing outside the population could hold it down. With `ISL_INHIBIT=1` a population
may carry an **inhibit input**, as it carries a gate, a pause or a heat input: a tag naming a
channel, an input port or an output port. Each step the lines that vector had on a frame ago are
taken before any of the population's kernels fires (folded onto its input words in order), so a
kernel whose mask lies more than half on them is held down, as if a rival had fired there.
Unlike a cool heat input, which raises every threshold in the population, it suppresses only the
kernels looking at the lines that vector is active on: inhibition between populations, a
winner-take-all across them, a loop quietening what feeds it. A new population has one one time
in four; two properties change it and what it names; with hints it takes the hint half the time.
With it, **the overlap rule as a gene** (`ISL_OVERLAP=1`): each population's `inhibit_quarters`
(1 to 4, starting at 2) sets how much of a kernel's mask the taken lines must cover to hold it
down, more than that many quarters (2 is the old "more than half"; 4: never), for its own kernels
and its inhibit input alike. The thought behind it: reading may need several kernels matched on
the same spot (an echo of one sound by more than one code), which the half rule forbids unless
`share` turns inhibition off for the whole population. Run 25 is run 21 with both
(`runs/run25.cmd`), after run 24.

From run 23 on the plasmid exchange stays off (`ISL_EVERY=0`), at the user's choice after run 22's
first half: without it run 22 was level with or ahead of run 21 on every island but recall (whose
fast delay rungs the exchange helped), and reached shapes' feel rung 183 generations sooner. Run 23
is so run 22 with thrift off and carrying paid as energy, compared with run 22; runs 24 and 25
build on it as before.


## Size classes (run 26)

Thrift is one pressure on size for a whole island: on (run 22) the networks stay small, 2 to 31
populations at generation 50; off (run 23) they all grow, 28 to 277, and the first rungs (simple
reflexes) come late. Either way the population has one size. With `ISL_SIZE_CLASSES=1` lexicase
keeps a **mixture**: each island's genomes are sorted by size (tokens) into classes a factor of
four apart (1-3, 4-15, 16-63, 64-255, 256+); each occupied class first keeps its own lexicase
choice among the elite (as far as the elite goes), and half of all children have a parent chosen
by lexicase within a class drawn at random from the occupied ones (the other parent of a
recombined child comes from the whole population, so circuits still move between sizes). A small
network that does one simple thing well is then not crowded out by bigger ones that score more on
other cases, and a large one in the middle of a bloat (loops, memory) is not cut down by small
ones that pass the rung sooner. It is selection only: nothing in the network changes, and no
deletion is aimed at parts that look unused (they may be needed over more than one step). Classes
fill as mutation grows and shrinks genomes; the op lines show how many genomes are in each. Run 26
is run 25 with it (`runs/run26.cmd`; `runs/run26-base21.cmd` if runs 24 and 25 fall back), after
run 25.

**Result of run 26** (600 generations; `runs/islands26/README.md`): 21 rungs passed, the most of
any run (run 25: 14, run 22: 15). The islands slowest in run 25 climbed fastest: shapes passed
five rungs between 55 and 61 (run 25: clean at 482), recall respond to lag2 by 55 and lag3 at 337
(run 25: respond at 193). Screen passed glow (300) at 274, a rung no run had passed, and the maze
feel-100 at 142 (run 25: 214). Every island kept genomes in three or four size classes all run;
none reached 256 tokens. Two islands did worse: switching sat on tell from 34 to 486 and
prediction on cycle1-100 until 480, rungs run 25 passed at 52 and 141. Reading cleared the easier
echo rungs again and then scored about 0 on echo2. So keeping a mixture of sizes helps most
islands climb; it does not by itself get past the rungs that need a new ability (echo2, feel).

## Thrift relaxed while stuck (run 27)

Run 22 (thrift on) passed its first rungs in 20 to 40 generations with small networks; run 23
(thrift off) grew networks of 30 to 460 populations and passed switching's first rung only at
123. A hard rung may need more than a small network has, an easy one does not. With
`ISL_THRIFT_RELAX=64` (and thrift on) each lexicase selection takes the thrift cases with chance
64 / (64 + g), g the generations the island has been on its current rung, and otherwise leaves
them out: full thrift on a fresh rung, half after 64 generations without a pass, a fifth after
256; a pass resets g, so what grew while stuck is trimmed on the next rung. Each island follows
its own ladder, so the bloat and the trim come where that island is held up, not at a rung number
fixed for all. It is selection only. (The count starts again at 0 when a run is resumed.) The
other way to let a network choose its size, an output it writes, was set aside: a network cannot
change its genome in a lifetime, so the only size it could choose is how it is judged, and pauses
and gates already let it choose how much of itself runs. Run 27 is run 26 with thrift back on and
relaxed while stuck (`runs/run27.cmd`; `runs/run27-base21.cmd` if the chain falls back), after
run 26.

## Steps up to feel (run 25)

The feel rung (click in the 3 steps after a touch stroke, scored as the click rate there less
the click rate elsewhere, per 1024; goal 300) sits before every repeat rung: switching's third,
shapes' sixth and the maze's first. In runs 16 to 24 no island passed it: the best 10-generation
means were 54 to 89 for switching, 75 to 93 for the maze and up to 87 for shapes, a third of the
goal, where the rung before it (tell) was passed in every run. So the maze has never climbed, and
switching and shapes stop there. `ISL_STEPS=<rung>=<goal>+<goal>` puts copies of a rung before it,
the same world and probes with lower goals; `feel=100+200` makes feel-100 > feel-200 > feel on all
three ladders. Only the harness's ladder changes. With passed rungs kept (run 24's
`ISL_KEEP_RUNGS`), a network on feel-200 is still selected on feel-100. Run 25 is run 24 with inhibit
inputs, the overlap gene and these steps; runs 26 and 27 keep them.

Other rungs no run has passed, by the same count (best 10-generation mean against the goal):
reading's echo2 (25 of 150: far off, and the signal measurement found no gradient there, so a
lower goal would not help; it needs an easier task first), screen's glow (170 of 300), selection's
cued (531 of 600) and prediction's cycle1 (172 of 200, passed once).

**Glow and cycle1 steps, and easier reading rungs** (also run 25): `ISL_STEPS` takes
`feel=100+200,glow=150,cycle1=100`, so screen climbs glow-150 > glow (most runs' best is 137 to 170
of 300) and prediction cycle1-100 > cycle1 (typical best 96 of 200).

Reading's echo2 had no gradient, and a hand-made genome shows why. Echo scores saying a token the
teacher said 1 to 4 steps before, less the same at 9 to 12 steps. The sound of a token and the
voice's code for it are drawn apart in every lifetime, so it must be learned. But that is not the
first obstacle: the ear also hears the network's own voice. A wire from the ear to the voice hears
the teacher and itself, the two codes tie, and the voice locks onto one token. Scored with
`examples/mirrorcheck.rs` over 24 worlds (two tokens, all taught):

| genome (`runs/echo-wires/`) | quiet, mirror 8 | heard, mirror 8 | heard, mirror 4 | echo2 (mirror 0) |
|---|---|---|---|---|
| one wire, ear to voice (`wire.txt`) | 503 | -6 | -10 | 0 |
| and the voice's last word taken away (ear at lag 2 ORed in, the voice at lag 3 cleared: `wire-less-own-voice.txt`) | 431 | 431 | 427 | 0 |

Other lags of the two wires score 19 to 309 heard, so there is a gradient between them.
`Lesson::mirror` (eighths) makes each token's sound share that many eighths of the voice code's
lines (`World::mirror_sounds`); `Lesson::quiet` leaves one's own voice unheard. `ISL_MIRROR=1` puts
echo2q (quiet, mirror 8: one wire), echo2m8 (heard, mirror 8: the wire and taking one's own voice
away), echo2m4 (heard, half the lines) before echo2, all with echo2's goal of 150. From echo2m4 to
echo2 the copy no longer says anything, and the sound's code must be learned in the lifetime;
whether the circuit built on the easier rungs helps there is what run 25 shows.

**The feel rung's ceiling** (`examples/feelcheck.rs`: the feel world driven by a fixed policy over 48
worlds, codes drawn afresh in each as the islands' are). In feel the strokes come whatever is done,
so nothing in the lifetime says which of the hand's 5 codes is the click (they are drawn afresh each
lifetime). What a policy can score:

| policy, in the 3 steps after a stroke | feel mean (worst, best) |
|---|---|
| nothing | 0 |
| a random move | 204 (152, 273) |
| one move kept for the lifetime | 170 (0, 1024) |
| the click (knowing which code it is) | 1024 |

So no network can reach the goal of 300 on average: the most a network that does not know the click
can expect is about 204, by moving at random after a stroke. The best networks so far reached 54 to
119, so there is room below that; but the goal itself is out of reach, and a mirrored-touch rung
(feeling one's own move as the hand's code for it) would not help, as it says nothing about which
move is the click either.

So from run 25 feel's goal is 160 (`ISL_FEEL_GOAL=160`), passed by moving at random after a
stroke (mean 204, worst world 152), with feel-100 before it (`ISL_STEPS=feel=100,...`; feel-200 is
dropped). Learning which move is the click is left to the repeat rung after it, where a stroke
follows one's own move.

**Result of run 25** (600 generations; `runs/islands25/README.md`): 14 rungs passed. The new rungs
were climbed: the maze's feel-100 at 214 and screen's glow-150 at 11, both islands' first rungs in
any run; prediction's cycle1-100 at 141 and cycle1 at 335; reading's echo2q, echo2m8 and echo2m4 at
91, 104 and 107, so evolution found an echo that works while the network hears its own voice. But
nothing carried over to the rungs after them: reading scored about 0 on echo2 for its last 490
generations, the maze about 30 of 160 on feel, switching about 0 on feel-100 after reaching 45.
The islands whose ladders did not change did worse than in run 24 (recall passed respond at 193
against 17, shapes clean at 482 against 64, selection averaged 194 against 277). Inhibit inputs
were kept unevenly (82% of reading's populations at the end, 49% of the generalist's, 8 to 31%
elsewhere); the overlap gene stayed at its default nearly everywhere. Three changes in one run, so
which one cost recall, shapes and selection is open; a run with the new rungs and without inhibit
inputs would tell.

**Result of run 27** (stopped at 536; `runs/islands27/README.md`): 17 rungs passed against run
26's 20 by 536. Relaxing thrift while stuck helped the islands whose rungs stall for long:
selection's cued averaged 385 against 162, screen's glow 148 against 71, prediction 53 against 9
(cycle2 at 121, the best of any run), and switching and prediction passed early rungs run 26 sat
on for 400 generations. It hurt the islands that climb by small fast steps (shapes passed clean at
152 against 55; screen never passed glow) and it let networks grow without a trim where no rung
passes: the generalist, with no ladder, is never reset and reached 897 populations at 200;
generations ran two to three times slower. Size classes with thrift on (run 26) stay the base.

## A lesion map (after run 26), and a wiring cost (run 28)

How brain-like is a network's inside: does it have parts, each for one task? `ISL_LESION=n` (with
`ISL_RESUME`) scores each island's best genome on n fresh worlds on each of its subtasks, then
again with each population taken out in turn (on the same worlds, so the comparison is paired),
and reports each subtask's loss as a share of its rung's goal. On run 26's final genomes, 24 worlds
(`runs/islands26/lesion.txt`):

- **The generalist has no inside.** 26 of its 29 populations read the whole senses (S) and write
  the whole act (A) directly; the other three are wires through a private channel that matter to
  nothing. There is no internal channel for a part to live in.
- 21 of the 29 matter (a loss of 5% of a goal or more over the nine subtasks). Their loss falls 45%
  on one subtask on average, against 11% if spread evenly: some concentration, all of it on
  switching's respond and tell (14 populations) and prediction (7). None is a part for selection,
  reading or recall, whose scores no single population carries.
- Many losses are gains: taking one of the S-to-A populations out raises switching's respond by up
  to 121 (#1398, eye to hand) while costing prediction and screen. The populations are not parts
  but competitors for the same act: interference.
- On the islands, few populations carry a task: selection's best has 103 populations and 2 matter;
  prediction's 19 and 3 matter.

So specialization first needs somewhere to happen. **The wiring cost** (`ISL_WIRE_COST=k`) charges
each population, each step, k for each bit of Hamming distance between the tags of its two ends,
as the carrying cost is charged. A channel's end is its tag; a port's end is the port's own tag, so
ports sit at fixed places in the one 8-bit tag space (the senses at 0x00, the act at 0xFF, the eye
0x07, ear 0x38, touch 0xC1, affect 0xE0, the self channel 0x80, voice 0x1F, hand 0xF8). A loop on one
channel is free, a link between neighbouring tags cheap, and a direct whole-senses-to-whole-act
population pays the most (8 bits). Circuits that work together are pulled into one part of tag
space, near the ports they use, and joined to the rest by few long links: the connection cost that
makes evolved networks modular (Clune, Mouret and Lipson 2013). At k = 1 a population costs up to 8
a step against a carrying cost of 4 a token; the generalist's 29 populations would pay about 232,
against its 2,850 of work. Run 28 is run 26 (size classes, thrift on, not relaxed) with
`ISL_WIRE_COST=1` (`runs/run28.cmd`), after run 27; the op lines report each best's wiring in bits.

**Competition between populations, and pruning** (also run 28). Kernels of one population
compete for their input (overlap inhibition); populations did not compete for their destination:
every one that fired ORed its write in, in genome order, so the act was the OR of every reflex,
which is what the lesion map showed (removing one raised another task's score), and against the
rule to decode from the winner only. At birth a muscle fibre is reached by several motor neurons
that compete by their activity until one keeps it. Two switches build that:
- `ISL_COMPETE=1`: at each destination the kernel populations writing it run best match first
  (each by its best fired kernel's share of its own lines, the step before; wires and noise keep
  their place), and each word an earlier population's kernels wrote this step is claimed: a kernel
  more than half of whose output words are claimed is held down and learns anti-Hebbian, as a
  kernel whose input lies on taken lines is (`KernelContext::taken_out`). Words, not lines: a move
  or a token is a 16-line code inside one word, and on run 26's networks competition by lines
  changed nothing (the reflexes write different lines of the hand's word: 53 lesions that raise a
  subtask with it and without), so one population writes each word.
- `ISL_PRUNE=1` (with it): each population's hold on its destination starts each lifetime at its
  gene `grip` (16, halved or doubled by mutation, 1 to 128), goes up one each step it writes there
  (to twice its start) and down one each step its kernels only lose; at 0 it lets go and runs no
  more that lifetime. Every lifetime starts wired broadly and keeps the winners; the genome is not
  changed, so a population that loses in one lifetime may win in the next.
Run 28 is run 26 with the wiring cost, competition and pruning together (`runs/run28.cmd`).
On run 26's final genomes, scored with word competition and pruning on (24 worlds, the same as the
lesion map; `runs/islands26/lesion-compete-words.txt`), the generalist's prediction rose from 9 to
45, screen from 53 to 71 and selection from 315 to 331 with nothing evolved for it, and lesions that
raise a subtask fell from 53 to 40 while lesions that cost one rose from 16 to 25: less
interference, more populations carrying something. On the islands, which evolved without it,
prediction fell from 170 to 132 and selection from 469 to 420 (21 of selection's 103 populations now
matter, against 2).

**Ordered by grip, not by match** (run 28). At a muscle fibre the inputs are not judged by how
well they match anything: the one whose firing drives the fibre is protected and its active
neighbours take ground from the silent (Balice-Gordon and Lichtman 1994: blocking part of a
junction removes it only while the rest is active), inputs firing in step both stay, and the one
holding more of the junction drives it more and gains more. So with pruning the populations into
a destination take its words in the order of their grip (ties in the genome's order), not their
best kernel's share: winning raises the grip, firing into a claimed word lowers it, and an input
that keeps driving the word (bursts) gains fastest. Without pruning the order stays best match.

**Result of run 28** (stopped at 255; `runs/islands28/README.md`): 11 rungs against run 26's 15
by 255. One writer a word helped where a single circuit should own the output: screen passed glow
at 150 (run 26: 274) and selection averaged 203 against 149. It slowed the islands whose first rungs
need several populations writing together or a wire into a port: reading's echo2q (one wire from
the ear to the voice) took 224 generations against at most 92 before, shapes passed clean at 193
against 55, and the maze and switching started slowly. The contest grew with the networks, from
tens to thousands of kernels held down per 1000 steps (the generalist 17,309 at 200), and pruning
let go of up to 41 populations per 1000 steps on prediction. The wiring cost pulled the best
networks' wiring from up to 332 bits at 50 to 12 to 174 at 250.

## Branches (run 29), and the contest counted

A population had one destination; a brain's axons branch, so one cell's firing reaches several
targets at once: an efference copy of every command, a collateral to an inhibitory interneuron,
two areas driven by one decision. Without it no population could know what it had just done (the
hand-made echo circuit rebuilds that from the OR of the whole act), and inhibition between
populations needed a channel the inhibitor wrote, while the inhibitor's one output went to the
act. With `ISL_BRANCH=1` a population may carry a branch (`Pop::branch`, a tag naming a channel or
an output port): its fired kernels also write their outputs there, at the same place (wrapping),
by OR. Output learning stays on the main destination. A new population has one one time in four;
mutation adds, removes, retags or moves it between channel and port, and circuit copies keep it.
The wiring cost charges the collateral from the source end. Under competition a branch writes
only words no other population claimed this step, and claims them, so it cannot get round one
writer a word. Run 29 is run 28 with branches (`runs/run29.cmd`), after run 28.

From run 28 the op lines also count the contest, per 1000 steps of each island's probing: kernels
held down by another population's claim, and populations that let go of their destination; and,
with branches, how many of the best's populations branch.

**The lesion map of run 28** (24 worlds, its settings; `runs/islands28/lesion.txt`) against run 26's:
the generalist has an inside now. Of its 38 populations 14 join two of its own channels and 15 a
channel and a port, where run 26's 29 were 25 port to port, 4 channel and port, none channel to
channel. Lesions that raise a subtask fell from 53 to 25 (those that cost one: 28), and the
populations that matter put 55% of their loss on one subtask (run 26: 45%; 11% if spread evenly).
On recall 28 of 34 populations matter, 24 of them for its respond and 4 for its memory: a part for
each. The wiring cost, word competition and pruning together are the first change that has built
internal structure.

## Codes a wire can use (a check, after run 29 started)

Every code in the world is 16 random lines in one random word of a 512-bit channel, drawn afresh
each lifetime: each token's glyph and sound, the lights, the stroke, and each of the voice's
tokens and the hand's moves; a move is the code holding at least half its lines. A wire copies
lines where they lie, so what it writes into the hand almost never holds half a move's code.
(Kernels into the act start with the body's codes as their outputs, chosen at random each
lifetime, so they can make moves, but not a move the genome picks.) Two lesson options try other
codes: `Lesson::motor_place` makes the body's moves and tokens place codes, the same every
lifetime, as a muscle is a labelled line (move i owns word i of the hand, and a command is read
as the move whose word has the most lines on: `Effector::fix_place`); `Lesson::sense_place` also
puts the two lights at words 0 and 1 of the eye and every stroke in the first word of the skin.
`examples/motorcheck.rs` scores one wire reading a whole sense port and landing at each word of
the hand or voice in turn (24 worlds; best landing word):

| codes | respond (eye to hand; goal 100) | tell (goal 100) | feel (affect to hand; goal 160) | echo, heard as said (ear to voice) |
|---|---|---|---|---|
| drawn each lifetime, as now | 23 | 42 | 2 | 461 |
| moves and tokens as place codes | 197 | 384 | 97 | 461 |
| and the lights and stroke placed | 515 | 1024 | 389 | 461 |

With the motor side as labelled lines a single wire passes respond and tell (the light lands on the
click's word in the lifetimes where the light's random word happens to be the right one, often
enough), and scores 97 on feel; with the stimuli placed too, those rungs become innate reflexes
(tell 1024, feel 389). What a light, glyph or sound means is still drawn each lifetime either way.

## Learning what to write: copy learning of outputs (a check, negative so far)

No kernel learns what it writes, only when it fires (output learning moved only cancelling
kernels' lines), so a network can say only the token codes its kernels were born with, and cannot
learn which voice code makes the sound it hears: what echo2 needs, and what an infant learns by
babbling. `ISL_OUT_COPY=1` adds a gene `out_copy` (0 off, k a frame offset): each kernel that fires
moves one output line (chance 2^-out_learn a line) toward what its destination held `read_back +
k - 1` frames ago, when what it read came in; such a population's starting and born kernels into a
port get outputs spanning the whole port, so their lines can move to any code there.
`examples/copycheck.rs` scores a hand-made pair on echo2 (two tokens, codes drawn each lifetime,
sound and voice unrelated; 24 worlds): a babbler (self channel to voice, firing one step in four)
and a listener (ear to voice, growing kernels on what it hears, with copy learning):

| | teacher says 8 in 8 | 4 in 8 | 2 in 8 |
|---|---|---|---|
| no copy learning, 200 tokens | 9 | -1 | 5 |
| best copy offset, 200 tokens | 19 (offset 3) | 3 | 9 |
| best copy offset, 800 tokens | 1 | 1 | 0 |

Echo2's goal is 150; nothing reaches it, and longer lifetimes wash it out. Why, as far as the check
shows: what the voice held includes the listener's own writes, so it mostly copies itself; its
kernels are born only when a move is being made, so on the babbler's random token; and the ear
hears the teacher and one's own voice at once, so a heard sound does not say which voice code made
it. The rule stays off by default.
