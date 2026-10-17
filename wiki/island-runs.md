# Island runs compared

Every island run so far (`runs/islands3` to `runs/islands18`), side by side: what each changed,
and how far each island climbed. Details are in experiments 13 to 16.

## What each run changed

| run | changed from the run before | generations | time |
|---|---|---|---|
| 3 | (old runtime; started from genomes evolved 100 generations without ladders; ladders changed mid-run) | 400 | about 3 h, mixed machines |
| 4 | fresh from naked; channels held in place, no history; reads counted as work; the `set` op; faster births | 300 | 51 min |
| 5 | plasmids of at most 6 populations, exclusion, a carrying cost of 16; the op choice | 161 (stopped) | - |
| 6 | population 32, elite 8; energy (a ration a step, food for score) | 300 | 26 min |
| 7 | fresh training worlds each generation; screen's dark torch first; carrying cost 4 | 600 | 50 min |
| 8 | 24 training worlds; a goal held once before the held-out check | 600 | 99 min |
| 9 | the win-stay probe fixed (nothing to compare scores 0) | 600 | about 1 h 45 min |
| 10 | the generalist at the frontier; a repeat rung for learning from reward | 341 (stopped: behind run 9, nothing new) | 1 h |
| 11 | epigenetic marks; exploration kept in the genome | 600 | 2 h 35 min |
| 12 | kernels that may expect zeros as well as ones | 250 (stopped: nothing new on repeat) | 57 min |
| 13 | freak accidents: a trial connection grown now and then even when not stuck | 250 (stopped: nothing new on repeat) | 1 h 5 min |
| 14 | a ninth island, task selection | 250 (stopped: reward rungs flat, selection falling) | 1 h 19 min |
| 15 | a feel rung before every repeat rung | 250 (stopped: feel rising slowly, moved to port gates) | 1 h 10 min |
| 16 | ports for gates, pauses and writes: one population can learn only after a stroke | 600 | 3 h 22 min |
| 17 | shadow explorers, and recall's long ladder | 250 (stopped: predictor bloat, feel at 0) | 1 h 11 min |
| 18 | pauses drawn as gates are: a port pause one step away | 548 (stopped for run 19) | 3 h 32 min |
| 19 | shadows as indicators: a find noted as a hint that gates and pauses take, nothing built; shadows started at random | 600 | 3 h 2 min |
| 20 | edges built on hints: a mutation of its own adds a population of any kind on a hinted path, the way that feeds back (closes a loop, or runs from a higher level to a lower) three times in four; output ports readable by any population (ISL_READ_OUT); hint bands (ISL_HINT_BAND); self channel writable (ISL_SELF_WRITE); loops closed (ISL_LOOPS) | 600 | 3 h 8 min |
| 21 | heat inputs: a channel or port whose fullness moves a population's thresholds, down when warm, up when cool | 600 | 4 h 24 min |
| 22 | no plasmid exchange (ISL_EVERY=0): run 21 with the islands apart, to measure the exchange | 600 | 3 h 43 min |
| 23 | thrift off (ISL_THRIFT=0) and carrying paid as energy (ISL_CARRY_ENERGY=1): run 22 (no plasmid exchange) with energy alone pricing compute and carrying | after 22 | stopped at 296: 4 rungs passed against run 22's 8 (switching feel 0 against 45, recall on lag1 against lag4, prediction 0 against 85); networks grew without limit (reading 684 populations) and passed nothing; runs 24 to 27 are built on run 22 instead, each on the one before (the run-23-based launchers kept as runN-base23.cmd) |
| 24 | goals judged on the elite (ISL_ELITE_PASS=1), partial credit in selection (ISL_PARTIAL=1) and passed rungs kept as cases (ISL_KEEP_RUNGS=1): run 22 with the three (run 23 showed no promise); no plasmid exchange | started 02:59Z when run 23 was stopped | stopped at 409: 11 rungs passed against run 22's 14 (recall lag1-lag3 at 17-19, shapes four rungs at 229-272, sooner than any run; within a rung far lower: switching feel mean 5 against 31, recall lag3 25 against 84) |
| 25 | inhibit inputs (ISL_INHIBIT=1), the overlap rule as a gene (ISL_OVERLAP=1) steps up to the rungs no run has passed (ISL_STEPS=feel=100,glow=150,cycle1=100), feel's goal lowered to 160 (ISL_FEEL_GOAL: 300 is out of reach) and easier reading rungs (ISL_MIRROR=1: echo2q > echo2m8 > echo2m4 > echo2): run 24 with them; no plasmid exchange | started 05:25Z when run 24 was stopped | |
| 26 | size classes (ISL_SIZE_CLASSES=1): lexicase keeps each size class's own choice among the elite and draws half the parents within a random class, so small and large networks both stay; run 25 with it; no plasmid exchange | after 25 | |
| 27 | thrift relaxed while stuck (ISL_THRIFT=1, ISL_THRIFT_RELAX=64): the thrift cases taken with chance 64/(64+g), g the generations on the current rung, full again after a pass; run 26 with it; no plasmid exchange | after 26 | |

## How far each island climbed

The highest stage reached, and the generation it was reached; "first" means still on the first
rung. Every move from run 6 on was also passed on 48 held-out worlds.

| run | switching | recall | reading | prediction | maze | screen | shapes | plasmids that helped |
|---|---|---|---|---|---|---|---|---|
| 3 | swap1 (172) | first | echo2 (124) | first | first | first | paying (309) | 6% of 64 |
| 4 | first | lag4 (294) | echo2 (26) | first | first | first | first | 6% of 223 |
| 5 | first | lag3 (126) | echo2 (116) | first | first | first | blur1 (31) | 11% of 128 |
| 6 | swap1 (153) | lag4 (173) | echo2 (268) | first | first | first | win-stay (159) | 12% of 223 |
| 7 | win-stay (94) | lag1 (399) | echo2 (129) | first | first | first | win-stay (83) | 12% of 460 |
| 8 | win-stay (172) | lag4 (420) | echo2 (13) | first | first | first | win-stay (89) | 8% of 462 |
| 9 | win-stay (229) | lag4 (186) | echo2 (20) | cycle2 (415) | first | first | win-stay (101) | 10% of 458 |
| 10 (to 341) | repeat (129) | lag1 (268) | echo2 (50) | first | first | first | repeat (156) | 13% of 272 |
| 11 | repeat (67) | lag3 (102) | **fade6 (369)** | first | first | first | repeat (250) | 13% of 462 |
| 12 (to 250) | repeat (110) | lag3 (191) | echo2 (13) | first | first | first | repeat (57) | 13% of 191 |
| 13 (to 250) | repeat (144) | lag1 (138) | echo2 (14) | first | first | first | repeat (67) | 11% of 192 |
| 14 (to 250) | repeat (59) | **lag4 (139)** | echo2 (14) | first | first | first | repeat (36) | 9% of 215 |
| 15 (to 250) | feel (59) | **lag4 (105)** | echo2 (12) | first | first (feel: mean 47) | first | feel (33) | 6% of 216 |
| 16 | feel (26; mean 67) | lag4 (353) | echo2 (23) | **cycle2 (332)** | first (feel: mean 36) | first (glow: mean 62) | feel (157; mean 46) | 4% of 523 |
| 17 (to 250) | feel (135; mean 0) | lag3 (209) | echo2 (12) | first | first (feel: mean 3) | first (glow: mean 43) | feel (96; mean 0) | 8% of 215 |
| 18 (to 548) | feel (158; mean 36) | lag3 (93) | echo2 (18) | first | first (feel: mean 26) | first (**glow: mean 105**) | first | 5% of 485 |
| 19 | feel (44; mean 40) | **lag4 (83)** | echo2 (27) | first | first (**feel: mean 55**) | first (glow: mean 58) | feel (140; mean 52) | 8% of 522 |
| 20 | feel (157; mean 51) | lag4 (127; **lag3 at 36**) | echo2 (13) | first | first (feel: mean 48) | first (**glow: mean 117**) | shift8 (593) | **10% of 521** |
| 21 | feel (44; mean 7) | **distract (104): every delay rung, lag16 at 88** | echo2 (21) | first (**cycle1: mean 58**) | first (feel: mean 24) | first (glow: mean 99) | feel (442) | 7% of 521 |
| 22 (no exchange) | feel (33; **mean 35**) | lag16 (582) | echo2 (17) | first (**cycle1: mean 78**) | first (**feel: mean 45**) | first (glow: mean 105) | feel (259) | none moved |

Two cautions for reading it:
- **Runs 3 to 6 trained on the same 12 worlds every generation**, and a population can fit them.
  Run 6's maze did (476-658 on its training worlds, -69 to -177 on fresh ones), and run 3's
  shapes climbed on training scores the held-out check would have stopped. From run 7 on, the
  training worlds are new each generation, so every climb is on worlds the population never saw.
- **A stage's name means the same rung only from run 6 on.** Earlier runs had shorter ladders
  (recall had no respond rung before run 4's, screen no dark torch before run 7's).

## What the changes did

- **Held memory (run 4)** made memory tasks reachable. Recall went from never passing its first
  rung (run 3) to lag4, and has reached lag4 in every full run since except run 7.
- **Energy and larger populations (run 6)** made runs fast (26 minutes for 300 generations) and
  let switching and shapes climb.
- **Fresh worlds (run 7)** ended the fitting but made goals hard to hold. **24 worlds and a goal
  held once (run 8)** restored the climbing.
- **The win-stay fix (run 9)** gave clean readings and the fastest recall (lag4 at 186), and with
  it prediction passed its first rung for the first time.
- **Marks and exploration (run 11)** got reading past echo2 for the first time (echo2 to fade6
  between 337 and 369), and kept a gradient on the repeat rung (switching's mean 48). They cost
  the simpler islands: shapes waited 230 generations on its first rung and recall stalled on lag3.
- **Kernels expecting zeros (run 12)** sped up the simple rungs (shapes reached repeat by 57,
  recall lag1 by 64) but left the repeat rung at about 0.
- **Freak accidents (run 13)** left the repeat rung and echo2 where they were; evolution kept
  them rare. Prediction's cycle1 averaged 90, the highest of any run.
- **Port gates (run 16)** got the stroke to the hand on all three feel islands at once (switching's
  mean 67, shapes' 46), screen's glow off 0 (mean 62), and prediction past cycle1 for the second
  time. Pauses on ports and writing a sense spread later, unaided.
- **Shadow explorers (run 17)** kept predictors that bloated islands in turn (recall 264, the
  generalist 605), each island doing worse while it carried them; the feel rung fell to 0.
- **Pauses drawn as gates are (run 18)** spread port pauses at once and gave screen its best
  glow (mean 105, best 201) from 1-4 populations mostly paused on a port; the kept predictors
  still held the other islands back.
- **Plasmids** help on arrival 6-12% of the time in every run, but none has made a lasting
  difference to an island. The first to fix whole in an island was run 13's #175 (all 32 of shapes'
  networks), and shapes' score did not change.

## Runs 16 to 19 compared (port gates, shadows, pauses, hints)

Each run adds one change to the one before; run 19 was compared at generation 342 (still
running). Means are of each island's goal score per generation, over generations 243-342 (runs
16, 18, 19) or 151-250 (run 17, stopped there).

| | run 16: port gates | run 17: + shadow predictors kept | run 18: + pauses drawn | run 19: + shadows as hints |
|---|---|---|---|---|
| switching on feel | **69** (feel at 26) | 0 (feel at 135) | 44 (feel at 158) | 49 (feel at 44) |
| maze on feel | 39 | 8 | 36 | **69** |
| shapes on feel | 20 (feel at 157) | 0 (feel at 96) | first rung, never left | **53** (feel at 140) |
| screen's glow | 48 | 58 | **121** | 44 |
| selection's cued | 248 | 262 | 172 | **355** |
| prediction's cycle1 | **100** (passed at 332) | 23 | 85 | 2 |
| recall | lag4 at 353 | lag3 at 209 | lag3 at 93, stuck | **lag4 at 83** (the earliest of any run) |
| reading's echo2 | -11 | -11 | -4 | -10 |
| kept predictors | none (no shadows) | hundreds (the generalist 605) | hundreds (the generalist 410) | none |

What each change did:
- **Port gates (run 16)** made learning only after a stroke one population, and the stroke
  reached the hand on all three feel islands for the first time. Switching led (69), and
  prediction passed cycle1.
- **Kept shadow predictors (run 17)** bloated the islands in turn: each island did worse while it
  carried them and recovered as it shed them. The feel rung fell to 0.
- **Pauses drawn as gates are (run 18)** spread port pauses at once and gave screen its best glow
  (mean 121 here, 105 over the run), from circuits of 1 to 4 populations mostly paused on a port.
  The kept predictors still held the other islands back.
- **Shadows as hints (run 19)** removed the bloat: no predictors, networks of 4 to 45
  populations. The maze (69) and shapes (53) have their best feel stretches of any run, and recall
  reached lag4 at 83. Hints sit on 11-49% of the populations, but only 2-7 gates or pauses on an
  island have taken one: a hint is used only when that population's gate or pause mutates.
  Switching's feel (49) and prediction (2) are behind run 16.

Across the four runs:
- the stroke reaches the hand on two or three islands in every run without bloat (16, 19), 45-70
  at best against a goal of 300; no feel rung has passed;
- the circuits that score are small and use ports: gated (run 16), paused (run 18), or both
  (run 19's maze and selection, 47-100% on each);
- a feel circuit moved between islands by plasmid twice: switching to shapes (run 16, #248),
  maze to shapes (run 19, #231, its senses classed as feel).

## Patterns in the evolved networks (runs 9-14)

`python runs/patterns.py` reads each run's log and last checkpoint. Two views:
- **Recursion against score:** each island's best at every 10-generation table (1,561 readings):
  its share of populations on a loop, against its held-out score on its current goal (as a share
  of the goal).
- **Wiring in the elites:** the top 8 of each island in each run's last checkpoint (6,465
  populations).

**Recursion helps memory and responding, not prediction or echo.** With loops against without,
same island and rung:

| island, rung | without loops | with loops |
|---|---|---|
| recall lag3 | 0.29 | **0.90** |
| recall respond | 0.61 | 0.93 |
| switching respond | 0.57 | 0.93 |
| screen glow | 0.19 | 0.36 |
| switching repeat | 0.02 (n 105) | 0.21 (n 9) |
| prediction cycle1 | **0.56** | 0.33 |
| reading echo2 | 0.06 | -0.01 |

Loops scored better on 9 island-rungs and worse on 7. Across all islands, heavy recursion (half
or more of the populations on a loop) did slightly worse (mean 0.20, 4% at the goal) than none
(0.27, 9%). It suits the tasks that need held state (recall's lags); prediction's best networks
were arithmetic without loops (sub and shr, runs 13-14).

**Inhibition between populations evolves everywhere.** A quarter of the elites' populations
(25%) write a clearing op (clear, and, nor, nand) into a channel that another population writes
with an exciting op. That is competition for a channel, on top of the built-in inhibition between
kernels. 28% clear a channel that something reads, or an output port. Run 14's switching had 52%
competing, run 12's shapes 57%.

**Switching circuits on and off is common.** 22% of the elites' populations run only while some
channel had lines on (or none) a frame ago: an execution gate, so one channel switches another
circuit on or off. It is high on islands that read context, such as run 9's generalist (53%) and
run 11's prediction (70%). Run 14's recall, which climbed its whole ladder by 139, had 59%.

**Learning gated by a channel: the shape internal reward would take.** 48% of the populations
learn only while a channel is on (or off). For 993 the gate channel is fed, within three hops,
from the affect port (the stroke); for 2,153 it is internal. That is reward-gated learning in
structure, but the repeat rung stayed near 0 in every run, so the stroke-gated ones are not used
that way yet. The genome shows the gate is there, not that the channel ever changes in a
lifetime.

**Strengthening populations that come true.** 41% of the populations replicate kernels that keep
coming true; 9% have repeat births (copying an act that was followed). With marks (runs 11-14),
populations whose predictions came true are duplicated more and deleted less: 90% of run 11's
generalist elite and 60% of run 12's recall were marked.

**What goes with a better best** (each island's last best against its last held-out score, 43
islands):

| in the best genome | islands with it | mean score | islands without | mean score |
|---|---|---|---|---|
| repeat births | 17 | **0.39** | 26 | 0.16 |
| loops | 17 | 0.34 | 26 | 0.19 |
| learn gates | 18 | 0.32 | 25 | 0.21 |
| run-while gates | 18 | 0.30 | 25 | 0.22 |
| clearing that competes | 20 | 0.28 | 23 | 0.23 |
| replication | 29 | 0.20 | 14 | **0.36** |

These are small samples and confounded (islands differ in difficulty, and the features are
correlated with each other and with genome size). They point the same way as the time series:
loops, gates and repeat births come with better networks; kernel replication does not.

## Still stuck in every run

- **Win-stay** (switching, shapes): reached in every run from 6 on, passed in none since fresh
  worlds. The fixed probe shows no genome learns from reward. The repeat rung below it (run 10)
  averaged about 0 too: the genomes choose, but the stroke changes nothing. Run 15 adds a feel
  rung below that (the stroke reaching the hand at all), and run 12 kernels that can match the
  stroke's absence.
- **The maze's turn**: never passed honestly. It needs the turn codes learned from reward, which
  is the repeat rung's job too (the maze's new first rung).
- **Screen's first rung**: no run has passed it, but scores rose from about 0 on the torch to a
  mean of 111 on the dark torch (run 9).
- **Reading's echo2**: reached in every run, passed only in run 11 (with marks and
  exploration), which went on to fade6.
