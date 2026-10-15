# 13: Islands, plasmid exchange and ladders

**Question.** Single-population lexicase on all the probes (experiment 12) grew tell and a little
remember, then stalled. Do populations pressed on different capacities, trading circuits, grow
more? And when a capacity scores nothing at all, can a ladder of easier stages give evolution a
way up?

**Setup.**
- **Islands** (`evolve_islands`). Each island is a tag-genome population from a naked start (4
  empty channels): 16 genomes, elite 4, epsilon-lexicase over (world, probe) cases on its own
  probes, plus one thrift case per world. 12 training worlds; gaps of 0 to 6 steps between trials.
  - Every 20 generations each island takes 2 plasmids into its new children. A plasmid is one
    population of a top-4 genome of another island, with the circuit of every channel it touches,
    made private (`take_population_plasmid`). Each one is logged.
  - Each population is checkpointed as text at every exchange. A run resumes from a checkpoint,
    which is how every change below went into a running experiment.
- **Probes**, in parts per 1024; chance and silence score 0:
  - respond, tell, discriminate, reverse, win-stay: from a serial-reversal lifetime;
  - remember: from a delayed lifetime;
  - added here:
    - **copy**: say the token shown 1 to 4 steps before, untaught tokens only;
    - **echo**: say the token the teacher said 1 to 4 steps before;
    - **babble**: how many of the shown tokens were ever said;
    - **surprise**: clicks in the 3 steps after a token that broke the stream's cycle, minus after
      a kept one.
- **Reading world.** A reading lifetime shows random tokens on the eye. A teacher says some of
  them aloud on the ear, and the network's own voice is heard there too.
- **Islands run:**
  - switching: respond, tell, discriminate, reverse, win-stay;
  - recall: respond, remember;
  - reading: copy (then copy and echo);
  - generalist: every probe;
  - from generation 140: prediction (surprise) and shapes.
- **Ladders** (from generation 100). A stage has its own probes (earlier ones kept), its own world
  (a `Lesson`) and a goal. An island moves on once its population's training mean meets the goal
  for 3 generations in a row. A lesson sets:
  - tokens (2 to 8) and how often the teacher speaks (eighths);
  - how many times the paying light swaps;
  - the delay before the response window (1 to 5);
  - a cycle and how often it is broken (surprise);
  - blur and shift for shapes: each trial shows a fresh exemplar, each line moved within its word
    with chance blur in 8 and the whole light rotated by up to `shift` lines.

  The stage lives only in the harness; the network is never told.

| Island | Stages (goal to move on) |
|---|---|
| switching | respond (respond 100), tell (tell 100), win-stay (80), discriminate (100), all with no swaps; one swap (reverse 80); three swaps |
| recall | delay 1, 2, 3 (remember 150 each), then 4 |
| reading | babble with 2 tokens (768); echo with 2, 4 and 8 tokens, the teacher on every token (echo 150); the teacher fading to 6, 4 and 2 in 8 (copy 100); 1 in 8 |
| prediction | cycles of 1, 2, 3, 4 tokens (surprise 200); 6 over 3 tokens (two tokens of context, 150); 8 |
| shapes | clean, blur 1, blur 2 (tell 150); blur 1 + shift 8 (120); blur 2 + shift 64 (100); then the paying shape |

    ISL_LADDERS=1 ISL_PREDICTION=1 ISL_SHAPES=1 ISL_HELD=48 cargo run --release --example islands
    RCK_TOP=4 RCK_FILE=checkpoint.txt cargo run --release --example recheck   # an elite on 64 fresh worlds
    PC_N=128 PC_TOKENS=2 PC_TEACH=8 cargo run --release --example port_cost   # blind controls under a lesson

## Generations 0 to 100: islands without ladders

Every island's best on 12 held-out worlds jumped around: win-stay 85 then 81 (generalist,
generations 40 and 50), copy 49 then 58 (reading, 60 and 70). Re-scored on 64 fresh worlds,
none held:

| generation 60, best genome, 64 fresh worlds | respond | tell | remember | win-stay | copy |
|---|---|---|---|---|---|
| switching (reads no port) | -2 ±5 | 4 ±11 | 3 ±8 | 3 ±12 | 8 ±6 |
| recall (reads the eye) | 2 ±7 | -1 ±11 | 2 ±5 | 9 ±11 | 0 |
| reading (reads no port) | 0 ±4 | -5 ±7 | 7 ±5 | 32 ±10 | -5 ±9 |
| generalist | 9 ±12 | **60 ±22** | -19 ±7 | 35 ±21 | 0 |

At generation 80 the generalist's elite all held tell on 64 fresh worlds: 71 ±26, 44, 44 and
50 ±19. Tell was the only capacity, as in experiment 12. Copy stayed at chance on every island
for 100 generations, including genomes that read the eye.

### Why copy scored nothing

The world's glyph, sound and voice codes for a token are three unrelated random codes. Copying
needs two learned maps chained:
- glyph to sound, from the teacher;
- sound to the motor code that makes it, from babbling: one's own voice is heard.

Copy scores nothing until both exist. A hand-built eye-to-voice population scores copy -2 even
with the teacher. Echo scores the second map alone; babble scores the voice the second map is
learned from.

## Generations 100 to 140: ladders

Everything about the islands with ladders before generation 140 is from the first stages.

**Each island's best on every probe, 48 held-out worlds, plain world (generation 120):**

| | respond | tell | discrim | reverse | remember | win-stay | copy | echo | babble |
|---|---|---|---|---|---|---|---|---|---|
| switching | **91** | **164** | 2 | 5 | 0 | -8 | 6 | 6 | 853 |
| recall | 4 | 46 | -12 | 4 | **51** | -9 | 0 | 0 | 0 |
| reading | 0 | -4 | 0 | -12 | -4 | -8 | 6 | 22 | 765 |
| generalist | 4 | 49 | 3 | 15 | 3 | -27 | 15 | -3 | 752 |

**Stage moves** (population training means):
- **Switching.** On respond from generation 100, it read no input port: its best genome scored
  respond 1 at generation 110. By generation 120 its genomes read the eye and the mean was 57.
  The mean held 99 to 112 and the island moved to tell at generation 133. Tell's mean was 166,
  then 226 a generation later (goal 100).
- **Reading.** Babble passed at generation 123 (mean 903, goal 768). On echo with 2 tokens the
  mean was 3 to 18 (goal 150) once the probe was fixed (below).
- **Recall.** Remember mean 19 to 60 (goal 150). Its "delay 1" stage was in fact the plain
  3-step task until generation 140 (below), so this is remember on the plain task. Its best's
  51 held out on that task agrees.
- **Generalist** (no ladder). It took up babble as soon as it was a case (mean 467 at generation
  110), and its tell fell from a mean of 32-47 to -11. With nine cases, the easiest one took the
  pressure.

## Faults found on the way

Each looked like progress until checked.
- **Lag-count bias.** Copy and echo compared matches at lags 1-4 against lags 9-12, each lag
  counted from its own start, so the near lags had about 32 more chances. A blind speaker scored
  echo 85 with 2 tokens and the teacher on every token (about 10 with 8 tokens). Reading's first
  echo2 mean of 89 was exactly this. Both probes now count only steps with 12 steps behind them.
  The blind speaker scores 3, and the hand-built eye-to-voice population 42 (with the teacher on
  every token, saying the glyph is echoing).
- **Lessons not reaching the conditioning lifetimes.** The reversal and delayed lifetimes behind
  the probes were built in the plain world, so a stage's delay (and blur and shift) never reached
  the world. Swaps did, through the task's period. Fixed at generation 140.
- **Too few held-out worlds.** On 12 held-out worlds one genome's mean moves by about ±25-50,
  which made the win-stay 85 and the copy 49 and 58. Reports now use 48 held-out worlds, and
  claims a recheck on 64 fresh ones.
- **Reading the wrong "best".** Lexicase ranks by training cases, so the top of the ranking
  changes almost every generation. A capacity held by most of the population (the generalist's
  tell) can be absent from the genome ranked first. Recheck an island's elite, not its first
  genome.

## Prediction and shapes (from generation 140)

**Surprise, not "say the next token".** Saying the next token needs a learned glyph-to-voice map,
the same gap as copy. A click needs no code, so prediction is scored by surprise. The world
marks the broken tokens for the observer only. A break in a cycle of k is a token unlike the one
k frames back: an XOR between the eye now and its history k frames back.

Controls on 128 worlds:
- blind clickers score 0 to 3;
- eye-to-hand populations score 35-78 on a cycle of 1 (a rare glyph wakes lines the kernels tuned
  to the common one do not cover) and about 0 on a cycle of 3.

**Shapes** reuse tell and discriminate with the light distorted each trial. Controls on 64
worlds:
- blind clickers stay near 0 at every blur and shift;
- the eye-to-hand populations' tell moves with the distortion: the best one falls from 339
  (plain) to 58-99, and the plain ones rise to 106-159.

Results to follow.

## Following plasmids

Plasmids grew large late in the run: 23 to 35 populations from switching, 71 into the
generalist at generation 260. The run could not tell whether any of them helped. From the next
restart on, each plasmid is numbered and followed:

- **Its carriers.** A moved population's id carries the plasmid's number, and every generation
  counts the recipient's genomes that carry any of it, the elite among them and those that carry
  it whole. The log says when the last carrier is gone. Mutation and one-point crossover keep
  ids. A copied segment, or a circuit taken by crossover inside an island, gets new ones, so
  spread inside an island is undercounted. A plasmid that was itself moved before names its
  parent (`#12 (from #5)`).
- **What it is made of:** populations, channels, kernels and wires, the ports it reads and
  writes, and whether it names the host's channels. Also anything that could hold state:
  holding channels, self-loops, history, repeats and growth.
- **What it does.**
  - Its role, from its wiring: a reflex (senses to act), a sensor (senses into the host), a
    driver (host to act), a babbler (acts on no sense), a relay (host to host), or sealed
    (reaches nothing).
  - Its scores alone on every probe, set against its blind control (the same circuit, its sense
    reads moved to the self channel). A probe it scores as well blind is marked as an artefact.
  - Its recipient's stage probes with it and without it: helps, hurts or neutral, at a change of
    40.

A short test from the generation-140 checkpoint (4 worlds, so noisy) already showed the pattern.
Most plasmids were gone within 2-3 generations. Several large reflexes from switching hurt their
recipients (recall's remember -96, screen -85). Some plasmids scored well blind, so what they
gave was an artefact, not sight.

## Speed and scaling

The run moved from the cloud container to a workstation (Ryzen 7 PRO 7840HS, 8 cores, 16
threads) at generation 140. With eight islands, a generation took about 60 s in the container
(generations 131-140, timed from commits) and about 20 s on the workstation. Scoring splits each
generation into (genome, world) jobs, about 8 islands x 16 genomes x 12 worlds = 1,536, and
spreads them over every thread `available_parallelism` reports. From the next restart on, every
generation's line carries its UTC time and its seconds.

If the run needs more speed, in order of cost:
1. **One bigger machine.** No code changes: the jobs already fill every core, and 1,536 of them
   leave room well past 16 threads. Clone, build and resume from a checkpoint as on the
   workstation.
2. **Islands across machines.** Islands meet only at the plasmid exchange every 20 generations,
   so each machine could run some islands and swap populations through checkpoint files at each
   exchange. This needs a coordinator, a merge of checkpoints and restarts kept in step. It waits
   until the results show it is worth it.

Run 3 ended at generation 399 (`runs/islands3`). A fresh run on a runtime with channels held
in place, against this one, is experiment 14.

## What it means

- **A capacity that needs two unlearned maps scores nothing**, and evolution has no gradient to
  it (copy, 100 generations). A ladder has to break it into stones one circuit apart, each scored
  alone (babble, then echo, then the teacher fading).
- **Ladders work when the step is one circuit.** Switching on respond went from reading no port
  to passing in about 30 generations, then passed tell in its first generation. Tell was already
  in the gene pool; the respond stage made the eye reader worth keeping.
- **Every probe needs a blind control under every stage's world.** Blind controls caught the clock
  artefact (experiment 12), the lag-count bias and the false copy and win-stay scores. A stage's
  world can create a bias the plain world hides: the lag bias was 10 at 8 tokens and 85 at 2.
- **Islands trade circuits, but plasmids are large.** One plasmid carried 18 populations because a
  busy channel drags its whole circuit along. Nothing yet shows a capacity crossing islands
  through a plasmid the way tell crossed into the generalist in the earlier island run.
- **An island with many cases favours the easiest.** The generalist took up babble and lost tell.
  A specialist with a ladder keeps the pressure on one step.
