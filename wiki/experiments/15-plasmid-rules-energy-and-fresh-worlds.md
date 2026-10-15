# 15: Plasmid rules, the op choice, energy, and fresh worlds (island runs 5 to 7)

Each run builds on experiment 14's runtime (channels held in place, reads counted as work).
Every switch below is off by default.

## Run 5: plasmids as bacteria keep them, and the op choice

**Setup.** Run 4's settings plus:
- `ISL_PLASMID_MAX=6`: a plasmid of more populations is cut to the chosen one and its nearest
  neighbours in the circuit;
- `ISL_EXCLUDE=1`: nothing is sent to a recipient that already carries every population of a
  plasmid;
- `ISL_CARRY=16`: each token a genome carries costs 16 work a step;
- `ISL_OPS=1`: a population's op is drawn again by a mutation of its own (one in 16, against
  about 1 in 100 as one of 38 properties), and ops are drawn without `rotr`, which is `rotl` by
  64 - n.

**Result.** Stopped at generation 161, when the session that launched it ended
(`runs/islands5`). Recall reached lag3 at 126 (run 4: 241). Shapes passed its first rung on
held-out worlds at 30, which run 4 never did. Reading took 115 generations to pass babble (run
4: 26).

Ops spread out. Run 4's islands sat at 80-98% one op each. In run 5, `xor` reached 22% of
prediction's and shapes' populations, `nand` 50-100% of some islands' best genomes, and `set`
appeared.

Generations slowed to 60-80 s at times. A few genomes cost 10-40 times their island's best
(shapes' population averaged 12,674 work a step at generation 60, its best 160). The generalist
was the costliest island in runs 4 and 5: thrift is one lexicase case per world next to one per
probe, so with 12 probes it is 1 case in 13.

## Run 6: larger populations and energy

**Setup.** Run 5 plus `ISL_POP=32 ISL_ELITE=8` and energy (`ISL_RATION=256 ISL_FOOD=32`):
- each lifetime step a network gets 256 work and pays its work;
- it eats what it scores: every 50 steps the last block is scored by its task's probes, and each
  1024 points are 32 rations;
- with its store below 0 it rests (does not run or act) until the ration refills it.

Compute beyond the ration must be earned by scoring. At these settings, rescoring the best
genomes of runs 4 and 5 kept every capacity while cutting the generalists' work from 6,708 and
1,886 a step to under 300 (`RCK_RATION`). Food first came from the world's strokes, but reading
and prediction have none, so food was made the score itself.

**Result** (`runs/islands6`). 300 generations in 26 minutes, 4-8 s a generation at twice run
5's population. Stage moves, each passed on held-out worlds:

| island | run 4 | run 6 |
|---|---|---|
| switching | never passed respond | respond, tell (21), win-stay (24), discrim (148), swap1 (153) |
| recall | lag1 225, lag2 238, lag3 293 | lag1 95, lag2 109, lag3 112, lag4 173 |
| shapes | never passed clean (held out) | blur1 147 to win-stay 159, five rungs |
| reading | echo2 at 26 | echo2 at 268 |

Plasmids: of 223 scored on arrival, 27 helped (12%, against 6% in runs 3 and 4). The median
size was 1 population, and two reached half an island. None made a lasting difference.

## What stalled, and why

- **Maze fitted its training worlds.** It held its training goal for 50 generations (up to 396
  against 200) and failed every held-out check (-44 to 88). Rescored on fresh worlds:

  | | training worlds 0-11 | fresh worlds |
  |---|---|---|
  | maze's top 4 | 658, 481, 481, 476 | -127, -177, -177, -69 |
  | their blind controls | 27, 7, 7, 33 | 5, 0, 0, 23 |

  It read the cue, since the blind controls fail, but turned the wrong way on fresh worlds. The
  hand's codes are drawn from a world's seed, and the 12 training worlds were the same every
  generation, so the elite had wired those worlds' codes instead of learning them each lifetime.
- **Screen's torch rung was a tell, not a respond.** The torch shows the glyph only under the
  cursor, but the cursor's mark is on the eye every step, so clicking at what is seen clicks at
  the cursor. A policy that clicks at anything seen scores about 0 with the cursor shown and
  above 600 with it hidden (tested). Runs 4-6 scored 0-35 on the torch.
- **Reading was slow to babble under the carrying cost.** Babble with 2 tokens is close to all
  or nothing per world (both tokens said: 1024; one: 512). The elite said one token in most
  worlds and was 2-4 populations. Energy did not cause it (rescored with and without energy,
  the same babble). The carrying cost is the common factor in runs 5 and 6, and energy now
  charges compute directly.

## Run 7: fresh worlds, a dark torch, a lighter carrying cost

Run 6 plus:
- `ISL_FRESH=1`: the training worlds are new every generation, the same for every genome in it;
- screen's ladder starts with a dark torch (`dark=1`: no cursor mark; the eye is empty unless the
  cursor is on the target);
- `ISL_CARRY=4` (was 16);
- 600 generations.

    runs/run7.cmd (started with Win32_Process.Create, so it outlives the session)

**Result** (`runs/islands7`, 600 generations in 50 minutes).

| island | run 6 | run 7 |
|---|---|---|
| shapes | clean to win-stay at 147-159 | clean to win-stay at 61-83 |
| reading | echo2 at 268 | echo2 at 129 |
| switching | to swap1 by 153 | tell 91, win-stay 94, then stuck (mean 3 over 500 generations) |
| recall | lag4 by 173 | lag1 at 399 |
| maze | fitted its worlds | honest: mean 37, best 275 on turn |
| screen | about 0 on the torch | mean 45, best 126 on the dark torch |

Final bests rescored on 48 fresh worlds (plain world; blind controls about 0 throughout): shapes
told the lights apart at 318 (run 6: 231), recall at 277 (235), prediction at 277 (165).

The dark torch and the lighter carrying cost did what they were for. Fresh worlds removed the
fitting, but they also made every generation's population mean swing by 50-100. A goal met 3
generations in a row on 12 such worlds became rare, so switching, shapes and recall, which each
got near a goal many times, mostly stopped climbing. The held-out check on 48 worlds is the
reliable test of a stage, and holding 3 in a row on few noisy worlds now only delays it.

## Run 8: more training worlds, a goal held once

Run 7 plus `ISL_LIVES=24` (24 training worlds, halving the noise of a mean) and `ISL_HOLD=1`
(the goal met once on the training worlds, then the held-out check on 48 decides). 600
generations, `runs/run8.cmd`.

**Result** (`runs/islands8`, 99 minutes). Recall climbed its whole ladder on fresh worlds: lag1 at
210, lag2 at 380, lag3 at 389, lag4 at 420 (run 7: lag1 at 399). Shapes again reached win-stay
(at 89), and switching reached it at 172. Over generations 100-599, win-stay averaged 3 on
switching and 1 on shapes, maze 0 (run 7: 37), screen 48 and prediction 59.

So with goals checked once against 48 held-out worlds, everything up to win-stay is climbed on
worlds the population has never fitted. Two steps stop every run since fresh worlds came in:
- **win-stay**, the first rung that needs learning within a lifetime (click a light again
  because it was just stroked);
- **maze's turn**, which needs the hand's two turn codes learned within each lifetime from the
  stroke.

Both are lifetime learning from reward, and both are where the next rungs are needed.
