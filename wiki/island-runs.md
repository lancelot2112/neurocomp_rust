# Island runs compared

Every island run so far (`runs/islands3` to `runs/islands10`), side by side: what each changed,
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
| 10 | the generalist at the frontier; a repeat rung for learning from reward | running | |
| 11 | epigenetic marks; exploration kept in the genome | next | |

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
- **Plasmids** help on arrival 6-12% of the time in every run, but none has made a lasting
  difference to an island.

## Still stuck in every run

- **Win-stay** (switching, shapes): reached in every run from 6 on, passed in none since fresh
  worlds. The fixed probe shows no genome learns from reward. Run 10 adds the repeat rung below
  it.
- **The maze's turn**: never passed honestly. It needs the turn codes learned from reward, which
  is the repeat rung's job too (the maze's new first rung).
- **Screen's first rung**: no run has passed it, but scores rose from about 0 on the torch to a
  mean of 111 on the dark torch (run 9).
- **Reading's echo2**: reached in every run, never passed. Its probe is sound (a hand-written
  echoer scores 483), so the gap is evolving the sound-to-voice circuit.
