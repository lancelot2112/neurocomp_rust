# Island run 7

Run 6's settings plus:
- `ISL_FRESH=1`: new training worlds every generation, the same for every genome in it;
- screen's ladder starting with a dark torch (no cursor mark);
- `ISL_CARRY=4` (was 16);
- 600 generations.

600 generations in 50 minutes. Started with `runs/run7.cmd` through `Win32_Process.Create`.

Stage moves (each passed on 48 held-out worlds): shapes to blur1 at 61, blur2 at 73, shift8 at
76, shift64 at 80 and win-stay at 83 (run 6: 147-159); switching to tell at 91 and win-stay at
94; reading to echo2 at 129 (run 6: 268); recall to lag1 at 399 (run 6: 95).

Each island's mean goal score over generations 100-599, on its stage:

| island | stage | mean | best |
|---|---|---|---|
| switching | win-stay | 3 | 157 |
| shapes | win-stay | 2 | 110 |
| recall | respond / lag1 | 54 / 100 | 229 / 240 |
| reading | echo2 | -10 | 199 |
| prediction | cycle1 | 74 | 185 |
| maze | turn | 37 | 275 |
| screen | glow (dark torch) | 45 | 126 |

Better than run 6: shapes and reading climbed sooner. Maze's scores are now honest (no fixed
worlds to fit), and screen scores on the dark rung, where it scored about 0 on the torch.

Worse than run 6: switching and shapes stuck on win-stay, recall only reached lag1, and
reading's echo2 averaged about 0. With fresh worlds the population mean swings by 50-100 from
one generation to the next, so a goal is rarely met 3 generations in a row. Run 8 trains on 24
worlds and holds a goal 1 generation before the held-out check.

Plasmids: of 460 scored on arrival, 59 helped (13%), 76 hurt, 15 were mixed and 310 neutral.
The median size was 1 population; 24 reached half an island. 3 were still carried at the end.

`checkpoint.txt` is generation 599. Recheck with
`RCK_INPLACE=1 RCK_READS=1 RCK_RATION=256 RCK_FOOD=32 RCK_FILE=runs/islands7/checkpoint.txt`.
