# Island run 8

Run 7's settings plus 24 training worlds a generation (`ISL_LIVES=24`, was 12) and a goal held 1
generation on them before the held-out check on 48 worlds (`ISL_HOLD=1`, was 3). 600 generations
in 99 minutes (`runs/run8.cmd`). Started overnight as the follow-up to run 7.

Stage moves (each passed on held-out worlds): reading to echo2 at 13; shapes clean to win-stay at
11-89 (run 7: 61-83); switching to tell at 171 and win-stay at 172 (run 7: 91, 94); recall to lag1
at 210, lag2 at 380, lag3 at 389 and lag4 at 420 (run 7 reached only lag1, at 399). Recall
climbed its whole ladder on fresh worlds.

Mean goal score over generations 100-599:

| island | stage | mean | best |
|---|---|---|---|
| switching | win-stay | 3 | 102 |
| shapes | win-stay | 1 | 71 |
| reading | echo2 | -12 | 70 |
| prediction | cycle1 | 59 | 180 |
| maze | turn | 0 | 93 |
| screen | glow | 48 | 162 |

Win-stay held both switching (428 generations) and shapes (500), as in run 7. It is the first
rung that needs learning within a lifetime: clicking a light again because it was just stroked.
Maze averaged 0 (run 7: 37); screen as run 7.

Final bests on 48 fresh worlds (blind controls about 0): prediction tell 384 and remember 213,
screen respond 171 and remember 267, switching respond 191 and tell 231, recall remember 83.

Plasmids: of 462 scored on arrival, 38 helped (8%), 72 hurt, 24 were mixed and 328 neutral; 11
reached half an island.

`checkpoint.txt` is generation 599. Recheck with
`RCK_INPLACE=1 RCK_READS=1 RCK_RATION=256 RCK_FOOD=32 RCK_FILE=runs/islands8/checkpoint.txt`.
