# Island run 10 (stopped at generation 341)

Run 9's settings plus the generalist scored at the frontier (`ISL_FRONTIER=1`) and a repeat rung
for learning from reward: a maze with the food fixed and no cue, the `repeat` probe. The rung
comes before switching's and shapes' win-stay, and is the maze's first rung (`learn`).

Stopped after generation 341 (1 hour), behind run 9 and with nothing new, to move to run 11.

Stage moves (each passed on held-out worlds): reading to echo2 at 50; switching tell at 128,
repeat at 129; shapes blur1 at 83 to repeat at 156; recall lag1 at 268.

At generation 341, against run 9 at the same point: recall on lag1 (run 9: lag4 by 186);
prediction's mean over the last 100 generations 23 (102); screen's 51 (107).

The repeat rung averaged -12 on switching (214 generations, best 318 once), -4 on shapes (187,
best 142) and -2 on the maze (243, best 98). `examples/maze_check` showed why. The genomes
turned on 57-100% of trials, but repeated a turn as often after a miss as after a stroke
(switching 80-96% either way), so the stroke changed nothing. Acting on it needs two populations
at once (affect copied into a channel, and a repeating population gated on it, since a gate
names a channel, not a port). Run 13 adds a feel rung below for the first.

Plasmids: of 272 scored, 36 helped (13%), 43 hurt, 9 were mixed and 184 neutral.

`checkpoint.txt` is generation 340.
