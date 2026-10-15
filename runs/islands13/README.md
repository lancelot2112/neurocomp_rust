# Island run 13 (stopped at generation 250)

Run 12's settings plus freak accidents (`ISL_ACCIDENT=1`): each population has its own rate, a
chance of 1 in 2^k each step (k drawn 12 to 18, moved by mutation), of growing its trial
connection even when it is not stuck. The trial is the same one exploration grows, still once a
lifetime, kept the same way. (The header's "2k" lost its caret to the shell: it is 2^k.)

Stopped after generation 250 (1 h 5 min), with nothing new on the rungs it was meant to help, to
move to run 14.

Stage moves (each passed on held-out worlds):
- reading: echo2 at 14, never passed.
- shapes: clean to repeat between 57 and 67.
- switching: tell at 135, repeat at 144.
- recall: lag1 at 138.

Means over the run: prediction's cycle1 90 (best 188), the highest of any run (run 11: 78, run
12: 14), without passing. On repeat, switching -10 (108 generations) and shapes -2 (185). Reading's
echo2 -7.

Accidents: evolution kept the rates rare. The elites' mean k settled at 14 to 17, about one
accident in 16,000 to 130,000 steps, and a few populations mutated to none.

Plasmids: of 192 scored, 22 helped (11%), 29 hurt, 11 were mixed and 130 neutral. #142 (recall to
shapes) gave repeat +543 on arrival at 180 and was gone by 200. #175 (prediction to shapes, from
220) was in all 32 of shapes' networks, whole, by 240: the first plasmid to fix in an island. Shapes'
repeat score did not change.

`checkpoint.txt` is generation 250; `plasmids.txt` records the plasmids up to 240.
