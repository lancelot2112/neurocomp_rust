# Island run 15 (stopped at generation 250)

Run 14's settings plus a feel rung (`ISL_FEEL=1`) before every repeat rung: strokes come at
random whatever the program does, and the `feel` probe scores clicks in the 3 steps after a
stroke minus clicks at other steps. It is first on the maze's ladder, after shift64 on shapes',
after tell on switching's.

Stopped after generation 250 (1 h 10 min), to move to run 16 (port gates), which keeps the feel
rung.

Stage moves (each passed on held-out worlds):
- shapes: clean to feel between 23 and 33;
- switching: tell at 46, feel at 59;
- recall: lag1 at 78, lag2 at 86, lag3 at 87, **lag4 (its whole ladder) at 105**, the fastest
  of any run (run 14: 139, run 9: 186);
- reading: echo2 at 12, never passed.

The feel rung (goal 300):
- the maze: mean 47 over 251 generations, 63 over the last 50, best 97;
- switching: mean 11 over 192, rising to 36 over the last 50 (54 at 250), best 72;
- shapes: 0 over 218 (best 3).

None passed. The stroke reaches the hand on two islands, weakly.

Prediction's cycle1 averaged 79 (best 172); selection's cued rung 229 (best 380).

Plasmids: of 216 scored, 12 helped (6%), 44 hurt, 5 were mixed and 155 neutral. #59 (selection to
reading) gave echo +512 on arrival at 80, the largest echo gain of any plasmid, and died out.
#145 (maze to switching, 180) was in 23 of switching's 32 networks at 200, as switching's feel
score rose.

`checkpoint.txt` is generation 250; `plasmids.txt` records the plasmids up to 240.
