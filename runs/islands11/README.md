# Island run 11 (600 generations)

Run 10's settings plus epigenetic marks (`ISL_EPI=1`) and exploration (`ISL_EXPLORE=1`): a
population whose kernels all go 64 steps without firing grows a trial XOR connection, and a trial
whose predictions come true 16 times in a generation is added to the genome, tagged `explored`.
Started 19:09Z, finished 21:44Z (2 h 35 min).

Stage moves (each passed on held-out worlds):
- reading: echo2 at 10, then **echo4 at 337, echo8 at 355 and fade6 at 369**. No run before had
  passed echo2.
- switching: tell at 66, repeat at 67 (run 10: 129).
- recall: lag1 at 78, lag2 at 88, lag3 at 102, then stuck on lag3 (mean 23 over 498 generations).
- shapes: stuck on clean until 230, then blur1 to repeat between 230 and 250.
- prediction, the maze and screen stayed on their first rungs.

The repeat rung (learning from reward):
- switching averaged 48 over 533 generations on it, best 231; run 10's averaged -12.
- shapes averaged 0 (350 generations, best 152); the maze's learn 10 (600, best 218).
- None passed (goal 300).

Prediction's cycle1 averaged 78 over the whole run (run 10: 20), best 192, without passing.

Exploration and marks:
- Reading's best had 87 marked populations and 13 explored when it passed echo2 to echo8. By
  580 it had 189 populations reading five ports.
- The generalist bloated: 278 explored connections and 69 of 90 populations on loops at 400,
  shed to 63 explored by 550.
- Recall and screen collapsed onto single ops at times (add 87%, sub 86%).

Plasmids: of 462 scored, 61 helped (13%), 80 hurt, 18 were mixed and 303 neutral. Many helped
the repeat rung on arrival (up to +207). One spread to 28 of the maze's 32 networks (#154,
generation 240), then died out. None was carried at the end.

`checkpoint.txt` is generation 599; `plasmids.txt` records every plasmid.
