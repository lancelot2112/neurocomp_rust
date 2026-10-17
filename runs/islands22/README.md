# Island run 22 (600 generations)

Run 21's settings (heat inputs, loops closed, self writes, hint bands, output reads, edges on
hints toward feedback, shadows as hints, pauses drawn, recall's long ladder, shadow explorers,
port gates) with **no plasmid exchange** (`ISL_EVERY=0`): the islands evolve apart, to measure what
the exchange does. Started 21:05Z, finished 00:48Z (3 h 43 min).

Stage moves (each passed on held-out worlds), 15 in all (run 21: 16):
- reading: echo2 at 17 (run 21: 21);
- switching: tell at 23, feel at 33 (run 21: 24 and 44);
- recall: lag1 at 42, lag2 at 67, lag3 at 72, lag4 at 161, lag5 at 400, lag8 at 408, lag16 at
  582 (run 21: lag1 at 31 to distract at 104);
- shapes: blur1 at 245, blur2 at 250, shift8 at 252, shift64 at 258, feel at 259 (run 21: blur1
  at 104, feel at 442).

Goal means against run 21 (best in brackets):
- **prediction's cycle1 78** (169; run 21: 58), the best of any run;
- **switching's feel 35** (67; run 21: 7) and **the maze's 45** (104; run 21: 24): the feel rung
  did far better without the exchange;
- screen's glow 105 (run 21: 99); selection's cued 251 (run 21: 274); reading's echo2 -6 in both;
- recall took much longer to climb the delay rungs (lag4 passed at 400, against 73), but reached
  lag16 on its own.

**What the exchange did:** it got recall up its delay rungs fast (in run 21 a plasmid came just
before each early pass) and shapes off its first rung early; without it most islands scored as
well or better within a rung, the feel rung most of all. From run 23 on the exchange stays off.

Networks: loops crossed channels on every island as in runs 20 and 21 (the generalist's best 92 of
97 populations on loops through 10 channels at 450; reading's 80 of 87 through 17 at 100); heat
inputs on every island, warm or cool by island.

`checkpoint.txt` is generation 599; no plasmids moved.
