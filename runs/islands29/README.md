# Island run 29 (stopped at generation 535)

Run 28's settings (run 26's with a wiring cost, populations competing for their destination by
grip, and pruning) with one change:
- **branches** (`ISL_BRANCH=1`): a population's fired kernels may also write their outputs into a
  second vector, a channel or an output port, at the same place by OR, as an axon's collaterals
  (an efference copy, a line to another population's inhibit input, two targets driven by one
  decision); a new population has one one time in four; under competition a branch writes only
  unclaimed words.

Started 3:20 PM ET (19:20Z), stopped by hand at 6:57 PM ET (22:57Z) on 2026-10-17 (3 h 37 min) to
start run 30.

Stage moves, 15 in all, every one by generation 74 (run 28 by 255: 11), none in the 460
generations after:
- reading: echo2q at 7, echo2m8 at 19, echo2m4 at 47, echo2 at 49 (run 28: echo2q at 224), then
  on echo2;
- shapes: blur1 at 17, blur2 at 18, shift8 at 20, shift64 at 23, feel-100 at 24 (run 28: blur2 by
  214), then on feel-100;
- switching: tell at 52, feel-100 at 54, then on feel-100;
- screen: glow at 53 (run 28: 150), then on glow;
- recall: lag1 at 59, lag2 at 73, lag3 at 74, then on lag3.

At 255 against run 28 at 255 (same ladders): ahead on switching (feel-100, run 28 tell), reading
(echo2, echo2m8) and shapes (feel-100, shift8), behind on screen (glow, torch), level on the rest.
Screen held its goal on training 17 times and shapes 14 but the elite fell short on the held-out
worlds every time (screen as near as 271/300 at 534).

Goal means over generations 0 to 255 against run 28's (on the rungs each was on): selection 270
(204), screen 112 (37), maze 25 (14), recall 25 (22), reading 8 (4), shapes 6 (21), switching 4 (1),
prediction 0 (1). Over the whole run: selection 315, screen 117, maze 30, recall 25.

Branches (op lines every 50 generations, the share of the elite's populations with a branch; a new
population has one a quarter of the time): selection 71%, prediction 34%, generalist 27%, shapes
26%, recall 22%, switching 16%, reading 15%, maze 3%, screen 2%. Selection keeps them (its
populations branch far more than chance), maze and screen lose them. The contest, per 1000 steps:
kernels held down by another population from 292 (screen) and 332 (selection) to 4166 (switching);
populations letting go from 0.08 (selection) to 12.83 (prediction).

So branches came with a fast start (every move by 74, against run 28's 11 moves by 255), then
nothing: the same rungs no run has passed (feel-100, echo2, lag3, glow, cycle1-100, cued) held
every island. Selection, the island that uses branches most, scores highest on its cued rung, but
not to its goal of 600.

Files: `log.txt`, `checkpoint.txt` (generation 535), `rungs.txt` (the best genome at each rung
passed).
