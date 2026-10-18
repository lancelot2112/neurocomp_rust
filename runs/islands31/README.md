# Island run 31 (stopped at generation 411)

Run 30's settings (sparse one-word codes on reading, a fixed body) with two changes:
- **the three-factor output rule** (`ISL_ELIG=1`): a kernel that fired k steps ago is checked
  while its population's gate (the modulator) is open, against what its destination holds then;
  right, it counts up; wrong, an output line moves toward what is there and it counts down, and
  wrong at 0 it dies, freeing its input for a new birth;
- **reading's paired-associate rungs** (`ISL_PAIRS=1`): pairs8 and pairs32 after echo2q,
  glyph-to-token pairs drawn each lifetime, each token heard like its voice code, goal copy 100.

Started 9:09 PM ET (01:09Z, 2026-10-18) on 2026-10-17, stopped by hand at 1:50 AM ET (05:50Z) on
2026-10-18 (4 h 40 min) to start run 32, after the rung audit (wiki experiment 17).

Stage moves, 25 in all (run 30: 24 in 266 generations):
- reading: echo2q at 11, pairs8 at 24, then on pairs8 to the end (copy about -20 to -30: no genome
  learns the pairs);
- switching: tell at 19, feel-100 at 79, feel at 88, repeat at 98, then on repeat;
- screen: glow at 20, torch at 47, then on torch;
- recall: lag1 to lag3 at 50 to 61, lag4 at 105, lag5 at 189, lag8 at 279, lag16 at 281,
  distract at 305, then on distract;
- shapes: blur1 at 67, blur2 at 149, then on blur2;
- maze: feel at 215, learn at 283, then on learn;
- prediction: cycle1 at 283, cycle2 at 295, cycle3 at 302, **cycle4 at 313** (the first run past
  cycle2), order2 at 314, then on order2.

At 250 against run 30 at 250: ahead on switching (repeat / feel-100) and reading (pairs8 /
echo2q; pairs8 is a rung run 30 did not have); behind on recall (lag5 / lag16), prediction
(cycle1-100 / cycle1), maze (feel / learn) and shapes (blur2 / win-stay); level on screen (torch)
and selection (cued). Then a burst from 279 to 314: recall lag8 to distract, prediction cycle1 to
order2, maze learn. Goals held on training but not held out: 53 times.

Goal means over generations 0 to 265 (on the rungs each run was on): selection 267 (run 30: 288),
shapes 27 (7), recall 23 (49), screen 21 (26), switching 8 (1), maze 6 (6), prediction -1 (14),
reading -14 (13; pairs8's copy).

What the passes are (rescored on fresh worlds, wiki experiment 17):
- **recall** lag16: 8 genomes 185-223 held out, by a latch (one 64-bit word reset by the light and
  refilled by OR over about 20 steps), not a loop that holds a code;
- **prediction** cycle4: 7 of the first 9 genomes 177-272 held out; genome 6's word trace shows a
  rare-token detector (an eye-to-C2 NOR that fires after the cycle's tokens; its silence lets a
  C2-hand loop click). The audit then showed every cycle rung of the old streams passed by rarity
  and order2 by a model of the token before: real passes, not prediction. Run 32 has fair streams;
- **recall** distract: passed at 305 on a rung a timer from any light could pass (run 32 fixes it).

The three-factor rule did not make pairs learnable here: reading sat on pairs8 for 387
generations.

Files: `log.txt`, `checkpoint.txt` (generation 410), `rungs.txt` (the elite at each rung passed),
`plasmids.txt`.
