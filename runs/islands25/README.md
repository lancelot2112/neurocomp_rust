# Island run 25 (600 generations)

Run 24's settings (run 22's, thrift on and no plasmid exchange, with goals judged on the elite,
partial credit and passed rungs kept) with **inhibit inputs** (`ISL_INHIBIT=1`), **the overlap
rule as a gene** (`ISL_OVERLAP=1`), **steps up to the rungs no run had passed**
(`ISL_STEPS=feel=100,glow=150,cycle1=100`, and feel's goal lowered from an unreachable 300 to 160,
`ISL_FEEL_GOAL=160`) and **easier reading rungs** (`ISL_MIRROR=1`: echo2q, echo2m8, echo2m4 before
echo2). Started 05:25Z, finished 08:49Z on 2026-10-17 (3 h 24 min).

Stage moves (each passed on held-out worlds), 14 in all (run 24 by its stop at 409: 11; ladders
differ, so the counts are not of the same rungs):
- reading: babble at 7, echo2q at 91, echo2m8 at 104, echo2m4 at 107 (the first echo rungs any run
  passed), then held on echo2;
- screen: glow-150 at 11 (its first rung in any run), then held on glow;
- switching: respond at 36, tell at 52, then held on feel-100;
- prediction: cycle1-100 at 141, cycle1 at 335 (passed once before, run 21), then held on cycle2;
- recall: respond at 193, lag1 at 335, lag2 at 336, then held on lag3 (run 24: respond at 17,
  lag3 by 19);
- maze: feel-100 at 214 (its first rung in any run), then held on feel (goal 160);
- shapes: clean at 482 (run 24: 64), then on blur1.

Goal means over the run (on whatever rung each island was on): switching 6, recall 25, reading 1,
prediction 30, maze 26, screen 116, shapes 24, selection 194. Over generations 0 to 409 against run
24's: maze 25 (24: 15), prediction 34 (30), screen 119 (99), shapes 21 (15), recall 25 (25),
switching 9 (5), selection 194 (277), reading 1 (1).

**What it showed:** the new rungs were climbable stepping stones: the maze and screen passed their
first rungs ever, prediction passed cycle1, and reading cleared every mirrored echo rung (it found
how to echo while hearing its own voice, as the hand-made two-wire circuit does). But none of it
carried over: reading scored about 0 on echo2 for 490 generations, where the sound's code must be
learned in the lifetime. The islands whose ladders did not change did worse than in run 24: recall
took 193 generations to pass respond (run 24: 17), shapes 482 to pass clean (64), and selection
averaged 194 against 277. Inhibit inputs were taken up unevenly: at the end 82% of reading's
populations and 49% of the generalist's carried one, 8 to 31% elsewhere; the overlap gene barely
moved off its default of 2 quarters. Whether inhibition is what cost recall, shapes and selection
is not settled by this run (three changes in one).

`checkpoint.txt` is generation 599; `rungs.txt` the 14 rung-passing genomes; no plasmids moved.
