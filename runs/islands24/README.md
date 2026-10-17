# Island run 24 (stopped at generation 409)

Run 22's settings (no plasmid exchange, thrift on, heat inputs, loops closed, self writes, hint
bands, output reads, edges on hints toward feedback, shadows as hints, pauses drawn, recall's long
ladder, shadow explorers, port gates) with **goals judged on the elite** (`ISL_ELITE_PASS=1`),
**partial credit** in selection (`ISL_PARTIAL=1`: graded echo, surprise and feel) and **passed
rungs kept** as selection cases (`ISL_KEEP_RUNGS=1`), and the first rung log
(`ISL_RUNG_LOG`: `rungs.txt` holds the best genome at each of the 11 passes). Started 02:59Z, stopped
by hand at 05:25Z on 2026-10-17 (2 h 26 min) to start run 25.

Stage moves (each passed on held-out worlds), 11 in all (run 22 by 409: 14):
- reading: echo2 at 6 (run 22: 17);
- switching: tell at 16, feel at 94 (run 22: 23 and 33);
- recall: lag1 at 17, lag2 at 18, lag3 at 19, then held on lag3 (run 22: lag1 at 42, lag2 at 67,
  lag3 at 72, lag4 at 161, lag5 at 400, lag8 at 408);
- shapes: blur1 at 64, blur2 at 229, shift8 at 235, shift64 at 247, feel at 272 (run 22: blur1 at
  245 to feel at 259).

Goal means over generations 0 to 409 against run 22's (best in brackets; on whatever rung each
island was on): switching 5 (140; run 22: 31), recall 25 (127; run 22: 84), prediction 30 (108;
run 22: 70), maze 15 (63; run 22: 37), screen 99 (201; run 22: 93), shapes 15 (138; run 22: 14),
selection 277 (449; run 22: 285), reading 1 (run 22: 12).

**What the three did:** judging the goal on the elite moved islands up as soon as their best
genomes passed: recall took lag1 to lag3 in three generations and shapes four rungs in 18
generations, the first rungs sooner than in any run. But the islands then scored far less within
a rung than run 22's did, most of all on the rungs that stall (switching's and the maze's feel 5
and 15 against 31 and 37; recall's lag3 25 against 84), and recall never passed lag4. The networks
stayed small (most islands' best 1 to 15 populations at 400; the generalist's 125 and the maze's
57). The feel rung's goal of 300 turned out to be out of reach for any network
(`examples/feelcheck.rs`), so three islands spent the run's second half on a rung they could not
pass; run 25 lowers it to 160.

`checkpoint.txt` is generation 405; `rungs.txt` the 11 rung-passing genomes; no plasmids moved.
