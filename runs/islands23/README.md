# Island run 23 (stopped at generation 296)

Run 22's settings (no plasmid exchange, heat inputs, loops closed, self writes, hint bands, output
reads, edges on hints toward feedback, shadows as hints, pauses drawn, recall's long ladder, shadow
explorers, port gates) with **thrift off** (`ISL_THRIFT=0`: no lexicase case for less work) and
**the carrying cost paid as energy** (`ISL_CARRY_ENERGY=1`: each step a network pays 4 a token
of its genome from the same store its work comes out of). Started 00:48Z, stopped by hand at
02:58Z on 2026-10-17 (2 h 10 min) after the check at generation 250.

Stage moves (each passed on held-out worlds), 4 in all (run 22 by 296: 8):
- reading: echo2 at 12 (run 22: 17);
- switching: tell at 123, feel at 124 (run 22: 23 and 33);
- recall: lag1 at 231 (run 22: lag1 at 42 to lag4 at 161).

Goal means over generations 0 to 296 against run 22's over the same generations (best in
brackets; on whatever rung each island was on):
- switching 8 (163; run 22: 28), recall 44 (107; run 22: 78), prediction's cycle1 0 (13; run 22: 62);
- maze's feel 9 (53; run 22: 36), screen's glow 54 (152; run 22: 77), shapes 0 (16; run 22: 18,
  past blur1 at 245);
- selection's cued 400 (477; run 22: 318), the only goal better than run 22's; reading -7 (run 22: 17).

**What thrift did:** without it the networks grew at once and kept growing: at generation 50
reading's best had 277 populations, at 200 684 (560 on loops through 27 channels), the generalist's 217
at 200, selection's 320 at 250; at 50 run 22's ranged from 2 to 31. None of it passed rungs:
the first rungs (simple reflexes: respond, tell) came 100 to 190 generations later than in run
22, and prediction never left 0. Paying for carrying out of the energy store did not hold size
down. Thrift stays on: from run 24 on the runs are built on run 22, with size classes (run 26) and
thrift relaxed while stuck (run 27) as gentler ways to let networks grow.

`checkpoint.txt` is generation 295; no plasmids moved.
