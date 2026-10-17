# Island run 27 (stopped at generation 536)

Run 26's settings (size classes, inhibit inputs, the overlap gene, the step rungs, feel at 160, the
easier reading rungs, goals judged on the elite, partial credit, passed rungs kept, no exchange)
with **thrift relaxed while an island is stuck** (`ISL_THRIFT_RELAX=64`: each selection takes the
thrift cases with chance 64/(64+g), g the generations on the current rung, full again after a
pass). Started 07:53 AM ET (11:52Z), stopped by hand at 1:41 PM ET (17:41Z) on 2026-10-17 (5 h 48
min; generations took 35 to 70 seconds against run 26's 17 to 20) to start run 28.

Stage moves (each passed on held-out worlds), 17 in all (run 26 by 536: 20):
- reading: babble at 6, echo2q at 92, echo2m8 at 403, echo2m4 at 404, then on echo2;
- switching: respond at 13, tell at 17 (run 26: tell at 486), then on feel-100;
- recall: respond at 32, lag1 at 58, lag2 at 63, then on lag3 (run 26: lag3 passed at 337);
- screen: glow-150 at 55, then on glow (run 26: glow passed at 274);
- shapes: clean at 152, blur1 at 243, blur2 at 244, shift8 at 265, shift64 at 266, then on
  feel-100 (run 26: clean to shift64 at 55 to 61, feel-100 at 281);
- prediction: cycle1-100 at 154, cycle1 at 270 (run 26: 480 and 565), then on cycle2.

Goal means over generations 0 to 536 against run 26's: selection 385 (run 26: 162), screen 148
(71), prediction 53 (9), maze 37 (43), recall 21 (24), shapes 18 (17), switching 7 (4), reading 1
(9).

**What relaxing thrift did:** the islands whose rungs stall for long (selection's cued, screen's
glow, prediction's cycles) scored much higher with room to grow (selection more than twice run
26's, prediction's cycle2 at 121 the best of any run), and switching and prediction passed the
early rungs run 26 sat on for 400 generations. But the islands that climb by small fast steps did
worse (shapes and recall later, screen never passed glow), and networks grew without a trim where
no rung passes: the generalist (no ladder, so never reset) reached 897 populations at 200, and
generations ran two to three times slower. Fewer rungs passed overall than run 26.

`checkpoint.txt` is generation 535; `rungs.txt` the 17 rung-passing genomes; no plasmids moved.
