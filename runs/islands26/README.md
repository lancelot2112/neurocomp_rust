# Island run 26 (600 generations)

Run 25's settings (run 24's with inhibit inputs, the overlap gene, steps before the rungs no run
had passed, feel's goal at 160 and the easier reading rungs) with **size classes**
(`ISL_SIZE_CLASSES=1`): each island's genomes sorted by token count into classes a factor of four
apart (1-3, 4-15, 16-63, 64-255, 256+); each occupied class keeps its own lexicase choice among the
elite, and half of all children have a parent chosen within a class drawn at random. Started
08:50Z, finished 11:52Z on 2026-10-17 (3 h 2 min).

Stage moves (each passed on held-out worlds), **21 in all, the most of any run** (run 25: 14, run
22: 15):
- shapes: clean at 55, blur1 at 57, blur2 at 59, shift8 at 60, shift64 at 61, feel-100 at 281, then
  held on feel (run 25: clean at 482);
- recall: respond at 51, lag1 at 54, lag2 at 55, lag3 at 337, then held on lag4 (run 25: respond at
  193, held on lag3; run 24 never passed lag3);
- reading: babble at 20, echo2q at 125, echo2m8 at 175, echo2m4 at 176, then held on echo2;
- screen: glow-150 at 41, **glow at 274** (no run had passed glow: run 25's best 10-generation mean
  was 213 of 300), then held on torch;
- maze: feel-100 at 142 (run 25: 214), then held on feel;
- switching: respond at 34, tell at 486, then held on feel-100;
- prediction: cycle1-100 at 480, cycle1 at 565, then on cycle2.

Goal means over the run (on whatever rung each island was on): maze 44 (run 25: 26), recall 24
(25), shapes 16 (24), selection 164 (194), screen 63 (116: here mostly on torch after 274),
prediction 17 (30), reading 8 (1), switching 4 (6).

Size classes kept every island's population spread over three or four classes all run (for example
at 150: switching 6/3/21/2 of 32 in 1-3/4-15/16-63/64-255 tokens, shapes 4/13/6/9); no genome
reached 256 tokens. The islands that were slowest in run 25 (recall, shapes) climbed fastest here,
which fits small genomes that do one simple thing well no longer being crowded out. Switching and
prediction were the exceptions: each sat on an early rung (tell, cycle1-100) for over 400
generations that run 25 passed at 52 and 141, and passed it only near the end. Reading again
cleared the easier echo rungs and then scored about 0 on echo2.

`checkpoint.txt` is generation 599; `rungs.txt` the 21 rung-passing genomes; no plasmids moved.
