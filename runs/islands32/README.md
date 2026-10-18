# Island run 32 (finished, 600 generations)

Run 31's settings with:
- **motions** (`ISL_MOTIONS=1`): conveyor channels (delay lines on time scales a factor of two
  apart) and rule-90 waves;
- **integer triggers** (`ISL_TRIGGERS=1`): a kernel may read a lane of 8 to 64 bits as a number
  (at least, at most or near a value, near a value it learns, or one lane greater or less than
  another), beside the popcount triggers;
- **the generalist on its own ladders** (`ISL_FRONTIER_LADDERS=1`): its own rung on every island's
  ladder, climbed as that island climbs it, with one passed rung per ladder kept in selection;
- **the rung audit's fixes**: recall's fair last rungs (`ISL_RECALL_FAIR`: two, then distract with
  two lights, then vary with a tone and early clicks spending the trial), partial credit on
  remember (`ISL_PARTIAL_REMEMBER`), fair prediction streams (`ISL_FAIR_SEQUENCES`), echo and copy
  per token shown (`ISL_FAIR_SAID`), feel's goals over what random moves reach (feel-300, feel 500).

Started 1:50 AM ET (05:50Z) and finished 6:44 AM ET (10:44Z) on 2026-10-18 (4 h 54 min). Run 33
started from it at once (`runs/after32.cmd`).

Rung moves, 28 in all (run 31: 25 in 411 generations): 19 by islands, 9 by the generalist on its own
ladders.
- recall: lag1 at 45 to lag4 at 104, lag5 at 218, lag8 at 225, lag16 at 329, **two at 409**
  (lag16 passed held out at 243), then on two at about 0 to the end: its memory holds that a light
  came, not which;
- shapes: blur1 at 91, blur2 at 131, then **shift8, shift64 and feel-300 at 321 to 323** (run 31
  stopped on blur2), then on feel-300 (best 78 of 300);
- reading: echo2q at 10, pairs8 at 65, then on pairs8 to the end (copy about -10 to -25);
- screen: glow at 16, torch at 21, then on torch;
- switching: tell at 12, then on tell (best about 49 of 100);
- maze: **feel at 474** (feel-300 passed; feel-100 in run 31), then on feel (goal 500);
- prediction: none. On the fair streams cycle1-100 stayed at about 0 all run; run 31 passed it at
  283 and climbed to order2 by rarity, which fair streams take away;
- selection: none, cued at 270 to 410 of 600 (the elite babbles and never clicks; see run 33's
  partial credit on select);
- the generalist on its own ladders: prediction cycle1-100 and cycle1 by 8 (one token: a new token
  is a break), reading babble at 23, switching respond and recall respond to lag2 at 53 to 56,
  screen glow-150 at 56 and glow at 76; then flat (its own rungs about 0 to 20).

At 250 against run 31 at 250: ahead on recall (lag8 / lag5), behind on switching (tell / repeat) and
maze (feel-300 / feel, a higher goal), level on the other five. Goals held on training, checked held
out: 45 times.

Goal means over generations 0 to 265: selection 269 (run 31: 267), maze 40 (6), shapes 16 (27),
switching 13 (8), recall 3 (23), screen 3 (21), prediction 0 (-1), reading -8 (-14).

Ops (`ops-NNN.txt`, every 50 from 225, written and read by): integer triggers spread where a word
can be read as a number. Switching read half its inputs as "a whole word near a value" (int~/64)
from 225 to 500, then dropped them by 550; screen read most of its inputs as "a word at most a value"
(int<=/64, 33% at 300, 80% at 550); recall kept "at least a value" (int>=/64) at 10 to 17%; the
generalist moved to two-lane comparisons (lanes>/32, 72% at 250) and then band triggers (75% at
500). Every integer trigger read whole 64-bit words but one 8-bit lane in recall at 550. The
generalist stayed small (3 to 15 populations), scored only on rungs it reached; maze's and recall's
best grew to 100 to 137 populations, nearly all on loops.

Files: `log.txt`, `checkpoint.txt` (generation 599), `rungs.txt` (the elite at each rung passed),
`plasmids.txt`, `ops-225.txt` to `ops-550.txt`.
