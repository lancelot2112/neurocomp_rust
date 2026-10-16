# Island run 19 (600 generations)

Run 18's settings (pauses drawn as gates are, recall's long ladder, shadow explorers, port gates)
with shadows as indicators (`ISL_SHADOW_HINT=1`): a shadow starts only by accident, at its
population's own rate, and what it predicts well is not built but noted on its population as a
hint, which that population's gate or pause takes half the time when they mutate. Started
10:31Z, finished 13:33Z (3 h 2 min).

Stage moves (each passed on held-out worlds):
- reading: echo2 at 27, never passed;
- switching: tell at 43, feel at 44;
- recall: lag1 at 61, lag2 at 70, lag3 at 74, **lag4 at 83** (the earliest of any run; run 16 at
  353), then stuck on lag4 (mean 40, best 101 of 150);
- shapes: clean at 134, then blur1 to feel between 134 and 140.

The feel rung (goal 300), none passed:
- **the maze: mean 55 over 600 generations, 66 over the last 100, best 118** (run 16: 36 and 69);
- shapes: mean 52 over 460 (best 97; run 16: 46);
- switching: mean 40 over 556, 65 over the last 100 (best 95; run 16: 67).

Other islands: screen's glow mean 58, 101 over the last 100 (best 164); selection's cued 292
(best 489); prediction's cycle1 22 (57 over the last 100, best 126; run 16 passed it at 332).

**No predictors and no bloat:** best networks of 2 to 58 populations all run (run 17's generalist
had 99). Hints sat on 0-82% of an island's populations; the maze used them most (31-35 gates or
pauses on a hint at 500-550). In the checkpoint at 445, hinted populations were on a loop less
often than the rest (45 of 790, 6%, against 24%), and most hints named things the network writes
or the runtime derives from it (the self channel 357, the hand 110).

**Every loop is a self-loop:** at 550 the nine best networks had 10 populations on loops, all
self-loops; 107 of 323 genomes declared no channel.

Plasmids: of 522 scored, 41 helped (8%), 84 hurt, 21 were mixed and 376 neutral. The largest gains
on arrival were on selection (#323 select +668 at 360, #448 +412 at 500) and reading (#365 echo
+352 at 420); each was gone within a few generations. #343 (maze to switching, 400) was whole in
all of switching's top 8 at 420.

`checkpoint.txt` is generation 599 (the run's last checkpoint); `plasmids.txt` records every
plasmid.
