# Island run 18 (stopped at generation 548)

Run 17's settings (shadow explorers keeping predictors, recall's long ladder, port gates) plus
pauses drawn as gates are (`ISL_PAUSES=1`): a new population is paused on a channel or port one
time in four, and a pause made by mutation draws what it names in the same step. Started 06:59Z;
stopped at 548 (3 h 32 min) to start run 19.

Stage moves (each passed on held-out worlds):
- recall: lag1 at 57, lag2 at 92, lag3 at 93 (then stuck: mean 14 over 456 generations);
- switching: tell at 114, feel at 158;
- reading: echo2 at 18, never passed;
- shapes never passed its first rung (clean, mean 40 over 549 generations).

Means over the run: **screen's glow 105** (best 201), the highest of any run (run 16: 62);
selection's cued 238 (best 549); prediction's cycle1 56 (best 193); switching's feel 36 (run 16:
67); the maze's feel 26.

**Pauses on ports spread at once:** by 50, 28-55% of the elite on four islands (run 16: 4 of
4,180 populations at 55). Screen's best scores came from 1 to 4 populations, 67-92% paused on a
port (and 75% gated at 250).

**The kept predictors bloated islands in turn again,** as in run 17: switching 155 at 50, the
generalist 410 at 200 (108 populations), selection 258 at 150, recall 144 at 300. The islands that
carried them did worse; shapes never left its first rung.

Plasmids: of 485 scored, 23 helped (5%), 84 hurt, 16 were mixed and 362 neutral. #232 (shapes to
selection, 260) gave select +307 on arrival.

`checkpoint.txt` is generation 545; `plasmids.txt` records the plasmids up to 540.
