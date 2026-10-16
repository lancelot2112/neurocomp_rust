# Island run 20 (600 generations)

Run 19's settings (shadows as hints, pauses drawn, recall's long ladder, shadow explorers, port
gates) with six changes, chosen together:
- **edges built on hints** (`ISL_HINT_USE=1`): a mutation adds a population of any kind on a
  hinted path, the way that feeds back (closes a loop, or runs from a higher level to a lower)
  three times in four;
- **output ports readable by any population** (`ISL_READ_OUT=1`): the act a step ago;
- **hint bands** (`ISL_HINT_BAND=1`): a shadow scores a target only by kernels partly right;
- **the self channel writable** (`ISL_SELF_WRITE=1`);
- **loops closed** (`ISL_LOOPS=1`): a mutation walks 1 to 4 populations forward and adds an edge
  back to the start.

Started 13:33Z, finished 16:41Z (3 h 8 min). Results are for the set, not any one change.

Stage moves (each passed on held-out worlds):
- reading: echo2 at 13;
- **recall: lag1 at 28, lag2 at 34, lag3 at 36** (the earliest of any run; run 19 at 61, 70, 74),
  lag4 at 127 (run 19 at 83);
- switching: tell at 62, feel at 157;
- shapes: blur1 at 167, then stuck until **blur2 at 572 and shift8 at 593**.

Goal means (best in brackets):
- **screen's glow 117 over 600 generations** (195), the best of any run (run 18: 105, run 19: 58);
- **selection's cued 388** (589 of its 600), the best of any run (run 19: 292);
- **recall's lag4 67** over 473 (run 19: 40);
- switching's feel 51 (58 over the last 100; run 19: 40, run 16: 67);
- the maze's feel 48 (run 19: 55); prediction's cycle1 20 (run 19: 22).

**Loops of every size:** at 50, four islands' best networks already had cross-channel loops (up
to 4 channels), where in run 19 every loop was a self-loop. The generalist's best network went
37, 72, 149, 78, 192, 77, 84, 47 and 5 populations at 50-generation checks, at its largest 184
of 192 on loops through 17 channels; recall's 67 of 71 through 11 at 250; prediction's 44 of 44
(no self-loops) through 5 at 350. By 550 every island had trimmed to 1 to 8 populations.

**Reading outputs spread:** a quarter of the populations on several islands, and 78-94% on the
maze and screen at times; screen's best network was often one population reading the act.

**Hints** fell from 47-82% of populations (run 19) to 0-49%, as the band rejected what the source
already determined, and few gates or pauses used them.

Plasmids: of 521 scored, 54 helped (10%, the most of any run), 88 hurt, 14 were mixed and 365
neutral. Largest on arrival: #346 (shapes to reading, 400) echo +469; #24 (reading to the
generalist, 40) babble +344; #292 (shapes to reading, 340) echo +342. #428 (screen to shapes,
480) was whole in 19 of shapes' 32 as its blur1 score rose to the pass at 572; #161 (screen to
selection, 180) was in all 8 of selection's elite at 200.

`checkpoint.txt` is generation 599; `plasmids.txt` records every plasmid.
