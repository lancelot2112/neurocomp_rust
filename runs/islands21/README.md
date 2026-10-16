# Island run 21 (600 generations)

Run 20's settings (edges on hints toward feedback, output reads, hint bands, self writes, loops
closed, shadows as hints, pauses drawn, recall's long ladder, shadow explorers, port gates) with
heat inputs (`ISL_HEAT=1`): a population may carry a channel or port whose fullness a frame ago
moves every kernel's threshold, down when warm and up when cool, by its own threshold times
popcount over bits. Started 16:41Z, finished 21:05Z (4 h 24 min).

Stage moves (each passed on held-out worlds), 16 in all:
- reading: echo2 at 21;
- switching: tell at 24 (the earliest of any run), feel at 44;
- **recall: lag1 at 31, lag2 at 41, lag3 at 48, lag4 at 66, lag5 at 73, lag8 at 74, lag16 at 88
  and distract at 104**: every delay rung, where no earlier run passed lag4 (run 20 stuck on it
  for 473 generations, run 19 for 517);
- shapes: blur1 at 104, then blur2 at 429, shift8 at 437, shift64 at 438 and feel at 442.

Goal means (best in brackets):
- **prediction's cycle1 58 over 600 generations** (148), the best of any run (run 20: 20);
- screen's glow 99 (241, the best of any run; run 20: 117);
- selection's cued 274 (456; run 20: 388);
- recall's distract 34 over 496 (127);
- switching's feel 7 over 556 (44; run 20: 51) and the maze's 24 (65; run 20: 48): the feel rung
  went backward;
- reading's echo2 -6.

**Heat inputs spread at once** and stayed: 20-90% of the elite's populations on every island,
warm on some (recall, prediction, selection, the generalist: hundreds warm) and cool on others
(the maze and reading, almost all cool). Loops crossed channels on most islands as in run 20 (the
generalist's best 99 of 114 through 12 channels at 400, the maze's 51 of 70 through 7 at 550).

Plasmids: of 521 scored, 34 helped (7%), 99 hurt, 14 were mixed and 374 neutral. Largest on
arrival: babble +488 and +472, select +464 (#511, shapes to the generalist at 580, a circuit that
reaches nothing: probably by changing which channels the host's own tags resolve to) and +452.
#385 and #456 were each whole in all 8 of the generalist's elite for a time.

`checkpoint.txt` is generation 599; `plasmids.txt` records every plasmid.
