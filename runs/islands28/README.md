# Island run 28 (stopped at generation 255)

Run 26's settings (size classes, thrift on and not relaxed, inhibit inputs, the overlap gene, the
step rungs, feel at 160, the easier reading rungs, goals judged on the elite, partial credit,
passed rungs kept, no exchange) with three changes:
- **a wiring cost** (`ISL_WIRE_COST=1`): each step a genome pays 1 for each bit of Hamming distance
  between the tags of each population's two ends, ports at fixed tags in the same space;
- **populations competing for their destination** (`ISL_COMPETE=1`): a word written by an earlier
  population's kernels this step is claimed, and a kernel whose output words are mostly claimed
  is held down and learns anti-Hebbian; the populations into a destination go strongest grip
  first;
- **pruning** (`ISL_PRUNE=1`): a population's grip (gene, 16 to start) goes up when it writes and
  down when it only loses; at 0 it lets go of its destination for the lifetime.

Started 1:42 PM ET (17:42Z), stopped by hand at 3:19 PM ET (19:19Z) on 2026-10-17 (1 h 37 min) to
start run 29.

Stage moves, 11 in all (run 26 by 255: 15):
- reading: babble at 6, echo2q at 224 (every run since 25 by 92 at the latest), then on echo2m8;
- recall: respond at 30, lag1 at 51, lag2 at 88, then on lag3;
- switching: respond at 95, then on tell;
- screen: glow-150 at 95, **glow at 150** (run 26: 274), then on torch;
- shapes: clean at 193, blur1 at 212, blur2 at 214, then on shift8 (run 26: clean to shift64 at 55
  to 61).

Goal means over generations 0 to 255 against run 26's: selection 203 (149), shapes 20 (9), recall 22
(13), maze 13 (43), reading 4 (19), screen 36 (140: here mostly on torch after 150), switching 0
(6), prediction 0 (0).

The contest (op lines, per 1000 steps of each island's probing): kernels held down by another
population's claim rose from tens to thousands as networks grew (the generalist 17,309 at 200,
switching 10,974), and populations let go at 0.1 to 7 per 1000 steps (prediction the most: 41 at
150). Best networks' wiring fell from up to 332 bits at 50 to 12 to 174 at 250.

So one writer a word helped where a single circuit should own the output (screen passed glow 120
generations sooner, selection scored higher) and slowed the islands whose first rungs need several
populations together or a wire into a port (reading's echo2q took 224 generations; shapes, the
maze and switching started slowly). `lesion.txt` is the lesion map of these genomes.

`checkpoint.txt` is generation 255; `rungs.txt` the 11 rung-passing genomes; no plasmids moved.
