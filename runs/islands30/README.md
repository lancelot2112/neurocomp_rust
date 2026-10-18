# Island run 30 (stopped at generation 266)

Run 29's settings (run 28's with branches) with two changes:
- **sparse one-word codes on reading** (`ISL_ONE_WORD=8`): every reading lifetime's glyphs, sounds
  and voice codes drawn as 8 random lines in the first word of their channels, so a glyph, its
  sound and its voice code sit at the same place, and a word holding three codes stays readable;
- **a fixed body** (`ISL_BODY=1`): the voice's and the hand's codes, the sound the voice makes for
  each token (the teacher's too) and the feel of each move drawn from one seed for the whole run,
  as anatomy is; glyphs, lights, tones, cues and what pays still drawn fresh each lifetime.

Started 6:58 PM ET (22:58Z), stopped by hand at 9:08 PM ET (01:08Z, 2026-10-18) on 2026-10-17
(2 h 10 min) to start run 31.

Stage moves, 24 in all (run 29: 15 in 535 generations), 7 of them past rungs run 29 never passed:
- reading: echo2q at 12, then on echo2q for 254 generations (run 29: echo2 by 49), although one
  wire from the ear to the voice scores 503 there under these codes (examples/mirrorcheck);
- shapes: blur1 to feel-100 at 12 to 18, **feel at 230, repeat at 233, win-stay at 234** (feel
  first passed by any island since its goal was set, repeat near 0 in earlier runs), then on
  win-stay;
- screen: **glow at 20** (run 29 on glow from 53 to the end), **torch at 42**, then on torch;
- switching: tell at 28, feel-100 at 29, then on feel-100;
- recall: lag1 to **lag4 at 33 to 36** (run 29 stopped on lag3), **lag5 at 63, lag8 at 155, lag16
  at 209**, then on lag16 (only run 21 had passed every delay rung);
- maze: **feel-100 at 49, feel at 50**, then on learn;
- prediction: **cycle1-100 at 215, cycle1 at 255** (no run had passed the step rung), then on
  cycle2.

At 250 against run 29 at 250: ahead on recall (lag16 / lag3), prediction (cycle1 / cycle1-100),
maze (learn / feel-100), screen (torch / glow) and shapes (win-stay / feel-100); behind on reading
(echo2q / echo2); level on switching and selection. Goals held on training but not on held-out
worlds: 44 times.

Goal means over generations 0 to 265 (on the rungs each run was on, so not like for like): recall
49 (run 29: 25), selection 288 (274), prediction 14 (0), reading 13 (8), shapes 7 (6), maze 6 (25),
screen 26 (113), switching 1 (4).

Networks grew where tasks gathered: at 200 the generalist's best had 389 wired populations and
2,417 bits of wiring (run 29's best at most 92 populations), and 15 of recall's 32 genomes had 64
to 255 tokens; reading's and selection's stayed at 2 or 3 wired populations.

The paired-associate capacity curve (examples/pairscheck, runs/checks/pairscheck-capacity.txt)
scored this run's best reading, generalist and recall genomes at about 0 for every N from 2 to 128
(a lookup table 148-243): the fixed body let evolution build in what used to be learned (moves,
feel, echo-like reflexes), but no genome learns an arbitrary map in its lifetime. That is what run
31 adds the three-factor output rule for.

Files: `log.txt`, `checkpoint.txt` (generation 265), `rungs.txt` (the best genome at each rung
passed).
