# Island run 6

Run 5's settings plus:
- a population of 32 and an elite of 8 (`ISL_POP=32`, `ISL_ELITE=8`; was 16 and 4);
- energy (`ISL_RATION=256`, `ISL_FOOD=32`). Each step a network gets 256 work and pays its
  work. It eats what it scores: every 50 steps the last block is scored by its task's probes,
  and each 1024 points are 32 rations. With its store below 0 it rests.

300 generations in 26 minutes (4-8 s a generation, at twice run 5's population). Started with
`runs/run6.cmd` through `Win32_Process.Create`, so it outlives the session.

Stage moves (each passed on 48 held-out worlds):
- switching: tell at 21, win-stay at 24, discrim at 148, swap1 at 153. Runs 4 and 5 never passed
  respond.
- recall: lag1 at 95, lag2 at 109, lag3 at 112, lag4 at 173.
- shapes: blur1 at 147, blur2 at 150, shift8 at 153, shift64 at 156, win-stay at 159.
- reading: echo2 at 268 (babble held out 1024).

Stalled:
- maze held its training goal for 50 generations (up to 396 against 200) and failed every
  held-out check (-44 to 88). Its elite scored 476-658 on the 12 training worlds and -69 to -177
  on fresh ones. Its blind controls scored about 0 on both, so it did read the cue. It had wired
  the hand's codes those training worlds drew, instead of learning them each lifetime, so on
  fresh worlds it turned the wrong way. Fixed in run 7 by fresh training worlds each generation.
- screen stayed at 3-35 on the torch. The torch shows the cursor's mark on every step, so
  clicking at what is seen clicks at the cursor: it is a tell, not a respond. Fixed in run 7 by a
  dark rung first (no cursor mark).
- prediction stayed on cycle1 (22-126).

Plasmids: of 223 scored on arrival, 27 helped (12%; 6% in runs 3 and 4), 32 hurt, 6 were mixed
and 158 neutral. The median size was 1 population, and 2 reached half an island; #160, a shapes
reflex trimmed by switching, was in 27 of 32 switching genomes. None made a lasting difference.

`checkpoint.txt` is generation 299. Recheck with
`RCK_INPLACE=1 RCK_READS=1 RCK_RATION=256 RCK_FOOD=32 RCK_FILE=runs/islands6/checkpoint.txt`.
