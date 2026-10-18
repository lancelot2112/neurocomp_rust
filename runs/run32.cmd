@echo off
rem Island run 32: run 31 (the three-factor output rule, the pairs rungs) with motions (ISL_MOTIONS=1: conveyor channels, delay lines on several time scales, and rule-90 waves) the generalist climbing every ladder itself (ISL_FRONTIER_LADDERS=1) and triggers with integer cases (ISL_TRIGGERS=1), and recall's fair last rungs (ISL_RECALL_FAIR=1) with partial credit on remember (ISL_PARTIAL_REMEMBER=1). 600 generations.
rem Started with Win32_Process.Create so it does not end with the session that launched it.
cd /d C:\Software\source\neurocomp
if not exist runs\islands32 mkdir runs\islands32
echo ---- run 32: run 31's settings (run 30's settings (run 29's settings (run 28's settings (run 26's settings [run 25's settings [run 24's settings [run 22's settings [no plasmid exchange, heat inputs, loops closed, self writes, hint bands, output reads, edges on hints toward feedback, shadows as hints, pauses drawn, recall's long ladder, shadow explorers, port gates, the feel rung, task selection, accidents, two-mask kernels, marks, exploration, the frontier generalist, fresh worlds] with goals judged on the elite [ISL_ELITE_PASS=1: a rung's goal is checked on the elite's mean, not the whole population's, before the held-out check] and partial credit [ISL_PARTIAL=1: selection uses graded echo, surprise and feel, a near or late answer counting in part; goals stay on the probes themselves; and passed rungs kept, ISL_KEEP_RUNGS=1: each generation one rung already passed is scored again in its own world and joins the selection cases]] with inhibit inputs [ISL_INHIBIT=1: a population may carry an inhibit input, a channel or port whose lines a frame ago are taken before its kernels fire, so a kernel whose mask lies more than half on them is held down; a new population has one one time in four] and the overlap rule as a gene [ISL_OVERLAP=1: a population's inhibit_quarters, 1 to 4 starting at 2, sets how much of a kernel's mask taken lines must cover to hold it down; 4 lets kernels on the same spot all fire] and steps up to the rungs no run has passed [ISL_STEPS=feel=100,glow=150,cycle1=100 and ISL_FEEL_GOAL=160: feel-100 before feel, whose goal is lowered from 300, out of reach, to 160, on switching's, shapes' and the maze's ladders, glow-150 before screen's glow, cycle1-100 before prediction's cycle1, the same worlds with lower goals] and easier reading rungs [ISL_MIRROR=1: echo2q, echo2m8 and echo2m4 before echo2, each token heard like the voice's code for it, first without one's own voice]] with size classes [ISL_SIZE_CLASSES=1: each island's genomes sorted by size into classes a factor of four apart, 1-3 tokens, 4-15, 16-63, 64-255 and 256+; each occupied class keeps its own lexicase choice among the elite, and half of all children have a parent chosen within a class drawn at random, so small and large networks both stay in the population]] with a wiring cost [ISL_WIRE_COST=1: each step a genome pays 1 for each bit of Hamming distance between the tags of the two ends of each of its populations, ports at fixed tags in the same space, as the carrying cost is paid; near ends cheap, far ones dear, so circuits that work together gather in one part of tag space near the ports they use], populations competing for their destination [ISL_COMPETE=1: the populations into a destination run strongest grip first, ties in genome order, and each word an earlier population's kernels wrote this step is claimed, and a kernel whose output words are mostly claimed is held down and learns anti-Hebbian, so the winners write the act, not the OR of every reflex] and pruning [ISL_PRUNE=1: each population holds its destination with its gene grip, 16 to start, up one a step it writes, down one a step it only loses, letting go for the lifetime at 0]) with branches (ISL_BRANCH=1: a population's fired kernels may also write their outputs into a second vector, a channel or an output port, at the same place by OR, as an axon's collaterals: an efference copy, a line to another population's inhibit input, two targets driven by one decision; a new population has one one time in four; under competition a branch writes only unclaimed words)) with sparse one-word codes on reading (ISL_ONE_WORD=8: every reading lifetime's glyphs, sounds and voice codes drawn as 8 random lines in the first word of their channels, so a glyph, its sound and its voice code sit at the same place and a word holding three codes stays readable) and a fixed body (ISL_BODY=1: the voice's and the hand's codes, the sound the voice makes for each token, which the teacher speaks too, and the feel of each move drawn from one seed for the whole run, as anatomy is, so evolution can build them in; glyphs, lights, tones, cues and what pays are still drawn fresh each lifetime, so learning is forced by the world)) with the three-factor output rule (ISL_ELIG=1: a kernel that fired k steps ago is checked while its population's gate, the modulator, is open, against what its destination holds then, less its own population's writes; right, it counts up; wrong, an output line moves toward what is there and it counts down, and wrong at 0 it dies, so its input is free for a new birth; a new population has it one time in four) and reading's paired-associate rungs (ISL_PAIRS=1: pairs8 and pairs32 after echo2q, glyph-to-token pairs drawn each lifetime, each token heard like its voice code, goal copy 100)) with motions (ISL_MOTIONS=1: a channel may move by itself each step before any population writes, a conveyor whose words move one place along every 2^(k-1) steps, so word j holds what was written j moves ago, delay lines on time scales a factor of two apart, or a travelling wave, each line the XOR of its neighbours, cellular rule 90; a new token is a motion one time in about twenty) and the generalist climbing every ladder itself (ISL_FRONTIER_LADDERS=1: a rung of its own on each island's ladder, scored in that island's world, with one rung below it drawn at random so what it passed stays in its selection; it moves up each ladder as that island does, the goal on its elite's mean for the rung's hold, then the held-out check) and triggers (ISL_TRIGGERS=1: a population's kernels match at least their threshold, at most it, in a band, or over the whole tile; or as integers, the lane of 8 to 64 bits holding a kernel's first mask line read as a number: at least, at most or near its threshold's share of the lane's range, near a number of its own that it learns, or the first of two lanes greater or less than the last); and recall's fair last rungs (ISL_RECALL_FAIR=1: two, then distract with two lights and the flash the light the trial did not show first, then vary with two lights, a tone as the window opens and a click before it spending the trial; the audit found the old distract and vary passed by clicking a fixed time after any light) with partial credit on remember (ISL_PARTIAL_REMEMBER=1: in selection, a click in the two steps after the window counts half; the goal stays on remember itself); 600 generations> runs\islands32\log.txt
set ISL_RATION=256
set ISL_FOOD=32
set ISL_POP=32
set ISL_ELITE=8
set ISL_OPS=1
set ISL_PLASMID_MAX=6
set ISL_EXCLUDE=1
set ISL_CARRY=4
set ISL_INPLACE=1
set ISL_READS=1
set ISL_SCREEN=1
set ISL_MAZE=1
set ISL_PREDICTION=1
set ISL_SHAPES=1
set ISL_LADDERS=1
set ISL_CKPT=5
set ISL_HELD=48
set ISL_GENS=600
set ISL_FRESH=1
set ISL_FRONTIER=1

set ISL_EPI=1

set ISL_EXPLORE=1
set ISL_ZEROS=1
set ISL_ACCIDENT=1

set ISL_SELECTION=1

set ISL_FEEL=1
set ISL_PORTGATE=1
set ISL_SHADOW=1
set ISL_RECALL_LONG=1
set ISL_PAUSES=1
set ISL_SHADOW_HINT=1
set ISL_HINT_USE=1
set ISL_READ_OUT=1
set ISL_HINT_BAND=1
set ISL_SELF_WRITE=1
set ISL_LOOPS=1
set ISL_HEAT=1
set ISL_EVERY=0
set ISL_ELITE_PASS=1
set ISL_PARTIAL=1
set ISL_KEEP_RUNGS=1
set ISL_LIVES=24
set ISL_HOLD=1
set ISL_CHECKPOINT=runs/islands32/checkpoint.txt
set ISL_INHIBIT=1
set ISL_OVERLAP=1
set ISL_STEPS=feel=100,glow=150,cycle1=100
set ISL_FEEL_GOAL=160
set ISL_MIRROR=1
set ISL_SIZE_CLASSES=1
set ISL_WIRE_COST=1
set ISL_COMPETE=1
set ISL_PRUNE=1
set ISL_BRANCH=1
set ISL_ONE_WORD=8
set ISL_BODY=1
set ISL_ELIG=1
set ISL_PAIRS=1
set ISL_MOTIONS=1
set ISL_FRONTIER_LADDERS=1
set ISL_TRIGGERS=1
set ISL_RECALL_FAIR=1
set ISL_PARTIAL_REMEMBER=1
set ISL_RUNG_LOG=runs/islands32/rungs.txt
set ISL_LOG=runs/islands32/plasmids.txt
target\release\examples\islands.exe >> runs\islands32\log.txt 2>&1
