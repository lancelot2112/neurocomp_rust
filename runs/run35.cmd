@echo off
rem Island run 35: run 34 with more branches (ISL_BRANCHES_MANY=1), long shifts (ISL_LONG_SHIFTS=1), the cycle3-100 step (ISL_STEPS ...,cycle3=100) and shaped surprise (ISL_PARTIAL_SURPRISE=1). Run 34: run 33 with the voice one-word everywhere (ISL_ONE_WORD_VOICE=1), selection's step rung (ISL_SELECT_STEP=1) the generalist's passes migrating (ISL_MIGRATE=1) and the islands' best joining the generalist (ISL_MIGRATE_UP=1, ISL_MIGRATE_UP_N=4, ISL_GENERALIST_ROOM=64) and step 2 of the learning plan: twin worlds (ISL_TWINS=1), the delay and the period drawn per lifetime (ISL_VARY_LAG=1, ISL_VARY_PERIOD=1) mixed lights as one-word codes (ISL_MIXED_CODES=1), and a novelty island with species (ISL_NOVELTY=1, ISL_NOVELTY_MAX=128). Run 33: run 32 with win-stay against the other light (ISL_FAIR_WINSTAY=1) the clock drawn (ISL_CLOCK_DRAWN=1) order cycles of varied shape (ISL_VARIED_ORDERS=1), held-out worlds fresh at every check (ISL_FRESH_HELD=1), the maze cue mirrored per lifetime (ISL_MAZE_MIRROR=1), the shapes' lights in one word (ISL_SAME_WORD=1) food from the world's strokes (ISL_FOOD_WORLD=1): the rest of the rung audit; and partial credit on select (ISL_PARTIAL_SELECT=1). Run 32: run 31 (the three-factor output rule, the pairs rungs) with motions (ISL_MOTIONS=1: conveyor channels, delay lines on several time scales, and rule-90 waves) the generalist climbing every ladder itself (ISL_FRONTIER_LADDERS=1) and triggers with integer cases (ISL_TRIGGERS=1), and recall's fair last rungs (ISL_RECALL_FAIR=1) with partial credit on remember (ISL_PARTIAL_REMEMBER=1), fair prediction streams (ISL_FAIR_SEQUENCES=1), echo and copy per token shown (ISL_FAIR_SAID=1), and feel's goals over what random moves reach (feel-300, feel 500). Open-ended: it runs until stopped by hand.
rem Started with Win32_Process.Create so it does not end with the session that launched it.
cd /d C:\Software\source\neurocomp
if not exist runs\islands35 mkdir runs\islands35
echo ---- run 35: run 34's settings (run 33's settings (run 32's settings (run 31's settings (run 30's settings (run 29's settings (run 28's settings (run 26's settings [run 25's settings [run 24's settings [run 22's settings [no plasmid exchange, heat inputs, loops closed, self writes, hint bands, output reads, edges on hints toward feedback, shadows as hints, pauses drawn, recall's long ladder, shadow explorers, port gates, the feel rung, task selection, accidents, two-mask kernels, marks, exploration, the frontier generalist, fresh worlds] with goals judged on the elite [ISL_ELITE_PASS=1: a rung's goal is checked on the elite's mean, not the whole population's, before the held-out check] and partial credit [ISL_PARTIAL=1: selection uses graded echo, surprise and feel, a near or late answer counting in part; goals stay on the probes themselves; and passed rungs kept, ISL_KEEP_RUNGS=1: each generation one rung already passed is scored again in its own world and joins the selection cases]] with inhibit inputs [ISL_INHIBIT=1: a population may carry an inhibit input, a channel or port whose lines a frame ago are taken before its kernels fire, so a kernel whose mask lies more than half on them is held down; a new population has one one time in four] and the overlap rule as a gene [ISL_OVERLAP=1: a population's inhibit_quarters, 1 to 4 starting at 2, sets how much of a kernel's mask taken lines must cover to hold it down; 4 lets kernels on the same spot all fire] and steps up to the rungs no run has passed [ISL_STEPS=feel=100,glow=150,cycle1=100 and ISL_FEEL_GOAL=160: feel-100 before feel, whose goal is lowered from 300, out of reach, to 160, on switching's, shapes' and the maze's ladders, glow-150 before screen's glow, cycle1-100 before prediction's cycle1, the same worlds with lower goals] and easier reading rungs [ISL_MIRROR=1: echo2q, echo2m8 and echo2m4 before echo2, each token heard like the voice's code for it, first without one's own voice]] with size classes [ISL_SIZE_CLASSES=1: each island's genomes sorted by size into classes a factor of four apart, 1-3 tokens, 4-15, 16-63, 64-255 and 256+; each occupied class keeps its own lexicase choice among the elite, and half of all children have a parent chosen within a class drawn at random, so small and large networks both stay in the population]] with a wiring cost [ISL_WIRE_COST=1: each step a genome pays 1 for each bit of Hamming distance between the tags of the two ends of each of its populations, ports at fixed tags in the same space, as the carrying cost is paid; near ends cheap, far ones dear, so circuits that work together gather in one part of tag space near the ports they use], populations competing for their destination [ISL_COMPETE=1: the populations into a destination run strongest grip first, ties in genome order, and each word an earlier population's kernels wrote this step is claimed, and a kernel whose output words are mostly claimed is held down and learns anti-Hebbian, so the winners write the act, not the OR of every reflex] and pruning [ISL_PRUNE=1: each population holds its destination with its gene grip, 16 to start, up one a step it writes, down one a step it only loses, letting go for the lifetime at 0]) with branches (ISL_BRANCH=1: a population's fired kernels may also write their outputs into a second vector, a channel or an output port, at the same place by OR, as an axon's collaterals: an efference copy, a line to another population's inhibit input, two targets driven by one decision; a new population has one one time in four; under competition a branch writes only unclaimed words)) with sparse one-word codes on reading (ISL_ONE_WORD=8: every reading lifetime's glyphs, sounds and voice codes drawn as 8 random lines in the first word of their channels, so a glyph, its sound and its voice code sit at the same place and a word holding three codes stays readable) and a fixed body (ISL_BODY=1: the voice's and the hand's codes, the sound the voice makes for each token, which the teacher speaks too, and the feel of each move drawn from one seed for the whole run, as anatomy is, so evolution can build them in; glyphs, lights, tones, cues and what pays are still drawn fresh each lifetime, so learning is forced by the world)) with the three-factor output rule (ISL_ELIG=1: a kernel that fired k steps ago is checked while its population's gate, the modulator, is open, against what its destination holds then, less its own population's writes; right, it counts up; wrong, an output line moves toward what is there and it counts down, and wrong at 0 it dies, so its input is free for a new birth; a new population has it one time in four) and reading's paired-associate rungs (ISL_PAIRS=1: pairs8 and pairs32 after echo2q, glyph-to-token pairs drawn each lifetime, each token heard like its voice code, goal copy 100)) with motions (ISL_MOTIONS=1: a channel may move by itself each step before any population writes, a conveyor whose words move one place along every 2^(k-1) steps, so word j holds what was written j moves ago, delay lines on time scales a factor of two apart, or a travelling wave, each line the XOR of its neighbours, cellular rule 90; a new token is a motion one time in about twenty) and the generalist climbing every ladder itself (ISL_FRONTIER_LADDERS=1: a rung of its own on each island's ladder, scored in that island's world, with one rung below it drawn at random so what it passed stays in its selection; it moves up each ladder as that island does, the goal on its elite's mean for the rung's hold, then the held-out check) and triggers (ISL_TRIGGERS=1: a population's kernels match at least their threshold, at most it, in a band, or over the whole tile; or as integers, the lane of 8 to 64 bits holding a kernel's first mask line read as a number: at least, at most or near its threshold's share of the lane's range, near a number of its own that it learns, or the first of two lanes greater or less than the last); and recall's fair last rungs (ISL_RECALL_FAIR=1: two, then distract with two lights and the flash the light the trial did not show first, then vary with two lights, a tone as the window opens and a click before it spending the trial; the audit found the old distract and vary passed by clicking a fixed time after any light) with partial credit on remember (ISL_PARTIAL_REMEMBER=1: in selection, a click in the two steps after the window counts half; the goal stays on remember itself), fair prediction streams (ISL_FAIR_SEQUENCES=1: a cycle uses each of its tokens equally often, period 6 of 3 as A A B B C C, and a break is one of the cycle's tokens that may follow the token before it, never two in a row, so a rare-token detector or a model of the token before scores about 0), echo and copy per token shown (ISL_FAIR_SAID=1: divided by the tokens said or the tokens shown that count, whichever is more, so one token said by luck cannot pass) and feel's goals raised over what random moves after a stroke reach (204, best 273; the fixed body makes the click wireable: feel-300 then feel 500)) with win-stay against the other light (ISL_FAIR_WINSTAY=1: the stroked-minus-unstroked rate of clicking a light's next trial, less the same on the other light's next trial, halved, so a clicker driven by any recent stroke scores below 0: -291 to -308, the old probe 314 to 319 against a goal of 80) and the clock drawn (ISL_CLOCK_DRAWN=1: the self channel's step count starts each lifetime at a number drawn from the lifetime's seed, so it cannot time a reversal or a block's end) and order cycles of varied shape (ISL_VARIED_ORDERS=1: order2's cycle says each token twice in a random order, all pairs different and no token twice in a row, and a break keeps the pairs with the tokens before and after it, so a fixed rule on the last frames, which passed run 32's order2 at 766, scores about 0), held-out worlds fresh at every check (ISL_FRESH_HELD=1: new worlds for each check of a goal, so repeated checks on one set cannot let a lucky genome through), the maze cue mirrored per lifetime (ISL_MAZE_MIRROR=1: in half the lifetimes the cue's half of the eye is the side that does not pay, so a wired turn scores about 0 and the side is learned from strokes), the shapes' lights in one word (ISL_SAME_WORD=1: with blur or shift the two lights share a word, so which word is lit no longer tells them apart) and food from the world (ISL_FOOD_WORLD=1: energy from each stroke the world gives, not from the observer's scores) and partial credit on select (ISL_PARTIAL_SELECT=1: in selection the voice and click costs at a quarter, so a click reflex not yet gated by the task, 0 on select, scores 769 against babbling's 512 and selection can climb past cued, where every run's elite babbled and never clicked)) with the voice one-word everywhere (ISL_ONE_WORD_VOICE=1: the voice's codes and the sounds they make the same 8-line one-word codes in every lifetime, not only those with tokens, so the voice is one body in every task; the hand's codes unchanged), selection's step rung (ISL_SELECT_STEP=1: mixclick before cued, the mixed lifetime scored on responding in its conditioning blocks alone, goal 100, so clicking at the lights pays before it must be gated by the task; no run's selection island ever held cued's goal on training) and the generalist's passes migrating (ISL_MIGRATE=1: when the generalist passes a rung on an island's ladder the island has not passed, its best two genomes are copied into that island's next population; run 32's generalist passed cycle1-100 with one population at generation 7 and the prediction island never did) and the islands' best migrating into the generalist (ISL_MIGRATE_UP=1, ISL_MIGRATE_UP_N=4, ISL_GENERALIST_ROOM=64: every generation each island's best four join the generalist beside its own 32 children, not in their place, up to 64 in all, and are scored with them; breeding brings it back to 32, so a migrant that wins a place among its parents or its elite carries the specialists' circuits on, as plasmids once did) and step 2 of the learning plan, what stayed the same from lifetime to lifetime drawn per lifetime (twin worlds, ISL_TWINS=1: lifetimes in pairs living one world, the second with its lights' roles swapped and the maze cue mirrored, so a policy fixed to a light or side nets about 0 over a pair; the delay, ISL_VARY_LAG=1, and the period, ISL_VARY_PERIOD=1, drawn from 1 to the rung's per lifetime, so a fixed delay line or comparator fits only some lifetimes; mixed lights as one-word codes, ISL_MIXED_CODES=1, so the size or place of a code no longer tells selection's two tasks apart) and a novelty island (ISL_NOVELTY=1: selected on how new its networks' behaviour is, click and voice rates in each kind of lifetime, clicks after strokes and odd tokens, other moves, tokens said, against its population and an archive, with no goal; a novel genome useful on some probe founds a species, whose founder is kept while the island grows by four genomes, up to ISL_NOVELTY_MAX=128; its most novel join the generalist through ISL_MIGRATE_UP)) with more branches (ISL_BRANCHES_MANY=1: a population may carry up to four more branches besides its one, each with its own target, a channel or an output port, its own write op and rotation, and its own trigger, a gate on a channel or a port, so one firing reaches several places, each written its own way and in its own context), long shifts (ISL_LONG_SHIFTS=1: write ops lsl:k and lsr:k that shift what is written by k bits across word boundaries, the bits leaving a word entering the next, so a pattern can travel the length of a vector; a new op is one one time in seven), and selection's longer ladder (ISL_SELECT_MORE=1: short given a goal, then held, the task cue felt only for a block's first 3 steps and the tasks drawn at random so the task must be held through the block, then three, a third task, still, neither click nor voice; a holder of the last cue scores 1022, a network acting only while the cue is felt 699, a two-task selector 680), and the cursor off the target (ISL_CURSOR_AWAY=1: on every screen rung but glow the cursor never starts on the target, so a hit takes a move; run 33's torch passer, which never moved and clicked when the glyph showed under the cursor, scores 322 without it and -341 with it), with partial credit on the screen (ISL_PARTIAL_SCREEN=1: in selection, a trial in which the cursor reached the target without a hit counts half a hit, so search pays before the click on what it finds: a sitter 0, a wanderer 512, a searcher 1024), and a step before prediction's cycle3 (ISL_STEPS cycle3=100: cycle3-100, the same world with goal 100; run 33's population scored about 0 on cycle3 for 600 generations, every genome alike, and a lag-3 detector right a third of the time scores 273, so a step needs one right about 1 time in 7) and surprise shaped from clicking at all (ISL_PARTIAL_SURPRISE=1: in selection prediction is scored by surprise+, a quarter for clicking after breaks at all and three quarters for clicking after breaks more than after other steps, so never clicking scores 0, clicking always 256, a third of the breaks 321, every break about 740; the goal stays on surprise); open-ended (1,000,000 generations: it runs until stopped by hand)> runs\islands35\log.txt
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
set ISL_GENS=1000000
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
set ISL_CHECKPOINT=runs/islands35/checkpoint.txt
set ISL_INHIBIT=1
set ISL_OVERLAP=1
set ISL_STEPS=feel=300,glow=150,cycle1=100,cycle3=100
set ISL_PARTIAL_SURPRISE=1
set ISL_FEEL_GOAL=500
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
set ISL_FAIR_SEQUENCES=1
set ISL_FAIR_SAID=1
set ISL_FAIR_WINSTAY=1
set ISL_CLOCK_DRAWN=1
set ISL_VARIED_ORDERS=1
set ISL_FRESH_HELD=1
set ISL_MAZE_MIRROR=1
set ISL_SAME_WORD=1
set ISL_FOOD_WORLD=1
set ISL_PARTIAL_SELECT=1
set ISL_ONE_WORD_VOICE=1
set ISL_SELECT_STEP=1
set ISL_SELECT_MORE=1
set ISL_CURSOR_AWAY=1
set ISL_PARTIAL_SCREEN=1
set ISL_MIGRATE=1
set ISL_MIGRATE_UP=1
set ISL_MIGRATE_UP_N=4
set ISL_GENERALIST_ROOM=64
set ISL_TWINS=1
set ISL_VARY_LAG=1
set ISL_VARY_PERIOD=1
set ISL_MIXED_CODES=1
set ISL_NOVELTY=1
set ISL_NOVELTY_MAX=128
set ISL_BRANCHES_MANY=1
set ISL_LONG_SHIFTS=1
set ISL_RUNG_LOG=runs/islands35/rungs.txt
set ISL_LOG=runs/islands35/plasmids.txt
target\release\examples\islands.exe >> runs\islands35\log.txt 2>&1
