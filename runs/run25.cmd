@echo off
rem Island run 25: run 24 (built on run 24: run 22 with goals judged on the elite, partial credit and passed rungs kept) with inhibit inputs (ISL_INHIBIT=1), the overlap rule as a gene (ISL_OVERLAP=1) and steps up to feel (ISL_STEPS=feel=100+200). 600 generations.
rem Started with Win32_Process.Create so it does not end with the session that launched it.
cd /d C:\Software\source\neurocomp
if not exist runs\islands25 mkdir runs\islands25
echo ---- run 25: run 24's settings (run 22's settings [no plasmid exchange, heat inputs, loops closed, self writes, hint bands, output reads, edges on hints toward feedback, shadows as hints, pauses drawn, recall's long ladder, shadow explorers, port gates, the feel rung, task selection, accidents, two-mask kernels, marks, exploration, the frontier generalist, fresh worlds] with goals judged on the elite [ISL_ELITE_PASS=1: a rung's goal is checked on the elite's mean, not the whole population's, before the held-out check] and partial credit [ISL_PARTIAL=1: selection uses graded echo, surprise and feel, a near or late answer counting in part; goals stay on the probes themselves; and passed rungs kept, ISL_KEEP_RUNGS=1: each generation one rung already passed is scored again in its own world and joins the selection cases]) with inhibit inputs (ISL_INHIBIT=1: a population may carry an inhibit input, a channel or port whose lines a frame ago are taken before its kernels fire, so a kernel whose mask lies more than half on them is held down; a new population has one one time in four) and the overlap rule as a gene (ISL_OVERLAP=1: a population's inhibit_quarters, 1 to 4 starting at 2, sets how much of a kernel's mask taken lines must cover to hold it down; 4 lets kernels on the same spot all fire) and steps up to the feel rung (ISL_STEPS=feel=100+200: feel100 and feel200 before feel on switching's, shapes' and the maze's ladders, the same world with goals 100 and 200; no run has passed feel at 300); 600 generations> runs\islands25\log.txt
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
set ISL_CHECKPOINT=runs/islands25/checkpoint.txt
set ISL_INHIBIT=1
set ISL_OVERLAP=1
set ISL_STEPS=feel=100+200
set ISL_RUNG_LOG=runs/islands25/rungs.txt
set ISL_LOG=runs/islands25/plasmids.txt
target\release\examples\islands.exe >> runs\islands25\log.txt 2>&1
