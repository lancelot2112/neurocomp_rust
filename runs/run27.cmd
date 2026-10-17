@echo off
rem (Fallback if run 23 does not show promise: based on run 22.) Island run 27: run 26 (run 22 with inhibit inputs, the overlap gene and size classes) with thrift relaxed while stuck (ISL_THRIFT_RELAX=64). 600 generations.
rem Started with Win32_Process.Create so it does not end with the session that launched it.
cd /d C:\Software\source\neurocomp
if not exist runs\islands27 mkdir runs\islands27
echo ---- run 27: run 26's settings (run 25's settings [run 22's settings [no plasmid exchange, heat inputs, loops closed, self writes, hint bands, output reads, edges on hints toward feedback, shadows as hints, pauses drawn, recall's long ladder, shadow explorers, port gates, the feel rung, task selection, accidents, two-mask kernels, marks, exploration, the frontier generalist, fresh worlds] with inhibit inputs [ISL_INHIBIT=1: a population may carry an inhibit input, a channel or port whose lines a frame ago are taken before its kernels fire, so a kernel whose mask lies more than half on them is held down; a new population has one one time in four] and the overlap rule as a gene [ISL_OVERLAP=1: a population's inhibit_quarters, 1 to 4 starting at 2, sets how much of a kernel's mask taken lines must cover to hold it down; 4 lets kernels on the same spot all fire]] with size classes [ISL_SIZE_CLASSES=1: each island's genomes sorted by size into classes a factor of four apart, 1-3 tokens, 4-15, 16-63, 64-255 and 256+; each occupied class keeps its own lexicase choice among the elite, and half of all children have a parent chosen within a class drawn at random, so small and large networks both stay in the population]) with thrift relaxed while stuck (ISL_THRIFT_RELAX=64: each selection takes the thrift cases with chance 64/(64+g), g the generations the island has been on its rung; full on a fresh rung, half after 64 generations stuck, a fifth after 256, full again after each pass); 600 generations> runs\islands27\log.txt
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
set ISL_INHIBIT=1
set ISL_OVERLAP=1
set ISL_LIVES=24
set ISL_HOLD=1
set ISL_CHECKPOINT=runs/islands27/checkpoint.txt
set ISL_SIZE_CLASSES=1
set ISL_THRIFT_RELAX=64
set ISL_RUNG_LOG=runs/islands27/rungs.txt
set ISL_LOG=runs/islands27/plasmids.txt
target\release\examples\islands.exe >> runs\islands27\log.txt 2>&1
