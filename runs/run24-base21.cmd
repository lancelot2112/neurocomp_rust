@echo off
rem (Fallback if run 23 does not show promise: based on run 21.) Island run 24: run 21 with goals judged on the elite (ISL_ELITE_PASS=1) and partial credit in selection (ISL_PARTIAL=1). 600 generations.
rem Started with Win32_Process.Create so it does not end with the session that launched it.
cd /d C:\Software\source\neurocomp
if not exist runs\islands24 mkdir runs\islands24
echo ---- run 24: run 22's settings (no plasmid exchange, heat inputs, loops closed, self writes, hint bands, output reads, edges on hints toward feedback, shadows as hints, pauses drawn, recall's long ladder, shadow explorers, port gates, the feel rung, task selection, accidents, two-mask kernels, marks, exploration, the frontier generalist, fresh worlds) with goals judged on the elite (ISL_ELITE_PASS=1: a rung's goal is checked on the elite's mean, not the whole population's, before the held-out check) and partial credit (ISL_PARTIAL=1: selection uses graded echo, surprise and feel, a near or late answer counting in part; goals stay on the probes themselves); 600 generations> runs\islands24\log.txt
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
set ISL_LIVES=24
set ISL_HOLD=1
set ISL_CHECKPOINT=runs/islands24/checkpoint.txt
set ISL_LOG=runs/islands24/plasmids.txt
target\release\examples\islands.exe >> runs\islands24\log.txt 2>&1
