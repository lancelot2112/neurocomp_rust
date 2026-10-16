@echo off
rem Island run 18: run 17 with pauses drawn as gates are (ISL_PAUSES=1). 600 generations.
rem Started with Win32_Process.Create so it does not end with the session that launched it.
cd /d C:\Software\source\neurocomp
if not exist runs\islands18 mkdir runs\islands18
echo ---- run 18: run 17's settings (shadow explorers, ports for gates, pauses and writes, the feel rung, the task-selection island, freak accidents, two-mask kernels, marks, exploration, the frontier generalist, fresh training worlds, 24 of them, a goal held once, population 32, elite 8, energy) with recall's long ladder (as run 17) and pauses drawn as gates are (ISL_PAUSES=1: a new population is paused on a channel or port one time in four, and a pause made by mutation draws what it names in the same step); 600 generations> runs\islands18\log.txt
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
set ISL_LIVES=24
set ISL_HOLD=1
set ISL_CHECKPOINT=runs/islands18/checkpoint.txt
set ISL_LOG=runs/islands18/plasmids.txt
target\release\examples\islands.exe >> runs\islands18\log.txt 2>&1
