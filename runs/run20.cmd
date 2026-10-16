@echo off
rem Island run 20: run 19 with edges built on hints, biased toward feedback (ISL_HINT_USE=1). 600 generations.
rem Started with Win32_Process.Create so it does not end with the session that launched it.
cd /d C:\Software\source\neurocomp
if not exist runs\islands20 mkdir runs\islands20
echo ---- run 20: run 19's settings (shadows as hints, pauses drawn as gates are, recall's long ladder, shadow explorers, ports for gates, pauses and writes, the feel rung, the task-selection island, freak accidents, two-mask kernels, marks, exploration, the frontier generalist, fresh training worlds, 24 of them, a goal held once, population 32, elite 8, energy) with edges built on hints (ISL_HINT_USE=1: a mutation of its own, one in sixteen while any population carries a hint, adds a new population of any kind on a hinted path, from the hinted population's input to the hinted vector or from that vector to its destination, the way that feeds back, closing a loop or running from a higher level to a lower, three times in four; an output port is read too, as the act a step ago, by any population: ISL_READ_OUT=1, a new population reading from outside reads an output one time in four and a property mutation switches it); 600 generations> runs\islands20\log.txt
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
set ISL_LIVES=24
set ISL_HOLD=1
set ISL_CHECKPOINT=runs/islands20/checkpoint.txt
set ISL_LOG=runs/islands20/plasmids.txt
target\release\examples\islands.exe >> runs\islands20\log.txt 2>&1
