@echo off
rem Island run 17: run 16 with shadow explorers (ISL_SHADOW=1). 600 generations.
rem Started with Win32_Process.Create so it does not end with the session that launched it.
cd /d C:\Software\source\neurocomp
if not exist runs\islands17 mkdir runs\islands17
echo ---- run 17: run 16's settings (ports for gates, pauses and writes, the feel rung, the task-selection island, freak accidents, two-mask kernels, marks, exploration, the frontier generalist, fresh training worlds, 24 of them, a goal held once, population 32, elite 8, energy) with shadow explorers (ISL_SHADOW=1: a stuck or unlucky population grows a shadow that predicts, a step ahead, one vector after another from what the population reads, writing nothing; kernels reinforced when their prediction came true, weakened and retrained when it failed; the best-predicted target kept as a predictor into a channel of its own, tagged explored); 600 generations> runs\islands17\log.txt
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
set ISL_LIVES=24
set ISL_HOLD=1
set ISL_CHECKPOINT=runs/islands17/checkpoint.txt
set ISL_LOG=runs/islands17/plasmids.txt
target\release\examples\islands.exe >> runs\islands17\log.txt 2>&1
