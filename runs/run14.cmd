@echo off
rem Island run 14: run 13 with the task-selection island (ISL_SELECTION=1). 600 generations.
rem Started with Win32_Process.Create so it does not end with the session that launched it.
cd /d C:\Software\source\neurocomp
if not exist runs\islands14 mkdir runs\islands14
echo ---- run 14: run 13's settings (freak accidents, two-mask kernels, marks, exploration, the frontier generalist, the repeat rung, fresh training worlds, 24 of them, a goal held once, population 32, elite 8, energy) with a ninth island, task selection (ISL_SELECTION=1): one lifetime of blocks alternating responding and babbling, scored by select; its ladder cued, uncued, short blocks; 600 generations> runs\islands14\log.txt
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
set ISL_LIVES=24
set ISL_HOLD=1
set ISL_CHECKPOINT=runs/islands14/checkpoint.txt
set ISL_LOG=runs/islands14/plasmids.txt
target\release\examples\islands.exe >> runs\islands14\log.txt 2>&1
