@echo off
rem Island run 13: run 12 with freak accidents (ISL_ACCIDENT=1): a population grows its trial
rem connection now and then even when it is not stuck, at a rate of its own. 600 generations.
rem Started with Win32_Process.Create so it does not end with the session that launched it.
cd /d C:\Software\source\neurocomp
if not exist runs\islands13 mkdir runs\islands13
echo ---- run 13: run 12's settings (two-mask kernels, marks, exploration, the frontier generalist, the repeat rung, fresh training worlds, 24 of them, a goal held once, population 32, elite 8, energy) with freak accidents (ISL_ACCIDENT=1: each population grows its trial connection by accident, one in 2 to the power k steps, k its own gene, drawn 12-18 and moved by mutation, even when it is not stuck; kept as explored trials are); 600 generations> runs\islands13\log.txt
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
set ISL_LIVES=24
set ISL_HOLD=1
set ISL_CHECKPOINT=runs/islands13/checkpoint.txt
set ISL_LOG=runs/islands13/plasmids.txt
target\release\examples\islands.exe >> runs\islands13\log.txt 2>&1
