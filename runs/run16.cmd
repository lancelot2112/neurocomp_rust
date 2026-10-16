@echo off
rem Island run 16: run 15 with ports for gates, pauses and writes (ISL_PORTGATE=1). 600 generations.
rem Started with Win32_Process.Create so it does not end with the session that launched it.
cd /d C:\Software\source\neurocomp
if not exist runs\islands16 mkdir runs\islands16
echo ---- run 16: run 15's settings (the feel rung, the task-selection island, freak accidents, two-mask kernels, marks, exploration, the frontier generalist, the repeat rung, fresh training worlds, 24 of them, a goal held once, population 32, elite 8, energy) with ports for gates, pauses and writes (ISL_PORTGATE=1: a population's gate and pause may name a sense, the self channel or an output port as well as a channel, so one population can learn only after a stroke; and a population may write a sense, which it senses with the world's input a step later); 600 generations> runs\islands16\log.txt
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
set ISL_LIVES=24
set ISL_HOLD=1
set ISL_CHECKPOINT=runs/islands16/checkpoint.txt
set ISL_LOG=runs/islands16/plasmids.txt
target\release\examples\islands.exe >> runs\islands16\log.txt 2>&1
