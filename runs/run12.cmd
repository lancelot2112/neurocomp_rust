@echo off
rem Island run 12: run 11 with two-mask kernels (ISL_ZEROS=1): a population may expect zeros as
rem well as ones on its lines, so it can match the absence of input. 600 generations.
rem Started with Win32_Process.Create so it does not end with the session that launched it.
cd /d C:\Software\source\neurocomp
if not exist runs\islands12 mkdir runs\islands12
echo ---- run 12: run 11's settings (marks, exploration, the frontier generalist, the repeat rung, fresh training worlds, 24 of them, a goal held once, population 32, elite 8, energy) with two-mask kernels (ISL_ZEROS=1: a population may expect, so each kernel matches ones and zeros alike on its lines, born half on off lines and half on on lines, expecting what they held; it can sense a click that went unstroked); 600 generations> runs\islands12\log.txt
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
set ISL_LIVES=24
set ISL_HOLD=1
set ISL_CHECKPOINT=runs/islands12/checkpoint.txt
set ISL_LOG=runs/islands12/plasmids.txt
target\release\examples\islands.exe >> runs\islands12\log.txt 2>&1
