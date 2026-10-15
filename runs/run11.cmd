@echo off
rem Island run 11: run 10 with epigenetic marks (ISL_EPI=1) and exploration (ISL_EXPLORE=1):
rem trial connections grown in a lifetime, kept in the genome tagged explored. 600 generations.
rem Started with Win32_Process.Create so it does not end with the session that launched it.
cd /d C:\Software\source\neurocomp
if not exist runs\islands11 mkdir runs\islands11
echo ---- run 11: run 10's settings (the frontier generalist, the repeat rung, fresh training worlds, 24 of them, a goal held once, the fixed win-stay probe, the dark torch, carrying cost 4, population 32, elite 8, energy) with epigenetic marks (ISL_EPI=1: a population's predictions that came true mark it, and marked populations are duplicated more and deleted less) and exploration (ISL_EXPLORE=1: a stuck population grows a trial XOR connection back into its source, kept in the genome tagged explored if it came true 16 times); 600 generations> runs\islands11\log.txt
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
set ISL_LIVES=24
set ISL_HOLD=1
set ISL_CHECKPOINT=runs/islands11/checkpoint.txt
set ISL_LOG=runs/islands11/plasmids.txt
target\release\examples\islands.exe >> runs\islands11\log.txt 2>&1
