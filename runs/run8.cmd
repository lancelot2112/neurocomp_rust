@echo off
rem Island run 8: run 7 with 24 training worlds (ISL_LIVES=24) and a goal held 1 generation
rem before the held-out check (ISL_HOLD=1), 600 generations. Started with
rem Win32_Process.Create so it does not end with the session that launched it.
cd /d C:\Software\source\neurocomp
if not exist runs\islands8 mkdir runs\islands8
echo ---- run 8: run 7's settings (fresh training worlds, the dark torch first, a carrying cost of 4, population 32, elite 8, energy) with 24 training worlds a generation (ISL_LIVES=24, was 12) and a goal held 1 generation on them before the held-out check (ISL_HOLD=1, was 3): with fresh worlds the population mean swung too much to hold a goal 3 generations in a row; 600 generations> runs\islands8\log.txt
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
set ISL_LIVES=24
set ISL_HOLD=1
set ISL_CHECKPOINT=runs/islands8/checkpoint.txt
set ISL_LOG=runs/islands8/plasmids.txt
target\release\examples\islands.exe >> runs\islands8\log.txt 2>&1
