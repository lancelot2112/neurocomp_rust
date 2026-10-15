@echo off
rem Island run 9: run 8 with the fixed win-stay probe (a lifetime with nothing to compare
rem scores 0), 600 generations. Started with
rem Win32_Process.Create so it does not end with the session that launched it.
cd /d C:\Software\source\neurocomp
if not exist runs\islands9 mkdir runs\islands9
echo ---- run 9: run 8's settings (fresh training worlds, 24 of them, a goal held once before the held-out check, the dark torch first, a carrying cost of 4, population 32, elite 8, energy) with the win-stay probe fixed: a lifetime with no stroked click to compare against an unstroked one scores 0 (a stubborn clicker scored +-1024 by which light it kept to); 600 generations> runs\islands9\log.txt
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
set ISL_CHECKPOINT=runs/islands9/checkpoint.txt
set ISL_LOG=runs/islands9/plasmids.txt
target\release\examples\islands.exe >> runs\islands9\log.txt 2>&1
