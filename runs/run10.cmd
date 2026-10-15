@echo off
rem Island run 10: run 9 with the generalist scored at the frontier (ISL_FRONTIER=1) and a repeat
rem rung for learning from reward below win-stay and the maze's turn. (Frontier: on each
rem other island's current stage, in its world), 600 generations. Started with
rem Win32_Process.Create so it does not end with the session that launched it.
cd /d C:\Software\source\neurocomp
if not exist runs\islands10 mkdir runs\islands10
echo ---- run 10: run 9's settings (fresh training worlds, 24 of them, a goal held once before the held-out check, the fixed win-stay probe, the dark torch first, a carrying cost of 4, population 32, elite 8, energy) with the generalist scored at the frontier (ISL_FRONTIER=1: on each other island's current stage, its probes in its world, instead of all twelve probes in the plain world, where nine scored about 0 for every genome) and a repeat rung for learning from reward (a maze with the food on one side all lifetime and no cue; the repeat probe) below switching's and shapes' win-stay and maze's turn; 600 generations> runs\islands10\log.txt
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
set ISL_LIVES=24
set ISL_HOLD=1
set ISL_CHECKPOINT=runs/islands10/checkpoint.txt
set ISL_LOG=runs/islands10/plasmids.txt
target\release\examples\islands.exe >> runs\islands10\log.txt 2>&1
