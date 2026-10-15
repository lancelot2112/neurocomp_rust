@echo off
rem Island run 6: run 5's settings with a population of 32, an elite of 8, and energy
rem (a ration of 256 a step, 32 rations of food for each 1024 points scored). Started with
rem Win32_Process.Create so it does not end with the session that launched it.
cd /d C:\Software\source\neurocomp
if not exist runs\islands6 mkdir runs\islands6
echo ---- run 6: run 5's settings (fresh from naked, channels held in place, reads counted, plasmids of at most 6 populations, exclusion, a carrying cost of 16 a token, the op choice) with a population of 32 and an elite of 8 (ISL_POP=32, ISL_ELITE=8) and energy: a ration of 256 a step, and it eats what it scores, 32 rations for each 1024 points (ISL_RATION=256, ISL_FOOD=32); 300 generations> runs\islands6\log.txt
set ISL_RATION=256
set ISL_FOOD=32
set ISL_POP=32
set ISL_ELITE=8
set ISL_OPS=1
set ISL_PLASMID_MAX=6
set ISL_EXCLUDE=1
set ISL_CARRY=16
set ISL_INPLACE=1
set ISL_READS=1
set ISL_SCREEN=1
set ISL_MAZE=1
set ISL_PREDICTION=1
set ISL_SHAPES=1
set ISL_LADDERS=1
set ISL_CKPT=5
set ISL_HELD=48
set ISL_GENS=300
set ISL_CHECKPOINT=runs/islands6/checkpoint.txt
set ISL_LOG=runs/islands6/plasmids.txt
target\release\examples\islands.exe >> runs\islands6\log.txt 2>&1
