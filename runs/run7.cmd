@echo off
rem Island run 7: run 6 with fresh training worlds each generation, the dark torch rung first for screen, and a carrying cost of 4
rem (ISL_FRESH=1, ISL_CARRY=4), 600 generations. Started with
rem Win32_Process.Create so it does not end with the session that launched it.
cd /d C:\Software\source\neurocomp
if not exist runs\islands7 mkdir runs\islands7
echo ---- run 7: run 6's settings (population 32, elite 8, energy with ration 256 and food 32, channels held in place, reads counted, plasmids of at most 6 populations, exclusion, the op choice) with fresh training worlds every generation (ISL_FRESH=1), screen's ladder starting with a dark torch (no cursor mark), and a carrying cost of 4 a token (ISL_CARRY=4, was 16); 600 generations> runs\islands7\log.txt
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
set ISL_CHECKPOINT=runs/islands7/checkpoint.txt
set ISL_LOG=runs/islands7/plasmids.txt
target\release\examples\islands.exe >> runs\islands7\log.txt 2>&1
