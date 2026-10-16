@echo off
rem Waits for island run 17 to finish, then builds and starts run 18 (started with
rem Win32_Process.Create, so it outlives the session that launched it).
cd /d C:\Software\source\neurocomp
:wait
timeout /t 30 /nobreak > nul
findstr /B /C:"plasmids recorded" runs\islands17\log.txt > nul 2>&1
if errorlevel 1 goto wait
timeout /t 10 /nobreak > nul
cargo build --release --examples > runs\build-run18.txt 2>&1
call runs\run18.cmd
