@echo off
rem Waits for island run 30 to finish, then builds and starts run 31 (started with
rem Win32_Process.Create, so it outlives the session that launched it).
cd /d C:\Software\source\neurocomp
:wait
timeout /t 30 /nobreak > nul
findstr /B /C:"plasmids recorded" runs\islands30\log.txt > nul 2>&1
if errorlevel 1 goto wait
timeout /t 10 /nobreak > nul
cargo build --release --examples > runs\build-run31.txt 2>&1
call runs\run31.cmd
