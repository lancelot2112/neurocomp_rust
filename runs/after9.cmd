@echo off
rem Waits for island run 9 to finish, then starts run 10 (started with Win32_Process.Create, so
rem it outlives the session that launched it).
cd /d C:\Software\source\neurocomp
:wait
timeout /t 30 /nobreak > nul
tasklist /FI "IMAGENAME eq islands.exe" | find /I "islands.exe" > nul
if not errorlevel 1 goto wait
findstr /B /C:"plasmids recorded" runs\islands9\log.txt > nul
if errorlevel 1 exit /b 1
cargo build --release --examples > runs\build-run10.txt 2>&1
call runs\run10.cmd
