@echo off
title Stocker - HARD RESTART
color 0C
echo ==================================================
echo      KILLING ALL STOCKER PROCESSES (FORCE)
echo ==================================================
echo.
echo Terminating Python...
taskkill /F /IM python.exe >nul 2>&1
echo.
echo Waiting for cleanup...
timeout /t 3 /nobreak >nul
echo.
echo ==================================================
echo      STARTING FRESH (DESKTOP APP)
echo ==================================================
timeout /t 1 /nobreak >nul
call start_stocker.bat
exit
