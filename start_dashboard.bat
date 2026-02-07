@echo off
title Stocker - Dashboard ^& Paper Trader
color 0B
cls
echo ========================================
echo   Stocker - Live Trading Dashboard
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo [1/2] Starting Paper Trader (Data Generator)...
start "Stocker Paper Trader" cmd /c "python paper_trader.py"
timeout /t 3 /nobreak >nul

echo [2/2] Starting Dashboard API Server...
echo Cleaning up any old instances...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :5001 ^| findstr LISTENING') do taskkill /f /pid %%a >nul 2>&1

echo.
echo Dashboard will be available at: 
echo - http://localhost:5001
echo - http://127.0.0.1:5001
echo.
echo Opening browser in 5 seconds...
timeout /t 5 /nobreak >nul
start http://127.0.0.1:5001

python dashboard_api.py

pause
