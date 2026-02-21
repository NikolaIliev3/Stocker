@echo off
title Stocker - Starting...
cd /d "%~dp0"
echo ========================================
echo   Stocker - Stock Trading App
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    echo.
    pause
    exit /b 1
)

echo [1/3] Checking dependencies...
python -c "import requests, yfinance, pandas, numpy, matplotlib, flask" >nul 2>&1
if errorlevel 1 (
    echo.
    echo Some dependencies are missing.
    echo.
    echo Please run install_dependencies.bat first to install them.
    echo Or run install_dependencies_admin.bat as Administrator if you get errors.
    echo.
    echo Press any key to exit...
    pause >nul
    exit /b 1
) else (
    echo Dependencies OK!
)

echo [2/3] Starting backend server...
start "Stocker Backend Server" /min cmd /c "python backend_proxy.py"
echo Waiting for backend to start...
timeout /t 4 /nobreak >nul

REM Check if backend started successfully
curl http://localhost:5000/api/health >nul 2>&1
if errorlevel 1 (
    echo WARNING: Backend may not have started properly
    echo Continuing anyway...
    timeout /t 2 /nobreak >nul
)

echo [2.5/3] Starting Dashboard ^& Paper Trader...
start "Stocker Dashboard" /min cmd /c "start_dashboard.bat"
timeout /t 2 /nobreak >nul

echo [3/3] Starting desktop application...
echo.
echo ========================================
echo Backend server is running in background
echo Close the app window when done
echo ========================================
echo.

python main.py
if errorlevel 1 (
    echo.
    echo ERROR: Failed to start application
    echo Check the error message above
    echo.
    pause
)

REM Cleanup: Kill backend server when app closes
echo.
echo Shutting down backend server...
taskkill /FI "WindowTitle eq Stocker Backend Server*" /T /F >nul 2>&1
taskkill /FI "WindowTitle eq Stocker Backend*" /T /F >nul 2>&1
echo Done!
timeout /t 2 /nobreak >nul

