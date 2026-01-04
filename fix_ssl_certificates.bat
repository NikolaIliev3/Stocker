@echo off
title Fix SSL Certificates for Stocker
echo ========================================
echo   Fixing SSL Certificate Configuration
echo ========================================
echo.
echo This will reinstall certifi to fix SSL issues with yfinance
echo.

python -m pip install --upgrade --force-reinstall certifi

if errorlevel 1 (
    echo.
    echo ERROR: Failed to reinstall certifi
    pause
    exit /b 1
)

echo.
echo Running SSL fix script...
python fix_yfinance_ssl.py

echo.
echo ========================================
echo SSL fix complete!
echo Restart the backend server and try again.
echo ========================================
echo.
pause


