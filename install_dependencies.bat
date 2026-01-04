@echo off
title Installing Stocker Dependencies
echo ========================================
echo   Installing Stocker Dependencies
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

echo Python found!
python --version
echo.
echo IMPORTANT: Close any running Python applications before continuing.
echo This includes any Flask servers, Python scripts, or IDEs.
echo.
pause

REM Kill any running Python processes that might lock files
echo [Step 0/3] Closing any running Python processes...
taskkill /F /IM python.exe /T >nul 2>&1
taskkill /F /IM pythonw.exe /T >nul 2>&1
timeout /t 2 /nobreak >nul
echo Done.
echo.

REM Upgrade pip first
echo [Step 1/3] Upgrading pip...
python -m pip install --upgrade pip --user
if errorlevel 1 (
    echo WARNING: pip upgrade failed, continuing anyway...
)
echo.

REM Try to uninstall conflicting packages first
echo [Step 2/3] Preparing for installation...
pip uninstall -y flask flask-cors >nul 2>&1
timeout /t 1 /nobreak >nul
echo Done.
echo.

REM Install requirements with retry logic
echo [Step 3/3] Installing packages from requirements.txt...
echo This may take several minutes. Please wait...
echo.

REM Try installing with --user flag to avoid permission issues
pip install --user -r requirements.txt
if errorlevel 1 (
    echo.
    echo First attempt failed. Trying alternative method...
    echo.
    
    REM Try without --user flag
    pip install -r requirements.txt --force-reinstall --no-cache-dir
    if errorlevel 1 (
        echo.
        echo ========================================
        echo ERROR: Installation failed
        echo ========================================
        echo.
        echo Possible solutions:
        echo 1. Close all Python applications and try again
        echo 2. Run this file as Administrator (Right-click ^> Run as Administrator)
        echo 3. Manually run: pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
)

echo.
echo ========================================
echo Installation complete!
echo You can now run START_HERE.bat
echo ========================================
echo.
pause

