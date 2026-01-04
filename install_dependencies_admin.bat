@echo off
REM Check for admin rights
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo This script requires Administrator privileges.
    echo Right-click this file and select "Run as Administrator"
    echo.
    pause
    exit /b 1
)

title Installing Stocker Dependencies (Admin)
echo ========================================
echo   Installing Stocker Dependencies
echo   (Running as Administrator)
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

REM Kill any running Python processes
echo Closing any running Python processes...
taskkill /F /IM python.exe /T >nul 2>&1
taskkill /F /IM pythonw.exe /T >nul 2>&1
timeout /t 2 /nobreak >nul
echo.

REM Upgrade pip
echo [1/3] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Uninstall conflicting packages
echo [2/3] Preparing for installation...
pip uninstall -y flask flask-cors >nul 2>&1
timeout /t 1 /nobreak >nul
echo.

REM Install requirements
echo [3/3] Installing packages...
echo This may take several minutes...
echo.
pip install -r requirements.txt --force-reinstall --no-cache-dir

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed even with admin rights
    echo Please check the error messages above
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Installation complete!
echo You can now run START_HERE.bat
echo ========================================
echo.
pause


