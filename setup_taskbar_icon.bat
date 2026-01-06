@echo off
title Stocker - Taskbar Icon Setup
echo ========================================
echo   Stocker - Taskbar Icon Setup
echo ========================================
echo.
echo This will create a shortcut on your Desktop
echo that you can pin to your taskbar with the icon.
echo.

python pin_to_taskbar.py

if errorlevel 1 (
    echo.
    echo ERROR: Setup failed!
    echo Make sure Python is installed and dependencies are available.
    echo.
    pause
) else (
    echo.
    echo ========================================
    echo Setup complete!
    echo ========================================
    echo.
    echo Next steps:
    echo 1. Find "Stocker.lnk" on your Desktop
    echo 2. Right-click it and select "Pin to taskbar"
    echo 3. The icon should now appear on your taskbar!
    echo.
)

pause

