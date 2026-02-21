@echo off
title Stocker - Quick Start
cd /d "%~dp0"
color 0A
cls
echo.
echo   ╔═══════════════════════════════════════╗
echo   ║   STOCKER - Stock Trading App        ║
echo   ║   Double-click this file to start!    ║
echo   ╚═══════════════════════════════════════╝
echo.
echo   Starting application...
echo   (If you get dependency errors, run install_dependencies.bat first)
echo.
timeout /t 2 /nobreak >nul
call start_stocker.bat

