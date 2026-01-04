@echo off
echo Creating Stocker Desktop Shortcut...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Install required packages if needed
echo Installing required packages...
pip install Pillow --quiet --user
pip install pywin32 --quiet --user 2>nul || echo Warning: pywin32 installation failed, will use VBScript method instead

REM Create icon
echo Creating icon...
python create_icon.py
if errorlevel 1 (
    echo Warning: Could not create icon, continuing without custom icon...
)

REM Create shortcut
echo Creating desktop shortcut...
python create_shortcut.py
if errorlevel 1 (
    echo.
    echo Attempting alternative method...
    if exist create_shortcut.vbs (
        cscript //nologo create_shortcut.vbs
        if errorlevel 1 (
            echo.
            echo Automatic shortcut creation failed.
            echo.
            echo Manual method:
            echo 1. Right-click on START_HERE.bat
            echo 2. Select "Create shortcut"
            echo 3. Right-click the shortcut, select "Properties"
            echo 4. Click "Change Icon" and browse to stocker_icon.ico
            echo 5. Move shortcut to Desktop and rename to "Stocker"
            pause
            exit /b 1
        )
    ) else (
        echo.
        echo Please run: python create_shortcut.py
        pause
        exit /b 1
    )
)

echo.
echo Shortcut created successfully on your Desktop!
echo You can now double-click "Stocker" on your desktop to start the app.
pause

