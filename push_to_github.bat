@echo off
echo Pushing Stocker to GitHub...
echo.

REM Check if remote exists
git remote get-url origin >nul 2>&1
if errorlevel 1 (
    echo Remote 'origin' not found.
    echo.
    echo Please provide your GitHub username:
    set /p GITHUB_USER="GitHub Username: "
    
    if "%GITHUB_USER%"=="" (
        echo Error: GitHub username is required
        pause
        exit /b 1
    )
    
    echo.
    echo Adding remote repository...
    git remote add origin https://github.com/%GITHUB_USER%/stocker.git
    if errorlevel 1 (
        echo Error: Could not add remote. Please check:
        echo 1. Repository 'stocker' exists on your GitHub account
        echo 2. You have access to push to it
        pause
        exit /b 1
    )
)

echo.
echo Current branch: 
git branch --show-current

echo.
echo Pushing to GitHub...
git push -u origin master
if errorlevel 1 (
    echo.
    echo Push failed. Trying 'main' branch instead...
    git branch -M main
    git push -u origin main
    if errorlevel 1 (
        echo.
        echo Error: Could not push to GitHub.
        echo.
        echo Possible issues:
        echo 1. Repository doesn't exist on GitHub - create it first at github.com
        echo 2. Authentication required - you may need to:
        echo    - Use GitHub CLI: gh auth login
        echo    - Or use SSH: git remote set-url origin git@github.com:USERNAME/stocker.git
        echo    - Or use Personal Access Token
        echo.
        pause
        exit /b 1
    )
)

echo.
echo Successfully pushed to GitHub!
echo Repository: https://github.com/%GITHUB_USER%/stocker
pause

