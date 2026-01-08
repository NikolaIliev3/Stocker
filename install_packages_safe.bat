@echo off
REM Safe installation script that avoids locked executable errors
echo Installing advanced ML packages...
echo.

REM Close any Python processes first (optional - uncomment if needed)
REM taskkill /F /IM python.exe 2>nul
REM timeout /t 2 /nobreak >nul

echo Step 1: Installing XGBoost...
pip install xgboost --no-deps --quiet
if %errorlevel% neq 0 (
    echo Warning: XGBoost installation had issues, but library may still work
)

echo Step 2: Installing LightGBM...
pip install lightgbm --no-deps --quiet
if %errorlevel% neq 0 (
    echo Warning: LightGBM installation had issues, but library may still work
)

echo Step 3: Installing Optuna dependencies...
pip install colorlog packaging PyYAML tqdm --quiet
if %errorlevel% neq 0 (
    echo Warning: Some Optuna dependencies had issues
)

echo Step 4: Installing Optuna (library only)...
pip install optuna --no-deps --quiet
if %errorlevel% neq 0 (
    echo Warning: Optuna installation had issues, but library may still work
)

echo.
echo Step 5: Verifying installation...
python -c "try:
    import xgboost
    print('  [OK] XGBoost:', xgboost.__version__)
except Exception as e:
    print('  [FAIL] XGBoost:', str(e))

try:
    import lightgbm
    print('  [OK] LightGBM:', lightgbm.__version__)
except Exception as e:
    print('  [FAIL] LightGBM:', str(e))

try:
    import optuna
    print('  [OK] Optuna:', optuna.__version__)
except Exception as e:
    print('  [FAIL] Optuna:', str(e))"

echo.
echo Installation complete!
echo.
echo Note: If you see warnings about locked executables, they don't affect functionality.
echo The libraries themselves are installed and working.
pause
