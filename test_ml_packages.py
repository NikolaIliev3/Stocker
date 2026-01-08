#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test script to verify ML packages are installed and working"""

import sys

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

print("Testing ML Package Installation...")
print("=" * 50)

results = {}

# Test XGBoost
try:
    import xgboost
    results['XGBoost'] = f"[OK] Version {xgboost.__version__}"
except Exception as e:
    results['XGBoost'] = f"[ERROR] {str(e)}"

# Test LightGBM
try:
    import lightgbm
    results['LightGBM'] = f"[OK] Version {lightgbm.__version__}"
except Exception as e:
    results['LightGBM'] = f"[ERROR] {str(e)}"

# Test Optuna
try:
    import optuna
    results['Optuna'] = f"[OK] Version {optuna.__version__}"
except Exception as e:
    results['Optuna'] = f"[ERROR] {str(e)}"

# Print results
for package, status in results.items():
    print(f"{package:15} {status}")

print("=" * 50)

# Summary
all_ok = all("[OK]" in status for status in results.values())
if all_ok:
    print("\nSUCCESS! All packages are installed and working!")
    print("\nYou can now use all advanced accuracy improvements:")
    print("  - XGBoost/LightGBM models")
    print("  - Stacking ensemble")
    print("  - Time-series features")
    print("  - Pattern detection")
    print("  - Feature interactions")
    print("  - Hyperparameter tuning")
    print("\nReady to retrain models!")
else:
    print("\nWARNING: Some packages have issues. Check errors above.")
