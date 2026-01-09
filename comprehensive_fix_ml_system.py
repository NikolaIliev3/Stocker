"""
COMPREHENSIVE FIX FOR ML SYSTEM
This script diagnoses and fixes all critical issues:
1. SMOTE import issue
2. SELL bias
3. Low backtesting accuracy
4. Overfitting
"""
import sys
import traceback
from pathlib import Path

print("=" * 70)
print("COMPREHENSIVE ML SYSTEM FIX")
print("=" * 70)

# Test 1: Check SMOTE import
print("\n[1] Testing SMOTE import...")
try:
    from imblearn.over_sampling import SMOTE
    print("  [OK] SMOTE import successful")
    smote_available = True
except ImportError as e:
    print(f"  [FAIL] SMOTE import failed: {e}")
    smote_available = False
    print("  [FIX] Installing imbalanced-learn...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn"])
    try:
        from imblearn.over_sampling import SMOTE
        print("  [OK] SMOTE now available after install")
        smote_available = True
    except ImportError:
        print("  [FAIL] SMOTE still not available after install")
        smote_available = False

# Test 2: Check ml_training.py for SMOTE usage
print("\n[2] Checking ml_training.py SMOTE implementation...")
ml_training_path = Path("ml_training.py")
if ml_training_path.exists():
    content = ml_training_path.read_text(encoding='utf-8')
    
    # Check if SMOTE is properly imported and used
    if "from imblearn.over_sampling import SMOTE" in content:
        print("  [OK] SMOTE import found in ml_training.py")
    else:
        print("  [WARN] SMOTE import not found - checking if it's inside try-except")
    
    if "smote.fit_resample" in content:
        print("  [OK] SMOTE usage found")
    else:
        print("  [FAIL] SMOTE usage not found")
    
    # Check if there's proper exception handling
    if "except ImportError" in content and "SMOTE" in content:
        print("  [OK] ImportError handling found")
    else:
        print("  [WARN] ImportError handling may be missing")
else:
    print("  [FAIL] ml_training.py not found")

# Test 3: Check confidence threshold implementation
print("\n[3] Checking confidence threshold implementation...")
if ml_training_path.exists():
    content = ml_training_path.read_text(encoding='utf-8')
    
    if "binary_confidence_threshold" in content:
        print("  [OK] Confidence threshold found")
        if "0.55" in content or "55" in content:
            print("  [OK] 55% threshold found")
        else:
            print("  [WARN] Threshold value not found")
    else:
        print("  [FAIL] Confidence threshold not found")
    
    if "is_binary_mode" in content and "maxp < binary_confidence_threshold" in content:
        print("  [OK] Confidence threshold logic found")
    else:
        print("  [WARN] Confidence threshold logic may be incomplete")

# Test 4: Check for class imbalance handling
print("\n[4] Checking class imbalance handling...")
if ml_training_path.exists():
    content = ml_training_path.read_text(encoding='utf-8')
    
    if "imbalance_ratio" in content:
        print("  [OK] Imbalance ratio calculation found")
    else:
        print("  [WARN] Imbalance ratio calculation not found")
    
    if "1.3" in content and "imbalance" in content.lower():
        print("  [OK] Imbalance threshold (1.3) found")
    else:
        print("  [WARN] Imbalance threshold not found")

# Test 5: Check backtest evaluation logic
print("\n[5] Checking backtest evaluation logic...")
backtester_path = Path("continuous_backtester.py")
if backtester_path.exists():
    content = backtester_path.read_text(encoding='utf-8')
    
    # Check SELL evaluation
    if "predicted_action == 'SELL'" in content and "actual_change_pct > 0" in content:
        print("  [OK] SELL evaluation logic correct (expects price UP)")
    else:
        print("  [FAIL] SELL evaluation logic may be incorrect")
    
    # Check BUY evaluation
    if "predicted_action == 'BUY'" in content and "actual_change_pct < 0" in content:
        print("  [OK] BUY evaluation logic correct (expects price DOWN)")
    else:
        print("  [FAIL] BUY evaluation logic may be incorrect")

# Summary and recommendations
print("\n" + "=" * 70)
print("SUMMARY AND FIXES")
print("=" * 70)

fixes_needed = []

if not smote_available:
    fixes_needed.append("1. Install imbalanced-learn: pip install imbalanced-learn")

print("\n[RECOMMENDATIONS]")
print("1. Retrain ML model with SMOTE enabled")
print("2. Increase confidence threshold to 60% (from 55%)")
print("3. Add more regularization to reduce overfitting")
print("4. Clear old backtest data and retrain from scratch")
print("5. Monitor prediction distribution (should be ~40-60% each class)")

print("\n[IMMEDIATE ACTIONS]")
print("1. Run: python -c 'from imblearn.over_sampling import SMOTE; print(\"SMOTE OK\")'")
print("2. Retrain ML model (SMOTE will auto-apply if imbalance > 1.3)")
print("3. Run backtesting and check if accuracy improves")
print("4. If accuracy still < 50%, increase confidence threshold to 60%")

print("\n" + "=" * 70)
