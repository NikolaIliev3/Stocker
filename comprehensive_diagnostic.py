"""
Comprehensive diagnostic to find why model performs worse than random
"""
import json
import numpy as np
from pathlib import Path
import sys

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

APP_DATA_DIR = Path.home() / ".stocker"
METADATA_FILE = APP_DATA_DIR / "ml_metadata_trading.json"

print("=" * 70)
print("COMPREHENSIVE DIAGNOSTIC - Why Model Performs Worse Than Random")
print("=" * 70)

if not METADATA_FILE.exists():
    print("\n[ERROR] No training metadata found")
    sys.exit(1)

with open(METADATA_FILE, 'r') as f:
    metadata = json.load(f)

train_acc = metadata.get('train_accuracy', 0)
test_acc = metadata.get('test_accuracy', 0)
cv_mean = metadata.get('cv_mean', 0)

print(f"\n1. CURRENT RESULTS:")
print(f"   Training: {train_acc*100:.1f}%")
print(f"   Test: {test_acc*100:.1f}%")
print(f"   CV: {cv_mean*100:.1f}%")
print(f"   Random Baseline: 33.3% (for 3 classes)")
print(f"   Status: {'WORSE than random' if test_acc < 0.333 else 'Better than random'}")

print(f"\n2. POSSIBLE ROOT CAUSES:")

issues = []

# Check if model is learning backwards
if test_acc < 0.33:
    issues.append("CRITICAL: Model performing worse than random")
    print("   [CRITICAL] Model is performing WORSE than random guessing!")
    print("   This suggests:")
    print("     - Labels might be inverted (BUY/SELL swapped)")
    print("     - Features might be inverted (negative correlation)")
    print("     - Model is learning the OPPOSITE pattern")

# Check feature importance
feature_importance = metadata.get('feature_importance', {})
if feature_importance:
    max_importance = max(feature_importance.values()) if feature_importance.values() else 0
    if max_importance < 0.05:
        issues.append("Very low feature importance (max < 0.05)")
        print("   [WARNING] Feature importances are very low (max: {:.4f})".format(max_importance))
        print("   This suggests features are not predictive")

# Check class distribution
label_classes = metadata.get('label_classes', [])
if len(label_classes) != 3:
    issues.append(f"Wrong number of classes: {len(label_classes)}")
    print(f"   [WARNING] Expected 3 classes, found {len(label_classes)}")

print(f"\n3. RECOMMENDED FIXES:")

if test_acc < 0.33:
    print("   [FIX 1] Test if labels are inverted:")
    print("     - Check training_pipeline.py labeling logic")
    print("     - Verify: price UP → BUY, price DOWN → SELL")
    print("     - If inverted, model learns backwards pattern")
    
    print("\n   [FIX 2] Simplify model to baseline:")
    print("     - Use single RandomForest (no ensemble)")
    print("     - Disable feature selection")
    print("     - Use default parameters")
    print("     - Goal: Get to at least 33% (random) first")
    
    print("\n   [FIX 3] Check feature extraction:")
    print("     - Verify features aren't all zeros")
    print("     - Check if features are being scaled correctly")
    print("     - Ensure features have variance")

print("\n" + "=" * 70)
print("NEXT: Test with simplified model to isolate the issue")
print("=" * 70)
