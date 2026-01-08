"""
Diagnostic script to check what's causing low accuracy
"""
import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

from config import APP_DATA_DIR
import json

print("=" * 70)
print("DIAGNOSING LOW ACCURACY ISSUE")
print("=" * 70)

data_dir = Path(APP_DATA_DIR)

# Check metadata file
metadata_file = data_dir / "ml_metadata_trading.json"
if metadata_file.exists():
    print("\n[CHECKING] ML Metadata...")
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        print(f"  Strategy: {metadata.get('strategy', 'N/A')}")
        print(f"  Training Samples: {metadata.get('samples', 'N/A')}")
        print(f"  Train Accuracy: {metadata.get('train_accuracy', 0)*100:.1f}%")
        print(f"  Test Accuracy: {metadata.get('test_accuracy', 0)*100:.1f}%")
        print(f"  CV Mean: {metadata.get('cv_mean', 0)*100:.1f}%")
        print(f"  CV Std: {metadata.get('cv_std', 0)*100:.1f}%")
        print(f"  Total Features: {metadata.get('total_features', 'N/A')}")
        print(f"  Selected Features: {metadata.get('selected_features_count', 'N/A')}")
        print(f"  Label Classes: {metadata.get('label_classes', 'N/A')}")
        print(f"  Model Type: {metadata.get('model_type', 'N/A')}")
        
        # Check if there's feature importance
        if 'feature_importance' in metadata:
            print(f"  Feature Importance: {len(metadata['feature_importance'])} features")
    except Exception as e:
        print(f"  [ERROR] Could not read metadata: {e}")
else:
    print("\n[INFO] No metadata file found (model not trained yet)")

print("\n" + "=" * 70)
print("POTENTIAL ISSUES:")
print("=" * 70)
print("\n1. SEVERE CLASS IMBALANCE:")
print("   - Check training logs for 'Training data distribution'")
print("   - If HOLD > 80%, thresholds too strict")
print("   - Solution: Adjust thresholds (±3% → ±2%)")

print("\n2. TEMPORAL SPLIT ISSUE:")
print("   - Test set might have different distribution than training")
print("   - Market regime changed between train/test periods")
print("   - Solution: Use more diverse date range")

print("\n3. REGULARIZATION TOO STRONG:")
print("   - We increased regularization to prevent overfitting")
print("   - Might have gone too far (underfitting)")
print("   - Solution: Reduce regularization slightly")

print("\n4. FEATURE EXTRACTION ISSUE:")
print("   - Features might be all zeros or NaN")
print("   - Check if features are being extracted correctly")
print("   - Solution: Verify feature extraction")

print("\n5. LABEL ENCODING ISSUE:")
print("   - Labels might be encoded incorrectly")
print("   - Check label_classes in metadata")
print("   - Solution: Verify label encoding")

print("\n" + "=" * 70)
print("IMMEDIATE ACTIONS:")
print("=" * 70)
print("\n1. Check training logs for:")
print("   - 'Training data distribution: {...}'")
print("   - Label counts (BUY, SELL, HOLD)")
print("   - If HOLD > 75%, that's the problem")

print("\n2. If class imbalance detected:")
print("   - Adjust thresholds in training_pipeline.py")
print("   - Trading: ±3% → ±2%")
print("   - Mixed: ±5% → ±3%")

print("\n3. If regularization too strong:")
print("   - Reduce max_depth restrictions")
print("   - Reduce min_samples requirements")

print("\n4. Check if temporal split is working:")
print("   - Look for 'Using TEMPORAL splitting' in logs")
print("   - Verify date ranges are correct")

print("\n" + "=" * 70)
