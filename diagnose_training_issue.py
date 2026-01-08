"""
Diagnostic script to identify why ML training accuracy is so low
"""
import json
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
print("DIAGNOSING LOW TRAINING ACCURACY")
print("=" * 70)

if not METADATA_FILE.exists():
    print(f"\n[ERROR] Metadata file not found: {METADATA_FILE}")
    print("        Model may not have been trained yet.")
    sys.exit(1)

try:
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    print(f"\n1. TRAINING METRICS:")
    print(f"   Training Accuracy: {metadata.get('train_accuracy', 0)*100:.1f}%")
    print(f"   Test Accuracy: {metadata.get('test_accuracy', 0)*100:.1f}%")
    print(f"   Cross-Validation: {metadata.get('cv_mean', 0)*100:.1f}% (±{metadata.get('cv_std', 0)*100:.1f}%)")
    
    print(f"\n2. CLASS DISTRIBUTION:")
    label_classes = metadata.get('label_classes', [])
    if label_classes:
        print(f"   Classes: {label_classes}")
    else:
        print(f"   [WARNING] No class information found")
    
    print(f"\n3. FEATURE INFORMATION:")
    total_features = metadata.get('total_features', 0)
    selected_count = metadata.get('selected_features_count', 0)
    print(f"   Total Features: {total_features}")
    print(f"   Selected Features: {selected_count}")
    if total_features > 0:
        print(f"   Feature Selection Ratio: {selected_count/total_features*100:.1f}%")
    
    print(f"\n4. FEATURE IMPORTANCE:")
    feature_importance = metadata.get('feature_importance', {})
    if feature_importance:
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"   Top 10 Features:")
        for name, importance in top_features:
            print(f"     - {name}: {importance:.4f}")
    else:
        print(f"   [WARNING] No feature importance data")
    
    print(f"\n5. DIAGNOSIS:")
    test_acc = metadata.get('test_accuracy', 0)
    train_acc = metadata.get('train_accuracy', 0)
    gap = train_acc - test_acc
    
    issues = []
    
    if test_acc < 0.35:  # Worse than random (33%)
        issues.append("CRITICAL: Test accuracy worse than random (33% for 3 classes)")
    
    if gap > 0.15:
        issues.append("WARNING: Large train/test gap (>15%) - possible overfitting")
    elif gap < 0.05 and test_acc < 0.50:
        issues.append("WARNING: Low accuracy with small gap - model may not be learning")
    
    if selected_count < 10:
        issues.append("WARNING: Very few features selected - may be losing important information")
    
    if not feature_importance:
        issues.append("WARNING: No feature importance data - model may not have trained properly")
    
    if issues:
        print("   Issues detected:")
        for issue in issues:
            print(f"     - {issue}")
    else:
        print("   No obvious issues detected in metadata")
    
    print(f"\n6. RECOMMENDATIONS:")
    if test_acc < 0.35:
        print("   [CRITICAL] Model is performing worse than random guessing!")
        print("   Possible causes:")
        print("     1. Severe class imbalance (check training logs for class distribution)")
        print("     2. Features are all zeros or invalid")
        print("     3. Labeling logic might be incorrect")
        print("     4. Data quality issues")
        print("   ")
        print("   Action: Check training logs for:")
        print("     - Class distribution (should be balanced)")
        print("     - SMOTE application (should be applied if imbalance detected)")
        print("     - Feature extraction errors")
        print("     - Sample count per class")
    
    if gap > 0.15:
        print("   [OVERFITTING] Model memorizing training data")
        print("   Action: Increase regularization or reduce model complexity")
    
    if selected_count < 10:
        print("   [FEATURE SELECTION] Too few features selected")
        print("   Action: Check if feature selection is too aggressive")
    
except Exception as e:
    print(f"\n[ERROR] Could not read metadata: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Next: Check training logs for class distribution and SMOTE application")
print("=" * 70)
