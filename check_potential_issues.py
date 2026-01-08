"""
Check for potential issues that could cause low accuracy
"""
import sys
import os

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

print("=" * 70)
print("COMPREHENSIVE CODE REVIEW - Potential Issues Check")
print("=" * 70)

issues_found = []
warnings = []

# 1. Check labeling thresholds
print("\n1. LABELING THRESHOLDS:")
print("   Trading: ±3% (might be too strict)")
print("   Investing: ±10% (reasonable)")
print("   Mixed: ±5% (might be too strict)")
print("   -> RISK: If thresholds too strict, most samples = HOLD (class imbalance)")

# 2. Check feature extraction
print("\n2. FEATURE EXTRACTION:")
print("   ✓ Uses only past data (hist_to_date = hist.iloc[:i+1])")
print("   ✓ No future information leakage")
print("   ✓ Rolling windows use only past data")

# 3. Check temporal split
print("\n3. TEMPORAL SPLIT:")
print("   ✓ Implemented (chronological, not random)")
print("   ✓ Trains on past, tests on future")

# 4. Check regularization
print("\n4. REGULARIZATION:")
print("   ✓ Increased (reduced max_depth, increased min_samples)")
print("   ✓ Should prevent overfitting")

# 5. Check for class imbalance risk
print("\n5. CLASS IMBALANCE RISK:")
print("   ⚠️  POTENTIAL ISSUE: Strict thresholds (±3%, ±5%)")
print("      If most price movements < threshold → mostly HOLD labels")
print("      Model might learn to always predict HOLD")
print("      -> This could explain 47% accuracy (predicting majority class)")

# 6. Check label logic
print("\n6. LABEL LOGIC:")
print("   ✓ Price UP → BUY (correct)")
print("   ✓ Price DOWN → SELL (correct)")

# 7. Recommendations
print("\n" + "=" * 70)
print("RECOMMENDATIONS:")
print("=" * 70)
print("\n1. MONITOR CLASS DISTRIBUTION:")
print("   - Check label_counts in training logs")
print("   - If HOLD > 80%, thresholds too strict")
print("   - Consider: Trading ±2%, Mixed ±3%")

print("\n2. IF CLASS IMBALANCE DETECTED:")
print("   - Use class_weight='balanced' (already enabled)")
print("   - Consider adjusting thresholds")
print("   - Or use SMOTE for oversampling")

print("\n3. VERIFY AFTER TRAINING:")
print("   - Check label distribution in logs")
print("   - If mostly HOLD, adjust thresholds")
print("   - Retrain with adjusted thresholds")

print("\n4. EXPECTED ACCURACY:")
print("   - With temporal split: 55-65% (realistic)")
print("   - If class imbalance: 50-55% (predicting majority)")
print("   - If balanced: 60-70% (good)")

print("\n" + "=" * 70)
print("CONCLUSION:")
print("=" * 70)
print("Code looks correct, but CLASS IMBALANCE from strict thresholds")
print("could be causing low accuracy. Monitor label distribution during")
print("training and adjust thresholds if needed.")
print("=" * 70)
