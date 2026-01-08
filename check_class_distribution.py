"""
Check what the actual class distribution is in training
This will help diagnose if thresholds are too strict
"""
import sys
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

print("=" * 70)
print("CHECKING CLASS DISTRIBUTION ISSUE")
print("=" * 70)

print("\n[INFO] The model is performing worse than random (40.2% vs 33%)")
print("       This suggests the model is not learning meaningful patterns.")
print("\n[POSSIBLE CAUSES]:")

print("\n1. SEVERE CLASS IMBALANCE:")
print("   - If thresholds (±3% for trading) are too strict,")
print("     most samples might be labeled as HOLD (90%+)")
print("   - This makes the model predict HOLD most of the time")
print("   - Check training logs for: 'Training data distribution (before split)'")

print("\n2. FEATURES NOT INFORMATIVE:")
print("   - Feature importances are very low (max 0.03)")
print("   - This suggests features don't predict labels well")
print("   - Possible causes:")
print("     * Features are mostly zeros or invalid")
print("     * Features aren't scaled properly")
print("     * Features don't capture price movement patterns")

print("\n3. SMOTE NOT APPLIED:")
print("   - If class imbalance exists but SMOTE wasn't applied,")
print("     the model will predict the majority class")
print("   - Check training logs for: 'Applying SMOTE' or 'After SMOTE'")

print("\n4. TEMPORAL SPLIT ISSUE:")
print("   - If temporal split creates very different train/test distributions,")
print("     the model won't generalize")
print("   - Check if train/test have similar class distributions")

print("\n[ACTION REQUIRED]:")
print("\nPlease check your training logs and look for:")
print("  1. 'Training data distribution (before split): {...}'")
print("     - If one class (usually HOLD) is >80%, that's the problem")
print("  2. 'Applying SMOTE to balance TRAINING classes only'")
print("     - If this doesn't appear, SMOTE wasn't applied")
print("  3. 'Balanced training distribution: {...}'")
print("     - If this appears, SMOTE was applied successfully")

print("\n[QUICK FIX - IF CLASS IMBALANCE DETECTED]:")
print("  If HOLD is >80% of samples:")
print("  - Reduce thresholds in training_pipeline.py:")
print("    Trading: ±3% → ±2% (more BUY/SELL labels)")
print("    Mixed: ±5% → ±3% (more BUY/SELL labels)")

print("\n" + "=" * 70)
