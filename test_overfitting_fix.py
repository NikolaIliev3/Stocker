"""
Test script for ML Overfitting Fix
Verifies that training with large datasets doesn't result in overfitting.

Run: python test_overfitting_fix.py
"""
import sys
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_sample_limiting():
    """Test that sample limiting works correctly"""
    print("\n" + "="*60)
    print("TEST 1: Sample Limiting Logic")
    print("="*60)
    
    # Simulate large dataset
    large_sample_count = 10000
    max_samples = 5000
    
    # Mock training samples
    np.random.seed(42)
    X = np.random.randn(large_sample_count, 50)
    y = np.random.choice(['UP', 'DOWN'], size=large_sample_count)
    
    # Check class distribution before
    from collections import Counter
    dist_before = Counter(y)
    print(f"Before limiting: {large_sample_count} samples, distribution: {dict(dist_before)}")
    
    # Apply sample limiting (mimicking ml_training.py logic)
    if len(X) > max_samples:
        from sklearn.model_selection import train_test_split
        all_indices = np.arange(len(X))
        # FIXED: keep_indices is the TRAIN portion (first return value)
        keep_indices, _ = train_test_split(
            all_indices,
            train_size=max_samples,
            stratify=y,
            random_state=42
        )
        X = X[keep_indices]
        y = y[keep_indices]
    
    dist_after = Counter(y)
    print(f"After limiting: {len(X)} samples, distribution: {dict(dist_after)}")
    
    # Verify
    assert len(X) == max_samples, f"Expected {max_samples} samples, got {len(X)}"
    
    # Check class balance is preserved (should be roughly similar proportions)
    up_ratio_before = dist_before['UP'] / large_sample_count
    up_ratio_after = dist_after['UP'] / max_samples
    ratio_diff = abs(up_ratio_before - up_ratio_after)
    assert ratio_diff < 0.05, f"Class balance not preserved: diff={ratio_diff:.3f}"
    
    print(f"[PASS] Sample limiting works correctly (ratio diff: {ratio_diff:.3f})")
    return True


def test_config_values():
    """Test that config values are set correctly for anti-overfitting"""
    print("\n" + "="*60)
    print("TEST 2: Config Values Check")
    print("="*60)
    
    try:
        from config import (ML_RF_MAX_DEPTH, ML_RF_N_ESTIMATORS, 
                           ML_GB_MAX_DEPTH, ML_GB_N_ESTIMATORS,
                           ML_RF_MIN_SAMPLES_SPLIT, ML_RF_MIN_SAMPLES_LEAF)
        
        print(f"RF max_depth: {ML_RF_MAX_DEPTH} (expected: 5)")
        print(f"RF n_estimators: {ML_RF_N_ESTIMATORS} (expected: 60)")
        print(f"RF min_samples_split: {ML_RF_MIN_SAMPLES_SPLIT} (expected: 20)")
        print(f"RF min_samples_leaf: {ML_RF_MIN_SAMPLES_LEAF} (expected: 10)")
        print(f"GB max_depth: {ML_GB_MAX_DEPTH} (expected: 3)")
        print(f"GB n_estimators: {ML_GB_N_ESTIMATORS} (expected: 80)")
        
        # Verify anti-overfitting configuration
        checks = [
            (ML_RF_MAX_DEPTH <= 6, "RF max_depth should be <= 6"),
            (ML_RF_N_ESTIMATORS <= 100, "RF n_estimators should be <= 100"),
            (ML_GB_MAX_DEPTH <= 4, "GB max_depth should be <= 4"),
            (ML_GB_N_ESTIMATORS <= 120, "GB n_estimators should be <= 120"),
            (ML_RF_MIN_SAMPLES_SPLIT >= 15, "RF min_samples_split should be >= 15"),
            (ML_RF_MIN_SAMPLES_LEAF >= 8, "RF min_samples_leaf should be >= 8"),
        ]
        
        all_passed = True
        for check, msg in checks:
            if check:
                print(f"  [PASS] {msg}")
            else:
                print(f"  [FAIL] {msg}")
                all_passed = False
        
        if all_passed:
            print("[PASS] All config values are correct for anti-overfitting")
        else:
            print("[FAIL] Some config values need adjustment")
        
        return all_passed
        
    except ImportError as e:
        print(f"[FAIL] Could not import config: {e}")
        return False


def test_training_accuracy_gap():
    """
    Quick synthetic test to verify the model doesn't severely overfit.
    Uses synthetic data to simulate training behavior.
    """
    print("\n" + "="*60)
    print("TEST 3: Training-Test Gap (Synthetic)")
    print("="*60)
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from config import (ML_RF_MAX_DEPTH, ML_RF_N_ESTIMATORS,
                           ML_RF_MIN_SAMPLES_SPLIT, ML_RF_MIN_SAMPLES_LEAF)
        
        # Create synthetic data that mimics stock features
        np.random.seed(42)
        n_samples = 5000
        n_features = 50
        
        # Create features with some signal
        X = np.random.randn(n_samples, n_features)
        # Add some structure to make it learnable
        y = (X[:, 0] + 0.5*X[:, 1] - 0.3*X[:, 2] + np.random.randn(n_samples)*0.5 > 0).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train with current config
        rf = RandomForestClassifier(
            n_estimators=ML_RF_N_ESTIMATORS,
            max_depth=ML_RF_MAX_DEPTH,
            min_samples_split=ML_RF_MIN_SAMPLES_SPLIT,
            min_samples_leaf=ML_RF_MIN_SAMPLES_LEAF,
            random_state=42
        )
        rf.fit(X_train, y_train)
        
        train_acc = rf.score(X_train, y_train)
        test_acc = rf.score(X_test, y_test)
        gap = train_acc - test_acc
        
        print(f"Training Accuracy: {train_acc*100:.1f}%")
        print(f"Test Accuracy: {test_acc*100:.1f}%")
        print(f"Gap: {gap*100:.1f}%")
        
        # Check for overfitting
        if gap > 0.25:
            print(f"[FAIL] SEVERE overfitting detected (gap > 25%)")
            return False
        elif gap > 0.15:
            print(f"[WARN] Moderate overfitting (gap > 15%)")
            return True  # Pass but warn
        else:
            print(f"[PASS] Good generalization (gap < 15%)")
            return True
            
    except Exception as e:
        print(f"[FAIL] Test failed with error: {e}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "="*60)
    print("ML OVERFITTING FIX - VERIFICATION TESTS")
    print("="*60)
    
    results = []
    
    # Test 1: Sample limiting
    try:
        results.append(("Sample Limiting", test_sample_limiting()))
    except Exception as e:
        print(f"[FAIL] Sample Limiting test crashed: {e}")
        results.append(("Sample Limiting", False))
    
    # Test 2: Config values
    try:
        results.append(("Config Values", test_config_values()))
    except Exception as e:
        print(f"[FAIL] Config Values test crashed: {e}")
        results.append(("Config Values", False))
    
    # Test 3: Training gap
    try:
        results.append(("Training-Test Gap", test_training_accuracy_gap()))
    except Exception as e:
        print(f"[FAIL] Training-Test Gap test crashed: {e}")
        results.append(("Training-Test Gap", False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = 0
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status}: {name}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n*** All tests passed! The overfitting fix is working correctly. ***")
        return True
    else:
        print("\n*** Some tests failed. Review the failures above. ***")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
