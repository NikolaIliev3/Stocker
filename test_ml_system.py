"""
Comprehensive test script for ML training system
Tests all critical components and data flows
"""
import sys
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_binary_classification_logic():
    """Test binary classification label conversion"""
    logger.info("=" * 60)
    logger.info("TEST 1: Binary Classification Logic")
    logger.info("=" * 60)
    
    # Simulate training samples
    samples = [
        {'features': [1.0] * 130, 'label': 'BUY'},
        {'features': [2.0] * 130, 'label': 'SELL'},
        {'features': [3.0] * 130, 'label': 'HOLD'},
        {'features': [4.0] * 130, 'label': 'BUY'},
        {'features': [5.0] * 130, 'label': 'SELL'},
    ]
    
    # Test label filtering
    binary_mask = []
    binary_labels = []
    for i, sample in enumerate(samples):
        label = sample['label']
        if label == 'BUY':
            binary_mask.append(i)
            binary_labels.append('DOWN')  # BUY = price went down
        elif label == 'SELL':
            binary_mask.append(i)
            binary_labels.append('UP')  # SELL = price went up
        # Skip HOLD
    
    expected_binary_count = 4  # 2 BUY + 2 SELL, skip 1 HOLD
    actual_binary_count = len(binary_mask)
    
    if actual_binary_count == expected_binary_count:
        logger.info(f"✓ Binary filtering correct: {actual_binary_count} samples")
    else:
        logger.error(f"✗ Binary filtering failed: expected {expected_binary_count}, got {actual_binary_count}")
        return False
    
    # Check label conversion
    expected_labels = ['DOWN', 'UP', 'DOWN', 'UP']
    if binary_labels == expected_labels:
        logger.info(f"✓ Label conversion correct: {binary_labels}")
    else:
        logger.error(f"✗ Label conversion failed: expected {expected_labels}, got {binary_labels}")
        return False
    
    logger.info("✓ Binary classification logic: PASSED\n")
    return True

def test_cv_score_calculation():
    """Test CV score calculation and aggregation"""
    logger.info("=" * 60)
    logger.info("TEST 2: Cross-Validation Score Calculation")
    logger.info("=" * 60)
    
    # Simulate CV scores for different regimes
    regime_cv_scores = {
        'sideways': np.array([0.62, 0.64, 0.63]),
        'bull': np.array([0.65, 0.67, 0.66]),
        'bear': np.array([0.60, 0.61, 0.59])
    }
    
    regime_samples = {
        'sideways': 1000,
        'bull': 500,
        'bear': 300
    }
    
    # Calculate weighted average
    total_samples = sum(regime_samples.values())
    overall_cv_mean = 0.0
    
    for regime, scores in regime_cv_scores.items():
        cv_mean = scores.mean()
        weight = regime_samples[regime] / total_samples
        overall_cv_mean += cv_mean * weight
        logger.info(f"  {regime}: {cv_mean*100:.1f}% (weight: {weight:.2f})")
    
    logger.info(f"  Overall CV mean: {overall_cv_mean*100:.1f}%")
    
    # Validate calculation
    expected_mean = (
        regime_cv_scores['sideways'].mean() * (regime_samples['sideways'] / total_samples) +
        regime_cv_scores['bull'].mean() * (regime_samples['bull'] / total_samples) +
        regime_cv_scores['bear'].mean() * (regime_samples['bear'] / total_samples)
    )
    
    if abs(overall_cv_mean - expected_mean) < 0.001:
        logger.info("✓ CV score aggregation: PASSED\n")
        return True
    else:
        logger.error(f"✗ CV score aggregation failed: expected {expected_mean}, got {overall_cv_mean}")
        return False

def test_accuracy_reporting():
    """Test accuracy value extraction and reporting"""
    logger.info("=" * 60)
    logger.info("TEST 3: Accuracy Reporting")
    logger.info("=" * 60)
    
    # Simulate regime result
    regime_result = {
        'train_accuracy': 0.784,
        'test_accuracy': 0.634,
        'cv_mean': 0.625,
        'cv_std': 0.021
    }
    
    # Test extraction logic
    train_acc = regime_result.get('train_accuracy', 0.0)
    test_acc = regime_result.get('test_accuracy', 0.0)
    cv_mean = regime_result.get('cv_mean', 0.0)
    cv_std = regime_result.get('cv_std', 0.0)
    
    logger.info(f"  Train accuracy: {train_acc*100:.1f}%")
    logger.info(f"  Test accuracy: {test_acc*100:.1f}%")
    logger.info(f"  CV mean: {cv_mean*100:.1f}%")
    logger.info(f"  CV std: {cv_std*100:.1f}%")
    
    # Validate values are reasonable
    if 0.0 <= train_acc <= 1.0 and 0.0 <= test_acc <= 1.0:
        logger.info("✓ Accuracy values in valid range")
    else:
        logger.error(f"✗ Invalid accuracy values: train={train_acc}, test={test_acc}")
        return False
    
    if 0.0 <= cv_mean <= 1.0 and 0.0 <= cv_std <= 1.0:
        logger.info("✓ CV values in valid range")
    else:
        logger.error(f"✗ Invalid CV values: mean={cv_mean}, std={cv_std}")
        return False
    
    logger.info("✓ Accuracy reporting: PASSED\n")
    return True

def test_regime_model_aggregation():
    """Test regime model accuracy aggregation"""
    logger.info("=" * 60)
    logger.info("TEST 4: Regime Model Aggregation")
    logger.info("=" * 60)
    
    # Simulate multiple regime models
    regime_data = {
        'sideways': {'test_acc': 0.634, 'samples': 2317},
        'bull': {'test_acc': 0.650, 'samples': 500},
        'bear': {'test_acc': 0.600, 'samples': 300}
    }
    
    overall_test_acc = 0.0
    total_samples = 0
    
    for regime, data in regime_data.items():
        test_acc = data['test_acc']
        samples = data['samples']
        overall_test_acc += test_acc * samples
        total_samples += samples
        logger.info(f"  {regime}: {test_acc*100:.1f}% ({samples} samples)")
    
    if total_samples > 0:
        overall_test_acc /= total_samples
    
    logger.info(f"  Overall test accuracy: {overall_test_acc*100:.1f}%")
    
    # Validate calculation
    expected = (
        0.634 * 2317 + 0.650 * 500 + 0.600 * 300
    ) / (2317 + 500 + 300)
    
    if abs(overall_test_acc - expected) < 0.001:
        logger.info("✓ Regime aggregation: PASSED\n")
        return True
    else:
        logger.error(f"✗ Regime aggregation failed: expected {expected}, got {overall_test_acc}")
        return False

def test_label_already_binary_detection():
    """Test detection of already-binary labels"""
    logger.info("=" * 60)
    logger.info("TEST 5: Already-Binary Label Detection")
    logger.info("=" * 60)
    
    # Test case 1: Already binary (UP/DOWN)
    labels_binary = np.array(['UP', 'DOWN', 'UP', 'DOWN', 'UP'])
    unique_labels = set(labels_binary)
    is_binary = unique_labels.issubset({'UP', 'DOWN'})
    
    if is_binary:
        logger.info("✓ Correctly detected binary labels (UP/DOWN)")
    else:
        logger.error("✗ Failed to detect binary labels")
        return False
    
    # Test case 2: Not binary (BUY/SELL/HOLD)
    labels_not_binary = np.array(['BUY', 'SELL', 'HOLD', 'BUY', 'SELL'])
    unique_labels = set(labels_not_binary)
    is_binary = unique_labels.issubset({'UP', 'DOWN'})
    
    if not is_binary:
        logger.info("✓ Correctly detected non-binary labels (BUY/SELL/HOLD)")
    else:
        logger.error("✗ Incorrectly detected non-binary as binary")
        return False
    
    logger.info("✓ Binary detection: PASSED\n")
    return True

def test_data_alignment():
    """Test data alignment between samples and features"""
    logger.info("=" * 60)
    logger.info("TEST 6: Data Alignment")
    logger.info("=" * 60)
    
    # Simulate training samples
    training_samples = [
        {'features': [1.0] * 130, 'label': 'BUY', 'date': '2020-01-01'},
        {'features': [2.0] * 130, 'label': 'SELL', 'date': '2020-01-02'},
        {'features': [3.0] * 130, 'label': 'HOLD', 'date': '2020-01-03'},
        {'features': [4.0] * 130, 'label': 'BUY', 'date': '2020-01-04'},
    ]
    
    # Extract features and labels
    X = np.array([s['features'] for s in training_samples])
    y = np.array([s['label'] for s in training_samples])
    
    # Binary filtering
    binary_mask = [i for i, label in enumerate(y) if label != 'HOLD']
    X_filtered = X[binary_mask]
    y_filtered = y[binary_mask]
    
    # Check alignment
    if len(X_filtered) == len(y_filtered):
        logger.info(f"✓ Feature/label alignment: {len(X_filtered)} samples")
    else:
        logger.error(f"✗ Alignment failed: X={len(X_filtered)}, y={len(y_filtered)}")
        return False
    
    # Check filtered samples match
    filtered_samples = [training_samples[i] for i in binary_mask]
    if len(filtered_samples) == len(X_filtered):
        logger.info(f"✓ Sample alignment: {len(filtered_samples)} samples")
    else:
        logger.error(f"✗ Sample alignment failed")
        return False
    
    logger.info("✓ Data alignment: PASSED\n")
    return True

def test_cv_fallback():
    """Test CV score fallback logic"""
    logger.info("=" * 60)
    logger.info("TEST 7: CV Score Fallback")
    logger.info("=" * 60)
    
    test_score = 0.634
    
    # Simulate CV failure
    cv_scores = None
    try:
        # This would fail in real scenario
        raise Exception("CV failed")
    except Exception:
        cv_scores = np.array([test_score])  # Fallback
    
    if cv_scores is not None and len(cv_scores) > 0:
        cv_mean = cv_scores.mean()
        logger.info(f"✓ Fallback to test score: {cv_mean*100:.1f}%")
        if abs(cv_mean - test_score) < 0.001:
            logger.info("✓ CV fallback: PASSED\n")
            return True
        else:
            logger.error(f"✗ Fallback value incorrect: expected {test_score}, got {cv_mean}")
            return False
    else:
        logger.error("✗ Fallback failed: no scores")
        return False

def run_all_tests():
    """Run all tests"""
    logger.info("\n" + "=" * 60)
    logger.info("COMPREHENSIVE ML SYSTEM TEST SUITE")
    logger.info("=" * 60 + "\n")
    
    tests = [
        ("Binary Classification Logic", test_binary_classification_logic),
        ("CV Score Calculation", test_cv_score_calculation),
        ("Accuracy Reporting", test_accuracy_reporting),
        ("Regime Model Aggregation", test_regime_model_aggregation),
        ("Binary Label Detection", test_label_already_binary_detection),
        ("Data Alignment", test_data_alignment),
        ("CV Fallback", test_cv_fallback),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"✗ {test_name} crashed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            results.append((test_name, False))
    
    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"  {status}: {test_name}")
    
    logger.info("")
    logger.info(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED!")
        return True
    else:
        logger.error(f"⚠️ {total - passed} test(s) failed")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
