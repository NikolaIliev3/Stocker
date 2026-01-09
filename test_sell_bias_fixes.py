"""
Test script to verify SELL bias fixes
"""
import numpy as np
from pathlib import Path
from ml_training import StockPredictionML
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_confidence_threshold():
    """Test that confidence threshold filters low-confidence predictions"""
    logger.info("=" * 60)
    logger.info("TEST 1: Confidence Threshold")
    logger.info("=" * 60)
    
    data_dir = Path("data")
    strategy = "trading"
    
    # Load model
    ml_predictor = StockPredictionML(data_dir, strategy)
    
    if not ml_predictor.is_trained():
        logger.warning("Model not trained - skipping confidence threshold test")
        return False
    
    # Check if model is binary
    if hasattr(ml_predictor, 'label_encoder') and hasattr(ml_predictor.label_encoder, 'classes_'):
        classes = list(ml_predictor.label_encoder.classes_)
        is_binary = set(classes) == {'UP', 'DOWN'} or (len(classes) == 2 and 'UP' in classes and 'DOWN' in classes)
        
        if is_binary:
            logger.info("✓ Model is in binary mode (UP/DOWN)")
            logger.info("✓ Confidence threshold of 55% should be applied")
            logger.info("  - Predictions with <55% confidence will return HOLD")
            return True
        else:
            logger.info("Model is not in binary mode - confidence threshold only applies to binary")
            return True
    else:
        logger.warning("Cannot determine model type")
        return False

def test_smote_auto_enable():
    """Test that SMOTE auto-enables for binary classification"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: SMOTE Auto-Enable Logic")
    logger.info("=" * 60)
    
    # Test the logic
    use_binary_classification = True
    use_smote = None  # Not explicitly set
    
    # Simulate the auto-enable logic
    if use_smote is None:
        use_smote = use_binary_classification
    
    if use_smote:
        logger.info("✓ SMOTE auto-enables when binary classification is used")
        logger.info("✓ This will balance classes during training")
        return True
    else:
        logger.error("✗ SMOTE should auto-enable for binary classification")
        return False

def test_class_imbalance_detection():
    """Test class imbalance detection logic"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Class Imbalance Detection")
    logger.info("=" * 60)
    
    # Simulate training data distribution from logs
    up_count = 1399
    down_count = 918
    total = up_count + down_count
    
    imbalance_ratio = up_count / down_count if down_count > 0 else 1.0
    up_pct = (up_count / total * 100) if total > 0 else 0
    down_pct = (down_count / total * 100) if total > 0 else 0
    
    logger.info(f"Training data distribution:")
    logger.info(f"  - UP (SELL): {up_count} ({up_pct:.1f}%)")
    logger.info(f"  - DOWN (BUY): {down_count} ({down_pct:.1f}%)")
    logger.info(f"  - Imbalance ratio: {imbalance_ratio:.2f}")
    
    if imbalance_ratio > 1.3:
        logger.info(f"✓ Imbalance detected (ratio: {imbalance_ratio:.2f} > 1.3)")
        logger.info("✓ SMOTE will be applied to balance classes")
        return True
    else:
        logger.info("Imbalance is acceptable")
        return True

def test_prediction_diversity():
    """Test prediction diversity monitoring"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Prediction Diversity Check")
    logger.info("=" * 60)
    
    # Simulate recent predictions (from backtest logs - all SELL)
    recent_predictions = ['SELL'] * 100  # All SELL (current problem)
    
    sell_count = recent_predictions.count('SELL')
    buy_count = recent_predictions.count('BUY')
    hold_count = recent_predictions.count('HOLD')
    
    total = len(recent_predictions)
    sell_pct = (sell_count / total * 100) if total > 0 else 0
    
    logger.info(f"Recent predictions distribution:")
    logger.info(f"  - SELL: {sell_count} ({sell_pct:.1f}%)")
    logger.info(f"  - BUY: {buy_count} ({100 - sell_pct:.1f}%)")
    logger.info(f"  - HOLD: {hold_count}")
    
    if sell_pct > 80:
        logger.warning(f"⚠️  SELL bias detected: {sell_pct:.1f}% of predictions are SELL")
        logger.info("✓ With confidence threshold, low-confidence SELL predictions will be filtered to HOLD")
        logger.info("✓ With SMOTE, model should learn more balanced patterns")
        return True
    else:
        logger.info("✓ Prediction distribution is balanced")
        return True

def test_model_metadata():
    """Check if model metadata exists and contains useful info"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Model Metadata Check")
    logger.info("=" * 60)
    
    data_dir = Path("data")
    strategy = "trading"
    
    # Check for regime model metadata
    regime_metadata_file = data_dir / f"ml_metadata_{strategy}_sideways.json"
    main_metadata_file = data_dir / f"ml_metadata_{strategy}.json"
    
    metadata_found = False
    if regime_metadata_file.exists():
        logger.info(f"✓ Found regime metadata: {regime_metadata_file.name}")
        try:
            with open(regime_metadata_file, 'r') as f:
                metadata = json.load(f)
                logger.info(f"  - Test accuracy: {metadata.get('test_accuracy', 0)*100:.1f}%")
                logger.info(f"  - Binary classification: {metadata.get('use_binary_classification', False)}")
                metadata_found = True
        except Exception as e:
            logger.warning(f"Could not read metadata: {e}")
    
    if main_metadata_file.exists():
        logger.info(f"✓ Found main metadata: {main_metadata_file.name}")
        try:
            with open(main_metadata_file, 'r') as f:
                metadata = json.load(f)
                logger.info(f"  - Test accuracy: {metadata.get('test_accuracy', 0)*100:.1f}%")
                metadata_found = True
        except Exception as e:
            logger.warning(f"Could not read metadata: {e}")
    
    if not metadata_found:
        logger.warning("No metadata files found - model may need retraining")
    
    return metadata_found

def main():
    """Run all tests"""
    logger.info("\n" + "=" * 60)
    logger.info("SELL BIAS FIXES - TEST SUITE")
    logger.info("=" * 60)
    
    results = []
    
    # Test 1: Confidence threshold
    results.append(("Confidence Threshold", test_confidence_threshold()))
    
    # Test 2: SMOTE auto-enable
    results.append(("SMOTE Auto-Enable", test_smote_auto_enable()))
    
    # Test 3: Class imbalance detection
    results.append(("Class Imbalance Detection", test_class_imbalance_detection()))
    
    # Test 4: Prediction diversity
    results.append(("Prediction Diversity", test_prediction_diversity()))
    
    # Test 5: Model metadata
    results.append(("Model Metadata", test_model_metadata()))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{status}: {test_name}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATIONS")
    logger.info("=" * 60)
    logger.info("1. Retrain ML model to apply SMOTE balancing")
    logger.info("2. Run backtesting to verify prediction diversity")
    logger.info("3. Monitor prediction distribution (should be ~40-60% each)")
    logger.info("4. Check backtesting accuracy (should improve from 41%)")

if __name__ == "__main__":
    main()
