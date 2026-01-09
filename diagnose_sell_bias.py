"""
Diagnostic script to investigate SELL bias in ML predictions
"""
import json
import numpy as np
from pathlib import Path
from ml_training import StockPredictionML, FeatureExtractor
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_model_bias():
    """Analyze why the model is biased toward SELL predictions"""
    
    data_dir = Path("data")
    strategy = "trading"
    
    # Load the model
    ml_predictor = StockPredictionML(data_dir, strategy)
    
    # Force reload if needed
    if not ml_predictor.is_trained():
        logger.warning("Model not loaded, attempting to reload...")
        try:
            ml_predictor._load_model()
        except:
            pass
    
    if not ml_predictor.is_trained():
        logger.error("Model is not trained!")
        # Try to read metadata directly
        metadata_file = data_dir / f"ml_metadata_{strategy}.json"
        if metadata_file.exists():
            logger.info(f"Reading metadata directly from {metadata_file}")
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                ml_predictor.metadata = metadata
                logger.info("Loaded metadata directly")
        else:
            logger.error("Metadata file not found!")
            return
    
    logger.info("=" * 60)
    logger.info("MODEL BIAS DIAGNOSTIC")
    logger.info("=" * 60)
    
    # Check metadata
    if ml_predictor.metadata:
        logger.info("\n1. MODEL METADATA:")
        logger.info(f"   - Use binary classification: {ml_predictor.metadata.get('use_binary_classification', False)}")
        logger.info(f"   - Training accuracy: {ml_predictor.metadata.get('train_accuracy', 0)*100:.1f}%")
        logger.info(f"   - Test accuracy: {ml_predictor.metadata.get('test_accuracy', 0)*100:.1f}%")
        
        # Check class distribution in training
        if 'training_label_distribution' in ml_predictor.metadata:
            logger.info(f"   - Training label distribution: {ml_predictor.metadata['training_label_distribution']}")
    
    # Check label encoder
    logger.info("\n2. LABEL ENCODER:")
    if hasattr(ml_predictor, 'label_encoder') and hasattr(ml_predictor.label_encoder, 'classes_'):
        classes = list(ml_predictor.label_encoder.classes_)
        logger.info(f"   - Classes: {classes}")
        logger.info(f"   - Number of classes: {len(classes)}")
        
        # Check if it's binary mode
        is_binary = set(classes) == {'UP', 'DOWN'} or (len(classes) == 2 and 'UP' in classes and 'DOWN' in classes)
        logger.info(f"   - Binary mode: {is_binary}")
        
        if is_binary:
            logger.info("   - UP → SELL (price goes up = sell signal)")
            logger.info("   - DOWN → BUY (price goes down = buy signal)")
    else:
        logger.warning("   - Label encoder not available!")
    
    # Check regime models
    logger.info("\n3. REGIME MODELS:")
    if hasattr(ml_predictor, 'regime_models') and ml_predictor.regime_models:
        logger.info(f"   - Number of regime models: {len(ml_predictor.regime_models)}")
        for regime, regime_ml in ml_predictor.regime_models.items():
            logger.info(f"   - {regime}:")
            if hasattr(regime_ml, 'label_encoder') and hasattr(regime_ml.label_encoder, 'classes_'):
                regime_classes = list(regime_ml.label_encoder.classes_)
                logger.info(f"     Classes: {regime_classes}")
            if hasattr(regime_ml, 'metadata') and regime_ml.metadata:
                logger.info(f"     Test accuracy: {regime_ml.metadata.get('test_accuracy', 0)*100:.1f}%")
                if 'training_label_distribution' in regime_ml.metadata:
                    logger.info(f"     Training distribution: {regime_ml.metadata['training_label_distribution']}")
    else:
        logger.info("   - No regime models (using single model)")
    
    # Test with dummy features to see default behavior
    logger.info("\n4. TESTING WITH DUMMY FEATURES:")
    feature_extractor = FeatureExtractor()
    num_features = 130  # Standard number of features
    
    # Create dummy features (all zeros - neutral)
    dummy_features = np.zeros((1, num_features))
    
    # Create dummy stock_data and history_data
    dummy_stock_data = {'price': 100.0}
    dummy_history_data = {'data': []}
    
    try:
        # Try to get prediction
        result = ml_predictor.predict(dummy_features, dummy_stock_data, dummy_history_data)
        if result and 'action' in result:
            logger.info(f"   - Dummy prediction: {result['action']}")
            logger.info(f"   - Confidence: {result.get('confidence', 0):.1f}%")
            logger.info(f"   - Probabilities: {result.get('probabilities', {})}")
        else:
            logger.warning(f"   - Prediction failed: {result}")
    except Exception as e:
        logger.error(f"   - Error making prediction: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Check if there's a class imbalance issue
    logger.info("\n5. CLASS IMBALANCE ANALYSIS:")
    if ml_predictor.metadata and 'training_label_distribution' in ml_predictor.metadata:
        dist = ml_predictor.metadata['training_label_distribution']
        if isinstance(dist, dict):
            total = sum(dist.values())
            for label, count in dist.items():
                pct = (count / total * 100) if total > 0 else 0
                logger.info(f"   - {label}: {count} ({pct:.1f}%)")
                
                # Check if there's significant imbalance
                if pct > 60:
                    logger.warning(f"     ⚠️  {label} is overrepresented ({pct:.1f}%)")
                    logger.warning(f"     This may cause the model to bias toward {label}")
    
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATIONS:")
    logger.info("=" * 60)
    
    # Provide recommendations
    if ml_predictor.metadata and 'training_label_distribution' in ml_predictor.metadata:
        dist = ml_predictor.metadata['training_label_distribution']
        if isinstance(dist, dict):
            total = sum(dist.values())
            for label, count in dist.items():
                pct = (count / total * 100) if total > 0 else 0
                if pct > 60:
                    logger.info(f"1. Class imbalance detected: {label} is {pct:.1f}% of training data")
                    logger.info("   → Consider using stronger class weights or SMOTE oversampling")
                    logger.info("   → Collect more samples for the minority class")
    
    logger.info("2. Model is predicting SELL almost exclusively")
    logger.info("   → Check if features are providing good separation")
    logger.info("   → Consider adding a confidence threshold to filter low-confidence predictions")
    logger.info("   → Review feature importance to see which features drive SELL predictions")
    
    logger.info("3. Backtesting accuracy is 41% (below random)")
    logger.info("   → Model may need more diverse training data")
    logger.info("   → Consider retraining with more balanced classes")
    logger.info("   → Check if there's a data leakage issue")

if __name__ == "__main__":
    analyze_model_bias()
