"""
Diagnostic script to investigate backtesting issues
- Check if ML model is loaded correctly
- Verify binary classification conversion
- Test prediction flow
"""
import warnings
warnings.filterwarnings('ignore')

from ml_training import StockPredictionML, FeatureExtractor
from hybrid_predictor import HybridStockPredictor
from data_fetcher import StockDataFetcher
from config import APP_DATA_DIR
import numpy as np

def test_ml_model():
    """Test if ML model is loaded and working"""
    print("="*60)
    print("TEST 1: ML MODEL STATUS")
    print("="*60)
    
    ml = StockPredictionML(APP_DATA_DIR, 'trading')
    
    print(f"Model loaded: {ml.model is not None}")
    print(f"Is trained: {ml.is_trained()}")
    
    if ml.metadata:
        print(f"Metadata loaded: Yes")
        print(f"  Strategy: {ml.metadata.get('strategy')}")
        print(f"  Binary mode: {ml.metadata.get('use_binary_classification', False)}")
        print(f"  Train accuracy: {ml.metadata.get('train_accuracy', 0)*100:.1f}%")
        print(f"  Test accuracy: {ml.metadata.get('test_accuracy', 0)*100:.1f}%")
        print(f"  Label classes: {ml.metadata.get('label_classes', [])}")
    else:
        print("Metadata: Not loaded")
    
    # Test prediction
    if ml.is_trained():
        # Create dummy features
        feature_extractor = FeatureExtractor()
        dummy_features = np.zeros(143)  # Typical feature count
        prediction = ml.predict(dummy_features)
        
        print(f"\nTest prediction:")
        print(f"  Action: {prediction.get('action')}")
        print(f"  Confidence: {prediction.get('confidence', 0):.1f}%")
        print(f"  Probabilities: {prediction.get('probabilities', {})}")
        if 'error' in prediction:
            print(f"  ERROR: {prediction['error']}")
    else:
        print("\nModel not trained - cannot test prediction")
    
    return ml.is_trained()

def test_hybrid_predictor():
    """Test hybrid predictor"""
    print("\n" + "="*60)
    print("TEST 2: HYBRID PREDICTOR")
    print("="*60)
    
    df = StockDataFetcher()
    
    # Get real data
    stock_data = df.fetch_stock_data('AAPL')
    history_data = df.fetch_stock_history('AAPL', period='3mo')
    
    if 'error' in stock_data:
        print(f"ERROR: {stock_data['error']}")
        return False
    
    hybrid = HybridStockPredictor(APP_DATA_DIR, 'trading')
    
    analysis = hybrid.predict(stock_data, history_data, None)
    
    if 'error' in analysis:
        print(f"ERROR: {analysis['error']}")
        return False
    
    recommendation = analysis.get('recommendation', {})
    method = analysis.get('method', 'unknown')
    ml_pred = analysis.get('ml_prediction', {})
    rule_pred = analysis.get('rule_prediction', {})
    ensemble_weights = analysis.get('ensemble_weights', {})
    
    print(f"Method: {method}")
    print(f"Final action: {recommendation.get('action')}")
    print(f"Final confidence: {recommendation.get('confidence', 0):.1f}%")
    print(f"\nRule-based:")
    print(f"  Action: {rule_pred.get('action', 'N/A')}")
    print(f"  Confidence: {rule_pred.get('confidence', 0):.1f}%")
    print(f"\nML Model:")
    print(f"  Action: {ml_pred.get('action', 'N/A')}")
    print(f"  Confidence: {ml_pred.get('confidence', 0):.1f}%")
    print(f"  Probabilities: {ml_pred.get('probabilities', {})}")
    print(f"\nEnsemble weights:")
    print(f"  Rule: {ensemble_weights.get('rule', 0):.2f}")
    print(f"  ML: {ensemble_weights.get('ml', 0):.2f}")
    
    return True

def test_binary_conversion():
    """Test binary classification conversion"""
    print("\n" + "="*60)
    print("TEST 3: BINARY CLASSIFICATION CONVERSION")
    print("="*60)
    
    ml = StockPredictionML(APP_DATA_DIR, 'trading')
    
    if not ml.is_trained():
        print("Model not trained - cannot test")
        return False
    
    # Check metadata
    is_binary = ml.metadata and ml.metadata.get('use_binary_classification', False)
    classes = ml.metadata.get('label_classes', []) if ml.metadata else []
    
    print(f"Binary mode in metadata: {is_binary}")
    print(f"Label classes: {classes}")
    
    # Test with dummy features
    feature_extractor = FeatureExtractor()
    dummy_features = np.zeros(143)
    
    # Make multiple predictions to see distribution
    predictions = []
    for i in range(10):
        pred = ml.predict(dummy_features + np.random.normal(0, 0.01, 143))
        predictions.append(pred.get('action'))
    
    print(f"\nSample predictions (10 random feature vectors):")
    from collections import Counter
    pred_counts = Counter(predictions)
    for action, count in pred_counts.items():
        print(f"  {action}: {count}")
    
    # Check if we see UP/DOWN (shouldn't - should be converted to BUY/SELL)
    if 'UP' in predictions or 'DOWN' in predictions:
        print("\n⚠️ WARNING: Found UP/DOWN in predictions - conversion not working!")
        return False
    else:
        print("\n✓ Conversion working - only BUY/SELL/HOLD found")
        return True

def main():
    print("="*60)
    print("BACKTESTING DIAGNOSTICS")
    print("="*60)
    
    results = {
        'ML Model': test_ml_model(),
        'Hybrid Predictor': test_hybrid_predictor(),
        'Binary Conversion': test_binary_conversion(),
    }
    
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY")
    print("="*60)
    for test, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test}: {status}")
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    
    if not results['ML Model']:
        print("  • ML model is not trained - train it first")
    
    if not results['Binary Conversion']:
        print("  • Binary conversion is not working - check predict() method")
    
    if results['ML Model'] and results['Hybrid Predictor']:
        print("  • Both ML model and hybrid predictor are working")
        print("  • If backtesting accuracy is still low, check:")
        print("    - Ensemble weights might favor rule-based too much")
        print("    - Evaluation logic might still have issues")
        print("    - Data/timing mismatch between training and backtesting")
    
    print("="*60)

if __name__ == '__main__':
    main()
