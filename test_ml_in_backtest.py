"""
Test ML predictions in backtesting context
This script tests if ML predictions are working correctly
"""
import sys
from pathlib import Path
from config import APP_DATA_DIR
from ml_training import StockPredictionML
from hybrid_predictor import HybridStockPredictor
from trading_analyzer import TradingAnalyzer
from data_fetcher import StockDataFetcher
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ml_prediction():
    """Test if ML predictions work"""
    print("="*60)
    print("TESTING ML PREDICTIONS")
    print("="*60)
    
    # Initialize components
    strategy = 'trading'
    ml_predictor = StockPredictionML(APP_DATA_DIR, strategy)
    hybrid_predictor = HybridStockPredictor(APP_DATA_DIR, strategy, TradingAnalyzer())
    data_fetcher = StockDataFetcher()
    
    # Test symbol
    symbol = 'AAPL'
    
    print(f"\n1. Checking if ML model is trained...")
    is_trained = ml_predictor.is_trained()
    print(f"   is_trained() = {is_trained}")
    
    if not is_trained:
        print("   ERROR: ML model is not trained!")
        print(f"   - model is None: {ml_predictor.model is None}")
        print(f"   - regime_models: {len(ml_predictor.regime_models) if hasattr(ml_predictor, 'regime_models') else 'N/A'}")
        if hasattr(ml_predictor, 'regime_models') and ml_predictor.regime_models:
            for regime, regime_ml in ml_predictor.regime_models.items():
                print(f"     - {regime}: model={regime_ml.model is not None}, encoder={hasattr(regime_ml.label_encoder, 'classes_')}")
        if hasattr(ml_predictor, 'label_encoder'):
            print(f"   - label_encoder has classes: {hasattr(ml_predictor.label_encoder, 'classes_')}")
            if hasattr(ml_predictor.label_encoder, 'classes_'):
                print(f"   - classes: {list(ml_predictor.label_encoder.classes_)}")
        return False
    
    print("   [OK] ML model is trained")
    
    print(f"\n2. Fetching data for {symbol}...")
    try:
        stock_data = data_fetcher.fetch_stock_data(symbol)
        history_data = data_fetcher.fetch_history(symbol, days=365)
        
        if 'error' in stock_data:
            print(f"   ERROR: {stock_data['error']}")
            return False
        
        print(f"   ✓ Got stock data: price=${stock_data.get('price', 0):.2f}")
        print(f"   ✓ Got history: {len(history_data.get('data', []))} days")
    except Exception as e:
        print(f"   ERROR fetching data: {e}")
        return False
    
    print(f"\n3. Testing ML prediction directly...")
    try:
        from ml_training import FeatureExtractor
        feature_extractor = FeatureExtractor()
        
        # Get base analysis for indicators
        trading_analyzer = TradingAnalyzer()
        base_analysis = trading_analyzer.analyze(stock_data, history_data)
        
        if 'error' in base_analysis:
            print(f"   ERROR in base analysis: {base_analysis['error']}")
            return False
        
        indicators = base_analysis.get('indicators', {})
        features = feature_extractor.extract_features(
            stock_data, history_data, None, indicators,
            base_analysis.get('market_regime'), None, None, None
        )
        
        ml_result = ml_predictor.predict(features, stock_data, history_data)
        
        if 'error' in ml_result:
            print(f"   ERROR in ML prediction: {ml_result['error']}")
            return False
        
        print(f"   ✓ ML prediction: {ml_result.get('action')} ({ml_result.get('confidence', 0):.1f}%)")
        print(f"   ✓ Probabilities: {ml_result.get('probabilities', {})}")
    except Exception as e:
        print(f"   ERROR in ML prediction: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print(f"\n4. Testing hybrid predictor...")
    try:
        hybrid_result = hybrid_predictor.predict(stock_data, history_data, None)
        
        if 'error' in hybrid_result:
            print(f"   ERROR in hybrid prediction: {hybrid_result['error']}")
            return False
        
        ml_pred = hybrid_result.get('ml_prediction', {})
        rule_pred = hybrid_result.get('rule_prediction', {})
        ensemble_weights = hybrid_result.get('ensemble_weights', {})
        
        print(f"   ✓ Final prediction: {hybrid_result.get('recommendation', {}).get('action')}")
        print(f"   ✓ Rule prediction: {rule_pred.get('action', 'N/A')} ({rule_pred.get('confidence', 0):.1f}%)")
        print(f"   ✓ ML prediction: {ml_pred.get('action', 'N/A')} ({ml_pred.get('confidence', 0):.1f}%)")
        print(f"   ✓ Ensemble weights: Rule={ensemble_weights.get('rule', 0):.2f}, ML={ensemble_weights.get('ml', 0):.2f}")
        
        if ml_pred.get('action') == 'N/A':
            print("   ERROR: ML prediction is N/A in hybrid result!")
            return False
        
    except Exception as e:
        print(f"   ERROR in hybrid prediction: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    return True

if __name__ == '__main__':
    success = test_ml_prediction()
    sys.exit(0 if success else 1)
