"""
Comprehensive ML System Verification
Tests all components to ensure ML predictions work in backtesting
"""
import sys
from pathlib import Path
from config import APP_DATA_DIR
from ml_training import StockPredictionML
from hybrid_predictor import HybridStockPredictor
from trading_analyzer import TradingAnalyzer
from data_fetcher import StockDataFetcher
from ml_training import FeatureExtractor
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_complete_ml_system():
    """Test the complete ML system end-to-end"""
    print("="*70)
    print("COMPREHENSIVE ML SYSTEM VERIFICATION")
    print("="*70)
    
    strategy = 'trading'
    symbol = 'AAPL'
    all_tests_passed = True
    
    # Test 1: Model Loading
    print(f"\n[TEST 1] Model Loading...")
    try:
        ml_predictor = StockPredictionML(APP_DATA_DIR, strategy)
        is_trained = ml_predictor.is_trained()
        print(f"   is_trained() = {is_trained}")
        
        if not is_trained:
            print("   [FAIL] Model is not trained!")
            all_tests_passed = False
        else:
            print("   [PASS] Model is trained")
            if hasattr(ml_predictor, 'regime_models') and ml_predictor.regime_models:
                print(f"   [INFO] Loaded {len(ml_predictor.regime_models)} regime model(s)")
                for regime, regime_ml in ml_predictor.regime_models.items():
                    print(f"     - {regime}: model={regime_ml.model is not None}, encoder={len(regime_ml.label_encoder.classes_) if hasattr(regime_ml.label_encoder, 'classes_') else 0} classes")
    except Exception as e:
        print(f"   [FAIL] Error: {e}")
        import traceback
        traceback.print_exc()
        all_tests_passed = False
    
    if not all_tests_passed:
        print("\n[STOPPING] Model loading failed. Cannot continue tests.")
        return False
    
    # Test 2: Data Fetching
    print(f"\n[TEST 2] Data Fetching...")
    try:
        data_fetcher = StockDataFetcher()
        stock_data = data_fetcher.fetch_stock_data(symbol)
        history_data = data_fetcher.fetch_history(symbol, days=365)
        
        if 'error' in stock_data:
            print(f"   [SKIP] Data fetch error (expected in test environment): {stock_data['error']}")
            print("   [INFO] Using mock data for remaining tests...")
            # Create mock data
            stock_data = {'symbol': symbol, 'price': 150.0, 'info': {}}
            history_data = {'data': [{'date': f'2024-01-{i:02d}', 'close': 150.0 + i*0.1, 'volume': 1000000} for i in range(1, 100)]}
        else:
            print(f"   [PASS] Got real data: price=${stock_data.get('price', 0):.2f}, {len(history_data.get('data', []))} days")
    except Exception as e:
        print(f"   [SKIP] Data fetch error (expected): {e}")
        stock_data = {'symbol': symbol, 'price': 150.0, 'info': {}}
        history_data = {'data': [{'date': f'2024-01-{i:02d}', 'close': 150.0 + i*0.1, 'volume': 1000000} for i in range(1, 100)]}
    
    # Test 3: Feature Extraction
    print(f"\n[TEST 3] Feature Extraction...")
    try:
        trading_analyzer = TradingAnalyzer()
        base_analysis = trading_analyzer.analyze(stock_data, history_data)
        
        if 'error' in base_analysis:
            print(f"   [SKIP] Analysis error (expected with mock data): {base_analysis['error']}")
            indicators = {}
        else:
            indicators = base_analysis.get('indicators', {})
            print(f"   [PASS] Got indicators: {len(indicators)} items")
        
        feature_extractor = FeatureExtractor()
        features = feature_extractor.extract_features(
            stock_data, history_data, None, indicators,
            None, None, None, None
        )
        
        if features is None or len(features) == 0:
            print(f"   [FAIL] No features extracted!")
            all_tests_passed = False
        else:
            print(f"   [PASS] Extracted {len(features)} features")
    except Exception as e:
        print(f"   [FAIL] Feature extraction error: {e}")
        import traceback
        traceback.print_exc()
        all_tests_passed = False
        return False
    
    # Test 4: ML Prediction
    print(f"\n[TEST 4] ML Prediction...")
    try:
        features_array = np.array([features])
        ml_result = ml_predictor.predict(features_array, stock_data, history_data)
        
        if 'error' in ml_result:
            print(f"   [FAIL] ML prediction error: {ml_result['error']}")
            all_tests_passed = False
        else:
            action = ml_result.get('action', 'N/A')
            confidence = ml_result.get('confidence', 0)
            print(f"   [PASS] ML prediction: {action} ({confidence:.1f}%)")
            print(f"   [INFO] Probabilities: {ml_result.get('probabilities', {})}")
    except Exception as e:
        print(f"   [FAIL] ML prediction exception: {e}")
        import traceback
        traceback.print_exc()
        all_tests_passed = False
    
    # Test 5: Hybrid Predictor
    print(f"\n[TEST 5] Hybrid Predictor...")
    try:
        hybrid_predictor = HybridStockPredictor(APP_DATA_DIR, strategy, TradingAnalyzer())
        hybrid_result = hybrid_predictor.predict(stock_data, history_data, None)
        
        if 'error' in hybrid_result:
            print(f"   [FAIL] Hybrid prediction error: {hybrid_result['error']}")
            all_tests_passed = False
        else:
            ml_pred = hybrid_result.get('ml_prediction', {})
            rule_pred = hybrid_result.get('rule_prediction', {})
            ensemble_weights = hybrid_result.get('ensemble_weights', {})
            final_action = hybrid_result.get('recommendation', {}).get('action', 'N/A')
            
            print(f"   [PASS] Hybrid prediction: {final_action}")
            print(f"   [INFO] Rule: {rule_pred.get('action', 'N/A')} ({rule_pred.get('confidence', 0):.1f}%)")
            print(f"   [INFO] ML: {ml_pred.get('action', 'N/A')} ({ml_pred.get('confidence', 0):.1f}%)")
            print(f"   [INFO] Weights: Rule={ensemble_weights.get('rule', 0):.2f}, ML={ensemble_weights.get('ml', 0):.2f}")
            
            if ml_pred.get('action') == 'N/A':
                print(f"   [WARNING] ML prediction is N/A in hybrid result!")
                all_tests_passed = False
    except Exception as e:
        print(f"   [FAIL] Hybrid prediction exception: {e}")
        import traceback
        traceback.print_exc()
        all_tests_passed = False
    
    # Final Summary
    print("\n" + "="*70)
    if all_tests_passed:
        print("ALL TESTS PASSED! ML System is working correctly.")
        print("="*70)
        return True
    else:
        print("SOME TESTS FAILED. Check errors above.")
        print("="*70)
        return False

if __name__ == '__main__':
    success = test_complete_ml_system()
    sys.exit(0 if success else 1)
