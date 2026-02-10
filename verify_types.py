import numpy as np
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from trend_change_predictor import TrendChangePredictor
from hybrid_predictor import HybridStockPredictor

def verify_predictor_types():
    print("--- Verifying TrendChangePredictor Types ---")
    tcp = TrendChangePredictor()
    
    # Create history with some volatility to trigger patterns
    dates = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
    prices = [100 - i/10 for i in range(50)] + [95 + i/10 for i in range(50)]
    
    history_data = {
        'data': [
            {'date': d.isoformat(), 'close': np.float64(p), 'high': np.float64(p+1), 'low': np.float64(p-1)} 
            for d, p in zip(dates, prices)
        ]
    }
    
    indicators = {
        'rsi': np.float64(28.5),
        'ema_20': np.float64(96.0),
        'ema_50': np.float64(100.0),
        'macd': np.float64(-1.5),
        'macd_signal': np.float64(-1.4),
        'macd_diff': np.float64(-0.1)
    }
    
    price_action = {
        'trend': 'downtrend',
        'support': np.float64(94.0),
        'resistance': np.float64(110.0)
    }
    
    predictions = tcp.predict_trend_changes("TEST", history_data, indicators, price_action)
    
    for pred in predictions:
        try:
            json.dumps(pred)
        except TypeError as e:
            print(f"  ❌ JSON serialization FAILED: {e}")

    print("\n--- Verifying HybridStockPredictor Types & UnboundLocalError ---")
    predictor = HybridStockPredictor(strategy="trading", data_dir=Path("./test_data"))
    
    # 1. Test _apply_learned_weights (BUY)
    base_analysis = {
        'recommendation': {
            'action': 'BUY',
            'confidence': np.float64(82.5),
            'entry_price': np.float64(150.0),
            'target_price': np.float64(160.0),
            'stop_loss': np.float64(145.0),
            'estimated_days': np.int64(12)
        }
    }
    
    result = predictor._apply_learned_weights(base_analysis)
    print(f"Learned Weights Result (BUY):")
    print(f"  confidence type: {type(result['confidence'])}")
    print(f"  estimated_days type: {type(result['estimated_days'])}")
    
    # 2. Test _ensemble_predictions with HOLD (Fixing the UnboundLocalError)
    print("Testing Ensemble with HOLD signal...")
    rule_pred = {'action': 'HOLD', 'confidence': 50, 'target_price': 100, 'stop_loss': 90, 'estimated_days': 7}
    ml_pred = None
    stock_data = {'symbol': 'TEST', 'price': 100}
    
    history_data_df = {
        'data': [
            {'close': 100.0, 'high': 101.0, 'low': 99.0} for _ in range(50)
        ]
    }
    
    weights = {'rule_weight': 0.7, 'ml_weight': 0.3}
    
    base_analysis_hold = {
        'recommendation': rule_pred,
        'indicators': {'atr': 1.5, 'price_above_ema200': True, 'rsi': 50},
        'market_regime': {'regime': 'sideways', 'confidence': 0.8},
        'volatility': {'atr_percent': 1.5}
    }
    
    try:
        # Signature: (rule_pred, ml_pred, weights, base_analysis, stock_data, history_data, is_backtest=False)
        ensemble_result = predictor._ensemble_predictions(
            rule_pred, ml_pred, weights, base_analysis_hold, stock_data, history_data_df
        )
        print("  ✅ Ensemble with HOLD successful (No UnboundLocalError)")
        print(f"  Result estimated_days: {ensemble_result['recommendation']['estimated_days']}")
    except Exception as e:
        import traceback
        print(f"  ❌ Ensemble with HOLD FAILED: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    verify_predictor_types()
