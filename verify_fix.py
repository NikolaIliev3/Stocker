
import sys
from pathlib import Path
import logging
import numpy as np
import warnings

# Add path
sys.path.append(str(Path.cwd()))

from ml_training import StockPredictionML, FeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO)

# Load model
print("--- Loading Model ---")
ml = StockPredictionML(Path.home()/'.stocker', 'trading')
print(f"Regime Models: {list(ml.regime_models.keys())}")
print(f"Main Model Loaded: {ml.model is not None}")

# Check Bull model metadata
if 'bull' in ml.regime_models:
    bull = ml.regime_models['bull']
    print(f"Bull Model Samples: {bull.metadata.get('samples', 'N/A')}")
    # Force loading selected features if not loaded (the fix handles this, but let's see)
    print(f"Bull Model Selected Features: {len(bull.selected_features) if bull.selected_features is not None else 0}")
    
    # Predict with Diverse Inputs
    print("\n--- Testing Predictions ---")
    fe = FeatureExtractor()
    
    # Case 1: Bullish input
    ind_bull = {
        'rsi': 30, 'mfi': 20, 'macd': 0.5, 'adx': 30, 'ema_20_above_50': True,
        'roc_5': 5.0, 'momentum_10': 10.0,
        'volume_ratio': 2.0
    }
    
    # Case 2: Bearish input
    ind_bear = {
        'rsi': 85, 'mfi': 90, 'macd': -0.5, 'adx': 25, 'ema_20_above_50': False,
        'roc_5': -5.0, 'momentum_10': -10.0,
        'volume_ratio': 0.5
    }
    
    stock_data = {'symbol': 'TEST', 'current_price': 100}
    hist_data = {'data': [{'date': '2025-01-01', 'close': 100, 'volume': 1000} for _ in range(60)]}
    
    # Extract features
    # Passing 'indicators' logic relies on FeatureExtractor implementation to respect them
    # We pass None for history because features are calculated from arguments or history
    # Wait, extract_features prioritizes calculated indicators if passed?
    # Yes, lines 376 in training_pipeline show passing indicators.
    
    f_bull = fe.extract_features(stock_data, hist_data, None, ind_bull, None, None, None, None, None, None, None, {'volume_ratio': 2.0})
    f_bear = fe.extract_features(stock_data, hist_data, None, ind_bear, None, None, None, None, None, None, None, {'volume_ratio': 0.5})
    
    # Reshape for prediction
    f_bull_reshaped = f_bull.reshape(1, -1)
    f_bear_reshaped = f_bear.reshape(1, -1)
    
    # Predict
    p_bull = bull.predict(f_bull_reshaped)
    p_bear = bull.predict(f_bear_reshaped)
    
    print(f"Bullish Input -> Action: {p_bull['action']}, Conf: {p_bull['confidence']:.4f}%")
    print(f"Bearish Input -> Action: {p_bear['action']}, Conf: {p_bear['confidence']:.4f}%")
    
    if abs(p_bull['confidence'] - p_bear['confidence']) > 0.001:
        print("\nSUCCESS: Confidence scores are different!")
    else:
        print("\nFAILURE: Confidence scores are identical.")

else:
    print("FAILURE: Bull model not loaded.")
