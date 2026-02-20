
import sys
import os
import pandas as pd
import numpy as np
import logging

try:
    from ml_training import StockPredictionML
    print("Successfully imported StockPredictionML")
except ImportError as e:
    print(f"Error importing ml_training: {e}")
    sys.exit(1)

def test_feature_extraction():
    print("\n--- Testing Feature Extraction for RSI ---")
    
    # Create dummy data
    dates = pd.date_range(end=pd.Timestamp.now(), periods=100)
    data = {
        'date': dates,
        'open': np.linspace(100, 110, 100),
        'high': np.linspace(101, 111, 100),
        'low': np.linspace(99, 109, 100),
        'close': np.linspace(100, 110, 100), # Constant uptrend
        'volume': np.random.randint(1000, 2000, 100)
    }
    df = pd.DataFrame(data)
    
    class MockStorage:
        def __init__(self, server_name=None):
            self.data_dir = 'mock_data'
        def model_exists(self, name): return False
        def load_model(self, name): return None
        def save_model(self, model, name): pass
        
    try:
        if not os.path.exists('mock_data'):
            os.makedirs('mock_data')
            
        print("Initializing StockPredictionML...")
        ml = StockPredictionML('test_strategy', MockStorage())
        
        if hasattr(ml, '_extract_time_series_features'):
            print("Found _extract_time_series_features method")
            features = ml._extract_time_series_features(df)
            
            print(f"\nExtracted {len(features)} time-series features.")
            found_rsi = False
            for key in features.keys():
                if 'rsi' in key.lower():
                    print(f"✅ FOUND RSI FEATURE: {key} = {features[key]}")
                    found_rsi = True
            
            if not found_rsi:
                print("\n❌ FAILURE: NO RSI FEATURES FOUND in extraction output!")
                
        else:
            print("❌ _extract_time_series_features method missing!")
            
    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_feature_extraction()
