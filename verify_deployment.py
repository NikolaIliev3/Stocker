import sys
from pathlib import Path
from hybrid_predictor import HybridStockPredictor

def verify_config_2():
    data_dir = Path.home() / ".stocker"
    
    # Force strategy='trading' for testing
    predictor = HybridStockPredictor(data_dir, 'trading')
    
    # Mock data
    stock_data = {'symbol': 'AAPL', 'price': 150.0}
    # Mock minimal history for predictor to work
    history_data = {
        'data': [{'date': '2026-01-01', 'close': 140.0, 'high': 145.0, 'low': 135.0, 'volume': 1000000}] * 50
    }
    
    # Check weights
    weights = predictor._calculate_dynamic_weights({}, {})
    print(f"Actual Weights: {weights}")
    
    if weights['rule'] == 1.0 and weights['ml'] == 0.0:
        print("✅ SUCCESS: Config 2 (Rules-Only) is ACTIVE.")
    else:
        print(f"❌ FAILURE: Unexpected weights {weights}")

if __name__ == "__main__":
    verify_config_2()
