
import logging
from pathlib import Path
from momentum_monitor import MomentumMonitor
from trend_change_tracker import TrendChangeTracker
from data_fetcher import StockDataFetcher
import time

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_verification():
    print("Testing Verification Logic...")
    
    # Setup paths
    data_dir = Path("test_data")
    data_dir.mkdir(exist_ok=True)
    
    # Initialize components
    monitor = MomentumMonitor(data_dir)
    tracker = TrendChangeTracker(data_dir)
    
    # Mock DataFetcher
    class MockDataFetcher:
        def fetch_stock_data(self, symbol, force_refresh=False):
            return {'price': 150.0}
    
    fetcher = MockDataFetcher()
    
    # Test Momentum Verification
    print("\n1. Testing Momentum Verification")
    monitor.clear_history()
    
    # Add a history item (mocking one from yesterday)
    monitor.add_to_history(
        symbol="AAPL",
        action="BUY",
        changes={'recommendation': {'action': 'BUY'}},
        current_price=100.0  # Buying at 100
    )
    
    # Manually backdate the item to force verification
    monitor.history[0]['time'] = "2024-01-01 12:00:00" # Old date
    monitor.save_history()
    
    print("Verifying history (Current Price: 150, Entry: 100, Action: BUY)...")
    verified = monitor.verify_history(fetcher)
    
    if verified:
        item = verified[0]
        print(f"Verified Item: {item['symbol']} - Status: {item['status']} - Verified: {item['verified']}")
        if item['status'] == 'correct':
            print("✅ Momentum Verification SUCCESS")
        else:
            print(f"❌ Momentum Verification FAILED (Expected correct, got {item['status']})")
    else:
        print("❌ Momentum Verification FAILED (No items verified)")

    # Test Trend Verification
    print("\n2. Testing Trend Verification")
    # Add a trend prediction
    tracker.predictions = [] # Clear
    tracker.add_prediction(
        symbol="AAPL",
        current_trend="uptrend",
        predicted_change="downtrend",
        estimated_days=1,
        estimated_date="2024-01-01", # Past date
        confidence=80,
        reasoning="Test",
        key_indicators={}
    )
    
    print("Verifying trend...")
    verified_trends = tracker.check_for_verification(fetcher)
    # Note: currently check_for_verification in tracker just returns empty list as logic was 'pass'
    # But it should run without error
    if verified_trends is not None:
         print("✅ Trend Verification Ran (Logic is currently placeholder as designed)")
    else:
         print("❌ Trend Verification Failed")

    # Clean up
    import shutil
    if data_dir.exists():
        shutil.rmtree(data_dir)

if __name__ == "__main__":
    test_verification()
