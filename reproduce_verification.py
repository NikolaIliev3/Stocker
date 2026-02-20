
import logging
from pathlib import Path
from datetime import datetime, timedelta
import shutil
import json
import time

# Mock dependencies
class MockStorage:
    def __init__(self, filepath):
        self.filepath = filepath
    def load(self, default=None):
        if not self.filepath.exists(): return default or {}
        with open(self.filepath, 'r') as f: return json.load(f)
    def write_locked(self, data):
        with open(self.filepath, 'w') as f: json.dump(data, f)

# Setup environment
TEST_DIR = Path("test_verification_fix")
if TEST_DIR.exists(): shutil.rmtree(TEST_DIR)
TEST_DIR.mkdir()

# Mock config
import sys
# Create a dummy config module if it doesn't exist in sys.modules
if 'config' not in sys.modules:
    from types import ModuleType
    config = ModuleType('config')
    config.PREDICTIONS_FILE = TEST_DIR / "predictions.json"
    config.STATE_FILE = TEST_DIR / "state.json"
    config.MIN_PROFIT_TARGET_PCT = 3.0
    sys.modules['config'] = config
else:
    # Patch existing config
    import config
    config.PREDICTIONS_FILE = TEST_DIR / "predictions.json"
    config.STATE_FILE = TEST_DIR / "state.json"

# Import PredictionsTracker (assuming it's in the python path or current dir)
# We need to make sure we import the *actual* file we want to test
import predictions_tracker
from predictions_tracker import PredictionsTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerificationTest")

def run_test():
    tracker = PredictionsTracker(TEST_DIR)
    
    # 1. Create a BUY prediction 2 days ago
    # Entry: 100, Target: 110 (+10%), Stop: 95 (-5%)
    # Current Price: 105 (No hit)
    # History: Yesterday High was 112 (Target Hit!)
    
    past_date = (datetime.now() - timedelta(days=2)).isoformat()
    
    pred_id = tracker.add_prediction(
        symbol="TEST",
        strategy="trading",
        action="BUY",
        entry_price=100.0,
        target_price=110.0,
        stop_loss=95.0,
        confidence=80.0,
        reasoning="Test"
    )['id']
    
    # Manually backdate the prediction timestamp
    with tracker.lock:
        tracker.predictions[0]['timestamp'] = past_date
        tracker.save()
        
    logger.info(f"Created retroactive prediction: {tracker.predictions[0]}")
    
    # 2. Verify with CURRENT logic (Should FAIL/Wait)
    # Current price 105 is between 95 and 110.
    logger.info("--- Testing Current Price Verification (Expect WAIT) ---")
    tracker.verify_prediction(pred_id, current_price=105.0)
    
    pred = tracker.get_prediction_by_id(pred_id)
    if pred['status'] == 'verified':
        logger.error("❌ Pre-fix logic FAILED: Verified prematurely!")
    else:
        logger.info("✅ Pre-fix logic correctly waited (status: active)")
        
    # 3. Verify with NEW logic (Retroactive History) - Simulated
    # We will manually pass high/low to simulate the fix being present
    # In the actual fix, verify_all_active_predictions will calculate these
    
    logger.info("--- Testing Retroactive Verification (Expect SUCCESS) ---")
    
    # Simulate data fetcher finding a high of 112 yesterday
    high_since_entry = 112.0 
    low_since_entry = 98.0
    
    # Call verify_prediction with the overrides (this supports the plan)
    try:
        tracker.verify_prediction(
            pred_id, 
            current_price=105.0, 
            high=high_since_entry, 
            low=low_since_entry
        )
        
        pred = tracker.get_prediction_by_id(pred_id)
        if pred['status'] == 'verified' and pred['was_correct']:
            logger.info("✅ Retroactive verification SUCCESS!")
        else:
            logger.error(f"❌ Retroactive verification FAILED. Status: {pred['status']}")
            
    except TypeError as e:
        logger.error(f"⚠️ verify_prediction failed with TypeError: {e}")

if __name__ == "__main__":
    run_test()
