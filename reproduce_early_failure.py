
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
TEST_DIR = Path("test_early_failure")
if TEST_DIR.exists(): shutil.rmtree(TEST_DIR)
TEST_DIR.mkdir()

# Mock config - ensures we can import predictions_tracker
import sys
if 'config' in sys.modules: del sys.modules['config']
from types import ModuleType
config = ModuleType('config')
config.PREDICTIONS_FILE = TEST_DIR / "predictions.json"
config.STATE_FILE = TEST_DIR / "state.json"
config.MIN_PROFIT_TARGET_PCT = 3.0
sys.modules['config'] = config

from predictions_tracker import PredictionsTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("EarlyFailureTest")

def run_test():
    tracker = PredictionsTracker(TEST_DIR)
    
    # 1. Create a BUY prediction (Active)
    # Entry: 100, Target: 110 (+10%), Stop: 95 (-5%)
    # Expiry: 10 days from now
    logger.info("--- Creating Test Prediction ---")
    pred = tracker.add_prediction(
        symbol="TEST_FAIL",
        strategy="trading",
        action="BUY",
        entry_price=100.0,
        target_price=110.0,
        stop_loss=95.0,
        confidence=80.0,
        reasoning="Test Early Failure"
    )
    pred_id = pred['id']
    logger.info(f"Created prediction #{pred_id}: Entry 100, Target 110, Stop 95")
    
    # Manually ensure timestamp is old enough for grace period (Default > 15 mins)
    with tracker.lock:
        tracker.predictions[0]['timestamp'] = (datetime.now() - timedelta(minutes=30)).isoformat()
        tracker.save()

    # 2. Simulate Price Drop to Stop Loss (94.0)
    # This represents a -6% drop, hitting the -5% stop loss.
    # Current Behavior: Should marked as FALSE (Failed)
    # Desired Behavior: Should remain ACTIVE (Ignore Stop Loss, Wait for Expiry)
    
    logger.info("\n--- Simulating Price Drop to 94.0 (-6%) ---")
    tracker.verify_prediction(pred_id, current_price=94.0)
    
    updated_pred = tracker.get_prediction_by_id(pred_id)
    status = updated_pred['status']
    was_correct = updated_pred.get('was_correct')
    
    logger.info(f"Post-Simulation Status: {status}")
    logger.info(f"Was Correct: {was_correct}")
    
    if status == 'active':
        logger.info("✅ reproduction: Prediction remained ACTIVE (Correct Patient Sniper Behavior)")
        
        # 3. Now Simulate Expiry with price still low
        logger.info("\n--- Simulating Expiry (Time Travel +15 days) ---")
        
        # Manually expire it
        with tracker.lock:
             # Set timestamp to 20 days ago (estimated days is 10)
             tracker.predictions[0]['timestamp'] = (datetime.now() - timedelta(days=20)).isoformat()
             tracker.predictions[0]['estimated_target_date'] = (datetime.now() - timedelta(days=5)).isoformat()
             tracker.save()
             
        # Verify again with low price
        tracker.verify_prediction(pred_id, current_price=94.0)
        
        final_pred = tracker.get_prediction_by_id(pred_id)
        final_status = final_pred['status']
        final_correct = final_pred.get('was_correct')
        
        logger.info(f"Post-Expiry Status: {final_status}")
        logger.info(f"Was Correct: {final_correct}")
        
        if final_status == 'verified' and final_correct is False:
             logger.info("✅ Expiry Test: Prediction FAILED at expiry (Correct)")
             print("RESULT: SUCCESS_FULL_CYCLE")
        else:
             logger.error(f"❌ Expiry Test: Prediction state unexpected: {final_status}, {final_correct}")
             print("RESULT: FAILED_EXPIRY")

    elif status == 'verified' and was_correct is False:
        logger.info("❌ reproduction: Prediction FAILED immediately on Stop Loss hit (Old Behavior)")
        print("RESULT: FAILED_EARLY")
    else:
        logger.info(f"❓ Unexpected state: {status}, {was_correct}")
        print(f"RESULT: {status}")

if __name__ == "__main__":
    run_test()
