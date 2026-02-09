
import logging
import sys
import os
from datetime import datetime
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('Verification')

from config import APP_DATA_DIR
from main import StockerApp
import tkinter as tk

def run_verification():
    logger.info("🧪 STARTING OUT-OF-SAMPLE VERIFICATION (Jan 2026)")
    
    # Headless App
    root = tk.Tk()
    root.withdraw()
    app = StockerApp(root)
    if app.ml_scheduler: app.ml_scheduler.stop()
    if app.auto_learner: app.auto_learner.stop()
    
    symbols = ['AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA']
    test_dates = [
        "2026-01-05", "2026-01-12", "2026-01-20"
    ]
    
    backtester = app.backtester
    predictor = app.hybrid_predictors['trading']
    
    results = []
    
    for symbol in symbols:
        for date_str in test_dates:
            try:
                test_date = datetime.strptime(date_str, "%Y-%m-%d")
                logger.info(f"Testing {symbol} on {date_str}...")
                
                # We need to manually trigger the backtest point
                # Since backtester usually picks random dates, we use _test_historical_point directly
                # Note: We need to respect the updated signature or logic
                
                res = backtester._test_historical_point(
                    symbol, test_date, "trading", predictor, test_number=len(results)
                )
                
                if res:
                    results.append(res)
                    logger.info(f"   -> Prediction: {res['predicted_action']} | Correct: {res['was_correct']}")
                else:
                    logger.warning(f"   -> No prediction generated (Insufficient data or filtered)")
            
            except Exception as e:
                logger.error(f"Error testing {symbol} on {date_str}: {e}")

    # Summary
    if results:
        df = pd.DataFrame(results)
        accuracy = df['was_correct'].mean() * 100
        logger.info("\n📊 VERIFICATION RESULTS (Jan-Feb 2026):")
        logger.info(f"   Total Tests: {len(results)}")
        logger.info(f"   Accuracy:    {accuracy:.1f}%")
        
        if accuracy == 100.0 and len(results) > 5:
            logger.warning("⚠️ WARNING: Accuracy is 100%. Check manually if this is luck or leakage.")
        elif accuracy > 55.0:
            logger.info("✅ PASSED: Performance is realistic and positive.")
        else:
            logger.info("⚠️ NOTE: Performance is low. This is expected for out-of-sample testing without tuning.")
    else:
        logger.error("❌ No results generated.")

if __name__ == "__main__":
    run_verification()
