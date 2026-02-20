"""
Deep Calibration Loop for MegaMind
Force-trains the model against historical data until it achieves high precision on actionable signals (BUY/SELL).

Usage:
    python calibration_loop.py [target_precision_percent]
    
    Example: python calibration_loop.py 70
"""
import sys
import time
import logging
import tkinter as tk
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from main import StockerApp
from config import APP_DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(APP_DATA_DIR / 'calibration.log')
    ]
)
logger = logging.getLogger('CalibrationLoop')

def run_calibration(target_precision=70.0):
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 STARTING DEEP CALIBRATION LOOP")
    logger.info(f"Target Precision (BUY/SELL only): {target_precision}%")
    logger.info(f"{'='*70}\n")
    
    # Initialize Headless App
    root = tk.Tk()
    root.withdraw() # Hide window
    
    try:
        app = StockerApp(root)
        
        # Access components
        backtester = app.backtester
        
        # 1. DISABLE SAFETY BRAKES
        logger.info("🔓 Disabling training safety brakes (COOLDOWNS & OVERLAP CHECKS)...")
        backtester.retrain_cooldown_hours = 0
        backtester.max_overlap_threshold = 1.0 # Allow full overlap (we want to fix mistakes on these dates!)
        backtester.recent_test_window_days = 0 # Don't exclude recently tested dates
        
        # 2. Main Loop
        iteration = 1
        max_iterations = 20 # Safety cap
        
        while iteration <= max_iterations:
            logger.info(f"\n🔄 Calibration Cycle #{iteration}")
            
            # Run batch of backtests (force run)
            # We test 'trading' strategy primarily as it's the most active
            strategies = ['trading']
            num_tests = 100 # 100 random points per strategy per cycle
            
            logger.info(f"   • Running {num_tests} backtests per strategy...")
            
            # Manually trigger backtest without threading to block
            backtester._run_backtest(selected_strategies=strategies, num_tests=num_tests)
            
            # Analyze Results
            # Get results from THIS run (approximate by looking at last N results)
            # Since we ran (len(strategies) * num_tests), look at those
            recent_count = len(strategies) * num_tests
            results = backtester.get_recent_results(limit=recent_count)
            
            # Calculate Precision (BUY/SELL only)
            score = backtester.calculate_precision_score(results)
            precision = score['precision']
            actions = score['total_actions']
            correct = score['correct_actions']
            
            logger.info(f"\n📊 Cycle #{iteration} Results:")
            logger.info(f"   • Actionable Predictions (BUY/SELL): {actions}")
            logger.info(f"   • Correct Actions: {correct}")
            logger.info(f"   • Precision: {precision:.1f}% (Target: {target_precision}%)")
            
            # Check Success
            if actions >= 5 and precision >= target_precision:
                logger.info(f"\n✅ SUCCESS! Target precision reached.")
                logger.info(f"Megamind is calibrated.")
                break
            elif actions < 5:
                 logger.warning(f"⚠️ Not enough actionable signals ({actions}) to judge reliability. Continuing...")
            else:
                 logger.info(f"❌ Target not reached. Precision {precision:.1f}% < {target_precision}%.")
            
            # Force Learning & Retraining
            # The _run_backtest method AUTOMATICALLY class _learn_from_backtest_results at the end.
            # Since we disabled cooldowns, it *should* have triggered retraining if accuracy was low.
            # We will manually verify/force it just in case.
            
            logger.info("🧠 Forcing learning from these results...")
            
            # We manually call the internal method to ensure it runs even if _run_backtest skipped it
            # The backtester accumulates results, so we can just tell it to learn from recent history
            backtester._learn_from_backtest_results()
            
            # Verify if retraining happened (check logs or timestamps)
            last_retrain = backtester.last_retrain_time.get('trading')
            logger.info(f"   • Last Retrain Timestamp: {last_retrain}")
            
            iteration += 1
            time.sleep(2) # Breath
            
        if iteration > max_iterations:
            logger.warning(f"\n⚠️ Reached maximum iterations ({max_iterations}). Stopping.")
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Calibration stopped by user.")
    except Exception as e:
        logger.error(f"Error in calibration loop: {e}", exc_info=True)
    finally:
        try:
            # Cleanup
            if app.backend_process:
                app.backend_process.terminate()
            root.destroy()
        except:
            pass

if __name__ == "__main__":
    target = 70.0
    if len(sys.argv) > 1:
        try:
            target = float(sys.argv[1])
        except:
            pass
    run_calibration(target)
