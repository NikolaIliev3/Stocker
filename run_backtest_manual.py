"""
Manual Backtest Runner
Runs a single backtest session to verify ML model performance.
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
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('ManualBacktest')

def run_manual_backtest():
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 STARTING MANUAL BACKTEST VERIFICATION")
    logger.info(f"{'='*70}\n")
    
    # Initialize Headless App
    root = tk.Tk()
    root.withdraw() # Hide window
    
    try:
        app = StockerApp(root)
        backtester = app.backtester
        
        # Stop Scheduler to prevent interference
        if hasattr(app, 'ml_scheduler') and app.ml_scheduler:
             logger.info("Stopping ML Scheduler...")
             app.ml_scheduler.stop()

        # Stop any running backtest
        if backtester.is_testing:
             logger.info("Stopping active background backtest...")
             backtester.stop()
             # Wait for it to finish/abort
             for _ in range(20):
                 if not backtester.is_testing:
                     break
                 time.sleep(0.5)
                 try:
                    root.update()
                 except: 
                    pass
        
        # Strategies to test
        strategies = ['trading']
        
        # Parse arguments for custom sample size
        import argparse
        parser = argparse.ArgumentParser(description='Run manual backtest')
        parser.add_argument('--samples', type=int, default=50, help='Number of samples to test (default: 50)')
        args, _ = parser.parse_known_args()
        
        num_tests = args.samples
        logger.info(f"Configuration: {num_tests} samples")
        
        logger.info(f"Running {num_tests} tests for strategy: {strategies[0]}...")
        
        # Run backtest (use internal method to block until complete if possible, or wait)
        # Using internal _run_backtest blocks execution in the current thread (mostly)
        backtester._run_backtest(selected_strategies=strategies, num_tests=num_tests)
        
        # Get results
        results = backtester.results
        
        if 'strategy_results' in results and 'trading' in results['strategy_results']:
            trading_res = results['strategy_results']['trading']
            
            # --- START PRECISION/RECALL CALCULATION ---
            # Retrieve detailed test results from the backtester instance directly
            test_details = backtester.results.get('test_details', [])
            
            true_positives = 0  # Predicted BUY and was CORRECT
            false_positives = 0 # Predicted BUY and was WRONG
            true_negatives = 0  # Predicted HOLD and was CORRECT (inaction)
            false_negatives = 0 # Predicted HOLD but should have BOUGHT (missed opportunity)
            
            total_buys = 0
            
            for res in test_details:
                if res['strategy'] != 'trading':
                    continue
                    
                pred_action = res['predicted_action']
                was_correct = res['was_correct']
                
                if pred_action == 'BUY':
                    total_buys += 1
                    if was_correct:
                        true_positives += 1
                    else:
                        false_positives += 1
                elif pred_action == 'HOLD':
                    if was_correct:
                        true_negatives += 1
                    else:
                        # In the current logic, HOLD is "correct" if price didn't move much. 
                        # So "Incorrect HOLD" implies the price DID move significantly (Missed Opportunity)
                        false_negatives += 1

            precision = (true_positives / total_buys * 100) if total_buys > 0 else 0.0
            recall = (true_positives / (true_positives + false_negatives) * 100) if (true_positives + false_negatives) > 0 else 0.0
            
            logger.info(f"\n📊 FINAL RESULTS (Precision-Focused):")
            logger.info(f"   • Strategy: Trading")
            logger.info(f"   • Total Samples: {trading_res.get('total', 0)}")
            logger.info(f"   ----------------------------------------")
            logger.info(f"   • 🟢 ACTUALLY TRADED (Buys): {total_buys}")
            logger.info(f"   • ✅ WINNING TRADES:         {true_positives}")
            logger.info(f"   • ❌ LOSING TRADES:          {false_positives}")
            logger.info(f"   ----------------------------------------")
            logger.info(f"   • 🎯 PRECISION (Win Rate):   {precision:.1f}%  <-- THE REAL METRIC")
            logger.info(f"   • 🔎 RECALL (Coverage):      {recall:.1f}%")
            logger.info(f"   • 🛡️ PASSIVE CORRECT (Holds): {true_negatives}")
        else:
            logger.error("❌ No results found for trading strategy.")

    except KeyboardInterrupt:
        logger.info("\n🛑 Backtest stopped by user.")
    except Exception as e:
        logger.error(f"Error in backtest: {e}", exc_info=True)
    finally:
        try:
            if app.backend_process:
                app.backend_process.terminate()
            root.destroy()
        except:
            pass

if __name__ == "__main__":
    run_manual_backtest()
