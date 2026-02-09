"""
Manual Backtest Runner (Large)
Runs a large backtest session to thoroughly verify ML model performance.
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

def run_large_backtest():
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 STARTING LARGE BACKTEST (1000 SAMPLES)")
    logger.info(f"{'='*70}\n")
    
    # Initialize Headless App
    root = tk.Tk()
    root.withdraw() # Hide window
    
    try:
        app = StockerApp(root)
        backtester = app.backtester
        
        # Strategies to test
        strategies = ['trading']
        num_tests = 500 
        
        # Run backtest
        logger.info(f"Running {num_tests} tests for strategy: {strategies[0]}...")
        
        # Ensure it's not already running
        if backtester.is_testing:
            logger.info("Backtest already running, waiting for it to finish...")
            while backtester.is_testing:
                time.sleep(1)
        
        # Start new backtest
        backtester._run_backtest(selected_strategies=strategies, num_tests=num_tests)
        
        # WAIT for it to finish (since it might be running in a thread or we want to be sure)
        logger.info("Waiting for backtest results...")
        start_wait = time.time()
        while backtester.is_testing:
            time.sleep(2)
            if time.time() - start_wait > 3600: # 1 hour timeout
                logger.error("Backtest timed out!")
                break
        
        # Get results
        results = backtester.results
        
        if 'strategy_results' in results and 'trading' in results['strategy_results']:
            trading_res = results['strategy_results']['trading']
            logger.info(f"\n📊 FINAL RESULTS:")
            logger.info(f"   • Strategy: Trading")
            logger.info(f"   • Total Tests: {trading_res.get('total', 0)}")
            logger.info(f"   • Correct: {trading_res.get('correct', 0)}")
            logger.info(f"   • Accuracy: {trading_res.get('accuracy', 0):.1f}%")
        else:
            logger.error("❌ No results found for trading strategy.")
            # Debug: what DOES results have?
            logger.info(f"Available strategy results: {list(results.get('strategy_results', {}).keys())}")

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
    run_large_backtest()
