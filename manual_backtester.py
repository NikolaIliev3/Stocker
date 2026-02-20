"""
Manual Backtester Tool
Run this script to manually execute a specific number of backtests and see the results immediately.
Usage: python manual_backtester.py [num_tests]
"""
import sys
import argparse
import logging
import tkinter as tk
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from main import StockerApp

# Configure logging to show everything clearly in console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s', # Simplified format for readability
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('ManualBacktester')

def run_manual_test(count=10):
    print(f"\n{'='*60}")
    print(f"🔬 STARTING MANUAL BACKTEST ({count} samples)")
    print(f"{'='*60}\n")
    
    # Initialize Headless App
    root = tk.Tk()
    root.withdraw() 
    
    try:
        app = StockerApp(root)
        backtester = app.backtester
        
        # Configure standard strategy
        strategies = ['trading']
        
        # Run the backtest (this blocks until completion)
        backtester._run_backtest(selected_strategies=strategies, num_tests=count)
        
        # Display Results
        results = backtester.results
        if 'strategy_results' in results and 'trading' in results['strategy_results']:
            stats = results['strategy_results']['trading']
            print(f"\n{'='*60}")
            print(f"📊 SUMMARY RESULTS")
            print(f"{'='*60}")
            print(f"Tests Run:      {stats.get('total', 0)}")
            print(f"Correct:        {stats.get('correct', 0)}")
            print(f"Accuracy:       {stats.get('accuracy', 0):.1f}%")
            print(f"{'='*60}\n")
            
            print("💡 Note: Detailed prediction logs (inputs/probabilities) are above.")
            print("Files updated: backtest_results.json, learning_tracker.json")
            
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user.")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
    finally:
        try:
            if hasattr(app, 'backend_process') and app.backend_process:
                app.backend_process.terminate()
            root.destroy()
        except:
            pass

if __name__ == "__main__":
    # Allow passing number of tests as argument
    count = 10
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            print("Usage: python manual_backtester.py [number_of_tests]")
            sys.exit(1)
            
    run_manual_test(count)
