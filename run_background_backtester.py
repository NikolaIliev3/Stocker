"""
Background Continuous Backtester
Runs the backtester in continuous mode as a background service.
"""
import sys
import time
import logging
import tkinter as tk
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from main import StockerApp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('BackgroundBacktester')

def start_background_service():
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 STARTING BACKGROUND CONTINUOUS BACKTESTER")
    logger.info(f"{'='*70}\n")
    
    # Initialize Headless App
    root = tk.Tk()
    root.withdraw() # Hide window
    
    try:
        app = StockerApp(root)
        backtester = app.backtester
        
        # Configure for background running
        backtester.continuous_delay_seconds = 10  # Fast interval for demo/initial purposes, then maybe slow down? 
                                                  # Actually 10s is fine, it just checks loop. 
                                                  # Actual testing takes time.
        
        # Start Continuous Mode
        strategies = ['trading'] # Focus on trading strategy
        logger.info(f"Enabling continuous mode for: {strategies}")
        backtester.start_continuous_mode(selected_strategies=strategies)
        
        logger.info("✅ Service is running relative to .stocker data directory.")
        logger.info("   • It will continuously validate predictions.")
        logger.info("   • It will learn from mistakes (updates learning_tracker.json).")
        logger.info("   • Press Ctrl+C to stop.")
        
        # Keep alive
        root.mainloop()

    except KeyboardInterrupt:
        logger.info("\n🛑 Service stopped by user.")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        try:
            if app and hasattr(app, 'backend_process') and app.backend_process:
                app.backend_process.terminate()
            root.destroy()
        except:
            pass

if __name__ == "__main__":
    start_background_service()
