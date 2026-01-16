
import os
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Configure logging to see progress
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ForceRetrain")

# Add current directory to path so we can import local modules
current_dir = Path(__file__).parent.absolute()
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

try:
    from config import APP_DATA_DIR
    from data_fetcher import StockDataFetcher
    from training_pipeline import MLTrainingPipeline
    from market_scanner import MarketScanner
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    sys.exit(1)

def run_retrain():
    logger.info("Starting forced retraining of ML models to sync feature sets...")
    
    data_fetcher = StockDataFetcher()
    pipeline = MLTrainingPipeline(data_fetcher, APP_DATA_DIR)
    
    # Use a subset of popular stocks for quick but effective retraining
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'JPM', 'V', 'JNJ']
    
    # Date range for training (last 2 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    strategies = ['trading'] # Focus on the one with mismatch first
    
    for strategy in strategies:
        try:
            logger.info(f"\n--- Retraining {strategy.upper()} Strategy ---")
            logger.info(f"Symbols: {symbols}")
            logger.info(f"Range: {start_date} to {end_date}")
            
            results = pipeline.train_on_symbols(
                symbols=symbols,
                start_date=start_date,
                end_date=end_date,
                strategy=strategy,
                progress_callback=lambda msg: logger.info(f"  [Progress] {msg}")
            )
            
            if 'error' in results:
                logger.error(f"❌ Failed to retrain {strategy}: {results['error']}")
            else:
                logger.info(f"✅ Successfully retrained {strategy}!")
                logger.info(f"   Accuracy: {results.get('test_accuracy', 0)*100:.2f}%")
                logger.info(f"   Samples: {results.get('total_samples', 0)}")
                
        except Exception as e:
            logger.error(f"Exception during {strategy} retraining: {e}", exc_info=True)

if __name__ == "__main__":
    # Ensure data directory exists
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    run_retrain()
