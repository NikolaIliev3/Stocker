
import logging
import sys
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('StrictRetrain')

from config import APP_DATA_DIR
from training_pipeline import MLTrainingPipeline
from data_fetcher import StockDataFetcher

def retrain_strictly_historical():
    logger.info("🚀 Starting Strict Historical Retraining...")
    logger.info("   • Training Range: 2015-01-01 to 2024-12-31")
    logger.info("   • Test Range:     2025-01-01 to Present (Out-of-Sample)")
    
    # Initialize components
    data_fetcher = StockDataFetcher()
    pipeline = MLTrainingPipeline(data_fetcher, APP_DATA_DIR)
    
    # Define symbols (Top 20 from S&P 500 for robust base)
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'LLY', 'V',
        'UNH', 'XOM', 'JPM', 'JNJ', 'MA', 'PG', 'HD', 'AVGO', 'MRK', 'ABBV'
    ]
    
    # Train heavily on historical data ONLY
    try:
        # Force end_date to 2024-12-31 to prevent ANY 2025 leakage
        result = pipeline.train_on_symbols(
            symbols, 
            start_date="2015-01-01",  # 10 years of history
            end_date="2024-12-31",    # STRICT CUTOFF
            strategy="trading",
            use_binary_classification=True  # Ensure we keep the Buy/Hold logic
        )
        
        logger.info("✅ Retraining Complete!")
        logger.info(f"   • Train Accuracy: {result.get('train_accuracy', 0)*100:.1f}%")
        logger.info(f"   • Test Accuracy:  {result.get('test_accuracy', 0)*100:.1f}%")
        
        if result.get('train_accuracy', 0) > 0.95:
             logger.warning("⚠️ Warning: High training accuracy might still indicate overfitting to the historical period.")
             
    except Exception as e:
        logger.error(f"❌ Retraining Failed: {e}", exc_info=True)

if __name__ == "__main__":
    retrain_strictly_historical()
