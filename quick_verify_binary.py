"""
Quick Verification Script
Tests the Binary ML Refactor on a single symbol.
"""
import logging
import sys
from pathlib import Path
from config import APP_DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("QuickVerify")

from training_pipeline import MLTrainingPipeline
from data_fetcher import StockDataFetcher
from auto_cleanup import perform_cleanup

def verify_binary_pipeline():
    logger.info("🧪 STARTING VERIFICATION TEST")
    
    # 1. Test Cleanup
    logger.info("\n1. Testing Auto-Cleanup...")
    perform_cleanup()
    
    # Check if files are gone
    if (Path(APP_DATA_DIR) / "ml_model_trading.pkl").exists():
        logger.error("❌ Cleanup failed! Model file still exists.")
        return
    logger.info("✅ Cleanup passed.")
    
    # 2. Test Label Generation & Training
    logger.info("\n2. Testing Training Pipeline (Single Symbol)...")
    
    fetcher = StockDataFetcher()
    pipeline = MLTrainingPipeline(fetcher, Path(APP_DATA_DIR))
    
    # Train on just one symbol for speed
    # 'AAPL' usually has enough data and volatility
    try:
        results = pipeline.train_on_symbols(
            symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN'], 
            start_date='2020-01-01', 
            end_date='2024-01-01',
            strategy='trading'
        )
        
        if 'error' in results:
            logger.error(f"❌ Training failed: {results['error']}")
        else:
            logger.info("✅ Training completed successfully.")
            logger.info(f"   Train Acc: {results.get('train_accuracy', 0)*100:.1f}%")
            logger.info(f"   Test Acc: {results.get('test_accuracy', 0)*100:.1f}%")
            logger.info(f"   Classes: {results.get('classes', 'N/A')}")
            
    except Exception as e:
        logger.error(f"❌ Exception during training: {e}", exc_info=True)

if __name__ == "__main__":
    verify_binary_pipeline()
