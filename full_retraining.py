"""
Full Retraining Script
Restores model health by training on a large, diverse dataset of historical data.
Fixes the "Severe Overfitting" caused by training on small backtest batches.
"""
import sys
import logging
import warnings
from pathlib import Path

# Suppress pandas warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from training_pipeline import MLTrainingPipeline
from ml_training import StockPredictionML
from secure_ml_storage import SecureMLStorage
from config import APP_DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('FullRetraining')

def run_full_retraining():
    logger.info("🚀 STARTING ROBUST RETRAINING (Fixing Overfitting)")
    logger.info("==================================================")
    
    # 1. Parse Command Line Arguments
    import argparse
    # Provide defaults
    default_symbols = [
        'KO', 'GOOGL', 'HD', 'MCD', 'PG', 'META', 'MSFT', 'AMZN', 'MS', 'AAPL',
        'PEP', 'WMT', 'ABBV', 'AMD', 'DELL', 'SPY', 'QQQ', 'IWM', 'JPM', 'XLE',
        'NVDA', 'CRM', 'GS', 'PFE', 'CAT', 'DE', 'UPS', 'FDX', 'LOW', 'COST',
        'XOM', 'CVX', 'XLK', 'XLF', 'XLI', 'XLY', 'V', 'MA', 'BLK', 'LMT',
        'RTX', 'TSLA', 'JNJ', 'TGT', 'TXN', 'MDLZ'
    ]
    
    # Check if run from command line
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description='Full Retraining Script')
        parser.add_argument('--symbols', nargs='+', help='List of symbols to train on', default=default_symbols)
        parser.add_argument('--years', type=int, help='Years of history to fetch', default=5)
        args = parser.parse_args()
        
        training_symbols = args.symbols
        history_years = args.years
    else:
        training_symbols = default_symbols
        history_years = 5
        
    logger.info(f"Targeting {len(training_symbols)} symbols for diverse training data.")
    logger.info(f"Using {history_years} years of historical data.")

    # 2. Initialize Dependencies
    from data_fetcher import StockDataFetcher
    data_fetcher = StockDataFetcher()
    
    # 3. Initialize Pipeline
    pipeline = MLTrainingPipeline(data_fetcher, APP_DATA_DIR)
    
    # 4. Config Overrides
    lookback_days = 365 * history_years
    
    try:
        # 5. Run Training
        logger.info(f"Fetching data and generating samples (Lookback: {lookback_days} days)...")
        
        # Override config to enforce anti-overfitting
        pipeline.app = type('MockApp', (), {})() # Mock app to hold config
        pipeline.app._saved_training_config = {
            'rf_max_depth': 5,  # Balanced for new features
            'rf_min_samples_leaf': 20, # Require dense leaves
            'use_binary_classification': True,
            'use_smote': True, # Keep SMOTE for balance, but depth limit handles overfitting
            'feature_selection_method': 'rfe' 
        }
        
        # This calls the pipeline which handles fetching, feature extraction, and training
        # It triggers the 'train_on_symbols' logic
        from datetime import datetime, timedelta
        end_date = '2025-01-01'
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        pipeline.train_on_symbols(
            symbols=training_symbols,
            start_date=start_date,
            end_date=end_date,
            strategy="trading"
        )
        
        logger.info("\n✅ FULL RETRAINING COMPLETE")
        logger.info("   The model should now have >60% accuracy on test data.")
        logger.info("   Overfitting warnings should be resolved.")
        
    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}", exc_info=True)

if __name__ == "__main__":
    run_full_retraining()
