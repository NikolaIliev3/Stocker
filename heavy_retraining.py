
import logging
import sys
import os
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("heavy_retraining.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('HeavyRetrain')

from config import APP_DATA_DIR
from training_pipeline import MLTrainingPipeline
from data_fetcher import StockDataFetcher

def run_heavy_retraining():
    logger.info("STARTING HEAVY HISTORICAL RETRAINING (Quant Mode)")
    logger.info("===================================================")
    logger.info("   • Training Range: 2010-01-01 to 2025-12-31 (15+ Years)")
    logger.info("   • Target:         100 Stocks")
    logger.info("   • Strategy:       trading")
    logger.info("   • Integrity:      Strict Cutoff @ 2025-12-31")
    
    # Initialize components
    data_fetcher = StockDataFetcher()
    pipeline = MLTrainingPipeline(data_fetcher, APP_DATA_DIR)
    
    # 100 Stocks List (User provided + Expanded for Diversification)
    symbols = [
        # User Provided (with duplicates removed)
        'KO', 'GOOGL', 'HD', 'MCD', 'PG', 'META', 'MSFT', 'AMZN', 'MS', 'AAPL', 
        'PEP', 'WMT', 'ABBV', 'AMD', 'DELL', 'SPY', 'QQQ', 'IWM', 'JPM', 'XLE', 
        'NVDA', 'CRM', 'GS', 'PFE', 'CAT', 'DE', 'UPS', 'FDX', 'LOW', 'COST', 
        'XOM', 'CVX', 'XLK', 'XLF', 'XLI', 'XLY', 'V', 'MA', 'BLK', 'LMT', 
        'RTX', 'TSLA', 'JNJ', 'TGT', 'TXN', 'MDLZ', 'SPGI', 'SIE.DE',
        
        # Additional Diversification (to reach 100)
        'UNH', 'LLY', 'AVGO', 'ORCL', 'ADBE', 'BAC', 'NFLX', 'DIS', 'NKE', 'SBUX',
        'GE', 'AMGN', 'IBM', 'HON', 'INTC', 'AMAT', 'QCOM', 'MU', 'CSCO', 'ACN',
        'NOW', 'ISRG', 'BKNG', 'ADP', 'T', 'VZ', 'CMCSA', 'CVS', 'EL', 'MO',
        'PM', 'PLD', 'AMT', 'PSA', 'O', 'SPG', 'DLR', 'CB', 'PGR', 'TRV',
        'ALL', 'MET', 'PRU', 'MMC', 'AON', 'SCHW', 'MAR', 'HLT', 'SYY', 'ADM',
        'SO', 'NEE', 'DUK', 'PLTR', 'SNOW', 'MSTR', 'TEAM', 'WDAY', 'DDOG', 'ZS'
    ]
    
    # Remove duplicates just in case and sort
    symbols = sorted(list(set(symbols)))
    logger.info(f"Final Symbol List: {len(symbols)} stocks")
    
    # Training Configuration
    config = {
        'use_binary_classification': True,
        'use_regime_models': True,
        'use_hyperparameter_tuning': True, # Enabled for best quality
        'model_family': 'voting',
        'test_size': 0.15 # 15% for internal validation
    }
    
    try:
        start_time = datetime.now()
        logger.info(f"Training started at {start_time}")
        
        # Execute Training
        result = pipeline.train_on_symbols(
            symbols,
            start_date="2010-01-01",
            end_date="2025-12-31", # THE CUTOFF (Updated per user request)
            strategy="trading",
            **config
        )
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        if 'error' in result:
            logger.error(f"❌ Retraining Failed: {result['error']}")
        else:
            logger.info("✅ SUCCESS: Heavy Retraining Complete")
            logger.info(f"   • Duration:          {duration}")
            logger.info(f"   • Total Samples:     {result.get('total_samples', 0)}")
            logger.info(f"   • Weighted Accuracy: {result.get('test_accuracy', 0)*100:.1f}%")
            logger.info(f"   • Model Path:        {APP_DATA_DIR}")
            
    except Exception as e:
        logger.critical(f"💥 Fatal error during retraining: {e}", exc_info=True)

if __name__ == "__main__":
    run_heavy_retraining()
