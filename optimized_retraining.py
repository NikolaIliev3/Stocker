"""
Optimized Retraining Script
Uses smaller training window (2-3 years) which produced the best 66.7% accuracy model.
The 15-year retraining introduced too much noise and lowered accuracy.
"""
import sys
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

from training_pipeline import MLTrainingPipeline
from config import APP_DATA_DIR
from data_fetcher import StockDataFetcher

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('OptimizedRetraining')

def run_optimized_retraining():
    logger.info("🎯 OPTIMIZED RETRAINING (Targeting 60%+ Accuracy)")
    logger.info("="*50)
    
    # OPTIMIZED SYMBOL LIST
    # The 66.7% model used fewer symbols - focus on quality over quantity
    training_symbols = [
        # Core Tech (High liquidity, good patterns)
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        # Core Finance
        'JPM', 'BAC', 'GS', 'V', 'MA',
        # Core Consumer
        'KO', 'PEP', 'WMT', 'HD', 'MCD', 'NKE',
        # Core Healthcare
        'JNJ', 'PFE', 'UNH',
        # Core Industrial
        'CAT', 'BA', 'GE',
        # ETFs for context
        'SPY', 'QQQ'
    ]
    
    logger.info(f"Using focused symbol list: {len(training_symbols)} symbols")
    
    # OPTIMIZED LOOKBACK: 2-3 years, not 15
    # The 66.7% model had ~1000 samples which is about 2-3 years of data
    lookback_days = 365 * 3  # 3 years
    
    logger.info(f"Lookback: {lookback_days} days (~3 years)")
    
    # Initialize
    data_fetcher = StockDataFetcher()
    pipeline = MLTrainingPipeline(data_fetcher, APP_DATA_DIR)
    
    try:
        from datetime import datetime, timedelta
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        logger.info(f"Date range: {start_date} to {end_date}")
        
        result = pipeline.train_on_symbols(
            symbols=training_symbols,
            start_date=start_date,
            end_date=end_date,
            strategy="trading"
        )
        
        if result and 'test_accuracy' in result:
            acc = result.get('test_accuracy', 0)
            logger.info(f"\n✅ RETRAINING COMPLETE")
            logger.info(f"   Test Accuracy: {acc:.1%}")
            if acc >= 0.60:
                logger.info(f"   🎯 TARGET ACHIEVED! (≥60%)")
            else:
                logger.info(f"   ⚠️ Below 60% target, may need further tuning")
        else:
            logger.info("\n✅ Retraining complete. Run backtest to verify accuracy.")
            
    except Exception as e:
        logger.error(f"❌ Retraining failed: {e}", exc_info=True)

if __name__ == "__main__":
    run_optimized_retraining()
