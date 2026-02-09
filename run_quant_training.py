"""
Run Quant Training
Triggers the ML training process using the fresh "Quant-Verified" data.
"""
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from training_pipeline import MLTrainingPipeline
from predictions_tracker import PredictionsTracker
from data_fetcher import StockDataFetcher
from config import APP_DATA_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("QuantTrainer")

def train():
    logger.info("🧠 Initializing Quant Training Pipeline...")
    
    # ENFORCE CLEANUP PROTOCOL
    try:
        from auto_cleanup import perform_cleanup
        perform_cleanup()
    except Exception as e:
        logger.warning(f"⚠️ Auto-cleanup failed: {e}")
    
    
    # Initialize Dependencies
    data_fetcher = StockDataFetcher()
    tracker = PredictionsTracker(APP_DATA_DIR)
    
    # Initialize Pipeline
    pipeline = MLTrainingPipeline(data_fetcher, APP_DATA_DIR)
    
    # Check if we have enough data (we should have ~200 samples)
    if not (APP_DATA_DIR / "predictions.json").exists():
        logger.warning("⚠️ predictions.json not found! Switching to BOOTSTRAP MODE.")
        logger.info("🔨 Starting Bootstrap Training on Top 20 Market Cap Stocks & Indices...")
        
        # Bootstrap symbols (Indices + Sector Leaders to ensure regime coverage)
        BOOTSTRAP_SYMBOLS = [
            'SPY', 'QQQ', 'IWM', 'DIA',       # Indices
            'AAPL', 'MSFT', 'GOOGL', 'AMZN',  # Big Tech
            'NVDA', 'META', 'TSLA', 'JPM',    # Momentum/Finance
            'JNJ', 'PG', 'XOM', 'CVX',        # Defensive/Energy
            'GLD', 'USO', 'TLT'               # Commodities/Rates
        ]
        
        try:
            # Train directly on historical data
            result = pipeline.train_on_symbols(
                symbols=BOOTSTRAP_SYMBOLS,
                model_family='voting',
                use_hyperparameter_tuning=True, # Enable for best results
                use_regime_models=True
            )
            
            if result and result.get('ml_retrained'):
                logger.info("\n🏆 BOOTSTRAP TRAINING COMPLETE")
                logger.info(f"   • Test Accuracy: {result.get('test_accuracy', 0)*100:.1f}%")
                logger.info("✅ The model is now initialized and ready for backtesting.")
            else:
                 logger.warning("⚠️ Bootstrap training finished with warnings.")
            
            return # Exit after bootstrap

        except Exception as e:
            logger.error(f"❌ Bootstrap Training Failed: {e}", exc_info=True)
            return

    logger.info("🚀 Starting Training on Verified Predictions...")
    
    # Train using the pipeline
    try:
        # train_on_verified_predictions handles Loading -> Converting -> Training
        result = pipeline.train_on_verified_predictions(
            predictions_tracker=tracker, 
            strategy='trading', 
            retrain_ml=True
        )
        
        if result and result.get('ml_retrained'):
            logger.info("\n🏆 TRAINING COMPLETE")
            logger.info(f"   • Test Accuracy: {result.get('test_accuracy', 0)*100:.1f}%")
            logger.info("✅ The model is now calibrated to Quant Mode logic.")
        else:
            err = result.get('error', 'Unknown Error') if result else 'No result'
            logger.warning(f"⚠️ Training finished with warning: {err}")
            
    except Exception as e:
        logger.error(f"❌ Training Failed: {e}", exc_info=True)

if __name__ == "__main__":
    train()
