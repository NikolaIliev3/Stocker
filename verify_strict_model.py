
import logging
import sys
import os
import time
from datetime import datetime
import tkinter as tk
import threading

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('VerifyStrictModel')

# Add project root to path
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import APP_DATA_DIR
from training_pipeline import MLTrainingPipeline
from data_fetcher import StockDataFetcher
from main import StockerApp

def verify_strict_model():
    logger.info("🚀 STARTING STRICT VERIFICATION PROTOCOL")
    logger.info("==========================================")
    
    # 1. TRAIN CLEAN MODEL (Pre-June 2024)
    # ----------------------------------------------------
    logger.info("PHASE 1: Training 'verification_test' model")
    logger.info("   • Training Range: 2015-01-01 to 2024-05-31")
    logger.info("   • Purpose: Ensure model assumes 2024-06+ is 'Future'")
    
    data_fetcher = StockDataFetcher()
    pipeline = MLTrainingPipeline(data_fetcher, APP_DATA_DIR)
    
    # Use top liquid stocks for robust verification
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AMD', 'NFLX', 'INTC',
        'JPM', 'BAC', 'GS', 'MS', 'V', 'MA', 'WFC', 'C', 'BLK', 'AXP'
    ]
    
    # Train
    try:
        # We use a custom strategy name to avoid overwriting production models
        strategy_name = "verification_test"
        
        # Override config to ensure binary classification (easier to verify)
        training_config = {
            'use_binary_classification': True, 
            'use_regime_models': True,
            'use_hyperparameter_tuning': False # Speed up verification
        }
        
        result = pipeline.train_on_symbols(
            symbols, 
            start_date="2015-01-01", 
            end_date="2024-05-31",    # STRICT CUTOFF
            strategy=strategy_name,
            **training_config
        )
        
        if 'error' in result:
             logger.error(f"❌ Training Failed: {result['error']}")
             return
             
        logger.info(f"✅ Clean Model Trained. Test Accuracy available directly from training split: {result.get('test_accuracy', 0)*100:.1f}%")
        
    except Exception as e:
        logger.error(f"❌ Training Exception: {e}", exc_info=True)
        return

    # 2. BACKTEST ON UNSEEN DATA (Post-June 2024)
    # ----------------------------------------------------
    logger.info("\nPHASE 2: Backtesting on UNSEEN DATA")
    logger.info("   • Test Range: 2024-06-01 to Present")
    logger.info("   • Strategy:   verification_test")
    
    # Initialize Headless App for Backtester
    root = tk.Tk()
    root.withdraw()
    app = StockerApp(root)
    
    # Stop background threads
    if app.ml_scheduler: app.ml_scheduler.stop()
    if app.auto_learner: app.auto_learner.stop()
    
    # Override Strategy Map in HybridPredictor to use our new model
    # The app loads 'trading' by default. We need to tell backtester to use 'verification_test'
    # ContinuousBacktester._run_backtest takes a list of strategy names.
    # It tries to verify against app.hybrid_predictors[strategy].
    
    # We must ensure 'verification_test' predictor exists in app.hybrid_predictors
    from hybrid_predictor import HybridStockPredictor
    from ml_training import StockPredictionML
    
    # Create the predictor using our new clean model
    clean_ml = StockPredictionML(APP_DATA_DIR, strategy_name)
    if not clean_ml.is_trained():
        # Force load if valid
        clean_ml._load_model()
        
    # Inject into app
    app.hybrid_predictors[strategy_name] = HybridStockPredictor(
        APP_DATA_DIR, 
        strategy_name, 
        ml_predictor=clean_ml
    )
    logger.info(f"💉 Injected 'verification_test' predictor into App")
    
    # Configure Backtester to test MAINLY recent dates (Post June 2024)
    # The backtester picks random dates. We can't easily force it without modifying code.
    # BUT, recently we modified continuous_backtester to be smarter? 
    # Let's just run it and see. The dates chosen will be random.
    # To be precise, we can patch the date selection or just run MANY tests and filter logs.
    
    # Let's run 100 tests. Statistically ~20-30% should be in the last year (2025).
    # Wait, the range 2015-2025 is 10 years. 2024-06 to 2025-02 is ~0.7 years.
    # That's only 7% of history.
    
    # Hack: We will MONKEY PATCH the date selector in backtester to prefer recent dates
    # or just accept that we need to look at specific dates.
    
    # BETTER: We use `_test_historical_point` directly loop over recent dates!
    
    logger.info("⚡ Running Explicit Out-of-Sample Tests (Monthly check from 2024-07-01)...")
    
    backtester = app.backtester
    predictor = app.hybrid_predictors[strategy_name]
    
    test_dates = [
        "2024-07-01", "2024-08-01", "2024-09-01", "2024-10-01", 
        "2024-11-01", "2024-12-01", "2025-01-02", "2025-02-01"
    ]
    
    success_count = 0
    total_count = 0
    
    for date_str in test_dates:
        test_date = datetime.strptime(date_str, "%Y-%m-%d")
        
        # Test on 5 symbols per date
        for symbol in symbols[:5]:
            logger.info(f"   Testing {symbol} on {date_str}...")
            try:
                result = backtester._test_historical_point(
                    symbol, test_date, strategy_name, predictor, test_number=total_count
                )
                
                if result:
                    total_count += 1
                    if result['was_correct']:
                        success_count += 1
                    
                    logger.info(f"   -> Result: {result['predicted_action']} | Correct: {result['was_correct']}")
            except Exception as e:
                logger.warning(f"   Test error: {e}")
                
    if total_count > 0:
        accuracy = (success_count / total_count) * 100
        logger.info(f"\n✅ VERIFICATION RESULTS (Out-of-Sample 2024-06+):")
        logger.info(f"   Accuracy: {accuracy:.1f}% ({success_count}/{total_count})")
        if accuracy == 100.0:
            logger.critical("❌❌❌ 100% ACCURACY DETECTED. LEAK STILL EXISTS! ❌❌❌")
        elif accuracy > 55.0:
            logger.info("✅ Valid Model (>55%). Leak likely fixed.")
        else:
            logger.warning("⚠️ Model performance weak (<55%). At least it's honest.")
    else:
        logger.warning("No valid tests completed.")

if __name__ == "__main__":
    verify_strict_model()
