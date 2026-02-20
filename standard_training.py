"""
Standard ML Training - 50 Stocks, 10 Years
Optimized for balanced coverage and training time (~1.5 hours)
"""
import os
import sys
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('StandardTraining')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import APP_DATA_DIR
from data_fetcher import StockDataFetcher
from ml_training import StockPredictionML, FeatureExtractor
from training_pipeline import TrainingDataGenerator

# STANDARD CONFIG - 50 stocks, 10 years
TRAINING_SYMBOLS = [
    # Tech Giants
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM',
    # Financial
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA', 'AXP', 'BLK', 'C',
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'MRK', 'ABBV', 'TMO', 'ABT', 'LLY', 'BMY', 'AMGN',
    # Consumer
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
    # Industrial/Energy
    'XOM', 'CVX', 'BA', 'CAT', 'GE', 'HON', 'UPS', 'RTX', 'LMT', 'DE'
]

YEARS_OF_DATA = 10

def run_standard_training():
    start_time = datetime.now()
    logger.info("="*60)
    logger.info("STANDARD ML TRAINING - 50 Stocks, 10 Years")
    logger.info("="*60)
    logger.info(f"Started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Estimated completion: ~1.5 hours")
    
    # Initialize
    data_fetcher = StockDataFetcher()
    feature_extractor = FeatureExtractor()
    generator = TrainingDataGenerator(data_fetcher, feature_extractor)
    
    # Calculate date range
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=YEARS_OF_DATA * 365)).strftime('%Y-%m-%d')
    
    logger.info(f"\n📅 Date range: {start_date} to {end_date}")
    logger.info(f"📊 Processing {len(TRAINING_SYMBOLS)} symbols...")
    
    # Collect training samples
    all_samples = []
    label_counts = {'UP': 0, 'DOWN': 0}
    successful_symbols = []
    failed_symbols = []
    
    for i, symbol in enumerate(TRAINING_SYMBOLS, 1):
        progress = i / len(TRAINING_SYMBOLS) * 100
        logger.info(f"\n[{i}/{len(TRAINING_SYMBOLS)}] ({progress:.0f}%) Processing {symbol}...")
        
        try:
            samples = generator.generate_training_samples(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                strategy='trading',
                lookforward_days=10
            )
            
            if samples and len(samples) > 0:
                all_samples.extend(samples)
                for s in samples:
                    label = s.get('label', 'UNKNOWN')
                    if label in label_counts:
                        label_counts[label] += 1
                successful_symbols.append(symbol)
                logger.info(f"    ✅ {len(samples)} samples")
            else:
                failed_symbols.append(symbol)
                logger.warning(f"    ⚠️ No samples generated")
                
        except Exception as e:
            failed_symbols.append(symbol)
            logger.error(f"    ❌ Error: {e}")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SAMPLE GENERATION COMPLETE")
    logger.info("="*60)
    logger.info(f"Total samples: {len(all_samples)}")
    logger.info(f"Successful symbols: {len(successful_symbols)}/{len(TRAINING_SYMBOLS)}")
    logger.info(f"Label distribution: UP={label_counts['UP']}, DOWN={label_counts['DOWN']}")
    
    total = sum(label_counts.values())
    if total > 0:
        up_pct = label_counts['UP'] / total * 100
        down_pct = label_counts['DOWN'] / total * 100
        logger.info(f"Balance: UP={up_pct:.1f}%, DOWN={down_pct:.1f}%")
        
        if abs(up_pct - 50) > 15:
            logger.warning("⚠️ Data imbalance detected - SMOTE will correct this")
    
    if len(all_samples) < 100:
        logger.error("❌ Not enough samples for training!")
        return False
    
    # Train the model
    logger.info("\n" + "="*60)
    logger.info("TRAINING ML MODEL")
    logger.info("="*60)
    
    ml = StockPredictionML(strategy='trading', data_dir=APP_DATA_DIR)
    
    try:
        result = ml.train(all_samples)
        
        if result and result.get('test_accuracy', 0) > 0:
            elapsed = datetime.now() - start_time
            
            logger.info("\n" + "="*60)
            logger.info("✅ TRAINING SUCCESSFUL!")
            logger.info("="*60)
            logger.info(f"Validation Accuracy: {result.get('test_accuracy', 0) * 100:.1f}%")
            logger.info(f"Cross-Validation: {result.get('cv_mean', 0) * 100:.1f}% ± {result.get('cv_std', 0) * 100:.1f}%")
            logger.info(f"Classes: {result.get('label_classes', [])}")
            logger.info(f"Features used: {result.get('selected_features_count', 0)}/{result.get('total_features', 0)}")
            logger.info(f"Total time: {elapsed}")
            
            return True
        else:
            logger.error("❌ Training failed - no result returned")
            return False
            
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_standard_training()
    
    print("\n" + "="*60)
    if success:
        print("✅ STANDARD TRAINING COMPLETE!")
        print("   Model is ready for backtesting.")
    else:
        print("❌ TRAINING FAILED - Check logs above")
    print("="*60)
