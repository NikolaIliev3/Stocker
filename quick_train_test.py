"""
Quick Training Test - 5-10 minute validation run
Uses minimal data to verify the training pipeline works correctly.
"""
import os
import sys
import logging
import numpy as np

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('QuickTrainTest')

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import APP_DATA_DIR
from data_fetcher import StockDataFetcher
from ml_training import StockPredictionML, FeatureExtractor
from training_pipeline import TrainingDataGenerator

# MINIMAL CONFIG - 5 stocks, 2 years, quick test
TEST_SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
DATA_PERIOD = '2y'

def run_quick_test():
    logger.info("="*60)
    logger.info("QUICK TRAINING TEST - 5 Stocks, 2 Years, ~100 Samples")
    logger.info("="*60)
    
    # Initialize
    data_fetcher = StockDataFetcher()
    feature_extractor = FeatureExtractor()
    generator = TrainingDataGenerator(data_fetcher, feature_extractor)
    
    # Calculate date range (2 years back)
    from datetime import datetime, timedelta
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    
    # Collect training samples
    logger.info(f"\n📊 Generating samples for {len(TEST_SYMBOLS)} stocks...")
    logger.info(f"   Date range: {start_date} to {end_date}")
    
    all_samples = []
    label_counts = {'UP': 0, 'DOWN': 0}
    
    for symbol in TEST_SYMBOLS:
        logger.info(f"  Processing {symbol}...")
        try:
            samples = generator.generate_training_samples(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                strategy='trading',
                lookforward_days=10
            )
            if samples:
                all_samples.extend(samples)
                # Count labels
                for s in samples:
                    label = s.get('label', 'UNKNOWN')
                    if label in label_counts:
                        label_counts[label] += 1
                logger.info(f"    ✅ Generated {len(samples)} samples from {symbol}")
        except Exception as e:
            logger.error(f"    ❌ Error with {symbol}: {e}")
    
    logger.info(f"\n📈 Total samples collected: {len(all_samples)}")
    logger.info(f"   Label distribution: {label_counts}")
    
    # Check balance
    total = sum(label_counts.values())
    if total > 0:
        up_pct = label_counts['UP'] / total * 100
        down_pct = label_counts['DOWN'] / total * 100
        logger.info(f"   UP: {up_pct:.1f}%, DOWN: {down_pct:.1f}%")
        
        if abs(up_pct - 50) > 20:
            logger.warning("⚠️ IMBALANCED DATA DETECTED! This will cause prediction bias.")
    
    if len(all_samples) < 30:
        logger.error("❌ Not enough samples to train. Need at least 30.")
        return False
    
    # Train a quick model
    logger.info(f"\n🔧 Training ML model with {len(all_samples)} samples...")
    ml = StockPredictionML(strategy='trading', data_dir=APP_DATA_DIR)
    
    try:
        # Extract features and labels (for logging only)
        features = np.array([s['features'] for s in all_samples if 'features' in s])
        labels = np.array([s['label'] for s in all_samples if 'label' in s])
        
        logger.info(f"   Features shape: {features.shape}")
        logger.info(f"   Labels shape: {labels.shape}")
        logger.info(f"   Unique labels: {np.unique(labels)}")
        
        # Pass the raw samples list - train() expects list of dicts with 'features' and 'label'
        result = ml.train(all_samples)
        
        if result and result.get('test_accuracy', 0) > 0:
            logger.info(f"\n✅ TRAINING SUCCESSFUL!")
            logger.info(f"   Accuracy: {result.get('test_accuracy', 0) * 100:.1f}%")
            logger.info(f"   Classes: {result.get('label_classes', [])}")
            return True
        else:
            logger.error("❌ Training returned no result or 0% accuracy")
            logger.error(f"   Result: {result}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_quick_test()
    if success:
        print("\n" + "="*60)
        print("✅ Quick test PASSED - Pipeline is working correctly!")
        print("   You can now run the full training with confidence.")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ Quick test FAILED - Fix issues before full training!")
        print("="*60)
