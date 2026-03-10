"""
ML Retraining Script — Anti-Overfitting Fixes Applied
Trains on 56 diverse stocks × 5 years
"""
import sys
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from datetime import datetime, timedelta
from data_fetcher import StockDataFetcher
from training_pipeline import MLTrainingPipeline

# 56 stocks — diverse universe, no ETFs
TRAINING_STOCKS = [
    # Tech
    'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN', 'NVDA', 'AMD', 'CRM', 'DELL', 'INTC', 'TXN',
    # Consumer
    'KO', 'PEP', 'MCD', 'PG', 'WMT', 'COST', 'HD', 'LOW', 'TGT', 'NKE', 'SBUX', 'MDLZ',
    # Healthcare
    'JNJ', 'PFE', 'ABBV', 'ABT', 'AMGN', 'BMY', 'LLY', 'MRK', 'TMO', 'UNH',
    # Financials
    'JPM', 'GS', 'MS', 'BAC', 'C', 'WFC', 'BLK', 'SPGI', 'AXP', 'V', 'MA',
    # Industrials
    'CAT', 'DE', 'UPS', 'FDX', 'BA', 'HON', 'GE', 'LMT', 'RTX',
    # Energy
    'XOM', 'CVX',
    # Automotive
    'TSLA',
]

# Remove duplicates
TRAINING_STOCKS = list(dict.fromkeys(TRAINING_STOCKS))

print(f"\n{'='*60}")
print(f"ML RETRAINING — ANTI-OVERFITTING FIXES")
print(f"{'='*60}")
print(f"Stocks: {len(TRAINING_STOCKS)}")
print(f"Period: 5 years (2020-01-01 to {datetime.now().strftime('%Y-%m-%d')})")
print(f"Changes applied:")
print(f"  ✅ Feature threshold: 0.005 → 0.02")
print(f"  ✅ SMOTE: DISABLED")
print(f"  ✅ Sample capping: Random → Keep most recent")
print(f"  ✅ Purging gap: 10 trading days")
print(f"  ✅ RF depth: 5→4, leaf: 50→80")
print(f"  ✅ GB depth: 3→2, estimators: 150→100")
print(f"{'='*60}\n")

# Initialize
data_dir = Path.home() / '.stocker'
data_fetcher = StockDataFetcher()
pipeline = MLTrainingPipeline(data_fetcher, data_dir)

# Train
start_date = "2020-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")

def progress_callback(msg):
    print(f"  → {msg}")

results = pipeline.train_on_symbols(
    symbols=TRAINING_STOCKS,
    start_date=start_date,
    end_date=end_date,
    strategy="trading",
    progress_callback=progress_callback
)

# Report
print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
if 'error' in results:
    print(f"❌ Error: {results['error']}")
else:
    train_acc = results.get('train_accuracy', 0) * 100
    test_acc = results.get('test_accuracy', 0) * 100
    gap = train_acc - test_acc
    total = results.get('total_samples', 0)
    
    print(f"Total samples: {total:,}")
    print(f"Train accuracy: {train_acc:.1f}%")
    print(f"Test accuracy:  {test_acc:.1f}%")
    print(f"Gap:            {gap:.1f}pp {'✅' if gap < 10 else '⚠️' if gap < 15 else '❌'}")
    
    if results.get('cv_mean'):
        cv_mean = results.get('cv_mean', 0) * 100
        cv_std = results.get('cv_std', 0) * 100
        print(f"Cross-val:      {cv_mean:.1f}% (±{cv_std:.1f}%)")
    
    # Regime results
    if results.get('regime_results'):
        print(f"\nRegime-specific results:")
        for regime, r in results['regime_results'].items():
            r_train = r.get('train_accuracy', 0) * 100
            r_test = r.get('test_accuracy', 0) * 100
            r_gap = r_train - r_test
            print(f"  {regime}: Train={r_train:.1f}%, Test={r_test:.1f}%, Gap={r_gap:.1f}pp")

print(f"{'='*60}")
