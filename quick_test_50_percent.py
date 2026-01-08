"""Quick test to achieve 50% validation accuracy"""
from ml_training import StockPredictionML
from training_pipeline import MLTrainingPipeline
from data_fetcher import StockDataFetcher
from config import APP_DATA_DIR

# Generate samples - USE EVEN MORE STOCKS for maximum training data
df = StockDataFetcher()
tp = MLTrainingPipeline(df, APP_DATA_DIR, app=None)
samples = []
# Maximum stocks = maximum training data = better accuracy (targeting 60%)
# Added more diverse sectors and market caps
symbols = [
    # Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'INTC',
    # Finance
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA',
    # Healthcare
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO',
    # Consumer
    'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
    # Industrial
    'BA', 'CAT', 'GE', 'HON', 'UPS',
    # Energy
    'XOM', 'CVX', 'SLB', 'COP',
    # Other
    'DIS', 'VZ', 'T', 'CMCSA', 'NEE'
]
for symbol in symbols:
    s = tp.data_generator.generate_training_samples(symbol, '2023-01-01', '2024-12-31', 'trading', 10)
    samples.extend(s)
    print(f"{symbol}: {len(s)} samples")

samples.sort(key=lambda x: x.get('date', '1900-01-01'))
print(f"\nTotal: {len(samples)} samples")

# Train
ml = StockPredictionML(APP_DATA_DIR, 'trading')
results = ml.train(samples, False, False, 'importance')

print(f"\n{'='*70}")
print(f"RESULTS:")
print(f"  Training: {results['train_accuracy']*100:.1f}%")
print(f"  Validation: {results['test_accuracy']*100:.1f}%")
print(f"{'='*70}")

if results['test_accuracy'] >= 0.50:
    print("\n✅ SUCCESS! Achieved 50%+ validation accuracy!")
else:
    print(f"\n❌ Still below 50%. Current: {results['test_accuracy']*100:.1f}%")
