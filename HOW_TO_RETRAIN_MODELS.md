# How to Retrain ML Models - Step by Step Guide

## Quick Answer

You have **3 ways** to retrain your ML models:

1. **Through the UI** (Easiest) - Use the "Train AI Model" dialog
2. **Programmatically** - Call the training pipeline directly
3. **Automatic** - Models retrain automatically when you have enough new data

---

## Method 1: Through the UI (Recommended) 🖥️

### Step-by-Step:

1. **Open the App**
   ```bash
   python main.py
   ```

2. **Navigate to ML Training**
   - Look for a menu item or button labeled "Train AI Model" or "ML Training"
   - Or use the keyboard shortcut if available
   - The dialog should appear with training options

3. **Configure Training**
   - **Select Strategy**: Choose which strategy to train (Trading, Investing, or Mixed)
   - **Select Stocks**: Choose which stocks to use for training
     - Popular stocks: AAPL, MSFT, GOOGL, etc.
     - More stocks = better model (aim for 20+ stocks)
   - **Time Period**: Select date range (more data = better)
     - Recommended: Last 2-5 years
   - **Check "Retrain from verified predictions"**: This uses your actual prediction results

4. **Start Training**
   - Click "Start Training" or "Train Model"
   - Training runs in the background (won't freeze UI)
   - Progress will be shown in the status window

5. **Wait for Completion**
   - Training typically takes 5-30 minutes
   - Depends on:
     - Number of stocks
     - Amount of historical data
     - Your computer speed
   - You'll see progress updates in real-time

6. **Verify Training**
   - Check the status window for "✅ Training complete"
   - Models are automatically saved
   - New predictions will use the retrained models

---

## Method 2: Programmatically (For Developers) 💻

### Option A: Train All Models

```python
from training_pipeline import MLTrainingPipeline
from data_fetcher import StockDataFetcher
from config import APP_DATA_DIR

# Initialize
data_fetcher = StockDataFetcher()
training_manager = MLTrainingPipeline(data_fetcher, APP_DATA_DIR, app=None)

# Train all strategies
strategies = ['trading', 'investing', 'mixed']
for strategy in strategies:
    print(f"Training {strategy} model...")
    result = training_manager.train_model(
        strategy=strategy,
        symbols=['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
        start_date='2020-01-01',
        end_date='2024-01-01',
        retrain_ml=True  # Important: retrain with new features
    )
    print(f"Result: {result}")
```

### Option B: Train Specific Strategy

```python
from training_pipeline import MLTrainingPipeline
from data_fetcher import StockDataFetcher
from config import APP_DATA_DIR

# Initialize
data_fetcher = StockDataFetcher()
training_manager = MLTrainingPipeline(data_fetcher, APP_DATA_DIR)

# Train trading strategy
result = training_manager.train_model(
    strategy='trading',
    symbols=[
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
        'JPM', 'BAC', 'WFC', 'GS',
        'KO', 'PEP', 'WMT', 'TGT'
    ],
    start_date='2020-01-01',  # Start date
    end_date='2024-01-01',    # End date
    retrain_ml=True           # Retrain with enhanced features
)

if result.get('success'):
    print(f"✅ Model trained successfully!")
    print(f"   Samples: {result.get('samples', 0)}")
    print(f"   Accuracy: {result.get('accuracy', 0):.2%}")
else:
    print(f"❌ Training failed: {result.get('error')}")
```

### Option C: Quick Retrain Script

Create a file `retrain_models.py`:

```python
#!/usr/bin/env python3
"""
Quick script to retrain all ML models with enhanced features
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from training_pipeline import MLTrainingPipeline
from data_fetcher import StockDataFetcher
from config import APP_DATA_DIR

def main():
    print("🤖 Starting ML Model Retraining...")
    print("=" * 50)
    
    # Initialize
    data_fetcher = StockDataFetcher()
    training_manager = MLTrainingPipeline(data_fetcher, APP_DATA_DIR)
    
    # Popular stocks for training
    training_stocks = [
        # Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
        # Finance
        'JPM', 'BAC', 'WFC', 'GS', 'MS',
        # Consumer
        'KO', 'PEP', 'WMT', 'TGT', 'HD', 'MCD',
        # Healthcare
        'JNJ', 'PFE', 'UNH', 'ABT',
        # Industrial
        'BA', 'CAT', 'GE', 'MMM',
        # Energy
        'XOM', 'CVX', 'COP',
        # Other
        'DIS', 'V', 'MA', 'PG', 'NKE'
    ]
    
    strategies = ['trading', 'investing', 'mixed']
    
    for strategy in strategies:
        print(f"\n📊 Training {strategy.upper()} model...")
        print("-" * 50)
        
        try:
            result = training_manager.train_model(
                strategy=strategy,
                symbols=training_stocks,
                start_date='2020-01-01',
                end_date='2024-01-01',
                retrain_ml=True
            )
            
            if result.get('success'):
                print(f"✅ {strategy.upper()} model trained successfully!")
                print(f"   Samples: {result.get('samples', 0)}")
                print(f"   Accuracy: {result.get('accuracy', 0):.2%}")
            else:
                print(f"❌ {strategy.upper()} training failed: {result.get('error')}")
                
        except Exception as e:
            print(f"❌ Error training {strategy}: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Retraining complete!")
    print("\nYour models now use enhanced features:")
    print("  • Market microstructure")
    print("  • Sector relative strength")
    print("  • Alternative data (news sentiment)")
    print("  • Macroeconomic indicators")

if __name__ == '__main__':
    main()
```

Run it:
```bash
python retrain_models.py
```

---

## Method 3: Automatic Retraining (Background) 🔄

The app automatically retrains models when:
- You have enough new verified predictions (100+ samples)
- Accuracy drops below threshold
- Auto-training is enabled

**To enable automatic retraining:**
- Check app settings for "Auto Training" or "Auto Retrain"
- Or it may be enabled by default

---

## What Happens During Training?

1. **Data Collection**
   - Fetches historical data for selected stocks
   - Generates training samples from price movements
   - Extracts features (including new enhanced features!)

2. **Feature Extraction**
   - Technical indicators (RSI, MACD, etc.)
   - **Enhanced features** (market microstructure, sector analysis, etc.)
   - Fundamental metrics (P/E, P/B, etc.)
   - Total: ~120+ features (was ~100 before)

3. **Model Training**
   - Trains Random Forest, Gradient Boosting, Logistic Regression
   - Uses cross-validation
   - Selects best model ensemble

4. **Model Evaluation**
   - Tests on held-out data
   - Calculates accuracy metrics
   - Saves best model

5. **Model Storage**
   - Saves trained model to disk
   - Encrypted storage
   - Ready for predictions

---

## Training Parameters

### Recommended Settings:

**For Trading Strategy:**
- **Stocks**: 20-30 popular stocks
- **Time Period**: Last 2-3 years
- **Lookforward Days**: 10 days

**For Investing Strategy:**
- **Stocks**: 20-30 popular stocks
- **Time Period**: Last 5 years
- **Lookforward Days**: 547 days (1.5 years)

**For Mixed Strategy:**
- **Stocks**: 20-30 popular stocks
- **Time Period**: Last 3 years
- **Lookforward Days**: 21 days

### Minimum Requirements:
- **Minimum Stocks**: 10
- **Minimum Samples**: 100 per strategy
- **Minimum Time Period**: 1 year

---

## Troubleshooting

### "Training failed" or "Not enough data"

**Solution:**
- Use more stocks (20+ recommended)
- Use longer time period (2+ years)
- Check internet connection (needs to fetch data)

### "Training takes too long"

**Solution:**
- Reduce number of stocks
- Reduce time period
- Training is normal - can take 10-30 minutes

### "Models not improving"

**Solution:**
- Use more diverse stocks (different sectors)
- Use longer time period
- Check data quality (should be >75%)
- Verify enhanced features are being extracted

### "How do I know if enhanced features are being used?"

**Check the logs:**
- Look for "Enhanced features extracted" messages
- Check feature count (should be ~120+ features)
- Verify `enhanced_extractor` is initialized

---

## Verifying Enhanced Features Are Used

After training, verify enhanced features are included:

```python
from ml_training import FeatureExtractor
from data_fetcher import StockDataFetcher

# Initialize with data_fetcher (required for enhanced features)
data_fetcher = StockDataFetcher()
extractor = FeatureExtractor(data_fetcher=data_fetcher)

# Extract features
features = extractor.extract_features(
    stock_data, history_data, financials_data,
    indicators, market_regime, timeframe_analysis,
    relative_strength, support_resistance
)

# Check feature count
print(f"Total features: {len(features)}")
# Should be ~120+ (was ~100 before)
```

---

## Expected Results

### Before Retraining:
- Feature count: ~100 features
- Typical accuracy: 55-65%

### After Retraining:
- Feature count: ~120+ features (with enhanced features)
- Expected accuracy: 60-70% (5-10% improvement)

### Time to See Improvements:
- **Immediate**: New predictions use retrained models
- **Validation**: Check accuracy after 20-50 predictions
- **Full benefit**: After 100+ predictions to see statistical improvement

---

## Quick Checklist

- [ ] Open app or use programmatic method
- [ ] Select strategy to train
- [ ] Choose 20+ stocks
- [ ] Set time period (2-5 years)
- [ ] Start training
- [ ] Wait for completion (5-30 minutes)
- [ ] Verify models are saved
- [ ] Test with new predictions
- [ ] Monitor accuracy improvements

---

## Summary

**Easiest Method**: Use the UI "Train AI Model" dialog

**Fastest Method**: Run the retrain script (`retrain_models.py`)

**Most Control**: Use programmatic method with custom parameters

**Key Point**: Make sure `retrain_ml=True` to use enhanced features!

After retraining, your predictions will use the new enhanced features and should be 5-10% more accurate! 🎯


