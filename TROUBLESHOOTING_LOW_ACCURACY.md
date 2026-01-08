# 🔧 Troubleshooting: Low Accuracy After Retraining

## Problem: Retrained Models Still Have Bad Accuracy

If you've retrained your models but accuracy is still low (<50%), here's a systematic approach to diagnose and fix the issue.

---

## Step 1: Diagnose the Problem 🔍

### Check Current Accuracy

```python
# Check your current accuracy
from predictions_tracker import PredictionsTracker
from config import APP_DATA_DIR

tracker = PredictionsTracker(APP_DATA_DIR)
stats = tracker.get_accuracy_stats()

print(f"Overall Accuracy: {stats.get('overall_accuracy', 0)*100:.1f}%")
print(f"Total Predictions: {stats.get('total_predictions', 0)}")
print(f"Correct: {stats.get('correct', 0)}")
print(f"Incorrect: {stats.get('incorrect', 0)}")
```

### What's "Bad" Accuracy?

- **< 40%**: Critical - Models are worse than random
- **40-50%**: Poor - Models barely better than random
- **50-55%**: Below expectations - Should be better
- **55-60%**: Acceptable but could improve
- **> 60%**: Good performance

---

## Step 2: Common Causes & Solutions

### Cause 1: Insufficient Training Data ❌

**Symptoms:**
- Training completes quickly (< 5 minutes)
- "Not enough samples" warnings
- Model accuracy similar to random guessing

**Diagnosis:**
```python
# Check training data
from training_pipeline import MLTrainingPipeline
from data_fetcher import StockDataFetcher
from config import APP_DATA_DIR

data_fetcher = StockDataFetcher()
training = MLTrainingPipeline(data_fetcher, APP_DATA_DIR)

# Check how many samples you have
result = training.train_model(
    strategy='trading',
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    start_date='2020-01-01',
    end_date='2024-01-01',
    retrain_ml=True
)

print(f"Samples: {result.get('samples', 0)}")
# Need at least 500+ samples for good training
```

**Solutions:**

1. **Use More Stocks** (Most Important!)
   ```python
   # Instead of 3 stocks, use 20-30
   symbols = [
       'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
       'JPM', 'BAC', 'WFC', 'GS', 'MS',
       'KO', 'PEP', 'WMT', 'TGT', 'HD', 'MCD',
       'JNJ', 'PFE', 'UNH', 'ABT',
       'BA', 'CAT', 'GE', 'MMM',
       'XOM', 'CVX', 'COP',
       'DIS', 'V', 'MA', 'PG', 'NKE'
   ]
   ```

2. **Use Longer Time Period**
   ```python
   # Instead of 1 year, use 3-5 years
   start_date = '2019-01-01'  # 5 years
   end_date = '2024-01-01'
   ```

3. **Check Data Availability**
   ```python
   # Verify you're getting data
   for symbol in symbols:
       data = data_fetcher.fetch_stock_history(symbol, period='5y')
       if not data or len(data.get('data', [])) < 100:
           print(f"⚠️ {symbol}: Insufficient data")
   ```

**Expected Improvement:** +10-15% accuracy

---

### Cause 2: Poor Data Quality ❌

**Symptoms:**
- Missing data points
- Outliers or invalid values
- Data quality score < 75%

**Diagnosis:**
```python
from data_fetcher import StockDataFetcher

data_fetcher = StockDataFetcher()
stock_data = data_fetcher.fetch_stock_data('AAPL')

# Check quality report
quality = stock_data.get('quality_report', {})
score = quality.get('overall_score', 100)

print(f"Data Quality Score: {score:.1f}%")
if score < 75:
    print("⚠️ Poor data quality detected!")
    print(f"Issues: {quality.get('issues', [])}")
```

**Solutions:**

1. **Use Data Quality Checks** (Already integrated)
   - The system automatically validates data
   - Check quality reports before training

2. **Use Multiple Data Sources**
   ```python
   # System automatically falls back to alternative sources
   # Check if fallback is working
   ```

3. **Clean Data Manually**
   ```python
   from data_quality import DataQualityChecker
   
   checker = DataQualityChecker()
   cleaned_data = checker.clean_data(raw_data)
   ```

**Expected Improvement:** +5-10% accuracy

---

### Cause 3: Wrong Features or Feature Engineering ❌

**Symptoms:**
- Feature count is low (< 100)
- Enhanced features not being used
- Model doesn't learn meaningful patterns

**Diagnosis:**
```python
from ml_training import FeatureExtractor
from data_fetcher import StockDataFetcher

data_fetcher = StockDataFetcher()
extractor = FeatureExtractor(data_fetcher=data_fetcher)

# Extract features and check count
features = extractor.extract_features(...)
print(f"Total features: {len(features)}")
# Should be ~120+ with enhanced features
```

**Solutions:**

1. **Verify Enhanced Features Are Used**
   ```python
   # Check if enhanced_extractor is initialized
   if hasattr(extractor, 'enhanced_extractor'):
       print("✅ Enhanced features enabled")
       enhanced = extractor.enhanced_extractor.extract_all_enhanced_features(...)
       print(f"Enhanced features: {len(enhanced)}")
   else:
       print("❌ Enhanced features NOT enabled!")
   ```

2. **Check Feature Importance** (if available)
   ```python
   # Some models provide feature importance
   # Check which features are most important
   # Remove or reduce weight of unimportant features
   ```

3. **Add More Features**
   - Market regime indicators
   - Sector relative strength
   - Volume profile analysis
   - Support/resistance levels

**Expected Improvement:** +5-10% accuracy

---

### Cause 4: Overfitting ❌

**Symptoms:**
- Training accuracy is high (>80%)
- But validation/walk-forward accuracy is low (<50%)
- Model performs well on training data but poorly on new data

**Diagnosis:**
```python
# Run walk-forward backtest
from walk_forward_backtester import WalkForwardBacktester
from config import APP_DATA_DIR

backtester = WalkForwardBacktester(APP_DATA_DIR)
result = backtester.run_walk_forward_test(...)

train_acc = result.get('train_accuracy', 0)
test_acc = result.get('test_accuracy', 0)

if train_acc > test_acc + 0.20:  # 20% gap
    print("⚠️ Overfitting detected!")
    print(f"Training: {train_acc*100:.1f}%, Test: {test_acc*100:.1f}%")
```

**Solutions:**

1. **Use Walk-Forward Validation**
   - Always validate with walk-forward backtesting
   - Don't trust training accuracy alone

2. **Reduce Model Complexity**
   ```python
   # Use simpler models or reduce parameters
   # Random Forest: reduce max_depth, n_estimators
   # Gradient Boosting: reduce learning_rate
   ```

3. **Add Regularization**
   - L1/L2 regularization
   - Dropout (for neural networks)
   - Early stopping

4. **Use More Diverse Data**
   - Different stocks
   - Different time periods
   - Different market conditions

**Expected Improvement:** +10-20% validation accuracy

---

### Cause 5: Market Conditions ❌

**Symptoms:**
- Accuracy varies significantly by time period
- Works well in trending markets, poorly in sideways markets
- Some stocks work, others don't

**Diagnosis:**
```python
# Check accuracy by market regime
# Bull markets: easier to predict
# Bear markets: harder to predict
# Sideways markets: very hard to predict
```

**Solutions:**

1. **Train on Multiple Market Regimes**
   ```python
   # Include bull, bear, and sideways markets
   start_date = '2018-01-01'  # Includes 2018 correction
   end_date = '2024-01-01'    # Includes recent markets
   ```

2. **Use Market Regime Detection**
   ```python
   # The system already detects market regimes
   # Models should adapt to different regimes
   ```

3. **Accept Lower Accuracy in Difficult Markets**
   - Sideways markets: 45-55% is acceptable
   - Trending markets: 60-70% is achievable
   - Volatile markets: Lower accuracy is expected

**Expected Improvement:** More consistent performance

---

### Cause 6: Wrong Strategy/Timeframe Mismatch ❌

**Symptoms:**
- Trading strategy used for long-term predictions
- Investing strategy used for short-term predictions
- Lookforward days don't match strategy

**Diagnosis:**
```python
# Check strategy and lookforward days match
# Trading: 10 days lookforward
# Mixed: 21 days lookforward
# Investing: 547 days lookforward
```

**Solutions:**

1. **Match Strategy to Timeframe**
   ```python
   # Trading: Use for 1-2 week predictions
   # Mixed: Use for 1-4 week predictions
   # Investing: Use for 6+ month predictions
   ```

2. **Use Correct Lookforward Days**
   ```python
   if strategy == 'trading':
       lookforward_days = 10
   elif strategy == 'mixed':
       lookforward_days = 21
   else:  # investing
       lookforward_days = 547
   ```

**Expected Improvement:** +10-15% accuracy

---

### Cause 7: Model Architecture Issues ❌

**Symptoms:**
- Models converge too quickly
- All predictions are the same
- Model doesn't learn from data

**Solutions:**

1. **Try Different Models**
   ```python
   # System uses ensemble: Random Forest, Gradient Boosting, Logistic Regression
   # If one model fails, others should compensate
   ```

2. **Adjust Hyperparameters**
   ```python
   # Increase model complexity if underfitting
   # Decrease if overfitting
   # Tune learning rate, depth, etc.
   ```

3. **Use Ensemble Methods**
   - Combine multiple models
   - Weight predictions by confidence
   - Use voting or averaging

**Expected Improvement:** +5-10% accuracy

---

## Step 3: Systematic Improvement Plan 📋

### Phase 1: Data Quality (Do First!)

1. ✅ Verify data quality > 75%
2. ✅ Use 20+ stocks from different sectors
3. ✅ Use 3-5 years of historical data
4. ✅ Check for missing data points

**Expected:** +5-10% accuracy

---

### Phase 2: Feature Engineering

1. ✅ Verify enhanced features are enabled
2. ✅ Check feature count is ~120+
3. ✅ Verify market regime detection works
4. ✅ Check sector relative strength features

**Expected:** +5-10% accuracy

---

### Phase 3: Training Optimization

1. ✅ Use correct lookforward days for strategy
2. ✅ Train on multiple market regimes
3. ✅ Use walk-forward validation
4. ✅ Avoid overfitting

**Expected:** +5-15% accuracy

---

### Phase 4: Model Tuning

1. ✅ Try different model parameters
2. ✅ Use ensemble methods
3. ✅ Adjust confidence thresholds
4. ✅ Filter low-confidence predictions

**Expected:** +3-8% accuracy

---

## Step 4: When to Accept Lower Accuracy ⚠️

### Realistic Expectations

**Stock Market Prediction is Hard!**

- **Random guessing**: 33% (3 actions: BUY, SELL, HOLD)
- **Good models**: 55-65% accuracy
- **Excellent models**: 65-75% accuracy
- **Perfect models**: Don't exist (markets are unpredictable)

### Accept Lower Accuracy If:

1. **Market Conditions**
   - Sideways/choppy markets: 45-55% is acceptable
   - High volatility: Lower accuracy expected
   - Unusual events: Models can't predict black swans

2. **Stock Type**
   - Small caps: Harder to predict (50-60%)
   - Large caps: Easier to predict (60-70%)
   - Penny stocks: Very hard (45-55%)

3. **Timeframe**
   - Short-term (1-5 days): Harder (50-60%)
   - Medium-term (1-4 weeks): Moderate (55-65%)
   - Long-term (6+ months): Easier (60-70%)

4. **You've Tried Everything**
   - Used 30+ stocks
   - Used 5+ years of data
   - Verified enhanced features
   - Used walk-forward validation
   - Still getting 50-55% accuracy

**This might be the best you can achieve!**

---

## Step 5: Alternative Approaches 🎯

### If Accuracy Still Low After Everything:

1. **Focus on Risk Management**
   - Use stop-losses
   - Position sizing
   - Portfolio diversification
   - Even 50% accuracy can be profitable with good risk management!

2. **Filter Predictions**
   - Only act on high-confidence predictions (>70%)
   - Ignore low-confidence predictions
   - This improves effective accuracy

3. **Use Multiple Strategies**
   - Combine trading + investing strategies
   - Use ensemble of models
   - Diversify prediction methods

4. **Focus on Specific Stocks**
   - Some stocks are easier to predict
   - Focus on stocks with >60% accuracy
   - Avoid stocks with <45% accuracy

5. **Use Hybrid Approach**
   - Combine ML predictions with rule-based analysis
   - Use ML for confirmation, not primary signal
   - Weight predictions by confidence

---

## Quick Diagnostic Checklist ✅

Run through this checklist:

- [ ] **Data**: Using 20+ stocks, 3+ years of data?
- [ ] **Quality**: Data quality score > 75%?
- [ ] **Features**: Feature count ~120+?
- [ ] **Enhanced**: Enhanced features enabled?
- [ ] **Strategy**: Correct lookforward days for strategy?
- [ ] **Validation**: Walk-forward accuracy checked?
- [ ] **Overfitting**: Training vs test accuracy gap < 20%?
- [ ] **Market**: Training includes multiple market regimes?
- [ ] **Models**: Multiple models in ensemble?
- [ ] **Confidence**: Filtering low-confidence predictions?

**If all checked but still low accuracy:**
- Accept that 50-55% might be your limit
- Focus on risk management
- Filter predictions by confidence
- Use hybrid approach

---

## Example: Complete Retraining with All Fixes

```python
from training_pipeline import MLTrainingPipeline
from data_fetcher import StockDataFetcher
from config import APP_DATA_DIR

# Initialize
data_fetcher = StockDataFetcher()
training = MLTrainingPipeline(data_fetcher, APP_DATA_DIR)

# Use MANY stocks (30+)
symbols = [
    # Tech (8)
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
    # Finance (5)
    'JPM', 'BAC', 'WFC', 'GS', 'MS',
    # Consumer (6)
    'KO', 'PEP', 'WMT', 'TGT', 'HD', 'MCD',
    # Healthcare (4)
    'JNJ', 'PFE', 'UNH', 'ABT',
    # Industrial (4)
    'BA', 'CAT', 'GE', 'MMM',
    # Energy (3)
    'XOM', 'CVX', 'COP',
    # Other (5)
    'DIS', 'V', 'MA', 'PG', 'NKE'
]

# Use LONG time period (5 years)
start_date = '2019-01-01'
end_date = '2024-01-01'

# Train all strategies
for strategy in ['trading', 'investing', 'mixed']:
    print(f"\n{'='*60}")
    print(f"Training {strategy.upper()} model...")
    print(f"{'='*60}")
    
    result = training.train_model(
        strategy=strategy,
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        retrain_ml=True  # Use enhanced features
    )
    
    if result.get('success'):
        print(f"✅ Success!")
        print(f"   Samples: {result.get('samples', 0)}")
        print(f"   Accuracy: {result.get('accuracy', 0)*100:.1f}%")
    else:
        print(f"❌ Failed: {result.get('error')}")

# Validate with walk-forward
print(f"\n{'='*60}")
print("Running walk-forward validation...")
print(f"{'='*60}")

# ... run walk-forward backtest ...
```

---

## Summary

**If accuracy is still low after retraining:**

1. ✅ **Check data**: 20+ stocks, 3+ years
2. ✅ **Check quality**: Data quality > 75%
3. ✅ **Check features**: ~120+ features, enhanced enabled
4. ✅ **Check validation**: Walk-forward accuracy
5. ✅ **Check overfitting**: Training vs test gap
6. ✅ **Accept limits**: 50-55% might be realistic
7. ✅ **Focus on risk**: Good risk management > high accuracy

**Remember**: Even 50% accuracy can be profitable with proper risk management! 🎯
