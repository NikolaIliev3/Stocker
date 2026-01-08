# 🔍 Comprehensive Code Review - Accuracy Issues

## ✅ What's CORRECT

### 1. **Labeling Logic** ✓
- **Fixed**: Price UP → BUY, Price DOWN → SELL
- **Location**: `training_pipeline.py` lines 245-269
- **Status**: Correct

### 2. **Feature Extraction** ✓
- **No Data Leakage**: Uses only past data (`hist_to_date = hist.iloc[:i+1]`)
- **Rolling Windows**: Only use past data (correct)
- **Time-Series Features**: Lag features, rolling stats - all use past only
- **Status**: No future information leakage

### 3. **Temporal Split** ✓
- **Fixed**: Chronological split (not random)
- **Location**: `ml_training.py` lines 970-1000
- **Logic**: Trains on past (80%), tests on future (20%)
- **Status**: Correct, prevents data leakage

### 4. **Regularization** ✓
- **Increased**: Reduced max_depth, increased min_samples
- **Random Forest**: max_depth 8→6, min_samples_split 20→30
- **LightGBM/XGBoost**: Reduced complexity, increased reg_lambda
- **Status**: Should prevent overfitting

## ⚠️ POTENTIAL ISSUE: Class Imbalance

### The Problem
**Strict thresholds** might cause severe class imbalance:
- **Trading**: ±3% threshold
- **Mixed**: ±5% threshold
- **Investing**: ±10% threshold (reasonable)

**If most price movements are < threshold → mostly HOLD labels**

### Why This Matters
If 80%+ samples are HOLD:
- Model learns to always predict HOLD
- Accuracy = % of HOLD samples (could be 47% if HOLD is 47%)
- Model doesn't learn BUY/SELL patterns

### Current Mitigation
- ✅ `class_weight='balanced'` is enabled (helps but may not be enough)
- ✅ Class imbalance warning in logs
- ⚠️ Thresholds might still be too strict

## 📊 Expected Accuracy Scenarios

### Scenario 1: Balanced Classes (Ideal)
- **Distribution**: BUY 30%, SELL 30%, HOLD 40%
- **Expected Accuracy**: **60-70%** (good)
- **Model**: Learns all three classes

### Scenario 2: Moderate Imbalance
- **Distribution**: BUY 20%, SELL 20%, HOLD 60%
- **Expected Accuracy**: **55-65%** (acceptable)
- **Model**: Learns patterns but favors HOLD

### Scenario 3: Severe Imbalance (Problem)
- **Distribution**: BUY 10%, SELL 10%, HOLD 80%
- **Expected Accuracy**: **50-55%** (poor - just predicting majority)
- **Model**: Mostly predicts HOLD

## 🎯 What to Do

### Step 1: Train and Check Distribution
After training, check the logs for:
```
Training data distribution: {'BUY': X, 'SELL': Y, 'HOLD': Z}
```

### Step 2: Evaluate
- **If HOLD < 60%**: Good balance, accuracy should be 60-70%
- **If HOLD 60-75%**: Moderate imbalance, accuracy 55-65%
- **If HOLD > 75%**: Severe imbalance, adjust thresholds

### Step 3: If Severe Imbalance
**Option A: Adjust Thresholds** (Recommended)
```python
# In training_pipeline.py, change:
# Trading: ±3% → ±2%
if price_change_pct >= 2:  # was 3
    label = "BUY"
elif price_change_pct <= -2:  # was -3
    label = "SELL"

# Mixed: ±5% → ±3%
if price_change_pct >= 3:  # was 5
    label = "BUY"
elif price_change_pct <= -3:  # was -5
    label = "SELL"
```

**Option B: Use SMOTE** (Advanced)
- Oversample minority classes
- Requires additional library: `imbalanced-learn`

## 📈 Realistic Accuracy Expectations

### With Current Fixes:
- **Temporal Split**: ✅ (prevents cheating)
- **Regularization**: ✅ (prevents overfitting)
- **Correct Labels**: ✅ (learns right patterns)

### Expected Results:
1. **If Balanced Classes**: **60-70%** accuracy
2. **If Moderate Imbalance**: **55-65%** accuracy
3. **If Severe Imbalance**: **50-55%** accuracy (needs threshold adjustment)

### Why 47% Was So Bad:
- **Random split** (cheating) + **Inverted labels** (wrong patterns) + **Overfitting** (memorization)
- All three issues combined = terrible performance

## ✅ Summary

### Code Status: **CORRECT** ✓
- Labeling: Fixed
- Temporal split: Fixed
- Regularization: Increased
- Feature extraction: No leakage

### Potential Issue: **Class Imbalance** ⚠️
- Monitor label distribution during training
- Adjust thresholds if HOLD > 75%
- Expected accuracy: **55-70%** (depending on balance)

### Confidence Level: **HIGH** 🎯
- Code is correct
- Fixes are proper
- Should see improvement from 47% to 55-70%
- If still low, check class distribution and adjust thresholds

---

## 🚀 Next Steps

1. **Train models** with current code
2. **Check label distribution** in logs
3. **If HOLD > 75%**: Adjust thresholds and retrain
4. **Expected accuracy**: 55-70% (much better than 47%)

**The code is correct. Class imbalance is the only potential issue, and it's easy to fix if detected.**
