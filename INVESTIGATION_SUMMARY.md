# Investigation Summary - Stocker ML System Issues

## Date: 2026-01-10

### Issues Investigated

1. **Benchmarking Baseline Calculation Error** ⚠️ **CRITICAL - FIXED**
2. **Log Message Displaying Incorrect Baseline** ⚠️ **CRITICAL - FIXED**
3. **Overfitting Warnings** ⚠️ **MODERATE - NEEDS ATTENTION**
4. **HOLD Prediction Bias** ⚠️ **MODERATE - MONITORING**

---

## 1. Benchmarking Baseline Calculation Error ✅ FIXED

### Problem
The random baseline was incorrectly calculated as **33.3%** (3-class) instead of **50%** (binary), leading to misleading comparison metrics.

### Root Cause
The `calculate_baselines()` method in `model_benchmarking.py` was inferring the classification type (binary vs 3-class) solely from the predictions:
- If `HOLD` was present in predictions, it assumed 3-class (33.3% baseline)
- However, the ML model is actually **binary** (UP/DOWN), and `HOLD` is returned when confidence is below the threshold (60%)

### Impact
- **Model accuracy**: 58.5%
- **Incorrect random baseline**: 33.3%
- **Wrong difference calculation**: 25.2 percentage points (should be 8.5%)
- **Misleading improvement %**: 75.7% (should be 17%)

### Fix Applied
1. **Metadata-based detection**: Added `_check_if_binary_classification()` method that reads ML model metadata to determine true classification type
2. **Improved fallback logic**: If metadata check fails, uses smarter inference:
   - If HOLD is <20% of predictions → binary (50% baseline)
   - If BUY and SELL are roughly balanced (40-60%) → binary
   - Otherwise → 3-class (33.3% baseline)
3. **Updated log message**: Now displays the actual random baseline instead of hardcoded "50%"

### Files Modified
- `model_benchmarking.py`: Updated `calculate_baselines()` and added `_check_if_binary_classification()`
- `continuous_backtester.py`: Updated to pass strategy parameter and display correct baseline

---

## 2. Log Message Displaying Incorrect Baseline ✅ FIXED

### Problem
The log message was hardcoded to say "Random 50%" even when the actual baseline was 33.3%, causing confusion.

### Fix Applied
Updated the log message in `continuous_backtester.py` to display the actual random baseline from stored baselines:
```python
random_baseline = baselines.get('random', {}).get('accuracy', 50)
logger.info(f"📊 Benchmarking ({strategy}): Model {strategy_acc:.1f}% vs Random {random_baseline:.1f}% "
           f"(diff: {diff:+.1f}pp, improvement: {improvement:+.1f}%)")
```

---

## 3. Overfitting Warnings ⚠️ NEEDS ATTENTION

### Problem
The logs show severe overfitting:
- **Training accuracy**: 93.5%
- **Test accuracy**: 75.0%
- **Gap**: 18.5% (SEVERE)

And moderate overfitting:
- **Training accuracy**: 84.5%
- **Test accuracy**: 59.9%
- **Gap**: 24.6% (MODERATE)

### Root Cause Analysis
1. **Feature Selection**: May be selecting features that work well on training data but don't generalize
2. **Model Complexity**: The ensemble model (LightGBM + XGBoost + RandomForest) may be too complex for the available data
3. **Data Size**: If training data is limited, the model may memorize patterns instead of learning generalizable rules
4. **Regime Models**: Separate models for bull/bear/sideways markets may be overfitting to specific market conditions

### Current Mitigation
The system already implements:
- ✅ **RFE Feature Selection**: Reduces feature count to avoid overfitting
- ✅ **Hyperparameter Tuning**: Adjusts parameters to balance bias-variance tradeoff
- ✅ **Class Weighting**: Uses `class_weight='balanced'` to handle imbalanced data
- ✅ **SMOTE**: Synthetically balances classes
- ✅ **Regularization**: L1/L2 penalties in XGBoost and LightGBM

### Recommendations
1. **Increase Training Data**: Collect more historical data to improve generalization
2. **Cross-Validation**: Use k-fold cross-validation during training instead of single train-test split
3. **Early Stopping**: Implement early stopping in XGBoost/LightGBM to prevent overfitting
4. **Feature Engineering**: Remove highly correlated features that may cause overfitting
5. **Model Simplification**: Consider using simpler models (e.g., Logistic Regression) for comparison
6. **Data Augmentation**: Generate more training samples using techniques like time-series augmentation

### Status
⚠️ **Monitoring**: The system already logs overfitting warnings. Continue monitoring and collect more data to improve generalization.

---

## 4. HOLD Prediction Bias ⚠️ MONITORING

### Problem
The model is frequently predicting `HOLD` with low confidence (~53%), even after implementing:
- Confidence threshold of 60%
- SMOTE for class balancing
- Binary classification conversion

### Root Cause Analysis
1. **Low Confidence Threshold**: The threshold is set to 60%, but many predictions hover around 53-55%, causing them to be converted to HOLD
2. **Market Conditions**: In uncertain markets, the model may correctly identify that signals are unclear
3. **Feature Quality**: If features are not discriminative enough, the model may struggle to make confident predictions
4. **Class Imbalance**: Despite SMOTE, there may still be imbalance in the underlying data

### Current Behavior
- Predictions with confidence <60% are converted to HOLD
- This is actually **correct behavior** - it's better to be cautious than make bad predictions
- However, if HOLD predictions are too frequent (>50%), the model may be too conservative

### Recommendations
1. **Lower Confidence Threshold**: Consider lowering from 60% to 55% to allow more predictions
   - **Risk**: May increase false positives
   - **Benefit**: More actionable predictions
2. **Feature Quality**: Investigate which features are most discriminative and focus on those
3. **Market Regime Detection**: Use regime models more aggressively - they may provide higher confidence
4. **Ensemble Calibration**: Calibrate ensemble probabilities to better reflect true confidence

### Status
⚠️ **Monitoring**: Continue tracking HOLD prediction frequency. If it stays >50%, consider adjusting the confidence threshold or investigating feature quality.

---

## Summary of Fixes

### ✅ Completed
1. **Fixed benchmarking baseline calculation** - Now correctly uses 50% for binary classification
2. **Fixed log message** - Now displays actual baseline instead of hardcoded value
3. **Improved binary classification detection** - Checks ML metadata first, then uses smart fallback

### ⚠️ Ongoing Monitoring
1. **Overfitting** - System logs warnings; continue monitoring and collect more data
2. **HOLD Prediction Bias** - System is behaving correctly (conservative), but monitor frequency

### 📊 Expected Improvements
After the baseline fix:
- **More accurate comparisons**: Model vs Random will now show correct differences (e.g., 8.5pp instead of 25.2pp)
- **Better decision-making**: Benchmarking reports will be more reliable for model evaluation
- **Clearer logs**: Users will see actual baselines instead of misleading hardcoded values

---

## Next Steps

1. ✅ **Run new backtest** - Verify baseline fix works correctly
2. ⚠️ **Monitor overfitting** - Continue collecting data and watch for improvement
3. ⚠️ **Adjust confidence threshold** - If HOLD predictions are too frequent, consider lowering threshold
4. 📊 **Review feature importance** - Identify which features are most valuable for predictions
5. 🔄 **Periodic retraining** - Continue regular retraining as more data becomes available

---

## Technical Details

### Binary Classification Detection Logic
```python
# Priority 1: Check ML model metadata
is_binary = metadata.get('use_binary_classification', False)

# Priority 2: Smart inference from predictions
if is_binary is None:
    if HOLD < 20% of predictions:
        is_binary = True  # Binary with confidence threshold
    elif BUY and SELL are balanced (40-60%):
        is_binary = True  # Binary classification
    else:
        is_binary = False  # 3-class classification
```

### Baseline Calculation
- **Binary**: 50% random accuracy (2 choices: UP or DOWN)
- **3-Class**: 33.3% random accuracy (3 choices: BUY, SELL, or HOLD)

---

**Investigation completed**: 2026-01-10
**Status**: ✅ Critical issues fixed, moderate issues under monitoring
