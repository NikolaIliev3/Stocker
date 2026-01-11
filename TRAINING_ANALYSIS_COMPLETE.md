# Training & Backtesting Analysis - Complete Review

## Date: 2026-01-10 (After Fresh Training)

### ✅ Benchmarking Fix Confirmed Working

The benchmarking fix is working correctly! The log shows:
```
📊 Benchmarking (trading): Model 54.8% vs Random 50.0% (diff: +4.8pp, improvement: +9.5%)
```

This is **correct** - the model is being compared to the proper 50% binary baseline, and the difference (+4.8pp) is accurate.

---

## 🎯 ML Training Results

### Training Metrics
- **Training Samples**: 3,800 (20 symbols × 190 samples each)
- **Training Accuracy**: 85.3%
- **Test Accuracy**: 63.8%
- **Cross-Validation**: 61.4% (±4.5%)
- **Overfitting Gap**: 21.5% (MODERATE)
- **Strategy**: Trading
- **Binary Classification**: ✅ Enabled
- **SMOTE Balancing**: ✅ Enabled
- **Feature Selection**: 70/130 features (53.8% retained)

### ⚠️ Critical Issue: All Samples Classified as "Sideways"

**Problem:**
- **Bull samples**: 0 (need 100 minimum)
- **Bear samples**: 0 (need 100 minimum)
- **Sideways samples**: 2,317 (all samples)

**Root Cause:**
The regime detector (`MarketRegimeDetector`) uses very strict criteria:
- **Bull**: `current_price > sma_50 > sma_200 AND trend_strength > 0.1`
- **Bear**: `current_price < sma_50 < sma_200 AND trend_strength < -0.1`
- **Otherwise**: Defaults to "sideways"

Additionally, regime detection requires SPY (market index) data, and if it can't fetch it properly or the conditions don't meet the strict thresholds, it defaults to sideways.

**Impact:**
- Only 1 regime model is trained (sideways) instead of 3 (bull/bear/sideways)
- Missing specialized models for different market conditions
- The regime-specific training feature is essentially disabled

**Recommendations:**
1. **Relax thresholds**: Lower the trend_strength threshold (0.05 instead of 0.1)
2. **Add fallback logic**: If SPY data unavailable, use the stock's own price action to detect regime
3. **Better regime detection**: Use relative performance vs SPY, volatility-based detection, or broader time windows
4. **Minimum sample adjustment**: Consider training regime models even with <100 samples (but with regularization)

---

## 📊 Backtesting Results

### Performance Metrics
- **Tests Run**: 100 (accumulated 800 total in history)
- **Overall Accuracy**: 54.8% (438/800 correct)
- **Trading Strategy**: 54.8% (438/800)
- **Baseline Comparison**: +4.8pp vs Random 50% (CORRECT - benchmarking fix working!)

### ⚠️ Issues Identified

#### 1. HOLD Predictions Failing
```
🔧 Action pattern detected (trading): HOLD predictions failing 76% of time (31/41 failures)
```

**Analysis:**
- HOLD is predicted when ML confidence < 60%
- 41 HOLD predictions were made
- 31 failed (76% failure rate)
- This suggests the confidence threshold (60%) may be too low, OR HOLD predictions are inherently less accurate because they're "uncertain" predictions

**Recommendations:**
- Consider raising confidence threshold from 60% to 65-70% to reduce HOLD predictions
- OR: Don't penalize HOLD predictions in backtesting (they're "pass" decisions, not wrong)
- Analyze why HOLD predictions fail: Are they made at the wrong times?

#### 2. Low Backtest Accuracy (54.8%)

**Analysis:**
- 54.8% is only 4.8pp above random (50%)
- Test accuracy during training was 63.8%, but backtest shows 54.8%
- This suggests the model doesn't generalize well to real backtesting scenarios

**Possible Causes:**
1. **Overfitting**: Train (85.3%) vs Test (63.8%) gap of 21.5% suggests overfitting
2. **Data distribution shift**: Training data (2020-2023) vs backtest data (different periods) may have different characteristics
3. **Regime mismatch**: All training was "sideways", but backtest periods may have different regimes
4. **Feature quality**: Features may not be discriminative enough in real market conditions

**Recommendations:**
1. **Reduce overfitting**: Increase regularization, reduce model complexity
2. **More diverse training**: Include more time periods and market conditions
3. **Better feature engineering**: Focus on features that generalize across market regimes
4. **Ensemble calibration**: Calibrate probabilities to better reflect true confidence

---

## 🔄 Model Retraining from Backtest Results

### Retraining Attempt
- **Samples**: 100 (from backtest results)
- **Training Accuracy**: 92.5%
- **Test Accuracy**: 55.6%
- **Overfitting Gap**: 36.9% (SEVERE)
- **Cross-Validation**: 75.5% (±18.0% - very high variance!)

**Result:**
```
⚠️ New model performs worse: Accuracy dropped by 15.9% (from 71.4% to 55.6%)
❌ Rejected new version 20260110_173948 due to performance degradation
✅ Rolled back to version 20260110_171144
```

**Analysis:**
- The retrained model was rejected correctly (versioning system working!)
- 100 samples is too small for training (causes severe overfitting)
- The system correctly prevented deployment of a worse model

**Recommendations:**
- **Minimum sample threshold**: Only retrain from backtest if >500 samples accumulated
- **Accumulate more data**: Wait until sufficient backtest results before retraining
- **Incremental learning**: Consider online learning or fine-tuning instead of full retraining

---

## 📈 Positive Findings

### ✅ System Improvements Working
1. **Benchmarking Fix**: ✅ Correctly comparing to 50% binary baseline
2. **Version Management**: ✅ Rejected worse model automatically
3. **SMOTE**: ✅ Balancing classes (UP: 1399, DOWN: 918 → balanced)
4. **Feature Selection**: ✅ Selecting 70/130 features (53.8%)
5. **Cross-Validation**: ✅ Provides robust estimates (61.4% ±4.5%)
6. **Data Validation**: ✅ Detecting class imbalance and low variance features

### ✅ Model Quality Indicators
- **CV Score Close to Test**: 61.4% CV vs 63.8% Test (good sign - model is consistent)
- **Binary Classification Working**: Properly converting BUY/SELL to UP/DOWN
- **Confidence Threshold Working**: HOLD predictions at ~51-53% confidence

---

## 🎯 Action Items

### High Priority
1. **Fix Regime Detection** ⚠️ **CRITICAL**
   - Relax thresholds (0.05 instead of 0.1 for trend_strength)
   - Add fallback when SPY data unavailable
   - Use stock's own price action as backup
   - Lower minimum sample requirement for regime models (50 instead of 100)

2. **Reduce Overfitting** ⚠️ **HIGH**
   - Increase regularization parameters
   - Consider simpler models (Logistic Regression baseline)
   - Add early stopping in XGBoost/LightGBM
   - Increase training data diversity

3. **Improve Backtest Accuracy** ⚠️ **HIGH**
   - Investigate why test accuracy (63.8%) doesn't match backtest (54.8%)
   - Check for data distribution shifts
   - Analyze which backtest periods fail most
   - Consider temporal validation (train on earlier, test on later periods)

### Medium Priority
4. **HOLD Prediction Analysis**
   - Determine if HOLD failures should count against accuracy
   - Analyze timing of HOLD predictions (are they at good times to hold?)
   - Consider raising confidence threshold

5. **Retraining Strategy**
   - Set minimum sample threshold (500+) before retraining
   - Implement incremental learning instead of full retraining
   - Use stratified sampling to ensure balanced retraining data

### Low Priority
6. **Feature Engineering**
   - Analyze which features contribute most to predictions
   - Remove redundant/correlated features
   - Add regime-specific features

7. **Monitoring & Alerts**
   - Set up alerts for overfitting (gap > 20%)
   - Monitor HOLD prediction frequency
   - Track accuracy trends over time

---

## 📊 Summary Statistics

| Metric | Value | Status |
|--------|-------|--------|
| **Training Accuracy** | 85.3% | ⚠️ Too high (overfitting) |
| **Test Accuracy** | 63.8% | ✅ Good baseline |
| **CV Accuracy** | 61.4% ± 4.5% | ✅ Consistent |
| **Backtest Accuracy** | 54.8% | ⚠️ Below test accuracy |
| **Overfitting Gap** | 21.5% | ⚠️ Moderate |
| **Regime Models** | 1/3 (sideways only) | ❌ **CRITICAL ISSUE** |
| **HOLD Failure Rate** | 76% (31/41) | ⚠️ High |
| **Baseline Comparison** | +4.8pp vs 50% | ✅ Correct |
| **Model Versioning** | ✅ Working | ✅ Good |

---

## 🔍 Next Steps

1. **Immediate**: Fix regime detection to classify samples into bull/bear/sideways
2. **Short-term**: Reduce overfitting and improve backtest accuracy
3. **Medium-term**: Analyze HOLD prediction failures and adjust strategy
4. **Long-term**: Implement incremental learning and better retraining strategy

---

**Analysis completed**: 2026-01-10  
**Status**: ⚠️ Critical issue with regime detection identified. Moderate overfitting and low backtest accuracy need attention. Benchmarking fix confirmed working correctly.
