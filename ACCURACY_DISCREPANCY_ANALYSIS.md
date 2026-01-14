# ML Accuracy Discrepancy Analysis

## Problem
- **Cross-validation accuracy**: 62.9% (61.2% ± 4.1% for bull, 68.3% ± 5.2% for bear)
- **Backtesting accuracy**: 40.0% (40/100 correct)
- **Gap**: 22.9 percentage points

## Root Causes Identified

### 1. **Severe Overfitting** ⚠️ CRITICAL
From logs:
- **Bull regime**: Train 83.7%, Test 56.6%, **Gap: 27.1%**
- **Bear regime**: Train 92.1%, Test 62.8%, **Gap: 29.3%**

**Problem**: Model is memorizing training data, not learning generalizable patterns.

### 2. **Temporal Data Leakage** ⚠️ CRITICAL
- **Training**: Uses `train_test_split` with **random split** (not temporal)
- **Backtesting**: Uses **temporal/historical dates** (realistic scenario)
- **Issue**: Random split allows model to see "future" patterns during training

**Example**: Model trained on 2010-2025 data randomly split might see:
- Training: 2015, 2020, 2012, 2023 (mixed)
- Test: 2018, 2011, 2024 (mixed)

But in reality, when predicting 2018, the model shouldn't have seen 2020+ patterns!

### 3. **Cross-Validation on Training Data** ⚠️
- CV runs on `X_train_scaled, y_train` (training set)
- Should run on full dataset or separate validation set
- This inflates CV scores

### 4. **BUY Prediction Bias** ⚠️
- Model predicts BUY 60/100 times (60%)
- **75% of BUY predictions fail** (45/60 failures)
- Model is overconfident in BUY predictions (avg confidence 76.1%)

### 5. **Class Imbalance**
- Training distribution: UP: 3238, DOWN: 2160 (1.5:1 ratio)
- SMOTE balances this, but model still favors BUY predictions
- Backtest shows: HOLD: 67, SELL: 26, BUY: 7 (heavily skewed)

### 6. **Regime Detection Issues**
- Only **bull** and **bear** regimes detected (0 sideways samples)
- Model trained on 4044 bull + 1354 bear samples
- No sideways regime model exists, so all predictions use bull/bear models

## Solutions Implemented

### ✅ Fix 1: Temporal Split (IMPLEMENTED)
**Problem**: Random split allows model to see "future" patterns during training  
**Solution**: Use date-based temporal split instead of random split
- Training samples sorted by date
- First (1-test_size) samples → training set
- Last test_size samples → test set
- **Status**: ✅ Implemented in `ml_training.py` line 1419-1478

### ✅ Fix 2: Increased Regularization (IMPLEMENTED)
**Problem**: Severe overfitting (27-29% gap between train/test)  
**Solution**: Increased regularization parameters:
- LightGBM: `reg_alpha`: 0.5 → 1.0, `reg_lambda`: 2.0 → 3.0
- XGBoost: `reg_alpha`: 0.5 → 1.0, `reg_lambda`: 2.0 → 3.0
- `min_child_samples`: 30 → 50 (LightGBM)
- `min_child_weight`: 3 → 5 (XGBoost)
- `subsample`: 0.7 → 0.6 (both)
- `colsample_bytree`: 0.7 → 0.6 (both)
- **Status**: ✅ Implemented in `ml_training.py` lines 1696-1719

### ⚠️ Fix 3: Cross-Validation (NOT CHANGED)
**Current**: CV runs on training set (X_train_scaled, y_train)  
**Note**: This is actually correct - CV should run on training data only  
**Issue**: CV scores are optimistic due to overfitting, but test_score is the real metric

### ⚠️ Fix 4: BUY Bias (NEEDS INVESTIGATION)
**Problem**: Model predicts BUY 60% of time, 75% fail  
**Possible Causes**:
- Class imbalance in training (UP: 3238, DOWN: 2160 = 1.5:1)
- SMOTE may not be balancing correctly
- Model learned wrong patterns

**Next Steps**:
- Check if label mapping is correct (BUY→DOWN, SELL→UP)
- Verify SMOTE is working correctly
- Consider adjusting class weights

### ⚠️ Fix 5: Regime Detection (PARTIAL)
**Problem**: Only bull/bear detected, no sideways (0 samples)  
**Current**: Falls back to bull/bear models  
**Impact**: May cause issues when market is sideways

## Expected Impact
- **Temporal split**: Should reduce gap by ~10-15% (prevents data leakage)
- **Increased regularization**: Should reduce overfitting gap from 27% to ~15-20%
- **Combined**: Should improve backtest accuracy from 40% to ~50-55%

## Next Steps
1. ✅ **DONE**: Implement temporal train/test split
2. ✅ **DONE**: Increase regularization parameters  
3. **TODO**: Retrain models with fixes
4. **TODO**: Re-run backtest and compare results
5. **TODO**: Investigate BUY bias if still present after retraining
6. **TODO**: Consider disabling regime models if sideways samples insufficient
