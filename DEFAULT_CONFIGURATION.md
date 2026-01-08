# ✅ DEFAULT CONFIGURATION - WORKING VERSION

**Status:** ✅ WORKING - 70% Backtesting Accuracy  
**Date Saved:** 2026-01-07  
**Performance:** Overall 70.0% (Trading: 70.0%, Mixed: 70.0%)

## ⚠️ IMPORTANT
**DO NOT MODIFY THIS CONFIGURATION** unless you have a specific reason and test thoroughly. This is the proven working configuration.

---

## Model Parameters

### RandomForest Classifier
```python
rf_params = {
    'n_estimators': 100,
    'max_depth': 8,  # DEFAULT
    'min_samples_split': 20,  # DEFAULT
    'min_samples_leaf': 10,  # DEFAULT
    'max_features': 'sqrt',
    'class_weight': 'balanced',
    'random_state': random_seed,
    'n_jobs': ML_TRAINING_CPU_CORES
}
```

### LightGBM Classifier
```python
gb_params = {
    'n_estimators': 200,  # DEFAULT
    'max_depth': 6,  # DEFAULT
    'learning_rate': 0.05,
    'subsample': 0.8,  # DEFAULT
    'colsample_bytree': 0.8,  # DEFAULT
    'min_child_samples': 20,  # DEFAULT
    'reg_alpha': 0,  # DEFAULT
    'reg_lambda': 1,  # DEFAULT
    'random_state': random_seed,
    'verbose': -1
}
```

### XGBoost Classifier (if LightGBM not available)
```python
gb_params = {
    'n_estimators': 200,  # DEFAULT
    'max_depth': 6,  # DEFAULT
    'learning_rate': 0.05,
    'subsample': 0.8,  # DEFAULT
    'colsample_bytree': 0.8,  # DEFAULT
    'min_child_weight': 1,  # DEFAULT
    'gamma': 0,  # DEFAULT
    'reg_alpha': 0,  # DEFAULT
    'reg_lambda': 1,  # DEFAULT
    'random_state': random_seed,
    'eval_metric': 'mlogloss',
    'enable_categorical': False
}
```

### LogisticRegression (Meta-learner)
```python
lr = LogisticRegression(
    max_iter=1000,
    C=0.1,  # DEFAULT
    class_weight='balanced',
    random_state=random_seed
)
```

---

## Training Configuration

### Data Split
- **Method:** Random split (stratified)
- **Test Size:** 0.2 (20%)
- **Stratify:** Yes (maintains class distribution)

### Labeling Thresholds
- **Trading Strategy:**
  - BUY: `price_change_pct >= 3%`
  - SELL: `price_change_pct <= -3%`
  - HOLD: Otherwise

- **Mixed Strategy:**
  - BUY: `price_change_pct >= 5%`
  - SELL: `price_change_pct <= -5%`
  - HOLD: Otherwise

- **Investing Strategy:**
  - BUY: `price_change_pct >= 10%`
  - SELL: `price_change_pct <= -10%`
  - HOLD: Otherwise

### Labeling Logic (CRITICAL - DO NOT CHANGE)
```python
# CORRECT LOGIC - Price UP → BUY, Price DOWN → SELL
if price_change_pct >= threshold:
    label = "BUY"  # Price went up → BUY
elif price_change_pct <= -threshold:
    label = "SELL"  # Price went down → SELL
else:
    label = "HOLD"
```

---

## Feature Configuration

### Feature Selection
- **Method:** Importance-based (default)
- **Alternative methods available:** RFE, mutual_info

### Advanced Features
- ✅ Time-series features (enabled)
- ✅ Pattern detection (enabled)
- ✅ Feature interactions (enabled)
- ✅ Enhanced features (enabled)

### Ensemble
- **Type:** Stacking Classifier
- **Base Models:** RandomForest, LightGBM/XGBoost, LogisticRegression
- **Meta-learner:** LogisticRegression

---

## Performance Settings

### CPU Usage
- **ML Training:** All cores (`ML_TRAINING_CPU_CORES = -1`)
- **Backtesting:** All cores (`BACKTESTING_CPU_CORES = -1`)
- **Hyperparameter Tuning:** Auto (`HYPERPARAMETER_TUNING_PARALLEL_TRIALS = None`)

### Hyperparameter Tuning
- **Default:** Disabled (use original parameters)
- **When to enable:** For final optimization after baseline is established

---

## What Makes This Configuration Work

1. ✅ **Correct Labeling** - Price UP → BUY, Price DOWN → SELL (fixed from inverted)
2. ✅ **Balanced Regularization** - Not too strict, not too loose
3. ✅ **Original Parameters** - Proven to work, not over-optimized
4. ✅ **Clean Hybrid Accuracy** - Cleared old 0/9 scores that affected weights
5. ✅ **Proper Ensemble** - Stacking with balanced base models

---

## How to Restore This Configuration

If the configuration gets changed and you need to restore:

1. **Check `ml_training.py`** around line 1103-1177
2. **Verify parameters match** the values above
3. **Check `training_pipeline.py`** around line 245-269 for labeling thresholds
4. **Verify labeling logic** is correct (Price UP → BUY)

---

## Performance History

- **Initial Baseline:** 47% test accuracy (before fixes)
- **After Labeling Fix:** Improved
- **After Clearing Hybrid Accuracy:** 70% backtesting accuracy
- **Current Status:** ✅ WORKING - 70% backtesting

---

## Notes

- This configuration was reverted from experimental changes that made accuracy worse
- The key fix was correcting the inverted labeling logic
- Clearing hybrid accuracy (0/9) also helped ensemble weights
- Keep this as the stable baseline for future improvements

---

**Last Updated:** 2026-01-07  
**Status:** ✅ ACTIVE DEFAULT CONFIGURATION
