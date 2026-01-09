# SELL Bias Investigation Report

## Problem Summary
The ML model is predicting SELL almost exclusively (100% of predictions in backtesting), resulting in:
- **Backtesting accuracy: 41%** (below 50% random chance)
- **SELL predictions failing 68% of the time** (40/59 failures)
- Model appears to be biased toward the majority class

## Root Cause Analysis

### 1. Class Imbalance in Training Data
From training logs:
- **UP (SELL)**: 1,399 samples (60.3%)
- **DOWN (BUY)**: 918 samples (39.7%)
- **Ratio**: ~1.5:1 imbalance

### 2. Label Mapping Logic
The inverted logic is correctly implemented:
- **Training**: BUY → DOWN, SELL → UP
- **Prediction**: UP → SELL, DOWN → BUY
- This is correct for the application's inverted SELL logic

### 3. Model Behavior
- Model is using `class_weight='balanced'` which should help, but:
  - If features don't provide good separation, model defaults to majority class
  - The 60/40 imbalance is enough to cause bias when features are weak
  - Model confidence is moderate (65-75%) but always choosing SELL

### 4. Feature Quality Issue
The model may not be learning meaningful patterns:
- All predictions are SELL regardless of input features
- This suggests features aren't providing good discrimination
- Model may be overfitting to majority class patterns

## Recommendations

### Immediate Fixes

1. **Add Prediction Confidence Threshold**
   - Filter predictions below 55% confidence → convert to HOLD
   - This will reduce false SELL predictions
   - Implementation: Modify `ml_training.py` predict method

2. **Improve Class Balancing**
   - Use SMOTE (Synthetic Minority Oversampling) to balance classes
   - Or use stronger class weights (e.g., `class_weight={0: 1.5, 1: 1.0}`)
   - Implementation: Add to `ml_training.py` train method

3. **Add Prediction Diversity Check**
   - Track prediction distribution in real-time
   - If >80% predictions are same class, trigger warning
   - Implementation: Add to `hybrid_predictor.py`

### Long-term Improvements

1. **Collect More Balanced Training Data**
   - Focus on collecting more DOWN (BUY) samples
   - Target: 50/50 or at least 55/45 distribution
   - Use more volatile stocks or longer time periods

2. **Feature Engineering**
   - Review feature importance to identify weak features
   - Add features that better distinguish UP vs DOWN scenarios
   - Consider market regime-specific features

3. **Model Architecture**
   - Consider using a threshold-based approach instead of argmax
   - Use calibrated probabilities (Platt scaling)
   - Implement ensemble voting with diversity requirements

4. **Evaluation Metrics**
   - Track per-class accuracy (not just overall)
   - Monitor prediction distribution over time
   - Alert when bias exceeds threshold

## Code Changes Needed

### 1. Add Confidence Threshold (Priority: High)
```python
# In ml_training.py predict() method
confidence_threshold = 0.55  # 55% minimum confidence
if max(probabilities) < confidence_threshold:
    action = 'HOLD'  # Don't make low-confidence predictions
```

### 2. Improve Class Balancing (Priority: High)
```python
# In ml_training.py train() method
from imblearn.over_sampling import SMOTE
if use_binary_classification and class_imbalance_ratio > 1.3:
    smote = SMOTE(random_state=random_seed)
    X, y = smote.fit_resample(X, y)
```

### 3. Add Prediction Diversity Monitoring (Priority: Medium)
```python
# In hybrid_predictor.py
self.recent_predictions = []  # Track last 100 predictions
# If >80% are same class, log warning
```

## Expected Impact

After implementing fixes:
- **Prediction diversity**: Should see ~40-60% SELL, ~40-60% BUY
- **Backtesting accuracy**: Should improve to 50-55%+
- **False positives**: Should decrease significantly

## Next Steps

1. ✅ **Investigation Complete** - Root cause identified
2. ⏳ **Implement confidence threshold** - Quick win
3. ⏳ **Add SMOTE balancing** - Medium effort, high impact
4. ⏳ **Monitor prediction distribution** - Ongoing
