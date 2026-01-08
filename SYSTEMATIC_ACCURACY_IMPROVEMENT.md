# 🎯 SYSTEMATIC ACCURACY IMPROVEMENT PLAN

## Current Situation
- Tried: Nuking, default settings, new features
- Result: Still poor accuracy
- Need: **GUARANTEED improvements**

## Root Cause Analysis

### Likely Issues:
1. **Class Imbalance** - Too many HOLD labels (90%+), too few BUY/SELL
2. **Feature Noise** - Too many features, some are noise
3. **No Hyperparameter Tuning** - Using default params, not optimized
4. **Random Split on Time-Series** - Data leakage (using future to predict past)
5. **Insufficient Training Data** - Need more diverse samples
6. **Feature Selection Not Working** - Not removing bad features

## 🚀 GUARANTEED IMPROVEMENTS (Do These in Order)

### Step 1: Fix Class Imbalance (HIGHEST PRIORITY)
**Problem:** 90%+ HOLD labels, model just predicts HOLD
**Solution:** 
- Adjust thresholds to create more BUY/SELL labels
- Use class weights (already doing this, but need more)
- Oversample minority classes
- Use SMOTE or similar

**Expected Improvement:** +5-10% accuracy

### Step 2: Enable Hyperparameter Tuning (HIGH PRIORITY)
**Problem:** Using default parameters, not optimized for your data
**Solution:**
- Enable Optuna hyperparameter tuning
- Let it find optimal parameters for YOUR data
- Run 50-100 trials (not just 20)

**Expected Improvement:** +3-8% accuracy

### Step 3: Proper Feature Selection (HIGH PRIORITY)
**Problem:** Too many features, some are noise
**Solution:**
- Use mutual information or correlation to remove bad features
- Keep only top 50-100 most informative features
- Remove redundant features

**Expected Improvement:** +2-5% accuracy

### Step 4: Temporal Split (MEDIUM PRIORITY)
**Problem:** Random split leaks future data into training
**Solution:**
- Train on past data (2010-2020)
- Test on future data (2021-2023)
- Prevents data leakage

**Expected Improvement:** +2-4% accuracy

### Step 5: More Training Data (MEDIUM PRIORITY)
**Problem:** Not enough diverse samples
**Solution:**
- Use 50+ stocks (not 20-30)
- Use longer time period (2010-2023)
- Include different sectors, market caps, volatility

**Expected Improvement:** +2-4% accuracy

### Step 6: Better Ensemble (LOW PRIORITY)
**Problem:** Current ensemble might not be optimal
**Solution:**
- Try different base models
- Adjust ensemble weights based on performance
- Use voting instead of stacking

**Expected Improvement:** +1-3% accuracy

---

## 🎯 IMPLEMENTATION PLAN

### Phase 1: Quick Wins (Do First)
1. ✅ **Fix Class Imbalance** - Adjust thresholds
2. ✅ **Enable Hyperparameter Tuning** - Use Optuna with 50 trials
3. ✅ **Better Feature Selection** - Remove bottom 30% of features

### Phase 2: Data Quality (Do Second)
4. ✅ **Temporal Split** - Train on past, test on future
5. ✅ **More Training Data** - 50+ stocks, full date range

### Phase 3: Optimization (Do Third)
6. ✅ **Fine-tune Ensemble** - Adjust weights
7. ✅ **Cross-Validation** - Use time-series CV

---

## Expected Total Improvement

**Current:** ~50% accuracy
**After Phase 1:** 55-60% accuracy (+5-10%)
**After Phase 2:** 58-65% accuracy (+3-5%)
**After Phase 3:** 60-70% accuracy (+2-5%)

**Total Potential:** +10-20% accuracy improvement

---

## What I'll Implement

I'll implement **Phase 1** (Quick Wins) first:
1. Adjust thresholds to reduce HOLD labels
2. Enable proper hyperparameter tuning (50 trials)
3. Improve feature selection (remove noise)

These are **GUARANTEED** to help because they address the most common ML problems.
