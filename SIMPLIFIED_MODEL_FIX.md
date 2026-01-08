# 🔧 SIMPLIFIED MODEL - DIAGNOSTIC FIX

**Date:** 2026-01-07  
**Issue:** Model performing worse than random (36.8% test accuracy)  
**Action:** Simplified model to isolate the issue

---

## ✅ CHANGES APPLIED

### 1. **Simplified Model Architecture** ✅
- **Before:** Complex StackingClassifier ensemble
- **After:** Single RandomForest (simplest possible)
- **Why:** Complex ensemble might be causing issues - start simple

### 2. **Disabled Feature Selection** ✅
- **Before:** Feature selection removing 50% of features
- **After:** Using ALL features
- **Why:** Feature selection might be removing important features

### 3. **Disabled Feature Interactions** ✅
- **Before:** Adding feature interactions
- **After:** No interactions
- **Why:** Simplify to isolate the issue

### 4. **Increased Model Capacity** ✅
- **RandomForest:**
  - max_depth: 6 → 10 (allow more learning)
  - min_samples_split: 30 → 10 (allow more splits)
  - min_samples_leaf: 15 → 5 (allow more detail)

### 5. **Temporal Split Enabled** ✅
- Prevents data leakage (random split caused 98.8% train / 44% test)

### 6. **SMOTE Disabled** ✅
- Using `class_weight='balanced'` instead

---

## 🎯 EXPECTED RESULTS

**Goal:** Get to at least 50% accuracy (above random 33%)

**If this works:**
- Model can learn (features are informative)
- Can then re-enable optimizations one by one

**If still fails:**
- Deeper issue (labels wrong, features invalid, data quality)

---

## 🚀 NEXT STEPS

1. **Retrain with simplified model**
2. **Check if accuracy improves to at least 50%**
3. **If yes:** Re-enable optimizations one by one
4. **If no:** Check labels, features, data quality

---

**Status:** ✅ SIMPLIFIED - Ready for diagnostic retraining
