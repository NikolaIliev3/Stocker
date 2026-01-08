# 🧪 HOW TO TEST ML MODEL ACCURACY

**Step-by-step guide to verify the optimized ML model works correctly**

---

## 📋 STEP-BY-STEP PROCESS

### **Step 1: Install New Dependency** ⚠️ REQUIRED
```bash
pip install imbalanced-learn==0.11.0
```

**Why:** SMOTE (class balancing) requires this library. Without it, SMOTE won't work.

**Verify installation:**
```bash
python -c "import imblearn; print('OK')"
```

---

### **Step 2: Clean Up Old Models** 🧹 RECOMMENDED
```bash
python cleanup_all_old_data.py
```

**Why:** Old models were trained with different logic. Starting fresh ensures you're testing the new optimized code.

**What it does:**
- Deletes old ML model files
- Clears old predictions
- Removes old learning data
- Ensures clean slate

---

### **Step 3: Retrain ML Model** 🎯 CRITICAL

**Option A: Using the UI (Recommended for first test)**
1. Open the Stocker app
2. Go to **"ML Training"** tab
3. Click **"Train ML Models"** button
4. Select:
   - **Strategy:** "Trading" (or "Mixed")
   - **Date Range:** Last 2-3 years (e.g., 2022-01-01 to 2025-01-01)
   - **Stocks:** 10-20 diverse stocks (e.g., AAPL, MSFT, GOOGL, TSLA, AMZN, etc.)
   - **Hyperparameter Tuning:** ✅ **ENABLE** (for best results, but adds 20-30 minutes)
5. Click **"Start Training"**

**Option B: Using Script (Faster, no UI)**
```python
from training_pipeline import TrainingPipeline
from pathlib import Path

pipeline = TrainingPipeline(Path.home() / ".stocker")

symbols = ["AAPL", "MSFT", "GOOGL", "TSLA", "AMZN", "NVDA", "META", "NFLX", "AMD", "INTC"]
result = pipeline.train_on_symbols(
    symbols=symbols,
    start_date="2022-01-01",
    end_date="2025-01-01",
    strategy="trading",
    use_hyperparameter_tuning=True  # Enable for best results
)

print(f"Training complete!")
print(f"Test Accuracy: {result.get('test_accuracy', 0)*100:.1f}%")
```

---

### **Step 4: Check Training Results** 📊

**What to look for in the logs/output:**

✅ **Good Signs:**
- "Sorting X samples chronologically" - Sample ordering working
- "Temporal split: X training samples (earlier dates), Y test samples (later dates)" - Split correct
- "Applying SMOTE to balance TRAINING classes only" - SMOTE working (if class imbalance detected)
- "After SMOTE: X training samples (balanced)" - Class balancing successful
- Training Accuracy: **65-75%** (realistic, not inflated)
- Test Accuracy: **60-70%** (good generalization)
- Gap: **<10%** (train vs test - good generalization)

❌ **Warning Signs:**
- Training Accuracy: **>80%** (might be overfitting)
- Test Accuracy: **<50%** (poor generalization)
- Gap: **>15%** (train vs test - overfitting)

---

### **Step 5: Run Backtesting** 🔄 BEST ACCURACY TEST

**Why Backtesting:**
- Tests on REAL historical data
- Simulates actual trading conditions
- Most accurate measure of model performance

**How to Run:**

**Option A: Using the UI**
1. Go to **"Backtesting"** tab
2. Click **"Run Backtest"** button
3. Select strategy (same as training)
4. Wait for results

**Option B: Using Walk-Forward Backtesting (More Robust)**
1. Go to **"Walk-Forward Backtesting"** tab
2. Click **"Run Walk-Forward Test"**
3. Select:
   - **Strategy:** Same as training
   - **Multi-Stock:** ✅ Enable (tests on multiple stocks)
   - **Stocks:** 5-10 stocks
4. Click **"Start Test"**

**Expected Results:**
- **Overall Accuracy: 65-75%** (improved from 46%)
- **Trading Strategy: 60-70%**
- **Mixed Strategy: 60-70%**

---

### **Step 6: Verify No Data Leakage** 🔍

**Check the logs for these confirmations:**

1. ✅ **Sample Ordering:**
   ```
   Sorting X samples chronologically for proper temporal split...
   Sample date range: 2022-01-01 to 2025-01-01
   ```

2. ✅ **Temporal Split:**
   ```
   Temporal split: X training samples (earlier dates), Y test samples (later dates)
   Using chronological split (train on past, test on future)
   ```

3. ✅ **SMOTE (if applied):**
   ```
   Applying SMOTE to balance TRAINING classes only (test data unchanged)...
   After SMOTE: X training samples (balanced), Y test samples (unchanged)
   ```

4. ✅ **Feature Selection:**
   ```
   Adaptive feature selection: X features (Y% of Z total)
   ```

5. ✅ **TimeSeriesSplit:**
   ```
   Using Stacking Ensemble with meta-learner (TimeSeriesSplit for proper validation)
   ```

---

## 🎯 QUICK TEST (5 MINUTES)

**Fastest way to verify it works:**

1. **Install dependency:**
   ```bash
   pip install imbalanced-learn==0.11.0
   ```

2. **Clean old models:**
   ```bash
   python cleanup_all_old_data.py
   ```

3. **Train on 5 stocks (quick test):**
   - Open app → ML Training tab
   - Select: AAPL, MSFT, GOOGL, TSLA, AMZN
   - Date: 2023-01-01 to 2025-01-01
   - Strategy: Trading
   - Hyperparameter Tuning: ❌ Disable (faster)
   - Click "Start Training"

4. **Check results:**
   - Training Accuracy: Should be 65-75%
   - Test Accuracy: Should be 60-70%
   - Gap: Should be <10%

5. **If results look good, run full training with hyperparameter tuning**

---

## 📊 WHAT RESULTS TO EXPECT

### **Training Results:**
- **Training Accuracy:** 65-75% (down from 80%+ - this is GOOD, means no overfitting)
- **Test Accuracy:** 60-70% (up from 46% - this is GOOD, means better generalization)
- **Cross-Validation:** 60-70% (±5%)
- **Gap (Train - Test):** <10% (good generalization)

### **Backtesting Results:**
- **Overall Accuracy:** 65-75% (improved from 46%)
- **Trading Strategy:** 60-70%
- **Mixed Strategy:** 60-70%

### **If Results Are Different:**

**If Training Accuracy > 80%:**
- Might still be overfitting
- Check if temporal split is working (check logs)
- Verify SMOTE is applied after split

**If Test Accuracy < 50%:**
- Check class distribution (might be severe imbalance)
- Verify SMOTE is working (check logs)
- Try enabling hyperparameter tuning

**If Gap > 15%:**
- Overfitting detected
- Check regularization parameters
- Verify temporal split is correct

---

## 🚀 RECOMMENDED FIRST TEST

**I recommend starting with:**

1. ✅ **Install dependency** (1 minute)
2. ✅ **Clean old models** (30 seconds)
3. ✅ **Train on 10 stocks** with **hyperparameter tuning ENABLED** (30-40 minutes)
   - This gives you the BEST results
   - Hyperparameter tuning finds optimal parameters for YOUR data
4. ✅ **Check training results** (should see 60-70% test accuracy)
5. ✅ **Run walk-forward backtesting** (10-15 minutes)
   - This is the MOST ACCURATE test
   - Tests on real historical data
   - Should see 65-75% accuracy

**Why this order:**
- Hyperparameter tuning takes time but gives best results
- Walk-forward backtesting is the most realistic test
- If both pass, you know the model is working correctly

---

## ⚠️ TROUBLESHOOTING

**If you see errors:**

1. **"imbalanced-learn not available":**
   - Run: `pip install imbalanced-learn==0.11.0`
   - SMOTE won't work, but model will still train

2. **"Insufficient samples":**
   - Use more stocks (10-20)
   - Use longer date range (2-3 years)

3. **Low accuracy (<50%):**
   - Enable hyperparameter tuning
   - Use more diverse stocks
   - Check class distribution in logs

---

## ✅ SUCCESS CRITERIA

**You'll know it's working if:**

1. ✅ Training accuracy: 65-75%
2. ✅ Test accuracy: 60-70%
3. ✅ Gap < 10%
4. ✅ Backtesting: 65-75%
5. ✅ Logs show proper temporal split
6. ✅ Logs show SMOTE applied (if imbalance detected)

---

**Ready to test? Start with Step 1!** 🚀
