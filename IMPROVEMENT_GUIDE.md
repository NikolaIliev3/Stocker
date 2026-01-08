# 🚀 Step-by-Step Guide: Improving Accuracy from 55% to 60-65%

## Current Status ✅
- **Overall Accuracy**: 55% (good baseline!)
- **Trading Strategy**: 60% (excellent!)
- **Mixed Strategy**: 50% (needs improvement)

## Goal 🎯
Reach **60-65% accuracy** using hyperparameter tuning and better training data.

---

## Step 1: Enable Hyperparameter Tuning ⚙️

### What It Does
- Uses Optuna to automatically find the best model parameters
- Tests 20 different parameter combinations
- Finds optimal settings for your specific data

### How to Enable

1. **Open the App**
   - Start backend: `python backend_proxy.py`
   - Start app: `python main.py`

2. **Go to ML Training Tab**
   - Click on "ML Training" tab in the app

3. **Click "Train AI Model" Button**
   - This opens the training dialog

4. **Check the Hyperparameter Tuning Box**
   - Look for: **"⚙️ Enable Hyperparameter Tuning (Optuna)"**
   - ✅ Check this box
   - You'll see: "Better accuracy, slower training"

5. **What to Expect**
   - Training will take **10-20 minutes longer**
   - You'll see messages like:
     ```
     Step 3: Tuning hyperparameters with Optuna...
     Trial 1/20...
     Trial 2/20...
     ...
     Best hyperparameters found: {...}
     ```

### Expected Improvement
- **+3-5% accuracy** (from 55% → 58-60%)

---

## Step 2: Select Better Training Data 📊

### Current Setup
- You're using 7,800 samples (good!)
- But diversity matters more than quantity

### Recommended: Diverse Portfolio

#### Option A: Quick Preset (Easiest)
1. In training dialog, click **"📊 Load Diverse Portfolio (20 symbols)"**
2. This loads: `AAPL,MSFT,GOOGL,NVDA,AMD,JPM,BAC,GS,JNJ,PFE,UNH,AMZN,WMT,HD,BA,CAT,XOM,CVX,TSLA,META`
3. Includes: Tech, Finance, Healthcare, Consumer, Industrial, Energy

#### Option B: Manual Selection (Better Control)
Select **20-30 stocks** from different sectors:

**Technology (5-7 stocks):**
- AAPL, MSFT, GOOGL, NVDA, AMD, META, TSLA

**Finance (3-4 stocks):**
- JPM, BAC, GS, WFC

**Healthcare (3-4 stocks):**
- JNJ, PFE, UNH, ABT

**Consumer (3-4 stocks):**
- AMZN, WMT, HD, MCD

**Industrial (2-3 stocks):**
- BA, CAT, GE

**Energy (2-3 stocks):**
- XOM, CVX, COP

**Example**: `AAPL,MSFT,GOOGL,NVDA,JPM,BAC,GS,JNJ,PFE,UNH,AMZN,WMT,HD,BA,CAT,XOM,CVX,TSLA,META,NFLX`

### Why Diversity Matters
- **Single sector**: Model learns sector-specific patterns (overfitting)
- **Diverse sectors**: Model learns general market patterns (better generalization)
- **Result**: Better accuracy on new stocks

---

## Step 3: Use Longer Date Range 📅

### Current Setup
- Default: 2010-2023 (14 years) - **This is good!**

### Recommended
- **Start Date**: 2010-01-01 (includes 2008 recovery, bull markets, bear markets)
- **End Date**: 2023-12-31 (leave 2024-2025 for testing)

### Why Longer is Better
- **More market regimes**: Bull, bear, sideways, recovery
- **More patterns**: Model sees more scenarios
- **Better generalization**: Works in different market conditions

### What to Avoid
- ❌ Too short (< 5 years): Not enough data
- ❌ Too recent (2024-2025): Use for testing, not training
- ✅ **14-15 years**: Perfect balance

---

## Step 4: Training Process 🎓

### Complete Workflow

1. **Open Training Dialog**
   - ML Training tab → "Train AI Model"

2. **Select Strategy**
   - Start with **"Trading"** (already at 60% - best performer)

3. **Load Symbols**
   - Click "📊 Load Diverse Portfolio" OR
   - Enter 20-30 diverse symbols manually

4. **Set Date Range**
   - Start: `2010-01-01`
   - End: `2023-12-31`

5. **Enable Hyperparameter Tuning**
   - ✅ Check "⚙️ Enable Hyperparameter Tuning (Optuna)"

6. **Optional: Retrain from Verified Predictions**
   - Leave unchecked for now (you have fresh start)

7. **Click "Start Training"**
   - Training runs in background
   - You can minimize and use the app
   - Status updates in real-time

8. **Wait for Completion**
   - Normal training: 10-30 minutes
   - With hyperparameter tuning: 20-50 minutes
   - You'll see "ML Training Complete!" dialog

---

## Step 5: Evaluate Results 📈

### What to Look For

#### Good Signs ✅
- **Test Accuracy**: 58-65% (up from 55%)
- **Training Accuracy**: 60-70% (close to test = no overfitting)
- **Cross-Validation**: 58-65% (±2-3%)
- **Train/Test Gap**: <5% (good generalization)

#### Warning Signs ⚠️
- **Test Accuracy**: <55% (worse than before)
- **Training Accuracy**: >> Test (>10% gap = overfitting)
- **Cross-Validation**: Much different from test

### If Results Are Good
- ✅ Keep the model
- ✅ Make predictions with it
- ✅ Monitor backtesting results

### If Results Are Bad
- Check label distribution in logs
- Try different symbols
- Try different date range
- Retrain without hyperparameter tuning first

---

## Step 6: Retrain Other Strategies 🔄

### After Trading Strategy Works

1. **Train Mixed Strategy**
   - Same process
   - Currently at 50% (needs improvement)
   - Use hyperparameter tuning

2. **Train Investing Strategy**
   - Same process
   - Use hyperparameter tuning

### Strategy Priority
1. **Trading** (60%) - Best performer, focus here
2. **Mixed** (50%) - Needs improvement
3. **Investing** - Train if you use it

---

## Step 7: Regular Maintenance 🔧

### Monthly Checks
- Run walk-forward backtesting
- Check if accuracy drops below 50%
- Retrain if needed

### When to Retrain
- ✅ Accuracy drops below 50%
- ✅ Walk-forward suggests retraining
- ✅ After 2-3 months (market changes)
- ✅ After major market events (crashes, rallies)

### Keep Training Data Fresh
- Add new verified predictions to training
- Use "Retrain from Verified Predictions" option
- Combine with historical data

---

## Expected Timeline 📅

### First Training (With Hyperparameter Tuning)
- **Time**: 20-50 minutes
- **Expected Accuracy**: 58-62%
- **Improvement**: +3-5% from 55%

### After Multiple Training Sessions
- **Time**: 20-50 minutes each
- **Expected Accuracy**: 60-65%
- **Improvement**: +5-10% from 55%

---

## Quick Reference Card 🎯

### Best Practices Checklist

- [ ] Use 20-30 diverse stocks (different sectors)
- [ ] Use 14-15 years of data (2010-2023)
- [ ] Enable hyperparameter tuning
- [ ] Start with Trading strategy (best performer)
- [ ] Monitor train/test gap (<5% is good)
- [ ] Retrain monthly or when accuracy drops

### Symbols to Use

**Quick Copy-Paste (20 stocks):**
```
AAPL,MSFT,GOOGL,NVDA,AMD,JPM,BAC,GS,JNJ,PFE,UNH,AMZN,WMT,HD,BA,CAT,XOM,CVX,TSLA,META
```

**Extended (30 stocks):**
```
AAPL,MSFT,GOOGL,NVDA,AMD,META,TSLA,NFLX,JPM,BAC,GS,WFC,JNJ,PFE,UNH,ABT,AMZN,WMT,HD,MCD,BA,CAT,GE,MMM,XOM,CVX,COP,DIS,V,MA
```

---

## Troubleshooting 🔍

### "Hyperparameter tuning failed"
- **Cause**: Optuna not installed or insufficient data
- **Fix**: `pip install optuna`
- **Alternative**: Train without tuning first

### "Insufficient samples"
- **Cause**: Not enough data for selected symbols/date range
- **Fix**: Use more symbols or longer date range

### "Accuracy still low"
- **Check**: Label distribution in logs
- **Check**: Train/test gap (overfitting?)
- **Try**: Different symbols or date range

---

## Summary 🎯

1. ✅ **Enable hyperparameter tuning** - Check the box in training dialog
2. ✅ **Use diverse symbols** - 20-30 stocks from different sectors
3. ✅ **Use long date range** - 2010-2023 (14 years)
4. ✅ **Start with Trading** - Already at 60%, best performer
5. ✅ **Monitor results** - Check train/test gap
6. ✅ **Retrain regularly** - Monthly or when accuracy drops

**Expected Result**: 60-65% accuracy (up from 55%) 🚀

---

**You're all set! The hyperparameter tuning checkbox is now in the training dialog. Just check it and train!**
