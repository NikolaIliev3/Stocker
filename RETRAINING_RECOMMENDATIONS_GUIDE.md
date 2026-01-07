# 🧠 Automatic Retraining Recommendations Guide

## Overview

The **Walk-Forward Backtester** now automatically analyzes results and provides intelligent retraining recommendations based on multiple performance metrics.

**NEW:** Multi-stock walk-forward testing is now available! Test your models across multiple stocks for more robust and reliable retraining recommendations.

---

## How It Works

After running a walk-forward backtest, the system automatically:

1. **Analyzes** overall accuracy, variance, trends, and risk metrics
2. **Evaluates** multiple indicators (accuracy thresholds, consistency, Sharpe ratio)
3. **Provides** clear recommendation with priority level and reasons
4. **Displays** visual indicator in the UI (color-coded)

---

## Recommendation Types

### 🔴 RETRAIN NOW (High Priority)

**When it appears:**
- Overall accuracy < 40%
- Minimum accuracy < 30%
- Declining trend detected (>10% drop across windows)
- Negative Sharpe ratio
- Multiple critical issues

**What to do:**
```bash
python retrain_models.py
```

Or use ML Training in the app to retrain all models.

**Visual Indicator:** Red background with dark red text

---

### 🟡 CONSIDER RETRAINING (Medium/Low Priority)

**When it appears:**
- Overall accuracy 40-50%
- High variance (std > 20%)
- Minimum accuracy 30-40%
- Low Sharpe ratio (< 0.5)
- Large accuracy spread (> 40%)

**What to do:**
- Review the specific reasons provided
- Consider retraining if you have new data or want better performance
- Monitor for a few more days if performance is borderline acceptable

**Visual Indicator:** Orange/yellow background with dark orange text

---

### 🟢 NO RETRAIN NEEDED (Good Performance)

**When it appears:**
- Overall accuracy > 60%
- Consistent performance (low variance)
- Minimum accuracy > 45%
- Good Sharpe ratio (> 1.0)
- Stable or improving trend

**What to do:**
- Continue using current models
- Run walk-forward tests weekly to monitor
- No action needed unless performance degrades

**Visual Indicator:** Green background with dark green text

---

## Metrics Analyzed

The recommendation system evaluates:

### 1. **Overall Accuracy**
- **< 40%** → RETRAIN NOW
- **40-50%** → CONSIDER RETRAINING
- **> 60%** → NO RETRAIN NEEDED

### 2. **Minimum Accuracy**
- **< 30%** → RETRAIN NOW (some windows very poor)
- **30-40%** → CONSIDER RETRAINING (inconsistent)
- **> 45%** → Positive indicator

### 3. **Variance (Consistency)**
- **Std > 20%** → CONSIDER RETRAINING (high variance)
- **Std > 15%** → CONSIDER RETRAINING (moderate variance)
- **Std < 10%** → Positive indicator (consistent)

### 4. **Accuracy Spread**
- **Max - Min > 40%** → CONSIDER RETRAINING (high variability)
- **Max - Min < 25%** → Positive indicator (consistent)

### 5. **Sharpe Ratio**
- **< 0** → RETRAIN NOW (negative returns)
- **< 0.5** → CONSIDER RETRAINING (suboptimal)
- **> 1.0** → Positive indicator (good risk-adjusted returns)

### 6. **Trend Analysis**
- **Declining trend** (>10% drop) → RETRAIN NOW
- **Stable/improving** → Positive indicator

---

## Example Recommendations

### Example 1: RETRAIN NOW
```
🔴 RETRAIN NOW (HIGH priority, 85% confidence)

Reasons:
1. Overall accuracy (35.2%) is below 40% - critical threshold
2. Minimum accuracy (18.0%) is very low - some windows performed poorly
3. Declining trend detected: 52.1% → 28.3% (model degrading)
4. Negative Sharpe ratio (-0.15) - poor risk-adjusted returns

💡 Run 'python retrain_models.py' to retrain models
```

### Example 2: CONSIDER RETRAINING
```
🟡 CONSIDER RETRAINING (MEDIUM priority, 45% confidence)

Reasons:
1. Overall accuracy (47.8%) is below 50% - below acceptable
2. High variance (std: 18.5%) - inconsistent across windows
3. Minimum accuracy (32.0%) is below 40% - inconsistent performance

✅ Positive Indicators:
1. Mean accuracy (49.1%) is acceptable
2. Low variance (std: 8.2%) - consistent performance

💡 Consider retraining models to improve performance
```

### Example 3: NO RETRAIN NEEDED
```
🟢 NO RETRAIN NEEDED (NONE priority, 0% confidence)

✅ Positive Indicators:
1. Overall accuracy (64.2%) is good
2. Minimum accuracy (52.0%) is acceptable
3. Low variance (std: 7.8%) - consistent performance
4. Good Sharpe ratio (1.15) - strong risk-adjusted returns

💡 Models are performing well - no retraining needed
```

---

## How to Use

### Step 1: Run Walk-Forward Backtest

**Option A: Single Stock (Faster)**
1. Go to "🔄 Walk-Forward" tab
2. Analyze a stock first (or use currently analyzed stock)
3. Set parameters (defaults are fine)
4. Click "Run Walk-Forward Test"

**Option B: Multiple Stocks (More Robust) ⭐ RECOMMENDED**
1. Go to "🔄 Walk-Forward" tab
2. Check "Test Multiple Stocks (More Robust)"
3. Set "Max stocks" (default: 10)
4. Set parameters (defaults are fine)
5. Click "Run Walk-Forward Test"
6. Wait for all stocks to be tested (may take several minutes)

**Why Multi-Stock?**
- Tests generalization across different stocks
- More reliable retraining recommendations
- Better statistical significance
- Identifies stocks where models struggle

### Step 2: Review Recommendation

**For Single Stock:**
- **Visual indicator** appears at top of results (color-coded)
- **Detailed analysis** in results text
- **Reasons** listed for transparency

**For Multiple Stocks:**
- **Aggregate recommendation** based on all stocks tested
- **Per-stock breakdown** showing which stocks need retraining
- **Statistical summary** across all stocks
- **Visual indicator** shows aggregate recommendation
- **Retraining breakdown** (how many stocks in each category)

### Step 3: Take Action
- **🔴 RETRAIN NOW** → Retrain immediately
- **🟡 CONSIDER RETRAINING** → Review and decide
- **🟢 NO RETRAIN NEEDED** → Continue monitoring

---

## Confidence Score

The recommendation includes a **confidence score (0-100%)**:

- **0-30%** → Low confidence, borderline case
- **30-60%** → Medium confidence, clear indicators
- **60-100%** → High confidence, strong recommendation

Higher confidence = More reliable recommendation

---

## Best Practices

### 1. **Run Walk-Forward Weekly**
- Regular validation prevents model degradation
- Catch issues early before they become critical

### 2. **Compare Recommendations Over Time**
- Track if recommendations change
- Identify when models start degrading

### 3. **Retrain After Major Changes**
- After adding new features
- After market regime changes
- After accumulating significant new data

### 4. **Don't Over-Retrain**
- If recommendation is "NO RETRAIN NEEDED", trust it
- Over-training can lead to overfitting
- Wait for clear signals

### 5. **Use Multiple Indicators**
- Don't rely on single metric
- Consider all reasons provided
- Look at trend patterns

---

## Troubleshooting

### "Recommendation seems wrong"
- Check if you have enough windows tested (need at least 3)
- Verify data quality (poor data = poor results)
- Consider market conditions (some markets harder to predict)

### "Always says RETRAIN NOW"
- May indicate fundamental model issues
- Check if enhanced features are being used
- Verify ML models are actually trained
- Consider retraining with more data

### "Never recommends retraining"
- If accuracy is consistently good, this is correct!
- Models may be performing well
- Continue monitoring regularly

---

## Multi-Stock Testing Benefits

### Why Test Multiple Stocks?

1. **Generalization**: Tests if models work across different stocks, not just one
2. **Statistical Significance**: More data points = more reliable metrics
3. **Sector Diversity**: Tests across sectors (tech, finance, healthcare, etc.)
4. **Robust Recommendations**: Aggregate results reduce single-stock bias
5. **Per-Stock Insights**: Identifies which stocks need attention

### Multi-Stock Recommendation Logic

The system aggregates results across all stocks and provides:

- **Aggregate accuracy** (average across all stocks)
- **Per-stock breakdown** (which stocks need retraining)
- **Retraining breakdown** (counts: RETRAIN NOW, CONSIDER, NO RETRAIN)
- **Aggregate recommendation** based on:
  - Overall aggregate metrics
  - Percentage of stocks needing retraining
  - If >30% need retraining → RETRAIN NOW
  - If >50% need attention → CONSIDER RETRAINING

### Example Multi-Stock Output

```
MULTI-STOCK Walk-Forward Backtest Results
============================================================

Stocks Tested: 10
Strategy: Trading
Symbols: AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, NFLX, JPM, BAC

🧠 RETRAINING RECOMMENDATION (Multi-Stock Aggregate)
============================================================

🟡 CONSIDER RETRAINING (MEDIUM priority, 45% confidence)

⚠️ Reasons to Retrain:
  1. Average accuracy (47.8%) is below 50% - below acceptable
  2. 3/10 stocks (30%) showed critical performance issues (RETRAIN NOW)
  3. 5/10 stocks (50%) need attention (CONSIDER RETRAINING)

Aggregate Results (Across All Stocks):
------------------------------------------------------------
Average Accuracy: 47.80%
Std Deviation: 12.50%
Min Accuracy: 28.00%
Max Accuracy: 65.00%
Average Return: 2.30%
Average Sharpe Ratio: 0.45

Retraining Breakdown:
  🔴 RETRAIN NOW: 3 stocks
  🟡 CONSIDER RETRAINING: 5 stocks
  🟢 NO RETRAIN NEEDED: 2 stocks

Per-Stock Summary:
------------------------------------------------------------
AAPL: 52.1% accuracy - NO RETRAIN NEEDED
MSFT: 48.3% accuracy - CONSIDER RETRAINING
GOOGL: 35.2% accuracy - RETRAIN NOW
...
```

---

## Summary

The automatic retraining recommendation system:

✅ **Analyzes** multiple performance metrics  
✅ **Provides** clear, actionable recommendations  
✅ **Shows** visual indicators in UI  
✅ **Explains** reasons for each recommendation  
✅ **Helps** you know when to retrain vs. when not to  
✅ **NEW:** Supports multi-stock testing for more robust recommendations  

**Key Takeaway:** Trust the recommendations! They're based on proven thresholds and multiple indicators. When it says "RETRAIN NOW", do it. When it says "NO RETRAIN NEEDED", your models are performing well! 

**Pro Tip:** Use multi-stock mode for the most reliable retraining decisions - it tests generalization across multiple stocks and provides aggregate recommendations! 🎯

