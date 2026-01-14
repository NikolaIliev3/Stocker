# Continuous Backtesting Guide

## What It Does

**Continuous Backtesting** runs backtests non-stop in a loop until you manually stop it. Each backtest:
1. Tests the model on 100 random historical dates
2. Evaluates accuracy
3. Uses results to improve the model (with safeguards)

## How to Use

1. **Click "🔄 Start Continuous Backtesting"**
   - Select which strategies to test (Trading, Mixed, Investing)
   - Confirm you understand the safeguards

2. **Monitor Progress**
   - Status label shows current state ("Running backtest..." or "Waiting for next run...")
   - Check logs for detailed progress
   - View backtest results to see accuracy improvements

3. **Click "⏹️ Stop Continuous Backtesting"** when satisfied
   - Current backtest will complete
   - No new backtests will start

## Will It Actually Improve Accuracy?

### ✅ **YES, but with important limitations:**

#### What WILL Improve:
1. **More Training Data**: Each backtest run adds new training samples (from fresh dates)
2. **Diverse Patterns**: Tests on different dates/stocks = learns more patterns
3. **Realistic Learning**: Safeguards ensure model learns from diverse data, not memorization

#### What WON'T Improve (Limitations):
1. **Cooldown Period**: Model only retrains **once per 24 hours** (anti-overfitting safeguard)
   - Running 100 backtests in one day = still only 1 retrain
   - You need to wait days/weeks to see multiple retrains

2. **Diminishing Returns**: After a while, most dates will be "recently tested"
   - First few runs: Lots of fresh dates → Good learning
   - After many runs: Most dates are recent → Less new data per run
   - Improvement slows down over time

3. **Data Limitations**: Can only learn from historical patterns
   - If market regime changes, historical patterns may not apply
   - Real-world accuracy depends on many factors beyond backtesting

4. **Overfitting Risk**: Even with safeguards, too much backtesting can still cause issues
   - Model might overfit to the specific date range you're testing
   - Real-world performance may not match backtest performance

## Realistic Timeline

### Day 1:
- **Run 1**: Tests 100 dates → ~20-30 fresh dates used for training → Model retrains
- **Run 2-10**: Tests more dates, but cooldown blocks retraining
- **Result**: 1 retrain, slight improvement possible

### Day 2-3:
- **Run 11-20**: More fresh dates accumulate
- **Day 2**: Cooldown expires → Model retrains again
- **Result**: 2 retrains total, moderate improvement possible

### Week 1:
- **Run 50-100**: Many dates tested, but many are now "recent"
- **Retrains**: ~7 retrains (once per day)
- **Result**: Noticeable improvement if model is learning well

### Month 1:
- **Run 500-1000**: Most dates in test range are "recent"
- **Retrains**: ~30 retrains
- **Result**: Significant improvement possible, but diminishing returns

## Best Practices

### ✅ DO:
1. **Run for days/weeks**, not hours
   - Improvement happens over time, not instantly
   - Cooldown period means you need patience

2. **Monitor accuracy trends**
   - Check if accuracy is actually improving
   - If it plateaus, you've hit diminishing returns

3. **Use diverse strategies**
   - Test Trading, Mixed, and Investing separately
   - Each learns different patterns

4. **Check logs regularly**
   - See when safeguards are active
   - Understand why retraining is blocked

### ❌ DON'T:
1. **Don't expect instant results**
   - First retrain happens after 24 hours
   - Improvement is gradual

2. **Don't run indefinitely**
   - After a month, diminishing returns kick in
   - Real-world testing becomes more valuable

3. **Don't ignore safeguards**
   - They're preventing overfitting
   - If they block retraining, it's for a good reason

## Expected Accuracy Improvements

### Conservative Estimate:
- **Starting accuracy**: 40-50%
- **After 1 week**: 45-55% (5-10% improvement)
- **After 1 month**: 50-60% (10-20% improvement)
- **After 3 months**: 55-65% (15-25% improvement)

### Optimistic Estimate (if model learns well):
- **Starting accuracy**: 40-50%
- **After 1 week**: 50-60% (10-20% improvement)
- **After 1 month**: 60-70% (20-30% improvement)
- **After 3 months**: 65-75% (25-35% improvement)

### Reality Check:
- **Market conditions matter**: If market changes, historical patterns may not apply
- **Backtest ≠ Real-world**: Backtest accuracy often overestimates real performance
- **Diminishing returns**: After a certain point, more backtesting won't help

## When to Stop

Stop continuous backtesting when:
1. ✅ **Accuracy plateaus** (not improving for several days)
2. ✅ **You're satisfied** with current accuracy
3. ✅ **You want to test in real-world** (forward testing)
4. ✅ **Safeguards block retraining** frequently (diminishing returns)

## Alternative: Forward Testing

After backtesting, consider **forward testing**:
- Use the model on **real, live data**
- Track predictions vs. actual outcomes
- This is the true test of accuracy
- Forward testing results are more valuable than more backtesting

## Summary

**Will continuous backtesting improve accuracy?**

**YES**, but:
- ✅ It takes **days/weeks**, not hours
- ✅ Improvement is **gradual** (5-25% over weeks/months)
- ✅ **Diminishing returns** after a while
- ✅ **Real-world testing** is ultimately more valuable

**Recommendation**: 
- Run continuous backtesting for **1-2 weeks** initially
- Monitor accuracy trends
- If improving, continue for up to **1 month**
- Then switch to **forward testing** on real data
- Use forward testing results to further improve the model

The safeguards ensure the model learns genuinely, not by memorization. But patience is required - improvement happens over time, not instantly.
