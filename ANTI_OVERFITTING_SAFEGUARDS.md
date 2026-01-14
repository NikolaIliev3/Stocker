# Anti-Overfitting Safeguards

## Problem: Data Leakage from Repeated Backtesting

**The Issue**: If you repeatedly run backtests on the same dates and use those results for training, the model can "cheat" by memorizing those specific dates instead of learning general patterns. This is called **data leakage** or **temporal leakage**.

**Example**:
- Backtest on AAPL @ 2023-01-15 → Result: BUY was correct
- Use this result to retrain model
- Backtest again on AAPL @ 2023-01-15 → Model now "knows" the answer
- Model gets better at predicting those specific dates, but not new ones

## Solution: Multiple Safeguards

### ✅ Safeguard 1: Date Tracking
**What it does**: Tracks every date that has been tested (symbol + date combinations)

**How it works**:
- Every time a date is tested, it's marked in `tested_dates.json`
- Before using a date for training, we check if it was tested recently
- Dates tested within the last 90 days are excluded from training

**File**: `tested_dates.json` (stored in app data directory)

### ✅ Safeguard 2: Cooldown Period
**What it does**: Prevents retraining too frequently

**Settings**:
- **Cooldown**: 24 hours between retrains (per strategy)
- Prevents the model from being retrained multiple times per day

**Why**: Frequent retraining on overlapping data causes overfitting

### ✅ Safeguard 3: Overlap Threshold
**What it does**: Limits how many training samples can come from recently tested dates

**Settings**:
- **Max overlap**: 30% (configurable via `max_overlap_threshold`)
- **Recent window**: 90 days (configurable via `recent_test_window_days`)

**How it works**:
- Calculates what % of training samples are from recently tested dates
- If overlap > 30%, retraining is blocked
- Prevents the model from memorizing test dates

### ✅ Safeguard 4: Fresh Date Filtering
**What it does**: Only uses "fresh" dates (not recently tested) for training

**How it works**:
- Filters out any results from dates tested within the last 90 days
- Only converts "fresh" backtest results to training samples
- Logs how many results were skipped due to recent testing

### ✅ Safeguard 5: Minimum Fresh Samples
**What it does**: Requires at least 20 fresh (non-recently-tested) samples to retrain

**Why**: Prevents retraining on too few samples, which could cause instability

## Configuration

You can adjust these settings in `continuous_backtester.py`:

```python
# Minimum cooldown between retrains (hours)
self.retrain_cooldown_hours = 24  # Don't retrain more than once per day

# Maximum overlap threshold: don't retrain if >X% of samples are from recently tested dates
self.max_overlap_threshold = 0.3  # 30% max overlap with tested dates

# Days to consider "recently tested" (dates tested within this window are excluded from training)
self.recent_test_window_days = 90  # 3 months
```

## How It Prevents "Cheating"

### Before (Without Safeguards):
1. Backtest on AAPL @ 2023-01-15 → Result saved
2. Use result to retrain model
3. Backtest again on AAPL @ 2023-01-15 → Model "remembers" the answer
4. Accuracy improves artificially (only on those specific dates)
5. Real-world accuracy doesn't improve

### After (With Safeguards):
1. Backtest on AAPL @ 2023-01-15 → Date marked as tested
2. Try to use result for training → **BLOCKED** (date tested recently)
3. Backtest on AAPL @ 2022-06-10 → Date not recently tested
4. Use this result for training → **ALLOWED** (fresh date)
5. Model learns from diverse, non-overlapping dates
6. Real-world accuracy improves genuinely

## Log Messages

You'll see these log messages when safeguards are active:

```
🛡️ Anti-overfitting: Skipped 45/100 results from recently tested dates (trading)
⏸️ Retrain cooldown active (trading): 12.3h < 24h. Skipping retrain to prevent overfitting.
🛡️ Anti-overfitting: Overlap too high (trading): 35.2% > 30.0%. Skipping retrain to prevent memorization.
⏸️ Insufficient fresh results for retraining (trading): 15 fresh (need 20). Skipped 85 recently tested dates.
```

## Best Practices

1. **Run backtests periodically** (e.g., daily), not constantly
2. **Monitor log messages** to see when safeguards are active
3. **Adjust thresholds** if you have very limited historical data
4. **Use diverse date ranges** when manually triggering backtests
5. **Trust the safeguards** - they're preventing overfitting, not blocking legitimate learning

## Expected Behavior

- **First backtest run**: All dates are "fresh" → Training allowed
- **Second backtest run (same day)**: Cooldown active → Training blocked
- **Third backtest run (next day)**: Some dates still recent → Only fresh dates used
- **After 90 days**: Old test dates become "fresh" again → Can be used for training

This ensures the model learns from diverse, non-overlapping data, preventing memorization and improving real-world accuracy.
