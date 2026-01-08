# Backtest Configuration Guide

## Overview
You now have full control over the number of backtest points to run. This allows you to balance speed vs. accuracy confidence based on your needs.

## How to Configure

### Method 1: In the App (Recommended)
1. Click **🧪 Run Backtest** in the ML Training tab
2. A dialog will appear with:
   - Strategy selection checkboxes
   - Number of test points input field
   - Quick preset buttons (Quick/Standard/Thorough)
3. Choose your settings and click **Start Backtest**

### Method 2: In config.py (Global Default)
Edit `config.py` and change:
```python
BACKTEST_TESTS_PER_RUN = 100  # Change this number
```

## Recommendations

### Quick Test (30 tests)
- **Duration**: ~2-5 minutes
- **Use case**: Quick validation after small changes
- **Accuracy confidence**: Low (small sample size)
- **Example**: Testing if a bug fix works

### Standard Test (100 tests) ⭐ RECOMMENDED
- **Duration**: ~10-15 minutes
- **Use case**: Regular validation and performance tracking
- **Accuracy confidence**: Good (sufficient for most cases)
- **Example**: After training a new model or adjusting parameters

### Thorough Test (200 tests)
- **Duration**: ~20-30 minutes
- **Use case**: High-confidence validation before major decisions
- **Accuracy confidence**: High (large sample, more statistically significant)
- **Example**: Before deploying to production or making strategy changes

### Deep Analysis (300-500 tests)
- **Duration**: ~1-2 hours
- **Use case**: Research, algorithm comparison, publication
- **Accuracy confidence**: Very high
- **Example**: Comparing multiple algorithms or documenting performance

## Understanding the Results

### Small Sample Size (< 50 tests)
- Results can vary wildly (±20% accuracy swings)
- Not reliable for decision-making
- Good for quick sanity checks only

### Medium Sample Size (50-150 tests)
- Results are reasonably stable (±10% accuracy swings)
- Good for most development and validation
- **This is the sweet spot for regular testing**

### Large Sample Size (150+ tests)
- Results are very stable (±5% accuracy swings)
- High confidence in reported accuracy
- Takes longer but worth it for important decisions

## Current Issue: BUY Predictions

Your latest log showed:
- **Overall accuracy**: 52.7% (58/110 correct)
- **BUY prediction accuracy**: 31% (11/36 correct) ⚠️
- **Issue**: When the system predicts BUY (expecting price to go DOWN), it's often wrong (69% failure rate)

### Why This Happens
1. **Small sample**: Only 10 new tests were run, not enough to measure accurately
2. **Historical data**: 110 total tests include older model versions (mixed results)
3. **BUY is harder**: Predicting market bottoms is notoriously difficult

### Next Steps
1. **Run 100-200 tests** with your newly trained model (62.4% validation accuracy)
2. **Check the logs** for diagnostic info we added:
   - Look for "BUY prediction evaluation" lines
   - Check if ML or rule-based is making the BUY predictions
   - See ensemble weights (ML vs rule-based)
3. **If BUY is still weak** after more tests, we can:
   - Add confidence thresholds (only BUY when >70% confident)
   - Adjust decision thresholds asymmetrically
   - Train specifically on BUY scenarios

## Tips
- **Start with 100 tests** (Standard) for your next backtest
- **Run tests after every model training** to track improvement
- **Compare "new tests" accuracy to ML validation accuracy** (should be close)
- **Ignore "overall accuracy"** if you've trained new models - it mixes old and new
- **Use logs** to debug when backtest accuracy doesn't match ML training

## Configuration Files Modified
- `config.py`: Added `BACKTEST_TESTS_PER_RUN` setting
- `continuous_backtester.py`: Now reads from config
- `main.py`: Added UI controls for test count selection

## Example Usage
```
User wants quick check: Select "Quick (30)" preset → 5 min wait
User wants regular validation: Select "Standard (100)" → 15 min wait
User wants high confidence: Select "Thorough (200)" → 30 min wait
User doing research: Enter "500" manually → 1-2 hour wait
```

---

**Remember**: More tests = more accurate measurement, but diminishing returns after ~200 tests.
For day-to-day work, 100 tests is the sweet spot between speed and accuracy.
