# AI Predictor Testing Report

## Purpose
Test whether SeekerAI actually improves prediction accuracy compared to rule-based + ML only.

## Test Methodology

The test compares:
1. **Hybrid System WITH AI**: Rule-based + ML + AI predictions
2. **Hybrid System WITHOUT AI**: Rule-based + ML predictions only

### Test Process:
1. Select historical data points (e.g., 50 points per symbol)
2. For each point:
   - Make prediction WITH AI
   - Make prediction WITHOUT AI
   - Check actual outcome after N days (e.g., 10 days)
   - Record if prediction was correct
3. Calculate accuracy for both approaches
4. Compare results

### Success Criteria:
- **AI is helpful** if accuracy WITH AI > accuracy WITHOUT AI by >2%
- **AI is harmful** if accuracy WITH AI < accuracy WITHOUT AI by >2%
- **AI has minimal impact** if difference is within ±2%

## Running the Test

```bash
python test_ai_accuracy.py [SYMBOLS] [STRATEGY] [TEST_POINTS]

# Example:
python test_ai_accuracy.py AAPL,MSFT,GOOGL trading 20
```

### Parameters:
- **SYMBOLS**: Comma-separated list (e.g., AAPL,MSFT,GOOGL)
- **STRATEGY**: trading, mixed, or investing
- **TEST_POINTS**: Number of historical points to test per symbol (default: 20)

## Expected Results

The test will output:
- Accuracy WITH AI
- Accuracy WITHOUT AI
- Improvement/decline percentage
- AI usage rate (how often AI predictions were actually used)
- Confidence statistics
- Action distribution

## Interpreting Results

### If AI Improves Accuracy:
✅ **Keep AI enabled** - It's adding value

### If AI Reduces Accuracy:
❌ **Consider disabling AI** - It's adding noise

### If AI Has Minimal Impact:
⚠️ **Test more** - May need more data points or different stocks

## Notes

- AI predictions require SeekerAI to be enabled and configured
- AI may not be available for all stocks (depends on news availability)
- Test uses historical data, so results reflect past performance
- Real-world performance may differ

## Recommendations

Based on test results:
1. If AI improves accuracy: Keep it enabled, consider increasing weight
2. If AI reduces accuracy: Disable it or reduce weight
3. If minimal impact: Test with more stocks/time periods before deciding
