# Training & Backtesting Guide - What to Look For

This guide explains what improvements you'll see when running ML training and backtesting with the new production-ready features.

## 🚀 Quick Start

1. **Train ML Models**: Use the training function in the app
2. **Run Backtesting**: Use the continuous backtester
3. **Watch the Logs**: Check console output for new features in action

---

## ✨ New Features You'll See

### 1. **Model Versioning & Rollback** (During Training)

**What to Look For:**

When you start training, you'll see:
```
📦 Created version backup: 20260110_123456
```

After training completes:
```
✅ Registered new active version 20260110_123456 (accuracy: 62.0%)
```

If new model is worse (>5% drop):
```
⚠️ New model performs worse: Accuracy dropped by 7.0%
❌ Rejected new version 20260110_123456 due to performance degradation
✅ Rolled back to previous version 20260110_123457
```

**Location:** `data/model_versions/{strategy}/` directory will contain version backups

---

### 2. **Data Quality Validation** (During Training)

**What to Look For:**

Before training starts:
```
✅ Training data validation passed
  Sample count: 1000
  Feature count: 75
```

If issues found:
```
❌ Training data validation failed:
   - High missing value percentage: 6.2%
   - Class imbalance detected: 1.8:1
```

**Benefits:** Prevents training on bad data, saves time

---

### 3. **Production Monitoring** (During Predictions)

**What to Look For:**

After making predictions:
- Predictions are automatically tracked
- No immediate output (works in background)

To check monitoring status:
```python
from production_monitor import ProductionMonitor
monitor = ProductionMonitor(Path("data"), "trading")
perf = monitor.get_current_performance()
print(f"Accuracy: {perf['accuracy']:.1f}%")
```

**Alerts** (in logs if issues detected):
```
🚨 PRODUCTION ALERT (trading): Production accuracy (38.5%) below threshold (40%)
🚨 PRODUCTION ALERT (trading): Accuracy dropped by 16.2%
```

**Location:** `data/production_monitoring_{strategy}.json`

---

### 4. **Model Explainability** (In Predictions)

**What to Look For:**

If SHAP is installed (`pip install shap`):
```python
prediction = ml_model.predict(features)
if 'explanation' in prediction:
    print(prediction['explanation']['top_contributors'])
    # Output: [('rsi', 0.12), ('macd', 0.08), ('volume_ratio', 0.06)]
    print(prediction['explanation']['reasoning'])
    # Output: ['Strongest positive signal: rsi (contribution: 0.12)']
```

If SHAP not installed, uses feature importance (still works):
```python
prediction['explanation']['method']  # 'feature_importance'
```

**Benefits:** Understand WHY predictions are made

---

### 5. **Prediction Quality Scoring** (In Predictions)

**What to Look For:**

All predictions now include quality scores:
```python
prediction = ml_model.predict(features)
if 'quality_score' in prediction:
    quality = prediction['quality_score']
    print(f"Quality: {quality['quality_level']}")  # 'high', 'medium', 'low'
    print(f"Score: {quality['quality_score']:.2f}")  # 0.0-1.0
    if quality.get('should_flag'):
        print("⚠️ Low quality prediction flagged!")
```

**In Logs:**
```
⚠️ Low quality prediction flagged: very_low (score: 0.25)
```

**Benefits:** Identify unreliable predictions before acting on them

---

### 6. **Performance Attribution** (During Predictions)

**What to Look For:**

Automatically tracks outcomes (works in background)

To view attribution report:
```python
from performance_attribution import PerformanceAttribution
attribution = PerformanceAttribution(Path("data"))
report = attribution.get_attribution_report()

print("By Strategy:", report['by_strategy'])
print("By Regime:", report['by_regime'])
print("By Confidence:", report['by_confidence'])
print("Insights:", report['insights'])
```

**Example Insights:**
```
✅ Outperforms random baseline by 12.3%
✅ Best performing strategy: trading (58.5% accuracy)
✅ Best performing regime: bull (62.1% accuracy)
⚠️ Confidence calibration issue: High confidence (55.2%) not much better than low (52.1%)
```

**Location:** `data/performance_attribution.json`

---

### 7. **Model Benchmarking** (During Backtesting)

**What to Look For:**

After backtesting completes:
```
📊 Benchmarking (trading): Model 58.5% vs Random 50% (difference: +8.5%)
📊 Benchmarking (trading): Model 58.5% vs Majority Class 52.3% (difference: +6.2%)
```

**In Logs:**
```
✅ Baselines calculated: 4 baselines
✅ Comparisons generated: 3 comparisons
```

To view benchmark report:
```python
from model_benchmarking import ModelBenchmarker
benchmarker = ModelBenchmarker(Path("data"))
report = benchmarker.get_benchmark_report('trading')

print("Model Accuracy:", report['model_accuracy'])
print("Baselines:", report['baselines'])
print("Comparisons:", report['comparisons'])
print("Summary:", report['summary'])
```

**Example Summary:**
```
✅ Outperforms random baseline by 8.5%
✅ Outperforms majority class by 6.2%
🎯 Model performance: Good (55-60%)
```

**Location:** `data/model_benchmarks.json`

---

## 📊 Expected Training Output

When you run training, you should see:

```
Training ML model on 1000 samples with random_seed=1234...
  • Hyperparameter tuning: True
  • Regime-specific models: True
  • Feature selection method: rfe
  • Model family: voting
  • Binary classification: False
  • SMOTE balancing: False

📦 Created version backup: 20260110_123456
✅ Training data validation passed
  Sample count: 1000
  Feature count: 75

Running 5-fold cross-validation...
Cross-validation complete: 61.2% ± 3.1%

✅ Registered new active version 20260110_123456 (accuracy: 62.0%)
Trained and saved ML model: 62.0% accuracy
```

---

## 📈 Expected Backtesting Output

When you run backtesting, you should see:

```
✅ Continuous backtest complete: 50 new tests run
   • Overall accuracy: 58.5% (29/50 correct)
   • Trading: 58.5% (29/50)
   
📊 Benchmarking (trading): Model 58.5% vs Random 50% (difference: +8.5%)
   Control: 58.5%
   Treatment: N/A (no A/B test active)
```

---

## 🔍 How to Verify Everything Works

### 1. Check Version Backups
```bash
ls data/model_versions/trading/
# Should show version directories with timestamps
```

### 2. Check Production Monitoring
```python
from production_monitor import ProductionMonitor
monitor = ProductionMonitor(Path("data"), "trading")
print(monitor.get_current_performance())
```

### 3. Check Performance Attribution
```python
from performance_attribution import PerformanceAttribution
attribution = PerformanceAttribution(Path("data"))
report = attribution.get_attribution_report()
print(f"Strategies tracked: {len(report['by_strategy'])}")
```

### 4. Check Benchmarks
```python
from model_benchmarking import ModelBenchmarker
benchmarker = ModelBenchmarker(Path("data"))
report = benchmarker.get_benchmark_report('trading')
print(f"Model accuracy: {report['model_accuracy']:.1f}%")
```

---

## 🎯 What to Test

### Test Scenario 1: First Training
- ✅ Should create version backup
- ✅ Should validate data
- ✅ Should save model files
- ✅ Should register version

### Test Scenario 2: Train Better Model
- ✅ Should create new version backup
- ✅ Should compare to previous version
- ✅ Should activate if better
- ✅ Should keep old version as backup

### Test Scenario 3: Train Worse Model
- ✅ Should create version backup
- ✅ Should detect worse performance
- ✅ Should reject new version
- ✅ Should rollback to previous

### Test Scenario 4: Make Predictions
- ✅ Should track in production monitor
- ✅ Should include quality scores
- ✅ Should include explanations (if SHAP installed)

### Test Scenario 5: Verify Predictions
- ✅ Should record outcomes in attribution
- ✅ Should update performance metrics
- ✅ Should generate alerts if issues

### Test Scenario 6: Run Backtesting
- ✅ Should calculate baselines
- ✅ Should compare to baselines
- ✅ Should show benchmark report

---

## 💡 Tips

1. **Check Logs**: All new features log their activity - watch the console output
2. **Data Files**: New JSON files will appear in `data/` directory
3. **Version History**: Check `data/model_versions/{strategy}/version_history.json`
4. **Install SHAP**: For better explainability: `pip install shap`
5. **Be Patient**: First training may take several minutes

---

## 🚨 Troubleshooting

**Issue:** "Model versioning not available" warning
- **Solution:** Feature is optional, will gracefully degrade

**Issue:** "SHAP not available" warning
- **Solution:** Install with `pip install shap` (optional)

**Issue:** No version backup created
- **Solution:** This is normal if no previous model exists

**Issue:** Low quality scores on all predictions
- **Solution:** This indicates model needs retraining

---

## ✅ Success Indicators

You'll know everything is working if you see:

1. ✅ Version backups created during training
2. ✅ Data validation messages before training
3. ✅ Benchmark reports after backtesting
4. ✅ Quality scores in predictions (check logs)
5. ✅ New JSON files in `data/` directory
6. ✅ Attribution insights after verifying predictions
7. ✅ Production monitoring alerts (if issues detected)

---

## 🎉 Ready to Go!

Run your training and backtesting now - all new features are automatically active and will work in the background. Check the logs and data files to see them in action!
