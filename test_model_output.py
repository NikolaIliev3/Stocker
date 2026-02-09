"""Quick model prediction test"""
import sys, os
import numpy as np
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_training import StockPredictionML

# Load model - CORRECT order: (data_dir, strategy)
data_dir = Path.home() / '.stocker'
ml = StockPredictionML(data_dir, 'trading')

print("="*50)
print("MODEL PREDICTION TEST")
print("="*50)

# Check regime models
if ml.regime_models:
    print(f"Regime models: {list(ml.regime_models.keys())}")
    
    for regime, model in ml.regime_models.items():
        if hasattr(model, 'label_encoder') and hasattr(model.label_encoder, 'classes_'):
            print(f"  {regime}: classes = {list(model.label_encoder.classes_)}")
        if hasattr(model, 'model') and model.model is not None:
            print(f"  {regime}: model loaded = True")
else:
    print("No regime models!")

# Create random test features (190 features)
print("\nTesting with random features...")
np.random.seed(42)
test_features = np.random.randn(190)

# Get bull model and test
bull = ml.regime_models.get('bull')
if bull and bull.model:
    pred = bull.predict(test_features, {}, {}, is_backtest=True)
    print(f"\nBull model prediction: {pred.get('action')} ({pred.get('confidence', 0):.1f}%)")
    print(f"  Probabilities: {pred.get('probabilities', {})}")

# Test with different feature patterns
print("\n--- Testing with biased features ---")

# All high values
test_high = np.ones(190) * 0.8
pred_high = bull.predict(test_high, {}, {}, is_backtest=True)
print(f"All high features: {pred_high.get('action')} ({pred_high.get('confidence', 0):.1f}%)")

# All low values
test_low = np.ones(190) * -0.8
pred_low = bull.predict(test_low, {}, {}, is_backtest=True)
print(f"All low features: {pred_low.get('action')} ({pred_low.get('confidence', 0):.1f}%)")

# Zero values
test_zero = np.zeros(190)
pred_zero = bull.predict(test_zero, {}, {}, is_backtest=True)
print(f"All zero features: {pred_zero.get('action')} ({pred_zero.get('confidence', 0):.1f}%)")

print("\n" + "="*50)
