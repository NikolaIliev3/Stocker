"""
Diagnose SELL Bias in ML Model
This script checks how labels are encoded and decoded to find the inversion bug.
"""
import sys
sys.path.insert(0, '.')

from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Find the model files
from config import APP_DATA_DIR
data_dir = Path(APP_DATA_DIR)

# Load secure storage
from secure_ml_storage import SecureMLStorage
storage = SecureMLStorage(data_dir)

print("=" * 70)
print("DIAGNOSING SELL BIAS - LABEL ENCODING CHECK")
print("=" * 70)

# Check each regime model
regimes = ['bull', 'bear', 'sideways']

for regime in regimes:
    print(f"\n{'='*30} {regime.upper()} MODEL {'='*30}")
    
    # Load label encoder
    encoder_file = f"label_encoder_trading_{regime}.pkl"
    encoder = storage.load_model(encoder_file)
    
    if encoder is None:
        print(f"  ❌ Could not load {encoder_file}")
        continue
    
    classes = list(encoder.classes_)
    print(f"  Label Encoder Classes: {classes}")
    print(f"  Class Index Mapping:")
    for i, cls in enumerate(classes):
        print(f"    {i} -> {cls}")
    
    # Check model
    model_file = f"ml_model_trading_{regime}.pkl"
    model = storage.load_model(model_file)
    
    if model is None:
        print(f"  ❌ Could not load {model_file}")
        continue
    
    print(f"  Model type: {type(model).__name__}")
    
    # If it's a calibrated classifier, get the base classes
    if hasattr(model, 'classes_'):
        print(f"  Model.classes_: {list(model.classes_)}")
    
    # Test prediction with dummy data
    print(f"\n  Testing prediction flow...")
    import numpy as np
    
    # Load scaler
    scaler_file = f"scaler_trading_{regime}.pkl"
    scaler = storage.load_model(scaler_file)
    
    if scaler is None:
        print(f"  ❌ Could not load {scaler_file}")
        continue
    
    n_features = scaler.n_features_in_
    print(f"  Scaler expects {n_features} features")
    
    # Create dummy feature vector (all zeros)
    dummy_features = np.zeros((1, n_features))
    dummy_scaled = scaler.transform(dummy_features)
    
    # Get probabilities
    try:
        probs = model.predict_proba(dummy_scaled)[0]
        pred_idx = model.predict(dummy_scaled)[0]
        
        print(f"\n  Dummy Test (all-zero features):")
        print(f"    Raw probabilities: {probs}")
        print(f"    Predicted class index: {pred_idx}")
        print(f"    Decoded label: {encoder.inverse_transform([pred_idx])[0]}")
        
        print(f"\n  Probability interpretation:")
        for i, p in enumerate(probs):
            label = encoder.classes_[i]
            # How the code converts to BUY/SELL
            display = 'SELL' if label == 'DOWN' else ('BUY' if label == 'UP' else label)
            print(f"    prob[{i}] = {p:.4f} -> {label} -> display as {display}")
        
        # Check which has higher probability
        max_idx = np.argmax(probs)
        max_label = encoder.classes_[max_idx]
        max_display = 'SELL' if max_label == 'DOWN' else ('BUY' if max_label == 'UP' else max_label)
        print(f"\n  ⚡ Final prediction: {max_display} (prob={probs[max_idx]:.4f})")
        
        # THE KEY CHECK: Is the model correctly predicting based on training data?
        # Training has MORE UP (839) than DOWN (360), so with neutral features,
        # the model's prior should lean toward UP (BUY), NOT DOWN (SELL)
        if max_display == 'SELL':
            print(f"\n  ⚠️ WARNING: Model predicts SELL even with neutral features!")
            print(f"  ⚠️ This suggests the model is biased toward SELL/DOWN class.")
            print(f"  ⚠️ But training had 839 UP vs 360 DOWN - should be biased toward UP!")
        else:
            print(f"\n  ✅ Model predicts BUY with neutral features (expected with 839 UP vs 360 DOWN)")
            
    except Exception as e:
        print(f"  ❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()

# Check metadata
print("\n" + "=" * 70)
print("CHECKING METADATA")
print("=" * 70)

for regime in regimes:
    metadata_file = data_dir / f"ml_metadata_trading_{regime}.json"
    if metadata_file.exists():
        with open(metadata_file) as f:
            meta = json.load(f)
        print(f"\n{regime.upper()} metadata:")
        print(f"  use_binary_classification: {meta.get('use_binary_classification', 'NOT SET')}")
        print(f"  train_accuracy: {meta.get('train_accuracy', 'N/A')}")  
        print(f"  test_accuracy: {meta.get('test_accuracy', 'N/A')}")
    else:
        print(f"\n{regime.upper()} metadata: FILE NOT FOUND")
