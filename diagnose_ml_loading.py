"""
Diagnose ML model loading issues
"""
import sys
from pathlib import Path
from config import APP_DATA_DIR
from secure_ml_storage import SecureMLStorage
import json

print("="*60)
print("DIAGNOSING ML MODEL LOADING")
print("="*60)

storage = SecureMLStorage(APP_DATA_DIR)
strategy = 'trading'

# Check what files exist
print(f"\n1. Checking for model files in {APP_DATA_DIR}...")
files_to_check = [
    f"ml_model_{strategy}.pkl",
    f"scaler_{strategy}.pkl",
    f"label_encoder_{strategy}.pkl",
    f"ml_metadata_{strategy}.json",
    f"ml_model_{strategy}_sideways.pkl",
    f"scaler_{strategy}_sideways.pkl",
    f"label_encoder_{strategy}_sideways.pkl",
]

for filename in files_to_check:
    exists = storage.model_exists(filename)
    print(f"   {filename}: {'EXISTS' if exists else 'MISSING'}")

# Check metadata
metadata_file = APP_DATA_DIR / f"ml_metadata_{strategy}.json"
if metadata_file.exists():
    print(f"\n2. Reading metadata...")
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        print(f"   use_regime_models: {metadata.get('use_regime_models', False)}")
        print(f"   test_accuracy: {metadata.get('test_accuracy', 0)*100:.1f}%")
        if 'regime_metadata' in metadata:
            print(f"   regime_metadata keys: {list(metadata['regime_metadata'].keys())}")
            for regime, meta in metadata['regime_metadata'].items():
                print(f"     - {regime}: {meta.get('test_accuracy', 0)*100:.1f}%")
    except Exception as e:
        print(f"   ERROR reading metadata: {e}")
else:
    print(f"\n2. Metadata file NOT FOUND")

# Try to load a regime model directly
print(f"\n3. Trying to load regime model directly...")
regime = 'sideways'
regime_model_file = f"ml_model_{strategy}_{regime}.pkl"
if storage.model_exists(regime_model_file):
    try:
        model = storage.load_model(regime_model_file)
        print(f"   ✓ Loaded model: {type(model)}")
        print(f"   ✓ Model is not None: {model is not None}")
    except Exception as e:
        print(f"   ✗ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"   ✗ Model file does not exist")

print("\n" + "="*60)
