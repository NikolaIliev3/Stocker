
import sys
import os
import joblib
from pathlib import Path

# Setup paths
data_dir = Path.home() / '.stocker'
models_dir = data_dir / 'models'

print(f"Checking models in {models_dir}...")

def check_model(strategy):
    model_path = models_dir / f"{strategy}_model_latest.pkl"
    if not model_path.exists():
        print(f"❌ Model for {strategy} not found.")
        return

    try:
        data = joblib.load(model_path)
        print(f"\n--- {strategy.upper()} MODEL METADATA ---")
        
        # Check if it's the new class wrapper or raw dict
        if hasattr(data, 'metadata'):
            metadata = data.metadata
            print("Loaded Class Object")
        elif isinstance(data, dict):
             if 'metadata' in data:
                 metadata = data['metadata']
                 print("Loaded Dict with metadata")
             else:
                 metadata = data
                 print("Loaded Raw Dict")
        else:
             print(f"Unknown type: {type(data)}")
             metadata = {}

        # Print Key Fields
        fields = ['training_date', 'training_end_date', 'training_samples', 'test_accuracy', 'cv_mean']
        for f in fields:
            val = metadata.get(f, 'N/A')
            print(f"{f}: {val}")
            
    except Exception as e:
        print(f"Error loading model: {e}")

check_model('trading')
