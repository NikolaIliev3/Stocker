import os
import sys
import pickle
import pandas as pd
from config import APP_DATA_DIR

def inspect_model():
    regimes = ['bull', 'bear', 'sideways']
    
    for regime in regimes:
        filename = f"ml_model_trading_{regime}.pkl"
        model_path = os.path.join(APP_DATA_DIR, filename)
        
        print(f"\n--- Checking {filename} ---")
        if not os.path.exists(model_path):
            print(f"Model not found at {model_path}")
            continue

        try:
            # All metadata (including regimes) is in the main trading metadata file
            meta_path = os.path.join(APP_DATA_DIR, "ml_metadata_trading.json")
            
            if os.path.exists(meta_path):
                import json
                with open(meta_path, 'r') as f:
                    full_metadata = json.load(f)
                
                # Check for global info or regime-specific info
                if regime == 'total' or regime not in full_metadata.get('regime_metadata', {}):
                    metadata = full_metadata
                else:
                    metadata = full_metadata['regime_metadata'][regime]
                
                print(f"Model Metadata ({regime}):")
                # Print high level metrics
                print(f"  Samples: {metadata.get('samples', 'N/A')}")
                print(f"  Test Accuracy: {metadata.get('test_accuracy', 0)*100:.1f}%")
                print(f"  CV Mean: {metadata.get('cv_mean', 0)*100:.1f}%")
                
                if 'training_data_start' in metadata:
                    print(f"  Range: {metadata['training_data_start']} to {metadata['training_data_end']}")
            else:
                 print(f"Metadata file not found: {meta_path}")
                
        except Exception as e:
            print(f"Error loading metadata: {e}")

if __name__ == "__main__":
    inspect_model()
