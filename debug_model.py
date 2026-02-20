import pickle
import numpy as np
import pandas as pd
from pathlib import Path

def debug_model():
    base_path = Path("c:/Users/Никола/.stocker")
    
    # Load Label Encoder
    le_path = base_path / "label_encoder_trading_bull.pkl"
    if le_path.exists():
        with open(le_path, 'rb') as f:
            le = pickle.load(f)
        print(f"Label Encoder Classes: {le.classes_}")
    else:
        print("Label encoder (bull) not found")

    # Load Model
    model_path = base_path / "ml_model_trading_bull.pkl"
    if model_path.exists():
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Model Object: {type(model)}")
        
        # Check Metadata
        meta_path = base_path / "ml_metadata_trading.json"
        import json
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            print(f"Meta Trained Acc: {meta.get('train_accuracy')}")
            print(f"Meta Selected Features: {meta.get('regime_metadata', {}).get('bull', {}).get('selected_feature_names')[:10]}")

if __name__ == "__main__":
    debug_model()
