"""
Safely clear ML model files (if you want to retrain from scratch)
This only clears the model, NOT your predictions or other data
"""
import os
from pathlib import Path
from config import APP_DATA_DIR

def clear_ml_model_safe(strategy='trading'):
    print("="*60)
    print("CLEARING ML MODEL (SAFE - keeps all other data)")
    print("="*60)
    
    data_dir = APP_DATA_DIR
    
    files_to_delete = [
        data_dir / f"ml_model_{strategy}.pkl",
        data_dir / f"scaler_{strategy}.pkl",
        data_dir / f"ml_metadata_{strategy}.json",
        data_dir / f"label_encoder_{strategy}.pkl",
        data_dir / f"checksums_{strategy}.json",
    ]
    
    deleted = []
    not_found = []
    
    for file_path in files_to_delete:
        if file_path.exists():
            try:
                os.remove(file_path)
                print(f"[DELETED] {file_path.name}")
                deleted.append(file_path.name)
            except Exception as e:
                print(f"[ERROR] Could not delete {file_path.name}: {e}")
        else:
            not_found.append(file_path.name)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Deleted: {len(deleted)} files")
    print(f"Not found: {len(not_found)} files")
    print("\nPRESERVED:")
    print("  - predictions.json (your predictions)")
    print("  - portfolio.json (your portfolio)")
    print("  - backtest_results.json (backtest history)")
    print("  - learning_tracker.json (statistics)")
    print("\nNext step: Retrain ML model from historical data")
    print("="*60)

if __name__ == '__main__':
    clear_ml_model_safe('trading')
