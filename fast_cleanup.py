import os
import glob
from pathlib import Path

# Path from config
APP_DATA_DIR = Path(r"c:\Users\Никола\.stocker")

files_to_delete = [
    "predictions.json",        # The "poisoned" verified history
    "backtest_results.json",   # Old backtest results
    "tested_dates.json",       # Cache of what we tested
    "calibration_data.json",   # Old calibration state
    "ml_model_*.pkl",          # The actual models (Force Retrain)
    "scaler_*.pkl",            # Old scalers
    "label_encoder_*.pkl",     # Old encoders
    "ml_metadata_*.json"       # Old metadata
]

print(f"cleaning up data in {APP_DATA_DIR}...")

count = 0
for pattern in files_to_delete:
    # Use glob to handle wildcards
    full_pattern = str(APP_DATA_DIR / pattern)
    matches = glob.glob(full_pattern)
    
    for file_path in matches:
        try:
            os.remove(file_path)
            print(f"Deleted: {Path(file_path).name}")
            count += 1
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

print(f"Done. Deleted {count} files.")
