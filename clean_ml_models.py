import os
import glob
from pathlib import Path

# Define the target directory
target_dir = Path(os.path.expanduser('~')) / ".stocker"

print(f"Cleaning files in: {target_dir}")

# List of patterns to delete
patterns = [
    "ml_model_trading*.pkl",
    "model.pkl",
    "ml_metadata_trading*.json",
    "scaler_trading*.pkl",
    "label_encoder_trading*.pkl",
    "model_checksums.json",
    "predictions.json",
    "accuracy_history.json"
]

deleted_count = 0

if target_dir.exists():
    for pattern in patterns:
        files = list(target_dir.glob(pattern))
        for file_path in files:
            try:
                os.remove(file_path)
                print(f"✓ Deleted: {file_path.name}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Error deleting {file_path.name}: {e}")
else:
    print(f"Directory not found: {target_dir}")

print(f"\nTotal files deleted: {deleted_count}")
