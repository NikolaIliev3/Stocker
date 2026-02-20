"""
Auto Cleanup Script
Enforces the rule: "IF you find faulty/outdated/tainted data YOU DO NOT wait for me to tell you to delete it."
This script wipes old models, scalers, and training caches to ensure a fresh, clean training state.
"""
import os
import shutil
import logging
from pathlib import Path
from config import APP_DATA_DIR

# Define local data dir matching project structure
DATA_DIR = Path(__file__).parent / "data"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("AutoCleanup")

def perform_cleanup(target_dirs=None):
    """
    Deletes all ML model artifacts and training cache.
    """
    if target_dirs is None:
        target_dirs = [
            Path(APP_DATA_DIR),
            Path(DATA_DIR) / "model_versions",
            Path(DATA_DIR) / "training_cache"
        ]
    
    logger.info("🧹 STORAGE CLEANUP PROTOCOL INITIATED")
    logger.info("------------------------------------")
    
    patterns_to_nuke = [
        "ml_model_*.pkl",          # The models
        "scaler_*.pkl",            # The scalers
        "label_encoder_*.pkl",     # The encoders (critical to reset for binary)
        "ml_metadata_*.json",      # Metadata
        "model_checksums.json",
        "predictions.json",        # Old predictions
        "accuracy_history.json",
        "*.joblib",                # Possible joblib dumps
        "calibration_*.pkl"        # Calibrators
    ]
    
    deleted_count = 0
    
    for directory in target_dirs:
        dir_path = Path(directory)
        if not dir_path.exists():
            continue
            
        logger.info(f"Scanning {dir_path}...")
        
        # 1. Delete by pattern
        for pattern in patterns_to_nuke:
            for file_path in dir_path.glob(pattern):
                try:
                    os.remove(file_path)
                    logger.info(f"   Deleted: {file_path.name}")
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"   ❌ Failed to delete {file_path.name}: {e}")

    logger.info("------------------------------------")
    logger.info(f"Cleanup Complete. {deleted_count} files removed.")
    logger.info("Ready for fresh training.")

if __name__ == "__main__":
    perform_cleanup()
