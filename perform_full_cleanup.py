
import os
import glob
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def perform_cleanup():
    """
    Simulates 'Clear All Data' from the Danger Zone.
    Removes ML models, metadata, scalers, encoders, and training history.
    Preserves: preferences.json, api_keys.json (if exists)
    """
    data_dir = Path.home() / ".stocker"
    
    if not data_dir.exists():
        logger.info("Data directory does not exist. Nothing to clear.")
        return

    logger.info(f"🧹 Starting Full Cleanup in {data_dir}...")
    
    # Patterns of files to delete
    patterns = [
        "ml_model_*.pkl",
        "ml_metadata_*.json",
        "scaler_*.pkl",
        "label_encoder_*.pkl",
        "backtest_results.json",
        "accuracy_history.json",
        "predictions.json",
        "learning_tracker.json",
        "model_checksums.json",
        "model_checksums.json.bak",
        "trained_weights.json",
        # Also clean up sector specific models if they follow a pattern or are in a subdir?
        # The sector models seem to be managed by SectorMLManager, possibly in a subdir or same dir.
        # Based on file list earlier, there is a 'ml_configs' dir and 'model_versions' dir.
    ]
    
    # Delete matchign files
    deleted_count = 0
    for pattern in patterns:
        for file_path in data_dir.glob(pattern):
            try:
                os.remove(file_path)
                logger.info(f"Deleted: {file_path.name}")
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete {file_path.name}: {e}")

    # Delete directories if needed? 
    # For now, let's stick to the main artifacts that affect the "Stalemate" and "Phantom" issues.
    
    if deleted_count == 0:
        logger.info("Directory was already clean.")
    else:
        logger.info(f"✨ Cleanup Complete. Removed {deleted_count} files.")
        logger.info("You can now Start Training with a clean slate.")

if __name__ == "__main__":
    perform_cleanup()
