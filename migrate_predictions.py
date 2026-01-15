
import json
import shutil
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path setup
APP_DATA_DIR = Path.home() / ".stocker"
PREDICTIONS_FILE = APP_DATA_DIR / "predictions.json"
BACKUP_FILE = APP_DATA_DIR / f"predictions_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

def migrate_predictions():
    if not PREDICTIONS_FILE.exists():
        logger.warning(f"No predictions file found at {PREDICTIONS_FILE}. Nothing to migrate.")
        return

    # Backup logic
    try:
        shutil.copy2(PREDICTIONS_FILE, BACKUP_FILE)
        logger.info(f"Created backup at {BACKUP_FILE}")
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return

    # Load predictions
    try:
        with open(PREDICTIONS_FILE, 'r') as f:
            data = json.load(f)
            predictions = data.get('predictions', [])
    except Exception as e:
        logger.error(f"Failed to load predictions: {e}")
        return

    migrated_count = 0
    skipped_count = 0
    
    logger.info(f"Scanning {len(predictions)} predictions for 'Inverted Logic'...")

    for pred in predictions:
        action = pred.get('action')
        reasoning = pred.get('reasoning', '').lower()
        
        # Logic Detection based on Reasoning vs Action
        # Old Logic (Inverted):
        # - SELL (Buy Action) comes from Bullish Signals ("rising", "bullish", "positive")
        # - BUY (Sell Action) comes from Bearish Signals ("falling", "bearish", "negative")
        
        # New Logic (Standard):
        # - BUY comes from Bullish Signals
        # - SELL comes from Bearish Signals
        
        needs_migration = False
        new_action = action
        
        if action == "SELL":
            # Check for Old Logic signatures in reasoning
            # E.g., "Bullish signals", "Overbought", "Take profit" (implies we held long)
            if any(x in reasoning for x in ['bullish', 'positive', 'rising', 'uptrend', 'strong momentum']):
                # It was a SELL based on Bullish signals -> Inverted Logic!
                # We need to map this to BUY (to capture the Bullish intent for ML)
                needs_migration = True
                new_action = "BUY"
        
        elif action == "BUY":
            # Check for Old Logic signatures
            # E.g., "Bearish signals", "Oversold", "Dip buy"
            if any(x in reasoning for x in ['bearish', 'negative', 'falling', 'downtrend', 'weak momentum']):
                # It was a BUY based on Bearish signals -> Inverted Logic!
                # We need to map this to SELL (to capture the Bearish intent for ML)
                needs_migration = True
                new_action = "SELL"
                
        if needs_migration:
            logger.info(f"Migrating #{pred.get('id')} ({action} -> {new_action}) | Reason: {reasoning[:50]}...")
            pred['action'] = new_action
            # Append note to reasoning
            pred['reasoning'] = pred['reasoning'] + " [MIGRATED: Flipped Label]"
            migrated_count += 1
        else:
            skipped_count += 1

    # Save changes
    if migrated_count > 0:
        try:
            with open(PREDICTIONS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info(f"Successfully migrated {migrated_count} predictions. {skipped_count} skipped.")
        except Exception as e:
            logger.error(f"Failed to save predictions: {e}")
    else:
        logger.info("No predictions needed migration.")

if __name__ == "__main__":
    migrate_predictions()
