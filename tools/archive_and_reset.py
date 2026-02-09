"""
Archive and Reset Script
Safely moves old data (predictions, models) to a backup folder to prepare for Quant Mode.
"""
import shutil
import os
from pathlib import Path
from datetime import datetime
import json

def archive_and_reset():
    # Setup Paths
    data_dir = Path.home() / ".stocker"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = data_dir / "backups" / f"legacy_logic_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📂 Archiving data to {backup_dir}...")
    
    # Files to Move (Tainted by old logic)
    files_to_move = [
        "predictions.json",
        "learning_tracker.json",
        "ml_metadata_trading.json",
        "ml_model_trading_bear.pkl",
        "ml_model_trading_bull.pkl",
        "ml_model_meta_trading.pkl",
        "scaler.pkl",
        "scaler_trading_bear.pkl",
        "scaler_trading_bull.pkl",
        "label_encoder_trading_bear.pkl",
        "label_encoder_trading_bull.pkl",
        "tested_dates.json" # Reset backtest history
    ]
    
    for filename in files_to_move:
        src = data_dir / filename
        if src.exists():
            dst = backup_dir / filename
            try:
                shutil.move(str(src), str(dst))
                print(f"  ✅ Moved {filename}")
            except Exception as e:
                print(f"  ❌ Failed to move {filename}: {e}")
        else:
            print(f"  ⚠️ {filename} not found (fresh start?)")
            
    # Initialize empty predictions.json to prevent crash
    # Note: LockingJSONStorage handles creation, but explicit is better
    preds_file = data_dir / "predictions.json"
    if not preds_file.exists():
        with open(preds_file, 'w') as f:
            json.dump({}, f)
        print("  ✨ Created empty predictions.json")

    print("\n✅ Archive Complete. System is clean and ready for Quant Backtesting.")

if __name__ == "__main__":
    archive_and_reset()
