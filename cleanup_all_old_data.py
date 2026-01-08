"""
Comprehensive cleanup script - deletes ALL old ML-related data files
Run this to ensure a completely fresh start
"""
import sys
import os
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

from config import APP_DATA_DIR

def cleanup_all_old_data():
    """Delete all old ML models, predictions, and learning data"""
    data_dir = Path(APP_DATA_DIR)
    
    if not data_dir.exists():
        print(f"Data directory doesn't exist: {data_dir}")
        return
    
    # All files that could contain old faulty data
    files_to_delete = [
        # ML Models (already deleted, but check again)
        "ml_model_trading.pkl",
        "ml_model_investing.pkl", 
        "ml_model_mixed.pkl",
        
        # Metadata
        "ml_metadata_trading.json",
        "ml_metadata_investing.json",
        "ml_metadata_mixed.json",
        
        # Encoders and Scalers
        "label_encoder_trading.pkl",
        "label_encoder_investing.pkl",
        "label_encoder_mixed.pkl",
        "scaler_trading.pkl",
        "scaler_investing.pkl",
        "scaler_mixed.pkl",
        
        # Predictions (contains old contaminated predictions)
        "predictions.json",
        
        # Learning tracker (contains stats from faulty period)
        "learning_tracker.json",
        
        # Weight adjusters (learned from wrong predictions)
        "adaptive_weights_trading.json",
        "adaptive_weights_investing.json",
        "adaptive_weights_mixed.json",
        
        # Trend change predictions (might have wrong data)
        "trend_change_predictions.json",
        
        # Backtesting results (might have wrong results)
        "backtest_results.json",
        "continuous_backtest_results.json",
        "walk_forward_results.json",
        
        # Any cached model files
        "*.pkl",
        "*.pickle",
    ]
    
    deleted = []
    not_found = []
    
    print("=" * 70)
    print("COMPREHENSIVE CLEANUP: Deleting ALL old ML-related data")
    print("=" * 70)
    print(f"Data directory: {data_dir}\n")
    
    # Delete specific files
    for filename in files_to_delete[:len(files_to_delete)-2]:  # Exclude wildcards
        filepath = data_dir / filename
        if filepath.exists():
            try:
                filepath.unlink()
                deleted.append(filename)
                print(f"[OK] Deleted: {filename}")
            except Exception as e:
                print(f"[ERROR] Error deleting {filename}: {e}")
        else:
            not_found.append(filename)
    
    # Delete all .pkl files (model files)
    pkl_files = list(data_dir.glob("*.pkl"))
    for pkl_file in pkl_files:
        try:
            pkl_file.unlink()
            deleted.append(pkl_file.name)
            print(f"[OK] Deleted: {pkl_file.name}")
        except Exception as e:
            print(f"[ERROR] Error deleting {pkl_file.name}: {e}")
    
    # Delete all .pickle files
    pickle_files = list(data_dir.glob("*.pickle"))
    for pickle_file in pickle_files:
        try:
            pickle_file.unlink()
            deleted.append(pickle_file.name)
            print(f"[OK] Deleted: {pickle_file.name}")
        except Exception as e:
            print(f"[ERROR] Error deleting {pickle_file.name}: {e}")
    
    print("\n" + "=" * 70)
    print(f"[OK] Deleted {len(deleted)} files")
    if not_found:
        print(f"[INFO] {len(not_found)} files not found (already deleted or never existed)")
    print("=" * 70)
    
    # Files to KEEP (don't delete these)
    keep_files = [
        "portfolio.json",  # Your portfolio balance/wins/losses
        "preferences.json",  # User preferences
        "alerts.json",  # Alert settings (probably OK)
    ]
    
    print("\nFiles KEPT (not deleted):")
    for keep_file in keep_files:
        keep_path = data_dir / keep_file
        if keep_path.exists():
            print(f"  - {keep_file} (your data)")
    
    print("\n" + "=" * 70)
    print("[OK] Cleanup complete! Ready for completely fresh training.")
    print("\nNext steps:")
    print("1. Train ML models (all old data deleted)")
    print("2. Make new predictions (will be correct)")
    print("3. Verify predictions (will learn correctly)")
    print("=" * 70)

if __name__ == "__main__":
    cleanup_all_old_data()
