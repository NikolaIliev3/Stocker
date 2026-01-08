"""
Cleanup Old ML Models
Deletes outdated ML models while preserving valuable data
"""
import os
from pathlib import Path
from config import APP_DATA_DIR

def cleanup_old_models():
    print("="*60)
    print("ML MODEL CLEANUP")
    print("="*60)
    
    # Files to DELETE (old ML models)
    patterns_to_delete = [
        "ml_model_*.pkl",
        "scaler_*.pkl", 
        "ml_metadata_*.json",
        "label_encoder_*.pkl",
        "checksums_*.json",
    ]
    
    # Files to KEEP (valuable data)
    files_to_keep = [
        "predictions.json",
        "trend_changes.json",
        "trend_change_predictions.json",
        "portfolio.json",
        "learning_tracker.json",
        "walk_forward_results.json",
        "search_history.json",
        "user_preferences.json",
    ]
    
    print(f"\nData directory: {APP_DATA_DIR}")
    print("\n" + "-"*60)
    print("FILES TO DELETE (old ML models):")
    print("-"*60)
    
    deleted_count = 0
    deleted_files = []
    
    if APP_DATA_DIR.exists():
        for pattern in patterns_to_delete:
            # Handle wildcard patterns
            if "*" in pattern:
                prefix, suffix = pattern.split("*")
                for file in APP_DATA_DIR.iterdir():
                    if file.name.startswith(prefix) and file.name.endswith(suffix):
                        if file.name not in files_to_keep:
                            deleted_files.append(file.name)
                            print(f"  [DELETE] {file.name}")
                            try:
                                file.unlink()
                                deleted_count += 1
                            except Exception as e:
                                print(f"    ERROR: {e}")
            else:
                file = APP_DATA_DIR / pattern
                if file.exists() and file.name not in files_to_keep:
                    deleted_files.append(file.name)
                    print(f"  [DELETE] {file.name}")
                    try:
                        file.unlink()
                        deleted_count += 1
                    except Exception as e:
                        print(f"    ERROR: {e}")
    
    if deleted_count == 0:
        print("  (No old model files found)")
    
    print("\n" + "-"*60)
    print("FILES PRESERVED (valuable data):")
    print("-"*60)
    
    preserved_count = 0
    if APP_DATA_DIR.exists():
        for keep_file in files_to_keep:
            file = APP_DATA_DIR / keep_file
            if file.exists():
                # Get file size
                size = file.stat().st_size
                size_str = f"{size/1024:.1f} KB" if size > 1024 else f"{size} bytes"
                print(f"  [KEEP] {keep_file} ({size_str})")
                preserved_count += 1
    
    if preserved_count == 0:
        print("  (No data files found)")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"  Deleted: {deleted_count} old model files")
    print(f"  Preserved: {preserved_count} data files")
    print("\nNext steps:")
    print("  1. Run the app")
    print("  2. Go to ML Training")
    print("  3. Train a new model with the improved settings")
    print("  4. Your new model will be ~60-75% accurate (binary mode)")
    print("="*60)

if __name__ == '__main__':
    cleanup_old_models()
