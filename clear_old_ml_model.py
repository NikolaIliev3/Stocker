"""
Clear Old ML Model
Deletes the old ML model that has low accuracy (50%)
"""
import json
from pathlib import Path
from config import APP_DATA_DIR

def clear_old_ml_model():
    print("="*60)
    print("CLEARING OLD ML MODEL")
    print("="*60)
    
    data_dir = APP_DATA_DIR
    strategy = 'trading'  # The one with 50% accuracy
    
    files_to_delete = [
        f"ml_model_{strategy}.pkl",
        f"scaler_{strategy}.pkl",
        f"ml_metadata_{strategy}.json",
        f"label_encoder_{strategy}.pkl",
        f"checksums_{strategy}.json",
    ]
    
    deleted = []
    not_found = []
    
    for filename in files_to_delete:
        filepath = data_dir / filename
        if filepath.exists():
            try:
                filepath.unlink()
                deleted.append(filename)
                print(f"[DELETED] {filename}")
            except Exception as e:
                print(f"[ERROR] Could not delete {filename}: {e}")
        else:
            not_found.append(filename)
    
    if not_found:
        print(f"\n[NOT FOUND] {len(not_found)} files already missing:")
        for f in not_found:
            print(f"  - {f}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Deleted: {len(deleted)} files")
    print(f"Not found: {len(not_found)} files")
    
    if deleted:
        print("\nNext steps:")
        print("  1. Open the app")
        print("  2. Go to ML Training")
        print("  3. Train a new model with:")
        print("     - Binary classification enabled (should be default)")
        print("     - 20+ diverse stock symbols")
        print("     - Date range: 2010-2023")
        print("     - This should give you 60-75% accuracy")
    else:
        print("\nNo files to delete - model already cleared or doesn't exist")
    
    print("="*60)

if __name__ == '__main__':
    clear_old_ml_model()
