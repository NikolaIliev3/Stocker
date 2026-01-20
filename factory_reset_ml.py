"""
Factory Reset ML System

This script clears all ML models, backtest results, and related data to allow
a fresh start with corrected training logic.

Run this when:
- Feature count mismatch errors persist
- Model accuracy is stuck at ~50% (random chance)
- SELL predictions are failing at high rates
- Training is being rejected due to "performance degradation"
"""

import os
import json
from pathlib import Path
from datetime import datetime
import shutil

def factory_reset_ml(data_dir: Path = None, confirm: bool = False):
    """
    Perform a complete factory reset of the ML system.
    
    WARNING: This will delete all ML models, backtest history, and learned data!
    
    Args:
        data_dir: Path to data directory. If None, uses default location.
        confirm: If True, skip confirmation prompt.
    """
    
    if data_dir is None:
        # Default data directory
        script_dir = Path(__file__).parent
        data_dir = script_dir / "data"
    
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    print("=" * 60)
    print("  ML SYSTEM FACTORY RESET")
    print("=" * 60)
    print()
    print("This will DELETE the following:")
    print("  • All ML model files (ml_model_*.pkl)")
    print("  • All scaler files (scaler_*.pkl)")
    print("  • All label encoder files (label_encoder_*.pkl)")
    print("  • All ML metadata files (ml_metadata_*.json)")
    print("  • Backtest results (backtest_results.json)")
    print("  • Tested dates tracking (tested_dates_*.json)")
    print("  • Model version history (model_versions_*.json)")
    print("  • SPY cache data (spy_cache_*.json)")
    print()
    print("This will PRESERVE:")
    print("  • Saved ML configurations (ml_configs/)")
    print()
    
    # Count files to be deleted
    patterns_to_delete = [
        "ml_model_*.pkl",
        "scaler_*.pkl", 
        "label_encoder_*.pkl",
        "ml_metadata_*.json",
        "backtest_results.json",
        "tested_dates_*.json",
        "model_versions_*.json",
        "spy_cache_*.json",
        "tmp_*.pkl",  # Temporary training files
    ]
    
    files_to_delete = []
    for pattern in patterns_to_delete:
        if '*' in pattern:
            files_to_delete.extend(data_dir.glob(pattern))
        else:
            file_path = data_dir / pattern
            if file_path.exists():
                files_to_delete.append(file_path)
    
    # Also check for model_versions subdirectory
    versions_dir = data_dir / "model_versions"
    if versions_dir.exists():
        print(f"  • Model versions backup directory ({versions_dir})")
    
    print(f"\nTotal files to delete: {len(files_to_delete)}")
    
    if len(files_to_delete) > 0:
        print("\nFiles:")
        for f in sorted(files_to_delete)[:20]:  # Show first 20
            print(f"  - {f.name}")
        if len(files_to_delete) > 20:
            print(f"  ... and {len(files_to_delete) - 20} more")
    
    print()
    
    if not confirm:
        response = input("[WARNING] Are you SURE you want to delete all this data? (type 'yes' to confirm): ")
        if response.lower() != 'yes':
            print("[X] Cancelled. No files were deleted.")
            return False
    
    # Perform deletion
    print()
    print("[*] Deleting files...")

    
    deleted_count = 0
    error_count = 0
    
    for file_path in files_to_delete:
        try:
            file_path.unlink()
            print(f"  ✓ Deleted: {file_path.name}")
            deleted_count += 1
        except Exception as e:
            print(f"  ✗ Error deleting {file_path.name}: {e}")
            error_count += 1
    
    # Delete model versions directory
    if versions_dir.exists():
        try:
            shutil.rmtree(versions_dir)
            print(f"  ✓ Deleted directory: model_versions/")
            deleted_count += 1
        except Exception as e:
            print(f"  ✗ Error deleting model_versions/: {e}")
            error_count += 1
    
    print()
    print("=" * 60)
    print(f"  Factory reset complete!")
    print(f"  • Deleted: {deleted_count} items")
    print(f"  • Errors: {error_count}")
    print("=" * 60)
    print()
    print("📋 NEXT STEPS:")
    print("  1. Restart the Stocker application")
    print("  2. Go to 'Machine Learning' tab")
    print("  3. Click 'Train Megamind' to train fresh models")
    print("  4. Run 'Run Backtest Now' to verify accuracy")
    print()
    print("  Expected result: 55-65% accuracy without feature mismatch errors")
    print()
    
    return True


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Factory reset the ML system")
    parser.add_argument("--confirm", "-y", action="store_true", 
                       help="Skip confirmation prompt")
    parser.add_argument("--data-dir", type=str, default=None,
                       help="Path to data directory")
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir) if args.data_dir else None
    factory_reset_ml(data_dir, confirm=args.confirm)
