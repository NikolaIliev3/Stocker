"""
Script to automatically clear ML model and backtesting data for a fresh start
Preserves user preferences, portfolio, and predictions history
"""
import os
import shutil
from pathlib import Path
from config import APP_DATA_DIR

def clear_ml_data():
    """Clear ML model files and backtesting data"""
    data_dir = APP_DATA_DIR
    
    if not data_dir.exists():
        print(f"Data directory {data_dir} does not exist. Nothing to clear.")
        return
    
    print(f"Clearing ML data from: {data_dir}")
    print()
    
    # Files to remove (ML models, scalers, encoders, metadata)
    files_to_remove = []
    for strategy in ['trading', 'mixed', 'investing']:
        files_to_remove.extend([
            f"ml_model_{strategy}.pkl",
            f"scaler_{strategy}.pkl",
            f"label_encoder_{strategy}.pkl",
            f"ml_metadata_{strategy}.json",
            # Regime-specific models
            f"ml_model_{strategy}_bull.pkl",
            f"scaler_{strategy}_bull.pkl",
            f"label_encoder_{strategy}_bull.pkl",
            f"ml_metadata_{strategy}_bull.json",
            f"ml_model_{strategy}_bear.pkl",
            f"scaler_{strategy}_bear.pkl",
            f"label_encoder_{strategy}_bear.pkl",
            f"ml_metadata_{strategy}_bear.json",
            f"ml_model_{strategy}_sideways.pkl",
            f"scaler_{strategy}_sideways.pkl",
            f"label_encoder_{strategy}_sideways.pkl",
            f"ml_metadata_{strategy}_sideways.json",
        ])
    
    # Also clear backtesting results and tested dates tracking
    files_to_remove.extend([
        "backtest_results.json",
        "continuous_backtest_results.json",
        "tested_dates.json",  # Clear tested dates tracking for fresh start
        "model_benchmarks.json",
        "model_benchmarks_backup.json",
        "production_monitoring_trading.json",
        "production_monitoring_mixed.json",
        "production_monitoring_investing.json",
        "performance_attribution.json",
        "ab_tests_trading.json",
        "ab_tests_mixed.json",
        "ab_tests_investing.json",
    ])
    
    # Directories to remove
    dirs_to_remove = [
        "model_versions",
        "ab_tests",
    ]
    
    removed_count = 0
    for filename in files_to_remove:
        filepath = data_dir / filename
        if filepath.exists():
            try:
                filepath.unlink()
                print(f"  Removed: {filename}")
                removed_count += 1
            except Exception as e:
                print(f"  Error removing {filename}: {e}")
    
    for dirname in dirs_to_remove:
        dirpath = data_dir / dirname
        if dirpath.exists() and dirpath.is_dir():
            try:
                shutil.rmtree(dirpath)
                print(f"  Removed directory: {dirname}/")
                removed_count += 1
            except Exception as e:
                print(f"  Error removing {dirname}: {e}")
    
    if removed_count == 0:
        print("  No ML model or backtesting files found to remove.")
    else:
        print(f"\n[SUCCESS] Cleared {removed_count} ML model/backtesting file(s)")
        print("  Preserved: preferences.json, portfolio.json, predictions.json")
    
    print("\n[SUCCESS] Fresh start ready! You can now train new models with the fixed feature extraction.")

if __name__ == "__main__":
    print("=" * 60)
    print("ML Data Cleanup for Fresh Start")
    print("=" * 60)
    print("\nThis will remove:")
    print("  - All ML model files (.pkl)")
    print("  - All scalers and label encoders")
    print("  - All ML metadata files")
    print("  - All backtesting results")
    print("  - Tested dates tracking (tested_dates.json)")
    print("  - Model version history")
    print("  - Production monitoring data")
    print("\nThis will PRESERVE:")
    print("  - User preferences")
    print("  - Portfolio data")
    print("  - Prediction history")
    print("\n" + "=" * 60)
    print("\nProceeding automatically...\n")
    
    clear_ml_data()
