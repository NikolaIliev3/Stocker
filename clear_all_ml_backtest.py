"""
Clear All ML Models and Backtest Data (Non-Interactive)
Deletes old models trained with buggy code
"""
import json
from pathlib import Path
from config import APP_DATA_DIR

def clear_all_ml_and_backtest():
    """Clear all ML models and backtest data"""
    print("="*60)
    print("CLEARING ML MODELS AND BACKTEST DATA")
    print("="*60)
    
    data_dir = APP_DATA_DIR
    strategies = ['trading', 'mixed', 'investing']
    regimes = ['bull', 'bear', 'sideways']
    
    deleted = []
    
    # Clear main strategy models
    for strategy in strategies:
        files_to_delete = [
            f"ml_model_{strategy}.pkl",
            f"scaler_{strategy}.pkl",
            f"ml_metadata_{strategy}.json",
            f"label_encoder_{strategy}.pkl",
        ]
        
        for filename in files_to_delete:
            filepath = data_dir / filename
            if filepath.exists():
                try:
                    filepath.unlink()
                    deleted.append(filename)
                    print(f"[DELETED] {filename}")
                except Exception as e:
                    print(f"[ERROR] Could not delete {filename}: {e}")
    
    # Clear regime-specific models
    for strategy in strategies:
        for regime in regimes:
            files_to_delete = [
                f"ml_model_{strategy}_{regime}.pkl",
                f"scaler_{strategy}_{regime}.pkl",
                f"label_encoder_{strategy}_{regime}.pkl",
            ]
            
            for filename in files_to_delete:
                filepath = data_dir / filename
                if filepath.exists():
                    try:
                        filepath.unlink()
                        deleted.append(filename)
                        print(f"[DELETED] {filename}")
                    except Exception as e:
                        print(f"[ERROR] Could not delete {filename}: {e}")
    
    # Clear checksum files
    checksum_file = data_dir / "model_checksums.json"
    if checksum_file.exists():
        try:
            checksum_file.unlink()
            deleted.append("model_checksums.json")
            print(f"[DELETED] model_checksums.json")
        except Exception as e:
            print(f"[ERROR] Could not delete checksums: {e}")
    
    # Clear backtest results
    backtest_file = data_dir / "backtest_results.json"
    if backtest_file.exists():
        try:
            with open(backtest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                entries = len(data.get('results', []))
            backtest_file.unlink()
            deleted.append("backtest_results.json")
            print(f"[DELETED] backtest_results.json ({entries} entries)")
        except Exception as e:
            print(f"[ERROR] Could not delete backtest_results.json: {e}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total files deleted: {len(deleted)}")
    
    if deleted:
        print("\n[SUCCESS] Successfully cleared ML models and backtest data!")
        print("\nNext steps:")
        print("  1. Train a new ML model (will use all bug fixes)")
        print("  2. Run fresh backtesting")
        print("  3. Make new predictions")
    else:
        print("\nNo files found to delete.")
    
    print("="*60)

if __name__ == '__main__':
    clear_all_ml_and_backtest()
