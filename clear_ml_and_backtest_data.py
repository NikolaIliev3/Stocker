"""
Clear ML Models and Backtest Data
Removes old models trained with buggy code and old backtest results
"""
import json
from pathlib import Path
from config import APP_DATA_DIR
import glob

def clear_ml_models():
    """Clear all ML model files"""
    print("="*60)
    print("CLEARING ML MODELS")
    print("="*60)
    
    data_dir = APP_DATA_DIR
    strategies = ['trading', 'mixed', 'investing']
    regimes = ['bull', 'bear', 'sideways']
    
    deleted = []
    not_found = []
    
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
            else:
                not_found.append(filename)
    
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
    
    print(f"\nDeleted: {len(deleted)} ML model files")
    return len(deleted)

def clear_backtest_results():
    """Clear backtest results"""
    print("\n" + "="*60)
    print("CLEARING BACKTEST RESULTS")
    print("="*60)
    
    data_dir = APP_DATA_DIR
    backtest_file = data_dir / "backtest_results.json"
    
    if backtest_file.exists():
        try:
            # Load to see what we're deleting
            with open(backtest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                entries = len(data.get('results', []))
            
            backtest_file.unlink()
            print(f"[DELETED] backtest_results.json ({entries} entries)")
            return 1
        except Exception as e:
            print(f"[ERROR] Could not delete backtest_results.json: {e}")
            return 0
    else:
        print("[NOT FOUND] backtest_results.json")
        return 0

def main():
    """Main clearing function"""
    print("\n" + "="*60)
    print("CLEARING ML MODELS AND BACKTEST DATA")
    print("="*60)
    print("\nThis will delete:")
    print("  - All ML model files (trading, mixed, investing)")
    print("  - All regime-specific models (bull, bear, sideways)")
    print("  - All scalers and label encoders")
    print("  - All metadata files")
    print("  - Backtest results")
    print("\nThis will NOT delete:")
    print("  - Predictions")
    print("  - Portfolio data")
    print("  - User preferences")
    print("  - Learning tracker data")
    print("\n" + "="*60)
    
    response = input("\nProceed with clearing? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Cancelled.")
        return
    
    ml_deleted = clear_ml_models()
    backtest_deleted = clear_backtest_results()
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"ML Model files deleted: {ml_deleted}")
    print(f"Backtest files deleted: {backtest_deleted}")
    
    if ml_deleted > 0 or backtest_deleted > 0:
        print("\nNext steps:")
        print("  1. Open the app")
        print("  2. Go to ML Training tab")
        print("  3. Train a new model with:")
        print("     - Binary classification enabled")
        print("     - Regime-specific models enabled")
        print("     - 10-20 diverse stock symbols")
        print("     - Date range: 2014-2023 (or longer)")
        print("  4. After training, run backtesting to get fresh results")
        print("\nThe new model will have:")
        print("  - Fixed CV score reporting")
        print("  - Fixed binary classification")
        print("  - Fixed regime model fallback")
        print("  - All recent bug fixes applied")
    else:
        print("\nNo files to delete - data already cleared or doesn't exist")
    
    print("="*60)

if __name__ == '__main__':
    main()
