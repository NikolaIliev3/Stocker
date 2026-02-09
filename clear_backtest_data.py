"""
Clear Old Backtest Data
Resets the backtest history to start fresh with Buy-Only system.
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import APP_DATA_DIR

def clear_backtest_data():
    results_file = Path(APP_DATA_DIR) / 'backtest_results.json'
    
    if results_file.exists():
        # Backup first
        backup_file = Path(APP_DATA_DIR) / 'backtest_results_backup.json'
        import shutil
        shutil.copy(results_file, backup_file)
        print(f"✅ Backed up old data to: {backup_file}")
    
    # Clear with empty structure
    empty_data = {
        'test_details': [],
        'accuracy_history': [],
        'strategy_results': {}
    }
    
    with open(results_file, 'w') as f:
        json.dump(empty_data, f, indent=2)
    
    print(f"✅ Cleared backtest results at: {results_file}")
    print("🚀 Ready for fresh 500-sample backtest!")

if __name__ == "__main__":
    clear_backtest_data()
