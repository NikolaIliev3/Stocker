"""
Clear Backtest Results
Removes old backtest results while preserving all other data
"""
import json
from pathlib import Path
from config import APP_DATA_DIR

def clear_backtest_results():
    print("="*60)
    print("CLEARING BACKTEST RESULTS")
    print("="*60)
    
    backtest_file = APP_DATA_DIR / "backtest_results.json"
    
    if backtest_file.exists():
        # Load to show what we're clearing
        try:
            with open(backtest_file, 'r') as f:
                data = json.load(f)
            
            total_tests = sum(r.get('total', 0) for r in data.get('strategy_results', {}).values())
            test_details_count = len(data.get('test_details', []))
            
            print(f"\nBacktest file found: {backtest_file}")
            print(f"  Total tests in history: {total_tests}")
            print(f"  Test details entries: {test_details_count}")
            
            # Show strategy breakdown
            if 'strategy_results' in data:
                print("\nStrategy results:")
                for strategy, results in data['strategy_results'].items():
                    correct = results.get('correct', 0)
                    total = results.get('total', 0)
                    accuracy = (correct / total * 100) if total > 0 else 0
                    print(f"  • {strategy}: {accuracy:.1f}% ({correct}/{total})")
            
            # Delete the file
            backtest_file.unlink()
            print(f"\n[OK] Deleted: {backtest_file.name}")
            print("\nNext backtest run will start fresh!")
            
        except Exception as e:
            print(f"[ERROR] Error reading/deleting file: {e}")
    else:
        print(f"\nNo backtest results file found at {backtest_file}")
        print("Nothing to clear.")
    
    print("\n" + "="*60)
    print("All other data preserved:")
    print("  - predictions.json")
    print("  - portfolio.json")
    print("  - ML models (*.pkl)")
    print("  - user preferences")
    print("  - learning tracker")
    print("="*60)

if __name__ == '__main__':
    clear_backtest_results()
