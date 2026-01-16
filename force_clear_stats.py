
import json
from pathlib import Path
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def clear_performance_stats():
    print("Cleaning up performance stats...")
    
    # Files to delete
    files = [
        "data/performance_attribution.json",
        "data/production_monitoring_trading.json",
        "data/ab_tests_trading.json",
        "data/model_benchmarks.json"
    ]
    
    for relative_path in files:
        path = Path(relative_path)
        if path.exists():
            try:
                # Option 1: Delete file
                path.unlink()
                print(f"Deleted {path}")
            except Exception as e:
                print(f"Failed to delete {path}: {e}")
                
                # Option 2: Clear content if delete fails
                try:
                    with open(path, 'w') as f:
                        json.dump({}, f)
                    print(f"Cleared content of {path}")
                except Exception as e2:
                    print(f"Failed to clear {path}: {e2}")
        else:
            print(f"{path} not found (already clean)")

if __name__ == "__main__":
    clear_performance_stats()
