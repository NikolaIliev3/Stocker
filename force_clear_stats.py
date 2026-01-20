
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
    
    # FIXED: Use APP_DATA_DIR instead of hardcoded paths
    from config import APP_DATA_DIR
    data_dir = APP_DATA_DIR
    
    print(f"Looking in: {data_dir}")
    
    # Files to delete (relative to data_dir)
    files = [
        "performance_attribution.json",
        "production_monitoring_trading.json",
        "production_monitoring_trading_bear.json",
        "production_monitoring_trading_bull.json",
        "production_monitoring_trading_sideways.json",
        "ab_tests_trading.json",
        "model_benchmarks.json",
        "performance_history.json"
    ]
    
    deleted = 0
    for filename in files:
        path = data_dir / filename
        if path.exists():
            try:
                path.unlink()
                print(f"✅ Deleted {filename}")
                deleted += 1
            except Exception as e:
                print(f"❌ Failed to delete {filename}: {e}")
                
                # Option 2: Clear content if delete fails
                try:
                    with open(path, 'w') as f:
                        json.dump({}, f)
                    print(f"✅ Cleared content of {filename}")
                    deleted += 1
                except Exception as e2:
                    print(f"❌ Failed to clear {filename}: {e2}")
        else:
            print(f"⏭️ {filename} not found (skipped)")
    
    print(f"\n✅ Cleaned {deleted} files")

if __name__ == "__main__":
    clear_performance_stats()
