"""
Script to clear hybrid accuracy scores from performance history
This resets the hybrid performance tracking to start fresh
"""
import json
import sys
from pathlib import Path
from config import APP_DATA_DIR

# Fix Windows console encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except:
        pass

def clear_hybrid_accuracy():
    """Clear hybrid accuracy scores from performance history"""
    performance_file = APP_DATA_DIR / "performance_history.json"
    
    if not performance_file.exists():
        print("❌ Performance history file not found.")
        print(f"   Expected location: {performance_file}")
        return False
    
    try:
        # Load current performance data
        with open(performance_file, 'r') as f:
            all_performance = json.load(f)
        
        print(f"📁 Found performance history file: {performance_file}")
        print(f"📊 Strategies found: {list(all_performance.keys())}")
        
        # Clear hybrid performance for all strategies
        cleared_count = 0
        for strategy in all_performance.keys():
            if 'hybrid' in all_performance[strategy]:
                old_count = len(all_performance[strategy]['hybrid'])
                all_performance[strategy]['hybrid'] = []
                cleared_count += old_count
                print(f"   ✓ Cleared {old_count} hybrid entries for '{strategy}' strategy")
        
        # Save updated performance data
        with open(performance_file, 'w') as f:
            json.dump(all_performance, f, indent=2)
        
        if cleared_count > 0:
            print(f"\n✅ Successfully cleared {cleared_count} hybrid accuracy entries!")
            print("   The hybrid accuracy score has been reset to 0/0")
            return True
        else:
            print("\n⚠️  No hybrid accuracy entries found to clear.")
            return False
            
    except Exception as e:
        print(f"❌ Error clearing hybrid accuracy: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🧹 Clearing Hybrid Accuracy Scores")
    print("=" * 60)
    print()
    
    success = clear_hybrid_accuracy()
    
    print()
    print("=" * 60)
    if success:
        print("✅ Done! Hybrid accuracy has been reset.")
        print("   Restart the app to see the changes.")
    else:
        print("⚠️  No changes made.")
    print("=" * 60)
