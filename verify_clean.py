
import json
from pathlib import Path
import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def verify_clean():
    home_dir = Path.home() / ".stocker"
    perf_file = home_dir / "performance_history.json"
    
    print(f"Checking {perf_file}...")
    
    if not perf_file.exists():
        print("File does not exist. (GOOD)")
        return
        
    try:
        with open(perf_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        is_clean = True
        
        for strategy, methods in data.items():
            for method, history in methods.items():
                if isinstance(history, list) and len(history) > 0:
                    print(f"❌ FOUND DIRTY DATA: {strategy} -> {method} has {len(history)} entries")
                    is_clean = False
                elif isinstance(history, list):
                    print(f"✓ {strategy} -> {method} is empty")
                    
        if is_clean:
            print("\n✅ FILE IS 100% CLEAN!")
        else:
            print("\n❌ FILE IS NOT CLEAN (See above)")
            # Attempt to clean right here
            print("Attempting emergency clean...")
            for strategy in data:
                for method in data[strategy]:
                    data[strategy][method] = []
            with open(perf_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            print("Emergency clean done.")
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    verify_clean()
