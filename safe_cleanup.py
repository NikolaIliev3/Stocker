import os
import sys
import time
from pathlib import Path

# Target file
data_dir = Path.home() / ".stocker"
target_file = data_dir / "backtest_results.json"

print(f"Target: {target_file}")

if not target_file.exists():
    print("✅ File already deleted or does not exist.")
    sys.exit(0)

print("Attempting to delete...")

try:
    # Try standard remove
    os.remove(target_file)
    print("✅ SUCCESS: backtest_results.json deleted.")
    print("You can now restart the app with a clean slate.")
except PermissionError:
    print("\n❌ DATA LOCKED BY APP")
    print("The Stocker App is currently running and holding this file open.")
    print("-" * 40)
    print("👉 ACTION REQUIRED: Close the Stocker App window completely.")
    print("-" * 40)
    print("Run this script again after closing the app.")
except Exception as e:
    print(f"❌ Error: {e}")
