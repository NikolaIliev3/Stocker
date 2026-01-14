"""
Quick script to save current ML training configuration
Usage: python save_ml_config.py [name] [description]
"""
import sys
from pathlib import Path
from config import APP_DATA_DIR
from ml_config_manager import MLConfigManager

def main():
    config_manager = MLConfigManager(APP_DATA_DIR)
    
    # Get config name and description from command line or prompt
    if len(sys.argv) > 1:
        name = sys.argv[1]
        description = sys.argv[2] if len(sys.argv) > 2 else ""
    else:
        from datetime import datetime
        name = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        description = "Current good configuration"
        print(f"No name provided, using: {name}")
    
    # Save current configuration
    if config_manager.save_config(name, description):
        print(f"\n[SUCCESS] Saved ML configuration: {name}")
        if description:
            print(f"   Description: {description}")
        print(f"\nTo load this configuration:")
        print(f"  1. Open the app")
        print(f"  2. Click 'Train AI Model'")
        print(f"  3. Click 'Load Config'")
        print(f"  4. Select '{name}'")
    else:
        print(f"\n[ERROR] Failed to save configuration")
        sys.exit(1)

if __name__ == "__main__":
    main()
