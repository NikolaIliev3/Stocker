"""
Complete Reset Script for Stocker ML System
Clears all predictions, learning data, and ML models for a fresh start.
"""
import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def find_data_files():
    """Find all prediction and ML model files"""
    # FIXED: Search in ~/.stocker where data actually lives, not current directory
    from config import APP_DATA_DIR
    base_dir = APP_DATA_DIR
    
    files_found = {
        'predictions': [],
        'learning': [],
        'ml_models': [],
        'config': []
    }
    
    # Patterns to search for
    prediction_patterns = [
        "predictions.json",
        "trend_change_predictions.json", 
        "verified_predictions*.json",
        "buy_opportunity_predictions.json",
        "peak_bottom_detection_history.json"
    ]
    
    learning_patterns = [
        "learning_data*.json",
        "learning_tracker*.json",
        "price_move_learning.json",
        "calibration_data*.json",
        "verified_outcomes*.json",
        "performance_history.json",      # Hybrid accuracy stats
        "performance_attribution.json",  # Detailed attribution stats
        "production_monitoring*.json",
        "ab_tests*.json",
        "model_benchmarks*.json"
    ]
    
    ml_model_patterns = [
        "ml_model*.pkl",
        "scaler*.pkl", 
        "label_encoder*.pkl",
        "selected_features*.json",
        "model_checksums.json"
    ]
    
    print(f"Searching in: {base_dir}")
    
    # Search in the data directory
    if base_dir.exists():
        for pattern in prediction_patterns:
            for file_path in base_dir.rglob(pattern):
                files_found['predictions'].append(file_path)
        
        for pattern in learning_patterns:
            for file_path in base_dir.rglob(pattern):
                files_found['learning'].append(file_path)
        
        for pattern in ml_model_patterns:
            for file_path in base_dir.rglob(pattern):
                files_found['ml_models'].append(file_path)
    else:
        print(f"Data directory does not exist: {base_dir}")
    
    # Also check current directory for any stray files
    current_dir = Path(".")
    for pattern in prediction_patterns + learning_patterns + ml_model_patterns:
        for file_path in current_dir.rglob(pattern):
            # Don't duplicate if already found
            if file_path not in files_found['predictions'] and file_path not in files_found['learning'] and file_path not in files_found['ml_models']:
                if 'prediction' in pattern or pattern.startswith('verified'):
                    files_found['predictions'].append(file_path)
                elif '.pkl' in pattern or 'checksums' in pattern or 'features' in pattern:
                    files_found['ml_models'].append(file_path)
                else:
                    files_found['learning'].append(file_path)
    
    return files_found

def clear_recursive(data):
    """Recursively clear all lists in a dictionary"""
    cleared = False
    if isinstance(data, dict):
        for key in data:
            if isinstance(data[key], list):
                data[key] = []
                cleared = True
            elif isinstance(data[key], dict):
                if clear_recursive(data[key]):
                    cleared = True
    return cleared

def clear_json_file(filepath: Path):
    """Clear a JSON file by setting its list to empty"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Use recursive clear
        if clear_recursive(data):
            data['cleared_at'] = datetime.now().isoformat()
            data['cleared_reason'] = 'Fresh start - reset to fix accuracy issues'
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            return True
        else:
            # If no lists found, maybe it's just empty or different format
            # For some files we might want to just empty/reset them
            return False
            
    except Exception as e:
        print(f"  Error clearing {filepath}: {e}")
    return False

def delete_file(filepath: Path):
    """Delete a file"""
    try:
        filepath.unlink()
        return True
    except Exception as e:
        print(f"  Error deleting {filepath}: {e}")
        return False

def main():
    print("=" * 60)
    print("🔄 STOCKER COMPLETE RESET - Fresh Start")
    print("=" * 60)
    print()
    print("This will clear:")
    print("  • All predictions (regular, trend change, buy opportunities)")
    print("  • All learning data")
    print("  • All ML models (will be retrained from scratch)")
    print()
    
    # Find all files
    files = find_data_files()
    
    total_files = sum(len(v) for v in files.values())
    print(f"Found {total_files} files to process:")
    print(f"  • Predictions: {len(files['predictions'])}")
    print(f"  • Learning Data: {len(files['learning'])}")
    print(f"  • ML Models: {len(files['ml_models'])}")
    print()
    
    if total_files == 0:
        print("✅ No files found - system is already clean!")
        return
    
    # Process predictions (clear JSON lists)
    print("📋 Clearing predictions...")
    for filepath in files['predictions']:
        if clear_json_file(filepath):
            print(f"  ✅ Cleared: {filepath.name}")
        else:
            print(f"  ⚠️  Could not clear: {filepath.name}")
    
    # Process learning data (clear JSON lists)
    print()
    print("🧠 Clearing learning data...")
    for filepath in files['learning']:
        if clear_json_file(filepath):
            print(f"  ✅ Cleared: {filepath.name}")
        else:
            print(f"  ⚠️  Could not clear: {filepath.name}")
    
    # Process ML models (delete files to force retrain)
    print()
    print("🤖 Deleting ML models (will be retrained)...")
    for filepath in files['ml_models']:
        if delete_file(filepath):
            print(f"  ✅ Deleted: {filepath.name}")
        else:
            print(f"  ⚠️  Could not delete: {filepath.name}")
    
    print()
    print("=" * 60)
    print("✅ RESET COMPLETE!")
    print()
    print("Next steps:")
    print("  1. Restart the Stocker application")
    print("  2. The ML model will retrain automatically")
    print("  3. Make new predictions using the Gun tab")
    print("  4. Accuracy will improve as new (correctly labeled) data is collected")
    print("=" * 60)

if __name__ == "__main__":
    main()
