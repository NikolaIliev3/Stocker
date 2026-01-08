"""
Script to completely remove all predictions and trend change predictions for a fresh start.
This will delete the files or clear their contents.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

def clear_all_predictions():
    """Delete all predictions and trend change predictions"""
    
    print("=" * 60)
    print("Clear All Predictions - Fresh Start")
    print("=" * 60)
    print()
    
    # Find data directory (check common locations)
    possible_data_dirs = [
        Path("data"),
        Path(".") / "data",
        Path.home() / ".stocker" / "data",
    ]
    
    data_dir = None
    for possible_dir in possible_data_dirs:
        if possible_dir.exists():
            data_dir = possible_dir
            break
    
    # If no data dir exists, check if we need to create it or if files are elsewhere
    if not data_dir:
        # Try to find prediction files in current directory or subdirectories
        current_dir = Path(".")
        predictions_file = None
        trend_predictions_file = None
        
        # Search for prediction files
        for file_path in current_dir.rglob("predictions.json"):
            if "data" in str(file_path.parent):
                data_dir = file_path.parent
                break
        
        if not data_dir:
            for file_path in current_dir.rglob("trend_change_predictions.json"):
                if "data" in str(file_path.parent):
                    data_dir = file_path.parent
                    break
        
        # If still not found, use default
        if not data_dir:
            data_dir = Path("data")
    
    print(f"Using data directory: {data_dir}")
    print()
    
    predictions_file = data_dir / "predictions.json"
    trend_predictions_file = data_dir / "trend_change_predictions.json"
    
    deleted_count = 0
    cleared_count = 0
    
    # Handle regular predictions
    if predictions_file.exists():
        try:
            with open(predictions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                predictions = data.get('predictions', [])
                count = len(predictions)
            
            if count > 0:
                # Clear predictions
                data['predictions'] = []
                data['last_updated'] = datetime.now().isoformat()
                data['cleared_at'] = datetime.now().isoformat()
                
                with open(predictions_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                
                cleared_count += count
                print(f"✅ Cleared {count} regular prediction(s)")
            else:
                print("ℹ️  No regular predictions found")
        except Exception as e:
            print(f"⚠️  Error processing regular predictions: {e}")
            # Try to delete the file instead
            try:
                predictions_file.unlink()
                print(f"✅ Deleted predictions.json file")
                deleted_count += 1
            except Exception as e2:
                print(f"❌ Could not delete predictions.json: {e2}")
    else:
        print("ℹ️  predictions.json not found")
    
    # Handle trend change predictions
    if trend_predictions_file.exists():
        try:
            with open(trend_predictions_file, 'r', encoding='utf-8') as f:
                trend_data = json.load(f)
                trend_predictions = trend_data.get('predictions', [])
                count = len(trend_predictions)
            
            if count > 0:
                # Clear trend predictions
                trend_data['predictions'] = []
                trend_data['last_updated'] = datetime.now().isoformat()
                trend_data['cleared_at'] = datetime.now().isoformat()
                
                with open(trend_predictions_file, 'w', encoding='utf-8') as f:
                    json.dump(trend_data, f, indent=2)
                
                cleared_count += count
                print(f"✅ Cleared {count} trend change prediction(s)")
            else:
                print("ℹ️  No trend change predictions found")
        except Exception as e:
            print(f"⚠️  Error processing trend change predictions: {e}")
            # Try to delete the file instead
            try:
                trend_predictions_file.unlink()
                print(f"✅ Deleted trend_change_predictions.json file")
                deleted_count += 1
            except Exception as e2:
                print(f"❌ Could not delete trend_change_predictions.json: {e2}")
    else:
        print("ℹ️  trend_change_predictions.json not found")
    
    print()
    print("=" * 60)
    if cleared_count > 0 or deleted_count > 0:
        print(f"✅ Successfully cleared/deleted predictions!")
        print(f"   • Cleared: {cleared_count} prediction(s)")
        print(f"   • Deleted: {deleted_count} file(s)")
    else:
        print("✅ No predictions found - you're already starting fresh!")
    print()
    print("💡 You can now make fresh predictions using the Gun option!")
    print("   All new predictions will use the corrected logic.")
    print("=" * 60)

if __name__ == "__main__":
    clear_all_predictions()
