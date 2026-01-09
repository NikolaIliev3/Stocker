"""
Clear all trend change predictions
"""
import json
from pathlib import Path

def clear_trend_change_predictions():
    """Clear all trend change predictions"""
    data_dir = Path(".stocker")
    predictions_file = data_dir / "trend_change_predictions.json"
    
    if predictions_file.exists():
        try:
            # Load to see how many predictions exist
            with open(predictions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                count = len(data.get('predictions', []))
            
            # Delete the file
            predictions_file.unlink()
            print(f"Successfully cleared {count} trend change predictions")
            print(f"Deleted file: {predictions_file}")
        except Exception as e:
            print(f"Error clearing trend change predictions: {e}")
    else:
        print("No trend change predictions file found. Nothing to clear.")

if __name__ == "__main__":
    clear_trend_change_predictions()
