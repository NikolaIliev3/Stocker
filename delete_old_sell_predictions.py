"""
Script to delete old SELL predictions that were created with incorrect logic.

With the inverted logic fix:
- SELL = bullish (expecting price to go UP)
- Target should be ABOVE entry (not below)

This script will delete all active SELL predictions with targets below entry.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

def delete_old_sell_predictions():
    """Delete old SELL predictions with incorrect target/stop loss logic"""
    
    # Find data directory
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found!")
        return
    
    predictions_file = data_dir / "predictions.json"
    if not predictions_file.exists():
        print("✅ No predictions file found - nothing to delete!")
        return
    
    # Load predictions
    try:
        with open(predictions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            predictions = data.get('predictions', [])
    except Exception as e:
        print(f"❌ Error loading predictions: {e}")
        return
    
    if not predictions:
        print("✅ No predictions found - nothing to delete!")
        return
    
    # Find SELL predictions with incorrect logic (target below entry)
    to_delete = []
    for pred in predictions:
        if pred.get('action') == 'SELL' and pred.get('status') == 'active':
            entry_price = pred.get('entry_price', 0)
            target_price = pred.get('target_price', 0)
            
            # Check if target is below entry (incorrect for SELL = bullish)
            if entry_price > 0 and target_price > 0 and target_price < entry_price:
                to_delete.append(pred)
    
    if not to_delete:
        print("✅ No incorrect SELL predictions found! All predictions are correct.")
        return
    
    print(f"\n🔍 Found {len(to_delete)} incorrect SELL prediction(s) to delete:\n")
    for pred in to_delete:
        print(f"  • Prediction #{pred.get('id')} ({pred.get('symbol')}):")
        print(f"    Entry: ${pred.get('entry_price', 0):.2f}")
        print(f"    Target: ${pred.get('target_price', 0):.2f} (❌ below entry - incorrect)")
        print()
    
    # Confirm deletion
    response = input(f"Delete these {len(to_delete)} prediction(s)? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("⏭️  Cancelled - no predictions deleted.")
        return
    
    # Keep only predictions that are NOT in the to_delete list
    predictions_to_keep = [p for p in predictions if p not in to_delete]
    
    # Reassign IDs to maintain sequential order
    for i, pred in enumerate(predictions_to_keep):
        pred['id'] = i + 1
    
    # Save updated predictions
    data['predictions'] = predictions_to_keep
    data['last_updated'] = datetime.now().isoformat()
    data['deleted_old_sell_predictions'] = {
        'date': datetime.now().isoformat(),
        'count': len(to_delete)
    }
    
    with open(predictions_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ Deleted {len(to_delete)} incorrect SELL prediction(s)!")
    print(f"✅ {len(predictions_to_keep)} prediction(s) remaining.")
    print("\n💡 You can now make new predictions with the correct logic!")

if __name__ == "__main__":
    print("=" * 60)
    print("Delete Old SELL Predictions Script")
    print("=" * 60)
    print("\nThis script deletes SELL predictions created with the old incorrect logic.")
    print("(SELL targets should be ABOVE entry, not below)\n")
    
    delete_old_sell_predictions()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
