"""
Script to delete all predictions and trend change predictions - clean slate for fresh predictions.
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
    
    # Find data directory
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found!")
        return
    
    predictions_file = data_dir / "predictions.json"
    trend_predictions_file = data_dir / "trend_change_predictions.json"
    
    # Load regular predictions
    predictions_count = 0
    if predictions_file.exists():
        try:
            with open(predictions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                predictions = data.get('predictions', [])
                predictions_count = len(predictions)
        except Exception as e:
            print(f"❌ Error loading predictions: {e}")
            return
    
    # Load trend change predictions
    trend_predictions_count = 0
    if trend_predictions_file.exists():
        try:
            with open(trend_predictions_file, 'r', encoding='utf-8') as f:
                trend_data = json.load(f)
                trend_predictions = trend_data.get('predictions', [])
                trend_predictions_count = len(trend_predictions)
        except Exception as e:
            print(f"❌ Error loading trend change predictions: {e}")
            return
    
    if predictions_count == 0 and trend_predictions_count == 0:
        print("✅ No predictions found - nothing to delete!")
        return
    
    print(f"\n📊 Found predictions:")
    if predictions_count > 0:
        print(f"   • Regular predictions: {predictions_count}")
        # Show breakdown
        with open(predictions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            predictions = data.get('predictions', [])
            active = sum(1 for p in predictions if p.get('status') == 'active')
            verified = sum(1 for p in predictions if p.get('status') == 'verified')
            expired = sum(1 for p in predictions if p.get('status') == 'expired')
            print(f"     - Active: {active}")
            print(f"     - Verified: {verified}")
            print(f"     - Expired: {expired}")
    
    if trend_predictions_count > 0:
        print(f"   • Trend change predictions: {trend_predictions_count}")
        # Show breakdown
        with open(trend_predictions_file, 'r', encoding='utf-8') as f:
            trend_data = json.load(f)
            trend_predictions = trend_data.get('predictions', [])
            active = sum(1 for p in trend_predictions if p.get('status') == 'active')
            verified = sum(1 for p in trend_predictions if p.get('status') == 'verified')
            expired = sum(1 for p in trend_predictions if p.get('status') == 'expired')
            print(f"     - Active: {active}")
            print(f"     - Verified: {verified}")
            print(f"     - Expired: {expired}")
    
    # Confirm deletion
    print("\n⚠️  This will delete ALL predictions AND trend change predictions!")
    response = input("Are you sure? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("⏭️  Cancelled - no predictions deleted.")
        return
    
    # Clear regular predictions
    if predictions_count > 0:
        data['predictions'] = []
        data['last_updated'] = datetime.now().isoformat()
        data['cleared_at'] = datetime.now().isoformat()
        
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✅ Deleted all {predictions_count} regular prediction(s)!")
    
    # Clear trend change predictions
    if trend_predictions_count > 0:
        trend_data['predictions'] = []
        trend_data['last_updated'] = datetime.now().isoformat()
        trend_data['cleared_at'] = datetime.now().isoformat()
        
        with open(trend_predictions_file, 'w', encoding='utf-8') as f:
            json.dump(trend_data, f, indent=2)
        
        print(f"✅ Deleted all {trend_predictions_count} trend change prediction(s)!")
    
    total_deleted = predictions_count + trend_predictions_count
    if total_deleted > 0:
        print(f"\n✅ Total deleted: {total_deleted} prediction(s)")
        print("💡 You can now make fresh predictions using the Gun option!")

if __name__ == "__main__":
    print("=" * 60)
    print("Clear All Predictions Script")
    print("=" * 60)
    print("\nThis will delete ALL predictions for a fresh start.\n")
    
    clear_all_predictions()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
