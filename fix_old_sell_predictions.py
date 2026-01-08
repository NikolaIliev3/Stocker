"""
Script to fix old SELL predictions that were created with incorrect logic.

With the inverted logic fix:
- SELL = bullish (expecting price to go UP)
- Target should be ABOVE entry (not below)
- Stop loss should be BELOW entry (not above)

This script will:
1. Find all active SELL predictions with targets below entry (incorrect)
2. Fix them by recalculating target above entry and stop loss below entry
3. Mark them as fixed for reference
"""
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

def fix_old_sell_predictions():
    """Fix old SELL predictions with incorrect target/stop loss logic"""
    
    # Find data directory
    data_dir = Path("data")
    if not data_dir.exists():
        print("❌ Data directory not found!")
        return
    
    predictions_file = data_dir / "predictions.json"
    if not predictions_file.exists():
        print("✅ No predictions file found - nothing to fix!")
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
        print("✅ No predictions found - nothing to fix!")
        return
    
    # Find SELL predictions with incorrect logic
    fixed_count = 0
    deleted_count = 0
    issues_found = []
    
    for pred in predictions:
        if pred.get('action') == 'SELL' and pred.get('status') == 'active':
            entry_price = pred.get('entry_price', 0)
            target_price = pred.get('target_price', 0)
            stop_loss = pred.get('stop_loss', 0)
            strategy = pred.get('strategy', 'trading')
            
            # Check if target is below entry (incorrect for SELL = bullish)
            if entry_price > 0 and target_price > 0 and target_price < entry_price:
                issues_found.append({
                    'id': pred.get('id'),
                    'symbol': pred.get('symbol'),
                    'entry': entry_price,
                    'old_target': target_price,
                    'old_stop': stop_loss,
                    'strategy': strategy
                })
    
    if not issues_found:
        print("✅ No incorrect SELL predictions found! All predictions are correct.")
        return
    
    print(f"\n🔍 Found {len(issues_found)} SELL prediction(s) with incorrect logic:\n")
    for issue in issues_found:
        print(f"  • Prediction #{issue['id']} ({issue['symbol']}):")
        print(f"    Entry: ${issue['entry']:.2f}")
        print(f"    Old Target: ${issue['old_target']:.2f} (❌ below entry)")
        print(f"    Old Stop: ${issue['old_stop']:.2f}")
        print()
    
    # Ask user what to do
    print("Options:")
    print("  1. Fix them (recalculate target above entry, stop loss below entry)")
    print("  2. Delete them (remove incorrect predictions)")
    print("  3. Skip (leave them as-is)")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == '1':
        # Fix predictions
        for issue in issues_found:
            pred_id = issue['id']
            pred = next((p for p in predictions if p.get('id') == pred_id), None)
            if not pred:
                continue
            
            entry_price = pred['entry_price']
            strategy = pred.get('strategy', 'trading')
            
            # Recalculate target ABOVE entry (bullish)
            if strategy == "trading":
                new_target = entry_price * 1.03  # 3% above
                new_stop = entry_price * 0.97    # 3% below
            elif strategy == "mixed":
                new_target = entry_price * 1.06  # 6% above
                new_stop = entry_price * 0.95    # 5% below
            else:  # investing
                new_target = entry_price * 1.12  # 12% above
                new_stop = entry_price * 0.80    # 20% below
            
            pred['target_price'] = new_target
            pred['stop_loss'] = new_stop
            pred['reasoning'] = f"[FIXED] {pred.get('reasoning', '')} - Target/stop loss corrected for inverted logic."
            pred['fixed_at'] = datetime.now().isoformat()
            
            fixed_count += 1
            print(f"  ✅ Fixed prediction #{pred_id} ({pred['symbol']}): Target ${new_target:.2f}, Stop ${new_stop:.2f}")
        
        # Save fixed predictions
        data['predictions'] = predictions
        data['last_updated'] = datetime.now().isoformat()
        data['fixed_old_sell_predictions'] = {
            'date': datetime.now().isoformat(),
            'count': fixed_count
        }
        
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✅ Fixed {fixed_count} prediction(s)!")
        
    elif choice == '2':
        # Delete incorrect predictions
        predictions_to_keep = [
            p for p in predictions 
            if not (p.get('action') == 'SELL' and p.get('status') == 'active' 
                   and p.get('entry_price', 0) > 0 and p.get('target_price', 0) > 0
                   and p.get('target_price', 0) < p.get('entry_price', 0))
        ]
        
        deleted_count = len(predictions) - len(predictions_to_keep)
        
        # Reassign IDs
        for i, pred in enumerate(predictions_to_keep):
            pred['id'] = i + 1
        
        data['predictions'] = predictions_to_keep
        data['last_updated'] = datetime.now().isoformat()
        data['deleted_old_sell_predictions'] = {
            'date': datetime.now().isoformat(),
            'count': deleted_count
        }
        
        with open(predictions_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✅ Deleted {deleted_count} incorrect prediction(s)!")
        
    else:
        print("\n⏭️  Skipped - predictions left unchanged.")
        print("⚠️  Note: These predictions may cause incorrect trade calculations!")

if __name__ == "__main__":
    print("=" * 60)
    print("Fix Old SELL Predictions Script")
    print("=" * 60)
    print("\nThis script fixes SELL predictions created with the old incorrect logic.")
    print("(SELL targets should be ABOVE entry, not below)\n")
    
    fix_old_sell_predictions()
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
