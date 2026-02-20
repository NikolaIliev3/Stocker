"""
Reset script: Reverts SNAPPED predictions to active status
and clears tainted performance_attribution.json.
Also removes legacy HOLD predictions.
"""
import json
from pathlib import Path

STOCKER_DIR = Path.home() / ".stocker"
PREDICTIONS_FILE = STOCKER_DIR / "predictions.json"
PERF_ATTR_FILE = STOCKER_DIR / "performance_attribution.json"

def reset():
    # --- 1. Reset SNAPPED predictions ---
    with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        predictions = data.get('predictions', [])
    elif isinstance(data, list):
        predictions = data
    else:
        print(f"Unexpected data type: {type(data)}")
        return
    
    snapped_count = 0
    hold_count = 0
    ids_to_remove = []
    
    for i, p in enumerate(predictions):
        if not isinstance(p, dict):
            continue
        
        # Remove legacy HOLD predictions
        if p.get('action') in ('HOLD', 'AVOID'):
            ids_to_remove.append(i)
            hold_count += 1
            print(f"  🗑️  Removing legacy HOLD prediction #{p['id']} ({p['symbol']})")
            continue
        
        # Reset SNAPPED predictions
        if p.get('verified') and p.get('was_correct'):
            actual = p.get('actual_price_at_target', 0)
            target = p.get('target_price', 0)
            if actual and target and abs(actual - target) < 0.01:
                # This is a SNAPPED prediction — reset it
                p['status'] = 'active'
                p['verified'] = False
                p['was_correct'] = None
                # Remove verification artifacts
                for key in ['actual_price_at_target', 'verification_date', 
                           'actual_price_change', 'days_to_target']:
                    p.pop(key, None)
                snapped_count += 1
                print(f"  🔄 Reset #{p['id']} {p['symbol']} ({p['action']}) to active")
    
    # Remove HOLD predictions (reverse order to preserve indices)
    for i in sorted(ids_to_remove, reverse=True):
        removed = predictions.pop(i)
        print(f"  ✅ Removed #{removed['id']} {removed['symbol']} (legacy HOLD)")
    
    # Save updated predictions
    with open(PREDICTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f"\n📊 Reset {snapped_count} SNAPPED predictions to active")
    print(f"🗑️  Removed {hold_count} legacy HOLD predictions")
    
    # --- 2. Reset performance attribution ---
    if PERF_ATTR_FILE.exists():
        # Reset to empty structure
        empty_attribution = {
            "by_strategy": {},
            "by_regime": {},
            "by_confidence": {},
            "by_action": {},
            "by_sector": {},
            "by_time_period": {},
            "total_predictions": 0,
            "total_correct": 0,
            "overall_accuracy": 0
        }
        with open(PERF_ATTR_FILE, 'w', encoding='utf-8') as f:
            json.dump(empty_attribution, f, indent=2)
        print("🧹 Reset performance_attribution.json to empty state")
    else:
        print("ℹ️  performance_attribution.json not found (nothing to reset)")
    
    print("\n✅ Done! Run 'python audit_predictions.py' to verify.")

if __name__ == "__main__":
    reset()
