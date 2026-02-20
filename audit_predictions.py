"""
Audit script to check verified predictions for false positives.
Checks if actual_price_at_target was snapped to target_price (suspicious).
"""
import json
from pathlib import Path

PREDICTIONS_FILE = Path.home() / ".stocker" / "predictions.json"

def audit():
    with open(PREDICTIONS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle both list and dict formats
    if isinstance(data, dict):
        predictions = data.get('predictions', [])
    elif isinstance(data, list):
        predictions = data
    else:
        print(f"Unexpected data type: {type(data)}")
        return
    
    verified = [p for p in predictions if isinstance(p, dict) and p.get('verified')]
    
    print(f"Total predictions: {len(predictions)}")
    print(f"Total verified: {len(verified)}")
    print(f"Correct: {sum(1 for p in verified if p.get('was_correct'))}")
    print(f"Incorrect: {sum(1 for p in verified if not p.get('was_correct'))}")
    print()
    
    print("=" * 120)
    print(f"{'ID':>4} {'Symbol':<8} {'Action':<6} {'Entry':>10} {'Target':>10} {'Actual':>10} {'Correct':>8} {'Days':>5} {'Suspicious':>12}")
    print("=" * 120)
    
    suspicious_count = 0
    for p in verified:
        entry = p.get('entry_price', 0)
        target = p.get('target_price', 0)
        actual = p.get('actual_price_at_target', 0)
        correct = p.get('was_correct', False)
        days = p.get('days_to_target', '?')
        
        # Flag: actual == target means price was "snapped" (retroactive high check hit)
        snapped = abs(actual - target) < 0.01 if actual and target else False
        
        # Flag: correct but actual < target for BUY (or actual > target for SELL)
        action = p.get('action', '?')
        wrong_direction = False
        if correct and action == 'BUY' and actual < target and not snapped:
            wrong_direction = True
        if correct and action == 'SELL' and actual > target and not snapped:
            wrong_direction = True
        
        suspicious = ""
        if snapped and correct:
            suspicious = "SNAPPED ⚠️"
            suspicious_count += 1
        if wrong_direction:
            suspicious = "WRONG DIR ⚠️"
            suspicious_count += 1
            
        print(f"{p['id']:>4} {p['symbol']:<8} {action:<6} ${entry:>9,.2f} ${target:>9,.2f} ${actual:>9,.2f} {'✓' if correct else '✗':>8} {days:>5} {suspicious:>12}")
    
    print("=" * 120)
    print(f"\n⚠️ Suspicious predictions: {suspicious_count}")
    print("\nSNAPPED = actual_price_at_target equals target_price exactly (retroactive high claimed to hit target)")
    print("WRONG DIR = marked correct but actual price didn't reach target")

if __name__ == "__main__":
    audit()
