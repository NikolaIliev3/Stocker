"""
Calibration Verification Test
Measures how well model confidence matches actual accuracy at each confidence level
"""
import sys
sys.path.insert(0, '.')

import logging
logging.basicConfig(level=logging.WARNING)

import random
import json
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict

def run_calibration_test():
    print("\n" + "="*60)
    print("CALIBRATION VERIFICATION TEST")
    print("Does confidence match actual accuracy?")
    print("="*60 + "\n")
    
    # Load backtest results
    data_dir = Path.home() / '.stocker'
    results_file = data_dir / 'backtest_results.json'
    
    if not results_file.exists():
        print("❌ No backtest results found. Run a backtest first.")
        return
    
    with open(results_file) as f:
        data = json.load(f)
    
    # Extract results
    results = data.get('results', [])
    if not results:
        print("❌ No results in backtest file.")
        return
    
    # Group by confidence buckets
    buckets = defaultdict(lambda: {'correct': 0, 'total': 0})
    
    for r in results:
        conf = r.get('confidence', 50)
        is_correct = r.get('correct', False)
        
        # Bucket by 10% increments
        bucket = int(conf // 10) * 10
        buckets[bucket]['total'] += 1
        if is_correct:
            buckets[bucket]['correct'] += 1
    
    print("Confidence vs Actual Accuracy:\n")
    print(f"{'Confidence':<15} {'Samples':<10} {'Correct':<10} {'Accuracy':<12} {'Calibration Error'}")
    print("-" * 70)
    
    total_calibration_error = 0
    weighted_count = 0
    
    for bucket in sorted(buckets.keys()):
        data = buckets[bucket]
        total = data['total']
        correct = data['correct']
        accuracy = (correct / total * 100) if total > 0 else 0
        
        # Expected accuracy is the bucket midpoint (e.g., 70-80% → 75% expected)
        expected = bucket + 5
        calibration_error = abs(accuracy - expected)
        
        # Skip buckets with too few samples
        if total >= 5:
            total_calibration_error += calibration_error * total
            weighted_count += total
            
            status = "✓" if calibration_error < 10 else "⚠️" if calibration_error < 20 else "❌"
            print(f"{bucket}-{bucket+10}%{'':<7} {total:<10} {correct:<10} {accuracy:.1f}%{'':<8} {calibration_error:+.1f}% {status}")
        else:
            print(f"{bucket}-{bucket+10}%{'':<7} {total:<10} {correct:<10} {accuracy:.1f}%{'':<8} (too few samples)")
    
    if weighted_count > 0:
        avg_error = total_calibration_error / weighted_count
        print("-" * 70)
        print(f"\nWeighted Average Calibration Error: {avg_error:.1f}%")
        
        if avg_error < 10:
            print("✅ WELL CALIBRATED - Confidence matches reality within 10%")
        elif avg_error < 15:
            print("⚠️ MODERATELY CALIBRATED - Some drift between confidence and accuracy")
        else:
            print("❌ POORLY CALIBRATED - Confidence does not match actual accuracy")
    
    print(f"\nTotal samples analyzed: {sum(b['total'] for b in buckets.values())}")

if __name__ == '__main__':
    run_calibration_test()
