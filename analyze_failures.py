
import json
import os
import pandas as pd
import numpy as np

path = os.path.expanduser(r'~/.stocker/backtest_results.json')
if not os.path.exists(path):
    print(f"File not found: {path}")
    exit()

with open(path, 'r') as f:
    data = json.load(f)

details = data.get('test_details', [])

# Filter for the last 200 entries (The Clean Run)
if len(details) > 200:
    details = details[-200:]

# Filter for FAILED BUYs
failed_buys = [x for x in details if x['predicted_action'] == 'BUY' and not x.get('was_correct', False)]
success_buys = [x for x in details if x['predicted_action'] == 'BUY' and x.get('was_correct', False)]

print(f"Total Failed BUYs: {len(failed_buys)}")
print(f"Total Success BUYs: {len(success_buys)}")

# Analyze Failure Modes
# Mode 1: Negative Return (Loss)
losses = [x for x in failed_buys if x['actual_change_pct'] < 0]
# Mode 2: Positive Return but missed target (Stagnation/Timeout)
stagnation = [x for x in failed_buys if x['actual_change_pct'] >= 0]

print(f"\nFailure Modes:")
print(f"1. Real Losses (Price dropped): {len(losses)} ({len(losses)/len(failed_buys)*100:.1f}%)")
if len(losses) > 0:
    avg_loss = np.mean([x['actual_change_pct'] for x in losses])
    max_loss = np.min([x['actual_change_pct'] for x in losses])
    print(f"   - Avg Loss: {avg_loss:.2f}%")
    print(f"   - Max Drawdown: {max_loss:.2f}%")

print(f"2. Stagnation (Price rose but missed target): {len(stagnation)} ({len(stagnation)/len(failed_buys)*100:.1f}%)")
if len(stagnation) > 0:
    avg_gain = np.mean([x['actual_change_pct'] for x in stagnation])
    print(f"   - Avg Gain (Missed): {avg_gain:.2f}%")
    # Check if they missed by a little
    near_miss = sum(1 for x in stagnation if x['actual_change_pct'] > 2.0) # Assuming target is around 3%
    print(f"   - Near Misses (>2% gain): {near_miss}")

# Check Confidence of Failures
avg_conf_fail = np.mean([x['confidence'] for x in failed_buys]) if failed_buys else 0
avg_conf_succ = np.mean([x['confidence'] for x in success_buys]) if success_buys else 0
print(f"\nConfidence Analysis:")
print(f"- Avg Confidence (Failures): {avg_conf_fail:.1f}%")
print(f"- Avg Confidence (Successes): {avg_conf_succ:.1f}%")

# Check Target Aggressiveness
# Calculate implied target %: (predicted_target / test_price) - 1
def get_target_pct(x):
    if x['predicted_target'] < x['test_price'] * 0.5: # It's a delta
        return (x['predicted_target'] / x['test_price']) * 100
    else:
        return ((x['predicted_target'] - x['test_price']) / x['test_price']) * 100

avg_target_fail = np.mean([get_target_pct(x) for x in failed_buys]) if failed_buys else 0
avg_target_succ = np.mean([get_target_pct(x) for x in success_buys]) if success_buys else 0

print(f"\nTarget Analysis:")
print(f"- Avg Target (Failures): {avg_target_fail:.2f}%")
print(f"- Avg Target (Successes): {avg_target_succ:.2f}%")
