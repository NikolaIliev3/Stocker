
import json
import os
import pandas as pd

path = os.path.expanduser('~/.stocker/backtest_results.json')
with open(path, 'r') as f:
    data = json.load(f)

details = data.get('test_details', [])[-50:]
df = pd.DataFrame(details)

def get_regime(r):
    if isinstance(r, dict): return r.get('regime', 'unknown')
    return str(r)

df['regime_str'] = df['market_regime'].apply(get_regime)

print("\n### BUY ACCURACY BY MARKET REGIME ###")
df_buys = df[df['predicted_action'] == 'BUY']
regime_stats = df_buys.groupby('regime_str')['was_correct'].agg(['count', 'sum', 'mean'])
regime_stats.columns = ['Total', 'Correct', 'Win Rate']
regime_stats['Win Rate'] = (regime_stats['Win Rate'] * 100).map('{:.1f}%'.format)
print(regime_stats)

# Group by Prediction Type and Confidence Bin
df['conf_bin'] = pd.cut(df['confidence'], bins=[0, 50, 60, 70, 75, 80, 90, 100])

# Filter for BUYs with SMA200 context
df_buys = df[df['predicted_action'] == 'BUY'].copy()

# Look for 'context' or 'reasoning' to find SMA200 status
def is_above_sma200(row):
    reasoning = str(row.get('reasoning', ''))
    if 'AboveSMA200=True' in reasoning: return True
    if 'AboveSMA200=False' in reasoning: return False
    # Fallback to checking indicators if context not in reasoning
    indicators = row.get('indicators', {})
    if isinstance(indicators, dict):
        return indicators.get('price_above_ema200', True)
    return True

df_buys['is_above_sma200'] = df_buys.apply(is_above_sma200, axis=1)

print("\n### BUY ACCURACY BY SMA200 CONTEXT ###")
sma_stats = df_buys.groupby('is_above_sma200')['was_correct'].agg(['count', 'sum', 'mean'])
sma_stats.columns = ['Total', 'Correct', 'Win Rate']
sma_stats['Win Rate'] = (sma_stats['Win Rate'] * 100).map('{:.1f}%'.format)
print(sma_stats)

print("\n### FINAL PERFORMANCE BY CONFIDENCE (BUYs ONLY) ###")
df_buys['conf_bin'] = pd.cut(df_buys['confidence'], bins=[0, 50, 60, 70, 75, 80, 85, 90, 100])
stats = df_buys.groupby(['is_above_sma200', 'conf_bin'])['was_correct'].agg(['count', 'sum', 'mean'])
stats.columns = ['Total', 'Correct', 'Win Rate']
stats['Win Rate'] = (stats['Win Rate'] * 100).map('{:.1f}%'.format)
print(stats.dropna())

print("\n### FINAL PERFORMANCE BY CONFIDENCE (Last 200 Samples - 2025 Market) ###")
for action in ['BUY', 'SELL', 'HOLD']:
    action_df = df[df['predicted_action'] == action]
    if action_df.empty: continue
    
    print(f"\n--- {action} Signals ---")
    stats = action_df.groupby('conf_bin')['was_correct'].agg(['count', 'sum', 'mean'])
    stats.columns = ['Total', 'Correct', 'Win Rate']
    stats['Win Rate'] = (stats['Win Rate'] * 100).map('{:.1f}%'.format)
    print(stats)

print("\n### OVERALL SUMMARY ###")
summary = df.groupby('predicted_action')['was_correct'].agg(['count', 'sum', 'mean'])
summary.columns = ['Total', 'Correct', 'Win Rate']
summary['Win Rate'] = (summary['Win Rate'] * 100).map('{:.1f}%'.format)
print(summary)
