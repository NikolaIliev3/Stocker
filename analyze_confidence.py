
import json
from pathlib import Path
import pandas as pd

def analyze_accuracy():
    file_path = Path("c:/Users/Никола/.stocker/backtest_results.json")
    if not file_path.exists():
        print("File not found")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    details = data.get('test_details', [])
    if not details:
        print("No test details found")
        return

    df = pd.DataFrame(details)
    
    # Clean up confidence
    df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
    df = df.dropna(subset=['confidence'])
    
    # Define buckets
    bins = [50, 60, 70, 75, 80, 85, 90, 101]
    labels = ["50-60%", "60-70%", "70-75%", "75-80%", "80-85%", "85-90%", "90-100%"]
    df['conf_bucket'] = pd.cut(df['confidence'], bins=bins, labels=labels, right=False)
    
    # Ensure was_correct is boolean
    df['was_correct'] = df['was_correct'].astype(bool)
    
    # Analysis 1: Global
    print("\n" + "="*50)
    print("🌍 GLOBAL ACCURACY BY CONFIDENCE")
    print("="*50)
    global_stats = df.groupby('conf_bucket')['was_correct'].agg(['count', 'mean']).reset_index()
    global_stats['mean'] = (global_stats['mean'] * 100).round(1).astype(str) + "%"
    print(global_stats.to_string(index=False))
    
    # Analysis 2: By Regime
    print("\n" + "="*50)
    print("🎭 ACCURACY BY REGIME AND CONFIDENCE")
    print("="*50)
    
    # Check what regimes we have
    regimes = df['market_regime'].unique()
    
    for regime in regimes:
        print(f"\n--- Regime: {regime} ---")
        regime_df = df[df['market_regime'] == regime]
        stats = regime_df.groupby('conf_bucket')['was_correct'].agg(['count', 'mean']).reset_index()
        stats['mean'] = (stats['mean'] * 100).round(1).astype(str) + "%"
        print(stats.to_string(index=False))

if __name__ == "__main__":
    analyze_accuracy()
