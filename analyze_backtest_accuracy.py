"""
Backtest Results Analyzer for Stocker App
Analyzes backtest_results.json to find which stocks have the worst prediction accuracy.
This helps identify unpredictable stocks that should be removed from ML training.

Usage: python analyze_backtest_accuracy.py
"""
import json
import sys
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8')

from config import APP_DATA_DIR


def analyze_backtest_results():
    """Analyze backtest results to find worst-performing stocks"""
    
    results_file = APP_DATA_DIR / "backtest_results.json"
    
    if not results_file.exists():
        print("❌ backtest_results.json not found!")
        print(f"   Expected at: {results_file}")
        return
    
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Error loading backtest results: {e}")
        return
    
    test_details = data.get('test_details', [])
    total_tests = data.get('total_tests', 0)
    overall_accuracy = (data.get('correct_predictions', 0) / total_tests * 100) if total_tests > 0 else 0
    
    print("=" * 70)
    print("📊 BACKTEST ACCURACY ANALYZER")
    print("   Identifies which stocks have the worst prediction accuracy")
    print("=" * 70)
    print(f"\n📈 Overall Stats: {total_tests} tests, {overall_accuracy:.1f}% accuracy\n")
    
    # Group tests by symbol
    symbol_stats = defaultdict(lambda: {'correct': 0, 'incorrect': 0, 'total': 0, 'buy': 0, 'sell': 0, 'hold': 0})
    
    for test in test_details:
        symbol = test.get('symbol', 'UNKNOWN')
        was_correct = test.get('was_correct', False)
        action = test.get('predicted_action', 'HOLD')
        
        symbol_stats[symbol]['total'] += 1
        if was_correct:
            symbol_stats[symbol]['correct'] += 1
        else:
            symbol_stats[symbol]['incorrect'] += 1
        
        if action == 'BUY':
            symbol_stats[symbol]['buy'] += 1
        elif action == 'SELL':
            symbol_stats[symbol]['sell'] += 1
        else:
            symbol_stats[symbol]['hold'] += 1
    
    # Calculate accuracy for each symbol
    results = []
    for symbol, stats in symbol_stats.items():
        accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        
        # Calculate problem score (higher = worse)
        score = 0
        
        # Poor accuracy is bad
        if accuracy < 40:
            score += 50
        elif accuracy < 45:
            score += 35
        elif accuracy < 50:
            score += 20
        elif accuracy < 55:
            score += 10
        
        # More tests with low accuracy = more problematic
        if accuracy < 50:
            score += min(20, stats['total'] * 1.5)
        
        results.append({
            'symbol': symbol,
            'accuracy': accuracy,
            'total': stats['total'],
            'correct': stats['correct'],
            'incorrect': stats['incorrect'],
            'buy_pct': stats['buy'] / stats['total'] * 100 if stats['total'] > 0 else 0,
            'sell_pct': stats['sell'] / stats['total'] * 100 if stats['total'] > 0 else 0,
            'hold_pct': stats['hold'] / stats['total'] * 100 if stats['total'] > 0 else 0,
            'score': score
        })
    
    # Sort by accuracy (worst first)
    results.sort(key=lambda x: x['accuracy'])
    
    print("=" * 70)
    print("📋 STOCKS RANKED BY BACKTEST ACCURACY (Worst First)")
    print("=" * 70)
    print(f"{'Symbol':<12} {'Acc%':>8} {'Tests':>8} {'Correct':>8} {'Wrong':>8} {'Score':>8}")
    print("-" * 70)
    
    remove_list = []
    watch_list = []
    keep_list = []
    
    for r in results:
        # Categorize
        if r['accuracy'] < 45 and r['total'] >= 5:
            category = "🔴"
            remove_list.append(r['symbol'])
        elif r['accuracy'] < 52 and r['total'] >= 5:
            category = "🟡"
            watch_list.append(r['symbol'])
        else:
            category = "🟢"
            keep_list.append(r['symbol'])
        
        print(f"{category} {r['symbol']:<10} {r['accuracy']:>7.1f}% {r['total']:>8} {r['correct']:>8} {r['incorrect']:>8} {r['score']:>8.0f}")
    
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    # Group by accuracy tiers
    tier_0_40 = [r for r in results if r['accuracy'] < 40 and r['total'] >= 5]
    tier_40_50 = [r for r in results if 40 <= r['accuracy'] < 50 and r['total'] >= 5]
    tier_50_60 = [r for r in results if 50 <= r['accuracy'] < 60 and r['total'] >= 5]
    tier_60_plus = [r for r in results if r['accuracy'] >= 60 and r['total'] >= 5]
    
    print(f"\n📉 Critical (<40% accuracy): {len(tier_0_40)} stocks")
    if tier_0_40:
        print(f"   {', '.join([r['symbol'] for r in tier_0_40])}")
    
    print(f"\n⚠️  Poor (40-50% accuracy): {len(tier_40_50)} stocks")
    if tier_40_50:
        print(f"   {', '.join([r['symbol'] for r in tier_40_50])}")
    
    print(f"\n✓  Acceptable (50-60% accuracy): {len(tier_50_60)} stocks")
    if tier_50_60:
        print(f"   {', '.join([r['symbol'] for r in tier_50_60])}")
    
    print(f"\n🎯 Good (60%+ accuracy): {len(tier_60_plus)} stocks")
    if tier_60_plus:
        print(f"   {', '.join([r['symbol'] for r in tier_60_plus])}")
    
    print("\n" + "=" * 70)
    print("💡 RECOMMENDATIONS")
    print("=" * 70)
    
    if remove_list:
        print(f"\n🔴 REMOVE FROM TRAINING ({len(remove_list)} stocks):")
        print(f"   These stocks have very poor prediction accuracy (<45%).")
        print(f"   Symbols: {', '.join(remove_list)}")
        print(f"\n   To remove: Delete predictions for these symbols, then retrain ML.")
    
    if watch_list:
        print(f"\n🟡 WATCH LIST ({len(watch_list)} stocks):")
        print(f"   Borderline accuracy (45-52%). Monitor these.")
        print(f"   Symbols: {', '.join(watch_list)}")
    
    if keep_list:
        print(f"\n🟢 GOOD FOR TRAINING ({len(keep_list)} stocks):")
        print(f"   Decent accuracy (52%+). Keep these.")
        print(f"   Symbols: {', '.join(keep_list)}")
    
    # Calculate what accuracy would be if we removed bad stocks
    if remove_list:
        good_tests = sum(r['correct'] for r in results if r['symbol'] not in remove_list)
        good_total = sum(r['total'] for r in results if r['symbol'] not in remove_list)
        potential_accuracy = (good_tests / good_total * 100) if good_total > 0 else 0
        
        print(f"\n📈 POTENTIAL IMPROVEMENT:")
        print(f"   Current overall accuracy: {overall_accuracy:.1f}%")
        print(f"   Accuracy without 🔴 stocks: {potential_accuracy:.1f}%")
        print(f"   Potential improvement: +{potential_accuracy - overall_accuracy:.1f}%")
    
    print("\n" + "=" * 70)
    print("📁 NEXT STEPS")
    print("=" * 70)
    print("""
1. Remove predictions for 🔴 stocks in the Predictions tab
2. Run 'python factory_reset_ml.py' to clear ML models
3. Retrain using only 🟢 GOOD stocks
4. Run backtests again to verify improvement
""")
    
    # Save results
    output_file = APP_DATA_DIR / "backtest_accuracy_analysis.json"
    try:
        with open(output_file, 'w') as f:
            json.dump({
                'analysis_date': datetime.now().isoformat(),
                'overall_accuracy': overall_accuracy,
                'total_tests': total_tests,
                'remove_list': remove_list,
                'watch_list': watch_list,
                'keep_list': keep_list,
                'details': results
            }, f, indent=2)
        print(f"📁 Full analysis saved to: {output_file}")
    except Exception as e:
        print(f"⚠️ Could not save analysis: {e}")


if __name__ == "__main__":
    analyze_backtest_results()
