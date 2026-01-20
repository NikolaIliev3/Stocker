"""
Volatile Stock Analyzer for Stocker App
Identifies which stocks in your training/predictions are most volatile
and should potentially be removed to improve ML accuracy.

Usage: python analyze_volatile_stocks.py
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Try to import yfinance for volatility calculation
try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False
    logger.warning("⚠️ yfinance not available - will use prediction accuracy instead")

from config import APP_DATA_DIR


def calculate_volatility(symbol: str, period: str = "3mo") -> dict:
    """Calculate volatility metrics for a stock"""
    if not HAS_YF:
        return None
    
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period)
        
        if len(hist) < 20:
            return None
        
        # Calculate daily returns
        returns = hist['Close'].pct_change().dropna()
        
        # Volatility metrics
        std_dev = returns.std() * 100  # Daily volatility %
        annualized_vol = std_dev * (252 ** 0.5)  # Annualized
        max_daily_move = returns.abs().max() * 100
        avg_daily_move = returns.abs().mean() * 100
        
        # Price range (high-low / avg price)
        price_range = ((hist['High'].max() - hist['Low'].min()) / hist['Close'].mean()) * 100
        
        return {
            'daily_volatility': std_dev,
            'annualized_volatility': annualized_vol,
            'max_daily_move': max_daily_move,
            'avg_daily_move': avg_daily_move,
            'price_range_pct': price_range
        }
    except Exception as e:
        logger.debug(f"Error calculating volatility for {symbol}: {e}")
        return None


def analyze_prediction_accuracy(predictions_file: Path) -> dict:
    """Analyze prediction accuracy per symbol"""
    if not predictions_file.exists():
        return {}
    
    try:
        with open(predictions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        return {}
    
    # Handle both {"predictions": [...]} and [...] formats
    if isinstance(data, dict):
        predictions = data.get('predictions', [])
    elif isinstance(data, list):
        predictions = data
    else:
        return {}
    
    symbol_stats = {}
    
    for pred in predictions:
        symbol = pred.get('symbol', '').upper()
        if not symbol:
            continue
        
        if symbol not in symbol_stats:
            symbol_stats[symbol] = {'correct': 0, 'incorrect': 0, 'total': 0}
        
        symbol_stats[symbol]['total'] += 1
        
        if pred.get('was_correct') is True:
            symbol_stats[symbol]['correct'] += 1
        elif pred.get('was_correct') is False:
            symbol_stats[symbol]['incorrect'] += 1
    
    # Calculate accuracy
    for symbol, stats in symbol_stats.items():
        verified = stats['correct'] + stats['incorrect']
        if verified > 0:
            stats['accuracy'] = (stats['correct'] / verified) * 100
        else:
            stats['accuracy'] = None  # Not enough data
    
    return symbol_stats


def main():
    print("=" * 60)
    print("🔍 VOLATILE STOCK ANALYZER")
    print("   Identifies unpredictable stocks in your ML training data")
    print("=" * 60)
    print()
    
    # 1. Analyze prediction accuracy
    predictions_file = APP_DATA_DIR / "predictions.json"
    symbol_accuracy = analyze_prediction_accuracy(predictions_file)
    
    # 2. Get unique symbols from predictions
    symbols = list(symbol_accuracy.keys())
    
    if not symbols:
        # Try to get from training symbols
        prefs_file = APP_DATA_DIR / "preferences.json"
        if prefs_file.exists():
            try:
                with open(prefs_file, 'r') as f:
                    prefs = json.load(f)
                training_symbols = prefs.get('training_symbols', {})
                for strategy, syms in training_symbols.items():
                    symbols.extend([s.strip().upper() for s in syms.split(',')])
                symbols = list(set(symbols))
            except:
                pass
    
    if not symbols:
        print("❌ No symbols found in predictions or training data.")
        return
    
    print(f"📊 Analyzing {len(symbols)} stocks...\n")
    
    # 3. Collect volatility data
    results = []
    
    for symbol in symbols:
        print(f"   Analyzing {symbol}...", end=" ")
        
        result = {
            'symbol': symbol,
            'volatility': None,
            'accuracy': symbol_accuracy.get(symbol, {}).get('accuracy'),
            'total_predictions': symbol_accuracy.get(symbol, {}).get('total', 0),
            'score': 0  # Lower = more stable/accurate
        }
        
        # Get volatility if possible
        if HAS_YF:
            vol_data = calculate_volatility(symbol)
            if vol_data:
                result['volatility'] = vol_data
                result['daily_vol'] = vol_data['daily_volatility']
                print(f"Vol: {vol_data['daily_volatility']:.2f}%", end=" ")
            else:
                print("Vol: N/A", end=" ")
        
        # Print accuracy
        if result['accuracy'] is not None:
            print(f"Acc: {result['accuracy']:.1f}%")
        else:
            print("Acc: N/A")
        
        # Calculate problem score (higher = more problematic)
        score = 0
        
        # High volatility is bad
        if result.get('daily_vol'):
            if result['daily_vol'] > 3:
                score += 30  # Very volatile
            elif result['daily_vol'] > 2:
                score += 20
            elif result['daily_vol'] > 1.5:
                score += 10
        
        # Low accuracy is bad
        if result['accuracy'] is not None:
            if result['accuracy'] < 40:
                score += 40  # Very bad accuracy
            elif result['accuracy'] < 50:
                score += 25
            elif result['accuracy'] < 55:
                score += 10
        
        # Many predictions with low accuracy = very problematic
        if result['accuracy'] is not None and result['accuracy'] < 50:
            score += min(20, result['total_predictions'] * 2)
        
        result['score'] = score
        results.append(result)
    
    # Sort by problem score (highest first = most problematic)
    results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n" + "=" * 60)
    print("📋 RESULTS - Stocks Ranked by Volatility/Unpredictability")
    print("   (Higher score = more problematic for ML training)")
    print("=" * 60)
    
    # Categorize
    remove_list = []
    watch_list = []
    keep_list = []
    
    for r in results:
        vol_str = f"{r.get('daily_vol', 0):.2f}%" if r.get('daily_vol') else "N/A"
        acc_str = f"{r['accuracy']:.1f}%" if r['accuracy'] is not None else "N/A"
        
        line = f"  {r['symbol']:8} | Score: {r['score']:3} | Volatility: {vol_str:7} | Accuracy: {acc_str:6} | Predictions: {r['total_predictions']}"
        
        if r['score'] >= 40:
            remove_list.append(r['symbol'])
            print(f"🔴 {line}")
        elif r['score'] >= 20:
            watch_list.append(r['symbol'])
            print(f"🟡 {line}")
        else:
            keep_list.append(r['symbol'])
            print(f"🟢 {line}")
    
    print("\n" + "=" * 60)
    print("📊 RECOMMENDATIONS")
    print("=" * 60)
    
    if remove_list:
        print(f"\n🔴 REMOVE from training ({len(remove_list)} stocks):")
        print(f"   These stocks are too volatile or have poor prediction accuracy.")
        print(f"   Symbols: {', '.join(remove_list)}")
    
    if watch_list:
        print(f"\n🟡 WATCH LIST ({len(watch_list)} stocks):")
        print(f"   Consider removing if accuracy doesn't improve.")
        print(f"   Symbols: {', '.join(watch_list)}")
    
    if keep_list:
        print(f"\n🟢 KEEP for training ({len(keep_list)} stocks):")
        print(f"   These stocks are stable enough for ML training.")
        print(f"   Symbols: {', '.join(keep_list)}")
    
    print("\n" + "=" * 60)
    print("💡 NEXT STEPS")
    print("=" * 60)
    print("""
1. To remove problematic stocks from predictions:
   - Go to Predictions tab → delete predictions for volatile stocks
   
2. To retrain with only stable stocks:
   - Go to ML Training → use only the 🟢 KEEP stocks
   
3. For a clean slate:
   - Run: python factory_reset_ml.py
   - Then retrain with stable stocks only
""")
    
    # Save results to file
    output_file = APP_DATA_DIR / "volatility_analysis.json"
    try:
        with open(output_file, 'w') as f:
            json.dump({
                'analysis_date': datetime.now().isoformat(),
                'remove_list': remove_list,
                'watch_list': watch_list,
                'keep_list': keep_list,
                'details': results
            }, f, indent=2)
        print(f"📁 Full analysis saved to: {output_file}")
    except Exception as e:
        logger.error(f"Could not save analysis: {e}")


if __name__ == "__main__":
    main()
