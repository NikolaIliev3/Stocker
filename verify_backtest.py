"""
Direct ML prediction test to verify model accuracy
Tests predictions on random historical points without needing GUI
"""
import sys
sys.path.insert(0, '.')

import logging
# Enable debug logging for this test
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from data_fetcher import StockDataFetcher
from trading_analyzer import TradingAnalyzer
from hybrid_predictor import HybridStockPredictor

def get_history_data(fetcher, symbol, start, end):
    try:
        # We need a bit more data for indicators
        start_date = (datetime.strptime(start, '%Y-%m-%d') - timedelta(days=100)).strftime('%Y-%m-%d')
        result = fetcher.fetch_stock_history(symbol, start_date=start_date, end_date=end)
        return result
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def run_verification():
    print("\n" + "="*60)
    print("ML MODEL VERIFICATION TEST (PHASE 3 AUDIT)")
    print("="*60 + "\n")
    
    import time
    random.seed(time.time())
    
    data_dir = Path.home() / ".stocker"
    fetcher = StockDataFetcher()
    
    # 1. MARKET REGIME FALLBACK
    print("Pre-fetching SPY data for market regime...")
    try:
        spy_result = fetcher.fetch_stock_history('SPY', period='3y', interval='1d')
    except Exception as e:
        print(f"⚠️ Market Regime Fallback: Could not fetch SPY ({e}).")
        spy_result = None
    
    analyzer = TradingAnalyzer(fetcher)
    
    if spy_result and 'data' in spy_result:
        spy_df = pd.DataFrame(spy_result['data'])
        for col in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']:
            if col in spy_df.columns: spy_df[col.lower()] = spy_df[col]
        spy_df['date'] = pd.to_datetime(spy_df['date'])
        spy_df.set_index('date', inplace=True)
        analyzer.market_history = spy_df.sort_index()
        analyzer.backtest_spy_history = spy_df.sort_index()
        print("✓ SPY history loaded and injected.")
    else:
        print("💡 Simulation: Using Bullish Market Regime for audit stability.")
        dummy_data = [{'Date': (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'), 
                       'close': 500 + (252-i)*0.5, 'high': 505 + (252-i)*0.5, 'low': 495 + (252-i)*0.5, 'volume': 1e6} 
                      for i in range(252)]
        dummy_df = pd.DataFrame(list(reversed(dummy_data)))
        dummy_df['date'] = pd.to_datetime(dummy_df['Date'])
        dummy_df.set_index('date', inplace=True)
        analyzer.market_history = dummy_df
        analyzer.backtest_spy_history = dummy_df

    predictor = HybridStockPredictor(data_dir, 'trading', trading_analyzer=analyzer, data_fetcher=fetcher)
    
    # 2. CONFIGURATION
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'PG', 'KO',
               'V', 'MA', 'DIS', 'NKE', 'MCD', 'WMT', 'HD', 'CAT', 'BA', 'XOM']
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', type=int, default=200, help='Number of samples to test')
    args_cli, _ = parser.parse_known_args()
    n_samples = args_cli.samples
    
    target_hurdle = 3.0 # Sniper Hurdle
    correct = 0
    total = 0
    results = []
    audit_records = []
    
    print(f"Testing {n_samples} samples with Sniper Hurdle ({target_hurdle}%)...\n")
    
    for i in range(n_samples):
        symbol = random.choice(symbols)
        days_ago = random.randint(30, 365)
        test_date = datetime.now() - timedelta(days=days_ago)
        
        try:
            # Get historical data up to test_date
            end_date = test_date.strftime('%Y-%m-%d')
            start_date = (test_date - timedelta(days=365)).strftime('%Y-%m-%d')
            
            history_dict = get_history_data(fetcher, symbol, start_date, end_date)
            if not history_dict or not history_dict.get('data'): continue
            
            history_df = pd.DataFrame(history_dict['data'])
            if len(history_df) < 50: continue
            
            stock_data = {
                'symbol': symbol,
                'current_price': float(history_df['close'].iloc[-1]),
                'volume': int(history_df['volume'].iloc[-1]) if 'volume' in history_df else 0
            }
            
            # Predict (is_backtest=True bypasses the 80% cap for audit)
            prediction = predictor.predict(stock_data, history_dict, is_backtest=True, current_date=test_date)
            if not prediction or 'recommendation' not in prediction: continue
            
            rec = prediction['recommendation']
            action = rec.get('action', 'HOLD')
            confidence = rec.get('confidence', 50)
            
            # --- PHASE 3 CAPTURE ---
            if confidence >= 70:
                print(f"DEBUG: prediction keys: {list(prediction.keys())}")
                indicators = prediction.get('indicators', {})
                print(f"DEBUG: indicators keys: {list(indicators.keys())}")
                vol_analysis = prediction.get('volume_analysis', {})
                current_price = stock_data['current_price']
                ema_50_price = indicators.get('ema_50', current_price)
                
                audit_records.append({
                    'id': i,
                    'symbol': symbol,
                    'confidence': confidence,
                    'is_correct': False, # Update later
                    'rsi': float(indicators.get('rsi', 50)),
                    'atr_pct': float(indicators.get('atr_percent', 0)),
                    'dist_sma50': ((current_price / ema_50_price) - 1) * 100 if ema_50_price else 0,
                    'volume_ratio': float(vol_analysis.get('volume_ratio', 1.0))
                })

            # --- VERIFY OUTCOME ---
            window_days = 15
            future_prices_dict = fetcher.fetch_stock_history(symbol, start_date=test_date.strftime('%Y-%m-%d'), period='1mo')
            if not future_prices_dict or not future_prices_dict.get('data'):
                print(f"  ⚠️ Skipped {symbol}: Could not fetch future data for {test_date.strftime('%Y-%m-%d')}")
                continue
            
            future_df = pd.DataFrame(future_prices_dict['data'])
            # Clean column names and convert types
            for col in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']:
                if col in future_df.columns: future_df[col.lower()] = future_df[col]
            
            if 'date' in future_df.columns:
                future_df['date'] = pd.to_datetime(future_df['date'])
            
            future_df = future_df[future_df['date'] > pd.to_datetime(test_date)].head(window_days)
            if future_df.empty:
                print(f"  ⚠️ Skipped {symbol}: Future window empty after {test_date.strftime('%Y-%m-%d')}")
                continue
            
            final_price = float(future_df['close'].iloc[-1])
            actual_return = ((final_price - stock_data['current_price']) / stock_data['current_price']) * 100 if action == 'BUY' else 0
            
            # Path Analysis (Orphan Lock)
            max_p = 0.0
            for _, row in future_df.iterrows():
                p = ((row['high'] - stock_data['current_price']) / stock_data['current_price']) * 100 if action == 'BUY' else 0
                max_p = max(max_p, p)
            
            is_win = max_p >= target_hurdle if action == 'BUY' else False
            
            if is_win:
                correct += 1
                outcome = "✓ (WIN)"
            else:
                outcome = "✗ (FAIL)"
            
            total += 1
            # Update audit record
            for r in audit_records:
                if r['id'] == i:
                    r['is_correct'] = is_win
                    break
            
            results.append({'symbol': symbol, 'confidence': confidence, 'win': is_win, 'action': action})
            
            if i % 20 == 0:
                print(f"  Processed {i}/{n_samples}...")

        except Exception as e:
            print(f"  ❌ Error on {symbol} / {test_date}: {e}")
            continue

    # --- REPORTING ---
    print(f"\n{'='*60}")
    print("AUDIT SUMMARY")
    print(f"{'='*60}")
    print(f"  Total Trades: {total}")
    if total > 0:
        print(f"  Accuracy: {correct/total*100:.1f}%")
    else:
        print("  Accuracy: N/A (No trades completed)")
    
    if audit_records:
        df = pd.DataFrame(audit_records)
        for zone, query in [("70-80% Zone", "(confidence >= 70) & (confidence < 80)"),
                            ("80%+ Zone", "confidence >= 80")]:
            group = df.query(query)
            if group.empty: continue
            wr = group['is_correct'].mean() * 100
            print(f"\n--- {zone} ---")
            print(f"  Samples: {len(group)} | Win Rate: {wr:.1f}%")
            winners = group[group['is_correct'] == True]
            losers = group[group['is_correct'] == False]
            for feat in ['rsi', 'dist_sma50', 'volume_ratio']:
                w = winners[feat].mean() if not winners.empty else 0
                l = losers[feat].mean() if not losers.empty else 0
                print(f"    {feat:<12} | Win: {w:>5.1f} | Loss: {l:>5.1f} | Delta: {l-w:>+5.1f}")

if __name__ == '__main__':
    run_verification()
