import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from data_fetcher import StockDataFetcher
from trading_analyzer import TradingAnalyzer
from hybrid_predictor import HybridStockPredictor
import logging

logging.basicConfig(level=logging.ERROR)

def get_outcome(fetcher, symbol, start_date, target_pct=3.0, stop_loss_pct=-4.0):
    try:
        future = fetcher.fetch_stock_history(symbol, start_date=start_date.strftime('%Y-%m-%d'), period='1mo')
        if not future or 'data' not in future: return None
        df = pd.DataFrame(future['data'])
        for col in ['High', 'high', 'Low', 'low', 'Close', 'close']:
            if col in df.columns: df[col.lower()] = df[col]
        
        start_price_search = df[pd.to_datetime(df['Date'] if 'Date' in df.columns else df['date']) <= start_date]
        if start_price_search.empty: return None
        start_price = start_price_search['close'].iloc[-1]
        
        window = df[pd.to_datetime(df['Date'] if 'Date' in df.columns else df['date']) > start_date].head(10)
        for _, row in window.iterrows():
            if (row['high'] - start_price)/start_price*100 >= target_pct: return True
            if (row['low'] - start_price)/start_price*100 <= stop_loss_pct: return False
        return ((window['close'].iloc[-1] - start_price)/start_price*100) >= target_pct
    except: return None

def run_calibration(n_samples=500):
    fetcher = StockDataFetcher()
    analyzer = TradingAnalyzer(fetcher)
    predictor = HybridStockPredictor(Path.home()/".stocker", 'trading', trading_analyzer=analyzer, data_fetcher=fetcher)
    
    # Enforce Config 2 logic (Rules + Filters, No ML)
    predictor._calculate_dynamic_weights = lambda r, m: {'rule': 1.0, 'ml': 0.0, 'ai': 0.0}
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'PG', 'KO',
               'V', 'MA', 'DIS', 'NKE', 'MCD', 'WMT', 'HD', 'CAT', 'BA', 'XOM']
    buckets = {'60-70': [], '70-80': [], '80-90': [], '90-100': []}
    
    processed = 0
    while processed < n_samples:
        symbol = random.choice(symbols)
        test_date = datetime(2026, 1, 23) + timedelta(days=random.randint(0, 15))
        if test_date > datetime.now(): continue
        
        hist = fetcher.fetch_stock_history(symbol, end_date=test_date.strftime('%Y-%m-%d'), period='1y')
        if not hist or 'data' not in hist: continue
        
    # Predict using Config 2 parameters
        hist_df = pd.DataFrame(hist['data'])
        for col in ['High', 'high', 'Low', 'low', 'Close', 'close']:
            if col in hist_df.columns: hist_df[col.lower()] = hist_df[col]
        
        current_price = hist_df['close'].iloc[-1]
        
        pred = predictor.predict({'symbol': symbol, 'price': current_price}, hist, is_backtest=True, current_date=test_date)
        if pred['recommendation']['action'] == 'BUY':
            conf = pred['recommendation']['confidence']
            outcome = get_outcome(fetcher, symbol, test_date)
            if outcome is not None:
                if 60 <= conf < 70: buckets['60-70'].append(outcome)
                elif 70 <= conf < 80: buckets['70-80'].append(outcome)
                elif 80 <= conf < 90: buckets['80-90'].append(outcome)
                elif 90 <= conf <= 100: buckets['90-100'].append(outcome)
                processed += 1
                print(f"  [{processed}/{n_samples}] {symbol} Conf: {conf:.1f}% -> {outcome}", end='\r')

    print("\n\n" + "="*40)
    print(f"{'Conf Bucket':<15} | {'Win Rate':<10} | {'Samples'}")
    print("-" * 40)
    for b, res in buckets.items():
        wr = (sum(res)/len(res)*100) if res else 0
        print(f"{b:<15} | {wr:>8.1f}% | {len(res)}")
    print("="*40)

if __name__ == "__main__":
    run_calibration(500)
