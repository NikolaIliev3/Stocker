import sys
sys.path.insert(0, '.')
import logging
logging.basicConfig(level=logging.ERROR)
import random
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from data_fetcher import StockDataFetcher
from trading_analyzer import TradingAnalyzer
from hybrid_predictor import HybridStockPredictor

def run_comparison(samples=100):
    print(f"\nRUNNING ML EDGE COMPARISON ({samples} samples)...")
    fetcher = StockDataFetcher()
    analyzer = TradingAnalyzer(fetcher)
    data_dir = Path.home() / ".stocker"
    
    # Pre-fetch SPY for regime
    spy_result = fetcher.fetch_stock_history('SPY', period='2y')
    if spy_result and 'data' in spy_result:
        spy_df = pd.DataFrame(spy_result['data'])
        # Handle case-insensitive columns
        for col in ['Date', 'date']:
            if col in spy_df.columns:
                spy_df['date'] = pd.to_datetime(spy_df[col])
                break
        spy_df.set_index('date', inplace=True)
        analyzer.market_history = spy_df
        analyzer.backtest_spy_history = spy_df

    predictor = HybridStockPredictor(data_dir, 'trading', trading_analyzer=analyzer, data_fetcher=fetcher)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'PG', 'KO']
    target_hurdle = 3.0
    
    hybrid_wins = 0
    hybrid_total = 0
    rules_wins = 0
    rules_total = 0
    
    for i in range(samples):
        symbol = random.choice(symbols)
        test_date = datetime.now() - timedelta(days=random.randint(30, 200))
        
        try:
            # Get historical data
            end_date = test_date.strftime('%Y-%m-%d')
            start_date = (test_date - timedelta(days=365)).strftime('%Y-%m-%d')
            history_dict = fetcher.fetch_stock_history(symbol, start_date=start_date, end_date=end_date)
            if not history_dict or not history_dict.get('data'): continue
            
            stock_data = {'symbol': symbol, 'current_price': float(history_dict['data'][-1]['close'])}
            
            # --- HYBRID PREDICTION ---
            # Monkeypatch to ensure normal weights
            predictor._calculate_dynamic_weights = lambda self, r, m: {'rule': 0.5, 'ml': 0.5, 'ai': 0.0}
            res_h = predictor.predict(stock_data, history_dict, is_backtest=True, current_date=test_date)
            
            # --- RULES ONLY PREDICTION ---
            # Monkeypatch to force Rules
            predictor._calculate_dynamic_weights = lambda self, r, m: {'rule': 1.0, 'ml': 0.0, 'ai': 0.0}
            res_r = predictor.predict(stock_data, history_dict, is_backtest=True, current_date=test_date)
            
            # Get Outcome
            future = fetcher.fetch_stock_history(symbol, start_date=test_date.strftime('%Y-%m-%d'), period='1mo')
            if not future or 'data' not in future or len(future['data']) < 5: continue
            
            f_df = pd.DataFrame(future['data'])
            f_df = f_df[pd.to_datetime(f_df['Date']) > test_date].head(10)
            if f_df.empty: continue
            
            max_p = ((f_df['high'].max() - stock_data['current_price']) / stock_data['current_price']) * 100
            is_winner = max_p >= target_hurdle
            
            # Hybrid Signal Check
            if res_h['recommendation']['action'] == 'BUY' and res_h['recommendation']['confidence'] >= 60:
                hybrid_total += 1
                if is_winner: hybrid_wins += 1
                
            # Rules Signal Check
            if res_r['recommendation']['action'] == 'BUY' and res_r['recommendation']['confidence'] >= 60:
                rules_total += 1
                if is_winner: rules_wins += 1
                
            if i % 20 == 0: print(f"  Sample {i}/{samples}...")
            
        except Exception: continue

    print("\n" + "="*40)
    print("FINAL ML EDGE RESULTS")
    print("="*40)
    h_wr = (hybrid_wins/hybrid_total*100) if hybrid_total > 0 else 0
    r_wr = (rules_wins/rules_total*100) if rules_total > 0 else 0
    
    print(f"Hybrid Conviction Win Rate: {h_wr:.1f}% ({hybrid_wins}/{hybrid_total})")
    print(f"Rules Only Conviction Win Rate: {r_wr:.1f}% ({rules_wins}/{rules_total})")
    print(f"ML Precision Edge: {h_wr - r_wr:+.1f}pp")
    print("="*40 + "\n")

if __name__ == '__main__':
    run_comparison(100)
