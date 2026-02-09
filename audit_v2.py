import sys
sys.path.insert(0, '.')
import logging
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from data_fetcher import StockDataFetcher
from trading_analyzer import TradingAnalyzer
from hybrid_predictor import HybridStockPredictor

# Mute noisy logs
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger('AuditV2')

def get_outcome(fetcher, symbol, start_date, horizon_days=10, target_pct=3.0):
    """Determine if a winner occurred in the horizon"""
    try:
        # Fetch 20 days to be safe
        future = fetcher.fetch_stock_history(symbol, start_date=start_date.strftime('%Y-%m-%d'), period='1mo')
        if not future or 'data' not in future: return None
        
        df = pd.DataFrame(future['data'])
        # Clean columns
        for col in ['Date', 'date', 'High', 'high', 'Close', 'close']:
            if col in df.columns: df[col.lower()] = df[col]
            
        df['date'] = pd.to_datetime(df['date'])
        
        # Current price (at start_date)
        start_price = df[df['date'] <= start_date]['close'].iloc[-1] if not df[df['date'] <= start_date].empty else None
        if start_price is None: return None
        
        # Future window
        window = df[df['date'] > start_date].head(horizon_days)
        if window.empty: return None
        
        max_high = window['high'].max()
        return (max_high - start_price) / start_price * 100 >= target_pct
    except Exception: return None

def run_audit_v2(n_samples=200):
    print(f"\n{'='*60}")
    print(f"AUDIT V2: THE 'RIGHT WAY' VALIDATION ({n_samples} Samples)")
    print(f"{'='*60}\n")
    
    fetcher = StockDataFetcher()
    analyzer = TradingAnalyzer(fetcher)
    data_dir = Path.home() / ".stocker"
    
    # Pre-fetch SPY for regime
    spy_result = fetcher.fetch_stock_history('SPY', period='2y')
    if spy_result and 'data' in spy_result:
        spy_df = pd.DataFrame(spy_result['data'])
        for col in ['Date', 'date']:
            if col in spy_df.columns: spy_df['date'] = pd.to_datetime(spy_df[col])
        spy_df.set_index('date', inplace=True)
        analyzer.market_history = spy_df
        analyzer.backtest_spy_history = spy_df

    predictor = HybridStockPredictor(data_dir, 'trading', trading_analyzer=analyzer, data_fetcher=fetcher)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'JNJ', 'PG', 'KO',
               'V', 'MA', 'DIS', 'NKE', 'MCD', 'WMT', 'HD', 'CAT', 'BA', 'XOM']
    
    # Metrics containers
    raw_ml_results = [] # [is_correct, action, confidence]
    rules_results = []
    filter_stats = {
        'buy_vetoed_losers': 0,
        'buy_vetoed_winners': 0,
        'buy_passed_winners': 0,
        'buy_passed_losers': 0,
        'total_buy_signals': 0
    }
    
    # Start date AFTER training end to avoid leakage
    # Heavy retraining ended approx Feb 8. Let's use late 2025 to avoid any potential overlap if we are testing history, 
    # but for "Live" accuracy we want the absolute latest.
    # The user is concerned about leakage. I will use Jan 2026 for out-of-sample if the model was trained on data up to 2025.
    
    processed = 0
    while processed < n_samples:
        symbol = random.choice(symbols)
        # Random date in late 2025/early 2026 (Out of Sample)
        days_ago = random.randint(30, 90)
        test_date = datetime.now() - timedelta(days=days_ago)
        
        try:
            # 1. Fetch data
            hist_end = test_date.strftime('%Y-%m-%d')
            hist_start = (test_date - timedelta(days=365)).strftime('%Y-%m-%d')
            history_dict = fetcher.fetch_stock_history(symbol, start_date=hist_start, end_date=hist_end)
            if not history_dict or 'data' not in history_dict or len(history_dict['data']) < 50: continue
            
            # 2. Outcomes
            is_winner = get_outcome(fetcher, symbol, test_date)
            if is_winner is None: continue
            
            # 3. Raw Predictions (Bypassing Filters)
            # Monkeypatch filters to see raw intent
            original_filters = predictor._apply_pro_filters
            predictor._apply_pro_filters = lambda f_a, f_c, s_d, h_d, b_a: (f_a, f_c)
            
            stock_data = {'symbol': symbol, 'current_price': float(history_dict['data'][-1]['close'])}
            
            # ML Only
            predictor._calculate_dynamic_weights = lambda r, m: {'rule': 0.0, 'ml': 1.0, 'ai': 0.0}
            res_ml = predictor.predict(stock_data, history_dict, is_backtest=True, current_date=test_date)
            
            # Rules Only
            predictor._calculate_dynamic_weights = lambda r, m: {'rule': 1.0, 'ml': 0.0, 'ai': 0.0}
            res_rules = predictor.predict(stock_data, history_dict, is_backtest=True, current_date=test_date)
            
            # 4. Filters Check (Restoring filters)
            predictor._apply_pro_filters = original_filters
            # Use original weighting logic
            from hybrid_predictor import HybridStockPredictor as HSP
            predictor._calculate_dynamic_weights = HSP._calculate_dynamic_weights.__get__(predictor, HSP)
            
            res_filtered = predictor.predict(stock_data, history_dict, is_backtest=True, current_date=test_date)
            
            # 5. Populate Metrics
            ml_action = res_ml['recommendation']['action']
            rules_action = res_rules['recommendation']['action']
            filtered_action = res_filtered['recommendation']['action']
            
            # Correctness: is_winner is True if move >= 3%
            ml_correct = (ml_action == 'BUY' and is_winner) or (ml_action != 'BUY' and not is_winner)
            rules_correct = (rules_action == 'BUY' and is_winner) or (rules_action != 'BUY' and not is_winner)
            
            raw_ml_results.append(ml_correct)
            rules_results.append(rules_correct)
            
            # Filter Stats
            if ml_action == 'BUY':
                filter_stats['total_buy_signals'] += 1
                if filtered_action == 'BUY':
                    if is_winner: filter_stats['buy_passed_winners'] += 1
                    else: filter_stats['buy_passed_losers'] += 1
                else: # Vetoed
                    if not is_winner: filter_stats['buy_vetoed_losers'] += 1
                    else: filter_stats['buy_vetoed_winners'] += 1
            
            processed += 1
            if processed % 10 == 0: print(f"  Processed {processed}/{n_samples}...")
            
        except Exception: continue

    # Results Reporting
    print("\n" + "="*40)
    print("AUDIT V2 RESULTS")
    print("="*40)
    
    ml_acc = np.mean(raw_ml_results) * 100
    r_acc = np.mean(rules_results) * 100
    
    print(f"RAW ML Full-Sample Accuracy (BUY+HOLD/SELL correct): {ml_acc:.1f}%")
    print(f"Rules Full-Sample Accuracy (BUY+HOLD/SELL correct): {r_acc:.1f}%")
    print(f"ML Edge over Rules: {ml_acc - r_acc:+.1f}pp")
    
    print("\n--- Filter Performance (Selection Quality) ---")
    total_buys = filter_stats['total_buy_signals']
    if total_buys > 0:
        veto_rate = (filter_stats['buy_vetoed_losers'] + filter_stats['buy_vetoed_winners']) / total_buys * 100
        veto_precision = (filter_stats['buy_vetoed_losers'] / (filter_stats['buy_vetoed_losers'] + filter_stats['buy_vetoed_winners']) * 100) if (filter_stats['buy_vetoed_losers'] + filter_stats['buy_vetoed_winners']) > 0 else 0
        pass_precision = (filter_stats['buy_passed_winners'] / (filter_stats['buy_passed_winners'] + filter_stats['buy_passed_losers']) * 100) if (filter_stats['buy_passed_winners'] + filter_stats['buy_passed_losers']) > 0 else 0
        
        print(f"  Total BUY Intents: {total_buys}")
        print(f"  Veto Rate (Blocked by Filters): {veto_rate:.1f}%")
        print(f"  Veto Accuracy (Correctly Blocked Losers): {veto_precision:.1f}%")
        print(f"  Pass Accuracy (Signals that hit 3% Hurdle): {pass_precision:.1f}%")
        print(f"  False Vetoes (Blocked Winners): {filter_stats['buy_vetoed_winners']}")
        
    print("="*40 + "\n")

if __name__ == '__main__':
    run_audit_v2(200)
