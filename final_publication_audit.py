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
logger = logging.getLogger('FinalAudit')

def get_outcome_details(fetcher, symbol, start_date, horizon_days=10, target_pct=3.0, stop_loss_pct=-4.0):
    """Path-dependent outcome: Exit at +3%, -4%, or EOW"""
    try:
        # Fetch slightly more to ensure we have intraday-like high/low behavior
        future = fetcher.fetch_stock_history(symbol, start_date=start_date.strftime('%Y-%m-%d'), period='1mo')
        if not future or 'data' not in future: return None
        
        df = pd.DataFrame(future['data'])
        for col in ['Date', 'date', 'High', 'high', 'Low', 'low', 'Close', 'close']:
            if col in df.columns: df[col.lower()] = df[col]
            
        df['date'] = pd.to_datetime(df['date'])
        
        # Current price (at start_date)
        start_price_search = df[df['date'] <= start_date]
        if start_price_search.empty: return None
        start_price = start_price_search['close'].iloc[-1]
        
        # 10-day window
        window = df[df['date'] > start_date].head(horizon_days)
        if window.empty: return None
        
        # Simulation loop: find FIRST exit trigger
        actual_price_change = 0
        is_winner = False
        
        for idx, row in window.iterrows():
            high_ret = (row['high'] - start_price) / start_price * 100
            low_ret = (row['low'] - start_price) / start_price * 100
            
            # 1. Target Hit (Orphan Lock)
            if high_ret >= target_pct:
                actual_price_change = target_pct
                is_winner = True
                break
            
            # 2. Hard Stop Loss Hit
            if low_ret <= stop_loss_pct:
                actual_price_change = stop_loss_pct
                is_winner = False
                break
        else:
            # 3. Time Stop (End of Window)
            actual_price_change = ((window['close'].iloc[-1] - start_price) / start_price * 100)
            is_winner = actual_price_change >= target_pct

        return {'is_winner': is_winner, 'price_change': actual_price_change, 'start_price': start_price}
    except Exception: return None

def run_publication_audit(n_samples=1000):
    print(f"\n{'='*70}")
    print(f"PUBLICATION STANDARD AUDIT ({n_samples} Samples)")
    print(f"Period: Jan 23, 2026 - Feb 8, 2026 (STRICT OUT-OF-SAMPLE)")
    print(f"{'='*70}\n")
    
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
    
    stats = {
        'baseline': {'correct': 0, 'intents': 0, 'passed': 0, 'wins': 0, 'profits': [], 'losses': []},
        'hardened': {'correct': 0, 'intents': 0, 'passed': 0, 'wins': 0, 'profits': [], 'losses': []},
        'full_system': {'correct': 0, 'intents': 0, 'passed': 0, 'wins': 0, 'profits': [], 'losses': []}
    }
    
    processed = 0
    while processed < n_samples:
        symbol = random.choice(symbols)
        # Random date Jan 23 - Feb 8
        test_date = datetime(2026, 1, 23) + timedelta(days=random.randint(0, 15))
        if test_date > datetime.now(): test_date = datetime.now() - timedelta(days=1)
        
        try:
            print(f"  [{processed+1}/{n_samples}] {symbol} @ {test_date.strftime('%Y-%m-%d')}...", end='\r')
            
            hist_end = test_date.strftime('%Y-%m-%d')
            hist_start = (test_date - timedelta(days=365)).strftime('%Y-%m-%d')
            history_dict = fetcher.fetch_stock_history(symbol, start_date=hist_start, end_date=hist_end)
            if not history_dict or 'data' not in history_dict or len(history_dict['data']) < 50: continue
            
            outcome = get_outcome_details(fetcher, symbol, test_date)
            if not outcome: continue
            
            stock_data = {'symbol': symbol, 'current_price': outcome['start_price']}
            is_winner = outcome['is_winner']
            price_change = outcome['price_change']
            
            original_filters = predictor._apply_pro_filters
            from hybrid_predictor import HybridStockPredictor as HSP
            original_weights = HSP._calculate_dynamic_weights
            
            # --- CONFIG 1: BASELINE (Rules, No Filters, No ML) ---
            predictor._apply_pro_filters = lambda f_a, f_c, s_d, h_d, b_a: (f_a, f_c)
            predictor._calculate_dynamic_weights = lambda r, m: {'rule': 1.0, 'ml': 0.0, 'ai': 0.0}
            res_b = predictor.predict(stock_data, history_dict, is_backtest=True, current_date=test_date)
            
            is_buy_b = res_b['recommendation']['action'] == 'BUY'
            correct_b = (is_buy_b and is_winner) or (not is_buy_b and not is_winner)
            stats['baseline']['correct'] += 1 if correct_b else 0
            stats['baseline']['intents'] += 1 if is_buy_b else 0
            stats['baseline']['passed'] += 1 if is_buy_b else 0
            if is_buy_b:
                stats['baseline']['wins'] += 1 if is_winner else 0
                stats['baseline']['profits'].append(price_change)
                if price_change < 0: stats['baseline']['losses'].append(price_change)
                
            # --- CONFIG 2: HARDENED (Rules + Filters, No ML) ---
            predictor._apply_pro_filters = original_filters
            res_h = predictor.predict(stock_data, history_dict, is_backtest=True, current_date=test_date)
            is_buy_h = res_h['recommendation']['action'] == 'BUY'
            correct_h = (is_buy_h and is_winner) or (not is_buy_h and not is_winner)
            stats['hardened']['correct'] += 1 if correct_h else 0
            stats['hardened']['intents'] += 1 if is_buy_b else 0
            if is_buy_h:
                stats['hardened']['passed'] += 1
                stats['hardened']['wins'] += 1 if is_winner else 0
                stats['hardened']['profits'].append(price_change)
                if price_change < 0: stats['hardened']['losses'].append(price_change)
                
            # --- CONFIG 3: FULL SYSTEM (Rules + ML + Filters) ---
            predictor._calculate_dynamic_weights = original_weights.__get__(predictor, HSP)
            res_f = predictor.predict(stock_data, history_dict, is_backtest=True, current_date=test_date)
            is_buy_f = res_f['recommendation']['action'] == 'BUY'
            correct_f = (is_buy_f and is_winner) or (not is_buy_f and not is_winner)
            stats['full_system']['correct'] += 1 if correct_f else 0
            stats['full_system']['intents'] += 1 if is_buy_b else 0 
            if is_buy_f:
                stats['full_system']['passed'] += 1
                stats['full_system']['wins'] += 1 if is_winner else 0
                stats['full_system']['profits'].append(price_change)
                if price_change < 0: stats['full_system']['losses'].append(price_change)
            
            processed += 1
            if processed % 50 == 0: 
                print(f"\n  >>> MILESTONE: {processed}/{n_samples} complete")
            
        except Exception: continue

    # Report
    print("\n" + "="*90)
    print(f"{'Metric':<25} | {'Baseline':<18} | {'Hardened':<18} | {'Full System':<18}")
    print("-" * 90)
    for key in ['Overall Accuracy', 'Signals Generated', 'Signals Passed', 'Win Rate (Passed)', 'Avg Win (3% Tgt)', 'Avg Loss (-4% Stop)', 'Net Expectancy', 'Total Return']:
        row = f"{key:<25} | "
        for cfg in ['baseline', 'hardened', 'full_system']:
            s = stats[cfg]
            wins = [p for p in s['profits'] if p >= 0]
            losses = [p for p in s['profits'] if p < 0]
            
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            win_rate = (s['wins']/s['passed']*100) if s['passed'] > 0 else 0
            expectancy = ((win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss))
            
            if key == 'Overall Accuracy': val = f"{(s['correct']/n_samples*100):.1f}%"
            elif key == 'Signals Generated': val = f"{s['intents']}"
            elif key == 'Signals Passed': val = f"{s['passed']}"
            elif key == 'Win Rate (Passed)': val = f"{win_rate:.1f}%"
            elif key == 'Avg Win (3% Tgt)': val = f"{avg_win:.2f}%"
            elif key == 'Avg Loss (-4% Stop)': val = f"{avg_loss:.2f}%"
            elif key == 'Net Expectancy': val = f"{expectancy:+.2f}%"
            elif key == 'Total Return': val = f"{np.sum(s['profits']):.1f}%"
            row += f"{val:<18} | "
        print(row)
    print("=" * 90 + "\n")

if __name__ == '__main__':
    run_publication_audit(1000)
