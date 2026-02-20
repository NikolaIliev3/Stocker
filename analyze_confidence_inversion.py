import os
import json
import logging
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta
from data_fetcher import StockDataFetcher
from hybrid_predictor import HybridStockPredictor
from config import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("confidence_audit")

def analyze_inversion(samples=50):
    fetcher = StockDataFetcher()
    data_dir = Path("c:/Users/Никола/Desktop/Coding/Stocker")
    predictor = HybridStockPredictor(data_dir, strategy='sniper', data_fetcher=fetcher)
    
    # Symbols from verify_backtest which we know have signals
    symbols = ['AAPL', 'MSFT', 'NVDA', 'AMZN', 'META', 'GOOGL', 'GOOG', 'AVGO', 'TSLA', 'COST', 
               'AMD', 'NFLX', 'QCOM', 'INTC', 'MU', 'CRM', 'ORCL', 'V', 'MA']
    
    results = []
    
    logger.info(f"🔍 Starting Confidence Inversion Audit ({samples} target samples)...")
    
    # Instead of purely random, let's sweep chunks of 2025
    # This ensures we don't miss "hot" regimes.
    start_date = datetime(2025, 1, 10) # start later for history buffer
    end_date = datetime(2025, 12, 1)
    
    current_test_date = start_date
    count = 0
    
    symbols_cycle = 0
    
    while count < samples and current_test_date < end_date:
        symbol = symbols[symbols_cycle % len(symbols)]
        symbols_cycle += 1
        
        # Move forward a few days if we cycle through all symbols
        if symbols_cycle % len(symbols) == 0:
            current_test_date += timedelta(days=2) # Scan every 2 days to find density
            
        if current_test_date.weekday() >= 5: # Skip weekends
            continue

        try:
            # 1. Fetch data
            history_dict = fetcher.fetch_stock_history(symbol, period='2y')
            if not history_dict or 'data' not in history_dict:
                continue
            
            df_full = pd.DataFrame(history_dict['data'])
            df_full['Date'] = pd.to_datetime(df_full['Date']).dt.tz_localize(None)
            
            # Use data as of current_test_date
            mask = df_full['Date'] <= current_test_date
            if not any(mask): continue
            
            current_row = df_full[mask].iloc[-1]
            stock_data = {
                'symbol': symbol,
                'price': float(current_row['close']),
                'change': 0,
                'change_percent': 0,
                'volume': int(current_row['volume'])
            }
            
            # 2. Get prediction
            # is_backtest=True allows us to see the original >80% confidence
            pred_result = predictor.predict(stock_data, history_dict, is_backtest=True, current_date=current_test_date)
            rec = pred_result.get('recommendation', {})
            action = rec.get('action', 'HOLD')
            confidence = rec.get('confidence', 0)
            
            if action == 'BUY' and confidence >= 70: # Lower for broader sweep
                # 3. VERIFY OUTCOME
                window_end = current_test_date + timedelta(days=14)
                future_data = df_full[(df_full['Date'] > current_test_date) & (df_full['Date'] <= window_end)]
                
                if len(future_data) < 5:
                    continue
                    
                entry_price = float(current_row['close'])
                max_high = future_data['high'].max()
                is_win = max_high >= entry_price * 1.03
                
                indicators = pred_result.get('base_analysis', {}).get('indicators', {})
                volatility = pred_result.get('base_analysis', {}).get('volatility', {})
                
                results.append({
                    'symbol': symbol,
                    'date': current_test_date.strftime('%Y-%m-%d'),
                    'confidence': confidence,
                    'is_win': is_win,
                    'rsi': indicators.get('rsi_14', 50),
                    'atr_pct': volatility.get('atr_percent', 0),
                    'dist_sma50': ((entry_price / indicators.get('sma_50', entry_price)) - 1) * 100 if indicators.get('sma_50') else 0,
                    'dist_sma200': ((entry_price / indicators.get('sma_200', entry_price)) - 1) * 100 if indicators.get('sma_200') else 0,
                    'volume_ratio': indicators.get('volume_sma_ratio', 1.0)
                })
                
                count += 1
                if count % 5 == 0:
                    logger.info(f"  ✅ Captured {count}/{samples} signals (Targeting inversion patterns)...")
                    
        except Exception:
            continue

    if not results:
        logger.error("❌ No valid signals found. Possible data alignment issue.")
        return

    # --- AGGREGATE ANALYSIS ---
    df_res = pd.DataFrame(results)
    
    bin_70_80 = df_res[(df_res['confidence'] >= 70) & (df_res['confidence'] < 80)]
    bin_80_plus = df_res[df_res['confidence'] >= 80]
    
    logger.info(f"\n{'='*60}")
    logger.info("CONFIDENCE INVERSION AUDIT RESULTS")
    logger.info(f"{'='*60}")
    
    for name, group in [("70-80% Zone", bin_70_80), ("80-90% Zone", bin_80_plus)]:
        if group.empty:
            continue
            
        win_rate = group['is_win'].mean() * 100
        logger.info(f"\n--- {name} ---")
        logger.info(f"  Signals: {len(group)}")
        logger.info(f"  Win Rate: {win_rate:.1f}%")
        
        winners = group[group['is_win'] == True]
        losers = group[group['is_win'] == False]
        
        logger.info(f"\n  Feature Heatmap (Losers vs Winners):")
        for feat in ['rsi', 'atr_pct', 'dist_sma50', 'dist_sma200', 'volume_ratio']:
            w_avg = winners[feat].mean() if not winners.empty else 0
            l_avg = losers[feat].mean() if not losers.empty else 0
            delta = l_avg - w_avg
            logger.info(f"    {feat:<12} | Win: {w_avg:>6.2f} | Loss: {l_avg:>6.2f} | Delta: {delta:>+6.2f}")

    if not bin_80_plus.empty:
        b80_losers = bin_80_plus[bin_80_plus['is_win'] == False]
        if not b80_losers.empty:
            logger.info(f"\n{'='*60}")
            logger.info("DIAGNOSTIC HYPOTHESES (Failures at High Conviction)")
            logger.info(f"{'='*60}")
            avg_rsi = b80_losers['rsi'].mean()
            avg_dist = b80_losers['dist_sma50'].mean()
            if avg_rsi > 70: logger.info(f"❗ [OVERBOUGHT]: Avg RSI {avg_rsi:.1f}. Model is chasing late exits.")
            if avg_dist > 12: logger.info(f"❗ [STRETCHED]: {avg_dist:.1f}% above SMA50. Regression risk ignored.")

if __name__ == "__main__":
    analyze_inversion(samples=20)
