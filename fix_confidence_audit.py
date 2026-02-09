
import sys
import time
import logging
import tkinter as tk
import pandas as pd
import numpy as np
import random
import threading
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from main import StockerApp
from hybrid_predictor import HybridStockPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('ConfidenceAudit')

# MONKEYPATCH: Allow all signals to measure full curve
# We want to see what the system *would* do if we let it trade, to measure confidence quality.
def bypassed_pro_filters(self, final_action: str, final_confidence: float, 
                      stock_data: dict, history_data: dict, base_analysis: dict = None) -> tuple:
    return final_action, final_confidence

HybridStockPredictor._apply_pro_filters = bypassed_pro_filters

def run_confidence_profiling():
    root = tk.Tk()
    root.withdraw()
    
    try:
        app = StockerApp(root)
        backtester = app.backtester
        
        symbols = backtester.test_symbols 
        num_tests = 300 
        strategy = 'trading'
        hybrid_predictor = app.hybrid_predictors.get(strategy)
        
        results = []
        
        print(f"Generating {num_tests} test points (2026 Strict Out-of-Sample)...")
        print("Sampling across Full Universe...")
        
        for i in range(num_tests):
            sym = random.choice(symbols)
            start_date = datetime(2026, 1, 1)
            end_date = datetime.now() - timedelta(days=5) # Ensure recent data needs
            
            days_range = (end_date - start_date).days
            if days_range <= 0: continue
            
            test_date = start_date + timedelta(days=random.randint(0, days_range))
            
            # Run test
            try:
                res = backtester._test_historical_point(sym, test_date, strategy, hybrid_predictor)
                
                if res and res.get('predicted_action') != 'HOLD':
                    conf = res.get('confidence', 0) # FIX: Use correct key from continuous_backtester
                    was_correct = res.get('was_correct', False)
                    action = res.get('predicted_action')
                    
                    # Ensure float
                    try:
                        conf = float(conf)
                    except:
                        conf = 0.0
                        
                    results.append({
                        'confidence': conf,
                        'correct': was_correct,
                        'action': action,
                        'symbol': sym
                    })
            except Exception as e:
                pass # Skip failed individual tests
                
            if len(results) % 50 == 0 and len(results) > 0:
                print(f"Collected {len(results)} valid predictions...")
                
        # Analysis
        if not results:
            print("No active predictions found.")
            return

        df = pd.DataFrame(results)
        
        print("\n" + "="*80)
        print("🎯 ACCURACY BY CONFIDENCE LEVEL (2026 Strict)")
        print("="*80)
        
        # Define ranges
        ranges = [
            (50, 60),
            (60, 70),
            (70, 75),
            (75, 80),
            (80, 85),
            (85, 90),
            (90, 101)
        ]
        
        print(f"{'Range':<12} | {'Count':<8} | {'Accuracy':<10} | {'Wins/Total':<15}")
        print("-" * 60)
        
        for low, high in ranges:
            mask = (df['confidence'] >= low) & (df['confidence'] < high)
            subset = df[mask]
            count = len(subset)
            if count > 0:
                wins = subset['correct'].sum()
                acc = (wins / count) * 100
                label = f"{low}-{high}%"
                print(f"{label:<12} | {count:<8} | {acc:.1f}%     | {wins}/{count}")
            else:
                label = f"{low}-{high}%"
                print(f"{label:<12} | 0        | N/A        | 0/0")
                
        # Specifically check user's claim: > 75%
        high_conf = df[df['confidence'] >= 75]
        cnt = len(high_conf)
        if cnt > 0:
            w = high_conf['correct'].sum()
            ac = (w / cnt) * 100
            print("-" * 60)
            print(f"> 75% Total  | {cnt:<8} | {ac:.1f}%     | {w}/{cnt}")
        else:
            print("-" * 60)
            print(f"> 75% Total  | 0        | N/A        | 0/0")
            
        print("="*80 + "\n")
        
        # Dump high confidence failures to see WHY
        fails = high_conf[high_conf['correct'] == False]
        if not fails.empty:
            print("\n❌ High Confidence Failures (>75%):")
            for _, row in fails.head(10).iterrows():
                print(f"   • {row['symbol']} {row['action']} @ {row['confidence']:.1f}%")

    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        try:
            if app.backend_process: app.backend_process.terminate()
            root.destroy()
        except: pass

if __name__ == "__main__":
    run_confidence_profiling()
