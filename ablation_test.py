
import sys
import time
import logging
import tkinter as tk
import pandas as pd
import numpy as np
import random
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from main import StockerApp
from hybrid_predictor import HybridStockPredictor

# Configure logging to be quiet
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger('AblationTest')

# GLOBAL TOGGLE FOR MONKEYPATCH
FORCE_RULES_ONLY = False

# MONKEYPATCH: Allow all signals to measure full curve
def bypassed_pro_filters(self, final_action: str, final_confidence: float, 
                      stock_data: dict, history_data: dict, base_analysis: dict = None) -> tuple:
    return final_action, final_confidence

# MONKEYPATCH: Control weights
original_calc_weights = HybridStockPredictor._calculate_dynamic_weights

def controlled_calc_weights(self, rule_pred, ml_pred):
    if FORCE_RULES_ONLY:
        return {'rule': 1.0, 'ml': 0.0, 'ai': 0.0}
    return original_calc_weights(self, rule_pred, ml_pred)

HybridStockPredictor._apply_pro_filters = bypassed_pro_filters
HybridStockPredictor._calculate_dynamic_weights = controlled_calc_weights

def run_ablation_test():
    global FORCE_RULES_ONLY
    
    root = tk.Tk()
    root.withdraw()
    root.after = MagicMock()
    
    try:
        print("Initializing Stocker Engine...")
        app = StockerApp(root)
        
        if hasattr(app, 'ml_scheduler') and app.ml_scheduler:
            app.ml_scheduler.stop()
            
        backtester = app.backtester
        symbols = backtester.test_symbols 
        num_tests = 50 
        strategy = 'trading'
        hybrid_predictor = app.hybrid_predictors.get(strategy)
        
        hybrid_results = []
        rules_results = []
        
        print(f"RUNNING ABLATION TEST ({num_tests} points)...")
        
        test_points = []
        start_date = datetime(2026, 1, 1)
        end_date = datetime.now() - timedelta(days=5)
        days_range = (end_date - start_date).days
        
        if days_range <= 0:
            print("Error: Invalid date range.")
            return

        print("Sampling points...")
        for _ in range(num_tests):
            sym = random.choice(symbols)
            test_date = start_date + timedelta(days=random.randint(0, days_range))
            test_points.append((sym, test_date))

        # Backup original weights
        original_weights = hybrid_predictor.ensemble_weights.copy()
        
        for i, (sym, test_date) in enumerate(test_points):
            try:
                # 1. HYBRID RUN
                FORCE_RULES_ONLY = False
                res_hybrid = backtester._test_historical_point(sym, test_date, strategy, hybrid_predictor)
                
                if res_hybrid and res_hybrid.get('predicted_action') != 'HOLD':
                    hybrid_results.append({
                        'confidence': float(res_hybrid.get('confidence', 0)),
                        'correct': bool(res_hybrid.get('was_correct', False))
                    })
                
                # 2. RULES ONLY RUN
                FORCE_RULES_ONLY = True
                res_rules = backtester._test_historical_point(sym, test_date, strategy, hybrid_predictor)
                
                if res_rules and res_rules.get('predicted_action') != 'HOLD':
                    rules_results.append({
                        'confidence': float(res_rules.get('confidence', 0)),
                        'correct': bool(res_rules.get('was_correct', False))
                    })
                
            except Exception as e:
                continue
                
            if (i+1) % 50 == 0:
                print(f"Progress: {i+1}/{num_tests} points ({len(hybrid_results)} Hybrid / {len(rules_results)} Rules signals)")

        def get_metrics(results):
            if not results: return 0, 0, 0, 0
            df = pd.DataFrame(results)
            count = len(df)
            acc = (df['correct'].sum() / count * 100) if count > 0 else 0
            
            high = df[df['confidence'] >= 75]
            h_count = len(high)
            h_acc = (high['correct'].sum() / h_count * 100) if h_count > 0 else 0
            return count, acc, h_count, h_acc

        h_count, h_acc, h_h_count, h_h_acc = get_metrics(hybrid_results)
        r_count, r_acc, r_h_count, r_h_acc = get_metrics(rules_results)

        print("\n" + "="*60)
        print("ABLATION FINAL VERDICT")
        print("="*60)
        print(f"{'Metric':<25} | {'Hybrid (ML+Rules)':<18} | {'Rules Only':<15}")
        print("-" * 60)
        print(f"{'Total Signals':<25} | {h_count:<18} | {r_count:<15}")
        print(f"{'Overall Accuracy':<25} | {h_acc:>17.1f}% | {r_acc:>14.1f}%")
        print(f"{'High-Conf (>75%) Count':<25} | {h_h_count:<18} | {r_h_count:<15}")
        print(f"{'High-Conf Accuracy':<25} | {h_h_acc:>17.1f}% | {r_h_acc:>14.1f}%")
        print("-" * 60)
        
        acc_edge = h_acc - r_acc
        h_acc_edge = h_h_acc - r_h_acc
        
        print(f"ML Edge (Accuracy): {acc_edge:+.2f}pp")
        print(f"ML Edge (High-Conf): {h_acc_edge:+.2f}pp")
        
        if h_acc > r_acc + 0.5:
            print("\nRESULT: ML provides a measurable performance boost.")
        elif h_acc < r_acc - 0.5:
            print("\nRESULT: ML is actively degrading Rule performance.")
        else:
            print("\nRESULT: ML contribution is within noise margin.")
        print("="*60 + "\n")

    except Exception as e:
        print(f"Fatal Error: {e}")
    finally:
        try:
            # Restore predictor weights
            hybrid_predictor.ensemble_weights = original_weights
        except: pass
        
        try:
            if app.backend_process: app.backend_process.terminate()
            root.destroy()
        except: pass

if __name__ == "__main__":
    run_ablation_test()
