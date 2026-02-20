
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
from config import APP_DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("final_audit_results.log")
    ]
)
logger = logging.getLogger('FinalAudit')

def run_audit_logic(app, root):
    try:
        backtester = app.backtester
        
        # 108 Symbols from heavy_retraining.py
        symbols = [
            'KO', 'GOOGL', 'HD', 'MCD', 'PG', 'META', 'MSFT', 'AMZN', 'MS', 'AAPL', 
            'PEP', 'WMT', 'ABBV', 'AMD', 'DELL', 'SPY', 'QQQ', 'IWM', 'JPM', 'XLE', 
            'NVDA', 'CRM', 'GS', 'PFE', 'CAT', 'DE', 'UPS', 'FDX', 'LOW', 'COST', 
            'XOM', 'CVX', 'XLK', 'XLF', 'XLI', 'XLY', 'V', 'MA', 'BLK', 'LMT', 
            'RTX', 'TSLA', 'JNJ', 'TGT', 'TXN', 'MDLZ', 'SPGI', 'SIE.DE',
            'UNH', 'LLY', 'AVGO', 'ORCL', 'ADBE', 'BAC', 'NFLX', 'DIS', 'NKE', 'SBUX',
            'GE', 'AMGN', 'IBM', 'HON', 'INTC', 'AMAT', 'QCOM', 'MU', 'CSCO', 'ACN',
            'NOW', 'ISRG', 'BKNG', 'ADP', 'T', 'VZ', 'CMCSA', 'CVS', 'EL', 'MO',
            'PM', 'PLD', 'AMT', 'PSA', 'O', 'SPG', 'DLR', 'CB', 'PGR', 'TRV',
            'ALL', 'MET', 'PRU', 'MMC', 'AON', 'SCHW', 'MAR', 'HLT', 'SYY', 'ADM',
            'SO', 'NEE', 'DUK', 'PLTR', 'SNOW', 'MSTR', 'TEAM', 'WDAY', 'DDOG', 'ZS'
        ]
        
        num_tests = 300  # 300 tests for 2026 window
        strategy = 'trading'
        hybrid_predictor = app.hybrid_predictors.get(strategy)
        
        regime_stats = {
            'bull': {'total': 0, 'correct': 0},
            'bear': {'total': 0, 'correct': 0},
            'sideways': {'total': 0, 'correct': 0}
        }
        
        test_points = []
        lookforward = backtester.lookforward_days[strategy]
        
        logger.info(f"Generating {num_tests} test points for 2026 STRICT OOS window...")
        
        for _ in range(num_tests):
            sym = random.choice(symbols)
            # STRICT OUT-OF-SAMPLE: 2026 Only (Training ended Dec 2025)
            start_date = datetime(2026, 1, 1)
            end_date = datetime.now() - timedelta(days=lookforward + 5)
            
            days_range = (end_date - start_date).days
            if days_range <= 0:
                test_date = start_date
            else:
                test_date = start_date + timedelta(days=random.randint(0, days_range))
            test_points.append((sym, test_date))

        logger.info(f"Starting test phase on {len(test_points)} points...")
        
        active_predictions = 0
        
        for sym, t_date in test_points:
            res = backtester._test_historical_point(sym, t_date, strategy, hybrid_predictor)
            
            if res and res.get('predicted_action') != 'HOLD':
                active_predictions += 1
                regime = res.get('market_regime', 'unknown').lower()
                is_correct = res.get('was_correct', False)
                
                if regime in regime_stats:
                    regime_stats[regime]['total'] += 1
                    if is_correct:
                        regime_stats[regime]['correct'] += 1
                
                if active_predictions % 5 == 0:
                    logger.info(f"✓ Active Pred {active_predictions}: {sym} {regime} -> {'CORRECT' if is_correct else 'WRONG'}")

        # Final Report
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 FINAL PERFORMANCE AUDIT REPORT (STRICT 2026 OOS)")
        logger.info(f"{'='*70}")
        
        total_active = 0
        total_correct = 0
        
        for regime, stats in regime_stats.items():
            reg_total = stats['total']
            reg_correct = stats['correct']
            reg_precision = (reg_correct / reg_total * 100) if reg_total > 0 else 0
            
            logger.info(f"   • {regime.upper()} Regime: {reg_precision:.1f}% Precision ({reg_correct}/{reg_total})")
            
            total_active += reg_total
            total_correct += reg_correct
            
        overall_precision = (total_correct / total_active * 100) if total_active > 0 else 0
        logger.info(f"{'-'*70}")
        logger.info(f"OVERALL ACTIVE PRECISION: {overall_precision:.1f}% ({total_correct}/{total_active})")
        logger.info(f"{'='*70}\n")
        
        with open("audit_summary.json", "w") as f:
            import json
            json.dump({
                "overall_precision": overall_precision,
                "regime_stats": regime_stats,
                "total_active": total_active,
                "period": "2026-01-01 to 2026-02-07"
            }, f)

    except Exception as e:
        logger.error(f"Error in audit thread: {e}", exc_info=True)
    finally:
        logger.info("Audit complete. Shutting down...")
        root.quit()

def run_large_audit():
    logger.info(f"\n{'='*70}")
    logger.info(f"🚀 STARTING FINAL REGIME-AWARE AUDIT (300 SAMPLES - 2026 OOS)")
    logger.info(f"{'='*70}\n")
    
    root = tk.Tk()
    root.withdraw()
    
    app = None
    try:
        app = StockerApp(root)
        audit_thread = threading.Thread(target=run_audit_logic, args=(app, root))
        audit_thread.daemon = True
        audit_thread.start()
        root.mainloop()

    except Exception as e:
        logger.error(f"Error in main loop: {e}", exc_info=True)
    finally:
        try:
            if app and hasattr(app, 'backend_process') and app.backend_process:
                app.backend_process.terminate()
            root.destroy()
        except:
            pass

if __name__ == "__main__":
    run_large_audit()
