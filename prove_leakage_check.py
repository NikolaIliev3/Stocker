
import sys
import logging
import tkinter as tk
import pandas as pd
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from main import StockerApp

# Configure logging to console
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger('LeakageProof')

def run_proof():
    root = tk.Tk()
    root.withdraw()
    
    try:
        app = StockerApp(root)
        backtester = app.backtester
        hybrid = app.hybrid_predictors['trading']
        
        # 1. INSPECT METADATA
        print("\n" + "="*60)
        print("🕵️  METADATA INSPECTION")
        print("="*60)
        
        if hasattr(hybrid, 'ml_predictor') and hybrid.ml_predictor:
            meta = hybrid.ml_predictor.metadata
            train_end = meta.get('training_data_end', 'UNKNOWN')
            print(f"✅  Model Training End Date: {train_end}")
            
            if train_end == 'UNKNOWN':
                print("❌  CRITICAL: Metadata missing 'training_data_end'")
                return
        else:
             print("❌  CRITICAL: ML Predictor not loaded")
             return

        # 2. TEST CASE A: LEAKAGE (2025)
        print("\n" + "="*60)
        print("🧪  TEST CASE A: INTENTIONAL LEAKAGE (2025-06-15)")
        print("    [Expect 'DATA LEAKAGE DETECTED' warning]")
        print("="*60)
        
        test_date_bad = datetime(2025, 6, 15)
        sym = backtester.test_symbols[0]
        
        # We manually invoke the check logic or run the test
        # We'll run the actual test function to see the log
        backtester._test_historical_point(sym, test_date_bad, 'trading', hybrid, test_number=0)
        
        # 3. TEST CASE B: CLEAN OOS (2026)
        print("\n" + "="*60)
        print("🧪  TEST CASE B: CLEAN OUT-OF-SAMPLE (2026-01-15)")
        print("    [Expect NO warning]")
        print("="*60)
        
        test_date_good = datetime(2026, 1, 15)
        backtester._test_historical_point(sym, test_date_good, 'trading', hybrid, test_number=0)
        
        print("\n" + "="*60)
        print("🏁  PROOF COMPLETE")
        print("="*60)

    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        try:
            if app.backend_process: app.backend_process.terminate()
            root.destroy()
        except: pass

if __name__ == "__main__":
    run_proof()
