
import sys
import os
import logging
import importlib
import traceback
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("SystemCheck")

def check_syntax(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        compile(source, file_path, 'exec')
        logger.info(f"✅ Syntax OK: {os.path.basename(file_path)}")
        return True
    except Exception as e:
        logger.error(f"❌ Syntax Error in {os.path.basename(file_path)}: {e}")
        return False

def check_imports():
    logger.info("Checking critical libraries...")
    try:
        import pandas
        logger.info(f"✅ pandas {pandas.__version__}")
        import numpy
        logger.info(f"✅ numpy {numpy.__version__}")
        import sklearn
        logger.info(f"✅ sklearn {sklearn.__version__}")
        import yfinance
        logger.info(f"✅ yfinance {yfinance.__version__}")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing library: {e}")
        return False

def check_structure_and_config():
    logger.info("Checking project structure and config...")
    try:
        import config
        
        # Verify Goldilocks settings
        if config.ML_RF_MAX_DEPTH == 6 and config.ML_RF_N_ESTIMATORS == 100:
             logger.info("✅ Config loaded and verified (Goldilocks settings apply)")
        else:
             logger.warning(f"⚠️ Config values mismatch: Depth={config.ML_RF_MAX_DEPTH}, Est={config.ML_RF_N_ESTIMATORS} (Expected 6, 100)")
             
        data_dir = getattr(config, 'APP_DATA_DIR', Path.home() / ".stocker")
        if not os.path.exists(data_dir):
            logger.warning(f"⚠️ Data directory missing: {data_dir} (Will be created on run)")
        else:
            logger.info(f"✅ Data directory exists: {data_dir}")
            
        return True
    except Exception as e:
        logger.error(f"❌ Config check failed: {e}")
        return False

def check_component_initialization():
    logger.info("Testing component initialization...")
    try:
        from pathlib import Path
        import config
        data_dir = getattr(config, 'APP_DATA_DIR', Path.home() / ".stocker")
        
        # 1. Hybrid Predictor (Complex init)
        from hybrid_predictor import HybridStockPredictor
        hp = HybridStockPredictor(data_dir, "trading")
        logger.info("✅ HybridStockPredictor initialized")
        
        # 2. ML Training (Feature Selection check)
        from ml_training import StockPredictionML
        ml = StockPredictionML(data_dir, "trading")
        logger.info("✅ StockPredictionML initialized")
        
        return True
    except Exception as e:
        logger.error(f"❌ Component Init Error: {e}")
        logger.error(traceback.format_exc())
        return False

def check_connectivity():
    logger.info("Testing data connectivity...")
    try:
        import yfinance as yf
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="1d")
        if not hist.empty:
            logger.info("✅ Data Fetching OK (yfinance connected)")
            return True
        else:
            logger.warning("⚠️ Data Fetching returned empty data for AAPL")
            return False
    except Exception as e:
        logger.error(f"❌ Connectivity Error: {e}")
        return False

def main():
    print("="*40)
    print(" STOCKER SYSTEM DIAGNOSTICS")
    print("="*40)
    
    # 1. Syntax Checks of Modified Files
    files_to_check = [
        "config.py",
        "ml_training.py",
        "hybrid_predictor.py",
        "smart_model_selector.py"
    ]
    
    syntax_ok = True
    for f in files_to_check:
        if os.path.exists(f):
            if not check_syntax(f):
                syntax_ok = False
        else:
            logger.warning(f"File not found: {f}")

    if not syntax_ok:
        logger.error("🛑 Aborting tests due to syntax errors.")
        return

    # 2. Imports
    if not check_imports(): return

    # 3. Config
    if not check_structure_and_config(): return

    # 4. Initialization
    if not check_component_initialization(): return

    # 5. Connectivity
    if not check_connectivity(): return

    print("\n" + "="*40)
    print("✅ GLOBAL SYSTEM STATUS: GREEN")
    print("   Ready for Training & Backtesting")
    print("="*40)

if __name__ == "__main__":
    main()
