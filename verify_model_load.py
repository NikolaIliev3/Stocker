import sys
import logging
from pathlib import Path
from ml_training import StockPredictionML

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def verify():
    print("Testing model load...")
    try:
        # Initialize
        data_dir = Path.home() / ".stocker"
        ml = StockPredictionML(data_dir, "trading")
        
        # Check initial state
        print(f"Initial is_trained: {ml.is_trained()}")
        
        # Force load
        ml._load_model()
        print(f"Post-load is_trained: {ml.is_trained()}")
        
        if ml.is_trained():
            print("SUCCESS: Model loaded.")
            if ml.use_regime_models:
                print(f"Loaded regimes: {list(ml.regime_models.keys())}")
        else:
            print("FAILURE: Model still says not trained.")
            
    except Exception as e:
        print(f"CRASH: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
