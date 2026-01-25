from pathlib import Path
from ml_training import StockPredictionML
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_loading():
    data_dir = Path("C:/Users/Никола/.stocker")
    logger.info(f"Initializing ML system from {data_dir}...")
    
    ml = StockPredictionML(data_dir, "trading")
    
    # Check initial state
    is_trained = ml.is_trained()
    logger.info(f"Initial is_trained(): {is_trained}")
    
    # Check internal state
    logger.info(f"Main model exists: {ml.model is not None}")
    logger.info(f"Regime models: {list(ml.regime_models.keys()) if ml.regime_models else 'None'}")
    
    # Attempt manual load
    if not is_trained:
        logger.info("Attempting manual _load_model()...")
        result = ml._load_model()
        logger.info(f"Load result: {result}")
        
        # Check state after load
        is_trained_after = ml.is_trained()
        logger.info(f"Post-load is_trained(): {is_trained_after}")
        logger.info(f"Post-load Main model: {ml.model is not None}")
        
    # Check file existence
    model_path = data_dir / "ml_models" / "trading_model.pkl"
    logger.info(f"Model file exists: {model_path.exists()}")
    if model_path.exists():
        logger.info(f"Model file size: {model_path.stat().st_size} bytes")

if __name__ == "__main__":
    test_model_loading()
