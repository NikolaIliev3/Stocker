import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from data_fetcher import StockDataFetcher
from trading_analyzer import TradingAnalyzer
from hybrid_predictor import HybridStockPredictor
from ml_training import FeatureExtractor

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def debug_prediction():
    data_dir = Path("data")
    
    # Initialize components
    logger.info("Initializing components...")
    fetcher = StockDataFetcher()
    analyzer = TradingAnalyzer(data_fetcher=fetcher)
    predictor = HybridStockPredictor(data_dir, 'trading', trading_analyzer=analyzer)
    
    symbol = "NVDA"
    logger.info(f"Fetching data for {symbol}...")
    
    # Fetch data
    stock_data = fetcher.fetch_stock_data(symbol)
    history = fetcher.fetch_stock_history(symbol, period="2y")
    
    if not stock_data or not history:
        logger.error("Failed to fetch data")
        return

    logger.info(f"Predicting for {symbol}...")
    try:
        result = predictor.predict(stock_data, history)
        
        logger.info("Prediction Result:")
        import json
        logger.info(json.dumps(result, indent=2, default=str))
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    debug_prediction()
