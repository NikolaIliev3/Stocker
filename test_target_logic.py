
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# FORCE UTF-8 OUTPUT FOR WINDOWS
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from data_fetcher import StockDataFetcher
from hybrid_predictor import HybridStockPredictor
from trading_analyzer import TradingAnalyzer
from config import APP_DATA_DIR

# Configure logging to see the TGT TRACE
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('TestPredict')

def test_target_logic(symbol):
    print(f"\n--- Testing Target Logic for {symbol} ---")
    fetcher = StockDataFetcher()
    analyzer = TradingAnalyzer(fetcher)
    predictor = HybridStockPredictor(APP_DATA_DIR, 'trading', trading_analyzer=analyzer)
    
    # Fetch data
    stock_data = fetcher.fetch_stock_data(symbol)
    history = fetcher.fetch_stock_history(symbol, period='1y')
    
    # Analyze Regime
    print("\n--- Regime Check ---")
    regime = analyzer._get_market_regime()
    print(f"Regime: {regime}")
    
    # Run prediction
    result = predictor.predict(stock_data, history)
    
    print(f"Action: {result.get('action')}")
    print(f"Confidence: {result.get('confidence', 0):.1f}%")
    print(f"Current Price: ${result.get('price', 0):.2f}")
    print(f"Target Price: ${result.get('target_price', 0):.2f}")
    
    if result.get('price', 0) > 0:
        expected_pct = ((result['target_price'] - result['price']) / result['price']) * 100
        print(f"Expeced Profit: {expected_pct:.2f}%")
    
    print("\nReasoning Extract:")
    print(result.get('reasoning', 'No reasoning provided.'))

if __name__ == "__main__":
    test_target_logic('XLF')
    test_target_logic('BA')
    test_target_logic('GLD')
