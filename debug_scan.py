
import logging
import sys
import os

# Setup logging to console
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("market_scanner")
logger.setLevel(logging.INFO)

# Add project root
sys.path.append(os.getcwd())

from market_scanner import MarketScanner
from data_fetcher import StockDataFetcher

def run_debug_scan():
    print("--- Starting Debug Scan ---")
    fetcher = StockDataFetcher()
    scanner = MarketScanner(data_fetcher=fetcher, data_dir=None) # No data_dir = basic analyzers, wait... 
    # WE NEED DATA_DIR to use HybridPredictor! Otherwise it uses basic analyzers which might lack the new logic?
    # Let's check: MarketScanner line 101: "Could not initialize hybrid predictors... using basic analyzers"
    # The new logic is in HybridPredictor. If I don't pass data_dir, I get basic logic (TradingAnalyzer).
    # Does TradingAnalyzer have the new logic? NO!
    
    # Use REAL data dir to load ML models
    from config import APP_DATA_DIR
    data_dir = APP_DATA_DIR
    
    scanner = MarketScanner(data_fetcher=fetcher, data_dir=data_dir)
    
    # Scan a mix of stocks
    tickers = ['TSLA', 'NVDA', 'KO', 'MCD']
    print(f"Scanning: {tickers}")
    
    results = scanner.scan_market(strategy='trading', custom_tickers=tickers)
    
    print("\n--- Results ---")
    for r in results:
        curr = r['price']
        tgt = r['target_price']
        gain = ((tgt - curr) / curr) * 100
        print(f"{r['symbol']}: Price ${curr:.2f} -> Target ${tgt:.2f} (+{gain:.2f}%) | Conf: {r['confidence']:.1f}%")

if __name__ == "__main__":
    run_debug_scan()
