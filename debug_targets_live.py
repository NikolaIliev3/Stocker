
import logging
import sys
import os
from config import APP_DATA_DIR

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("debug_targets")

from market_scanner import MarketScanner
from data_fetcher import StockDataFetcher

def run_debug():
    print("--- Debugging Targets for XLF, INTC, GLD ---")
    fetcher = StockDataFetcher()
    scanner = MarketScanner(data_fetcher=fetcher, data_dir=APP_DATA_DIR)
    
    tickers = ['XLF', 'INTC', 'GLD', 'SBUX']
    results = scanner.scan_market(strategy='trading', custom_tickers=tickers)
    
    print("\n--- Summary ---")
    for r in results:
        curr = r['price']
        tgt = r['target_price']
        gain = ((tgt - curr) / curr) * 100
        print(f"{r['symbol']}: Price ${curr:.2f} -> Target ${tgt:.2f} ({gain:.2f}%)")

if __name__ == "__main__":
    run_debug()
