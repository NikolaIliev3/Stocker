from data_fetcher import StockDataFetcher
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_fetch():
    fetcher = StockDataFetcher()
    symbol = 'AAPL'
    start = '2024-01-01'
    end = '2024-02-01'
    
    print(f"Fetching {symbol} from {start} to {end}...")
    result = fetcher.fetch_stock_history(symbol, start_date=start, end_date=end, interval='1d')
    
    if result and 'data' in result and result['data']:
        print(f"Success! Got {len(result['data'])} rows.")
        print("First row:", result['data'][0])
    else:
        print("FAILED:", result)

if __name__ == "__main__":
    test_fetch()
