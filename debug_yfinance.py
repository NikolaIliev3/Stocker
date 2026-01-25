
import yfinance as yf
import pandas as pd
import requests

print(f"YFinance Version: {yf.__version__}")

def test_ticker_history():
    print("\n--- Test 1: Ticker.history('SPY') ---")
    try:
        t = yf.Ticker('SPY')
        h = t.history(period='5d')
        print(f"Result: {len(h)} rows")
        if not h.empty: print(h.tail(1))
    except Exception as e:
        print(f"Failed: {e}")

def test_download():
    print("\n--- Test 2: yf.download('SPY') ---")
    try:
        h = yf.download('SPY', period='5d', progress=False)
        print(f"Result: {len(h)} rows")
        if not h.empty: print(h.tail(1))
    except Exception as e:
        print(f"Failed: {e}")

def test_with_custom_session():
    print("\n--- Test 3: Custom Session with User-Agent ---")
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        
        # Method A: Ticker with session (if supported in this version, usually passed implicitly or globally?)
        # yfinance doesn't easily take a session object in Ticker directly in all versions, 
        # but unlikely to help if yf internal logic is broken. 
        # However, usually yfinance uses requests.
        
        print("Attempting to use custom headers workaround via yfinance override...")
        # Simplest way: Download with specific request
        pass
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    test_ticker_history()
    test_download()
