import yfinance as yf
print("Testing SIE.DE with yfinance...")
try:
    ticker = yf.Ticker("SIE.DE")
    hist = ticker.history(period="1mo")
    print(f"History shape: {hist.shape}")
    if not hist.empty:
        print(hist.head())
    else:
        print("History is empty")
    
    print("\nTesting SIE-DE (normalized)...")
    ticker2 = yf.Ticker("SIE-DE")
    hist2 = ticker2.history(period="1mo")
    print(f"History shape: {hist2.shape}")
except Exception as e:
    print(f"Error: {e}")
