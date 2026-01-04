"""
Quick test script to see if we can get yfinance working
Run this to diagnose the SSL issue
"""
import os
import ssl
import certifi
import yfinance as yf

print("Testing yfinance SSL configuration...")
print()

# Try different SSL configurations
configs = [
    ("Default", {}),
    ("Disable SSL context", {"ssl_context": ssl._create_unverified_context()}),
]

for name, config in configs:
    print(f"Trying: {name}")
    try:
        # Set environment
        cert_path = certifi.where()
        if os.path.exists(cert_path):
            os.environ['CURL_CA_BUNDLE'] = cert_path
            os.environ['REQUESTS_CA_BUNDLE'] = cert_path
            print(f"  Certificate path: {cert_path}")
        
        # Try to fetch data
        ticker = yf.Ticker("AAPL")
        hist = ticker.history(period="5d")
        
        if not hist.empty:
            print(f"  ✓ SUCCESS! Got {len(hist)} days of data")
            print(f"  Latest price: ${hist['Close'].iloc[-1]:.2f}")
            break
        else:
            print("  ✗ No data returned")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    print()

print("\nIf all failed, the issue is with curl/SSL certificates.")
print("The workaround in backend_proxy.py should handle this.")


