
import yfinance as yf
import ssl
import os

# Bypass SSL
ssl._create_default_https_context = ssl._create_unverified_context

# Also try to set verify=False for requests if yfinance uses it
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

print("Fetching MSFT info with SSL bypass...")
try:
    msft = yf.Ticker("MSFT")
    hist = msft.history(period="1d")
    if not hist.empty:
        print(f"Price: {hist['Close'].iloc[-1]}")
    else:
        print("Empty history")
except Exception as e:
    print(f"Error: {e}")
