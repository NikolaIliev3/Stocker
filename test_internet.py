
import requests
import ssl

try:
    print("Testing connectivity to google.com...")
    r = requests.get("https://www.google.com", timeout=5, verify=False)
    print(f"Status: {r.status_code}")
    
    print("Testing connectivity to yahoo.com...")
    r = requests.get("https://finance.yahoo.com", timeout=5, verify=False)
    print(f"Status: {r.status_code}")
except Exception as e:
    print(f"Error: {e}")
