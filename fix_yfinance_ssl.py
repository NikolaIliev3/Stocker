"""
Script to fix yfinance SSL certificate issues
Run this once to configure yfinance to work with SSL certificates
"""
import ssl
import certifi
import os

print("Fixing yfinance SSL certificate configuration...")

try:
    # Get the certifi certificate path
    cert_path = certifi.where()
    print(f"Certificate path: {cert_path}")
    
    # Verify the file exists
    if os.path.exists(cert_path):
        print("✓ Certificate file exists")
    else:
        print("✗ Certificate file not found!")
        print("Installing certifi...")
        import subprocess
        subprocess.check_call(["pip", "install", "--upgrade", "certifi"])
        cert_path = certifi.where()
        print(f"New certificate path: {cert_path}")
    
    # Test SSL context
    try:
        context = ssl.create_default_context(cafile=cert_path)
        print("✓ SSL context created successfully")
    except Exception as e:
        print(f"✗ SSL context creation failed: {e}")
        print("Trying to reinstall certifi...")
        import subprocess
        subprocess.check_call(["pip", "install", "--force-reinstall", "certifi"])
    
    print("\n✓ SSL configuration check complete!")
    print("You can now use yfinance. If issues persist, try:")
    print("  pip install --upgrade certifi yfinance")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()


