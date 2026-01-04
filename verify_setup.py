"""
Quick verification script to check if all dependencies are installed correctly
"""
import sys

def check_imports():
    """Check if all required modules can be imported"""
    errors = []
    
    modules = [
        ('requests', 'requests'),
        ('yfinance', 'yfinance'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('ta', 'ta'),
        ('flask', 'flask'),
        ('flask_cors', 'flask-cors'),
        ('cryptography', 'cryptography'),
    ]
    
    print("Checking dependencies...")
    for module_name, package_name in modules:
        try:
            __import__(module_name)
            print(f"✓ {package_name}")
        except ImportError:
            print(f"✗ {package_name} - NOT INSTALLED")
            errors.append(package_name)
    
    # Check local modules
    print("\nChecking local modules...")
    local_modules = [
        'config',
        'security',
        'data_fetcher',
        'trading_analyzer',
        'investing_analyzer',
        'chart_generator',
        'portfolio',
    ]
    
    for module_name in local_modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name}.py")
        except ImportError as e:
            print(f"✗ {module_name}.py - ERROR: {e}")
            errors.append(module_name)
    
    if errors:
        print(f"\n❌ Setup incomplete. Missing: {', '.join(errors)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies installed correctly!")
        return True

if __name__ == "__main__":
    success = check_imports()
    sys.exit(0 if success else 1)

