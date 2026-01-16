
from momentum_monitor import MomentumMonitor
from pathlib import Path

def test_logic():
    print("Testing Momentum Logic Correction...")
    
    monitor = MomentumMonitor(Path("test_logic_data"))
    
    # Test 1: Downtrend -> Uptrend (Should be BUY)
    prev = {'trend': 'downtrend'}
    curr = {'trend': 'uptrend', 'rsi': 50, 'mfi': 50, 'macd_diff': 0, 'momentum_score': 0, 'golden_cross': False, 'death_cross': False}
    
    rec = monitor._determine_recommendation(prev, curr, True, True)
    print(f"Test 1 (Downtrend -> Uptrend): Action={rec['action']} (Expected BUY)")
    
    # Test 2: Golden Cross (Should be BUY)
    prev = {'golden_cross': False}
    curr = {'golden_cross': True, 'trend': 'uptrend', 'rsi': 50}
    
    rec = monitor._determine_recommendation(prev, curr, True, True)
    print(f"Test 2 (Golden Cross): Action={rec['action']} (Expected BUY)")
    
    # Test 3: RSI Overbought (Should be SELL)
    prev = {'trend': 'uptrend'}
    curr = {'trend': 'uptrend', 'rsi': 75}
    
    rec = monitor._determine_recommendation(prev, curr, False, True)
    print(f"Test 3 (RSI > 70): Action={rec['action']} (Expected SELL)")
    
    # Test 4: RSI Oversold (Should be BUY)
    prev = {'trend': 'downtrend'}
    curr = {'trend': 'downtrend', 'rsi': 25}
    
    rec = monitor._determine_recommendation(prev, curr, False, True)
    print(f"Test 4 (RSI < 30): Action={rec['action']} (Expected BUY)")

    # Cleanup
    import shutil
    if Path("test_logic_data").exists():
        shutil.rmtree(Path("test_logic_data"))

if __name__ == "__main__":
    test_logic()
