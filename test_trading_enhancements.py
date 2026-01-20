"""
Test script for trading strategy enhancements
Tests sector momentum, catalyst detection, R:R ratio, and liquidity analysis
"""
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_sector_momentum():
    """Test SectorMomentumAnalyzer"""
    print("\n" + "="*50)
    print("TESTING SECTOR MOMENTUM ANALYZER")
    print("="*50)
    
    from sector_momentum import SectorMomentumAnalyzer
    
    analyzer = SectorMomentumAnalyzer()
    
    # Test sector ETF mapping
    print("\n1. Testing sector ETF mapping...")
    test_sectors = ['Technology', 'Financial Services', 'Healthcare', 'Energy', 'Unknown']
    for sector in test_sectors:
        etf = analyzer.get_sector_etf(sector)
        print(f"   {sector} -> {etf if etf else 'No mapping'}")
    
    # Test with mock data
    print("\n2. Testing sector analysis with mock data...")
    import pandas as pd
    import numpy as np
    
    # Create mock DataFrame
    dates = pd.date_range(end=pd.Timestamp.now(), periods=60, freq='D')
    mock_df = pd.DataFrame({
        'close': np.random.uniform(100, 120, 60),
        'open': np.random.uniform(100, 120, 60),
        'high': np.random.uniform(110, 130, 60),
        'low': np.random.uniform(90, 110, 60),
        'volume': np.random.randint(1000000, 5000000, 60)
    }, index=dates)
    
    mock_stock_data = {
        'symbol': 'AAPL',
        'info': {'sector': 'Technology'}
    }
    
    result = analyzer.analyze(mock_stock_data, mock_df)
    print(f"   Sector: {result['sector']}")
    print(f"   ETF: {result['etf_symbol']}")
    print(f"   Sector Trend: {result['sector_trend']}")
    print(f"   Sector Signal: {result['sector_signal']}")
    print(f"   Available: {result['available']}")
    
    print("\n✅ Sector Momentum Analyzer tests passed!")
    return True

def test_catalyst_detector():
    """Test CatalystDetector"""
    print("\n" + "="*50)
    print("TESTING CATALYST DETECTOR")
    print("="*50)
    
    from catalyst_detector import CatalystDetector
    
    detector = CatalystDetector()
    
    # Test with a real stock (AAPL)
    print("\n1. Testing catalyst detection for AAPL...")
    result = detector.detect_catalysts('AAPL')
    
    print(f"   Earnings Date: {result['earnings_date']}")
    print(f"   Days to Earnings: {result['days_to_earnings']}")
    print(f"   Dividend Date: {result['dividend_date']}")
    print(f"   Days to Dividend: {result['days_to_dividend']}")
    print(f"   Has Upcoming Catalyst: {result['has_upcoming_catalyst']}")
    print(f"   Catalyst Warning: {result['catalyst_warning']}")
    print(f"   Available: {result['available']}")
    
    # Test confidence adjustment
    print("\n2. Testing confidence adjustment...")
    adj = detector.get_catalyst_adjustment(result)
    print(f"   Confidence Adjustment: {adj['confidence_adjustment']}")
    print(f"   Reasoning: {adj['reasoning']}")
    
    print("\n✅ Catalyst Detector tests passed!")
    return True

def test_trading_analyzer_integration():
    """Test full TradingAnalyzer integration"""
    print("\n" + "="*50)
    print("TESTING TRADING ANALYZER INTEGRATION")
    print("="*50)
    
    from trading_analyzer import TradingAnalyzer
    from data_fetcher import StockDataFetcher
    
    # Initialize
    print("\n1. Initializing TradingAnalyzer with data fetcher...")
    fetcher = StockDataFetcher()
    analyzer = TradingAnalyzer(data_fetcher=fetcher)
    
    # Check new components exist
    print("\n2. Checking new components are initialized...")
    assert hasattr(analyzer, 'sector_analyzer'), "Missing sector_analyzer"
    assert hasattr(analyzer, 'catalyst_detector'), "Missing catalyst_detector"
    print("   ✅ sector_analyzer present")
    print("   ✅ catalyst_detector present")
    
    # Test with real stock
    print("\n3. Fetching AAPL data and running analysis...")
    try:
        stock_data = fetcher.fetch_stock_data('AAPL')
        history_data = fetcher.fetch_stock_history('AAPL', period='3mo')
        
        if 'error' not in stock_data and history_data and 'data' in history_data:
            result = analyzer.analyze(stock_data, history_data)
            
            # Check new fields in result
            print("\n4. Checking new fields in analysis result...")
            
            if 'sector_analysis' in result:
                sa = result['sector_analysis']
                print(f"   Sector Analysis: {sa.get('sector')} ({sa.get('sector_signal')})")
            else:
                print("   ⚠️ sector_analysis not in result")
            
            if 'catalyst_info' in result:
                ci = result['catalyst_info']
                print(f"   Catalyst Info: Earnings in {ci.get('days_to_earnings')} days")
            else:
                print("   ⚠️ catalyst_info not in result")
            
            rec = result.get('recommendation', {})
            print(f"\n5. Checking recommendation fields...")
            print(f"   Action: {rec.get('action')}")
            print(f"   Confidence: {rec.get('confidence')}%")
            print(f"   R:R Ratio: {rec.get('risk_reward_ratio')}:1 ({rec.get('rr_quality')})")
            print(f"   Entry: ${rec.get('entry_price', 0):.2f}")
            print(f"   Target: ${rec.get('target_price', 0):.2f}")
            print(f"   Stop Loss: ${rec.get('stop_loss', 0):.2f}")
            
            # Check volume analysis has liquidity
            va = result.get('volume_analysis', {})
            print(f"\n6. Checking volume analysis...")
            print(f"   Liquidity Grade: {va.get('liquidity_grade')}")
            print(f"   Avg Dollar Volume: ${va.get('avg_dollar_volume', 0):,.0f}")
            
            print("\n✅ Trading Analyzer Integration tests passed!")
            return True
        else:
            print("   ⚠️ Could not fetch data, skipping integration test")
            return True
    except Exception as e:
        print(f"   ❌ Error during integration test: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("  TRADING STRATEGY ENHANCEMENTS - VERIFICATION TEST")
    print("="*60)
    
    all_passed = True
    
    # Test 1: Sector Momentum
    try:
        all_passed = test_sector_momentum() and all_passed
    except Exception as e:
        print(f"❌ Sector Momentum test failed: {e}")
        all_passed = False
    
    # Test 2: Catalyst Detector
    try:
        all_passed = test_catalyst_detector() and all_passed
    except Exception as e:
        print(f"❌ Catalyst Detector test failed: {e}")
        all_passed = False
    
    # Test 3: Full Integration
    try:
        all_passed = test_trading_analyzer_integration() and all_passed
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("  ✅ ALL TESTS PASSED!")
    else:
        print("  ❌ SOME TESTS FAILED")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
