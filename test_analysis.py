"""
Test script to verify trading analysis works correctly
"""
import logging
logging.basicConfig(level=logging.INFO)

from trading_analyzer import TradingAnalyzer
from data_fetcher import StockDataFetcher

def test_analysis():
    fetcher = StockDataFetcher()
    analyzer = TradingAnalyzer(fetcher)
    
    # Test with V (Visa)
    print("Testing V (Visa) analysis...")
    stock_data = fetcher.fetch_stock_data('V', force_refresh=True)
    history = fetcher.fetch_stock_history('V', period='1y')
    
    if not stock_data or not history:
        print("ERROR: Failed to fetch data for V")
        return False
    
    print(f"Current Price: ${stock_data.get('price', 0):.2f}")
    
    # Run analysis
    analysis = analyzer.analyze(stock_data, history)
    
    # Check price_action
    pa = analysis.get('price_action', {})
    trend = pa.get('trend', 'MISSING')
    short_change = pa.get('short_change_pct', 'MISSING')
    bullish = pa.get('bullish_signals', 0)
    bearish = pa.get('bearish_signals', 0)
    
    print(f"\n=== PRICE ACTION ===")
    print(f"Trend: {trend}")
    print(f"Short-term change: {short_change}%")
    print(f"Bullish signals: {bullish}")
    print(f"Bearish signals: {bearish}")
    
    # Check indicators
    indicators = analysis.get('indicators', {})
    print(f"\n=== INDICATORS ===")
    print(f"RSI: {indicators.get('rsi', 'N/A')}")
    print(f"MACD: {indicators.get('macd', 'N/A')}")
    
    # Check recommendation
    rec = analysis.get('recommendation', {})
    print(f"\n=== RECOMMENDATION ===")
    print(f"Action: {rec.get('action', 'MISSING')}")
    print(f"Confidence: {rec.get('confidence', 0)}%")
    
    # Validation
    print(f"\n=== VALIDATION ===")
    if short_change != 'MISSING' and float(short_change) < -2:
        if trend == 'downtrend':
            print("✅ Trend correctly detected as downtrend")
        else:
            print(f"❌ BUG: Short change is {short_change}% but trend is {trend}")
            return False
    
    if trend == 'MISSING':
        print("❌ BUG: trend is MISSING from price_action")
        return False
        
    print("✅ All checks passed!")
    return True

if __name__ == "__main__":
    test_analysis()
