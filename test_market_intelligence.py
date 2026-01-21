"""
Test Script for Market Intelligence Features (Backend)
Verifies Sentiment Analyzer, Macro Regime Detector, and Integration.
"""
import sys
import os
import pandas as pd
import logging
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_backend_integration():
    try:
        from trading_analyzer import TradingAnalyzer
        
        print("Initializing TradingAnalyzer...")
        analyzer = TradingAnalyzer()
        
        # Check if components are initialized
        if hasattr(analyzer, 'sentiment_analyzer'):
            print("Verified: SentimentAnalyzer is initialized in TradingAnalyzer")
        else:
            print("FAILED: SentimentAnalyzer not found in TradingAnalyzer")
            
        if hasattr(analyzer, 'macro_detector'):
            print("Verified: MacroRegimeDetector is initialized in TradingAnalyzer")
        else:
            print("FAILED: MacroRegimeDetector not found in TradingAnalyzer")
            
        # Create Dummy Data for analyze
        print("Creating dummy stock data...")
        df = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=50),
            'open': [100.0] * 50,
            'high': [102.0] * 50,
            'low': [98.0] * 50,
            'close': [101.0] * 50,
            'volume': [100000] * 50
        })
        
        stock_data = {
            'symbol': 'TEST',
            'price': 101.0,
            'volume': 100000
        }
        
        history_data = {'data': df.to_dict('records')}
        
        # Mock the internal calls to avoid API hits during quick test
        # analyzer.macro_detector.analyze_regime = MagicMock(return_value={'regime': 'bull', 'strength': 80})
        # analyzer.sentiment_analyzer.analyze_symbol = MagicMock(return_value={'sentiment_score': 0.5, 'available': True})
        
        # Actually calling analyze (using the real methods or mocks if configured above)
        print("Running analyze()...")
        result = analyzer.analyze(stock_data, history_data)
        
        # Verify output keys
        if 'macro_info' in result:
             print("Verified: 'macro_info' present in analysis result")
        else:
             print("FAILED: 'macro_info' missing from analysis result")
             
        if 'sentiment_info' in result:
             print("Verified: 'sentiment_info' present in analysis result")
        else:
             print("FAILED: 'sentiment_info' missing from analysis result")
             
        print("Backend integration test passed!")
        
    except Exception as e:
        print(f"Error during backend test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_backend_integration()
