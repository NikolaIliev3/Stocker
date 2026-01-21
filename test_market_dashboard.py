"""
Test Script for Market Dashboard UI
Verifies that the MarketDashboard can be instantiated and updated with data without errors.
"""
import tkinter as tk
from tkinter import ttk
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from market_dashboard import MarketDashboard
    print("Successfully imported MarketDashboard")
except ImportError as e:
    print(f"Failed to import MarketDashboard: {e}")
    sys.exit(1)

def test_dashboard_ui():
    root = tk.Tk()
    root.title("Test Market Dashboard")
    root.geometry("800x600")
    
    # Mock analysis data
    mock_data = {
        'macro_info': {
            'available': True,
            'vix_value': 15.5,
            'tnx_value': 4.2,
            'dxy_value': 102.5,
            'risk_level': 'low',
            'regime_summary': 'Risk On'
        },
        'sentiment_info': {
            'available': True,
            'sentiment_score': 0.8,
            'sentiment_rating': 'bullish',
            'headlines': [
                {'title': 'Stock Market Rallies on Great News', 'score': 0.9},
                {'title': 'Inflation Data Better Than Expected', 'score': 0.5},
                {'title': 'Tech Sector Booming', 'score': 0.7}
            ]
        },
        'sector_analysis': {
            'available': True,
            'sector': 'Technology',
            'sector_signal': 'outperform',
            'stock_vs_sector': 2.5,
            'sector_trend': 'bullish'
        }
    }
    
    try:
        dashboard = MarketDashboard(root)
        dashboard.pack(fill="both", expand=True)
        print("MarketDashboard instantiated successfully")
        
        dashboard.update_data(mock_data)
        print("MarketDashboard updated with data successfully")
        
        # Check updates with empty data (should not crash)
        dashboard.update_data({})
        print("MarketDashboard handled empty data successfully")
        
        dashboard.update_data(None)
        print("MarketDashboard handled None data successfully")
        
        # Schedule auto-close
        root.after(1000, root.destroy)
        root.mainloop()
        print("UI Loop finished without crashes")
        
    except Exception as e:
        print(f"Error during UI test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_dashboard_ui()
