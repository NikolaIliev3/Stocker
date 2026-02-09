
import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from paper_trader import PaperTrader
from config import BASE_POSITION_SIZE_PCT, MAX_CONCURRENT_POSITIONS, MAX_PORTFOLIO_ALLOCATION, GLOBAL_STOP_LOSS_PCT, TIME_STOP_DAYS

class TestSafeguards(unittest.TestCase):
    def setUp(self):
        # Mock dependencies
        self.mock_broker = MagicMock()
        self.mock_fetcher = MagicMock()
        self.mock_predictor = MagicMock()
        self.mock_tracker = MagicMock()
        self.mock_scanner = MagicMock()
        self.mock_learning = MagicMock()

        # Initialize PaperTrader with mocks
        with patch('paper_trader.MockBroker', return_value=self.mock_broker), \
             patch('paper_trader.StockDataFetcher', return_value=self.mock_fetcher), \
             patch('paper_trader.HybridStockPredictor', return_value=self.mock_predictor), \
             patch('paper_trader.PredictionsTracker', return_value=self.mock_tracker), \
             patch('paper_trader.MarketScanner', return_value=self.mock_scanner), \
             patch('paper_trader.LearningTracker', return_value=self.mock_learning):
            
            self.trader = PaperTrader(['AAPL'], initial_balance=10000)
            
        # Standardize mock returns
        self.mock_broker.get_positions.return_value = {}
        self.mock_broker.get_account_summary.return_value = {'cash': 10000, 'total_value': 10000, 'holdings_count': 0}
        self.mock_fetcher.get_current_data.return_value = {'price': 100.0}
        self.mock_fetcher.fetch_stock_history.return_value = {'data': []}
        
    def test_confidence_tiered_sizing_sweet_spot(self):
        """Test that 75-80% confidence gets 1.5x size"""
        # Setup
        self.mock_predictor.predict.return_value = {
            'recommendation': {'action': 'BUY', 'confidence': 76.0, 'target_price': 105, 'stop_loss': 95}
        }
        self.trader.min_confidence = 70.0 # Ensure it passes filter
        
        # Execute
        predictions = {}
        self.trader._scan_for_entries(predictions)
        
        # Verify
        # Base = 10000 * 0.02 = 200. Sweet Spot 1.5x = 300.
        # Price = 100. Qty = 3.
        self.mock_broker.execute_order.assert_called_with('AAPL', 'BUY', 3.0, 100.0, target_price=105, stop_loss=95)
        print("✅ Sweet Spot Sizing Verified (1.5x Base)")

    def test_confidence_tiered_sizing_momentum(self):
        """Test that 85%+ confidence gets 0.5x size"""
        # Setup
        self.mock_predictor.predict.return_value = {
            'recommendation': {'action': 'BUY', 'confidence': 88.0, 'target_price': 105, 'stop_loss': 95}
        }
        
        # Execute
        predictions = {}
        self.trader._scan_for_entries(predictions)
        
        # Verify
        # Base = 200. Momentum 0.5x = 100.
        # Price = 100. Qty = 1.
        self.mock_broker.execute_order.assert_called_with('AAPL', 'BUY', 1.0, 100.0, target_price=105, stop_loss=95)
        print("✅ Momentum Sizing Verified (0.5x Base)")

    def test_hard_stop_loss(self):
        """Test that -4% drop triggers Hard Stop"""
        # Setup
        entry_price = 100.0
        current_price = 95.0 # -5% drop
        self.mock_broker.get_positions.return_value = {
            'AAPL': {'qty': 10, 'avg_price': entry_price, 'stop_loss': 90}
        }
        self.mock_fetcher.get_current_data.return_value = {'price': current_price}
        
        # Execute
        predictions = {}
        self.trader._manage_existing_positions(predictions)
        
        # Verify
        self.mock_broker.execute_order.assert_called_with('AAPL', 'SELL', 10, current_price)
        print("✅ Hard Stop Loss Verified (-4% Trigger)")

    def test_time_stop(self):
        """Test that > 15 days triggers Time Stop"""
        # Setup
        entry_price = 100.0
        current_price = 100.0
        self.mock_broker.get_positions.return_value = {
            'AAPL': {'qty': 10, 'avg_price': entry_price}
        }
        self.mock_fetcher.get_current_data.return_value = {'price': current_price}
        
        # Mock active prediction 16 days ago
        old_date = (datetime.now() - timedelta(days=16)).isoformat()
        self.mock_tracker.get_active_predictions.return_value = [
            {'symbol': 'AAPL', 'timestamp': old_date}
        ]
        
        # Execute
        predictions = {}
        self.trader._manage_existing_positions(predictions)
        
        # Verify
        self.mock_broker.execute_order.assert_called_with('AAPL', 'SELL', 10, current_price)
        print("✅ Time Stop Verified (16 Days)")

    def test_concurrency_limit(self):
        """Test that max positions blocks new entries"""
        # Setup
        # Mock 8 positions
        positions = {f'SYM{i}': {} for i in range(8)}
        self.mock_broker.get_positions.return_value = positions
        
        # Execute
        predictions = {}
        self.trader._scan_for_entries(predictions)
        
        # Verify
        self.mock_predictor.predict.assert_not_called()
        print("✅ Concurrency Limit Verified (Max 8)")

if __name__ == '__main__':
    unittest.main()
