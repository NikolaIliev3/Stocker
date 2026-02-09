import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
from datetime import datetime
from paper_trader import PaperTrader

class TestTieredRisk(unittest.TestCase):
    def setUp(self):
        self.data_dir = Path.home() / ".stocker"
        with patch('paper_trader.StockDataFetcher'), \
             patch('paper_trader.HybridStockPredictor'), \
             patch('paper_trader.MockBroker'), \
             patch('paper_trader.PredictionsTracker'):
            self.trader = PaperTrader(['AAPL', 'MSFT'])
            self.trader.fetcher = MagicMock()
            self.trader.predictor = MagicMock()
            self.trader.broker = MagicMock()
            self.trader.tracker = MagicMock()
            self.trader.min_confidence = 70.0

    def test_plateau_skip(self):
        """95% Confidence should be skipped"""
        self.trader.broker.get_positions.return_value = {}
        self.trader.broker.get_account_summary.return_value = {'cash': 1000, 'total_value': 1000}
        self.trader.fetcher.get_current_data.return_value = {'price': 100}
        self.trader.fetcher.fetch_stock_history.return_value = {'data': []}
        
        # Mock 95% BUY
        self.trader.predictor.predict.return_value = {
            'recommendation': {'action': 'BUY', 'confidence': 95.0, 'target_price': 103, 'stop_loss': 96}
        }
        
        with self.assertLogs('paper_trader', level='INFO') as cm:
            self.trader._scan_for_entries({})
            self.assertTrue(any("PLATEAU SKIP" in line for line in cm.output))
        
        self.trader.broker.execute_order.assert_not_called()

    def test_elite_priority_standard_skip(self):
        """Hold 1 Elite -> 75% Signal should be skipped"""
        # Mock Hold 1 Elite
        self.trader.broker.get_positions.return_value = {'AAPL': {}}
        self.trader.tracker.get_active_predictions.return_value = [
            {'symbol': 'AAPL', 'status': 'active', 'confidence': 85.0} # ELITE
        ]
        self.trader.broker.get_account_summary.return_value = {'cash': 1000, 'total_value': 1000}
        self.trader.fetcher.get_current_data.return_value = {'price': 100}
        self.trader.fetcher.fetch_stock_history.return_value = {'data': []}
        
        # Mock 75% BUY (STANDARD)
        self.trader.predictor.predict.return_value = {
            'recommendation': {'action': 'BUY', 'confidence': 75.0, 'target_price': 103, 'stop_loss': 96}
        }
        
        with self.assertLogs('paper_trader', level='INFO') as cm:
            self.trader._scan_for_entries({})
            self.assertTrue(any("ELITE PRIORITY: Skipping Standard signal" in line for line in cm.output))
            
        self.trader.broker.execute_order.assert_not_called()

    def test_elite_allow_with_standard_held(self):
        """Hold 1 Standard -> 85% Elite signal should be ALLOWED"""
        # Mock Hold 1 Standard
        self.trader.broker.get_positions.return_value = {'MSFT': {}}
        self.trader.tracker.get_active_predictions.return_value = [
            {'symbol': 'MSFT', 'status': 'active', 'confidence': 75.0} # STANDARD
        ]
        self.trader.broker.get_account_summary.return_value = {'cash': 1000, 'total_value': 1000}
        self.trader.fetcher.get_current_data.return_value = {'price': 100}
        self.trader.fetcher.fetch_stock_history.return_value = {'data': []}
        
        # Mock 85% BUY (ELITE)
        self.trader.predictor.predict.return_value = {
            'recommendation': {'action': 'BUY', 'confidence': 85.0, 'target_price': 103, 'stop_loss': 96}
        }
        
        self.trader._scan_for_entries({})
        
        # Verify execution
        self.trader.broker.execute_order.assert_called_once()
        args = self.trader.broker.execute_order.call_args[0]
        # Check multiplier logic (ELITE_MULTIPLIER = 1.5)
        # Total Value 1000 * 2% Base = 20. 20 * 1.5 = 30. Qty = 30/100 = 0.3
        self.assertAlmostEqual(args[2], 0.3) 

if __name__ == '__main__':
    unittest.main()
