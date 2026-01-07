"""
Unit tests for Risk Management module
"""
import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from risk_management import RiskManager


class TestRiskManager(unittest.TestCase):
    """Test RiskManager class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.risk_manager = RiskManager(portfolio_value=10000.0, max_position_size=0.25)
    
    def test_position_size_fixed_fraction(self):
        """Test fixed fraction position sizing"""
        result = self.risk_manager.calculate_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            confidence=80.0,
            risk_per_trade=0.02,
            method='fixed_fraction'
        )
        
        self.assertNotIn('error', result)
        self.assertIn('shares', result)
        self.assertIn('position_value', result)
        self.assertIn('risk_amount', result)
        self.assertGreater(result['shares'], 0)
        self.assertLessEqual(result['position_value'], 2500.0)  # Max 25% of portfolio
    
    def test_position_size_kelly(self):
        """Test Kelly Criterion position sizing"""
        result = self.risk_manager.calculate_position_size(
            entry_price=100.0,
            stop_loss=95.0,
            confidence=70.0,
            method='kelly'
        )
        
        self.assertNotIn('error', result)
        self.assertIn('shares', result)
        self.assertLessEqual(result['position_value'], 2500.0)
    
    def test_stop_loss_recommendation(self):
        """Test stop loss recommendation"""
        result = self.risk_manager.recommend_stop_loss(
            entry_price=100.0,
            current_price=102.0,
            volatility=2.0,
            method='atr'
        )
        
        self.assertNotIn('error', result)
        self.assertIn('stop_loss', result)
        self.assertLess(result['stop_loss'], 100.0)
        self.assertGreater(result['stop_loss'], 90.0)  # Not more than 10% below
    
    def test_var_calculation(self):
        """Test VaR calculation"""
        import pandas as pd
        returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.02, -0.015, 0.01])
        
        result = self.risk_manager.calculate_var(returns, confidence_level=0.95)
        
        self.assertNotIn('error', result)
        self.assertIn('var', result)
        self.assertIn('var_percentage', result)
    
    def test_portfolio_risk(self):
        """Test portfolio risk calculation"""
        positions = [
            {'symbol': 'AAPL', 'value': 5000, 'volatility': 0.20},
            {'symbol': 'MSFT', 'value': 3000, 'volatility': 0.18},
            {'symbol': 'GOOGL', 'value': 2000, 'volatility': 0.22}
        ]
        
        result = self.risk_manager.calculate_portfolio_risk(positions)
        
        self.assertNotIn('error', result)
        self.assertIn('total_value', result)
        self.assertIn('portfolio_volatility', result)
        self.assertEqual(result['total_value'], 10000.0)
    
    def test_max_drawdown(self):
        """Test maximum drawdown calculation"""
        import pandas as pd
        prices = pd.Series([100, 105, 110, 95, 100, 108, 112])
        
        result = self.risk_manager.calculate_max_drawdown(prices)
        
        self.assertNotIn('error', result)
        self.assertIn('max_drawdown', result)
        self.assertLess(result['max_drawdown'], 0)


if __name__ == '__main__':
    unittest.main()

