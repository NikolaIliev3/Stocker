"""
Unit tests for Data Quality module
"""
import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_quality import DataQualityChecker


class TestDataQualityChecker(unittest.TestCase):
    """Test DataQualityChecker class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.checker = DataQualityChecker()
    
    def test_completeness_check(self):
        """Test data completeness check"""
        stock_data = {
            'symbol': 'AAPL',
            'currentPrice': 150.0,
            'info': {'name': 'Apple Inc.'}
        }
        
        history_data = {
            'data': [
                {'date': '2024-01-01', 'open': 150, 'high': 152, 'low': 149, 'close': 151, 'volume': 1000000}
            ]
        }
        
        result = self.checker._check_completeness(stock_data, history_data)
        
        self.assertIn('completeness_score', result)
        self.assertGreater(result['completeness_score'], 80)
    
    def test_price_validity_check(self):
        """Test price validity check"""
        history_data = {
            'data': [
                {'date': '2024-01-01', 'open': 150, 'high': 152, 'low': 149, 'close': 151, 'volume': 1000000},
                {'date': '2024-01-02', 'open': 151, 'high': 153, 'low': 150, 'close': 152, 'volume': 1100000}
            ]
        }
        
        result = self.checker._check_price_validity(history_data)
        
        self.assertIn('price_valid', result)
        self.assertTrue(result['price_valid'])
    
    def test_outlier_detection(self):
        """Test outlier detection"""
        history_data = {
            'data': [
                {'date': f'2024-01-{i:02d}', 'open': 150 + i, 'high': 152 + i, 
                 'low': 149 + i, 'close': 151 + i, 'volume': 1000000}
                for i in range(20)
            ]
        }
        # Add an outlier
        history_data['data'].append({
            'date': '2024-01-21', 'open': 300, 'high': 310, 'low': 290, 
            'close': 305, 'volume': 1000000
        })
        
        result = self.checker._detect_outliers(history_data)
        
        self.assertIn('outlier_count', result)
        self.assertGreaterEqual(result['outlier_count'], 0)
    
    def test_data_quality_check(self):
        """Test comprehensive data quality check"""
        stock_data = {
            'symbol': 'AAPL',
            'currentPrice': 150.0,
            'info': {'name': 'Apple Inc.'}
        }
        
        history_data = {
            'data': [
                {'date': f'2024-01-{i:02d}', 'open': 150 + i*0.1, 'high': 152 + i*0.1,
                 'low': 149 + i*0.1, 'close': 151 + i*0.1, 'volume': 1000000 + i*10000}
                for i in range(30)
            ]
        }
        
        result = self.checker.check_stock_data_quality(stock_data, history_data)
        
        self.assertIn('overall_score', result)
        self.assertIn('quality_level', result)
        self.assertGreaterEqual(result['overall_score'], 0)
        self.assertLessEqual(result['overall_score'], 100)


if __name__ == '__main__':
    unittest.main()

