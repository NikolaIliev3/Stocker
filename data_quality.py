"""
Data Quality Module
Implements data quality checks, cleaning pipelines, and reliability scoring
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Checks and validates data quality"""
    
    def __init__(self):
        self.quality_scores = {}
        self.issues_found = []
    
    def check_stock_data_quality(self, stock_data: dict, history_data: dict) -> Dict:
        """
        Comprehensive data quality check for stock data
        
        Args:
            stock_data: Stock info dict
            history_data: Historical price data
            
        Returns:
            Dict with quality metrics and issues
        """
        quality_report = {
            'overall_score': 100,
            'issues': [],
            'warnings': [],
            'checks_passed': 0,
            'checks_failed': 0
        }
        
        # Check 1: Data completeness
        completeness = self._check_completeness(stock_data, history_data)
        quality_report.update(completeness)
        if completeness['completeness_score'] < 80:
            quality_report['issues'].append("Low data completeness")
            quality_report['overall_score'] -= 10
        
        # Check 2: Price data validity
        price_validity = self._check_price_validity(history_data)
        quality_report.update(price_validity)
        if not price_validity['price_valid']:
            quality_report['issues'].append("Invalid price data detected")
            quality_report['overall_score'] -= 20
        
        # Check 3: Volume data validity
        volume_validity = self._check_volume_validity(history_data)
        quality_report.update(volume_validity)
        if not volume_validity['volume_valid']:
            quality_report['warnings'].append("Volume data may be unreliable")
            quality_report['overall_score'] -= 5
        
        # Check 4: Outlier detection
        outliers = self._detect_outliers(history_data)
        quality_report['outliers'] = outliers
        if outliers['outlier_count'] > len(history_data.get('data', [])) * 0.05:  # More than 5%
            quality_report['warnings'].append(f"High number of outliers: {outliers['outlier_count']}")
            quality_report['overall_score'] -= 5
        
        # Check 5: Data freshness
        freshness = self._check_data_freshness(history_data)
        quality_report.update(freshness)
        if freshness['is_stale']:
            quality_report['warnings'].append("Data may be stale")
            quality_report['overall_score'] -= 10
        
        # Check 6: Missing values
        missing = self._check_missing_values(history_data)
        quality_report['missing_values'] = missing
        if missing['missing_percentage'] > 5:
            quality_report['issues'].append(f"High percentage of missing values: {missing['missing_percentage']:.2f}%")
            quality_report['overall_score'] -= 10
        
        # Check 7: Data consistency
        consistency = self._check_consistency(history_data)
        quality_report.update(consistency)
        if not consistency['is_consistent']:
            quality_report['warnings'].append("Data consistency issues detected")
            quality_report['overall_score'] -= 5
        
        # Calculate final score
        quality_report['overall_score'] = max(0, min(100, quality_report['overall_score']))
        quality_report['quality_level'] = self._classify_quality(quality_report['overall_score'])
        
        return quality_report
    
    def _check_completeness(self, stock_data: dict, history_data: dict) -> Dict:
        """Check data completeness"""
        required_fields = ['symbol', 'currentPrice', 'info']
        stock_complete = all(field in stock_data or (field == 'info' and 'info' in stock_data) 
                           for field in required_fields)
        
        history_complete = 'data' in history_data and len(history_data.get('data', [])) > 0
        
        required_price_fields = ['date', 'open', 'high', 'low', 'close', 'volume']
        if history_data.get('data'):
            sample = history_data['data'][0] if history_data['data'] else {}
            price_complete = all(field in sample for field in required_price_fields)
        else:
            price_complete = False
        
        completeness_score = (
            (100 if stock_complete else 50) * 0.4 +
            (100 if history_complete else 0) * 0.4 +
            (100 if price_complete else 0) * 0.2
        )
        
        return {
            'completeness_score': completeness_score,
            'stock_data_complete': stock_complete,
            'history_data_complete': history_complete,
            'price_fields_complete': price_complete
        }
    
    def _check_price_validity(self, history_data: dict) -> Dict:
        """Check price data validity"""
        if not history_data or 'data' not in history_data:
            return {'price_valid': False, 'invalid_prices': []}
        
        df = pd.DataFrame(history_data['data'])
        if df.empty:
            return {'price_valid': False, 'invalid_prices': []}
        
        invalid_prices = []
        
        # Check for negative or zero prices
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                invalid = df[df[col] <= 0]
                if len(invalid) > 0:
                    invalid_prices.extend(invalid.index.tolist())
        
        # Check for high > low violations
        if 'high' in df.columns and 'low' in df.columns:
            invalid = df[df['high'] < df['low']]
            if len(invalid) > 0:
                invalid_prices.extend(invalid.index.tolist())
        
        # Check for unrealistic price changes (>50% in one day)
        if 'close' in df.columns:
            pct_change = df['close'].pct_change().abs()
            invalid = df[pct_change > 0.5]
            if len(invalid) > 0:
                invalid_prices.extend(invalid.index.tolist())
        
        return {
            'price_valid': len(invalid_prices) == 0,
            'invalid_prices': list(set(invalid_prices)),
            'invalid_count': len(set(invalid_prices))
        }
    
    def _check_volume_validity(self, history_data: dict) -> Dict:
        """Check volume data validity"""
        if not history_data or 'data' not in history_data:
            return {'volume_valid': False}
        
        df = pd.DataFrame(history_data['data'])
        if df.empty or 'volume' not in df.columns:
            return {'volume_valid': False}
        
        # Check for negative volume
        negative_volume = df[df['volume'] < 0]
        
        # Check for zero volume (may indicate data issue)
        zero_volume = df[df['volume'] == 0]
        
        # Check for unrealistic volume spikes (>10x average)
        if len(df) > 0:
            avg_volume = df['volume'].mean()
            if avg_volume > 0:
                spike_threshold = avg_volume * 10
                volume_spikes = df[df['volume'] > spike_threshold]
            else:
                volume_spikes = pd.DataFrame()
        else:
            volume_spikes = pd.DataFrame()
        
        is_valid = len(negative_volume) == 0
        
        return {
            'volume_valid': is_valid,
            'negative_volume_count': len(negative_volume),
            'zero_volume_count': len(zero_volume),
            'volume_spikes_count': len(volume_spikes)
        }
    
    def _detect_outliers(self, history_data: dict, method: str = 'iqr') -> Dict:
        """Detect outliers in price data"""
        if not history_data or 'data' not in history_data:
            return {'outlier_count': 0, 'outliers': []}
        
        df = pd.DataFrame(history_data['data'])
        if df.empty or 'close' not in df.columns:
            return {'outlier_count': 0, 'outliers': []}
        
        outliers = []
        
        if method == 'iqr':
            # IQR method
            Q1 = df['close'].quantile(0.25)
            Q3 = df['close'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df['close'] < lower_bound) | (df['close'] > upper_bound)
            outliers = df[outlier_mask].index.tolist()
        
        elif method == 'zscore':
            # Z-score method
            z_scores = np.abs((df['close'] - df['close'].mean()) / df['close'].std())
            outliers = df[z_scores > 3].index.tolist()
        
        return {
            'outlier_count': len(outliers),
            'outliers': outliers,
            'outlier_percentage': (len(outliers) / len(df)) * 100 if len(df) > 0 else 0
        }
    
    def _check_data_freshness(self, history_data: dict, max_age_hours: int = 24) -> Dict:
        """Check if data is fresh"""
        if not history_data or 'data' not in history_data:
            return {'is_stale': True, 'last_update': None}
        
        data = history_data['data']
        if not data:
            return {'is_stale': True, 'last_update': None}
        
        # Get most recent date
        try:
            dates = [item.get('date') for item in data if 'date' in item]
            if dates:
                if isinstance(dates[0], str):
                    last_date = pd.to_datetime(max(dates))
                else:
                    last_date = max(dates)
                
                age_hours = (datetime.now() - last_date).total_seconds() / 3600
                is_stale = age_hours > max_age_hours
                
                return {
                    'is_stale': is_stale,
                    'last_update': last_date.isoformat() if hasattr(last_date, 'isoformat') else str(last_date),
                    'age_hours': age_hours
                }
        except Exception as e:
            logger.error(f"Error checking data freshness: {e}")
        
        return {'is_stale': True, 'last_update': None}
    
    def _check_missing_values(self, history_data: dict) -> Dict:
        """Check for missing values"""
        if not history_data or 'data' not in history_data:
            return {'missing_percentage': 100, 'missing_fields': []}
        
        df = pd.DataFrame(history_data['data'])
        if df.empty:
            return {'missing_percentage': 100, 'missing_fields': []}
        
        required_fields = ['open', 'high', 'low', 'close', 'volume']
        missing_fields = []
        total_cells = len(df) * len(required_fields)
        missing_cells = 0
        
        for field in required_fields:
            if field in df.columns:
                missing_count = df[field].isna().sum()
                missing_cells += missing_count
                if missing_count > 0:
                    missing_fields.append(field)
        
        missing_percentage = (missing_cells / total_cells * 100) if total_cells > 0 else 0
        
        return {
            'missing_percentage': missing_percentage,
            'missing_fields': missing_fields,
            'missing_cells': missing_cells
        }
    
    def _check_consistency(self, history_data: dict) -> Dict:
        """Check data consistency"""
        if not history_data or 'data' not in history_data:
            return {'is_consistent': False}
        
        df = pd.DataFrame(history_data['data'])
        if df.empty:
            return {'is_consistent': False}
        
        issues = []
        
        # Check date ordering
        if 'date' in df.columns:
            df_sorted = df.sort_values('date')
            if not df_sorted['date'].equals(df['date']):
                issues.append('date_ordering')
        
        # Check price relationships
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # High should be >= all other prices
            invalid_high = df[df['high'] < df[['open', 'low', 'close']].max(axis=1)]
            if len(invalid_high) > 0:
                issues.append('high_price')
            
            # Low should be <= all other prices
            invalid_low = df[df['low'] > df[['open', 'high', 'close']].min(axis=1)]
            if len(invalid_low) > 0:
                issues.append('low_price')
        
        return {
            'is_consistent': len(issues) == 0,
            'consistency_issues': issues
        }
    
    def _classify_quality(self, score: float) -> str:
        """Classify data quality level"""
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 60:
            return "Fair"
        elif score >= 40:
            return "Poor"
        else:
            return "Very Poor"
    
    def clean_data(self, history_data: dict, quality_report: Dict) -> dict:
        """
        Clean data based on quality report
        
        Args:
            history_data: Historical data to clean
            quality_report: Quality check results
            
        Returns:
            Cleaned data dict
        """
        if not history_data or 'data' not in history_data:
            return history_data
        
        df = pd.DataFrame(history_data['data'])
        original_len = len(df)
        
        # Remove invalid prices
        if 'invalid_prices' in quality_report.get('price_validity', {}):
            invalid_indices = quality_report['price_validity']['invalid_prices']
            df = df.drop(df.index[invalid_indices])
        
        # Remove outliers (optional - can also cap instead)
        if quality_report.get('outliers', {}).get('outlier_count', 0) > 0:
            # For now, we'll keep outliers but could remove them
            pass
        
        # Fill missing values: STRICT FORWARD FILL ONLY (Rule #5)
        # Never use bfill() as it introduces look-ahead bias from future prices.
        if 'close' in df.columns:
            df['close'] = df['close'].ffill()
        
        # Also ffill other price columns to be safe
        for col in ['open', 'high', 'low', 'volume']:
            if col in df.columns:
                df[col] = df[col].ffill()
        
        # Ensure price relationships are valid
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # Fix high/low violations
            df['high'] = df[['open', 'high', 'low', 'close']].max(axis=1)
            df['low'] = df[['open', 'high', 'low', 'close']].min(axis=1)
        
        cleaned_len = len(df)
        
        return {
            'data': df.to_dict('records'),
            'cleaned': original_len != cleaned_len,
            'rows_removed': original_len - cleaned_len
        }

