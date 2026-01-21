"""
Enhanced Feature Engineering Module
Adds market microstructure, alternative data, sector analysis, and macroeconomic features
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class EnhancedFeatureExtractor:
    """Extracts enhanced features for ML models"""
    
    def __init__(self, data_fetcher=None):
        self.data_fetcher = data_fetcher
        self.sector_cache = {}
        self.industry_cache = {}
    
    def _safe_correlation(self, s1: pd.Series, s2: pd.Series) -> float:
        """Calculate correlation safely, handling NaNs and constants"""
        try:
            # Drop NaNs
            valid = pd.concat([s1, s2], axis=1).dropna()
            if len(valid) < 2:
                return 0.0
            
            # Check for constants (std dev near 0)
            if valid.iloc[:, 0].std() < 1e-8 or valid.iloc[:, 1].std() < 1e-8:
                return 0.0
                
            return float(valid.iloc[:, 0].corr(valid.iloc[:, 1]))
        except Exception:
            return 0.0

    def extract_market_microstructure_features(self, df: pd.DataFrame, 
                                               stock_data: dict) -> Dict:
        """
        Extract market microstructure features
        
        Args:
            df: DataFrame with OHLCV data
            stock_data: Stock info dict
            
        Returns:
            Dict with microstructure features
        """
        features = {}
        
        if len(df) < 2:
            return features
        
        # Price spread (simplified - would need bid/ask data)
        high_low_spread = (df['high'] - df['low']) / df['close']
        features['avg_spread'] = high_low_spread.mean()
        features['spread_volatility'] = high_low_spread.std()
        
        # Price impact (simplified)
        price_changes = df['close'].pct_change().dropna()
        volume_changes = df['volume'].pct_change().dropna()
        
        if len(price_changes) > 0 and len(volume_changes) > 0:
            # Correlation between price and volume changes
            features['price_volume_correlation'] = self._safe_correlation(price_changes, volume_changes)
        else:
            features['price_volume_correlation'] = 0
        
        # Liquidity proxy (volume / price volatility)
        if df['close'].std() > 0:
            features['liquidity_proxy'] = df['volume'].mean() / df['close'].std()
        else:
            features['liquidity_proxy'] = 0
        
        # Intraday volatility (high-low range)
        features['intraday_volatility'] = ((df['high'] - df['low']) / df['close']).mean()
        
        # Volume profile features
        features['volume_trend'] = self._calculate_trend(df['volume'])
        features['volume_ma_ratio'] = df['volume'].tail(5).mean() / df['volume'].mean() if df['volume'].mean() > 0 else 1
        
        return features
    
    def extract_sector_relative_strength(self, stock_data: dict, 
                                        history_data: dict) -> Dict:
        """
        Calculate sector and industry relative strength
        
        Args:
            stock_data: Stock info dict
            history_data: Historical price data
            
        Returns:
            Dict with sector/industry relative strength
        """
        features = {}
        
        try:
            sector = stock_data.get('info', {}).get('sector', '')
            industry = stock_data.get('info', {}).get('industry', '')
            
            if not sector or not self.data_fetcher:
                return features
            
            # Get sector ETF (simplified mapping)
            sector_etf_map = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financial Services': 'XLF',
                'Consumer Cyclical': 'XLY',
                'Consumer Defensive': 'XLP',
                'Energy': 'XLE',
                'Industrials': 'XLI',
                'Utilities': 'XLU',
                'Real Estate': 'XLRE',
                'Communication Services': 'XLC',
                'Basic Materials': 'XLB'
            }
            
            sector_etf = sector_etf_map.get(sector, 'SPY')  # Default to SPY
            
            # Fetch sector ETF data
            try:
                sector_data = self.data_fetcher.fetch_stock_data(sector_etf)
                if sector_data and 'history' in sector_data:
                    sector_history = sector_data['history']
                    
                    # Calculate relative strength
                    stock_returns = self._calculate_returns(history_data)
                    sector_returns = self._calculate_returns(sector_history)
                    
                    if len(stock_returns) > 0 and len(sector_returns) > 0:
                        # Align timeframes
                        min_len = min(len(stock_returns), len(sector_returns))
                        stock_ret = stock_returns.tail(min_len)
                        sector_ret = sector_returns.tail(min_len)
                        
                        # Relative strength
                        features['sector_relative_strength'] = (stock_ret.mean() - sector_ret.mean()) * 100
                        features['sector_correlation'] = self._safe_correlation(stock_ret, sector_ret)
                        
            except Exception as e:
                logger.debug(f"Could not fetch sector data: {e}")
        
        except Exception as e:
            logger.error(f"Error calculating sector relative strength: {e}")
        
        return features
    
    def extract_alternative_data_features(self, stock_data: dict, 
                                         news_data: Optional[List] = None) -> Dict:
        """
        Extract alternative data features (news sentiment, social media, etc.)
        
        Args:
            stock_data: Stock info dict
            news_data: Optional list of news articles
            
        Returns:
            Dict with alternative data features
        """
        features = {}
        
        # News sentiment (if available)
        if news_data:
            # Simple sentiment scoring (would need NLP in production)
            positive_keywords = ['growth', 'profit', 'gain', 'up', 'bullish', 'buy', 'outperform']
            negative_keywords = ['loss', 'decline', 'down', 'bearish', 'sell', 'underperform']
            
            positive_count = sum(1 for article in news_data 
                               if any(kw in str(article).lower() for kw in positive_keywords))
            negative_count = sum(1 for article in news_data 
                               if any(kw in str(article).lower() for kw in negative_keywords))
            
            total = len(news_data)
            if total > 0:
                features['news_sentiment'] = (positive_count - negative_count) / total
                features['news_volume'] = total
            else:
                features['news_sentiment'] = 0
                features['news_volume'] = 0
        else:
            features['news_sentiment'] = 0
            features['news_volume'] = 0
        
        # Market cap category (as feature)
        market_cap = stock_data.get('info', {}).get('market_cap', 0)
        if market_cap >= 200_000_000_000:
            features['market_cap_category'] = 5  # Mega cap
        elif market_cap >= 10_000_000_000:
            features['market_cap_category'] = 4  # Large cap
        elif market_cap >= 2_000_000_000:
            features['market_cap_category'] = 3  # Mid cap
        elif market_cap >= 300_000_000:
            features['market_cap_category'] = 2  # Small cap
        else:
            features['market_cap_category'] = 1  # Micro cap
        
        return features
    
    def extract_macroeconomic_features(self, stock_data: dict) -> Dict:
        """
        Extract macroeconomic indicator features
        
        Args:
            stock_data: Stock info dict
            
        Returns:
            Dict with macroeconomic features
        """
        features = {}
        
        # Sector sensitivity to economic cycles
        sector = stock_data.get('info', {}).get('sector', '')
        
        # Cyclical vs defensive classification
        cyclical_sectors = ['Technology', 'Consumer Cyclical', 'Financial Services', 
                           'Industrials', 'Energy', 'Basic Materials']
        defensive_sectors = ['Consumer Defensive', 'Healthcare', 'Utilities', 
                           'Real Estate', 'Communication Services']
        
        if sector in cyclical_sectors:
            features['sector_type'] = 1  # Cyclical
        elif sector in defensive_sectors:
            features['sector_type'] = -1  # Defensive
        else:
            features['sector_type'] = 0  # Neutral
        
        # Beta (if available)
        beta = stock_data.get('info', {}).get('beta', 1.0)
        features['beta'] = beta if beta else 1.0
        
        # Dividend yield (income vs growth)
        dividend_yield = stock_data.get('info', {}).get('dividendYield', 0)
        features['dividend_yield'] = dividend_yield if dividend_yield else 0
        
        return features
    
    def extract_all_enhanced_features(self, stock_data: dict, history_data: dict,
                                     news_data: Optional[List] = None) -> Dict:
        """
        Extract all enhanced features
        
        Args:
            stock_data: Stock info dict
            history_data: Historical price data
            news_data: Optional news data
            
        Returns:
            Dict with all enhanced features
        """
        all_features = {}
        
        # Convert history to DataFrame
        df = pd.DataFrame(history_data.get('data', []))
        if not df.empty and 'close' in df.columns:
            # Market microstructure
            microstructure = self.extract_market_microstructure_features(df, stock_data)
            all_features.update(microstructure)
            
            # Sector relative strength
            sector_features = self.extract_sector_relative_strength(stock_data, history_data)
            all_features.update(sector_features)
        
        # Alternative data
        alt_features = self.extract_alternative_data_features(stock_data, news_data)
        all_features.update(alt_features)
        
        # Macroeconomic
        macro_features = self.extract_macroeconomic_features(stock_data)
        all_features.update(macro_features)
        
        return all_features
    
    def _calculate_returns(self, history_data: dict) -> pd.Series:
        """Calculate returns from history data"""
        if not history_data or 'data' not in history_data:
            return pd.Series()
        
        df = pd.DataFrame(history_data['data'])
        if df.empty or 'close' not in df.columns:
            return pd.Series()
        
        df = df.sort_values('date')
        returns = df['close'].pct_change().dropna()
        return returns
    
    def _calculate_trend(self, series: pd.Series, window: int = 10) -> float:
        """Calculate trend strength"""
        if len(series) < window:
            return 0
        
        recent = series.tail(window)
        older = series.tail(window * 2).head(window)
        
        if older.mean() > 0:
            return (recent.mean() - older.mean()) / older.mean()
        return 0


