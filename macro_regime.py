"""
Macro Regime Detector
Analyzes macroeconomic indicators (VIX, 10Y Yield, Dollar Index) 
to determine the broader market risk environment.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class MacroRegimeDetector:
    """
    Analyzes macro indicators to determine market risk regime.
    """
    
    # Tickers for macro indicators on Yahoo Finance
    # ^VIX = CBOE Volatility Index
    # ^TNX = CBOE Interest Rate 10 Year T Note (Yield)
    # DX-Y.NYB = US Dollar/USDX - Index - Cash
    MACRO_TICKERS = {
        'VIX': '^VIX',
        'TNX': '^TNX',
        'DXY': 'DX-Y.NYB'
    }
    
    # Cache
    _macro_cache = {}
    _cache_time = None
    CACHE_DURATION_HOURS = 1
    
    def __init__(self, data_fetcher=None):
        """
        Initialize macro detector
        """
        self.data_fetcher = data_fetcher
        
    def _fetch_history(self, start_date: datetime, end_date: Optional[datetime] = None) -> Dict[str, pd.DataFrame]:
        """Fetch history for all macro tickers with caching and date range support"""
        if end_date is None:
            end_date = datetime.now()
            
        # Unique cache key based on full date range (Daily resolution for backtest parity)
        cache_key = f"{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}"
        
        # Check cache (1 hour duration for real-time, but keep backtest data longer)
        if self._cache_time and (datetime.now() - self._cache_time) < timedelta(hours=self.CACHE_DURATION_HOURS):
            if cache_key in self._macro_cache:
                return self._macro_cache[cache_key]
            
        results = {}
        try:
            import yfinance as yf
            tickers = list(self.MACRO_TICKERS.values())
            
            # Use specific start/end dates for accuracy in backtesting
            logger.debug(f"Fetching macro data from {start_date.date()} to {end_date.date()}...")
            data = yf.download(tickers, start=start_date.strftime('%Y-%m-%d'), 
                               end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'), 
                               progress=False, auto_adjust=True)
            
            if not data.empty:
                # Handle multi-index columns
                if isinstance(data.columns, pd.MultiIndex):
                    try:
                        closes = data['Close']
                    except KeyError:
                        closes = data['Adj Close']
                else:
                    closes = data
                    
                # Map back to internal names
                for name, ticker in self.MACRO_TICKERS.items():
                    if ticker in closes.columns:
                        results[name] = closes[ticker].dropna()
                    elif ticker in data.columns:
                        results[name] = data[ticker].dropna()
            
            # Store in cache
            if results:
                self._macro_cache[cache_key] = results
                self._cache_time = datetime.now()
            
        except Exception as e:
            logger.warning(f"Error fetching macro data: {e}")
            
        return results

    def analyze_regime_from_data(self, data: Dict[str, pd.Series]) -> Dict:
        """
        Analyze macroeconomic regime from provided historical data.
        Useful for training and backtesting to avoid data leakage.
        """
        result = {
            'risk_level': 'neutral',
            'regime_summary': 'Normal Market Conditions',
            'vix_status': 'unknown',
            'yield_status': 'unknown',
            'dxy_status': 'unknown',
            'score': 50,
            'available': False,
            'reasoning': []
        }
        
        if not data:
            return result
            
        try:
            risk_score = 0
            reasons = []
            
            # --- VIX Analysis (Fear Gauge) ---
            if 'VIX' in data and len(data['VIX']) >= 50:
                vix_series = data['VIX']
                current_vix = vix_series.iloc[-1]
                vix_ma50 = vix_series.rolling(50).mean()
                curr_vix_ma50 = vix_ma50.iloc[-1]
                prev_vix = vix_series.iloc[-2] if len(vix_series) > 1 else current_vix
                
                result['vix_value'] = float(current_vix)
                result['vix_change_pct'] = ((current_vix / prev_vix) - 1) * 100 if prev_vix > 0 else 0
                
                if current_vix > 30:
                    result['vix_status'] = 'extreme_fear'
                    risk_score += 3
                    reasons.append(f"VIX is EXTREMELY HIGH ({current_vix:.1f})")
                elif current_vix > 20:
                    result['vix_status'] = 'high_fear'
                    risk_score += 2
                    reasons.append(f"VIX is Elevated ({current_vix:.1f})")
                elif current_vix < 15:
                    result['vix_status'] = 'complacency'
                    risk_score -= 1 
                    reasons.append(f"VIX is Low ({current_vix:.1f})")
                else:
                    result['vix_status'] = 'normal'
                
                # VIX Trend
                if curr_vix_ma50 > 0 and current_vix > curr_vix_ma50 * 1.2:
                    risk_score += 1
            
            # --- TNX Analysis (10Y Yield) ---
            if 'TNX' in data and len(data['TNX']) >= 50:
                tnx_series = data['TNX']
                current_tnx = tnx_series.iloc[-1]
                tnx_ma50 = tnx_series.rolling(50).mean().iloc[-1]
                
                result['tnx_value'] = float(current_tnx)
                
                if current_tnx > tnx_ma50 * 1.1:
                    result['yield_status'] = 'rising_rapidly'
                    risk_score += 1
                elif current_tnx < tnx_ma50 * 0.9:
                    result['yield_status'] = 'falling'
                    risk_score -= 1
                else:
                    result['yield_status'] = 'stable'
            
            # --- DXY Analysis (Dollar) ---
            if 'DXY' in data and len(data['DXY']) > 0:
                dxy_series = data['DXY']
                current_dxy = dxy_series.iloc[-1]
                result['dxy_value'] = float(current_dxy)
                
                if current_dxy > 105:
                    result['dxy_status'] = 'very_strong'
                    risk_score += 1
                elif current_dxy < 90:
                    result['dxy_status'] = 'weak'
                    risk_score -= 1
                else:
                    result['dxy_status'] = 'neutral'
                    
            result['available'] = True
            
            # Map risk score
            if risk_score >= 3:
                result['risk_level'] = 'extreme'
            elif risk_score >= 1:
                result['risk_level'] = 'high'
            elif risk_score <= -2:
                result['risk_level'] = 'low'
            else:
                result['risk_level'] = 'neutral'
                
            result['reasoning'] = reasons
            
        except Exception as e:
            logger.warning(f"Error in macro analysis from data: {e}")
            
        return result

    def analyze_regime(self, as_of_date: Optional[datetime] = None) -> Dict:
        """
        Analyze macroeconomic regime (current or as of a specific date)
        
        Returns:
            Dict with regime classification and risk level
        """
        try:
            if as_of_date is None:
                as_of_date = datetime.now()
                
            # Fetch data (ensure we have enough history)
            # Use a fixed start date for backtesting to avoid multiple downloads
            fetch_start = as_of_date - timedelta(days=400)
            data = self._fetch_history(fetch_start)
            
            # Slice data up to as_of_date to avoid look-ahead bias
            sliced_data = {}
            for name, series in data.items():
                if not series.empty:
                    # Filter index <= as_of_date
                    # Ensure series index is datetime for comparison
                    if not isinstance(series.index, pd.DatetimeIndex):
                        series.index = pd.to_datetime(series.index)
                    
                    mask = series.index <= pd.Timestamp(as_of_date)
                    sliced_data[name] = series[mask]
                else:
                    sliced_data[name] = series
            
            return self.analyze_regime_from_data(sliced_data)
        except Exception as e:
            logger.warning(f"Error in macro regime analysis: {e}")
            return {
                'risk_level': 'neutral',
                'regime_summary': 'Error in Analysis',
                'available': False
            }
        
    def get_macro_adjustment(self, macro_result: Dict) -> Dict:
        """
        Get confidence adjustment based on macro regime
        """
        if not macro_result.get('available', False):
            return {'confidence_adjustment': 0, 'reasoning': []}
            
        risk_level = macro_result['risk_level']
        adj = 0
        
        if risk_level == 'extreme':
            adj = -15
        elif risk_level == 'high':
            adj = -5
        elif risk_level == 'low':
            adj = 5
            
        return {
            'confidence_adjustment': adj,
            'reasoning': macro_result.get('reasoning', [])
        }
