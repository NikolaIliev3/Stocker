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
        
    def _fetch_history(self, start_date: datetime) -> Dict[str, pd.DataFrame]:
        """Fetch history for all macro tickers"""
        # Check cache
        if self._cache_time and (datetime.now() - self._cache_time) < timedelta(hours=self.CACHE_DURATION_HOURS):
            return self._macro_cache
            
        results = {}
        try:
            import yfinance as yf
            # Fetch data
            tickers = list(self.MACRO_TICKERS.values())
            # Use '1y' to get enough data for trends
            data = yf.download(tickers, period="1y", interval="1d", progress=False, auto_adjust=True)
            
            # yfinance download with multiple tickers returns a detailed multi-index columns DataFrame
            # e.g. columns are (PriceType, Ticker)
            # We want simple close prices
            
            if not data.empty:
                # Handle multi-index columns
                if isinstance(data.columns, pd.MultiIndex):
                    # Try to get 'Close' or 'Adj Close'
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
                    elif ticker in data.columns: # Sometimes it flattens differently
                         results[name] = data[ticker].dropna()
                         
            self._macro_cache = results
            self._cache_time = datetime.now()
            
        except Exception as e:
            logger.warning(f"Error fetching macro data: {e}")
            
        return results

    def analyze_regime(self) -> Dict:
        """
        Analyze current macroeconomic regime
        
        Returns:
            Dict with regime classification and risk level
        """
        result = {
            'risk_level': 'neutral', # low, neutral, high, extreme
            'regime_summary': 'Normal Market Conditions',
            'vix_status': 'unknown',
            'yield_status': 'unknown',
            'dxy_status': 'unknown',
            'score': 50, # 0 (extreme fear) to 100 (extreme greed/safe)
            'available': False,
            'reasoning': []
        }
        
        try:
            # Fetch data (1 year history)
            start_date = datetime.now() - timedelta(days=365)
            data = self._fetch_history(start_date)
            
            if not data:
                return result
                
            risk_score = 0 # Higher = Higher Risk
            reasons = []
            
            # --- VIX Analysis (Fear Gauge) ---
            if 'VIX' in data:
                vix_series = data['VIX']
                current_vix = vix_series.iloc[-1]
                vix_ma50 = vix_series.rolling(50).mean().iloc[-1]
                
                result['vix_value'] = float(current_vix)
                
                if current_vix > 30:
                    result['vix_status'] = 'extreme_fear'
                    risk_score += 3
                    reasons.append(f"VIX is EXTREMELY HIGH ({current_vix:.1f}) - Market Panicking")
                elif current_vix > 20:
                    result['vix_status'] = 'high_fear'
                    risk_score += 2
                    reasons.append(f"VIX is Elevated ({current_vix:.1f}) - High Volatility")
                elif current_vix < 15:
                    result['vix_status'] = 'complacency'
                    # Low VIX is usually good for stocks but can mean complacency
                    risk_score -= 1 
                    reasons.append(f"VIX is Low ({current_vix:.1f}) - Stable Market")
                else:
                    result['vix_status'] = 'normal'
                    reasons.append(f"VIX is Normal ({current_vix:.1f})")
                    
                # VIX Trend
                if current_vix > vix_ma50 * 1.2:
                    reasons.append("VIX is trending sharply higher (+20% vs avg)")
                    risk_score += 1
            
            # --- TNX Analysis (10Y Yield) ---
            if 'TNX' in data:
                tnx_series = data['TNX']
                current_tnx = tnx_series.iloc[-1]
                tnx_ma50 = tnx_series.rolling(50).mean().iloc[-1] # Simple MA
                
                result['tnx_value'] = float(current_tnx)
                
                # Yield trend is often more important than absolute level
                if current_tnx > tnx_ma50 * 1.1:
                    result['yield_status'] = 'rising_rapidly'
                    risk_score += 1
                    reasons.append(f"10Y Yields Rising Rapidly ({current_tnx:.2f}%) - Headwind for Growth")
                elif current_tnx < tnx_ma50 * 0.9:
                    result['yield_status'] = 'falling'
                    risk_score -= 1
                    reasons.append(f"10Y Yields Falling ({current_tnx:.2f}%) - Supportive")
                else:
                    result['yield_status'] = 'stable'
            
            # --- DXY Analysis (Dollar) ---
            if 'DXY' in data:
                dxy_series = data['DXY']
                current_dxy = dxy_series.iloc[-1]
                
                result['dxy_value'] = float(current_dxy)
                
                if current_dxy > 105:
                    result['dxy_status'] = 'very_strong'
                    risk_score += 1
                    reasons.append(f"US Dollar Very Strong ({current_dxy:.1f}) - Headwind for Earnings")
                elif current_dxy < 90:
                    result['dxy_status'] = 'weak'
                    risk_score -= 1
                    reasons.append(f"US Dollar Weak ({current_dxy:.1f}) - Supportive for Exports")
                else:
                    result['dxy_status'] = 'neutral'
                    
            # --- Synthesis ---
            result['available'] = True
            
            # Map risk score to risk level
            # Base risk: 0 (neutral) to +X (risky) or -X (safe)
            if risk_score >= 3:
                result['risk_level'] = 'extreme'
                result['regime_summary'] = 'Defensive / High Risk Environment'
            elif risk_score >= 1:
                result['risk_level'] = 'high'
                result['regime_summary'] = 'Caution Warranted'
            elif risk_score <= -2:
                result['risk_level'] = 'low'
                result['regime_summary'] = 'Supportive Growth Environment'
            else:
                result['risk_level'] = 'neutral'
                result['regime_summary'] = 'Neutral / Mixed signals'
                
            result['reasoning'] = reasons
            
        except Exception as e:
            logger.warning(f"Error in macro regime analysis: {e}")
            
        return result
        
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
