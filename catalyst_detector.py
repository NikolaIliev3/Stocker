"""
Catalyst Detector
Detects upcoming earnings, dividends, and other market catalysts
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class CatalystDetector:
    """Detects upcoming market catalysts for trading decisions"""
    
    # Cache for catalyst data
    _catalyst_cache = {}
    _cache_expiry = {}
    CACHE_DURATION_HOURS = 4  # Earnings dates don't change often
    
    def __init__(self, data_fetcher=None):
        """
        Initialize with optional data fetcher
        
        Args:
            data_fetcher: StockDataFetcher instance (optional)
        """
        self.data_fetcher = data_fetcher
    
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid"""
        if symbol not in self._cache_expiry:
            return False
        return datetime.now() < self._cache_expiry[symbol]
    
    def get_earnings_date(self, symbol: str, stock_info: dict = None) -> Optional[datetime]:
        """
        Get next earnings date for a stock
        
        Args:
            symbol: Stock ticker symbol
            stock_info: Optional pre-fetched stock info
            
        Returns:
            Next earnings date or None
        """
        cache_key = f"{symbol}_earnings"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self._catalyst_cache.get(cache_key)
        
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            
            # Try calendar first (most reliable)
            try:
                calendar = ticker.calendar
                if calendar is not None:
                    if isinstance(calendar, pd.DataFrame) and not calendar.empty:
                        # Calendar can have 'Earnings Date' or similar
                        if 'Earnings Date' in calendar.index:
                            earnings_dates = calendar.loc['Earnings Date']
                            if earnings_dates is not None:
                                # Can be a single date or range
                                if isinstance(earnings_dates, pd.Series):
                                    next_date = earnings_dates.iloc[0]
                                else:
                                    next_date = earnings_dates
                                if pd.notna(next_date):
                                    if isinstance(next_date, str):
                                        next_date = pd.to_datetime(next_date)
                                    self._catalyst_cache[cache_key] = next_date
                                    self._cache_expiry[cache_key] = datetime.now() + timedelta(hours=self.CACHE_DURATION_HOURS)
                                    return next_date
                    elif isinstance(calendar, dict):
                        # Some versions return dict
                        earnings_date = calendar.get('Earnings Date')
                        if earnings_date:
                            if isinstance(earnings_date, list) and len(earnings_date) > 0:
                                next_date = pd.to_datetime(earnings_date[0])
                            else:
                                next_date = pd.to_datetime(earnings_date)
                            self._catalyst_cache[cache_key] = next_date
                            self._cache_expiry[cache_key] = datetime.now() + timedelta(hours=self.CACHE_DURATION_HOURS)
                            return next_date
            except Exception as e:
                logger.debug(f"Could not get calendar for {symbol}: {e}")
            
            # Fallback: check earnings_dates attribute
            try:
                earnings = ticker.earnings_dates
                if earnings is not None and not earnings.empty:
                    future_dates = earnings[earnings.index > datetime.now()]
                    if not future_dates.empty:
                        next_date = future_dates.index[0]
                        self._catalyst_cache[cache_key] = next_date
                        self._cache_expiry[cache_key] = datetime.now() + timedelta(hours=self.CACHE_DURATION_HOURS)
                        return next_date
            except Exception as e:
                logger.debug(f"Could not get earnings_dates for {symbol}: {e}")
            
        except Exception as e:
            logger.warning(f"Error getting earnings date for {symbol}: {e}")
        
        return None
    
    def get_dividend_date(self, symbol: str, stock_info: dict = None) -> Optional[datetime]:
        """
        Get next ex-dividend date for a stock
        
        Args:
            symbol: Stock ticker symbol
            stock_info: Optional pre-fetched stock info
            
        Returns:
            Next ex-dividend date or None
        """
        cache_key = f"{symbol}_dividend"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self._catalyst_cache.get(cache_key)
        
        try:
            # Try from provided info first
            if stock_info:
                info = stock_info.get('info', stock_info)
                ex_div = info.get('exDividendDate')
                if ex_div:
                    if isinstance(ex_div, (int, float)):
                        ex_date = datetime.fromtimestamp(ex_div)
                    else:
                        ex_date = pd.to_datetime(ex_div)
                    
                    if ex_date > datetime.now():
                        self._catalyst_cache[cache_key] = ex_date
                        self._cache_expiry[cache_key] = datetime.now() + timedelta(hours=self.CACHE_DURATION_HOURS)
                        return ex_date
            
            # Fetch from yfinance
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            ex_div = info.get('exDividendDate')
            if ex_div:
                if isinstance(ex_div, (int, float)):
                    ex_date = datetime.fromtimestamp(ex_div)
                else:
                    ex_date = pd.to_datetime(ex_div)
                
                if ex_date > datetime.now():
                    self._catalyst_cache[cache_key] = ex_date
                    self._cache_expiry[cache_key] = datetime.now() + timedelta(hours=self.CACHE_DURATION_HOURS)
                    return ex_date
                    
        except Exception as e:
            logger.warning(f"Error getting dividend date for {symbol}: {e}")
        
        return None
    
    def days_to_catalyst(self, catalyst_date: Optional[datetime]) -> Optional[int]:
        """
        Calculate days until a catalyst event
        
        Args:
            catalyst_date: Date of the catalyst
            
        Returns:
            Days until catalyst or None
        """
        if catalyst_date is None:
            return None
        
        try:
            if isinstance(catalyst_date, pd.Timestamp):
                catalyst_date = catalyst_date.to_pydatetime()
            
            # Make timezone-naive for comparison
            if hasattr(catalyst_date, 'tzinfo') and catalyst_date.tzinfo is not None:
                catalyst_date = catalyst_date.replace(tzinfo=None)
            
            delta = catalyst_date - datetime.now()
            return max(0, delta.days)
        except Exception as e:
            logger.warning(f"Error calculating days to catalyst: {e}")
            return None
    
    def detect_catalysts(self, symbol: str, stock_info: dict = None) -> Dict:
        """
        Detect all upcoming catalysts for a stock
        
        Args:
            symbol: Stock ticker symbol
            stock_info: Optional pre-fetched stock info
            
        Returns:
            Dict with catalyst information
        """
        result = {
            'symbol': symbol,
            'earnings_date': None,
            'days_to_earnings': None,
            'dividend_date': None,
            'days_to_dividend': None,
            'has_upcoming_catalyst': False,
            'catalyst_warning': False,
            'catalyst_type': None,
            'available': False
        }
        
        try:
            # Get earnings date
            earnings_date = self.get_earnings_date(symbol, stock_info)
            if earnings_date:
                result['earnings_date'] = earnings_date.strftime('%Y-%m-%d')
                result['days_to_earnings'] = self.days_to_catalyst(earnings_date)
                result['has_upcoming_catalyst'] = True
                result['available'] = True
                
                # Warning if earnings within 7 days
                if result['days_to_earnings'] is not None and result['days_to_earnings'] <= 7:
                    result['catalyst_warning'] = True
                    result['catalyst_type'] = 'earnings'
            
            # Get dividend date
            dividend_date = self.get_dividend_date(symbol, stock_info)
            if dividend_date:
                result['dividend_date'] = dividend_date.strftime('%Y-%m-%d')
                result['days_to_dividend'] = self.days_to_catalyst(dividend_date)
                result['has_upcoming_catalyst'] = True
                result['available'] = True
                
                # Warning if dividend within 3 days (ex-date affects price)
                if result['days_to_dividend'] is not None and result['days_to_dividend'] <= 3:
                    result['catalyst_warning'] = True
                    result['catalyst_type'] = result['catalyst_type'] or 'dividend'
            
        except Exception as e:
            logger.warning(f"Error detecting catalysts for {symbol}: {e}")
        
        return result
    
    def get_catalyst_adjustment(self, catalyst_info: Dict) -> Dict:
        """
        Generate confidence adjustment based on catalyst proximity
        
        Args:
            catalyst_info: Result from detect_catalysts()
            
        Returns:
            Dict with confidence_adjustment and reasoning
        """
        if not catalyst_info.get('available', False):
            return {'confidence_adjustment': 0, 'reasoning': []}
        
        conf_adj = 0
        reasoning = []
        
        days_to_earnings = catalyst_info.get('days_to_earnings')
        days_to_dividend = catalyst_info.get('days_to_dividend')
        
        # Earnings proximity adjustment
        if days_to_earnings is not None:
            if days_to_earnings <= 2:
                conf_adj -= 15
                reasoning.append(f"⚠️ EARNINGS in {days_to_earnings} days - high volatility expected")
            elif days_to_earnings <= 7:
                conf_adj -= 10
                reasoning.append(f"📅 Earnings in {days_to_earnings} days ({catalyst_info['earnings_date']})")
            elif days_to_earnings <= 14:
                reasoning.append(f"📅 Earnings upcoming: {catalyst_info['earnings_date']}")
        
        # Dividend proximity adjustment
        if days_to_dividend is not None:
            if days_to_dividend <= 3:
                conf_adj -= 5
                reasoning.append(f"💰 Ex-dividend in {days_to_dividend} days - expect price drop")
            elif days_to_dividend <= 7:
                reasoning.append(f"💰 Dividend ex-date: {catalyst_info['dividend_date']}")
        
        return {
            'confidence_adjustment': conf_adj,
            'reasoning': reasoning
        }
