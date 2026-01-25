"""
Data fetching module for Stocker App
Handles secure communication with backend proxy
Falls back to direct yfinance if backend is unavailable
Integrated with data provider abstraction and data quality checks
"""
import requests
import logging
from security import get_secure_session, DataValidator, SecurityError
from config import BACKEND_API_URL, APP_DATA_DIR
import json
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time
import random
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*Timestamp.utcnow is deprecated.*')
try:
    from pandas.errors import Pandas4Warning
    warnings.filterwarnings('ignore', category=Pandas4Warning)
except ImportError:
    pass

# Import new modules
try:
    from data_provider_abstraction import MultiDataProvider, YahooFinanceProvider
    from data_quality import DataQualityChecker
    from error_handler import ErrorHandler, safe_call
    HAS_NEW_MODULES = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"New modules not available: {e}")
    HAS_NEW_MODULES = False

logger = logging.getLogger(__name__)


class StockDataFetcher:
    """Fetches stock data securely from backend proxy, with fallback to direct yfinance"""
    
    def __init__(self):
        self.session = get_secure_session()
        self.base_url = BACKEND_API_URL
        self._backend_available = None  # Cache backend availability check
        self._last_backend_check = 0  # Timestamp of last check
        self._backend_check_interval = 30  # Recheck every 30 seconds
        
        # Initialize new modules if available
        if HAS_NEW_MODULES:
            try:
                self.data_provider = MultiDataProvider([YahooFinanceProvider()])
                self.quality_checker = DataQualityChecker()
                logger.info("Data provider abstraction and quality checker initialized")
            except Exception as e:
                logger.warning(f"Could not initialize new modules: {e}")
                self.data_provider = None
                self.quality_checker = None
        else:
            self.data_provider = None
            self.quality_checker = None
    
    def _check_backend_available(self) -> bool:
        """Check if backend server is available"""
        import time
        current_time = time.time()
        
        # Recheck periodically (every 30 seconds) to detect when backend becomes available
        if (self._backend_available is None or 
            current_time - self._last_backend_check > self._backend_check_interval):
            try:
                health_url = f"{self.base_url.replace('/api', '')}/api/health"
                response = self.session.get(health_url, timeout=2)
                is_available = response.status_code == 200
                was_available = self._backend_available
                self._backend_available = is_available
                self._last_backend_check = current_time
                
                # Log when backend becomes available
                if is_available and (was_available is False or was_available is None):
                    logger.info("✅ Backend server is now available")
                elif not is_available and was_available is True:
                    logger.warning("⚠️ Backend server became unavailable")
                
                return is_available
            except Exception as e:
                was_available = self._backend_available
                self._backend_available = False
                self._last_backend_check = current_time
                if was_available is True:
                    logger.warning(f"⚠️ Backend server check failed: {e}")
                return False
        
        # Return cached value if check interval hasn't passed
        return self._backend_available is True
    
    # Class-level verification timestamp to prevent multiple threads hammering API when rate limited
    _global_rate_limit_cooldown = 0
    
    def _retry_operation(self, operation, max_retries=5, base_delay=3.0):
        """Execute operation with retry logic for rate limits"""
        last_exception = None
        
        # Check global cooldown first
        current_time = time.time()
        if current_time < self._global_rate_limit_cooldown:
            wait_time = self._global_rate_limit_cooldown - current_time
            logger.warning(f"Global rate limit cooldown active. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)
        
        for attempt in range(max_retries):
            try:
                # Provide a small randomized delay between any request to be polite
                if attempt > 0:
                    time.sleep(random.random() * 0.5)
                    
                return operation()
            except Exception as e:
                last_exception = e
                error_msg = str(e).lower()
                
                # Check for rate limit indicators
                is_rate_limit = (
                    "rate limited" in error_msg or 
                    "too many requests" in error_msg or 
                    "429" in error_msg or
                    "connection refused" in error_msg or
                    "read timed out" in error_msg
                )
                
                if is_rate_limit:
                    # Set global cooldown on the first sign of trouble to protect other threads
                    # Cooldown increases with attempts
                    cooldown_duration = (base_delay * (2 ** attempt)) 
                    StockDataFetcher._global_rate_limit_cooldown = time.time() + cooldown_duration
                    
                    if attempt < max_retries - 1:
                        # Exponential backoff: 3s, 6s, 12s, 24s... + jitter
                        sleep_time = cooldown_duration + (random.random() * 2.0)
                        logger.warning(f"Rate limited (attempt {attempt+1}/{max_retries}). Global cooldown set for {cooldown_duration:.1f}s. Retrying in {sleep_time:.2f}s...")
                        time.sleep(sleep_time)
                        continue
                
                # If valid error but not rate limit, or max retries reached, raise it
                raise e
                
        if last_exception:
            raise last_exception
    
    def _fetch_direct_yfinance(self, symbol: str) -> dict:
        """Fetch stock data directly using yfinance (fallback method)"""
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol.upper())
            
            # Get info
            try:
                info = ticker.info
                if not info or not isinstance(info, dict):
                    info = {}
            except:
                info = {}
            
            # Get history
            def fetch_history_op():
                h = ticker.history(period="1y", timeout=30)
                if h is None or h.empty:
                    # Try shorter period
                    h = ticker.history(period="6mo", timeout=30)
                if h is None or h.empty:
                    h = ticker.history(period="3mo", timeout=30)
                return h
                
            hist = self._retry_operation(fetch_history_op)
            
            if hist is None or hist.empty:
                raise Exception("No historical data available")
            
            # Get current price
            current_price = float(hist['Close'].iloc[-1])
            current_volume = int(hist['Volume'].iloc[-1])
            
            # Calculate previous close
            previous_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
            change = current_price - previous_close
            change_percent = (change / previous_close * 100) if previous_close > 0 else 0.0
            
            # Convert history to list
            history_list = []
            for date, row in hist.iterrows():
                history_list.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "close": float(row['Close']),
                    "volume": int(row['Volume'])
                })
            
            # Build data dict
            data = {
                "symbol": symbol.upper(),
                "price": current_price,
                "volume": current_volume,
                "high": float(hist['High'].iloc[-1]),
                "low": float(hist['Low'].iloc[-1]),
                "open": float(hist['Open'].iloc[-1]),
                "previous_close": previous_close,
                "change": change,
                "change_percent": change_percent,
                "history": history_list,
                "info": info
            }
            
            # Validate data
            if not DataValidator.validate_stock_data(data):
                raise SecurityError("Invalid stock data from yfinance")
            
            logger.info(f"Fetched {symbol} data directly from yfinance (backend unavailable)")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data directly from yfinance: {e}")
            raise
    
    def fetch_stock_data(self, symbol: str, force_refresh: bool = False, 
                        check_quality: bool = True) -> dict:
        """Fetch current stock data for a symbol
        
        Args:
            symbol: Stock symbol to fetch
            force_refresh: If True, bypasses cache to get fresh data
            check_quality: If True, perform data quality checks
        """
        # Normalize symbol (e.g., BRK.B -> BRK-B)
        original_symbol = symbol
        symbol = symbol.replace('.', '-')
        
        # Try to use data provider abstraction first (it now has robust direct API access)
        if self.data_provider:
            try:
                # If backend is available, we might still want to try it first for caching benefits
                # But if we have direct provider access, it's often faster and more reliable
                provider_data = self.data_provider.fetch_stock_data(symbol)
                
                # If direct provider works, use it (it has its own internal fallback)
                if provider_data:
                    # Convert to expected format
                    data = {
                        "symbol": symbol.upper(),
                        "price": provider_data.get('currentPrice', 0),
                        "info": provider_data.get('info', {})
                    }
                    
                    # Fill in other fields if available in provider_data
                    for field in ['volume', 'high', 'low', 'open', 'previous_close', 'change', 'change_percent']:
                        if field in provider_data:
                            data[field] = provider_data[field]
                    
                    # Get history for quality check
                    history = self.data_provider.fetch_history(symbol)
                    if history and 'data' in history:
                        data['history'] = history['data']
                    
                    # Perform quality check if requested
                    if check_quality and self.quality_checker and history:
                        quality_report = self.quality_checker.check_stock_data_quality(
                            data, history
                        )
                        data['quality_report'] = quality_report
                        
                        # Auto-clean if quality is poor
                        if quality_report.get('overall_score', 100) < 60:
                            logger.warning(f"Data quality low ({quality_report['overall_score']:.1f}), cleaning...")
                            cleaned = self.quality_checker.clean_data(history, quality_report)
                            if cleaned.get('cleaned'):
                                data['history'] = cleaned['data']
                                logger.info(f"Cleaned {cleaned.get('rows_removed', 0)} invalid rows")
                    
                    return data
            except Exception as e:
                logger.debug(f"Data provider failed: {e}")

        # Fallback to backend if data provider failed (or if not available)
        # Try backend if available (prioritized for better reliability/workarounds)
        if self._check_backend_available():
            try:
                url = f"{self.base_url}/stock/{symbol}"
                params = {}
                if force_refresh:
                    # Add cache-busting parameter to force fresh data
                    import time
                    params['_refresh'] = str(int(time.time() * 1000))
                
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                # Validate data integrity
                if not DataValidator.validate_stock_data(data):
                    raise SecurityError("Invalid stock data received from server")
                
                # Perform quality check if requested
                if check_quality and self.quality_checker:
                    history_data = self.fetch_stock_history(symbol)
                    if history_data and 'data' in history_data:
                        quality_report = self.quality_checker.check_stock_data_quality(
                            data, history_data
                        )
                        data['quality_report'] = quality_report
                
                return data
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Backend request failed: {e}, falling back to other providers")
                # Don't mark as unavailable here, just continue to other providers
            except Exception as e:
                logger.warning(f"Backend error: {e}, falling back to other providers")
        
        # Fallback to direct yfinance (last resort)
        try:
            return self._fetch_direct_yfinance(symbol)
        except Exception as e:
            # If normalized symbol failed, try original symbol if it's different
            if original_symbol != symbol:
                logger.info(f"Retrying with original symbol: {original_symbol}")
                try:
                    return self._fetch_direct_yfinance(original_symbol)
                except Exception as e2:
                    logger.error(f"Both normalized ({symbol}) and original ({original_symbol}) symbols failed: {e2}")
                    raise e2
            raise e

    def fetch_batch(self, symbols: list, max_workers: int = 10, 
                   force_refresh: bool = False, check_quality: bool = True,
                   progress_callback=None) -> dict:
        """
        Fetch data for multiple stocks in parallel
        
        Args:
            symbols: List of stock symbols
            max_workers: Maximum number of concurrent requests
            force_refresh: Whether to force fresh data
            check_quality: Whether to check data quality
            progress_callback: Optional callback(current, total)
        
        Returns:
            Dict mapping symbol -> data
        """
        results = {}
        total = len(symbols)
        
        # Use ThreadPoolExecutor for parallel fetching
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # Adjust workers based on list size but cap at reasonable limit
        actual_workers = min(max_workers, total)
        
        logger.info(f"Fetching batch of {total} stocks with {actual_workers} workers...")
        
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # Create a future for each symbol
            future_to_symbol = {
                executor.submit(self.fetch_stock_data, symbol, force_refresh, check_quality): symbol 
                for symbol in symbols
            }
            
            completed = 0
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                completed += 1
                
                try:
                    data = future.result()
                    if data and 'error' not in data:
                        results[symbol] = data
                    else:
                        logger.warning(f"Failed to fetch batch data for {symbol}")
                except Exception as e:
                    logger.warning(f"Error fetching {symbol} in batch: {e}")
                
                if progress_callback:
                    progress_callback(completed, total)
        
        return results
    
    def fetch_stock_history(self, symbol: str, period: str = "1y", interval: str = "1d", 
                           start_date: str = None, end_date: str = None) -> dict:
        """Fetch historical stock data for charting
        
        Args:
            symbol: Stock symbol
            period: Period string (e.g., "1y", "6mo") - used if start_date/end_date not provided
            interval: Data interval (e.g., "1d", "1wk")
            start_date: Start date in YYYY-MM-DD format (optional, overrides period)
            end_date: End date in YYYY-MM-DD format (optional, overrides period)
        """
        # Normalize symbol (e.g., BRK.B -> BRK-B)
        symbol = symbol.replace('.', '-')

        # If start_date and end_date are provided, use them directly with yfinance
        if start_date and end_date:
            try:
                def fetch_op():
                    import yfinance as yf
                    ticker = yf.Ticker(symbol.upper())
                    return ticker.history(start=start_date, end=end_date, interval=interval, timeout=30)
                
                # Use retry logic
                hist = self._retry_operation(fetch_op)
                
                if hist is None or hist.empty:
                    return {"error": "No historical data available", "data": []}
                
                # Convert to list format
                history_list = []
                for date, row in hist.iterrows():
                    volume_val = row['Volume']
                    volume = int(volume_val) if not pd.isna(volume_val) else 0
                    history_list.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "open": float(row['Open']),
                        "high": float(row['High']),
                        "low": float(row['Low']),
                        "close": float(row['Close']),
                        "volume": volume
                    })
                
                return {"data": history_list}
                
            except Exception as e:
                logger.error(f"Error fetching stock history with dates: {e}")
                return {"error": str(e), "data": []}
        
        # Try using data provider abstraction first (it has robust direct API access)
        if self.data_provider:
            try:
                # If backend is available, we might still want to try it first for caching benefits
                # But if we have direct provider access, it's often faster and more reliable
                history = self.data_provider.fetch_history(symbol, period=period)
                if history and 'data' in history:
                    return history  # It's already in {"data": [...]} format
            except Exception as e:
                logger.debug(f"Data provider history fetch failed: {e}")

        # Fallback to backend if data provider failed (or if not available)
        # Try backend if available (prioritized for better reliability/workarounds)
        if self._check_backend_available():
            try:
                url = f"{self.base_url}/stock/{symbol}/history"
                params = {"period": period, "interval": interval}
                
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                # Backend returns {"data": [...]} format
                if "data" in data:
                    return data
                # Legacy format {"history": [...]} - convert to new format
                if "history" in data:
                    return {"data": data["history"]}
                # If neither, return as-is (might be error format)
                return data
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Backend history request failed: {e}, falling back to direct yfinance")
                # Don't mark as unavailable here, just continue to other providers
            except Exception as e:
                logger.warning(f"Backend history error: {e}, falling back to direct yfinance")
        
        # Fallback to direct yfinance
        try:
            def fetch_op():
                import yfinance as yf
                ticker = yf.Ticker(symbol.upper())
                return ticker.history(period=period, interval=interval, timeout=30)
            
            # Use retry logic
            hist = self._retry_operation(fetch_op)
            
            if hist is None or hist.empty:
                raise Exception("No historical data available")
            
            # Convert to list format
            history_list = []
            for date, row in hist.iterrows():
                volume_val = row['Volume']
                volume = int(volume_val) if not pd.isna(volume_val) else 0
                history_list.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "open": float(row['Open']),
                    "high": float(row['High']),
                    "low": float(row['Low']),
                    "close": float(row['Close']),
                    "volume": volume
                })
            
            return {"data": history_list}
            
        except Exception as e:
            logger.error(f"Error fetching stock history: {e}")
            raise
    
    def fetch_financials(self, symbol: str) -> dict:
        """Fetch financial statements for investing analysis"""
        # Normalize symbol (e.g., BRK.B -> BRK-B)
        symbol = symbol.replace('.', '-')

        # Try backend first if available
        if self._check_backend_available():
            try:
                url = f"{self.base_url}/stock/{symbol}/financials"
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                
                return response.json()
            
            except requests.exceptions.RequestException as e:
                logger.warning(f"Backend financials request failed: {e}, falling back to direct yfinance")
                self._backend_available = False
            except Exception as e:
                logger.warning(f"Backend financials error: {e}, falling back to direct yfinance")
                self._backend_available = False
        
        # Fallback to direct yfinance
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol.upper())
            
            # Get financial statements
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cashflow = ticker.cashflow
            
            # Convert DataFrames to dict
            def df_to_dict_safe(df):
                if df.empty:
                    return {}
                result = {}
                for col in df.columns:
                    result[str(col)] = {}
                    for idx, val in df[col].items():
                        key = str(idx) if not hasattr(idx, 'strftime') else idx.strftime('%Y-%m-%d')
                        result[str(col)][key] = float(val) if not pd.isna(val) else None
                return result
            
            return {
                "symbol": symbol.upper(),
                "financials": df_to_dict_safe(financials),
                "balance_sheet": df_to_dict_safe(balance_sheet),
                "cashflow": df_to_dict_safe(cashflow)
            }
            
        except Exception as e:
            logger.error(f"Error fetching financials: {e}")
            raise

