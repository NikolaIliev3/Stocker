"""
Backend Proxy Server for Stocker App
Acts as a secure intermediary between the desktop app and stock data APIs
Prevents API key exposure and enables rate limiting, caching, and abuse prevention
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
import time
from functools import wraps
from collections import defaultdict
import logging
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for desktop app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress yfinance's misleading error messages (e.g., "possibly delisted" false positives)
yfinance_logger = logging.getLogger('yfinance')
yfinance_logger.setLevel(logging.WARNING)  # Only show warnings and errors, suppress INFO/DEBUG
# Further suppress the "possibly delisted" messages by filtering
class YFinanceFilter(logging.Filter):
    def filter(self, record):
        # Suppress "possibly delisted" messages - they're often false positives
        if 'possibly delisted' in str(record.getMessage()).lower():
            return False
        return True
yfinance_logger.addFilter(YFinanceFilter())

# Fix SSL certificate issues at startup
import ssl
import certifi
import os
import warnings
import urllib3
import requests

# Suppress SSL warnings
warnings.filterwarnings('ignore', message='Unverified HTTPS request')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Workaround for Windows SSL certificate issues with yfinance/curl
# The issue is that curl (used by yfinance) can't find certificates
# We'll try to point curl to a valid certificate bundle or disable verification

# Try to find and set a valid certificate path
try:
    cert_path = certifi.where()
    if os.path.exists(cert_path):
        # Set for curl (yfinance uses curl)
        os.environ['CURL_CA_BUNDLE'] = cert_path
        os.environ['REQUESTS_CA_BUNDLE'] = cert_path
        os.environ['SSL_CERT_FILE'] = cert_path
        logger.info(f"Using certificate bundle: {cert_path}")
    else:
        # If cert file doesn't exist, disable verification
        logger.warning("Certificate file not found, disabling SSL verification")
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        os.environ['PYTHONHTTPSVERIFY'] = '0'
except Exception as e:
    logger.warning(f"Certificate setup failed: {e}, disabling SSL verification")
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['PYTHONHTTPSVERIFY'] = '0'

# Patch requests library to disable SSL verification
# This is necessary because yfinance uses requests internally
original_get = requests.Session.get
original_post = requests.Session.post
original_request = requests.Session.request

def patched_get(self, url, **kwargs):
    kwargs['verify'] = False
    return original_get(self, url, **kwargs)

def patched_post(self, url, **kwargs):
    kwargs['verify'] = False
    return original_post(self, url, **kwargs)

def patched_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return original_request(self, method, url, **kwargs)

requests.Session.get = patched_get
requests.Session.post = patched_post
requests.Session.request = patched_request

# Also patch the default SSL context
ssl._create_default_https_context = ssl._create_unverified_context

logger.info("SSL verification disabled for yfinance (Windows certificate workaround)")

# Rate limiting storage (in production, use Redis)
rate_limit_store = defaultdict(list)
MAX_REQUESTS_PER_MINUTE = 60

# Cache storage (in production, use Redis or Memcached)
cache = {}
CACHE_DURATION = 300  # 5 minutes

# API Keys (in production, use environment variables or secret management)
# These should NEVER be exposed to the client
STOCK_API_KEY = os.getenv("STOCK_API_KEY", "")
YAHOO_FINANCE_ENABLED = True  # yfinance doesn't need API keys


def rate_limit(f):
    """Rate limiting decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        client_id = request.remote_addr
        current_time = time.time()
        
        # Clean old entries
        rate_limit_store[client_id] = [
            req_time for req_time in rate_limit_store[client_id]
            if current_time - req_time < 60
        ]
        
        # Check rate limit
        if len(rate_limit_store[client_id]) >= MAX_REQUESTS_PER_MINUTE:
            return jsonify({"error": "Rate limit exceeded"}), 429
        
        # Add current request
        rate_limit_store[client_id].append(current_time)
        
        return f(*args, **kwargs)
    return decorated_function


def get_cached(key: str):
    """Get cached data if not expired"""
    if key in cache:
        data, timestamp = cache[key]
        if time.time() - timestamp < CACHE_DURATION:
            return data
        else:
            del cache[key]
    return None


def set_cache(key: str, data):
    """Cache data with timestamp"""
    cache[key] = (data, time.time())


def _fetch_alternative_history(symbol: str, period: str = "1y", interval: str = "1d"):
    """
    Alternative history fetching method using requests directly
    This bypasses yfinance's curl dependency
    """
    try:
        import requests
        from datetime import datetime, timedelta
        import time
        
        # Convert period to days
        period_days = {
            "1d": 1,
            "5d": 5,
            "1mo": 30,
            "3mo": 90,
            "6mo": 180,
            "1y": 365,
            "2y": 730,
            "5y": 1825,
            "10y": 3650,
            "ytd": 365,  # Approximate
            "max": 3650  # Approximate
        }
        days = period_days.get(period.lower(), 365)
        
        # Yahoo Finance API endpoint
        base_url = "https://query1.finance.yahoo.com/v8/finance/chart"
        
        # Get timestamps
        end_time = int(datetime.now().timestamp())
        start_time = int((datetime.now() - timedelta(days=days)).timestamp())
        
        url = f"{base_url}/{symbol}?period1={start_time}&period2={end_time}&interval={interval}"
        
        # Headers to avoid rate limiting
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
        }
        
        # Make request with retry logic
        max_retries = 3
        response = None
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    time.sleep(1)
                
                response = requests.get(url, headers=headers, verify=False, timeout=30)
                
                if response.status_code == 429 and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3
                    logger.warning(f"Rate limited, waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                else:
                    raise
        
        if response is None:
            return None
        
        data = response.json()
        
        if 'chart' not in data or 'result' not in data['chart'] or len(data['chart']['result']) == 0:
            return None
        
        result = data['chart']['result'][0]
        timestamps = result.get('timestamp', [])
        indicators = result.get('indicators', {})
        quote = indicators.get('quote', [{}])[0]
        
        if not timestamps or not quote:
            return None
        
        # Extract price data
        closes = quote.get('close', [])
        opens = quote.get('open', [])
        highs = quote.get('high', [])
        lows = quote.get('low', [])
        volumes = quote.get('volume', [])
        
        if not closes:
            return None
        
        # Build history list
        history_list = []
        for i, ts in enumerate(timestamps):
            if i < len(closes) and closes[i] is not None:
                history_list.append({
                    "date": datetime.fromtimestamp(ts).strftime('%Y-%m-%d'),
                    "open": float(opens[i]) if i < len(opens) and opens[i] else 0.0,
                    "high": float(highs[i]) if i < len(highs) and highs[i] else 0.0,
                    "low": float(lows[i]) if i < len(lows) and lows[i] else 0.0,
                    "close": float(closes[i]),
                    "volume": int(volumes[i]) if i < len(volumes) and volumes[i] else 0
                })
        
        return {
            "symbol": symbol.upper(),
            "period": period,
            "interval": interval,
            "data": history_list
        }
    except Exception as e:
        logger.error(f"Alternative history fetch failed: {e}")
        return None


def _fetch_alternative_data(symbol: str):
    """
    Alternative data fetching method using requests directly
    This bypasses yfinance's curl dependency
    """
    try:
        import requests
        from datetime import datetime, timedelta
        import json
        import time
        
        # Yahoo Finance API endpoint (public, no auth needed)
        base_url = "https://query1.finance.yahoo.com/v8/finance/chart"
        
        # Get current timestamp and 1 year ago
        end_time = int(datetime.now().timestamp())
        start_time = int((datetime.now() - timedelta(days=365)).timestamp())
        
        url = f"{base_url}/{symbol}?period1={start_time}&period2={end_time}&interval=1d"
        
        # Add proper headers to avoid rate limiting
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://finance.yahoo.com/',
        }
        
        # Make request with SSL verification disabled (workaround) and retry logic
        max_retries = 3
        response = None
        for attempt in range(max_retries):
            try:
                # Add small delay to avoid rate limiting (except first attempt)
                if attempt > 0:
                    time.sleep(1)
                
                response = requests.get(url, headers=headers, verify=False, timeout=30)
                
                if response.status_code == 429:
                    # Rate limited - wait and retry
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 3  # 3, 6, 9 seconds
                        logger.warning(f"Rate limited (429), waiting {wait_time} seconds before retry {attempt + 1}/{max_retries}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        raise requests.exceptions.HTTPError(f"Rate limited after {max_retries} attempts", response=response)
                
                response.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Request failed (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(2)
                    continue
                else:
                    raise
        
        if response is None:
            raise Exception("Failed to get response after all retries")
        
        data = response.json()
        
        if 'chart' not in data or 'result' not in data['chart'] or len(data['chart']['result']) == 0:
            return None
        
        result = data['chart']['result'][0]
        meta = result.get('meta', {})
        timestamps = result.get('timestamp', [])
        indicators = result.get('indicators', {})
        quote = indicators.get('quote', [{}])[0]
        
        if not timestamps or not quote:
            return None
        
        # Log meta data for debugging price extraction
        logger.debug(f"Meta data for {symbol}: regularMarketPrice={meta.get('regularMarketPrice')}, "
                    f"previousClose={meta.get('previousClose')}, chartPreviousClose={meta.get('chartPreviousClose')}")
        
        # Extract price data
        closes = quote.get('close', [])
        opens = quote.get('open', [])
        highs = quote.get('high', [])
        lows = quote.get('low', [])
        volumes = quote.get('volume', [])
        
        if not closes:
            return None
        
        # Try to get real-time current price from meta first (most accurate)
        current_price = None
        current_idx = None
        
        # Meta contains various price fields - prioritize real-time price
        if meta:
            # Priority 1: regularMarketPrice (real-time during market hours, most accurate)
            if 'regularMarketPrice' in meta and meta['regularMarketPrice'] is not None:
                current_price = float(meta['regularMarketPrice'])
                logger.info(f"✅ Using REAL-TIME price from meta (regularMarketPrice): {current_price} for {symbol}")
            # Priority 2: Check if there's a current price in the last timestamp
            elif len(timestamps) > 0 and len(closes) > 0:
                # Get the last timestamp and check if it's today
                last_ts = timestamps[-1]
                last_date = datetime.fromtimestamp(last_ts).date()
                today = datetime.now().date()
                
                # If last data point is from today, use it (might be intraday)
                if last_date == today and closes[-1] is not None and closes[-1] > 0:
                    current_price = float(closes[-1])
                    current_idx = len(closes) - 1
                    logger.info(f"✅ Using today's latest close price: {current_price} for {symbol}")
            # Priority 3: Use most recent non-null close from historical data
            if current_price is None or current_price <= 0:
                for i in range(len(closes) - 1, -1, -1):
                    if closes[i] is not None and closes[i] > 0:
                        current_price = float(closes[i])
                        current_idx = i
                        logger.info(f"Using most recent historical close price: {current_price} for {symbol} (index {i})")
                        break
        
        if current_price is None or current_price <= 0:
            logger.warning(f"No valid current price found for {symbol} in alternative data")
            return None
        
        # Get previous price (for change calculation)
        previous_price = current_price
        if current_idx is not None and current_idx > 0:
            # We have an index, find the previous valid price
            for i in range(current_idx - 1, -1, -1):
                if i < len(closes) and closes[i] is not None and closes[i] > 0:
                    previous_price = float(closes[i])
                    break
        else:
            # current_idx is None (we used meta price), find the most recent close from history
            for i in range(len(closes) - 1, -1, -1):
                if closes[i] is not None and closes[i] > 0:
                    # Use this as previous price (it's the last close we have)
                    previous_price = float(closes[i])
                    break
        
        # Build history list
        history_list = []
        for i, ts in enumerate(timestamps[-100:]):  # Last 100 days
            idx = len(timestamps) - 100 + i if len(timestamps) > 100 else i
            if idx < len(closes) and closes[idx] is not None and closes[idx] > 0:
                history_list.append({
                    "date": datetime.fromtimestamp(ts).strftime('%Y-%m-%d'),
                    "open": float(opens[idx]) if idx < len(opens) and opens[idx] is not None else 0.0,
                    "high": float(highs[idx]) if idx < len(highs) and highs[idx] is not None else 0.0,
                    "low": float(lows[idx]) if idx < len(lows) and lows[idx] is not None else 0.0,
                    "close": float(closes[idx]),
                    "volume": int(volumes[idx]) if idx < len(volumes) and volumes[idx] is not None else 0
                })
        
        # Get the most recent non-null values for other fields (use same index as current_price)
        current_volume = 0
        current_high = current_price
        current_low = current_price
        current_open = current_price
        
        if current_idx is not None and current_idx >= 0:
            # We have a valid index, use it
            if current_idx < len(volumes) and volumes[current_idx] is not None:
                current_volume = int(volumes[current_idx])
            if current_idx < len(highs) and highs[current_idx] is not None:
                current_high = float(highs[current_idx])
            if current_idx < len(lows) and lows[current_idx] is not None:
                current_low = float(lows[current_idx])
            if current_idx < len(opens) and opens[current_idx] is not None:
                current_open = float(opens[current_idx])
        else:
            # current_idx is None (we used meta price), use the most recent values from history
            for i in range(len(volumes) - 1, -1, -1):
                if i < len(volumes) and volumes[i] is not None:
                    current_volume = int(volumes[i])
                    break
            for i in range(len(highs) - 1, -1, -1):
                if i < len(highs) and highs[i] is not None:
                    current_high = float(highs[i])
                    break
            for i in range(len(lows) - 1, -1, -1):
                if i < len(lows) and lows[i] is not None:
                    current_low = float(lows[i])
                    break
            for i in range(len(opens) - 1, -1, -1):
                if i < len(opens) and opens[i] is not None:
                    current_open = float(opens[i])
                    break
        
        return {
            "symbol": symbol.upper(),
            "price": current_price,
            "volume": current_volume,
            "high": current_high,
            "low": current_low,
            "open": current_open,
            "previous_close": previous_price,
            "change": current_price - previous_price,
            "change_percent": ((current_price - previous_price) / previous_price * 100) if previous_price > 0 else 0.0,
            "history": history_list,
            "info": {
                "name": meta.get("longName", ""),
                "sector": meta.get("sector", ""),
                "industry": meta.get("industry", ""),
                "market_cap": meta.get("marketCap", 0),
                "pe_ratio": meta.get("trailingPE", None),
                "dividend_yield": meta.get("dividendYield", None),
            }
        }
    except Exception as e:
        logger.error(f"Alternative data fetch failed: {e}", exc_info=True)
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


@app.after_request
def add_security_headers(response):
    """Add security headers to all responses"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "service": "stocker-backend"})


@app.route('/api/stock/<symbol>', methods=['GET'])
@rate_limit
def get_stock_data(symbol: str):
    """
    Get stock data for a symbol
    This endpoint proxies requests to stock data providers
    """
    try:
        # Check if client wants to bypass cache (force refresh)
        force_refresh = request.args.get('_refresh') is not None
        
        # Check cache first (unless force refresh is requested)
        cache_key = f"stock_{symbol.upper()}"
        if not force_refresh:
            cached_data = get_cached(cache_key)
            if cached_data:
                logger.debug(f"Cache hit for {symbol}")  # Changed to debug to reduce log spam
                return jsonify(cached_data)
        else:
            logger.debug(f"Force refresh requested for {symbol}, bypassing cache")
        
        # Fetch from data provider
        # Note: yfinance often has issues (SSL, "delisted" false positives), so we try alternative method first
        # Try alternative method first for better reliability
        alt_data = _fetch_alternative_data(symbol.upper())
        if alt_data:
            # Alternative method succeeded - use it (more reliable than yfinance)
            if not force_refresh:
                set_cache(cache_key, alt_data)
            logger.debug(f"Using alternative data fetch for {symbol} (more reliable)")
            return jsonify(alt_data)
        
        # Fallback to yfinance if alternative method fails
        try:
            import yfinance as yf
            # Suppress yfinance's misleading "delisted" warnings
            import warnings
            warnings.filterwarnings('ignore', message='.*possibly delisted.*', category=UserWarning)
            
            # Approach 1: Try using yfinance.download() directly with session
            # This sometimes works better than Ticker for SSL issues
            try:
                import yfinance.utils as yf_utils
                # Try to get data using download function
                hist = yf_utils.download(
                    symbol.upper(),
                    period="1y",
                    interval="1d",
                    progress=False,
                    show_errors=False
                )
                
                if hist is not None and not hist.empty and symbol.upper() in hist.columns.levels[1] if hasattr(hist.columns, 'levels') else True:
                    # Success with download approach
                    if hasattr(hist.columns, 'levels'):
                        # Multi-index columns - extract the symbol
                        hist = hist.xs(symbol.upper(), level=1, axis=1)
                    ticker = None  # We got data, don't need ticker
                    info = {}
                else:
                    raise Exception("Download returned empty data")
            except Exception as download_err:
                logger.warning(f"Direct download failed: {download_err}, trying Ticker approach")
                hist = None
                ticker = yf.Ticker(symbol.upper())
            
            # Approach 2: Use Ticker if download didn't work
            if hist is None:
                ticker = yf.Ticker(symbol.upper())
                
                # Pre-initialize info to prevent None errors
                if not hasattr(ticker, '_info') or ticker._info is None:
                    ticker._info = {}
                
                # Try to get info, but don't fail if it doesn't work
                try:
                    info_result = ticker.info
                    if info_result and isinstance(info_result, dict) and len(info_result) > 0:
                        ticker._info = info_result
                    else:
                        ticker._info = {}
                except Exception as info_err:
                    logger.warning(f"Could not fetch info: {info_err}")
                    ticker._info = {}
                
                info = ticker._info if hasattr(ticker, '_info') and ticker._info else {}
                
                # Get history - this is the critical part
                # yfinance may have issues, so we'll try multiple approaches
                hist_error = None
                try:
                    hist = ticker.history(period="1y", timeout=30)
                except Exception as hist_err:
                    hist_error = str(hist_err)
                    logger.warning(f"Error fetching 1y history: {hist_err}")
                    try:
                        hist = ticker.history(period="6mo", timeout=30)
                        hist_error = None  # Success
                    except Exception as hist_err2:
                        hist_error = str(hist_err2)
                        logger.warning(f"Error fetching 6mo history: {hist_err2}")
                        try:
                            hist = ticker.history(period="3mo", timeout=30)
                            hist_error = None  # Success
                        except Exception as hist_err3:
                            hist_error = str(hist_err3)
                            logger.error(f"Error fetching 3mo history: {hist_err3}")
            else:
                # Download approach worked, we already have hist
                info = {}
            
            # If we got empty data, try alternative method (yfinance often fails silently with curl issues)
            if hist is None or hist.empty:
                logger.debug(f"yfinance returned empty data for {symbol}, trying alternative method...")
                alt_data = _fetch_alternative_data(symbol.upper())
                if alt_data:
                    # Don't cache if force_refresh was requested
                    if not force_refresh:
                        set_cache(cache_key, alt_data)
                    logger.debug(f"Alternative data fetch succeeded for {symbol}")
                    return jsonify(alt_data)
                else:
                    # If alternative also fails, return error
                    return jsonify({"error": f"No data found for symbol {symbol}. This may be due to SSL certificate issues with yfinance on Windows."}), 404
            
            # Validate we have required columns
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in hist.columns]
            if missing_cols:
                return jsonify({"error": f"Missing required data columns: {missing_cols}"}), 500
                
        except ImportError:
            logger.error("yfinance not installed")
            return jsonify({"error": "Stock data provider not available. Please install yfinance."}), 500
        except Exception as e:
            error_str = str(e).lower()
            logger.error(f"Error fetching from yfinance: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            # If yfinance fails due to SSL/curl, try alternative approach
            if any(keyword in error_str for keyword in ["curl", "certificate", "ssl", "77", "noneType", "delisted"]):
                logger.debug(f"Attempting alternative data fetch method for {symbol} (bypassing curl)...")
                try:
                    # Try using requests directly to Yahoo Finance (simpler, bypasses curl)
                    alt_data = _fetch_alternative_data(symbol.upper())
                    if alt_data:
                        # Don't cache if force_refresh was requested
                        if not force_refresh:
                            set_cache(cache_key, alt_data)
                        logger.debug(f"Alternative data fetch succeeded for {symbol}")
                        return jsonify(alt_data)
                    else:
                        logger.warning(f"Alternative data fetch returned no data for {symbol}")
                except Exception as alt_err:
                    logger.error(f"Alternative fetch also failed for {symbol}: {alt_err}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            return jsonify({
                "error": f"Failed to fetch data: {str(e)}",
                "hint": "This is often a Windows SSL certificate issue with yfinance. The app tried an alternative method but it also failed."
            }), 500
        
        # Prepare response
        try:
            current_price = float(hist['Close'].iloc[-1])
            current_volume = int(hist['Volume'].iloc[-1]) if not pd.isna(hist['Volume'].iloc[-1]) else 0
            
            # Convert history to JSON-serializable format
            history_list = []
            for date, row in hist.tail(100).iterrows():  # Limit to last 100 days
                try:
                    history_list.append({
                        "date": date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                        "open": float(row['Open']) if not pd.isna(row['Open']) else 0.0,
                        "high": float(row['High']) if not pd.isna(row['High']) else 0.0,
                        "low": float(row['Low']) if not pd.isna(row['Low']) else 0.0,
                        "close": float(row['Close']) if not pd.isna(row['Close']) else 0.0,
                        "volume": int(row['Volume']) if not pd.isna(row['Volume']) else 0
                    })
                except Exception as e:
                    logger.warning(f"Error converting history row: {e}")
                    continue
            
            # Get info safely
            info_dict = {}
            if info:
                info_dict = {
                    "name": info.get("longName") or info.get("shortName") or "",
                    "sector": info.get("sector") or "",
                    "industry": info.get("industry") or "",
                    "market_cap": info.get("marketCap") or 0,
                    "pe_ratio": info.get("trailingPE") or None,
                    "dividend_yield": info.get("dividendYield") or None,
                }
            
            previous_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
            change = current_price - previous_close if len(hist) > 1 else 0.0
            change_percent = (change / previous_close * 100) if previous_close > 0 and len(hist) > 1 else 0.0
            
            data = {
                "symbol": symbol.upper(),
                "price": current_price,
                "volume": current_volume,
                "high": float(hist['High'].iloc[-1]) if not pd.isna(hist['High'].iloc[-1]) else current_price,
                "low": float(hist['Low'].iloc[-1]) if not pd.isna(hist['Low'].iloc[-1]) else current_price,
                "open": float(hist['Open'].iloc[-1]) if not pd.isna(hist['Open'].iloc[-1]) else current_price,
                "previous_close": previous_close,
                "change": change,
                "change_percent": change_percent,
                "history": history_list,
                "info": info_dict
            }
        except Exception as e:
            logger.error(f"Error preparing response data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
        
        # Cache the result (unless force refresh was requested)
        if not force_refresh:
            set_cache(cache_key, data)
        else:
            logger.debug(f"Skipping cache for {symbol} (force refresh requested)")
        
        return jsonify(data)
    
    except Exception as e:
        logger.error(f"Error fetching stock data for {symbol}: {str(e)}")
        return jsonify({"error": f"Failed to fetch stock data: {str(e)}"}), 500


@app.route('/api/stock/<symbol>/history', methods=['GET'])
@rate_limit
def get_stock_history(symbol: str):
    """Get historical stock data for charting"""
    try:
        period = request.args.get('period', '1y')
        interval = request.args.get('interval', '1d')
        
        cache_key = f"history_{symbol}_{period}_{interval}"
        cached_data = get_cached(cache_key)
        if cached_data:
            return jsonify(cached_data)
        
        # Try alternative method first (more reliable than yfinance)
        alt_data = _fetch_alternative_history(symbol.upper(), period, interval)
        if alt_data:
            set_cache(cache_key, alt_data)
            logger.debug(f"Using alternative history fetch for {symbol}")
            return jsonify(alt_data)
        
        # Fallback to yfinance if alternative method fails
        try:
            import yfinance as yf
            import warnings
            # Suppress yfinance's misleading "delisted" warnings
            warnings.filterwarnings('ignore', message='.*possibly delisted.*', category=UserWarning)
            
            # SSL verification is already disabled at startup
            ticker = yf.Ticker(symbol.upper())
            
            # Handle None info gracefully
            if not hasattr(ticker, '_info') or ticker._info is None:
                ticker._info = {}
            
            hist = ticker.history(period=period, interval=interval, timeout=30)
            
            # If yfinance fails, try alternative method again
            if hist is None or hist.empty:
                logger.debug(f"yfinance returned empty history for {symbol}, trying alternative method...")
                alt_data = _fetch_alternative_history(symbol.upper(), period, interval)
                if alt_data:
                    set_cache(cache_key, alt_data)
                    return jsonify(alt_data)
                else:
                    return jsonify({"error": f"No history found for {symbol}"}), 404
            
            # Convert history to JSON-serializable format
            history_list = []
            for date, row in hist.iterrows():
                try:
                    history_list.append({
                        "date": date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
                        "open": float(row['Open']) if not pd.isna(row['Open']) else 0.0,
                        "high": float(row['High']) if not pd.isna(row['High']) else 0.0,
                        "low": float(row['Low']) if not pd.isna(row['Low']) else 0.0,
                        "close": float(row['Close']) if not pd.isna(row['Close']) else 0.0,
                        "volume": int(row['Volume']) if not pd.isna(row['Volume']) else 0
                    })
                except Exception as e:
                    logger.warning(f"Error converting history row: {e}")
                    continue
            
            data = {
                "symbol": symbol.upper(),
                "period": period,
                "interval": interval,
                "data": history_list
            }
            
            set_cache(cache_key, data)
            return jsonify(data)
        except Exception as e:
            error_str = str(e).lower()
            logger.error(f"Error fetching history: {e}")
            
            # If it's an SSL/curl error, try alternative
            if any(keyword in error_str for keyword in ["curl", "certificate", "ssl", "77", "nonetype"]):
                logger.info("yfinance history failed due to SSL/curl, trying alternative method...")
                alt_data = _fetch_alternative_history(symbol.upper(), period, interval)
                if alt_data:
                    set_cache(cache_key, alt_data)
                    return jsonify(alt_data)
            
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Failed to fetch history: {str(e)}"}), 500
    
    except Exception as e:
        logger.error(f"Error fetching history for {symbol}: {str(e)}")
        return jsonify({"error": f"Failed to fetch history: {str(e)}"}), 500


@app.route('/api/stock/<symbol>/financials', methods=['GET'])
@rate_limit
def get_financials(symbol: str):
    """Get financial statements for investing strategy"""
    try:
        cache_key = f"financials_{symbol}"
        cached_data = get_cached(cache_key)
        if cached_data:
            return jsonify(cached_data)
        
        try:
            import yfinance as yf
            
            # SSL verification is already disabled at startup
            ticker = yf.Ticker(symbol.upper())
            
            # Handle None info gracefully
            if not hasattr(ticker, '_info') or ticker._info is None:
                ticker._info = {}
            
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cashflow = ticker.cashflow
        except Exception as e:
            logger.error(f"Error fetching financials: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({"error": f"Failed to fetch financials: {str(e)}"}), 500
        
        # Convert DataFrames to JSON-serializable format
        def df_to_dict_safe(df):
            """Safely convert DataFrame to dict"""
            if df.empty:
                return {}
            try:
                # Convert to dict and handle NaN values
                result = {}
                for col in df.columns:
                    result[str(col)] = {}
                    for idx, val in df[col].items():
                        key = str(idx) if hasattr(idx, 'strftime') else idx.strftime('%Y-%m-%d')
                        result[str(col)][key] = float(val) if not pd.isna(val) else None
                return result
            except Exception as e:
                logger.warning(f"Error converting DataFrame: {e}")
                return {}
        
        data = {
            "symbol": symbol.upper(),
            "financials": df_to_dict_safe(financials),
            "balance_sheet": df_to_dict_safe(balance_sheet),
            "cashflow": df_to_dict_safe(cashflow),
        }
        
        set_cache(cache_key, data)
        return jsonify(data)
    
    except Exception as e:
        logger.error(f"Error fetching financials for {symbol}: {str(e)}")
        return jsonify({"error": f"Failed to fetch financials: {str(e)}"}), 500


if __name__ == '__main__':
    # In production, use a proper WSGI server like gunicorn
    app.run(host='0.0.0.0', port=5000, debug=False)

