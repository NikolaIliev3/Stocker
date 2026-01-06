"""
Seeker AI Module - Security Enhanced
Automatically researches stocks using news, sentiment analysis, and LLM insights
WITH COMPREHENSIVE SECURITY MEASURES
"""
import requests
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json
from pathlib import Path
import os
from cryptography.fernet import Fernet

from security import get_secure_session, SecurityError, DataValidator
from config import (
    APP_DATA_DIR, SEEKER_AI_ENABLED, SEEKER_AI_RATE_LIMIT,
    SEEKER_AI_CACHE_HOURS, SEEKER_AI_AUDIT_LOGGING,
    API_KEYS_ENCRYPTED, API_KEYS_FILE_PERMISSIONS
)
from secure_logging import SecureLogger
from input_validator import InputValidator
from secure_api_client import SecureAPIClient
from response_validator import ResponseValidator
from anomaly_detector import AnomalyDetector
from data_retention import DataRetentionManager

logger = logging.getLogger(__name__)


class SecureCredentialManager:
    """Manages API keys securely with encryption"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.keys_file = data_dir / ".api_keys.encrypted"
        self._encryption_key = self._get_or_create_encryption_key()
        self._cipher = Fernet(self._encryption_key)
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for API keys"""
        key_file = self.data_dir / ".encryption_key"
        
        if key_file.exists():
            try:
                with open(key_file, 'rb') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading encryption key: {e}")
                # Generate new key if read fails
                return self._generate_new_key(key_file)
        else:
            return self._generate_new_key(key_file)
    
    def _generate_new_key(self, key_file: Path) -> bytes:
        """Generate new encryption key"""
        key = Fernet.generate_key()
        try:
            # Set restrictive permissions (Unix)
            with open(key_file, 'wb') as f:
                f.write(key)
            try:
                os.chmod(key_file, API_KEYS_FILE_PERMISSIONS)
            except:
                pass  # Windows doesn't support chmod
        except Exception as e:
            logger.error(f"Error saving encryption key: {e}")
        return key
    
    def store_api_key(self, service: str, api_key: str):
        """Store API key encrypted"""
        try:
            # Load existing keys
            keys = self.load_all_keys()
            keys[service] = api_key
            
            # Encrypt and save
            encrypted_data = {}
            for svc, key in keys.items():
                encrypted_data[svc] = self._cipher.encrypt(key.encode()).decode()
            
            with open(self.keys_file, 'w') as f:
                json.dump(encrypted_data, f)
            
            # Set restrictive permissions
            try:
                os.chmod(self.keys_file, API_KEYS_FILE_PERMISSIONS)
            except:
                pass
            
            SecureLogger.info(f"API key stored for {service}")
        except Exception as e:
            logger.error(f"Error storing API key: {e}")
            raise SecurityError(f"Failed to store API key securely: {e}")
    
    def get_api_key(self, service: str) -> Optional[str]:
        """Retrieve API key (decrypted)"""
        # First check environment variables (most secure)
        env_key = os.getenv(f"{service.upper()}_API_KEY")
        if env_key:
            return env_key
        
        # Then check encrypted file
        try:
            if not self.keys_file.exists():
                return None
            
            with open(self.keys_file, 'r') as f:
                encrypted_data = json.load(f)
            
            if service in encrypted_data:
                encrypted_key = encrypted_data[service].encode()
                decrypted = self._cipher.decrypt(encrypted_key)
                return decrypted.decode()
        except Exception as e:
            SecureLogger.error(f"Error retrieving API key for {service}: {e}")
        
        return None
    
    def load_all_keys(self) -> Dict[str, str]:
        """Load all stored keys (for migration/backup)"""
        keys = {}
        try:
            if self.keys_file.exists():
                with open(self.keys_file, 'r') as f:
                    encrypted_data = json.load(f)
                
                for service, encrypted_key in encrypted_data.items():
                    try:
                        decrypted = self._cipher.decrypt(encrypted_key.encode())
                        keys[service] = decrypted.decode()
                    except:
                        pass
        except Exception as e:
            SecureLogger.error(f"Error loading keys: {e}")
        
        return keys
    
    def delete_api_key(self, service: str):
        """Delete stored API key"""
        try:
            keys = self.load_all_keys()
            if service in keys:
                del keys[service]
                
                encrypted_data = {}
                for svc, key in keys.items():
                    encrypted_data[svc] = self._cipher.encrypt(key.encode()).decode()
                
                with open(self.keys_file, 'w') as f:
                    json.dump(encrypted_data, f)
                
                SecureLogger.info(f"API key deleted for {service}")
        except Exception as e:
            SecureLogger.error(f"Error deleting API key: {e}")


class RateLimiter:
    """Rate limiting for API calls"""
    
    def __init__(self, max_calls: int, time_window: int):
        self.max_calls = max_calls
        self.time_window = time_window  # seconds
        self.calls = {}  # {service: [timestamps]}
    
    def can_make_request(self, service: str) -> bool:
        """Check if request can be made"""
        from time import time
        now = time()
        
        if service not in self.calls:
            self.calls[service] = []
        
        # Remove old calls outside time window
        self.calls[service] = [
            ts for ts in self.calls[service]
            if now - ts < self.time_window
        ]
        
        return len(self.calls[service]) < self.max_calls
    
    def record_request(self, service: str):
        """Record API request"""
        from time import time
        if service not in self.calls:
            self.calls[service] = []
        self.calls[service].append(time())


class SeekerAI:
    """Seeker AI - AI-powered stock research system with comprehensive security"""
    
    def __init__(self, data_dir: Path):
        if not SEEKER_AI_ENABLED:
            logger.warning("Seeker AI is disabled in configuration")
        
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = data_dir / "research_cache"
        self.cache_dir.mkdir(exist_ok=True)
        
        # Security components
        self.credential_manager = SecureCredentialManager(data_dir)
        self.session = get_secure_session()  # Use secure session from security.py
        self.api_client = SecureAPIClient(self.session)
        self.rate_limiter = RateLimiter(max_calls=SEEKER_AI_RATE_LIMIT, time_window=60)
        self.anomaly_detector = AnomalyDetector()
        self.retention_manager = DataRetentionManager(data_dir)
        
        # API endpoints (use HTTPS only)
        self.news_api_url = "https://newsapi.org/v2/everything"
        self.openai_api_url = "https://api.openai.com/v1/chat/completions"
        
        # Audit log
        self.audit_log_file = data_dir / "api_audit.log"
    
    def _log_api_call(self, service: str, symbol: str, success: bool, error: str = None):
        """Log API calls for security auditing"""
        if not SEEKER_AI_AUDIT_LOGGING:
            return
        
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                'timestamp': timestamp,
                'service': service,
                'symbol': symbol,
                'success': success,
                'error': SecureLogger.redact_sensitive(error) if error else None,
                'ip': 'localhost'  # Desktop app
            }
            
            with open(self.audit_log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            SecureLogger.error(f"Error logging API call: {e}")
    
    def research_stock(self, symbol: str, stock_data: dict, 
                      history_data: dict) -> Dict:
        """
        Comprehensive stock research using AI
        WITH SECURITY VALIDATION
        """
        try:
            # SECURITY: Validate input
            if not InputValidator.validate_symbol(symbol):
                SecureLogger.warning(f"Invalid symbol format: {symbol}")
                return {'error': 'Invalid symbol format'}
            
            symbol = symbol.upper().strip()
            
            # SECURITY: Validate stock data
            if not DataValidator.validate_stock_data(stock_data):
                SecureLogger.warning(f"Invalid stock data for {symbol}")
                return {'error': 'Invalid stock data'}
            
            # Check cache first (cache for configured hours)
            cached = self._load_from_cache(symbol)
            if cached and self._is_cache_valid(cached):
                logger.debug(f"Using cached research for {symbol}")
                return cached
            
            research = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'news_articles': [],
                'sentiment_score': 0.0,
                'reputation_score': 0.0,
                'price_movement_explanation': '',
                'key_insights': [],
                'risk_factors': [],
                'opportunities': [],
                'summary': ''
            }
            
            # 1. Fetch news articles (with rate limiting)
            if self.rate_limiter.can_make_request('news'):
                news_articles = self._fetch_news(symbol)
                research['news_articles'] = news_articles
                self.rate_limiter.record_request('news')
                self._log_api_call('news', symbol, len(news_articles) > 0)
                
                # Check for anomalies
                self.anomaly_detector.check_anomaly('news', symbol, len(news_articles) > 0)
            else:
                SecureLogger.warning(f"Rate limit exceeded for news API")
                research['news_articles'] = []
            
            # 2. Analyze sentiment (local processing, no API)
            sentiment = self._analyze_sentiment(research['news_articles'], symbol)
            research['sentiment_score'] = sentiment['score']
            research['sentiment_label'] = sentiment['label']
            
            # 3. Analyze reputation (local processing)
            reputation = self._analyze_reputation(research['news_articles'], symbol)
            research['reputation_score'] = reputation['score']
            research['reputation_factors'] = reputation['factors']
            
            # 4. Explain price movements
            price_explanation = self._explain_price_movement(
                symbol, stock_data, history_data, research['news_articles']
            )
            research['price_movement_explanation'] = price_explanation
            
            # 5. Generate AI summary using LLM (with rate limiting)
            if self.rate_limiter.can_make_request('llm'):
                ai_summary = self._generate_ai_summary(
                    symbol, stock_data, research['news_articles'], sentiment, reputation
                )
                research['summary'] = ai_summary['summary']
                research['key_insights'] = ai_summary['insights']
                research['risk_factors'] = ai_summary['risks']
                research['opportunities'] = ai_summary['opportunities']
                self.rate_limiter.record_request('llm')
                self._log_api_call('llm', symbol, 'summary' in ai_summary)
            else:
                SecureLogger.warning(f"Rate limit exceeded for LLM API")
                # Use fallback summary
                research['summary'] = self._generate_fallback_summary(symbol, sentiment, reputation)
            
            # Cache the research
            self._save_to_cache(symbol, research)
            
            return research
            
        except SecurityError as e:
            SecureLogger.error(f"Security error researching {symbol}: {e}")
            self._log_api_call('research', symbol, False, str(e))
            return {'error': 'Security validation failed'}
        except Exception as e:
            SecureLogger.error(f"Error researching stock {symbol}: {e}")
            self._log_api_call('research', symbol, False, str(e))
            return {'error': 'Research failed'}
    
    def _fetch_news(self, symbol: str, days: int = 7) -> List[Dict]:
        """Fetch recent news articles - SECURE VERSION"""
        try:
            # SECURITY: Validate symbol again
            if not InputValidator.validate_symbol(symbol):
                return []
            
            # Get API key securely
            news_api_key = self.credential_manager.get_api_key('newsapi')
            
            # Option 1: Use NewsAPI (if key available)
            if news_api_key:
                try:
                    params = {
                        'q': symbol,  # Sanitized by validate_symbol
                        'apiKey': news_api_key,
                        'sortBy': 'publishedAt',
                        'language': 'en',
                        'pageSize': 20
                    }
                    
                    # SECURITY: Use secure API client (HTTPS enforced)
                    response = self.api_client.get(
                        self.news_api_url, 
                        params=params
                    )
                    
                    # Validate response
                    if not ResponseValidator.validate_response(response):
                        SecureLogger.warning("Invalid response from NewsAPI")
                        self._log_api_call('newsapi', symbol, False, "Invalid response")
                        return []
                    
                    if response.status_code == 200:
                        data = response.json()
                        articles = []
                        for article in data.get('articles', [])[:20]:  # Limit to 20
                            # SECURITY: Sanitize article data
                            articles.append({
                                'title': InputValidator.sanitize_string(article.get('title', ''), 200),
                                'description': InputValidator.sanitize_string(article.get('description', ''), 500),
                                'url': article.get('url', '') if InputValidator.validate_url(article.get('url', '')) else '',
                                'published_at': article.get('publishedAt', ''),
                                'source': InputValidator.sanitize_string(
                                    article.get('source', {}).get('name', ''), 50
                                )
                            })
                        return articles
                    else:
                        SecureLogger.warning(f"NewsAPI returned status {response.status_code}")
                        self._log_api_call('newsapi', symbol, False, f"Status {response.status_code}")
                        
                except requests.exceptions.SSLError as e:
                    SecureLogger.error(f"SSL error fetching news: {e}")
                    self._log_api_call('newsapi', symbol, False, "SSL error")
                except Exception as e:
                    SecureLogger.error(f"Error fetching news from NewsAPI: {e}")
                    self._log_api_call('newsapi', symbol, False, "Request failed")
            
            # Option 2: Use yfinance news (fallback, no API key needed)
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                news = ticker.news
                articles = []
                for item in news[:20]:
                    articles.append({
                        'title': InputValidator.sanitize_string(item.get('title', ''), 200),
                        'description': InputValidator.sanitize_string(item.get('summary', ''), 500),
                        'url': item.get('link', ''),
                        'published_at': datetime.fromtimestamp(
                            item.get('providerPublishTime', 0)
                        ).isoformat(),
                        'source': 'Yahoo Finance'
                    })
                return articles
            except Exception as e:
                logger.debug(f"Could not fetch news from yfinance: {e}")
            
            return []
            
        except Exception as e:
            SecureLogger.error(f"Error fetching news: {e}")
            return []
    
    def _analyze_sentiment(self, articles: List[Dict], symbol: str) -> Dict:
        """Analyze sentiment from news articles"""
        if not articles:
            return {'score': 0.0, 'label': 'neutral'}
        
        # Simple sentiment analysis (can be enhanced with NLP library)
        positive_keywords = ['growth', 'profit', 'gain', 'surge', 'rally', 
                           'beat', 'exceed', 'upgrade', 'bullish', 'strong']
        negative_keywords = ['decline', 'loss', 'fall', 'drop', 'crash',
                           'miss', 'warning', 'bearish', 'weak', 'concern']
        
        positive_count = 0
        negative_count = 0
        total_words = 0
        
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}".lower()
            words = text.split()
            total_words += len(words)
            
            for word in words:
                if any(kw in word for kw in positive_keywords):
                    positive_count += 1
                if any(kw in word for kw in negative_keywords):
                    negative_count += 1
        
        # Calculate sentiment score (-1 to 1, normalized to 0-1)
        if total_words > 0:
            sentiment = (positive_count - negative_count) / total_words * 10
            sentiment = max(-1, min(1, sentiment))  # Clamp to -1 to 1
            sentiment_normalized = (sentiment + 1) / 2  # Normalize to 0-1
        else:
            sentiment_normalized = 0.5
        
        label = 'positive' if sentiment_normalized > 0.6 else 'negative' if sentiment_normalized < 0.4 else 'neutral'
        
        return {
            'score': sentiment_normalized,
            'label': label,
            'positive_count': positive_count,
            'negative_count': negative_count
        }
    
    def _analyze_reputation(self, articles: List[Dict], symbol: str) -> Dict:
        """Analyze company reputation from news"""
        if not articles:
            return {'score': 0.5, 'factors': []}
        
        reputation_keywords = {
            'positive': ['innovative', 'leader', 'strong', 'reliable', 'trusted',
                        'excellent', 'outstanding', 'award', 'recognition'],
            'negative': ['scandal', 'lawsuit', 'fraud', 'controversy', 'criticism',
                        'violation', 'fine', 'investigation', 'concern']
        }
        
        positive_score = 0
        negative_score = 0
        factors = []
        
        for article in articles:
            text = f"{article.get('title', '')} {article.get('description', '')}".lower()
            
            for keyword in reputation_keywords['positive']:
                if keyword in text:
                    positive_score += 1
                    factors.append(f"Positive: {keyword} mentioned")
            
            for keyword in reputation_keywords['negative']:
                if keyword in text:
                    negative_score += 1
                    factors.append(f"Negative: {keyword} mentioned")
        
        # Calculate reputation score (0-1)
        total_signals = positive_score + negative_score
        if total_signals > 0:
            reputation = positive_score / total_signals
        else:
            reputation = 0.5  # Neutral if no signals
        
        return {
            'score': reputation,
            'factors': factors[:10]  # Top 10 factors
        }
    
    def _explain_price_movement(self, symbol: str, stock_data: dict,
                               history_data: dict, articles: List[Dict]) -> str:
        """Explain why price has surged or fallen"""
        try:
            current_price = stock_data.get('price', 0)
            change_percent = stock_data.get('change_percent', 0)
            
            if abs(change_percent) < 2:
                return "Price movement is within normal range (<2%)."
            
            # Check recent news for price movement explanations
            significant_movements = []
            for article in articles[:5]:  # Check top 5 recent articles
                title = article.get('title', '').lower()
                description = article.get('description', '').lower()
                
                # Look for price-related keywords
                if any(kw in title or kw in description for kw in 
                      ['earnings', 'revenue', 'profit', 'loss', 'guidance', 
                       'upgrade', 'downgrade', 'merger', 'acquisition', 
                       'fda', 'approval', 'lawsuit', 'settlement']):
                    significant_movements.append({
                        'title': InputValidator.sanitize_string(article.get('title', ''), 200),
                        'description': InputValidator.sanitize_string(article.get('description', ''), 200)
                    })
            
            if significant_movements:
                explanation = f"Price {'surged' if change_percent > 0 else 'fell'} {abs(change_percent):.2f}%. "
                explanation += f"Recent news suggests: {significant_movements[0]['title']}"
                return explanation
            
            # Fallback explanation
            if change_percent > 5:
                return f"Significant price surge ({change_percent:.2f}%) detected. Review recent news and technical indicators."
            elif change_percent < -5:
                return f"Significant price decline ({abs(change_percent):.2f}%) detected. Review recent news and technical indicators."
            else:
                return f"Price movement of {change_percent:.2f}% may be due to market conditions or recent developments."
                
        except Exception as e:
            SecureLogger.error(f"Error explaining price movement: {e}")
            return "Unable to determine price movement explanation."
    
    def _generate_fallback_summary(self, symbol: str, sentiment: Dict, reputation: Dict) -> str:
        """Generate basic summary without LLM"""
        summary_parts = [
            f"{symbol} Analysis:",
            f"Sentiment: {sentiment['label']} ({sentiment['score']:.2f})",
            f"Reputation: {reputation['score']:.2f}",
        ]
        return '\n'.join(summary_parts)
    
    def _generate_ai_summary(self, symbol: str, stock_data: dict,
                            articles: List[Dict], sentiment: Dict, 
                            reputation: Dict) -> Dict:
        """Generate AI summary using LLM (OpenAI/Anthropic)"""
        try:
            # Get API key securely
            openai_key = self.credential_manager.get_api_key('openai')
            
            if not openai_key:
                return {
                    'summary': self._generate_fallback_summary(symbol, sentiment, reputation),
                    'insights': [],
                    'risks': [],
                    'opportunities': []
                }
            
            # Prepare context
            news_summary = "\n".join([f"- {InputValidator.sanitize_string(a['title'], 100)}" for a in articles[:10]])
            
            prompt = f"""Analyze the stock {symbol} based on the following information:

Stock Data:
- Current Price: ${stock_data.get('price', 0):.2f}
- Change: {stock_data.get('change_percent', 0):.2f}%

Recent News Headlines:
{news_summary}

Sentiment Analysis:
- Score: {sentiment['score']:.2f}
- Label: {sentiment['label']}

Reputation Analysis:
- Score: {reputation['score']:.2f}

Please provide:
1. A concise summary (2-3 sentences)
2. Key insights (3-5 bullet points)
3. Risk factors (2-3 items)
4. Opportunities (2-3 items)

Format as JSON with keys: summary, insights (array), risks (array), opportunities (array)
"""
            
            # SECURITY: Sanitize prompt
            prompt = InputValidator.sanitize_string(prompt, 2000)
            
            headers = {
                'Authorization': f'Bearer {openai_key}',
                'Content-Type': 'application/json'
            }
            
            payload = {
                'model': 'gpt-4',
                'messages': [
                    {'role': 'system', 'content': 'You are a financial analyst providing stock research.'},
                    {'role': 'user', 'content': prompt}
                ],
                'temperature': 0.7,
                'max_tokens': 500
            }
            
            # SECURITY: Use secure API client
            response = self.api_client.post(
                self.openai_api_url,
                headers=headers,
                json=payload
            )
            
            # Validate response
            if not ResponseValidator.validate_response(response):
                SecureLogger.warning("Invalid response from OpenAI API")
                return {
                    'summary': self._generate_fallback_summary(symbol, sentiment, reputation),
                    'insights': [],
                    'risks': [],
                    'opportunities': []
                }
            
            if response.status_code == 200:
                data = response.json()
                result_text = data['choices'][0]['message']['content']
                
                # Try to parse as JSON
                try:
                    result = json.loads(result_text)
                    # Validate JSON structure
                    if ResponseValidator.validate_json_structure(result):
                        return result
                    else:
                        SecureLogger.warning("Invalid JSON structure from LLM")
                except json.JSONDecodeError:
                    # Not JSON, return as summary
                    return {
                        'summary': InputValidator.sanitize_string(result_text, 1000),
                        'insights': [],
                        'risks': [],
                        'opportunities': []
                    }
            else:
                error_msg = f"API returned status {response.status_code}"
                SecureLogger.warning(error_msg)
                self._log_api_call('openai', symbol, False, error_msg)
                return {
                    'summary': self._generate_fallback_summary(symbol, sentiment, reputation),
                    'insights': [],
                    'risks': [],
                    'opportunities': []
                }
                
        except requests.exceptions.SSLError as e:
            SecureLogger.error(f"SSL error calling LLM API: {e}")
            self._log_api_call('openai', symbol, False, "SSL error")
            return {
                'summary': self._generate_fallback_summary(symbol, sentiment, reputation),
                'insights': [],
                'risks': [],
                'opportunities': []
            }
        except Exception as e:
            SecureLogger.error(f"Error calling LLM API: {e}")
            self._log_api_call('openai', symbol, False, "Request failed")
            return {
                'summary': self._generate_fallback_summary(symbol, sentiment, reputation),
                'insights': [],
                'risks': [],
                'opportunities': []
            }
    
    def _load_from_cache(self, symbol: str) -> Optional[Dict]:
        """Load research from cache"""
        cache_file = self.cache_dir / f"{symbol}_research.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None
    
    def _save_to_cache(self, symbol: str, research: Dict):
        """Save research to cache"""
        cache_file = self.cache_dir / f"{symbol}_research.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(research, f, indent=2)
        except Exception as e:
            SecureLogger.error(f"Error saving cache: {e}")
    
    def _is_cache_valid(self, cached: Dict, max_age_hours: int = None) -> bool:
        """Check if cache is still valid"""
        if max_age_hours is None:
            max_age_hours = SEEKER_AI_CACHE_HOURS
        
        try:
            timestamp = datetime.fromisoformat(cached.get('timestamp', ''))
            age = datetime.now() - timestamp
            return age < timedelta(hours=max_age_hours)
        except:
            return False
    
    def cleanup_old_data(self):
        """Cleanup old research data"""
        return self.retention_manager.cleanup_old_data()
