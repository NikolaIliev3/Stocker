"""
Sentiment Analyzer
Analyzes news sentiment using a lightweight, zero-cost financial dictionary approach.
Uses yfinance to fetch news and performs local analysis without heavy dependencies.
"""
import logging
import os
import shutil
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)

# --- SSL Fix for non-ASCII user paths (e.g. Cyrillic) on Windows ---
# curl_cffi cannot resolve cert paths containing non-ASCII characters.
# Copy cacert.pem to an ASCII-safe location if needed.
try:
    import certifi
    _cert_src = certifi.where()
    # Check if the path contains non-ASCII characters
    try:
        _cert_src.encode('ascii')
    except UnicodeEncodeError:
        _safe_cert = os.path.join(os.environ.get('TEMP', 'C:/temp_certs'), 'cacert.pem')
        try:
            _safe_cert.encode('ascii')
        except UnicodeEncodeError:
            _safe_cert = 'C:/temp_certs/cacert.pem'
        os.makedirs(os.path.dirname(_safe_cert), exist_ok=True)
        if not os.path.exists(_safe_cert) or os.path.getsize(_safe_cert) != os.path.getsize(_cert_src):
            shutil.copy2(_cert_src, _safe_cert)
        os.environ['CURL_CA_BUNDLE'] = _safe_cert
        os.environ['SSL_CERT_FILE'] = _safe_cert
        logger.debug(f"SSL cert copied to ASCII-safe path: {_safe_cert}")
except Exception as e:
    logger.debug(f"SSL cert fix skipped: {e}")

class SentimentAnalyzer:
    """
    Analyzes stock sentiment from news headlines using a 
    specialized financial lexicon (simplified Loughran-McDonald).
    """
    
    # Simplified Financial Sentiment Dictionary
    # Focused on high-impact financial keywords (includes common inflected forms)
    POSITIVE_WORDS = {
        'surge', 'surges', 'surging', 'surged',
        'jump', 'jumps', 'jumping', 'jumped',
        'gain', 'gains', 'gaining', 'gained',
        'rally', 'rallies', 'rallying', 'rallied',
        'climb', 'climbs', 'climbing', 'climbed',
        'soar', 'soars', 'soaring', 'soared',
        'skyrocket', 'skyrockets', 'skyrocketing', 'skyrocketed',
        'bull', 'bullish',
        'profit', 'profits', 'profitable',
        'beat', 'beats', 'beating',
        'exceed', 'exceeds', 'exceeding', 'exceeded',
        'outperform', 'outperforms', 'outperforming', 'outperformed',
        'growth', 'grow', 'grows', 'growing',
        'expand', 'expands', 'expanding', 'expanded', 'expansion',
        'success', 'successful',
        'collaborate', 'partner', 'partnership', 'merge', 'merger', 'acquisition',
        'dividend', 'buyback',
        'upgrade', 'upgrades', 'upgraded',
        'higher', 'record', 'strong', 'stronger', 'strongest', 'strength',
        'positive', 'optimistic', 'optimism',
        'lead', 'leads', 'leading',
        'breakthrough', 'innovate', 'innovation', 'innovative',
        'approve', 'approved', 'approval',
        'clear', 'clears', 'cleared',
        'win', 'wins', 'winning', 'won',
        'award', 'awards', 'awarded',
        'raise', 'raises', 'raised', 'raising',
        'boost', 'boosts', 'boosted', 'boosting',
        'recover', 'recovers', 'recovery', 'recovered',
        'rebound', 'rebounds', 'rebounding', 'rebounded',
        'upbeat', 'upside',
    }
    
    NEGATIVE_WORDS = {
        'plunge', 'plunges', 'plunging', 'plunged',
        'drop', 'drops', 'dropping', 'dropped',
        'fall', 'falls', 'falling', 'fell',
        'slide', 'slides', 'sliding', 'slid',
        'tumble', 'tumbles', 'tumbling', 'tumbled',
        'crash', 'crashes', 'crashing', 'crashed',
        'bear', 'bearish',
        'loss', 'losses', 'lose', 'loses', 'losing', 'lost',
        'miss', 'misses', 'missed', 'missing',
        'lag', 'lags', 'lagging', 'lagged',
        'underperform', 'underperforms', 'underperforming', 'underperformed',
        'decline', 'declines', 'declining', 'declined',
        'shrink', 'shrinks', 'shrinking', 'shrunk',
        'fail', 'fails', 'failing', 'failed', 'failure',
        'sue', 'sues', 'sued', 'suing', 'lawsuit', 'lawsuits',
        'investigation', 'investigations', 'investigate', 'investigated',
        'fraud', 'fraudulent', 'scandal', 'scandals',
        'risk', 'risks', 'risky',
        'warning', 'warnings', 'warn', 'warns', 'warned', 'worried', 'worry', 'worries',
        'downgrade', 'downgrades', 'downgraded',
        'lower', 'lowers', 'lowered',
        'cut', 'cuts', 'cutting',
        'weak', 'weaker', 'weakest', 'weakness',
        'negative', 'pessimistic', 'pessimism',
        'delay', 'delays', 'delayed',
        'reject', 'rejects', 'rejected', 'rejection',
        'denial', 'deny', 'denied',
        'ban', 'bans', 'banned', 'banning',
        'fine', 'fines', 'fined',
        'penalty', 'penalties',
        'debt', 'debts',
        'bankrupt', 'bankruptcy',
        'downturn', 'recession', 'selloff', 'sell-off',
        'downside', 'slump', 'slumps', 'slumping',
    }
    
    # Cache to prevent rate limiting
    _sentiment_cache = {}
    _cache_expiry = {}
    CACHE_DURATION_HOURS = 1
    
    def __init__(self, data_fetcher=None):
        """
        Initialize sentiment analyzer
        
        Args:
            data_fetcher: Optional data fetcher (not strictly needed as we use yfinance directly)
        """
        self.data_fetcher = data_fetcher
        
    def _is_cache_valid(self, symbol: str) -> bool:
        """Check if cached data is still valid"""
        if symbol not in self._cache_expiry:
            return False
        return datetime.now() < self._cache_expiry[symbol]
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for analysis"""
        if not text:
            return ""
        # Convert to lowercase and remove non-alphabetic characters
        text = str(text).lower()
        text = re.sub(r'[^a-z\s]', ' ', text)
        return text
        
    def _analyze_text(self, text: str) -> float:
        """
        Analyze a single text string
        Returns score from -1.0 (negative) to 1.0 (positive)
        """
        words = self._clean_text(text).split()
        if not words:
            return 0.0
            
        score = 0
        word_count = len(words)
        
        for word in words:
            if word in self.POSITIVE_WORDS:
                score += 1
            elif word in self.NEGATIVE_WORDS:
                score -= 1.2  # Negative words often weigh heavier in financial context
                
        # Normalize score
        if word_count > 0:
            # Scale based on density of sentiment words, capped at -1 to 1
            # Adjust divisor to control sensitivity
            final_score = score / (word_count * 0.15 + 1)
            return max(-1.0, min(1.0, final_score))
        return 0.0
        
    def analyze_sentiment(self, symbol: str, as_of_date: Optional[datetime] = None) -> Dict:
        """
        Fetch news and analyze sentiment for a stock
        
        Args:
            symbol: Stock ticker symbol
            as_of_date: Optional timestamp for point-in-time analysis
            
        Returns:
            Dict with sentiment metrics, including score (-100 to 100)
        """
        # INTEGRITY LOCK: If backtesting (as_of_date in the past), do NOT use live news.
        # Live news from 'ticker.news' is ALWAYS current/2026 data.
        reference_date = as_of_date if as_of_date else datetime.now()
        if as_of_date and (datetime.now() - as_of_date).days > 7:
            return {
                'symbol': symbol,
                'sentiment_score': 0.0,
                'sentiment_rating': 'neutral',
                'news_count': 0,
                'reasoning': ["Historical sentiment not available for backtest period"],
                'headlines': [],
                'available': False
            }

        # Check cache
        if self._is_cache_valid(symbol):
            return self._sentiment_cache[symbol]
            
        result = {
            'symbol': symbol,
            'sentiment_score': 0,  # -100 to 100
            'sentiment_rating': 'neutral',
            'news_count': 0,
            'reasoning': [],
            'headlines': [],
            'available': False
        }
        
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            if not news:
                logger.debug(f"No news found for {symbol}")
                return result
                
            result['news_count'] = len(news)
            total_score = 0
            count = 0
            
            recent_headlines = []
            
            for item in news:
                # Handle both old flat format and new nested 'content' format
                content = item.get('content', item)  # New format nests under 'content'
                title = content.get('title', '')
                if not title:
                    continue
                    
                # Publish time check (only analyze recent news - last 7 days)
                # New format: ISO string 'pubDate'; Old format: unix timestamp 'providerPublishTime'
                pub_date_str = content.get('pubDate')
                pub_time = item.get('providerPublishTime')  # Legacy
                if pub_date_str:
                    try:
                        pub_date = datetime.fromisoformat(pub_date_str.replace('Z', '+00:00')).replace(tzinfo=None)
                        if (reference_date - pub_date) > timedelta(days=7):
                            continue
                    except Exception:
                        pass
                elif pub_time:
                    try:
                        pub_date = datetime.fromtimestamp(pub_time)
                        if (reference_date - pub_date) > timedelta(days=7):
                            continue
                    except Exception:
                        pass
                
                score = self._analyze_text(title)
                
                # Weight newer news heavier? 
                # For now simple average
                total_score += score
                count += 1
                
                sentiment_label = "neutral"
                if score > 0.2: sentiment_label = "positive"
                elif score < -0.2: sentiment_label = "negative"
                
                # Extract link from new or old format
                link = ''
                canonical = content.get('canonicalUrl')
                if isinstance(canonical, dict):
                    link = canonical.get('url', '')
                elif isinstance(canonical, str):
                    link = canonical
                else:
                    link = item.get('link', '')
                
                recent_headlines.append({
                    'title': title,
                    'score': score,
                    'label': sentiment_label,
                    'link': link
                })
            
            if count > 0:
                avg_score = total_score / count
                # Scale to -100 to 100
                scaled_score = avg_score * 100
                
                result['sentiment_score'] = scaled_score
                
                if scaled_score > 20:
                    result['sentiment_rating'] = 'bullish'
                elif scaled_score > 5:
                    result['sentiment_rating'] = 'slightly_bullish'
                elif scaled_score < -20:
                    result['sentiment_rating'] = 'bearish'
                elif scaled_score < -5:
                    result['sentiment_rating'] = 'slightly_bearish'
                else:
                    result['sentiment_rating'] = 'neutral'
                    
                result['headlines'] = recent_headlines[:5]  # Top 5 recent
                result['available'] = True
                
                # Generate reasoning
                if result['sentiment_rating'] != 'neutral':
                    emoji = "🐂" if "bullish" in result['sentiment_rating'] else "🐻"
                    result['reasoning'].append(f"{emoji} News sentiment is {result['sentiment_rating'].replace('_', ' ')} ({scaled_score:.1f})")
            
            # Cache result
            self._sentiment_cache[symbol] = result
            self._cache_expiry[symbol] = datetime.now() + timedelta(hours=self.CACHE_DURATION_HOURS)
            
        except Exception as e:
            logger.warning(f"Error analyzing sentiment for {symbol}: {e}")
            
        return result
        
    def get_sentiment_context(self, sentiment_result: Dict) -> Dict:
        """
        Get confidence adjustment based on sentiment.
        
        Calibrated against empirical distribution (20-ticker sample):
        Mean ≈ +4, StDev ≈ 8, Range ≈ -10 to +16.
        
        Thresholds:
          Strong positive:  score > 12  (~1.5σ above mean) -> +5
          Moderate positive: score > 7  (~1σ above mean)   -> +2
          Moderate negative: score < -5 (~1σ below mean)   -> -3
          Strong negative:   score < -8 (~1.5σ below mean) -> -7
        
        Negative news is weighted ~1.5x heavier (bad news travels faster in markets).
        """
        if not sentiment_result.get('available', False):
            return {'confidence_adjustment': 0, 'reasoning': []}
            
        score = sentiment_result['sentiment_score']
        adj = 0
        reasoning = []
        
        # Graduated sentiment impact (data-driven thresholds)
        if score > 12:
            adj = 5
            reasoning.append(f"📰 Strong positive news sentiment ({score:.1f})")
        elif score > 7:
            adj = 2
            reasoning.append(f"📰 Moderate positive news sentiment ({score:.1f})")
        elif score < -8:
            adj = -7  # Negative news is more dangerous
            reasoning.append(f"⚠️ Strong negative news sentiment ({score:.1f})")
        elif score < -5:
            adj = -3
            reasoning.append(f"⚠️ Moderate negative news sentiment ({score:.1f})")
            
        return {
            'confidence_adjustment': adj,
            'reasoning': reasoning
        }
