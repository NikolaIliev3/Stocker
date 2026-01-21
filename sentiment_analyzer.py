"""
Sentiment Analyzer
Analyzes news sentiment using a lightweight, zero-cost financial dictionary approach.
Uses yfinance to fetch news and performs local analysis without heavy dependencies.
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """
    Analyzes stock sentiment from news headlines using a 
    specialized financial lexicon (simplified Loughran-McDonald).
    """
    
    # Simplified Financial Sentiment Dictionary
    # Focused on high-impact financial keywords
    POSITIVE_WORDS = {
        'surge', 'jump', 'gain', 'rally', 'climb', 'soar', 'skyrocket', 'bull', 'bullish',
        'profit', 'beat', 'exceed', 'outperform', 'growth', 'expand', 'success',
        'collaborate', 'partner', 'merge', 'acquisition', 'dividend', 'buyback',
        'upgrade', 'higher', 'record', 'strong', 'positive', 'optimistic', 'lead',
        'breakthrough', 'innovate', 'approve', 'clear', 'win', 'award', 'raise'
    }
    
    NEGATIVE_WORDS = {
        'plunge', 'drop', 'fall', 'slide', 'tumble', 'crash', 'bear', 'bearish',
        'loss', 'miss', 'lag', 'underperform', 'decline', 'shrink', 'fail',
        'sue', 'lawsuit', 'investigation', 'fraud', 'scandal', 'risk', 'warning',
        'downgrade', 'lower', 'cut', 'weak', 'negative', 'pessimistic', 'delay',
        'reject', 'denial', 'ban', 'fine', 'penalty', 'debt', 'bankrupt'
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
        
    def analyze_sentiment(self, symbol: str) -> Dict:
        """
        Fetch news and analyze sentiment for a stock
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dict with sentiment metrics, including score (-100 to 100)
        """
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
                title = item.get('title', '')
                if not title:
                    continue
                    
                # Publish time check (only analyze recent news - last 7 days)
                # yfinance returns timestamp
                pub_time = item.get('providerPublishTime')
                if pub_time:
                    try:
                        pub_date = datetime.fromtimestamp(pub_time)
                        if (datetime.now() - pub_date) > timedelta(days=7):
                            continue
                    except:
                        pass
                
                score = self._analyze_text(title)
                
                # Weight newer news heavier? 
                # For now simple average
                total_score += score
                count += 1
                
                sentiment_label = "neutral"
                if score > 0.2: sentiment_label = "positive"
                elif score < -0.2: sentiment_label = "negative"
                
                recent_headlines.append({
                    'title': title,
                    'score': score,
                    'label': sentiment_label,
                    'link': item.get('link', '')
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
        Get confidence adjustment based on sentiment
        """
        if not sentiment_result.get('available', False):
            return {'confidence_adjustment': 0, 'reasoning': []}
            
        score = sentiment_result['sentiment_score']
        adj = 0
        reasoning = []
        
        # Strong sentiment affects confidence
        if score > 30:
            adj = 5
            reasoning.append(f"📰 Strong positive news sentiment ({score:.1f})")
        elif score < -30:
            adj = -10  # Negative news is more dangerous
            reasoning.append(f"⚠️ Strong negative news sentiment ({score:.1f})")
            
        return {
            'confidence_adjustment': adj,
            'reasoning': reasoning
        }
