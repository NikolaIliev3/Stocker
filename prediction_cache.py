"""
Prediction Cache - LRU cache for stock predictions with TTL
"""
import time
import threading
from typing import Dict, Optional, Any
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class PredictionCache:
    """Thread-safe LRU cache for predictions with time-to-live"""
    
    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        """Initialize prediction cache
        
        Args:
            max_size: Maximum number of cached predictions
            ttl_seconds: Time-to-live in seconds (default: 5 minutes)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        
        logger.info(f"Prediction cache initialized (max_size={max_size}, ttl={ttl_seconds}s)")
    
    def _make_key(self, symbol: str, strategy: str) -> str:
        """Create cache key from symbol and strategy
        
        Args:
            symbol: Stock symbol
            strategy: Strategy name (trading, mixed, investing)
            
        Returns:
            Cache key string
        """
        return f"{symbol.upper()}:{strategy}"
    
    def get(self, symbol: str, strategy: str) -> Optional[Dict[str, Any]]:
        """Get cached prediction if available and not expired
        
        Args:
            symbol: Stock symbol
            strategy: Strategy name
            
        Returns:
            Cached prediction dict or None if not found/expired
        """
        key = self._make_key(symbol, strategy)
        
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # Check if expired
            age = time.time() - entry['timestamp']
            if age > self.ttl_seconds:
                # Expired, remove it
                del self._cache[key]
                logger.debug(f"Cache expired for {key} (age: {age:.1f}s)")
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(key)
            
            logger.debug(f"Cache hit for {key} (age: {age:.1f}s)")
            return entry['prediction']
    
    def set(self, symbol: str, strategy: str, prediction: Dict[str, Any]):
        """Store prediction in cache
        
        Args:
            symbol: Stock symbol
            strategy: Strategy name
            prediction: Prediction dictionary to cache
        """
        key = self._make_key(symbol, strategy)
        
        with self._lock:
            # Create cache entry
            entry = {
                'timestamp': time.time(),
                'prediction': prediction
            }
            
            # Add to cache
            self._cache[key] = entry
            self._cache.move_to_end(key)
            
            # Evict oldest if over max size
            if len(self._cache) > self.max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                logger.debug(f"Cache evicted oldest entry: {oldest_key}")
            
            logger.debug(f"Cache stored {key} (cache size: {len(self._cache)})")
    
    def invalidate(self, symbol: str, strategy: Optional[str] = None):
        """Invalidate cached prediction(s) for a symbol
        
        Args:
            symbol: Stock symbol
            strategy: Optional strategy name. If None, invalidates all strategies for symbol
        """
        with self._lock:
            if strategy:
                # Invalidate specific strategy
                key = self._make_key(symbol, strategy)
                if key in self._cache:
                    del self._cache[key]
                    logger.debug(f"Cache invalidated: {key}")
            else:
                # Invalidate all strategies for this symbol
                keys_to_remove = [k for k in self._cache.keys() 
                                 if k.startswith(f"{symbol.upper()}:")]
                for key in keys_to_remove:
                    del self._cache[key]
                    logger.debug(f"Cache invalidated: {key}")
    
    def clear(self):
        """Clear all cached predictions"""
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.info(f"Cache cleared ({count} entries removed)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'ttl_seconds': self.ttl_seconds,
                'entries': list(self._cache.keys())
            }


# Global cache instance
_global_cache: Optional[PredictionCache] = None


def get_cache() -> PredictionCache:
    """Get global prediction cache instance
    
    Returns:
        Global PredictionCache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = PredictionCache()
    return _global_cache


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    cache = PredictionCache(max_size=3, ttl_seconds=2)
    
    # Store predictions
    cache.set('AAPL', 'trading', {'action': 'BUY', 'confidence': 75})
    cache.set('MSFT', 'trading', {'action': 'SELL', 'confidence': 65})
    cache.set('GOOGL', 'trading', {'action': 'HOLD', 'confidence': 50})
    
    # Retrieve
    print("Get AAPL:", cache.get('AAPL', 'trading'))
    print("Get MSFT:", cache.get('MSFT', 'trading'))
    
    # Add one more (should evict oldest)
    cache.set('TSLA', 'trading', {'action': 'BUY', 'confidence': 80})
    print("Stats:", cache.get_stats())
    
    # Wait for expiration
    print("\nWaiting 3 seconds for TTL expiration...")
    time.sleep(3)
    
    print("Get MSFT (should be expired):", cache.get('MSFT', 'trading'))
    print("Stats:", cache.get_stats())
