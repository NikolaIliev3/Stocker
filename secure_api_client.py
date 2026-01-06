"""
Secure API Client Module
API client with security timeouts and connection limits
"""
import logging
import time
from typing import Optional
from security import SecureSession, SecurityError
from config import API_REQUEST_TIMEOUT, API_MAX_RETRIES, API_MAX_CONNECTIONS

logger = logging.getLogger(__name__)


class SecureAPIClient:
    """API client with security timeouts and limits"""
    
    CONNECT_TIMEOUT = 5  # seconds
    READ_TIMEOUT = API_REQUEST_TIMEOUT
    MAX_RETRIES = API_MAX_RETRIES
    MAX_CONNECTIONS = API_MAX_CONNECTIONS
    MAX_REDIRECTS = 5
    
    def __init__(self, session: Optional[SecureSession] = None):
        self.session = session or SecureSession()
        self.active_connections = 0
        self._connection_lock = False  # Simple lock mechanism
    
    def _acquire_connection(self) -> bool:
        """Acquire a connection slot"""
        if self.active_connections >= self.MAX_CONNECTIONS:
            logger.warning("Maximum connections exceeded")
            return False
        self.active_connections += 1
        return True
    
    def _release_connection(self):
        """Release a connection slot"""
        if self.active_connections > 0:
            self.active_connections -= 1
    
    def make_request(self, method: str, url: str, **kwargs):
        """
        Make request with security limits
        
        Args:
            method: HTTP method (GET, POST, etc.)
            url: Request URL
            **kwargs: Additional request parameters
        
        Returns:
            Response object
        
        Raises:
            SecurityError: If security limits are exceeded
        """
        # Check connection limit
        if not self._acquire_connection():
            raise SecurityError("Maximum connections exceeded")
        
        try:
            # Set timeouts
            kwargs['timeout'] = (
                self.CONNECT_TIMEOUT,
                self.READ_TIMEOUT
            )
            
            # Limit redirects
            kwargs['allow_redirects'] = True
            if 'max_redirects' not in kwargs:
                kwargs['max_redirects'] = self.MAX_REDIRECTS
            
            # Retry logic
            last_exception = None
            for attempt in range(self.MAX_RETRIES):
                try:
                    response = self.session.request(method, url, **kwargs)
                    return response
                except Exception as e:
                    last_exception = e
                    if attempt < self.MAX_RETRIES - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        logger.debug(f"Request failed, retrying in {wait_time}s (attempt {attempt + 1}/{self.MAX_RETRIES})")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Request failed after {self.MAX_RETRIES} attempts: {e}")
            
            raise last_exception
            
        finally:
            self._release_connection()
    
    def get(self, url: str, **kwargs):
        """Make GET request"""
        return self.make_request('GET', url, **kwargs)
    
    def post(self, url: str, **kwargs):
        """Make POST request"""
        return self.make_request('POST', url, **kwargs)
    
    def put(self, url: str, **kwargs):
        """Make PUT request"""
        return self.make_request('PUT', url, **kwargs)
    
    def delete(self, url: str, **kwargs):
        """Make DELETE request"""
        return self.make_request('DELETE', url, **kwargs)
