"""
Request Signing Module
Signs requests to prevent tampering
"""
import hmac
import hashlib
import json
import time
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class RequestSigner:
    """Sign requests to prevent tampering"""
    
    def __init__(self, secret_key: str):
        if not secret_key:
            raise ValueError("Secret key cannot be empty")
        self.secret_key = secret_key.encode()
    
    def sign_request(self, data: dict, timestamp: Optional[float] = None) -> str:
        """Create HMAC signature for request"""
        if timestamp is None:
            timestamp = time.time()
        
        # Sort keys for consistent hashing
        payload = json.dumps(data, sort_keys=True) + str(timestamp)
        signature = hmac.new(
            self.secret_key,
            payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def verify_request(self, data: dict, timestamp: float, signature: str, 
                      max_age: int = 300) -> bool:
        """Verify request signature with timestamp validation"""
        # Check timestamp freshness (prevent replay attacks)
        current_time = time.time()
        if abs(current_time - timestamp) > max_age:
            logger.warning(f"Request timestamp too old: {abs(current_time - timestamp)}s")
            return False
        
        # Verify signature
        expected = self.sign_request(data, timestamp)
        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(expected, signature)
    
    def create_signed_request(self, data: dict) -> Dict:
        """Create a signed request payload"""
        timestamp = time.time()
        signature = self.sign_request(data, timestamp)
        
        return {
            'data': data,
            'timestamp': timestamp,
            'signature': signature
        }
