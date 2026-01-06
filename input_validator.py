"""
Input Validation Module
Comprehensive input validation and sanitization
"""
import re
import logging
from typing import Optional
from config import MAX_SYMBOL_LENGTH, MAX_REQUEST_SIZE, MAX_STRING_LENGTH

logger = logging.getLogger(__name__)


class InputValidator:
    """Comprehensive input validation"""
    
    MAX_SYMBOL_LENGTH = MAX_SYMBOL_LENGTH
    MAX_REQUEST_SIZE = MAX_REQUEST_SIZE
    MAX_STRING_LENGTH = MAX_STRING_LENGTH
    ALLOWED_SYMBOL_CHARS = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_')
    
    # Patterns to detect potential injection attacks
    DANGEROUS_PATTERNS = [
        r'<script',
        r'javascript:',
        r'onerror=',
        r'onload=',
        r'eval\(',
        r'exec\(',
        r'__import__',
        r'subprocess',
    ]
    
    @staticmethod
    def validate_symbol(symbol: str) -> bool:
        """Validate stock symbol with strict rules"""
        if not symbol or not isinstance(symbol, str):
            return False
        
        symbol = symbol.strip()
        
        # Length check
        if len(symbol) < 1 or len(symbol) > InputValidator.MAX_SYMBOL_LENGTH:
            return False
        
        # Character check - only allow alphanumeric, dots, and hyphens
        if not all(c in InputValidator.ALLOWED_SYMBOL_CHARS for c in symbol):
            return False
        
        # Must start with letter
        if not symbol[0].isalpha():
            return False
        
        # Check for dangerous patterns
        symbol_lower = symbol.lower()
        for pattern in InputValidator.DANGEROUS_PATTERNS:
            if re.search(pattern, symbol_lower):
                logger.warning(f"Dangerous pattern detected in symbol: {symbol}")
                return False
        
        return True
    
    @staticmethod
    def validate_json_size(data: dict) -> bool:
        """Check if JSON data is within size limits"""
        import json
        try:
            size = len(json.dumps(data))
            if size > InputValidator.MAX_REQUEST_SIZE:
                logger.warning(f"JSON data too large: {size} bytes")
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating JSON size: {e}")
            return False
    
    @staticmethod
    def sanitize_string(text: str, max_length: Optional[int] = None) -> str:
        """Sanitize string input"""
        if not isinstance(text, str):
            text = str(text)
        
        if max_length is None:
            max_length = InputValidator.MAX_STRING_LENGTH
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Truncate to max length
        text = text[:max_length]
        
        # Remove control characters except newline, tab, carriage return
        text = ''.join(c for c in text if ord(c) >= 32 or c in '\n\r\t')
        
        return text
    
    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate URL format and security"""
        if not url or not isinstance(url, str):
            return False
        
        # Must start with https://
        if not url.startswith('https://'):
            logger.warning(f"Non-HTTPS URL rejected: {url[:50]}")
            return False
        
        # Check for dangerous patterns
        url_lower = url.lower()
        for pattern in InputValidator.DANGEROUS_PATTERNS:
            if pattern in url_lower:
                logger.warning(f"Dangerous pattern in URL: {url[:50]}")
                return False
        
        return True
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email format"""
        if not email or not isinstance(email, str):
            return False
        
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_numeric(value: any, min_val: Optional[float] = None, 
                        max_val: Optional[float] = None) -> bool:
        """Validate numeric value within range"""
        try:
            num = float(value)
            
            if min_val is not None and num < min_val:
                return False
            if max_val is not None and num > max_val:
                return False
            
            return True
        except (ValueError, TypeError):
            return False
