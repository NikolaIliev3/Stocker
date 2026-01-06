"""
Secure Logging Module
Prevents sensitive data leakage in logs
"""
import re
import logging
from typing import Any

logger = logging.getLogger(__name__)


class SecureLogger:
    """Logger that redacts sensitive information"""
    
    SENSITIVE_PATTERNS = [
        (r'api[_-]?key["\s:=]+([a-zA-Z0-9_-]{20,})', r'api_key=***REDACTED***'),
        (r'bearer\s+([a-zA-Z0-9_-]{20,})', r'bearer ***REDACTED***'),
        (r'password["\s:=]+([^\s"]+)', r'password=***REDACTED***'),
        (r'token["\s:=]+([a-zA-Z0-9_-]{20,})', r'token=***REDACTED***'),
        (r'secret["\s:=]+([a-zA-Z0-9_-]{20,})', r'secret=***REDACTED***'),
        (r'authorization["\s:=]+([^\s"]+)', r'authorization=***REDACTED***'),
    ]
    
    @staticmethod
    def redact_sensitive(text: str) -> str:
        """Redact sensitive information from log messages"""
        if not isinstance(text, str):
            text = str(text)
        
        redacted = text
        for pattern, replacement in SecureLogger.SENSITIVE_PATTERNS:
            redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)
        
        return redacted
    
    @staticmethod
    def safe_log(level: int, message: str, *args, **kwargs):
        """Log with sensitive data redaction"""
        safe_message = SecureLogger.redact_sensitive(str(message))
        
        # Also redact args
        safe_args = tuple(SecureLogger.redact_sensitive(str(arg)) for arg in args)
        
        # Redact kwargs values
        safe_kwargs = {}
        for key, value in kwargs.items():
            if key in ['exc_info', 'stack_info', 'stacklevel']:
                safe_kwargs[key] = value
            else:
                safe_kwargs[key] = SecureLogger.redact_sensitive(str(value))
        
        logger.log(level, safe_message, *safe_args, **safe_kwargs)
    
    @staticmethod
    def info(message: str, *args, **kwargs):
        """Log info level with redaction"""
        SecureLogger.safe_log(logging.INFO, message, *args, **kwargs)
    
    @staticmethod
    def warning(message: str, *args, **kwargs):
        """Log warning level with redaction"""
        SecureLogger.safe_log(logging.WARNING, message, *args, **kwargs)
    
    @staticmethod
    def error(message: str, *args, **kwargs):
        """Log error level with redaction"""
        SecureLogger.safe_log(logging.ERROR, message, *args, **kwargs)
    
    @staticmethod
    def debug(message: str, *args, **kwargs):
        """Log debug level with redaction"""
        SecureLogger.safe_log(logging.DEBUG, message, *args, **kwargs)
