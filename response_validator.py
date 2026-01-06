"""
Response Validation Module
Validates API responses for security
"""
import json
import logging
from typing import Dict, Any
import requests

logger = logging.getLogger(__name__)


class ResponseValidator:
    """Validate API responses for security"""
    
    MAX_RESPONSE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_CONTENT_TYPES = [
        'application/json',
        'text/plain',
        'text/html',
        'application/xml',
    ]
    
    # Suspicious patterns that might indicate injection
    SUSPICIOUS_PATTERNS = [
        '<script',
        'javascript:',
        'onerror=',
        'onload=',
        'eval(',
        'exec(',
        '__import__',
        'subprocess',
    ]
    
    @staticmethod
    def validate_response(response: requests.Response) -> bool:
        """
        Validate API response for security
        
        Args:
            response: Response object to validate
        
        Returns:
            True if valid, False otherwise
        """
        # Check status code
        if response.status_code >= 500:
            logger.warning(f"Server error: {response.status_code}")
            return False
        
        # Check content type
        content_type = response.headers.get('Content-Type', '').lower()
        if not any(allowed in content_type for allowed in ResponseValidator.ALLOWED_CONTENT_TYPES):
            logger.warning(f"Unexpected content type: {content_type}")
            return False
        
        # Check response size
        content_length = response.headers.get('Content-Length')
        if content_length:
            try:
                size = int(content_length)
                if size > ResponseValidator.MAX_RESPONSE_SIZE:
                    logger.warning(f"Response too large: {size} bytes")
                    return False
            except ValueError:
                pass
        
        # Check actual content size
        try:
            content = response.content
            if len(content) > ResponseValidator.MAX_RESPONSE_SIZE:
                logger.warning(f"Response content too large: {len(content)} bytes")
                return False
        except Exception as e:
            logger.error(f"Error checking response size: {e}")
            return False
        
        # Validate JSON structure if JSON
        if 'application/json' in content_type:
            try:
                data = response.json()
                # Check for suspicious patterns
                data_str = json.dumps(data).lower()
                for pattern in ResponseValidator.SUSPICIOUS_PATTERNS:
                    if pattern in data_str:
                        logger.warning(f"Suspicious content in response: {pattern}")
                        return False
            except json.JSONDecodeError:
                logger.warning("Invalid JSON in response")
                return False
            except Exception as e:
                logger.error(f"Error validating JSON response: {e}")
                return False
        
        return True
    
    @staticmethod
    def validate_json_structure(data: Dict[str, Any], required_fields: list = None) -> bool:
        """
        Validate JSON structure
        
        Args:
            data: JSON data to validate
            required_fields: List of required field names
        
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(data, dict):
            return False
        
        # Check required fields
        if required_fields:
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Missing required field: {field}")
                    return False
        
        # Check for suspicious patterns in string values
        def check_value(value):
            if isinstance(value, str):
                value_lower = value.lower()
                for pattern in ResponseValidator.SUSPICIOUS_PATTERNS:
                    if pattern in value_lower:
                        logger.warning(f"Suspicious pattern in data: {pattern}")
                        return False
            elif isinstance(value, dict):
                for v in value.values():
                    if not check_value(v):
                        return False
            elif isinstance(value, list):
                for item in value:
                    if not check_value(item):
                        return False
            return True
        
        return check_value(data)
