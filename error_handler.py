"""
Enhanced Error Handling Module
Provides comprehensive error handling, logging, and user-friendly error messages
"""
import logging
import traceback
from typing import Dict, Optional, Callable
from functools import wraps
import sys

logger = logging.getLogger(__name__)


class ErrorHandler:
    """Centralized error handling"""
    
    ERROR_CODES = {
        'DATA_FETCH_ERROR': 'DF001',
        'DATA_VALIDATION_ERROR': 'DV001',
        'ANALYSIS_ERROR': 'AN001',
        'ML_MODEL_ERROR': 'ML001',
        'NETWORK_ERROR': 'NW001',
        'CONFIGURATION_ERROR': 'CF001',
        'UNKNOWN_ERROR': 'UN001'
    }
    
    @staticmethod
    def handle_error(error: Exception, context: str = "", 
                    user_message: Optional[str] = None) -> Dict:
        """
        Handle an error and return structured error information
        
        Args:
            error: Exception object
            context: Context where error occurred
            user_message: User-friendly error message
            
        Returns:
            Dict with error information
        """
        error_type = type(error).__name__
        error_message = str(error)
        error_code = ErrorHandler._classify_error(error)
        
        # Log error
        logger.error(f"[{error_code}] {context}: {error_type}: {error_message}")
        logger.debug(traceback.format_exc())
        
        # Generate user-friendly message
        if user_message is None:
            user_message = ErrorHandler._generate_user_message(error, context)
        
        return {
            'error': True,
            'error_code': error_code,
            'error_type': error_type,
            'error_message': error_message,
            'user_message': user_message,
            'context': context
        }
    
    @staticmethod
    def _classify_error(error: Exception) -> str:
        """Classify error type"""
        error_type = type(error).__name__
        
        if 'Connection' in error_type or 'Timeout' in error_type or 'Network' in error_type:
            return ErrorHandler.ERROR_CODES['NETWORK_ERROR']
        elif 'Value' in error_type or 'Key' in error_type or 'Index' in error_type:
            return ErrorHandler.ERROR_CODES['DATA_VALIDATION_ERROR']
        elif 'Model' in error_type or 'ML' in error_type:
            return ErrorHandler.ERROR_CODES['ML_MODEL_ERROR']
        elif 'Config' in error_type or 'Setting' in error_type:
            return ErrorHandler.ERROR_CODES['CONFIGURATION_ERROR']
        else:
            return ErrorHandler.ERROR_CODES['UNKNOWN_ERROR']
    
    @staticmethod
    def _generate_user_message(error: Exception, context: str) -> str:
        """Generate user-friendly error message"""
        error_type = type(error).__name__
        
        if 'Connection' in error_type:
            return "Unable to connect to data source. Please check your internet connection."
        elif 'Timeout' in error_type:
            return "Request timed out. The server may be busy. Please try again."
        elif 'KeyError' in error_type:
            return "Data format error. Some required information is missing."
        elif 'ValueError' in error_type:
            return "Invalid data value. Please check the stock symbol and try again."
        elif 'FileNotFound' in error_type:
            return "Required file not found. The application may need to be reinstalled."
        else:
            return f"An error occurred: {str(error)}. Please try again or contact support."
    
    @staticmethod
    def safe_execute(func: Callable, default_return=None, context: str = ""):
        """
        Decorator for safe function execution with error handling
        
        Args:
            func: Function to wrap
            default_return: Value to return on error
            context: Context description
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_info = ErrorHandler.handle_error(e, context or func.__name__)
                if default_return is not None:
                    return default_return
                return error_info
        return wrapper


def safe_call(func: Callable, *args, default_return=None, context: str = "", **kwargs):
    """
    Safely call a function with error handling
    
    Args:
        func: Function to call
        *args: Positional arguments
        default_return: Value to return on error
        context: Context description
        **kwargs: Keyword arguments
        
    Returns:
        Function result or error dict/default_return
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_info = ErrorHandler.handle_error(e, context or func.__name__)
        if default_return is not None:
            return default_return
        return error_info


class ValidationError(Exception):
    """Custom exception for validation errors"""
    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field = field
        super().__init__(self.message)


class DataFetchError(Exception):
    """Custom exception for data fetching errors"""
    def __init__(self, message: str, symbol: str = None):
        self.message = message
        self.symbol = symbol
        super().__init__(self.message)


class AnalysisError(Exception):
    """Custom exception for analysis errors"""
    def __init__(self, message: str, analysis_type: str = None):
        self.message = message
        self.analysis_type = analysis_type
        super().__init__(self.message)

