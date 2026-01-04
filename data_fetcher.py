"""
Data fetching module for Stocker App
Handles secure communication with backend proxy
"""
import requests
import logging
from security import get_secure_session, DataValidator, SecurityError
from config import BACKEND_API_URL
import json

logger = logging.getLogger(__name__)


class StockDataFetcher:
    """Fetches stock data securely from backend proxy"""
    
    def __init__(self):
        self.session = get_secure_session()
        self.base_url = BACKEND_API_URL
    
    def fetch_stock_data(self, symbol: str) -> dict:
        """Fetch current stock data for a symbol"""
        try:
            url = f"{self.base_url}/stock/{symbol}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Validate data integrity
            if not DataValidator.validate_stock_data(data):
                raise SecurityError("Invalid stock data received from server")
            
            return data
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching stock data: {e}")
            raise
        except SecurityError as e:
            logger.error(f"Security error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def fetch_stock_history(self, symbol: str, period: str = "1y", interval: str = "1d") -> dict:
        """Fetch historical stock data for charting"""
        try:
            url = f"{self.base_url}/stock/{symbol}/history"
            params = {"period": period, "interval": interval}
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching stock history: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    def fetch_financials(self, symbol: str) -> dict:
        """Fetch financial statements for investing analysis"""
        try:
            url = f"{self.base_url}/stock/{symbol}/financials"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            return response.json()
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching financials: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise

