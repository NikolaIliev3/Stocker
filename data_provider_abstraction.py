"""
Data Provider Abstraction Layer
Allows multiple data sources to be used interchangeably
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import logging
import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """Abstract base class for data providers"""
    
    @abstractmethod
    def fetch_stock_data(self, symbol: str) -> Optional[Dict]:
        """Fetch current stock data"""
        pass
    
    @abstractmethod
    def fetch_history(self, symbol: str, period: str = "1y") -> Optional[Dict]:
        """Fetch historical price data"""
        pass
    
    @abstractmethod
    def fetch_financials(self, symbol: str) -> Optional[Dict]:
        """Fetch financial statements"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Get provider name"""
        pass


class YahooFinanceProvider(DataProvider):
    """Yahoo Finance data provider (current implementation)"""
    
    def __init__(self):
        self.name = "Yahoo Finance"
        self.available = True
    
    def fetch_stock_data(self, symbol: str) -> Optional[Dict]:
        """Fetch stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or 'symbol' not in info:
                return None
            
            current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
            
            return {
                'symbol': symbol.upper(),
                'currentPrice': current_price,
                'info': info
            }
        except Exception as e:
            logger.error(f"Yahoo Finance error for {symbol}: {e}")
            return None
    
    def fetch_history(self, symbol: str, period: str = "1y") -> Optional[Dict]:
        """Fetch historical data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return None
            
            # Convert to dict format
            data = []
            for date, row in hist.iterrows():
                data.append({
                    'date': date.strftime('%Y-%m-%d'),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
            
            return {'data': data}
        except Exception as e:
            logger.error(f"Yahoo Finance history error for {symbol}: {e}")
            return None
    
    def fetch_financials(self, symbol: str) -> Optional[Dict]:
        """Fetch financial statements from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            financials = ticker.financials
            balance_sheet = ticker.balance_sheet
            cashflow = ticker.cashflow
            
            return {
                'info': info,
                'financials': financials.to_dict() if not financials.empty else {},
                'balance_sheet': balance_sheet.to_dict() if not balance_sheet.empty else {},
                'cashflow': cashflow.to_dict() if not cashflow.empty else {}
            }
        except Exception as e:
            logger.error(f"Yahoo Finance financials error for {symbol}: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if Yahoo Finance is available"""
        return self.available
    
    def get_name(self) -> str:
        """Get provider name"""
        return self.name


class MultiDataProvider:
    """Manages multiple data providers with fallback"""
    
    def __init__(self, providers: List[DataProvider] = None):
        """
        Initialize with list of providers (in priority order)
        
        Args:
            providers: List of data providers, ordered by priority
        """
        if providers is None:
            # Default: Yahoo Finance
            providers = [YahooFinanceProvider()]
        
        self.providers = providers
        self.current_provider = None
        self.provider_stats = {p.get_name(): {'success': 0, 'failures': 0} for p in providers}
    
    def _get_available_provider(self) -> Optional[DataProvider]:
        """Get first available provider"""
        for provider in self.providers:
            if provider.is_available():
                return provider
        return None
    
    def fetch_stock_data(self, symbol: str) -> Optional[Dict]:
        """Fetch stock data using available provider"""
        provider = self._get_available_provider()
        if not provider:
            logger.error("No available data providers")
            return None
        
        self.current_provider = provider
        result = provider.fetch_stock_data(symbol)
        
        # Update stats
        if result:
            self.provider_stats[provider.get_name()]['success'] += 1
        else:
            self.provider_stats[provider.get_name()]['failures'] += 1
        
        return result
    
    def fetch_history(self, symbol: str, period: str = "1y") -> Optional[Dict]:
        """Fetch historical data using available provider"""
        provider = self._get_available_provider()
        if not provider:
            return None
        
        result = provider.fetch_history(symbol, period)
        
        if result:
            self.provider_stats[provider.get_name()]['success'] += 1
        else:
            self.provider_stats[provider.get_name()]['failures'] += 1
        
        return result
    
    def fetch_financials(self, symbol: str) -> Optional[Dict]:
        """Fetch financials using available provider"""
        provider = self._get_available_provider()
        if not provider:
            return None
        
        result = provider.fetch_financials(symbol)
        
        if result:
            self.provider_stats[provider.get_name()]['success'] += 1
        else:
            self.provider_stats[provider.get_name()]['failures'] += 1
        
        return result
    
    def get_provider_stats(self) -> Dict:
        """Get statistics for each provider"""
        stats = {}
        for name, stat in self.provider_stats.items():
            total = stat['success'] + stat['failures']
            stats[name] = {
                'success': stat['success'],
                'failures': stat['failures'],
                'success_rate': stat['success'] / total if total > 0 else 0,
                'current_provider': name == (self.current_provider.get_name() if self.current_provider else None)
            }
        return stats
    
    def add_provider(self, provider: DataProvider, priority: int = None):
        """
        Add a new data provider
        
        Args:
            provider: Data provider instance
            priority: Insertion position (None = append)
        """
        if priority is not None:
            self.providers.insert(priority, provider)
        else:
            self.providers.append(provider)
        
        self.provider_stats[provider.get_name()] = {'success': 0, 'failures': 0}
    
    def get_current_provider_name(self) -> str:
        """Get name of current provider"""
        if self.current_provider:
            return self.current_provider.get_name()
        return "None"

