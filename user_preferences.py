"""
User Preferences module for Stocker App
Saves and loads user settings like budget, theme, language, currency
"""
import json
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class UserPreferences:
    """Manages user preferences and settings"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.preferences_file = data_dir / "preferences.json"
        self.preferences = {}
        self.load()
    
    def load(self) -> Dict:
        """Load preferences from file"""
        if self.preferences_file.exists():
            try:
                with open(self.preferences_file, 'r') as f:
                    self.preferences = json.load(f)
            except Exception as e:
                logger.error(f"Error loading preferences: {e}")
                self.preferences = self._get_defaults()
        else:
            self.preferences = self._get_defaults()
        return self.preferences
    
    def _get_defaults(self) -> Dict:
        """Get default preferences"""
        return {
            'budget': 1000.0,
            'budget_currency': 'USD',  # Currency when budget was saved
            'theme': 'light',
            'language': 'en',
            'currency': 'USD',
            'strategy': 'trading',
            'search_history': [],  # List of previously searched symbols
            'training_symbols': {  # Last used symbols for training per strategy
                'trading': 'AAPL,MSFT,GOOGL,AMZN,TSLA',
                'mixed': 'AAPL,MSFT,GOOGL,AMZN,TSLA,META,NVDA,NFLX',
                'investing': 'AAPL,MSFT,GOOGL,AMZN,TSLA'
            },
            'test_symbols': {  # Last used symbols for testing per strategy
                'trading': 'AAPL,MSFT,GOOGL,AMZN,TSLA',
                'mixed': 'AAPL,MSFT,GOOGL,AMZN,TSLA',
                'investing': 'AAPL,MSFT,GOOGL,AMZN,TSLA'
            },
            'auto_learner_scan_interval_hours': 6,  # Default scan interval
            'auto_learner_predictions_per_scan': 5,  # Default predictions per scan
            'auto_learner_min_confidence': 60,  # Default minimum confidence
            'backtest_strategies': ['trading', 'mixed', 'investing']  # Default: all strategies
        }
    
    def save(self):
        """Save preferences to file"""
        try:
            with open(self.preferences_file, 'w') as f:
                json.dump(self.preferences, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving preferences: {e}")
    
    def get(self, key: str, default=None):
        """Get a preference value"""
        return self.preferences.get(key, default)
    
    def set(self, key: str, value):
        """Set a preference value"""
        self.preferences[key] = value
        self.save()
    
    def get_budget(self, current_currency: str = None) -> float:
        """Get saved budget, converted to current currency if needed"""
        from localization import Localization
        
        budget = self.preferences.get('budget', 1000.0)
        saved_currency = self.preferences.get('budget_currency', 'USD')
        
        # If currency changed, convert budget
        if current_currency and saved_currency != current_currency:
            # Convert from saved currency to USD first, then to current currency
            rates = Localization.CURRENCY_RATES
            usd_budget = budget / rates.get(saved_currency, 1.0)
            converted_budget = usd_budget * rates.get(current_currency, 1.0)
            return converted_budget
        
        return budget
    
    def set_budget(self, budget: float, currency: str = 'USD'):
        """Save budget with currency"""
        self.set('budget', budget)
        self.set('budget_currency', currency)
    
    def get_search_history(self) -> list:
        """Get search history (list of symbols)"""
        return self.preferences.get('search_history', [])
    
    def add_search(self, symbol: str):
        """Add a symbol to search history"""
        history = self.get_search_history()
        symbol = symbol.upper().strip()
        # Remove if exists and add to front
        if symbol in history:
            history.remove(symbol)
        history.insert(0, symbol)
        # Keep only last 20 searches
        history = history[:20]
        self.set('search_history', history)
    
    def get_theme(self) -> str:
        """Get saved theme"""
        return self.preferences.get('theme', 'light')
    
    def set_theme(self, theme: str):
        """Save theme"""
        self.set('theme', theme)
    
    def get_language(self) -> str:
        """Get saved language"""
        return self.preferences.get('language', 'en')
    
    def set_language(self, language: str):
        """Save language"""
        self.set('language', language)
    
    def get_currency(self) -> str:
        """Get saved currency"""
        return self.preferences.get('currency', 'USD')
    
    def set_currency(self, currency: str):
        """Save currency"""
        self.set('currency', currency)
    
    def get_strategy(self) -> str:
        """Get saved strategy"""
        return self.preferences.get('strategy', 'trading')
    
    def set_strategy(self, strategy: str):
        """Save strategy"""
        self.set('strategy', strategy)
    
    def get_training_symbols(self, strategy: str) -> str:
        """Get last used training symbols for a strategy"""
        training_symbols = self.preferences.get('training_symbols', {})
        return training_symbols.get(strategy, 'AAPL,MSFT,GOOGL,AMZN,TSLA')
    
    def set_training_symbols(self, strategy: str, symbols: str):
        """Save training symbols for a strategy"""
        if 'training_symbols' not in self.preferences:
            self.preferences['training_symbols'] = {}
        self.preferences['training_symbols'][strategy] = symbols
        self.save()
    
    def get_test_symbols(self, strategy: str) -> str:
        """Get last used test symbols for a strategy"""
        test_symbols = self.preferences.get('test_symbols', {})
        return test_symbols.get(strategy, 'AAPL,MSFT,GOOGL,AMZN,TSLA')
    
    def set_test_symbols(self, strategy: str, symbols: str):
        """Save test symbols for a strategy"""
        if 'test_symbols' not in self.preferences:
            self.preferences['test_symbols'] = {}
        self.preferences['test_symbols'][strategy] = symbols
        self.save()
    
    def get_backtest_strategies(self) -> list:
        """Get selected strategies for backtesting"""
        return self.preferences.get('backtest_strategies', ['trading', 'mixed', 'investing'])
    
    def set_backtest_strategies(self, strategies: list):
        """Save selected strategies for backtesting"""
        self.set('backtest_strategies', strategies)



