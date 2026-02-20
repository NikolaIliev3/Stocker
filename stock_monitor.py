"""
Stock Monitoring System
Tracks stocks that have been analyzed and notifies when recommendations change to BUY
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class StockMonitor:
    """Monitors stocks for recommendation changes"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.monitor_file = self.data_dir / "monitored_stocks.json"
        self.monitored_stocks = self._load_monitored()
    
    def _load_monitored(self) -> Dict:
        """Load monitored stocks from file"""
        if self.monitor_file.exists():
            try:
                with open(self.monitor_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading monitored stocks: {e}")
        return {}
    
    def save(self):
        """Save monitored stocks to file"""
        try:
            with open(self.monitor_file, 'w') as f:
                json.dump(self.monitored_stocks, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving monitored stocks: {e}")
    
    def add_stock(self, symbol: str, action: str, strategy: str, confidence: float = 0):
        """Add or update a monitored stock
        
        Args:
            symbol: Stock symbol
            action: Current recommendation (BUY, SELL, HOLD, AVOID)
            strategy: Strategy used (trading, mixed, investing)
            confidence: Confidence level
        """
        symbol_upper = symbol.upper()
        self.monitored_stocks[symbol_upper] = {
            'symbol': symbol_upper,
            'last_action': action,
            'strategy': strategy,
            'last_confidence': confidence,
            'last_checked': datetime.now().isoformat(),
            'added_date': self.monitored_stocks.get(symbol_upper, {}).get('added_date', datetime.now().isoformat())
        }
        self.save()
        logger.debug(f"Added/updated monitored stock: {symbol_upper} ({action})")
    
    def update_stock_action(self, symbol: str, new_action: str, confidence: float = 0):
        """Update the action for a monitored stock"""
        symbol_upper = symbol.upper()
        if symbol_upper in self.monitored_stocks:
            old_action = self.monitored_stocks[symbol_upper].get('last_action', 'UNKNOWN')
            self.monitored_stocks[symbol_upper]['last_action'] = new_action
            self.monitored_stocks[symbol_upper]['last_confidence'] = confidence
            self.monitored_stocks[symbol_upper]['last_checked'] = datetime.now().isoformat()
            self.save()
            return old_action
        return None
    
    def get_monitored_stocks(self) -> List[str]:
        """Get list of all monitored stock symbols"""
        return list(self.monitored_stocks.keys())
    
    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """Get monitoring info for a stock"""
        return self.monitored_stocks.get(symbol.upper())
    
    def check_for_bullish_signal(self, symbol: str, new_action: str, confidence: float = 0) -> bool:
        """Check if stock changed from non-bullish to bullish (SELL in inverted logic)
        
        Returns:
            True if changed from HOLD/AVOID/BUY to SELL
        """
        symbol_upper = symbol.upper()
        if symbol_upper not in self.monitored_stocks:
            return False
        
        old_action = self.monitored_stocks[symbol_upper].get('last_action', 'UNKNOWN')
        
        # STANDARD logic: BUY action means bullish (buy opportunity)
        # Check if changed from non-BUY to BUY
        if old_action != 'BUY' and new_action == 'BUY':
            # Update the action
            self.update_stock_action(symbol_upper, new_action, confidence)
            return True
        
        # Update even if no signal (to track current state)
        self.update_stock_action(symbol_upper, new_action, confidence)
        return False
    
    # Alias for backward compatibility if needed, but we should use the new name
    def check_for_buy_signal(self, symbol: str, new_action: str, confidence: float = 0) -> bool:
        return self.check_for_bullish_signal(symbol, new_action, confidence)
    
    def remove_stock(self, symbol: str):
        """Remove a stock from monitoring"""
        symbol_upper = symbol.upper()
        if symbol_upper in self.monitored_stocks:
            del self.monitored_stocks[symbol_upper]
            self.save()
            logger.debug(f"Removed monitored stock: {symbol_upper}")
    
    def get_statistics(self) -> Dict:
        """Get monitoring statistics"""
        total = len(self.monitored_stocks)
        buy_count = sum(1 for s in self.monitored_stocks.values() if s.get('last_action') == 'BUY')
        sell_count = sum(1 for s in self.monitored_stocks.values() if s.get('last_action') == 'SELL')
        hold_count = sum(1 for s in self.monitored_stocks.values() if s.get('last_action') == 'HOLD')
        avoid_count = sum(1 for s in self.monitored_stocks.values() if s.get('last_action') == 'AVOID')
        
        return {
            'total_monitored': total,
            'buy': buy_count,
            'sell': sell_count,
            'hold': hold_count,
            'avoid': avoid_count
        }

