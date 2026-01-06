"""
Portfolio tracking module for Stocker App
Tracks wins, losses, balance, and investment performance
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class Portfolio:
    """Manages portfolio tracking and performance"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.portfolio_file = self.data_dir / "portfolio.json"
        self.load()
    
    def load(self):
        """Load portfolio data from file"""
        if self.portfolio_file.exists():
            try:
                with open(self.portfolio_file, 'r') as f:
                    data = json.load(f)
                    self.balance = data.get('balance', 10000.0)
                    self.initial_balance = data.get('initial_balance', 10000.0)
                    self.trades = data.get('trades', [])
                    self.wins = data.get('wins', 0)
                    self.losses = data.get('losses', 0)
            except Exception as e:
                logger.error(f"Error loading portfolio: {e}")
                self._initialize_default()
        else:
            self._initialize_default()
    
    def _initialize_default(self):
        """Initialize portfolio with default values"""
        self.balance = 10000.0
        self.initial_balance = 10000.0
        self.trades = []
        self.wins = 0
        self.losses = 0
    
    def save(self):
        """Save portfolio data to file"""
        try:
            data = {
                'balance': self.balance,
                'initial_balance': self.initial_balance,
                'trades': self.trades,
                'wins': self.wins,
                'losses': self.losses,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.portfolio_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving portfolio: {e}")
    
    def set_balance(self, balance: float):
        """Set initial balance"""
        self.balance = balance
        self.initial_balance = balance
        self.save()
    
    def calculate_potential_trade(self, symbol: str, budget: float, 
                                 entry_price: float, target_price: float,
                                 stop_loss: float, action: str) -> Dict:
        """Calculate potential win/loss for a trade (supports fractional shares)"""
        if budget <= 0 or entry_price <= 0:
            return {
                "error": "Invalid budget or entry price",
                "potential_win": 0,
                "potential_loss": 0,
                "shares": 0
            }
        
        # Calculate fractional shares (supports partial stock purchases)
        shares = budget / entry_price
        
        if action == "BUY":
            potential_win = (target_price - entry_price) * shares
            potential_loss = (entry_price - stop_loss) * shares
        elif action == "SELL":  # SELL (short)
            potential_win = (entry_price - target_price) * shares
            potential_loss = (stop_loss - entry_price) * shares
        else:  # HOLD or other - treat as BUY scenario if target > entry, else SELL
            if target_price > entry_price:
                # Bullish scenario - calculate as BUY
                potential_win = (target_price - entry_price) * shares
                potential_loss = (entry_price - stop_loss) * shares
                action = "BUY"  # Update action for display
            else:
                # Bearish scenario - calculate as SELL
                potential_win = (entry_price - target_price) * shares
                potential_loss = (stop_loss - entry_price) * shares
                action = "SELL"  # Update action for display
        
        return {
            "symbol": symbol,
            "action": action,
            "shares": shares,
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "budget_used": shares * entry_price,
            "potential_win": potential_win,
            "potential_loss": potential_loss,
            "potential_win_percent": (potential_win / (shares * entry_price)) * 100 if shares * entry_price > 0 else 0,
            "potential_loss_percent": (potential_loss / (shares * entry_price)) * 100 if shares * entry_price > 0 else 0,
            "risk_reward_ratio": abs(potential_win / potential_loss) if potential_loss > 0 else 0
        }
    
    def record_trade(self, symbol: str, action: str, shares: int,
                    entry_price: float, exit_price: float, 
                    budget_used: float) -> Dict:
        """Record a completed trade"""
        pnl = (exit_price - entry_price) * shares if action == "BUY" else (entry_price - exit_price) * shares
        pnl_percent = (pnl / budget_used) * 100 if budget_used > 0 else 0
        
        trade = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "action": action,
            "shares": shares,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "budget_used": budget_used,
            "pnl": pnl,
            "pnl_percent": pnl_percent,
            "is_win": pnl > 0
        }
        
        self.trades.append(trade)
        self.balance += pnl
        
        if pnl > 0:
            self.wins += 1
        else:
            self.losses += 1
        
        self.save()
        
        return trade
    
    def get_statistics(self) -> Dict:
        """Get portfolio statistics"""
        total_trades = len(self.trades)
        win_rate = (self.wins / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum(trade.get('pnl', 0) for trade in self.trades)
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        return {
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "total_return": total_return,
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": win_rate
        }
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent trades"""
        return self.trades[-limit:] if self.trades else []

