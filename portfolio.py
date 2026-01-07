"""
Portfolio tracking module for Stocker App
Tracks wins, losses, balance, and investment performance
Integrated with risk management and advanced analytics
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging
import pandas as pd

# Import new modules
try:
    from risk_management import RiskManager
    from advanced_analytics import AdvancedAnalytics
    HAS_NEW_MODULES = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"New modules not available: {e}")
    HAS_NEW_MODULES = False

logger = logging.getLogger(__name__)


class Portfolio:
    """Manages portfolio tracking and performance"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.portfolio_file = self.data_dir / "portfolio.json"
        self.load()
        
        # Initialize new modules if available
        if HAS_NEW_MODULES:
            try:
                self.risk_manager = RiskManager(
                    portfolio_value=self.balance,
                    max_position_size=0.25
                )
                self.analytics = AdvancedAnalytics()
                logger.info("Risk management and analytics initialized for portfolio")
            except Exception as e:
                logger.warning(f"Could not initialize risk/analytics modules: {e}")
                self.risk_manager = None
                self.analytics = None
        else:
            self.risk_manager = None
            self.analytics = None
    
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
                                 stop_loss: float, action: str,
                                 confidence: float = 50.0,
                                 use_risk_management: bool = True) -> Dict:
        """Calculate potential win/loss for a trade (supports fractional shares)
        
        Args:
            symbol: Stock symbol
            budget: Investment budget
            entry_price: Entry price
            target_price: Target price
            stop_loss: Stop loss price
            action: Trade action (BUY/SELL/HOLD)
            confidence: Confidence score (0-100)
            use_risk_management: If True, use risk management for position sizing
        """
        if budget <= 0 or entry_price <= 0:
            return {
                "error": "Invalid budget or entry price",
                "potential_win": 0,
                "potential_loss": 0,
                "shares": 0
            }
        
        # Use risk management for position sizing if available
        if use_risk_management and self.risk_manager and stop_loss > 0:
            try:
                # Update portfolio value
                self.risk_manager.portfolio_value = self.balance
                
                # Calculate optimal position size
                position_info = self.risk_manager.calculate_position_size(
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    confidence=confidence,
                    method='fixed_fraction'
                )
                
                if 'error' not in position_info:
                    shares = position_info['shares']
                    budget_used = position_info['position_value']
                    
                    # Calculate win/loss with optimal position
                    if action == "BUY":
                        potential_win = (target_price - entry_price) * shares
                        potential_loss = (entry_price - stop_loss) * shares
                    elif action == "SELL":
                        potential_win = (entry_price - target_price) * shares
                        potential_loss = (stop_loss - entry_price) * shares
                    else:
                        if target_price > entry_price:
                            potential_win = (target_price - entry_price) * shares
                            potential_loss = (entry_price - stop_loss) * shares
                        else:
                            potential_win = (entry_price - target_price) * shares
                            potential_loss = (stop_loss - entry_price) * shares
                    
                    result = {
                        "symbol": symbol,
                        "action": action,
                        "shares": shares,
                        "entry_price": entry_price,
                        "target_price": target_price,
                        "stop_loss": stop_loss,
                        "budget_used": budget_used,
                        "potential_win": potential_win,
                        "potential_loss": potential_loss,
                        "potential_win_percent": (potential_win / budget_used) * 100 if budget_used > 0 else 0,
                        "potential_loss_percent": (potential_loss / budget_used) * 100 if budget_used > 0 else 0,
                        "risk_reward_ratio": abs(potential_win / potential_loss) if potential_loss > 0 else 0,
                        "risk_management": {
                            "position_percentage": position_info.get('position_percentage', 0),
                            "risk_percentage": position_info.get('risk_percentage', 0),
                            "method": position_info.get('method', 'fixed_fraction')
                        }
                    }
                    
                    # Get stop loss recommendation if not provided
                    if stop_loss <= 0:
                        stop_rec = self.risk_manager.recommend_stop_loss(
                            entry_price=entry_price,
                            current_price=entry_price,
                            volatility=entry_price * 0.02,  # Estimate 2% volatility
                            method='percentage'
                        )
                        if 'stop_loss' in stop_rec:
                            result['recommended_stop_loss'] = stop_rec['stop_loss']
                            result['stop_loss_reasoning'] = stop_rec.get('reasoning', '')
                    
                    return result
            except Exception as e:
                logger.warning(f"Risk management calculation failed: {e}, using basic calculation")
        
        # Fallback to basic calculation
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
        """Get portfolio statistics with advanced analytics"""
        total_trades = len(self.trades)
        win_rate = (self.wins / total_trades * 100) if total_trades > 0 else 0
        
        total_pnl = sum(trade.get('pnl', 0) for trade in self.trades)
        total_return = ((self.balance - self.initial_balance) / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        stats = {
            "balance": self.balance,
            "initial_balance": self.initial_balance,
            "total_return": total_return,
            "total_pnl": total_pnl,
            "total_trades": total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": win_rate
        }
        
        # Add advanced analytics if available
        if self.analytics and len(self.trades) > 0:
            try:
                # Calculate returns from trades
                returns = pd.Series([trade.get('pnl_percent', 0) / 100 for trade in self.trades])
                
                # Calculate drawdown
                if len(returns) > 0:
                    prices = pd.Series([self.initial_balance])
                    for trade in self.trades:
                        prices = pd.concat([prices, pd.Series([prices.iloc[-1] + trade.get('pnl', 0)])])
                    
                    drawdown_analysis = self.analytics.calculate_drawdown_analysis(prices)
                    stats['drawdown'] = drawdown_analysis
                
                # Calculate VaR if we have enough data
                if len(returns) >= 10:
                    var_result = self.risk_manager.calculate_var(returns, confidence_level=0.95) if self.risk_manager else None
                    if var_result and 'error' not in var_result:
                        stats['var_95'] = var_result.get('var_percentage', 0)
            except Exception as e:
                logger.debug(f"Could not calculate advanced analytics: {e}")
        
        return stats
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent trades"""
        return self.trades[-limit:] if self.trades else []

