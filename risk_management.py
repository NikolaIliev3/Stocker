"""
Risk Management Module
Implements position sizing, stop-loss recommendations, VaR calculations, and portfolio risk metrics
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class RiskManager:
    """Manages risk calculations and position sizing recommendations"""
    
    def __init__(self, portfolio_value: float = 10000.0, max_position_size: float = 0.25):
        """
        Initialize Risk Manager
        
        Args:
            portfolio_value: Total portfolio value
            max_position_size: Maximum position size as fraction of portfolio (default 25%)
        """
        self.portfolio_value = portfolio_value
        self.max_position_size = max_position_size
    
    def calculate_position_size(self, entry_price: float, stop_loss: float, 
                               confidence: float, risk_per_trade: float = 0.02,
                               method: str = 'fixed_fraction', target_price: float = 0.0) -> Dict:
        """
        Calculate optimal position size using various methods
        
        Args:
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            confidence: Confidence score (0-100)
            risk_per_trade: Risk per trade as fraction of portfolio (default 2%)
            method: Position sizing method ('fixed_fraction', 'kelly', 'optimal_f')
            
        Returns:
            Dict with position size, shares, and risk metrics
        """
        if entry_price <= 0 or stop_loss <= 0:
            return {"error": "Invalid prices"}
        
        # HARD LIMIT: Enforce -2.0% Quant Safety Floor globally
        max_allowed_stop = entry_price * 0.98
        if stop_loss < max_allowed_stop:
             # Stop is too loose, force 2%
             stop_loss = max_allowed_stop
             
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            return {"error": "Stop loss equals entry price"}
        
        # Risk amount per trade
        risk_amount = self.portfolio_value * risk_per_trade
        
        # Base position size calculation
        if method == 'fixed_fraction':
            shares = risk_amount / risk_per_share
            position_value = shares * entry_price
            
            # Apply max position size constraint
            max_position_value = self.portfolio_value * self.max_position_size
            if position_value > max_position_value:
                shares = max_position_value / entry_price
                position_value = max_position_value
                risk_amount = shares * risk_per_share
            
            # Adjust for confidence
            confidence_multiplier = confidence / 100.0
            shares = shares * confidence_multiplier
            position_value = shares * entry_price
            risk_amount = shares * risk_per_share
            
        elif method == 'kelly':
            # Full Kelly Criterion (Fractional 0.5 for stability)
            # win_loss_ratio (b) = avg_win / avg_loss
            win_probability = confidence / 100.0
            
            # Reward-to-Risk ratio
            potential_win = abs(target_price - entry_price) if target_price > 0 else risk_per_share * 2
            b = potential_win / risk_per_share if risk_per_share > 0 else 1.0
            
            # Kelly Formula: f* = (bp - q) / b
            # q = 1 - p
            p = win_probability
            q = 1.0 - p
            
            k_star = (b * p - q) / b if b > 0 else 0
            
            # Use Half-Kelly (conservative) and cap at max_position_size
            fractional_kelly = k_star * 0.5
            position_fraction = max(0, min(fractional_kelly, self.max_position_size))
            
            position_value = self.portfolio_value * position_fraction
            shares = position_value / entry_price
            risk_amount = shares * risk_per_share
            
        elif method == 'optimal_f':
            # Optimal F (Ralph Vince)
            # Simplified version - would need historical win rate and avg win/loss
            # Using confidence as proxy for win rate
            win_rate = confidence / 100.0
            avg_win = risk_per_share * 2  # Assume 2:1 reward:risk
            avg_loss = risk_per_share
            
            if avg_loss > 0:
                f_value = ((win_rate * avg_win) - ((1 - win_rate) * avg_loss)) / avg_win
                f_value = max(0, min(f_value, self.max_position_size))
            else:
                f_value = self.max_position_size
            
            position_value = self.portfolio_value * f_value
            shares = position_value / entry_price
            risk_amount = shares * risk_per_share
            position_value = self.portfolio_value * f_value
            shares = position_value / entry_price
            risk_amount = shares * risk_per_share

        elif method == 'volatility_scaled':
            # Inverse Volatility Scaling (Risk Parity approach)
            # Target Volatility for the Position (e.g., impact on portfolio)
            # Default target: 0.5% daily volatility impact (relatively conservative)
            target_vol_impact = 0.005 
            
            # We need the asset's volatility (std dev of returns)
            # Use 'risk_per_trade' as a proxy for Target Volatility % if provided, otherwise default
            if risk_per_trade > 0:
                target_vol_impact = risk_per_trade / 4.0 # Scale down aggressive risk settings
            
            # Asset volatility estimate
            # If not provided explicitly, estimate from stop loss distance (assuming stop = 2*ATR)
            # ATR ~ Volatility. Stop dist = 2*Vol -> Vol = StopDist / 2
            # Vol % = (StopDist / Price) / 2
            stop_dist_pct = abs(entry_price - stop_loss) / entry_price
            estimated_volatility = stop_dist_pct / 2.0
            
            if estimated_volatility > 0:
                # Formula: (Portfolio * TargetVol) / AssetVol
                optimal_position_value = (self.portfolio_value * target_vol_impact) / estimated_volatility
                
                # Cap at max position size
                optimal_position_value = min(optimal_position_value, self.portfolio_value * self.max_position_size)
                
                shares = optimal_position_value / entry_price
                position_value = optimal_position_value
                risk_amount = shares * abs(entry_price - stop_loss)
            else:
                # Fallback if volatility is zero (shouldn't happen)
                shares = 0
                position_value = 0
                risk_amount = 0

        else:
            # Default to fixed fraction
            shares = risk_amount / risk_per_share
            position_value = shares * entry_price
        
        return {
            'shares': float(shares), # Ensure float for JSON serialization
            'position_value': round(position_value, 2),
            'risk_amount': round(risk_amount, 2),
            'stop_loss': round(stop_loss, 2), # Return enforced stop loss
            'risk_percentage': round((risk_amount / self.portfolio_value) * 100, 2),
            'position_percentage': round((position_value / self.portfolio_value) * 100, 2),
            'method': method,
            'risk_reward_ratio': round(abs(target_price - entry_price) / risk_per_share, 2) if risk_per_share > 0 and target_price > 0 else 1.0
        }
    
    def recommend_stop_loss(self, entry_price: float, current_price: float,
                           volatility: float, support_level: Optional[float] = None,
                           method: str = 'atr') -> Dict:
        """
        Recommend stop loss price using various methods
        
        Args:
            entry_price: Entry price
            current_price: Current market price
            volatility: ATR or standard deviation
            support_level: Nearest support level (optional)
            method: Stop loss method ('atr', 'percentage', 'support', 'trailing')
            
        Returns:
            Dict with stop loss price and reasoning
        """
        if entry_price <= 0:
            return {"error": "Invalid entry price"}
        
        stop_loss = None
        reasoning = []
        
        if method == 'atr' and volatility > 0:
            # Stop loss at 2x ATR below entry
            stop_loss = entry_price - (2 * volatility)
            reasoning.append(f"ATR-based: {volatility:.2f} ATR, stop at {stop_loss:.2f}")
        
        elif method == 'percentage':
            # Stop loss at 5% below entry
            stop_loss = entry_price * 0.95
            reasoning.append("Percentage-based: 5% below entry")
        
        elif method == 'support' and support_level:
            # Stop loss just below support
            stop_loss = support_level * 0.98
            reasoning.append(f"Support-based: Just below support at {support_level:.2f}")
        
        elif method == 'trailing':
            # Trailing stop loss (5% below current price)
            stop_loss = current_price * 0.95
            reasoning.append("Trailing stop: 5% below current price")
        
        else:
            # Default to percentage
            stop_loss = entry_price * 0.95
            reasoning.append("Default: 5% below entry")
        
        # HARD LIMIT: Never allow more than -2% risk per share on entry for Quant Mode
        max_allowed_stop = entry_price * 0.98
        if stop_loss > max_allowed_stop:
            # If the user-calculated stop is tighter than 2% (e.g. -1%), keep it.
            # We only intervene if it's too loose.
            pass
        elif stop_loss < max_allowed_stop:
             stop_loss = max_allowed_stop
             reasoning.append("Quant Safety: Capped at -2.0% Hard Stop Loss")
        
        risk_amount = entry_price - stop_loss
        risk_percentage = (risk_amount / entry_price) * 100
        
        return {
            'stop_loss': round(stop_loss, 2),
            'risk_amount': round(risk_amount, 2),
            'risk_percentage': round(risk_percentage, 2),
            'method': method,
            'reasoning': '; '.join(reasoning)
        }
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95,
                     method: str = 'historical') -> Dict:
        """
        Calculate Value at Risk (VaR)
        
        Args:
            returns: Series of returns
            confidence_level: Confidence level (default 95%)
            method: VaR method ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            Dict with VaR values and metrics
        """
        if len(returns) == 0:
            return {"error": "No returns data"}
        
        returns_clean = returns.dropna()
        if len(returns_clean) == 0:
            return {"error": "No valid returns data"}
        
        alpha = 1 - confidence_level
        
        if method == 'historical':
            # Historical simulation
            var = np.percentile(returns_clean, alpha * 100)
            var_abs = abs(var)
            
        elif method == 'parametric':
            # Parametric (assumes normal distribution)
            mean_return = returns_clean.mean()
            std_return = returns_clean.std()
            var = stats.norm.ppf(alpha, mean_return, std_return)
            var_abs = abs(var)
            
        elif method == 'monte_carlo':
            # Monte Carlo simulation (simplified)
            mean_return = returns_clean.mean()
            std_return = returns_clean.std()
            simulations = 10000
            simulated_returns = np.random.normal(mean_return, std_return, simulations)
            var = np.percentile(simulated_returns, alpha * 100)
            var_abs = abs(var)
        else:
            # Default to historical
            var = np.percentile(returns_clean, alpha * 100)
            var_abs = abs(var)
        
        # Calculate Expected Shortfall (CVaR)
        if method == 'historical':
            cvar = returns_clean[returns_clean <= var].mean()
        else:
            cvar = var * 1.5  # Approximation
        
        return {
            'var': round(var, 4),
            'var_percentage': round(var * 100, 2),
            'var_absolute': round(var_abs, 4),
            'cvar': round(cvar, 4),
            'confidence_level': confidence_level,
            'method': method,
            'expected_shortfall': round(abs(cvar), 4)
        }
    
    def calculate_portfolio_risk(self, positions: list, correlations: Optional[pd.DataFrame] = None) -> Dict:
        """
        Calculate portfolio-level risk metrics
        
        Args:
            positions: List of dicts with 'symbol', 'value', 'volatility' keys
            correlations: Optional correlation matrix between positions
            
        Returns:
            Dict with portfolio risk metrics
        """
        if not positions:
            return {"error": "No positions provided"}
        
        total_value = sum(p.get('value', 0) for p in positions)
        if total_value == 0:
            return {"error": "Total portfolio value is zero"}
        
        # Calculate weighted average volatility
        weighted_vol = sum(p.get('value', 0) * p.get('volatility', 0) for p in positions) / total_value
        
        # Calculate portfolio variance (simplified - assumes equal correlation if not provided)
        if correlations is not None and len(positions) > 1:
            # Full portfolio variance calculation
            weights = np.array([p.get('value', 0) / total_value for p in positions])
            cov_matrix = np.outer(
                [p.get('volatility', 0) for p in positions],
                [p.get('volatility', 0) for p in positions]
            ) * correlations.values
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
        else:
            # Simplified: assume 0.5 correlation
            portfolio_variance = weighted_vol ** 2
            if len(positions) > 1:
                # Add diversification benefit
                portfolio_variance = portfolio_variance * (1 + 0.5 * (len(positions) - 1) / len(positions))
            portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Concentration risk (Herfindahl index)
        weights = [p.get('value', 0) / total_value for p in positions]
        concentration = sum(w ** 2 for w in weights)
        
        # Number of positions
        num_positions = len(positions)
        
        return {
            'total_value': round(total_value, 2),
            'weighted_volatility': round(weighted_vol, 4),
            'portfolio_volatility': round(portfolio_volatility, 4),
            'concentration_index': round(concentration, 4),
            'num_positions': num_positions,
            'diversification_score': round(1 - concentration, 4),  # Higher is better
            'risk_level': self._classify_risk(portfolio_volatility)
        }
    
    def _classify_risk(self, volatility: float) -> str:
        """Classify risk level based on volatility"""
        if volatility < 0.15:
            return "Low"
        elif volatility < 0.30:
            return "Medium"
        elif volatility < 0.50:
            return "High"
        else:
            return "Very High"
    
    def calculate_max_drawdown(self, prices: pd.Series) -> Dict:
        """
        Calculate maximum drawdown
        
        Args:
            prices: Series of prices
            
        Returns:
            Dict with drawdown metrics
        """
        if len(prices) == 0:
            return {"error": "No price data"}
        
        # Calculate running maximum
        running_max = prices.expanding().max()
        
        # Calculate drawdown
        drawdown = (prices - running_max) / running_max
        
        max_drawdown = drawdown.min()
        max_drawdown_idx = drawdown.idxmin()
        
        # Find recovery point
        recovery_idx = None
        if max_drawdown_idx is not None:
            recovery_data = drawdown.loc[max_drawdown_idx:]
            recovery_points = recovery_data[recovery_data >= -0.01]  # Within 1% of peak
            if len(recovery_points) > 0:
                recovery_idx = recovery_points.index[0]
        
        return {
            'max_drawdown': round(max_drawdown, 4),
            'max_drawdown_percentage': round(max_drawdown * 100, 2),
            'max_drawdown_date': max_drawdown_idx,
            'recovery_date': recovery_idx,
            'duration_days': (recovery_idx - max_drawdown_idx).days if recovery_idx and max_drawdown_idx else None
        }

