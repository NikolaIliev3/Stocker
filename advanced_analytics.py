"""
Advanced Analytics Module
Implements correlation analysis, portfolio optimization, and Monte Carlo simulation
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from scipy.optimize import minimize
from scipy.stats import norm

logger = logging.getLogger(__name__)


class AdvancedAnalytics:
    """Advanced analytics for portfolio and market analysis"""
    
    def calculate_correlation_matrix(self, symbols: List[str], 
                                    returns_data: Dict[str, pd.Series],
                                    method: str = 'pearson') -> Dict:
        """
        Calculate correlation matrix between multiple stocks
        
        Args:
            symbols: List of stock symbols
            returns_data: Dict mapping symbols to return Series
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dict with correlation matrix and insights
        """
        if len(symbols) < 2:
            return {"error": "Need at least 2 symbols for correlation"}
        
        # Align all return series
        returns_df = pd.DataFrame()
        for symbol in symbols:
            if symbol in returns_data and len(returns_data[symbol]) > 0:
                returns_df[symbol] = returns_data[symbol]
        
        if returns_df.empty:
            return {"error": "No valid returns data"}
        
        # Calculate correlation
        corr_matrix = returns_df.corr(method=method)
        
        # Find highest and lowest correlations
        corr_values = corr_matrix.values
        np.fill_diagonal(corr_values, np.nan)  # Remove diagonal
        
        max_corr_idx = np.unravel_index(np.nanargmax(corr_values), corr_values.shape)
        min_corr_idx = np.unravel_index(np.nanargmin(corr_values), corr_values.shape)
        
        max_corr = corr_values[max_corr_idx]
        min_corr = corr_values[min_corr_idx]
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'max_correlation': {
                'value': float(max_corr),
                'symbols': [symbols[max_corr_idx[0]], symbols[max_corr_idx[1]]]
            },
            'min_correlation': {
                'value': float(min_corr),
                'symbols': [symbols[min_corr_idx[0]], symbols[min_corr_idx[1]]]
            },
            'average_correlation': float(np.nanmean(corr_values)),
            'method': method
        }
    
    def optimize_portfolio(self, symbols: List[str], 
                          returns_data: Dict[str, pd.Series],
                          risk_free_rate: float = 0.02,
                          method: str = 'sharpe') -> Dict:
        """
        Optimize portfolio weights using Modern Portfolio Theory
        
        Args:
            symbols: List of stock symbols
            returns_data: Dict mapping symbols to return Series
            risk_free_rate: Risk-free rate (annual)
            method: Optimization method ('sharpe', 'min_variance', 'max_return')
            
        Returns:
            Dict with optimal weights and portfolio metrics
        """
        if len(symbols) < 2:
            return {"error": "Need at least 2 symbols for optimization"}
        
        # Prepare data
        returns_df = pd.DataFrame()
        for symbol in symbols:
            if symbol in returns_data and len(returns_data[symbol]) > 0:
                returns_df[symbol] = returns_data[symbol]
        
        if returns_df.empty:
            return {"error": "No valid returns data"}
        
        # Calculate expected returns and covariance
        expected_returns = returns_df.mean() * 252  # Annualize
        cov_matrix = returns_df.cov() * 252  # Annualize
        
        n_assets = len(symbols)
        
        # Objective functions
        def portfolio_return(weights):
            return np.sum(expected_returns * weights)
        
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        
        def negative_sharpe(weights):
            ret = portfolio_return(weights)
            vol = portfolio_volatility(weights)
            if vol == 0:
                return -np.inf
            return -(ret - risk_free_rate) / vol
        
        def portfolio_variance(weights):
            return portfolio_volatility(weights) ** 2
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: weights between 0 and 1 (long-only)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        try:
            if method == 'sharpe':
                result = minimize(negative_sharpe, initial_weights,
                                method='SLSQP', bounds=bounds,
                                constraints=constraints)
            elif method == 'min_variance':
                result = minimize(portfolio_variance, initial_weights,
                                method='SLSQP', bounds=bounds,
                                constraints=constraints)
            elif method == 'max_return':
                result = minimize(lambda w: -portfolio_return(w), initial_weights,
                                method='SLSQP', bounds=bounds,
                                constraints=constraints)
            else:
                result = minimize(negative_sharpe, initial_weights,
                                method='SLSQP', bounds=bounds,
                                constraints=constraints)
            
            optimal_weights = result.x
            optimal_return = portfolio_return(optimal_weights)
            optimal_vol = portfolio_volatility(optimal_weights)
            sharpe_ratio = (optimal_return - risk_free_rate) / optimal_vol if optimal_vol > 0 else 0
            
            # Create weight dict
            weights_dict = {symbols[i]: float(optimal_weights[i]) 
                          for i in range(len(symbols))}
            
            return {
                'optimal_weights': weights_dict,
                'expected_return': float(optimal_return),
                'expected_volatility': float(optimal_vol),
                'sharpe_ratio': float(sharpe_ratio),
                'method': method,
                'optimization_success': result.success
            }
        
        except Exception as e:
            logger.error(f"Portfolio optimization error: {e}")
            return {"error": str(e)}
    
    def monte_carlo_simulation(self, initial_price: float, 
                              expected_return: float,
                              volatility: float,
                              days: int = 252,
                              simulations: int = 10000,
                              confidence_level: float = 0.95) -> Dict:
        """
        Monte Carlo simulation for price forecasting
        
        Args:
            initial_price: Starting price
            expected_return: Expected daily return
            volatility: Daily volatility
            days: Number of days to simulate
            simulations: Number of simulation runs
            confidence_level: Confidence level for intervals
            
        Returns:
            Dict with simulation results
        """
        if initial_price <= 0 or volatility < 0:
            return {"error": "Invalid parameters"}
        
        # Generate random returns
        np.random.seed(42)  # For reproducibility
        random_returns = np.random.normal(expected_return, volatility, (simulations, days))
        
        # Calculate price paths
        price_paths = np.zeros((simulations, days + 1))
        price_paths[:, 0] = initial_price
        
        for i in range(1, days + 1):
            price_paths[:, i] = price_paths[:, i - 1] * np.exp(random_returns[:, i - 1])
        
        # Calculate statistics
        final_prices = price_paths[:, -1]
        
        mean_price = np.mean(final_prices)
        median_price = np.median(final_prices)
        std_price = np.std(final_prices)
        
        # Confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(final_prices, lower_percentile)
        upper_bound = np.percentile(final_prices, upper_percentile)
        
        # Probability of profit
        prob_profit = np.sum(final_prices > initial_price) / simulations
        
        # Expected return
        expected_final_price = initial_price * np.exp(expected_return * days)
        
        return {
            'initial_price': float(initial_price),
            'mean_final_price': float(mean_price),
            'median_final_price': float(median_price),
            'std_final_price': float(std_price),
            'expected_final_price': float(expected_final_price),
            'confidence_interval': {
                'lower': float(lower_bound),
                'upper': float(upper_bound),
                'level': confidence_level
            },
            'probability_profit': float(prob_profit),
            'simulations': simulations,
            'days': days,
            'price_paths_sample': price_paths[:100].tolist()  # Sample for visualization
        }
    
    def calculate_beta(self, stock_returns: pd.Series,
                     market_returns: pd.Series) -> Dict:
        """
        Calculate beta (market sensitivity)
        
        Args:
            stock_returns: Stock return series
            market_returns: Market return series (e.g., SPY)
            
        Returns:
            Dict with beta and related metrics
        """
        if len(stock_returns) == 0 or len(market_returns) == 0:
            return {"error": "Insufficient data"}
        
        # Align series
        aligned = pd.concat([stock_returns, market_returns], axis=1).dropna()
        if len(aligned) < 2:
            return {"error": "Insufficient aligned data"}
        
        stock = aligned.iloc[:, 0]
        market = aligned.iloc[:, 1]
        
        # Calculate beta
        covariance = np.cov(stock, market)[0, 1]
        market_variance = np.var(market)
        
        if market_variance == 0:
            return {"error": "Market variance is zero"}
        
        beta = covariance / market_variance
        
        # Calculate alpha (risk-adjusted return)
        stock_mean = stock.mean()
        market_mean = market.mean()
        alpha = stock_mean - (beta * market_mean)
        
        # R-squared (with NaN protection)
        try:
            # Check for constant series which causes NaN in corrcoef
            if stock.std() == 0 or market.std() == 0:
                correlation = 0.0
                r_squared = 0.0
            else:
                correlation = np.corrcoef(stock, market)[0, 1]
                if np.isnan(correlation):
                    correlation = 0.0
                r_squared = correlation ** 2
        except Exception:
            correlation = 0.0
            r_squared = 0.0
        
        return {
            'beta': float(beta),
            'alpha': float(alpha),
            'r_squared': float(r_squared),
            'correlation': float(correlation),
            'interpretation': self._interpret_beta(beta)
        }
    
    def _interpret_beta(self, beta: float) -> str:
        """Interpret beta value"""
        if beta < 0.5:
            return "Low volatility relative to market"
        elif beta < 1.0:
            return "Less volatile than market"
        elif beta == 1.0:
            return "Moves with market"
        elif beta < 1.5:
            return "More volatile than market"
        else:
            return "High volatility relative to market"
    
    def calculate_drawdown_analysis(self, prices: pd.Series) -> Dict:
        """
        Comprehensive drawdown analysis
        
        Args:
            prices: Price series
            
        Returns:
            Dict with drawdown metrics
        """
        if len(prices) == 0:
            return {"error": "No price data"}
        
        # Calculate running maximum
        running_max = prices.expanding().max()
        
        # Calculate drawdown
        drawdown = (prices - running_max) / running_max
        
        # Maximum drawdown
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        # Average drawdown
        avg_dd = drawdown[drawdown < 0].mean() if len(drawdown[drawdown < 0]) > 0 else 0
        
        # Drawdown duration
        drawdown_periods = []
        in_drawdown = False
        start_idx = None
        
        for i, dd in enumerate(drawdown):
            if dd < -0.01 and not in_drawdown:  # Enter drawdown (>1%)
                in_drawdown = True
                start_idx = i
            elif dd >= -0.01 and in_drawdown:  # Exit drawdown
                in_drawdown = False
                if start_idx is not None:
                    drawdown_periods.append(i - start_idx)
                start_idx = None
        
        if in_drawdown and start_idx is not None:
            drawdown_periods.append(len(drawdown) - start_idx)
        
        avg_duration = np.mean(drawdown_periods) if drawdown_periods else 0
        
        return {
            'max_drawdown': float(max_dd),
            'max_drawdown_percentage': float(max_dd * 100),
            'max_drawdown_date': str(max_dd_idx),
            'average_drawdown': float(avg_dd),
            'average_drawdown_percentage': float(avg_dd * 100),
            'average_duration_days': float(avg_duration),
            'drawdown_count': len(drawdown_periods),
            'current_drawdown': float(drawdown.iloc[-1]) if len(drawdown) > 0 else 0
        }

