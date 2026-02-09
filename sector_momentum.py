"""
Sector Momentum Analyzer
Tracks sector ETF performance and compares individual stocks to their sector
"""
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import pandas as pd

logger = logging.getLogger(__name__)


class SectorMomentumAnalyzer:
    """Analyzes sector momentum and stock relative strength within sector"""
    
    # Sector ETF mapping (SPDR Select Sector ETFs)
    SECTOR_ETF_MAP = {
        'Technology': 'XLK',
        'Financial Services': 'XLF',
        'Financials': 'XLF',
        'Healthcare': 'XLV',
        'Health Care': 'XLV',
        'Consumer Cyclical': 'XLY',
        'Consumer Discretionary': 'XLY',
        'Consumer Defensive': 'XLP',
        'Consumer Staples': 'XLP',
        'Energy': 'XLE',
        'Industrials': 'XLI',
        'Basic Materials': 'XLB',
        'Materials': 'XLB',
        'Real Estate': 'XLRE',
        'Utilities': 'XLU',
        'Communication Services': 'XLC',
        'Telecommunications': 'XLC',
    }
    
    # Cache for sector ETF data to minimize API calls
    _sector_cache = {}
    _cache_expiry = {}
    CACHE_DURATION_HOURS = 1
    
    def __init__(self, data_fetcher=None):
        """
        Initialize with optional data fetcher for API calls
        
        Args:
            data_fetcher: StockDataFetcher instance for fetching ETF data
        """
        self.data_fetcher = data_fetcher
    
    def get_stock_sector(self, stock_info: dict) -> str:
        """
        Extract sector from stock info
        
        Args:
            stock_info: Stock info dict from yfinance
            
        Returns:
            Sector name or 'Unknown'
        """
        return stock_info.get('sector', stock_info.get('info', {}).get('sector', 'Unknown'))
    
    def get_sector_etf(self, sector: str) -> Optional[str]:
        """
        Get the corresponding sector ETF for a given sector
        
        Args:
            sector: Sector name
            
        Returns:
            ETF symbol or None if not found
        """
        return self.SECTOR_ETF_MAP.get(sector, None)
    
    def _is_cache_valid(self, etf_symbol: str) -> bool:
        """Check if cached data is still valid"""
        if etf_symbol not in self._cache_expiry:
            return False
        return datetime.now() < self._cache_expiry[etf_symbol]
    
    def _fetch_sector_etf_data(self, etf_symbol: str, as_of_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Fetch sector ETF historical data (date-aware for backtesting)
        """
        if as_of_date is None:
            as_of_date = datetime.now()
        
        # Use as_of_date for deterministic cache expiry check
        # This prevents live "now" from affecting historical state
        reference_date = as_of_date
            
        # Cache key includes full date to allow daily resolution in backtests (Rule #21)
        cache_key = f"{etf_symbol}_{as_of_date.strftime('%Y-%m-%d')}"
        
        # Check cache
        if cache_key in self._sector_cache and self._is_cache_valid(etf_symbol):
            return self._sector_cache[cache_key]
        
        # Determine fetch window (need enough for 20-day momentum)
        start_date = as_of_date - timedelta(days=90)
        end_date = as_of_date
        
        if not self.data_fetcher:
            try:
                import yfinance as yf
                ticker = yf.Ticker(etf_symbol)
                # Use start/end for backtest accuracy
                hist = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                                     end=(end_date + timedelta(days=1)).strftime('%Y-%m-%d'))
                if hist is not None and not hist.empty:
                    hist.columns = [c.lower() for c in hist.columns]
                    self._sector_cache[cache_key] = hist
                    self._cache_expiry[etf_symbol] = datetime.now() + timedelta(hours=self.CACHE_DURATION_HOURS)
                    return hist
            except Exception as e:
                logger.warning(f"Error fetching ETF {etf_symbol} directly for {as_of_date}: {e}")
                return None
        else:
            try:
                # Use data_fetcher for consistency and rate-limiting
                history = self.data_fetcher.fetch_stock_history(
                    etf_symbol, 
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                if history and 'data' in history and history['data']:
                    df = pd.DataFrame(history['data'])
                    df['date'] = pd.to_datetime(df['date'])
                    df.set_index('date', inplace=True)
                    self._sector_cache[cache_key] = df
                    self._cache_expiry[etf_symbol] = datetime.now() + timedelta(hours=self.CACHE_DURATION_HOURS)
                    return df
            except Exception as e:
                logger.warning(f"Error fetching ETF {etf_symbol} via fetcher for {as_of_date}: {e}")
                return None
        
        return None
    
    def _calculate_momentum(self, df: pd.DataFrame, periods: int = 20) -> Dict:
        """
        Calculate momentum metrics for price data
        
        Args:
            df: DataFrame with 'close' column
            periods: Number of periods for momentum calculation
            
        Returns:
            Dict with momentum metrics
        """
        if df is None or len(df) < periods:
            return {'momentum': 0, 'trend': 'unknown', 'change_pct': 0}
        
        close = df['close'] if 'close' in df.columns else df['Close']
        
        # Calculate percentage change over period
        if len(close) >= periods:
            change_pct = ((close.iloc[-1] / close.iloc[-periods]) - 1) * 100
        else:
            change_pct = ((close.iloc[-1] / close.iloc[0]) - 1) * 100
        
        # Calculate short-term momentum (5 days)
        short_change = ((close.iloc[-1] / close.iloc[-5]) - 1) * 100 if len(close) >= 5 else 0
        
        # Determine trend
        if change_pct > 5:
            trend = 'strong_uptrend'
        elif change_pct > 2:
            trend = 'uptrend'
        elif change_pct > -2:
            trend = 'sideways'
        elif change_pct > -5:
            trend = 'downtrend'
        else:
            trend = 'strong_downtrend'
        
        return {
            'momentum': float(change_pct),
            'short_momentum': float(short_change),
            'trend': trend,
            'change_pct': float(change_pct)
        }
    
    def analyze(self, stock_data: dict, stock_df: pd.DataFrame, as_of_date: Optional[datetime] = None) -> Dict:
        """
        Analyze sector momentum and stock's relative performance
        
        Args:
            stock_data: Stock data dict with 'info' containing sector
            stock_df: Stock price DataFrame
            as_of_date: Optional timestamp for backtesting
            
        Returns:
            Dict with sector analysis results
        """
        if as_of_date is None and not stock_df.empty:
            as_of_date = stock_df.index[-1]
        result = {
            'sector': 'Unknown',
            'etf_symbol': None,
            'sector_trend': 'unknown',
            'sector_momentum': 0,
            'stock_momentum': 0,
            'stock_vs_sector': 0,
            'outperforming_sector': False,
            'sector_signal': 'neutral',
            'available': False
        }
        
        try:
            # Get sector from stock data
            info = stock_data.get('info', stock_data)
            sector = self.get_stock_sector(info)
            result['sector'] = sector
            
            if sector == 'Unknown':
                logger.debug("Could not determine stock sector")
                return result
            
            # Get corresponding ETF
            etf_symbol = self.get_sector_etf(sector)
            if not etf_symbol:
                logger.debug(f"No ETF mapping for sector: {sector}")
                return result
            
            result['etf_symbol'] = etf_symbol
            
            # Fetch sector ETF data (passing as_of_date)
            etf_df = self._fetch_sector_etf_data(etf_symbol, as_of_date)
            if etf_df is None or etf_df.empty:
                logger.debug(f"Could not fetch ETF data for {etf_symbol}")
                return result
            
            # Calculate sector momentum
            sector_momentum = self._calculate_momentum(etf_df)
            result['sector_trend'] = sector_momentum['trend']
            result['sector_momentum'] = sector_momentum['momentum']
            
            # Calculate stock momentum
            stock_momentum = self._calculate_momentum(stock_df)
            result['stock_momentum'] = stock_momentum['momentum']
            
            # Calculate relative performance (stock vs sector)
            result['stock_vs_sector'] = result['stock_momentum'] - result['sector_momentum']
            result['outperforming_sector'] = result['stock_vs_sector'] > 0
            
            # Generate signal
            if result['stock_vs_sector'] > 5:
                result['sector_signal'] = 'strong_outperform'
            elif result['stock_vs_sector'] > 2:
                result['sector_signal'] = 'outperform'
            elif result['stock_vs_sector'] > -2:
                result['sector_signal'] = 'inline'
            elif result['stock_vs_sector'] > -5:
                result['sector_signal'] = 'underperform'
            else:
                result['sector_signal'] = 'strong_underperform'
            
            result['available'] = True
            logger.debug(f"Sector analysis complete: {sector} ({etf_symbol}) - {result['sector_signal']}")
            
        except Exception as e:
            logger.warning(f"Error in sector analysis: {e}")
        
        return result
    
    def get_sector_context_for_recommendation(self, sector_analysis: Dict) -> Dict:
        """
        Generate scoring adjustments and reasoning based on sector analysis
        
        Args:
            sector_analysis: Result from analyze()
            
        Returns:
            Dict with score_adjustment and reasoning
        """
        if not sector_analysis.get('available', False):
            return {'score_adjustment': 0, 'reasoning': []}
        
        score_adj = 0
        reasoning = []
        
        sector = sector_analysis['sector']
        etf = sector_analysis['etf_symbol']
        sector_trend = sector_analysis['sector_trend']
        stock_vs_sector = sector_analysis['stock_vs_sector']
        
        # Sector trend scoring
        if sector_trend == 'strong_uptrend':
            score_adj += 1
            reasoning.append(f"📈 {sector} sector ({etf}) in strong uptrend")
        elif sector_trend == 'strong_downtrend':
            score_adj -= 1
            reasoning.append(f"📉 {sector} sector ({etf}) in strong downtrend")
        
        # Stock vs Sector scoring
        if stock_vs_sector > 5:
            score_adj += 2
            reasoning.append(f"💪 Outperforming sector by {stock_vs_sector:.1f}%")
        elif stock_vs_sector > 2:
            score_adj += 1
            reasoning.append(f"✅ Outperforming sector by {stock_vs_sector:.1f}%")
        elif stock_vs_sector < -5:
            score_adj -= 2
            reasoning.append(f"⚠️ Underperforming sector by {abs(stock_vs_sector):.1f}%")
        elif stock_vs_sector < -2:
            score_adj -= 1
            reasoning.append(f"📊 Underperforming sector by {abs(stock_vs_sector):.1f}%")
        
        return {
            'score_adjustment': score_adj,
            'reasoning': reasoning
        }
