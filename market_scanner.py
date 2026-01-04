"""
Market Scanner module for Stocker App
Scans the market and recommends stocks based on strategy
"""
import logging
from typing import List, Dict, Optional
from data_fetcher import StockDataFetcher
from trading_analyzer import TradingAnalyzer
from investing_analyzer import InvestingAnalyzer
from mixed_analyzer import MixedAnalyzer

logger = logging.getLogger(__name__)


class MarketScanner:
    """Scans market and recommends stocks based on strategy"""
    
    # Popular stocks to scan (can be expanded)
    POPULAR_STOCKS = [
        # Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
        # Finance
        'JPM', 'BAC', 'WFC', 'GS', 'MS',
        # Consumer
        'KO', 'PEP', 'WMT', 'TGT', 'HD', 'MCD',
        # Healthcare
        'JNJ', 'PFE', 'UNH', 'ABT',
        # Industrial
        'BA', 'CAT', 'GE', 'MMM',
        # Energy
        'XOM', 'CVX', 'COP',
        # Other
        'DIS', 'V', 'MA', 'PG', 'NKE'
    ]
    
    def __init__(self, data_fetcher: StockDataFetcher):
        self.data_fetcher = data_fetcher
        self.trading_analyzer = TradingAnalyzer()
        self.investing_analyzer = InvestingAnalyzer()
        self.mixed_analyzer = MixedAnalyzer()
    
    def scan_market(self, strategy: str, max_results: int = 10) -> List[Dict]:
        """
        Scan market and return recommended stocks
        
        Args:
            strategy: 'trading', 'investing', or 'mixed'
            max_results: Maximum number of recommendations to return
        
        Returns:
            List of recommended stocks with analysis
        """
        recommendations = []
        
        logger.info(f"Scanning market with {strategy} strategy...")
        
        for symbol in self.POPULAR_STOCKS:
            try:
                # Fetch data
                stock_data = self.data_fetcher.fetch_stock_data(symbol)
                if not stock_data or 'error' in stock_data:
                    continue
                
                history_data = self.data_fetcher.fetch_stock_history(symbol)
                if not history_data or 'error' in history_data:
                    continue
                
                # Analyze based on strategy
                if strategy == 'trading':
                    analysis = self.trading_analyzer.analyze(stock_data, history_data)
                elif strategy == 'mixed':
                    financials_data = self.data_fetcher.fetch_financials(symbol)
                    analysis = self.mixed_analyzer.analyze(stock_data, financials_data, history_data)
                else:  # investing
                    financials_data = self.data_fetcher.fetch_financials(symbol)
                    analysis = self.investing_analyzer.analyze(stock_data, financials_data, history_data)
                
                if 'error' in analysis:
                    continue
                
                recommendation = analysis.get('recommendation', {})
                action = recommendation.get('action', 'HOLD')
                confidence = recommendation.get('confidence', 0)
                
                # Only include BUY recommendations with decent confidence
                if action == 'BUY' and confidence >= 50:
                    recommendations.append({
                        'symbol': symbol,
                        'name': stock_data.get('name', symbol),
                        'price': stock_data.get('price', 0),
                        'action': action,
                        'confidence': confidence,
                        'entry_price': recommendation.get('entry_price', 0),
                        'target_price': recommendation.get('target_price', 0),
                        'stop_loss': recommendation.get('stop_loss', 0),
                        'reasoning': analysis.get('reasoning', '')[:200]  # Truncate
                    })
                
                # Limit to prevent too many API calls
                if len(recommendations) >= max_results:
                    break
                    
            except Exception as e:
                logger.warning(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by confidence (highest first)
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        logger.info(f"Found {len(recommendations)} recommendations")
        return recommendations

