"""
Market Scanner module for Stocker App
Scans the market and recommends stocks based on strategy
Uses hybrid predictor (trained ML + learned weights) for better recommendations
"""
import logging
from typing import List, Dict, Optional
from pathlib import Path
from data_fetcher import StockDataFetcher
from trading_analyzer import TradingAnalyzer
from investing_analyzer import InvestingAnalyzer
from mixed_analyzer import MixedAnalyzer
from hybrid_predictor import HybridStockPredictor

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
    
    def __init__(self, data_fetcher: StockDataFetcher, data_dir: Path = None, 
                 trading_analyzer: TradingAnalyzer = None,
                 investing_analyzer: InvestingAnalyzer = None,
                 mixed_analyzer: MixedAnalyzer = None):
        self.data_fetcher = data_fetcher
        self.data_dir = data_dir
        
        # Store base analyzers (needed for hybrid predictor)
        self.trading_analyzer = trading_analyzer or TradingAnalyzer(data_fetcher=self.data_fetcher)
        self.investing_analyzer = investing_analyzer or InvestingAnalyzer()
        self.mixed_analyzer = mixed_analyzer or MixedAnalyzer()
    
        # Initialize hybrid predictors (will use trained models if available)
        self.hybrid_predictors = {}
        if data_dir:
            try:
                self.hybrid_predictors = {
                    'trading': HybridStockPredictor(
                        data_dir, 'trading', 
                        trading_analyzer=self.trading_analyzer
                    ),
                    'mixed': HybridStockPredictor(
                        data_dir, 'mixed',
                        mixed_analyzer=self.mixed_analyzer
                    ),
                    'investing': HybridStockPredictor(
                        data_dir, 'investing',
                        investing_analyzer=self.investing_analyzer
                    )
                }
                logger.info("Market scanner initialized with hybrid predictors (trained AI enabled)")
            except Exception as e:
                logger.warning(f"Could not initialize hybrid predictors: {e}, using basic analyzers")
                self.hybrid_predictors = {}
    
    def scan_market(self, strategy: str, max_results: int = 10, 
                   predictions_tracker=None) -> List[Dict]:
        """
        Scan market and return recommended stocks
        
        Args:
            strategy: 'trading', 'investing', or 'mixed'
            max_results: Maximum number of recommendations to return
            predictions_tracker: Optional PredictionsTracker to skip stocks with active predictions
        
        Returns:
            List of recommended stocks with analysis
        """
        recommendations = []
        
        logger.info(f"Scanning market with {strategy} strategy...")
        
        for symbol in self.POPULAR_STOCKS:
            # Skip stocks that already have active predictions
            if predictions_tracker:
                symbol_upper = symbol.upper()
                has_active = any(
                    p.get('symbol', '').upper() == symbol_upper and 
                    p.get('status') == 'active'
                    for p in predictions_tracker.predictions
                )
                if has_active:
                    logger.debug(f"Skipping {symbol} - already has active prediction")
                    continue
            try:
                # Fetch data
                stock_data = self.data_fetcher.fetch_stock_data(symbol)
                if not stock_data or 'error' in stock_data:
                    continue
                
                history_data = self.data_fetcher.fetch_stock_history(symbol)
                if not history_data or 'error' in history_data:
                    continue
                
                # Use hybrid predictor if available (includes trained ML + learned weights)
                # Otherwise fall back to basic analyzers
                hybrid_predictor = self.hybrid_predictors.get(strategy)
                
                if hybrid_predictor:
                    # Use trained hybrid system (ML + learned weights)
                    try:
                        if strategy == 'trading':
                            analysis = hybrid_predictor.predict(stock_data, history_data, None)
                        elif strategy == 'mixed':
                            financials_data = self.data_fetcher.fetch_financials(symbol)
                            analysis = hybrid_predictor.predict(stock_data, history_data, financials_data)
                        else:  # investing
                            financials_data = self.data_fetcher.fetch_financials(symbol)
                            analysis = hybrid_predictor.predict(stock_data, history_data, financials_data)
                    except Exception as hybrid_error:
                        logger.warning(f"Hybrid predictor failed for {symbol}, using basic analyzer: {hybrid_error}")
                        # Fallback to basic analyzer
                        if strategy == 'trading':
                            analysis = self.trading_analyzer.analyze(stock_data, history_data)
                        elif strategy == 'mixed':
                            financials_data = self.data_fetcher.fetch_financials(symbol)
                            analysis = self.mixed_analyzer.analyze(stock_data, financials_data, history_data)
                        else:  # investing
                            financials_data = self.data_fetcher.fetch_financials(symbol)
                            analysis = self.investing_analyzer.analyze(stock_data, financials_data, history_data)
                else:
                    # Use basic analyzers (no training available yet)
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

