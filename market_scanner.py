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
                logger.info("Market scanner initialized with hybrid predictors (Rules + ML)")
            except Exception as e:
                logger.warning(f"Could not initialize hybrid predictors: {e}, using basic analyzers")
                self.hybrid_predictors = {}
    
    def scan_market(self, strategy: str, max_results: int = 10, 
                   predictions_tracker=None, custom_tickers: List[str] = None) -> List[Dict]:
        """
        Scan market and return recommended stocks
        
        Args:
            strategy: 'trading', 'investing', or 'mixed'
            max_results: Maximum number of recommendations to return
            predictions_tracker: Optional PredictionsTracker to skip stocks with active predictions
                                (IGNORED IF custom_tickers IS PROVIDED)
            custom_tickers: Optional list of tickers to scan instead of POPULAR_STOCKS
        
        Returns:
            List of recommended stocks with analysis
        """
        recommendations = []
        
        # Use custom tickers if provided, otherwise default list
        stocks_to_scan = custom_tickers if custom_tickers else self.POPULAR_STOCKS
        is_custom_scan = custom_tickers is not None
        
        logger.info(f"Scanning market with {strategy} strategy (Custom Scan: {is_custom_scan}, Stocks: {len(stocks_to_scan)})...")
        
        for symbol in stocks_to_scan:
            # Skip stocks that already have active predictions ONLY if this is a general market scan
            # If we are scanning specific tickers (e.g. existing predictions), we want to re-analyze them
            if not is_custom_scan and predictions_tracker:
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
                
                # Only include recommendations with decent confidence
                # STANDARD LOGIC: BUY is the bullish/entry signal
                # For custom scans (existing predictions), include EVERYTHING (even HOLD)
                if is_custom_scan or (action == 'BUY' and confidence >= 50):
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
                
                # Limit to prevent too many API calls (unless custom scan)
                if not is_custom_scan and len(recommendations) >= max_results:
                    break
                    
            except Exception as e:
                logger.warning(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by Action (BUY first) then Confidence
        # Action priority: BUY (2), HOLD (1), SELL (0) - assuming standard logic
        def get_sort_key(rec):
            action = rec.get('action', 'HOLD')
            action_score = 2 if action == 'BUY' else 1 if action == 'HOLD' else 0
            return (action_score, rec.get('confidence', 0))
            
        recommendations.sort(key=get_sort_key, reverse=True)
        
        logger.info(f"Found {len(recommendations)} recommendations")
        return recommendations

