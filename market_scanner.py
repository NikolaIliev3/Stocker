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
        'DIS', 'V', 'MA', 'PG', 'NKE',
        # Semiconductors
        'AMD', 'INTC', 'TXN',
        # Additional Diverse
        'SBUX', 'BLK'
    ]
    
    # Extended list for broad market dip scanning
    BROAD_MARKET_STOCKS = POPULAR_STOCKS + [
        # More Tech
        'CRM', 'ADBE', 'ORCL', 'IBM', 'CSCO', 'QCOM', 'AVGO',
        'NOW', 'SHOP', 'SQ', 'PYPL', 'UBER', 'ABNB', 'SNAP', 'PINS', 'ZM', 'DOCU',
        'PLTR', 'CRWD', 'DDOG', 'NET', 'MDB', 'SNOW', 'U', 'RBLX', 'COIN', 'HOOD',
        # More Finance
        'C', 'SCHW', 'AXP', 'USB', 'PNC', 'TFC', 'COF', 'CME', 'ICE',
        # More Consumer
        'COST', 'LOW', 'YUM', 'CMG', 'DG', 'DLTR', 'ROST', 'TJX', 'LULU',
        # More Healthcare
        'MRK', 'LLY', 'BMY', 'GILD', 'AMGN', 'BIIB', 'VRTX', 'REGN', 'MRNA', 'ZTS',
        # More Industrial
        'HON', 'UPS', 'FDX', 'RTX', 'LMT', 'NOC', 'GD', 'DE', 'EMR', 'ITW',
        # More Energy
        'SLB', 'EOG', 'PSX', 'VLO', 'MPC', 'OXY', 'HAL', 'BKR', 'DVN', 'FANG',
        # REITs & Utilities
        'AMT', 'PLD', 'CCI', 'EQIX', 'O', 'SPG', 'NEE', 'DUK', 'SO', 'D',
        # Materials
        'LIN', 'APD', 'SHW', 'ECL', 'NEM', 'FCX', 'NUE', 'CLF', 'X', 'AA',
        # ETFs (for broad market signals)
        'SPY', 'QQQ', 'IWM', 'DIA', 'XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB',
        # International ADRs
        'BABA', 'TSM', 'NIO', 'BIDU', 'JD', 'PDD', 'SE', 'GRAB', 'NU', 'MELI'
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

        # Initialize Topology Analyzer (Market Graph)
        try:
            from topology_analyzer import TopologyAnalyzer
            self.topology_analyzer = TopologyAnalyzer(data_fetcher)
            logger.info("Topology Analyzer initialized (Market Graph enabled)")
        except ImportError:
            self.topology_analyzer = None
            logger.warning("Topology Analyzer not available")
    
    def scan_market(self, strategy: str, max_results: int = 10, 
                   predictions_tracker=None, custom_tickers: List[str] = None,
                   progress_callback=None) -> List[Dict]:
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
        
        total_stocks = len(stocks_to_scan)
        
        # Start scanning loop
        for idx, symbol in enumerate(stocks_to_scan):
            if progress_callback:
                progress_callback(idx + 1, total_stocks, symbol, len(recommendations))

        # TOPOLOGY ANALYSIS (Global Context)
        # Pre-scan the market to build the graph and find mispricings
        laplacian_scores = {}
        if self.topology_analyzer and len(stocks_to_scan) > 5 and not is_custom_scan:
             try:
                 logger.info("Building Market Graph for topological analysis...")
                 # We need batch history. Fetching 3mo history for all stocks.
                 # This might be slow, so we do it in parallel
                 # Using data_fetcher.fetch_batch but asking for history via check_quality=True logic
                 # Or better: just assume we iterate and build as we go? No, need simultaneous data.
                 # Optimization: Only do this for "Popular Stocks" scan, not custom
                 
                 # Create a dataframe of returns
                 import pandas as pd
                 import numpy as np
                 
                 # We need to fetch history efficiently. 
                 # Let's skip the expensive pre-fetch for now and build it progressively or use a cached approach?
                 # Better: Use the `dataset_fetcher` style if available, or just fetch batch.
                 # For now, to ensure speed, we might skip this step if list is huge.
                 # But user wants "everything integrated".
                 
                 # Let's try to fetch batch history for the top 50 stocks to build a "Core Graph"
                 # Limit to first 50 stocks for graph building to save time
                 graph_stocks = stocks_to_scan[:50] 
                 batch_results = self.data_fetcher.fetch_batch(graph_stocks, max_workers=20, check_quality=True)
                 
                 # Extract standard returns
                 price_data = {}
                 for sym, data in batch_results.items():
                     if 'history' in data and data['history']:
                         # Create series
                         hist = data['history']
                         # Filter to last 60 days
                         df = pd.DataFrame(hist)
                         if 'date' in df.columns and 'close' in df.columns:
                             df['date'] = pd.to_datetime(df['date'])
                             df.set_index('date', inplace=True)
                             # Calculate returns
                             price_data[sym] = df['close'].pct_change()
                 
                 if price_data:
                     returns_df = pd.DataFrame(price_data)
                     # Build graph
                     if self.topology_analyzer.build_market_graph(returns_df):
                         # Compute residuals for latest timeframe
                         latest_returns = returns_df.iloc[-1]
                         laplacian_scores = self.topology_analyzer.compute_laplacian_score(latest_returns)
                         logger.info(f"✅ Market Graph Built. Calculated {len(laplacian_scores)} residual scores.")
             except Exception as e:
                 logger.warning(f"Topology analysis failed: {e}")

        for idx, symbol in enumerate(stocks_to_scan):
            # ... loop continues ...

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
                
                # INJECT TOPOLOGY SCORE
                # If we have a mispricing score for this stock, add it to data
                if symbol in laplacian_scores:
                    stock_data['laplacian_score'] = laplacian_scores[symbol]
                    # Log if significant
                    if abs(laplacian_scores[symbol]) > 0.02: # arbitrarily threshold
                         logger.info(f"Topology Signal for {symbol}: Residual {laplacian_scores[symbol]:.4f}")
                
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
                # --- TIERED UI SYNC: Only show signals that meet Standard or Elite Tiers ---
                from config import STANDARD_CONF_RANGE
                min_ui_conf = STANDARD_CONF_RANGE[0] # Usually 70.0
                
                if is_custom_scan or (action == 'BUY' and confidence >= min_ui_conf):
                    # Get timing metadata robustly
                    est_days = recommendation.get('estimated_days') or analysis.get('estimated_days') or 7
                    est_date = recommendation.get('estimated_target_date') or analysis.get('estimated_target_date')
                    
                    recommendations.append({
                        'symbol': symbol,
                        'name': stock_data.get('name', symbol),
                        'price': stock_data.get('price', 0),
                        'action': action,
                        'confidence': confidence,
                        'entry_price': recommendation.get('entry_price', 0),
                        'target_price': recommendation.get('target_price', 0),
                        'stop_loss': recommendation.get('stop_loss', 0),
                        'estimated_days': est_days,
                        'estimated_target_date': est_date,
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

    def scan_for_dips(self, max_results: int = 20, 
                      rsi_threshold: float = 35.0,
                      mfi_threshold: float = 25.0,
                      progress_callback=None) -> List[Dict]:
        """
        Scan broad market for oversold dip opportunities.
        
        Looks for stocks that meet the "aggressive dip buying" criteria:
        - RSI below threshold (oversold)
        - MFI below threshold (money flow oversold)
        - In a downtrend (price below EMAs)
        
        Args:
            max_results: Maximum number of dips to return
            rsi_threshold: RSI level to consider oversold (default 35)
            mfi_threshold: MFI level to consider oversold (default 25)
            progress_callback: Optional callback(current, total, symbol) for progress updates
        
        Returns:
            List of oversold stocks with analysis
        """
        import pandas as pd
        import numpy as np
        
        dip_opportunities = []
        stocks_to_scan = self.BROAD_MARKET_STOCKS
        total_stocks = len(stocks_to_scan)
        
        logger.info(f"🔻 Scanning {total_stocks} stocks for market dips (RSI < {rsi_threshold}, MFI < {mfi_threshold})...")
        
        for idx, symbol in enumerate(stocks_to_scan):
            try:
                if progress_callback:
                    progress_callback(idx + 1, total_stocks, symbol)
                
                # Log progress every 10 stocks
                if idx % 10 == 0:
                    logger.info(f"🔻 Scanning dips progress: {idx+1}/{total_stocks} ({symbol})")
                
                # Fetch historical data
                history_data = self.data_fetcher.fetch_stock_history(symbol, period='3mo')
                if not history_data or 'error' in history_data:
                    continue
                
                # FIX: fetch_stock_history returns {'data': [...]}, not {'prices': [...]}
                prices = history_data.get('data', [])
                if len(prices) < 20:
                    continue
                
                # Convert to DataFrame for easier calculation
                df = pd.DataFrame(prices)
                if 'close' not in df.columns:
                    continue
                
                close = df['close'].values
                high = df.get('high', df['close']).values
                low = df.get('low', df['close']).values
                volume = df.get('volume', pd.Series([1]*len(df))).values
                
                # Calculate RSI
                delta = np.diff(close)
                gains = np.where(delta > 0, delta, 0)
                losses = np.where(delta < 0, -delta, 0)
                
                avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else np.mean(gains)
                avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else np.mean(losses)
                
                if avg_loss == 0:
                    rsi = 100
                else:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                
                # Calculate MFI (simplified)
                typical_price = (high + low + close) / 3
                raw_money_flow = typical_price * volume
                
                positive_flow = np.sum(raw_money_flow[-14:][np.diff(np.append([typical_price[-15]], typical_price[-14:])) > 0])
                negative_flow = np.sum(raw_money_flow[-14:][np.diff(np.append([typical_price[-15]], typical_price[-14:])) <= 0])
                
                if negative_flow == 0:
                    mfi = 100
                else:
                    money_ratio = positive_flow / negative_flow
                    mfi = 100 - (100 / (1 + money_ratio))
                
                # Calculate trend (EMA20 vs EMA50)
                ema20 = pd.Series(close).ewm(span=20).mean().iloc[-1]
                ema50 = pd.Series(close).ewm(span=50).mean().iloc[-1] if len(close) >= 50 else ema20
                current_price = close[-1]
                
                is_downtrend = current_price < ema20 and ema20 < ema50
                is_oversold = rsi < rsi_threshold or mfi < mfi_threshold
                
                # Check for dip opportunity
                if is_oversold:
                    # Get stock info
                    stock_data = self.data_fetcher.fetch_stock_data(symbol)
                    if not stock_data or 'error' in stock_data:
                        stock_data = {'name': symbol, 'price': current_price}
                    
                    # Calculate dip severity score (lower RSI/MFI = deeper dip = higher score)
                    rsi_score = max(0, (rsi_threshold - rsi) / rsi_threshold) * 50
                    mfi_score = max(0, (mfi_threshold - mfi) / mfi_threshold) * 50
                    dip_score = rsi_score + mfi_score
                    
                    # Price distance from EMA20 (how far price has fallen)
                    distance_from_ema = ((ema20 - current_price) / ema20) * 100 if ema20 > 0 else 0
                    
                    # Calculate targets
                    from config import MIN_PROFIT_TARGET_PCT
                    entry_price = current_price
                    
                    # Target: Return to EMA20, but ensure at least MIN_PROFIT_TARGET_PCT
                    base_target = ema20
                    min_target = entry_price * (1 + MIN_PROFIT_TARGET_PCT/100)
                    
                    target_price = max(base_target, min_target)
                    stop_loss = current_price * 0.95  # 5% stop loss (wide for catching falling knives)
                    
                    potential_gain = ((target_price - entry_price) / entry_price) * 100
                    
                    dip_opportunities.append({
                        'symbol': symbol,
                        'name': stock_data.get('name', symbol),
                        'price': current_price,
                        'rsi': rsi,
                        'mfi': mfi,
                        'trend': 'Downtrend' if is_downtrend else 'Sideways/Mixed',
                        'dip_score': dip_score,
                        'distance_from_ema': distance_from_ema,
                        'entry_price': entry_price,
                        'target_price': target_price,
                        'stop_loss': stop_loss,
                        'potential_gain': potential_gain,
                        'action': 'BUY',
                        'confidence': min(90, 60 + dip_score / 2),  # Higher dip score = higher confidence
                        'reasoning': f"💎 OVERSOLD DIP: RSI={rsi:.1f}, MFI={mfi:.1f}. Price {distance_from_ema:.1f}% below EMA20. Target: ${target_price:.2f} (+{potential_gain:.1f}%)"
                    })
                    
                    # Limit results
                    if len(dip_opportunities) >= max_results:
                        break
                        
            except Exception as e:
                logger.debug(f"Error scanning {symbol} for dips: {e}")
                continue
        
        # Sort by dip score (deepest dips first)
        dip_opportunities.sort(key=lambda x: x.get('dip_score', 0), reverse=True)
        
        logger.info(f"🔻 Found {len(dip_opportunities)} dip opportunities")
        return dip_opportunities
