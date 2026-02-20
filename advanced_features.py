"""
Advanced Feature Engineering Module v2.0
Includes: FinBERT Sentiment, Fundamentals, Macro Indicators, Analyst Ratings
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime, timedelta
import yfinance as yf

logger = logging.getLogger(__name__)

# Try to import FinBERT for sentiment analysis
HAS_FINBERT = False
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch
    HAS_FINBERT = True
except ImportError:
    logger.warning("FinBERT not available. Install with: pip install transformers torch")


class AdvancedFeatureExtractor:
    """
    Advanced feature extractor with:
    - FinBERT news sentiment
    - Fundamental analysis (P/E, revenue growth, debt ratios)
    - Macro indicators (VIX, bond yields, DXY)
    - Analyst ratings
    - Sector rotation signals
    """
    
    def __init__(self, data_fetcher=None):
        self.data_fetcher = data_fetcher
        self._sector_cache = {}
        self._macro_cache = {}
        self._macro_cache_time = None
        self._finbert_model = None
        self._finbert_tokenizer = None
        self._sentiment_cache = {}
        
    def _load_finbert(self):
        """Lazy load FinBERT model"""
        if not HAS_FINBERT:
            return False
        if self._finbert_model is None:
            try:
                logger.info("Loading FinBERT sentiment model...")
                self._finbert_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self._finbert_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                self._finbert_model.eval()
                logger.info("FinBERT loaded successfully")
                return True
            except Exception as e:
                logger.error(f"Failed to load FinBERT: {e}")
                return False
        return True
    
    # =========================================================================
    # 1. SENTIMENT ANALYSIS FEATURES
    # =========================================================================
    
    def extract_finbert_sentiment(self, headlines: List[str]) -> Dict:
        """
        Analyze news headlines using FinBERT.
        Returns sentiment scores: positive, negative, neutral, compound.
        """
        features = {
            'finbert_positive': 0.0,
            'finbert_negative': 0.0,
            'finbert_neutral': 0.0,
            'finbert_compound': 0.0,
            'news_count': 0
        }
        
        if not headlines:
            return features
            
        if not self._load_finbert():
            # Fallback to simple keyword analysis
            return self._simple_sentiment(headlines)
        
        try:
            all_probs = []
            for headline in headlines[:10]:  # Limit to 10 headlines for speed
                inputs = self._finbert_tokenizer(
                    headline, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=512
                )
                with torch.no_grad():
                    outputs = self._finbert_model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
                    all_probs.append(probs)
            
            if all_probs:
                avg_probs = np.mean(all_probs, axis=0)
                # FinBERT: [positive, negative, neutral]
                features['finbert_positive'] = float(avg_probs[0])
                features['finbert_negative'] = float(avg_probs[1])
                features['finbert_neutral'] = float(avg_probs[2])
                # Compound: positive - negative (range -1 to +1)
                features['finbert_compound'] = float(avg_probs[0] - avg_probs[1])
                features['news_count'] = len(headlines)
                
        except Exception as e:
            logger.warning(f"FinBERT analysis failed: {e}")
            return self._simple_sentiment(headlines)
            
        return features
    
    def _simple_sentiment(self, headlines: List[str]) -> Dict:
        """Fallback simple keyword-based sentiment"""
        positive_kw = ['beat', 'surge', 'rally', 'growth', 'profit', 'gain', 'bullish', 
                       'upgrade', 'outperform', 'buy', 'strong', 'record', 'soar']
        negative_kw = ['miss', 'drop', 'fall', 'loss', 'decline', 'bearish', 'downgrade',
                       'underperform', 'sell', 'weak', 'cut', 'crash', 'plunge']
        
        pos_count = neg_count = 0
        for h in headlines:
            h_lower = h.lower()
            pos_count += sum(1 for kw in positive_kw if kw in h_lower)
            neg_count += sum(1 for kw in negative_kw if kw in h_lower)
        
        total = pos_count + neg_count
        if total > 0:
            compound = (pos_count - neg_count) / total
        else:
            compound = 0
            
        return {
            'finbert_positive': pos_count / max(1, len(headlines)),
            'finbert_negative': neg_count / max(1, len(headlines)),
            'finbert_neutral': 1 - (pos_count + neg_count) / max(1, len(headlines) * 2),
            'finbert_compound': compound,
            'news_count': len(headlines)
        }
    
    def extract_analyst_ratings(self, symbol: str) -> Dict:
        """
        Extract analyst ratings and upgrades/downgrades.
        """
        features = {
            'analyst_buy_pct': 0.5,
            'analyst_hold_pct': 0.3,
            'analyst_sell_pct': 0.2,
            'analyst_score': 0.0,  # -1 (all sell) to +1 (all buy)
            'recent_upgrades': 0,
            'recent_downgrades': 0
        }
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get recommendations
            recs = ticker.recommendations
            if recs is not None and len(recs) > 0:
                # Get recent recommendations (last 30 days)
                recent = recs.tail(30)
                
                if 'To Grade' in recent.columns:
                    grades = recent['To Grade'].str.lower()
                    buy_grades = ['buy', 'strong buy', 'outperform', 'overweight']
                    hold_grades = ['hold', 'neutral', 'equal-weight', 'market perform']
                    sell_grades = ['sell', 'underperform', 'underweight']
                    
                    buy_count = grades.apply(lambda x: any(g in str(x) for g in buy_grades)).sum()
                    hold_count = grades.apply(lambda x: any(g in str(x) for g in hold_grades)).sum()
                    sell_count = grades.apply(lambda x: any(g in str(x) for g in sell_grades)).sum()
                    
                    total = buy_count + hold_count + sell_count
                    if total > 0:
                        features['analyst_buy_pct'] = buy_count / total
                        features['analyst_hold_pct'] = hold_count / total
                        features['analyst_sell_pct'] = sell_count / total
                        features['analyst_score'] = (buy_count - sell_count) / total
                
                # Count upgrades/downgrades
                if 'Action' in recent.columns:
                    actions = recent['Action'].str.lower()
                    features['recent_upgrades'] = (actions == 'upgrade').sum()
                    features['recent_downgrades'] = (actions == 'downgrade').sum()
                    
        except Exception as e:
            logger.debug(f"Could not get analyst ratings for {symbol}: {e}")
            
        return features
    
    # =========================================================================
    # 2. FUNDAMENTAL FEATURES
    # =========================================================================
    
    def extract_fundamental_features(self, symbol: str, stock_info: dict = None) -> Dict:
        """
        Extract fundamental analysis features.
        """
        features = {
            'pe_ratio': 0.0,
            'pe_vs_sector': 0.0,  # Relative P/E
            'peg_ratio': 0.0,
            'revenue_growth': 0.0,
            'earnings_growth': 0.0,
            'debt_to_equity': 0.0,
            'current_ratio': 0.0,
            'profit_margin': 0.0,
            'roe': 0.0,
            'free_cash_flow_yield': 0.0
        }
        
        try:
            if stock_info is None:
                ticker = yf.Ticker(symbol)
                stock_info = ticker.info
            
            # P/E Ratio
            pe = stock_info.get('trailingPE') or stock_info.get('forwardPE', 0)
            features['pe_ratio'] = min(100, max(-100, pe)) if pe else 0
            
            # PEG Ratio (P/E relative to growth)
            peg = stock_info.get('pegRatio', 0)
            features['peg_ratio'] = min(10, peg) if peg else 0
            
            # Revenue Growth
            rev_growth = stock_info.get('revenueGrowth', 0)
            features['revenue_growth'] = rev_growth if rev_growth else 0
            
            # Earnings Growth
            earn_growth = stock_info.get('earningsGrowth', 0)
            features['earnings_growth'] = earn_growth if earn_growth else 0
            
            # Debt to Equity
            de = stock_info.get('debtToEquity', 0)
            features['debt_to_equity'] = min(5, de / 100) if de else 0  # Normalize
            
            # Current Ratio (liquidity)
            curr_ratio = stock_info.get('currentRatio', 0)
            features['current_ratio'] = min(5, curr_ratio) if curr_ratio else 0
            
            # Profit Margin
            margin = stock_info.get('profitMargins', 0)
            features['profit_margin'] = margin if margin else 0
            
            # Return on Equity
            roe = stock_info.get('returnOnEquity', 0)
            features['roe'] = roe if roe else 0
            
            # Free Cash Flow Yield
            fcf = stock_info.get('freeCashflow', 0)
            market_cap = stock_info.get('marketCap', 1)
            if fcf and market_cap:
                features['free_cash_flow_yield'] = fcf / market_cap
            
            # P/E vs Sector (simplified - compare to SPY average ~22)
            sector_pe = 22  # Market average
            if features['pe_ratio'] > 0:
                features['pe_vs_sector'] = (features['pe_ratio'] - sector_pe) / sector_pe
                
        except Exception as e:
            logger.debug(f"Could not get fundamentals for {symbol}: {e}")
            
        return features
    
    # =========================================================================
    # 3. MACRO INDICATORS
    # =========================================================================
    
    def extract_macro_features(self, as_of_date: Optional[datetime] = None, 
                             macro_history: Optional[pd.DataFrame] = None) -> Dict:
        """
        Extract macroeconomic indicators: VIX, Bond Yields, DXY.
        If as_of_date and macro_history are provided, it slices historically (Rule #6).
        Otherwise fetches most recent data (Production mode).
        """
        features = {
            'vix_level': 20.0,
            'vix_change_pct': 0.0,
            'vix_regime': 1,  # 0=low vol, 1=normal, 2=high vol
            'treasury_10y': 4.0,
            'treasury_change': 0.0,
            'yield_curve': 0.0,
            'dxy_level': 100.0,
            'dxy_change_pct': 0.0,
            'available': False
        }
        
        # 🟢 HISTORICAL MODE (Backtest/Training)
        if as_of_date and macro_history is not None:
            try:
                # Slice indices up to as_of_date
                slice = macro_history[macro_history.index <= as_of_date].tail(5)
                if len(slice) >= 2:
                    features['vix_level'] = float(slice['VIX'].iloc[-1])
                    features['vix_change_pct'] = float((slice['VIX'].iloc[-1] - slice['VIX'].iloc[-2]) / slice['VIX'].iloc[-2] * 100)
                    features['treasury_10y'] = float(slice['TNX'].iloc[-1])
                    features['treasury_change'] = float(slice['TNX'].iloc[-1] - slice['TNX'].iloc[-2])
                    features['dxy_level'] = float(slice['DXY'].iloc[-1])
                    features['dxy_change_pct'] = float((slice['DXY'].iloc[-1] - slice['DXY'].iloc[-2]) / slice['DXY'].iloc[-2] * 100)
                    features['available'] = True
                    
                    # Yield Curve (10Y - 3M proxy if available)
                    if 'IRX' in slice.columns:
                        features['yield_curve'] = features['treasury_10y'] - float(slice['IRX'].iloc[-1])
                    
                    # VIX Regime
                    if features['vix_level'] < 15: features['vix_regime'] = 0
                    elif features['vix_level'] > 25: features['vix_regime'] = 2
                    
                    return features
            except Exception as e:
                logger.debug(f"Historical macro extraction failed for {as_of_date}: {e}")

        if as_of_date and macro_history is None:
             # If backtesting but no history provided, do NOT fetch live data (Leakage)
             return features

        # 🔵 PRODUCTION MODE (Current Data)
        # Check cache (1 hour)
        now = datetime.now()
        if not as_of_date and self._macro_cache_time and (now - self._macro_cache_time).seconds < 3600:
            return self._macro_cache
        
        # Guard against leakage: If as_of_date is set, we should have returned above.
        # This explicit check prevents accidental fall-through
        if as_of_date:
            return features

        try:
            # VIX (Fear Index)
            vix = yf.Ticker("^VIX")
            vix_hist = vix.history(period="5d", auto_adjust=True)
            if not vix_hist.empty:
                features['vix_level'] = float(vix_hist['Close'].iloc[-1])
                if len(vix_hist) > 1:
                    features['vix_change_pct'] = float(
                        (vix_hist['Close'].iloc[-1] - vix_hist['Close'].iloc[-2]) / 
                        vix_hist['Close'].iloc[-2] * 100
                    )
                # VIX Regime
                if features['vix_level'] < 15:
                    features['vix_regime'] = 0  # Low volatility
                elif features['vix_level'] < 25:
                    features['vix_regime'] = 1  # Normal
                else:
                    features['vix_regime'] = 2  # High volatility
            
            # 10-Year Treasury Yield
            tny = yf.Ticker("^TNX")
            tny_hist = tny.history(period="5d")
            if not tny_hist.empty:
                features['treasury_10y'] = float(tny_hist['Close'].iloc[-1])
                if len(tny_hist) > 1:
                    features['treasury_change'] = float(
                        tny_hist['Close'].iloc[-1] - tny_hist['Close'].iloc[-2]
                    )
            
            # 2-Year Treasury for yield curve
            try:
                t2y = yf.Ticker("^IRX")  # 3-month as proxy
                t2y_hist = t2y.history(period="5d")
                if not t2y_hist.empty:
                    features['yield_curve'] = features['treasury_10y'] - float(t2y_hist['Close'].iloc[-1])
            except:
                pass
            
            # DXY (Dollar Strength)
            dxy = yf.Ticker("DX-Y.NYB")
            dxy_hist = dxy.history(period="5d")
            if not dxy_hist.empty:
                features['dxy_level'] = float(dxy_hist['Close'].iloc[-1])
                if len(dxy_hist) > 1:
                    features['dxy_change_pct'] = float(
                        (dxy_hist['Close'].iloc[-1] - dxy_hist['Close'].iloc[-2]) / 
                        dxy_hist['Close'].iloc[-2] * 100
                    )
                    
        except Exception as e:
            logger.warning(f"Could not get macro data: {e}")
        
        # Cache results
        self._macro_cache = features
        self._macro_cache_time = now
        
        return features
    
    # =========================================================================
    # 4. SECTOR ROTATION
    # =========================================================================
    
    def extract_sector_rotation_features(self, symbol: str, stock_info: dict = None, 
                                        as_of_date: Optional[datetime] = None) -> Dict:
        """
        Extract sector rotation and relative strength features (Date-aware).
        """
        features = {
            'sector_momentum_1w': 0.0,
            'sector_momentum_1m': 0.0,
            'sector_vs_spy': 0.0,
            'stock_vs_sector': 0.0
        }
        
        # Sector ETF mapping
        sector_etfs = {
            'Technology': 'XLK',
            'Financial Services': 'XLF',
            'Healthcare': 'XLV',
            'Communication Services': 'XLC',
            'Consumer Cyclical': 'XLY',
            'Consumer Discretionary': 'XLY',
            'Consumer Defensive': 'XLP',
            'Consumer Staples': 'XLP',
            'Energy': 'XLE',
            'Industrials': 'XLI',
            'Basic Materials': 'XLB',
            'Real Estate': 'XLRE',
            'Utilities': 'XLU'
        }
        
        try:
            # Get stock's sector
            if stock_info is None:
                ticker = yf.Ticker(symbol)
                stock_info = ticker.info
            
            sector = stock_info.get('sector', '')
            sector_etf = sector_etfs.get(sector, 'SPY')
            
            # Get sector & SPY ETF data (Point-in-Time)
            if as_of_date:
                # Backtest mode: Specific window
                start = as_of_date - timedelta(days=45)
                end = as_of_date
                
                etf = yf.Ticker(sector_etf)
                spy = yf.Ticker("SPY")
                
                etf_hist = etf.history(start=start.strftime('%Y-%m-%d'), end=(end + timedelta(days=1)).strftime('%Y-%m-%d'), auto_adjust=True)
                spy_hist = spy.history(start=start.strftime('%Y-%m-%d'), end=(end + timedelta(days=1)).strftime('%Y-%m-%d'), auto_adjust=True)
                
                # Also get stock history for this period
                stock_ticker = yf.Ticker(symbol)
                stock_hist = stock_ticker.history(start=start.strftime('%Y-%m-%d'), end=(end + timedelta(days=1)).strftime('%Y-%m-%d'), auto_adjust=True)
            else:
                # Production mode: Last month
                etf = yf.Ticker(sector_etf)
                spy = yf.Ticker("SPY")
                stock_ticker = yf.Ticker(symbol)
                
                etf_hist = etf.history(period="1mo", auto_adjust=True)
                spy_hist = spy.history(period="1mo", auto_adjust=True)
                stock_hist = stock_ticker.history(period="1mo", auto_adjust=True)
            
            if not etf_hist.empty and len(etf_hist) >= 10:
                # Sector momentum (1 week approx 5 trading days)
                features['sector_momentum_1w'] = float(
                    (etf_hist['Close'].iloc[-1] - etf_hist['Close'].iloc[-5]) / 
                    etf_hist['Close'].iloc[-5] * 100
                )
                
                # Sector momentum (1 month approx 20 trading days) - Use 21 as proxy for trading month
                m_start_idx = max(0, len(etf_hist) - 21)
                features['sector_momentum_1m'] = float(
                    (etf_hist['Close'].iloc[-1] - etf_hist['Close'].iloc[m_start_idx]) / 
                    etf_hist['Close'].iloc[m_start_idx] * 100
                )
            
            # Sector vs SPY
            if not spy_hist.empty and not etf_hist.empty and len(spy_hist) >= 10:
                m_start_idx = max(0, len(spy_hist) - 21)
                spy_return = (spy_hist['Close'].iloc[-1] - spy_hist['Close'].iloc[m_start_idx]) / spy_hist['Close'].iloc[m_start_idx]
                
                e_start_idx = max(0, len(etf_hist) - 21)
                sector_return = (etf_hist['Close'].iloc[-1] - etf_hist['Close'].iloc[e_start_idx]) / etf_hist['Close'].iloc[e_start_idx]
                
                features['sector_vs_spy'] = float((sector_return - spy_return) * 100)
                
            # Stock vs Sector
            if not stock_hist.empty and not etf_hist.empty and len(stock_hist) >= 10:
                s_start_idx = max(0, len(stock_hist) - 21)
                stock_return = (stock_hist['Close'].iloc[-1] - stock_hist['Close'].iloc[s_start_idx]) / stock_hist['Close'].iloc[s_start_idx]
                
                e_start_idx = max(0, len(etf_hist) - 21)
                sector_return = (etf_hist['Close'].iloc[-1] - etf_hist['Close'].iloc[e_start_idx]) / etf_hist['Close'].iloc[e_start_idx]
                
                features['stock_vs_sector'] = float((stock_return - sector_return) * 100)
                
        except Exception as e:
            logger.debug(f"Could not get sector rotation for {symbol}: {e}")
            
        return features
    
    # =========================================================================
    # MAIN EXTRACTION METHOD
    # =========================================================================
    
    def extract_all_features(self, symbol: str, stock_data: dict = None, 
                            news_headlines: List[str] = None,
                            as_of_date: Optional[datetime] = None,
                            macro_history: Optional[pd.DataFrame] = None) -> Dict:
        """
        Extract ALL advanced features for a symbol.
        
        Args:
            symbol: Stock ticker symbol
            stock_data: Optional stock info dict (will fetch if None)
            news_headlines: Optional list of news headlines
            as_of_date: Date for historical slicing
            macro_history: Pre-fetched macro dataframe
            
        Returns:
            Dict with all advanced features
        """
        all_features = {}
        
        try:
            # Get stock info if not provided (Avoid fetching info in training mode)
            if stock_data is None:
                if as_of_date: # Training/Backtest
                    # Use a skeleton dict to avoid yfinance.info (which is always "current")
                    stock_data = {'symbol': symbol, 'info': {}}
                else: # Production
                    ticker = yf.Ticker(symbol)
                    stock_data = ticker.info
            
            # 1. Sentiment (FinBERT and fallback) - Date Aware
            if news_headlines:
                # Use provided headlines (for training override)
                sentiment = self.extract_finbert_sentiment(news_headlines)
                all_features.update(sentiment)
            else:
                # Fetch from yfinance (Production or restricted Backtest)
                sentiment = self.sentiment_analyzer.analyze_sentiment(symbol, as_of_date)
                all_features.update({
                    'sentiment_score': sentiment['sentiment_score'],
                    'news_count': sentiment['news_count']
                })
            
            # 2. Analyst Ratings (PRODUCTION ONLY - ticker.recommendations is not point-in-time)
            if not as_of_date:
                analyst = self.extract_analyst_ratings(symbol)
                all_features.update(analyst)
            else:
                all_features.update({'analyst_buy_pct': 0.5, 'analyst_score': 0.0})
            
            # 3. Fundamentals (PRODUCTION ONLY - ticker.info is not point-in-time)
            if not as_of_date:
                fundamentals = self.extract_fundamental_features(symbol, stock_data)
                all_features.update(fundamentals)
            else:
                # INTEGRITY LOCK: Zero-out fundamentals in historical mode to prevent drift
                all_features.update({f: 0.0 for f in ['pe_ratio', 'revenue_growth', 'roe', 'peg_ratio']})
            
            # 4. Macro Indicators (SYNCED)
            macro = self.extract_macro_features(as_of_date, macro_history)
            all_features.update(macro)
            
            # 5. Sector Rotation (SYNCED)
            sector = self.extract_sector_rotation_features(symbol, stock_data, as_of_date)
            all_features.update(sector)
            
        except Exception as e:
            logger.error(f"Error extracting features for {symbol}: {e}")
        
        return all_features


# Convenience function for quick feature extraction
def get_advanced_features(symbol: str, headlines: List[str] = None) -> Dict:
    """Quick extraction of all advanced features"""
    extractor = AdvancedFeatureExtractor()
    return extractor.extract_all_features(symbol, news_headlines=headlines)
