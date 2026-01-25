"""
Trading Strategy Analyzer
Focuses on technical analysis, price action, momentum, and short-term indicators
"""
import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
try:
    from ta.volume import MFIIndicator
    HAS_MFI = True
except ImportError:
    HAS_MFI = False
    logger.warning("MFIIndicator not available in ta library, will calculate manually")
import logging

from algorithm_improvements import MarketRegimeDetector, VolumeWeightedLevels, MultiTimeframeAnalyzer, RelativeStrengthAnalyzer
from pattern_recognition import PatternRecognizer
from sector_momentum import SectorMomentumAnalyzer
from catalyst_detector import CatalystDetector
from sentiment_analyzer import SentimentAnalyzer
from macro_regime import MacroRegimeDetector

logger = logging.getLogger(__name__)


class TradingAnalyzer:
    """Analyzes stocks using trading/technical analysis approach"""
    
    def __init__(self, data_fetcher=None, calibration_manager=None):
        self.calibration_manager = calibration_manager
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.regime_detector = MarketRegimeDetector()
        self.volume_levels_calculator = VolumeWeightedLevels()
        self.timeframe_analyzer = MultiTimeframeAnalyzer()
        self.relative_strength_analyzer = RelativeStrengthAnalyzer()
        self.pattern_recognizer = PatternRecognizer()
        self.pattern_recognizer = PatternRecognizer()
        self.sector_analyzer = SectorMomentumAnalyzer(data_fetcher)
        self.catalyst_detector = CatalystDetector(data_fetcher)
        self.sentiment_analyzer = SentimentAnalyzer(data_fetcher)
        self.macro_detector = MacroRegimeDetector(data_fetcher)
        self.data_fetcher = data_fetcher  # For fetching SPY data
        self._market_regime_cache = None
        self._market_regime_cache_time = None
        self._spy_data_cache = None
        self._spy_data_cache_time = None
    
    def analyze(self, stock_data: dict, history_data: dict) -> dict:
        """
        Perform trading analysis on stock data
        Returns buy/sell recommendation with reasoning
        """
        try:
            # Convert history to DataFrame
            df = pd.DataFrame(history_data.get('data', []))
            if df.empty:
                return {"error": "No historical data available"}
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.sort_index()
            
            # Get symbol for catalyst detection
            symbol = stock_data.get('symbol', stock_data.get('info', {}).get('symbol', ''))
            
            # Detect market regime (using SPY as market proxy)
            market_regime = self._get_market_regime()
            
            # Multi-timeframe analysis (daily, weekly, monthly trends)
            timeframe_analysis = self.timeframe_analyzer.analyze_timeframes(df)
            
            # Relative strength analysis (stock vs market)
            relative_strength = self._calculate_relative_strength(df)
            
            # Sector momentum analysis (stock vs sector ETF)
            sector_analysis = self.sector_analyzer.analyze(stock_data, df)
            
            # Catalyst detection (earnings, dividends)
            catalyst_info = self.catalyst_detector.detect_catalysts(symbol, stock_data)
            
            # Sentiment Analysis (News)
            sentiment_info = self.sentiment_analyzer.analyze_sentiment(symbol)
            
            # Macro Regime (VIX, TNX, DXY) - mostly cached
            macro_info = self.macro_detector.analyze_regime()
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(df)
            
            # Analyze price action
            price_action = self._analyze_price_action(df)
            
            # Analyze volume with enhanced liquidity
            volume_analysis = self._analyze_volume(df)
            
            # Analyze momentum
            momentum_analysis = self._analyze_momentum(indicators)
            
            # Identify support and resistance
            support_resistance = self._identify_levels(df)
            
            # Determine recommendation (with all analysis inputs)
            recommendation = self._make_recommendation(
                indicators, price_action, volume_analysis, 
                momentum_analysis, support_resistance, df, market_regime, 
                timeframe_analysis, relative_strength, sector_analysis, 
                catalyst_info, sentiment_info, macro_info
            )
            
            return {
                "strategy": "trading",
                "recommendation": recommendation,
                "indicators": indicators,
                "price_action": price_action,
                "volume_analysis": volume_analysis,
                "momentum_analysis": momentum_analysis,
                "support_resistance": support_resistance,
                "market_regime": market_regime,
                "timeframe_analysis": timeframe_analysis,
                "relative_strength": relative_strength,
                "sector_analysis": sector_analysis,
                "catalyst_info": catalyst_info,
                "sentiment_info": sentiment_info,
                "macro_info": macro_info,
                "reasoning": self._generate_reasoning(
                    recommendation, indicators, price_action, 
                    volume_analysis, momentum_analysis, support_resistance, 
                    market_regime, timeframe_analysis, relative_strength,
                    sector_analysis, catalyst_info, sentiment_info, macro_info
                )
            }
        
        except Exception as e:
            logger.error(f"Error in trading analysis: {e}")
            return {"error": str(e)}
    
    def _calculate_indicators(self, df: pd.DataFrame) -> dict:
        """Calculate technical indicators"""
        indicators = {}
        
        try:
            # RSI
            rsi_indicator = RSIIndicator(close=df['close'], window=14)
            rsi_series = rsi_indicator.rsi()
            indicators['rsi'] = float(rsi_series.iloc[-1])
            indicators['rsi_prev'] = float(rsi_series.iloc[-2]) if len(rsi_series) > 1 else 50.0
            
            # MACD
            macd_indicator = MACD(close=df['close'])
            indicators['macd'] = float(macd_indicator.macd().iloc[-1])
            indicators['macd_signal'] = float(macd_indicator.macd_signal().iloc[-1])
            indicators['macd_diff'] = float(macd_indicator.macd_diff().iloc[-1])
            
            # Moving Averages
            ema_20 = EMAIndicator(close=df['close'], window=20)
            ema_50 = EMAIndicator(close=df['close'], window=50)
            ema_200 = EMAIndicator(close=df['close'], window=200)
            
            indicators['ema_20'] = float(ema_20.ema_indicator().iloc[-1])
            indicators['ema_50'] = float(ema_50.ema_indicator().iloc[-1])
            indicators['ema_200'] = float(ema_200.ema_indicator().iloc[-1])
            
            # Bollinger Bands
            bb = BollingerBands(close=df['close'], window=20, window_dev=2)
            indicators['bb_upper'] = float(bb.bollinger_hband().iloc[-1])
            indicators['bb_lower'] = float(bb.bollinger_lband().iloc[-1])
            indicators['bb_middle'] = float(bb.bollinger_mavg().iloc[-1])
            
            # ATR (Average True Range) for volatility
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            indicators['atr'] = float(true_range.rolling(14).mean().iloc[-1])
            indicators['atr_percent'] = float((indicators['atr'] / df['close'].iloc[-1]) * 100)
            
            # Stochastic Oscillator - Momentum indicator
            try:
                stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
                indicators['stoch_k'] = float(stoch.stoch().iloc[-1])
                indicators['stoch_d'] = float(stoch.stoch_signal().iloc[-1])
            except:
                indicators['stoch_k'] = 50
                indicators['stoch_d'] = 50
            
            # Williams %R - Momentum indicator
            try:
                williams = WilliamsRIndicator(high=df['high'], low=df['low'], close=df['close'])
                indicators['williams_r'] = float(williams.williams_r().iloc[-1])
            except:
                indicators['williams_r'] = -50
            
            # ADX (Average Directional Index) - Trend strength
            try:
                adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'])
                indicators['adx'] = float(adx.adx().iloc[-1])
                indicators['adx_pos'] = float(adx.adx_pos().iloc[-1])
                indicators['adx_neg'] = float(adx.adx_neg().iloc[-1])
            except:
                indicators['adx'] = 25
                indicators['adx_pos'] = 0
                indicators['adx_neg'] = 0
            
            # On-Balance Volume (OBV) - Volume momentum
            try:
                obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
                obv_values = obv.on_balance_volume()
                indicators['obv'] = float(obv_values.iloc[-1])
                # OBV trend (increasing/decreasing)
                if len(obv_values) >= 10:
                    obv_trend = obv_values.iloc[-1] - obv_values.iloc[-10]
                    indicators['obv_trend'] = 'bullish' if obv_trend > 0 else 'bearish'
                else:
                    indicators['obv_trend'] = 'neutral'
            except:
                indicators['obv'] = 0
                indicators['obv_trend'] = 'neutral'
            
            # Money Flow Index (MFI) - Combines price and volume to identify overbought/oversold
            try:
                if HAS_MFI:
                    mfi_indicator = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
                    indicators['mfi'] = float(mfi_indicator.money_flow_index().iloc[-1])
                else:
                    # Calculate MFI manually
                    typical_price = (df['high'] + df['low'] + df['close']) / 3
                    raw_money_flow = typical_price * df['volume']
                    positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
                    negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
                    positive_mf = positive_flow.rolling(window=14).sum()
                    negative_mf = negative_flow.rolling(window=14).sum()
                    money_ratio = positive_mf / negative_mf.replace([np.inf, -np.inf, np.nan], 0)
                    mfi_values = 100 - (100 / (1 + money_ratio.replace([np.inf, -np.inf, np.nan], 1)))
                    indicators['mfi'] = float(mfi_values.iloc[-1]) if not pd.isna(mfi_values.iloc[-1]) else 50
            except Exception as e:
                logger.warning(f"Could not calculate MFI: {e}")
                indicators['mfi'] = 50  # Neutral if calculation fails
            
            # Moving Average Crossovers
            indicators['ema_20_above_50'] = indicators['ema_20'] > indicators['ema_50']
            indicators['ema_50_above_200'] = indicators['ema_50'] > indicators['ema_200']
            indicators['golden_cross'] = indicators['ema_20_above_50'] and indicators['ema_50_above_200']
            indicators['death_cross'] = not indicators['ema_20_above_50'] and not indicators['ema_50_above_200']
            
            # Price position relative to EMAs
            current_price = df['close'].iloc[-1]
            indicators['price_above_ema20'] = current_price > indicators['ema_20']
            indicators['price_above_ema50'] = current_price > indicators['ema_50']
            indicators['price_above_ema200'] = current_price > indicators['ema_200']
            
            # Advanced pattern recognition (replaces basic candlestick detection)
            pattern_analysis = self.pattern_recognizer.detect_all_patterns(df)
            indicators['pattern_analysis'] = pattern_analysis
            indicators['candlestick_pattern'] = pattern_analysis.get('primary_pattern', 'unknown')
            # Keep old method for backward compatibility
            indicators['basic_candlestick'] = self._detect_candlestick_pattern(df)
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        
        return indicators
    
    def _analyze_price_action(self, df: pd.DataFrame) -> dict:
        """Analyze price action and trends"""
        current_price = df['close'].iloc[-1]
        
        # Use shorter window for trading (more responsive)
        short_prices = df['close'].tail(5)  # Last week
        medium_prices = df['close'].tail(10)  # Last 2 weeks
        recent_prices = df['close'].tail(20)  # Last month
        
        # Calculate EMAs for trend
        ema_5 = df['close'].ewm(span=5).mean().iloc[-1]
        ema_10 = df['close'].ewm(span=10).mean().iloc[-1]
        ema_20 = df['close'].ewm(span=20).mean().iloc[-1]
        
        # Short-term momentum (for trading)
        short_change = ((short_prices.iloc[-1] / short_prices.iloc[0]) - 1) * 100  # 5-day change
        medium_change = ((medium_prices.iloc[-1] / medium_prices.iloc[0]) - 1) * 100  # 10-day change
        
        # Determine trend based on multiple factors
        bullish_signals = 0
        bearish_signals = 0
        
        # EMA alignment
        if ema_5 > ema_10 > ema_20:
            bullish_signals += 2  # Strong bullish
        elif ema_5 > ema_10:
            bullish_signals += 1
        elif ema_5 < ema_10 < ema_20:
            bearish_signals += 2  # Strong bearish
        elif ema_5 < ema_10:
            bearish_signals += 1
        
        # Short-term momentum (most important for trading)
        if short_change > 2:
            bullish_signals += 2
        elif short_change > 0:
            bullish_signals += 1
        elif short_change < -2:
            bearish_signals += 2
        elif short_change < 0:
            bearish_signals += 1
        
        # Price vs EMAs
        if current_price > ema_5:
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        # Determine trend
        if bullish_signals > bearish_signals + 1:
            trend = "uptrend"
        elif bearish_signals > bullish_signals + 1:
            trend = "downtrend"
        else:
            trend = "sideways"
        
        # Price position relative to recent range
        recent_high = recent_prices.max()
        recent_low = recent_prices.min()
        price_position = ((current_price - recent_low) / (recent_high - recent_low)) * 100 if recent_high != recent_low else 50
        
        return {
            "trend": trend,
            "current_price": float(current_price),
            "recent_high": float(recent_high),
            "recent_low": float(recent_low),
            "price_position_percent": float(price_position),
            "short_change_pct": float(short_change),
            "medium_change_pct": float(medium_change),
            "ema_5": float(ema_5),
            "ema_10": float(ema_10),
            "ema_20": float(ema_20),
            "bullish_signals": bullish_signals,
            "bearish_signals": bearish_signals
        }
    
    def _analyze_volume(self, df: pd.DataFrame) -> dict:
        """Analyze volume patterns and liquidity"""
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].tail(20).mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Volume trend
        recent_volumes = df['volume'].tail(10)
        volume_trend = "increasing" if recent_volumes.iloc[-1] > recent_volumes.iloc[0] else "decreasing"
        
        # Calculate average daily dollar volume for liquidity assessment
        avg_price = df['close'].tail(20).mean()
        avg_dollar_volume = avg_volume * avg_price
        
        # Liquidity grade based on daily dollar volume
        if avg_dollar_volume >= 50_000_000:  # $50M+/day
            liquidity_grade = 'high'
        elif avg_dollar_volume >= 10_000_000:  # $10M-$50M/day
            liquidity_grade = 'medium'
        elif avg_dollar_volume >= 1_000_000:  # $1M-$10M/day
            liquidity_grade = 'low'
        else:
            liquidity_grade = 'very_low'
        
        return {
            "current_volume": int(current_volume),
            "average_volume": int(avg_volume),
            "volume_ratio": float(volume_ratio),
            "volume_trend": volume_trend,
            "is_high_volume": volume_ratio > 1.5,
            "avg_dollar_volume": float(avg_dollar_volume),
            "liquidity_grade": liquidity_grade
        }
    
    def _analyze_momentum(self, indicators: dict) -> dict:
        """Analyze momentum indicators"""
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_diff = indicators.get('macd_diff', 0)
        stoch_k = indicators.get('stoch_k', 50)
        stoch_d = indicators.get('stoch_d', 50)
        williams_r = indicators.get('williams_r', -50)
        adx = indicators.get('adx', 25)
        adx_pos = indicators.get('adx_pos', 0)
        adx_neg = indicators.get('adx_neg', 0)
        obv_trend = indicators.get('obv_trend', 'neutral')
        
        momentum_signals = []
        momentum_score = 0
        
        # RSI signals
        if rsi < self.rsi_oversold:
            momentum_signals.append("RSI oversold - potential buy")
            momentum_score += 2
        elif rsi > self.rsi_overbought:
            momentum_signals.append("RSI overbought - potential sell")
            momentum_score -= 2
        
        # MACD signals
        if macd_diff > 0 and macd > macd_signal:
            momentum_signals.append("MACD bullish crossover")
            momentum_score += 1
        elif macd_diff < 0 and macd < macd_signal:
            momentum_signals.append("MACD bearish crossover")
            momentum_score -= 1
        
        # Stochastic signals
        if stoch_k < 20 and stoch_d < 20:
            momentum_signals.append("Stochastic oversold")
            momentum_score += 1
        elif stoch_k > 80 and stoch_d > 80:
            momentum_signals.append("Stochastic overbought")
            momentum_score -= 1
        
        # Williams %R signals
        if williams_r < -80:
            momentum_signals.append("Williams %R oversold")
            momentum_score += 1
        elif williams_r > -20:
            momentum_signals.append("Williams %R overbought")
            momentum_score -= 1
        
        # ADX trend strength
        if adx > 25:
            if adx_pos > adx_neg:
                momentum_signals.append(f"Strong uptrend (ADX: {adx:.1f})")
                momentum_score += 1
            elif adx_neg > adx_pos:
                momentum_signals.append(f"Strong downtrend (ADX: {adx:.1f})")
                momentum_score -= 1
        else:
            momentum_signals.append("Weak trend - sideways movement")
        
        # OBV trend
        if obv_trend == 'bullish':
            momentum_signals.append("On-Balance Volume trending up")
            momentum_score += 1
        elif obv_trend == 'bearish':
            momentum_signals.append("On-Balance Volume trending down")
            momentum_score -= 1
        
        # Money Flow Index signals
        mfi = indicators.get('mfi', 50)
        if mfi < 20:
            momentum_signals.append(f"MFI oversold ({mfi:.2f})")
            momentum_score += 2
        elif mfi > 80:
            momentum_signals.append(f"MFI overbought ({mfi:.2f})")
            momentum_score -= 2
        elif mfi > 70:
            momentum_signals.append(f"MFI approaching overbought ({mfi:.2f})")
            momentum_score -= 1
        
        return {
            "rsi_status": "oversold" if rsi < self.rsi_oversold else "overbought" if rsi > self.rsi_overbought else "neutral",
            "macd_bullish": macd > macd_signal,
            "stoch_oversold": stoch_k < 20,
            "stoch_overbought": stoch_k > 80,
            "trend_strength": "strong" if adx > 25 else "weak",
            "obv_trend": obv_trend,
            "momentum_score": momentum_score,
            "signals": momentum_signals
        }
    
    def _detect_candlestick_pattern(self, df: pd.DataFrame) -> str:
        """Detect common candlestick patterns"""
        try:
            if len(df) < 3:
                return "insufficient_data"
            
            # Get last few candles
            last = df.iloc[-1]
            prev = df.iloc[-2] if len(df) >= 2 else None
            prev2 = df.iloc[-3] if len(df) >= 3 else None
            
            open_price = last['open']
            close_price = last['close']
            high_price = last['high']
            low_price = last['low']
            
            body = abs(close_price - open_price)
            upper_shadow = high_price - max(open_price, close_price)
            lower_shadow = min(open_price, close_price) - low_price
            total_range = high_price - low_price
            
            if total_range == 0:
                return "doji"
            
            body_ratio = body / total_range
            
            # Doji pattern (small body, similar open/close)
            if body_ratio < 0.1:
                return "doji"
            
            # Hammer pattern (small body at top, long lower shadow)
            if body_ratio < 0.3 and lower_shadow > body * 2 and upper_shadow < body * 0.5:
                return "hammer"
            
            # Shooting star (small body at bottom, long upper shadow)
            if body_ratio < 0.3 and upper_shadow > body * 2 and lower_shadow < body * 0.5:
                return "shooting_star"
            
            # Engulfing patterns
            if prev is not None:
                prev_body = abs(prev['close'] - prev['open'])
                is_bullish_engulfing = (prev['close'] < prev['open'] and  # Previous was bearish
                                       close_price > open_price and  # Current is bullish
                                       open_price < prev['close'] and  # Current opens below prev close
                                       close_price > prev['open'])  # Current closes above prev open
                
                is_bearish_engulfing = (prev['close'] > prev['open'] and  # Previous was bullish
                                       close_price < open_price and  # Current is bearish
                                       open_price > prev['close'] and  # Current opens above prev close
                                       close_price < prev['open'])  # Current closes below prev open
                
                if is_bullish_engulfing:
                    return "bullish_engulfing"
                elif is_bearish_engulfing:
                    return "bearish_engulfing"
            
            # Regular bullish/bearish
            if close_price > open_price:
                return "bullish"
            else:
                return "bearish"
                
        except Exception as e:
            logger.error(f"Error detecting candlestick pattern: {e}")
            return "unknown"
    
    def _identify_levels(self, df: pd.DataFrame) -> dict:
        """Identify volume-weighted support and resistance levels"""
        try:
            # Ensure we have required columns
            required_cols = ['high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                # Fallback to simple method if volume data is missing
                return self._identify_levels_simple(df)
            
            # Use volume-weighted calculation
            levels = self.volume_levels_calculator.calculate_levels(df, lookback=100)
            
            current_price = df['close'].iloc[-1]
            support = levels.get('support', df['low'].min())
            resistance = levels.get('resistance', df['high'].max())
            
            # Calculate distances
            distance_to_support = float((current_price - support) / support * 100) if support > 0 else 0
            distance_to_resistance = float((resistance - current_price) / current_price * 100) if current_price > 0 else 0
            
            return {
                "support": support,
                "resistance": resistance,
                "support_strength": levels.get('support_strength', 50),
                "resistance_strength": levels.get('resistance_strength', 50),
                "support_levels": levels.get('support_levels', []),
                "resistance_levels": levels.get('resistance_levels', []),
                "current_distance_to_support": distance_to_support,
                "current_distance_to_resistance": distance_to_resistance,
                "volume_weighted": True  # Flag to indicate volume-weighted calculation
            }
        except Exception as e:
            logger.warning(f"Error calculating volume-weighted levels: {e}. Falling back to simple method.")
            return self._identify_levels_simple(df)
    
    def _identify_levels_simple(self, df: pd.DataFrame) -> dict:
        """Fallback: Simple method to find local highs and lows"""
        prices = df['close'].tail(100)
        
        # Simple method: find local highs and lows
        highs = []
        lows = []
        
        for i in range(1, len(prices) - 1):
            if prices.iloc[i] > prices.iloc[i-1] and prices.iloc[i] > prices.iloc[i+1]:
                highs.append(float(prices.iloc[i]))
            if prices.iloc[i] < prices.iloc[i-1] and prices.iloc[i] < prices.iloc[i+1]:
                lows.append(float(prices.iloc[i]))
        
        resistance = max(highs) if highs else float(prices.max())
        support = min(lows) if lows else float(prices.min())
        
        return {
            "support": support,
            "resistance": resistance,
            "support_strength": 50,  # Default strength
            "resistance_strength": 50,
            "support_levels": [support] if support else [],
            "resistance_levels": [resistance] if resistance else [],
            "current_distance_to_support": float((df['close'].iloc[-1] - support) / support * 100) if support > 0 else 0,
            "current_distance_to_resistance": float((resistance - df['close'].iloc[-1]) / df['close'].iloc[-1] * 100) if df['close'].iloc[-1] > 0 else 0,
            "volume_weighted": False
        }
    
    def _get_market_regime(self) -> dict:
        """Get current market regime by analyzing SPY (S&P 500)"""
        # Cache regime for 1 hour to avoid excessive API calls
        from datetime import datetime, timedelta
        if (self._market_regime_cache and self._market_regime_cache_time and 
            (datetime.now() - self._market_regime_cache_time) < timedelta(hours=1)):
            return self._market_regime_cache
        
        if not self.data_fetcher:
            # No data fetcher available, return neutral
            return {'regime': 'unknown', 'strength': 50, 'reason': 'No data fetcher available'}
        
        try:
            # Fetch SPY data (S&P 500 index as market proxy)
            spy_data = self.data_fetcher.fetch_stock_data('SPY', force_refresh=False)
            if not spy_data or 'error' in spy_data:
                logger.debug("Could not fetch SPY data for regime detection")
                return {'regime': 'unknown', 'strength': 50, 'reason': 'SPY data unavailable'}
            
            spy_history = self.data_fetcher.fetch_stock_history('SPY', period='1y')
            if not spy_history or 'error' in spy_history:
                logger.debug("Could not fetch SPY history for regime detection")
                return {'regime': 'unknown', 'strength': 50, 'reason': 'SPY history unavailable'}
            
            # Convert to DataFrame
            spy_df = pd.DataFrame(spy_history.get('data', []))
            if spy_df.empty or len(spy_df) < 50:
                return {'regime': 'unknown', 'strength': 50, 'reason': 'Insufficient SPY data'}
            
            spy_df['date'] = pd.to_datetime(spy_df['date'])
            spy_df.set_index('date', inplace=True)
            spy_df = spy_df.sort_index()
            
            # Detect regime
            regime_info = self.regime_detector.detect_regime(spy_df[['close']])
            
            # Cache the result
            self._market_regime_cache = regime_info
            self._market_regime_cache_time = datetime.now()
            
            logger.debug(f"Market regime detected: {regime_info['regime']} (strength: {regime_info['strength']})")
            return regime_info
            
        except Exception as e:
            logger.warning(f"Error detecting market regime: {e}")
            return {'regime': 'unknown', 'strength': 50, 'reason': f'Error: {str(e)}'}
    
    def _get_spy_data(self) -> pd.DataFrame:
        """Get SPY (S&P 500) data for relative strength comparison"""
        # Cache SPY data for 1 hour to avoid excessive API calls
        from datetime import datetime, timedelta
        if (self._spy_data_cache is not None and self._spy_data_cache_time and 
            (datetime.now() - self._spy_data_cache_time) < timedelta(hours=1)):
            return self._spy_data_cache
        
        if not self.data_fetcher:
            return None
        
        try:
            spy_history = self.data_fetcher.fetch_stock_history('SPY', period='1y')
            if not spy_history or 'error' in spy_history:
                logger.debug("Could not fetch SPY history for relative strength")
                return None
            
            spy_df = pd.DataFrame(spy_history.get('data', []))
            if spy_df.empty or len(spy_df) < 20:
                return None
            
            spy_df['date'] = pd.to_datetime(spy_df['date'])
            spy_df.set_index('date', inplace=True)
            spy_df = spy_df.sort_index()
            
            # Cache the result
            self._spy_data_cache = spy_df
            self._spy_data_cache_time = datetime.now()
            
            return spy_df
        except Exception as e:
            logger.warning(f"Error fetching SPY data for relative strength: {e}")
            return None
    
    def _calculate_relative_strength(self, stock_df: pd.DataFrame) -> dict:
        """Calculate relative strength vs market (SPY)"""
        try:
            spy_df = self._get_spy_data()
            if spy_df is None or len(spy_df) < 20:
                return {'rs_ratio': 1.0, 'outperformance': 0, 'trend': 'neutral', 
                       'stock_return': 0, 'market_return': 0, 'available': False}
            
            # Align dates - use common dates between stock and SPY
            common_dates = stock_df.index.intersection(spy_df.index)
            if len(common_dates) < 20:
                return {'rs_ratio': 1.0, 'outperformance': 0, 'trend': 'neutral',
                       'stock_return': 0, 'market_return': 0, 'available': False}
            
            # Get aligned price series
            stock_prices = stock_df.loc[common_dates, 'close']
            spy_prices = spy_df.loc[common_dates, 'close']
            
            # Calculate relative strength
            rs_data = self.relative_strength_analyzer.calculate_relative_strength(
                stock_prices, spy_prices
            )
            rs_data['available'] = True
            
            return rs_data
        except Exception as e:
            logger.warning(f"Error calculating relative strength: {e}")
            return {'rs_ratio': 1.0, 'outperformance': 0, 'trend': 'neutral',
                   'stock_return': 0, 'market_return': 0, 'available': False}
    
    def _make_recommendation(self, indicators: dict, price_action: dict, 
                           volume_analysis: dict, momentum_analysis: dict,
                           support_resistance: dict, df: pd.DataFrame, 
                           market_regime: dict = None, timeframe_analysis: dict = None,
                           relative_strength: dict = None, sector_analysis: dict = None,
                           catalyst_info: dict = None, sentiment_info: dict = None,
                           macro_info: dict = None) -> dict:
        """Make buy/sell recommendation based on analysis"""
        score = 0
        reasons = []
        
        current_price = df['close'].iloc[-1]
        rsi = indicators.get('rsi', 50)
        macd_diff = indicators.get('macd_diff', 0)
        trend = price_action.get('trend', 'sideways')
        volume_ratio = volume_analysis.get('volume_ratio', 1)
        
        # RSI analysis (Smarter "Confirmation" Logic)
        rsi_prev = indicators.get('rsi_prev', 50)
        
        if rsi < self.rsi_oversold:
            if rsi > rsi_prev:
                # Rebounding from lows - Strong Signal
                score += 2
                reasons.append(f"RSI oversold and rebounding ({rsi:.1f}) - Confirmed Buy")
            else:
                # Still falling - Risky "Falling Knife"
                score += 1 
                reasons.append(f"RSI oversold but still falling ({rsi:.1f}) - Watch for reversal")
                
        elif rsi > 80:  # Extremely overbought
            score -= 3
            reasons.append(f"RSI extremely overbought ({rsi:.2f}) - strong sell signal")
            
        elif rsi > self.rsi_overbought:
            if rsi < rsi_prev:
                # Cooling off from highs - Strong Sell
                score -= 2
                reasons.append(f"RSI overbought and correcting ({rsi:.1f})")
            else:
                # Still pumping - could go higher
                score -= 1
                reasons.append(f"RSI overbought but trending up ({rsi:.1f})")
        
        # MACD analysis
        if macd_diff > 0:
            score += 1
            reasons.append("MACD shows bullish momentum")
        else:
            score -= 1
            reasons.append("MACD shows bearish momentum")
        
        # Trend analysis
        if trend == "uptrend":
            score += 1
            reasons.append("Price is in uptrend")
        elif trend == "downtrend":
            score -= 1
            reasons.append("Price is in downtrend")
        
        # Volume confirmation
        if volume_ratio > 1.5:
            score += 1
            reasons.append("High volume confirms move")
        
        # Support/Resistance (volume-weighted)
        distance_to_support = support_resistance.get('current_distance_to_support', 0)
        support_strength = support_resistance.get('support_strength', 50)
        resistance_strength = support_resistance.get('resistance_strength', 50)
        is_volume_weighted = support_resistance.get('volume_weighted', False)
        
        if distance_to_support < 5:
            # Near support - stronger support = more bullish
            if support_strength > 70:
                score += 2
                reasons.append(f"Near strong support level (strength: {support_strength:.0f}%)")
            elif support_strength > 50:
                score += 1
                reasons.append(f"Near support level (strength: {support_strength:.0f}%)")
            else:
                score += 0.5
                reasons.append("Near weak support level")
        
        # Near resistance - stronger resistance = more bearish
        distance_to_resistance = support_resistance.get('current_distance_to_resistance', 0)
        if distance_to_resistance < 5:
            if resistance_strength > 70:
                score -= 2
                reasons.append(f"Near strong resistance level (strength: {resistance_strength:.0f}%)")
            elif resistance_strength > 50:
                score -= 1
                reasons.append(f"Near resistance level (strength: {resistance_strength:.0f}%)")
            else:
                score -= 0.5
                reasons.append("Near weak resistance level")
        
        if is_volume_weighted:
            reasons.append("📊 Using volume-weighted support/resistance levels")

        # --- NEW: Sentiment & Macro Adjustments ---
        
        # Sentiment
        if sentiment_info and sentiment_info.get('available'):
            sent_adj = self.sentiment_analyzer.get_sentiment_context(sentiment_info)
            if sent_adj['confidence_adjustment'] != 0:
                 # Adjust score based on conf adjustment? Or just add reasoning
                 # Modifying score directly for cleaner impact
                 if sent_adj['confidence_adjustment'] > 0: score += 1
                 elif sent_adj['confidence_adjustment'] < 0: score -= 1
                 reasons.extend(sent_adj['reasoning'])
                 
        # Macro Regime
        if macro_info and macro_info.get('available'):
            macro_adj = self.macro_detector.get_macro_adjustment(macro_info)
            if macro_adj['confidence_adjustment'] < 0:
                 score -= 1 # Penalty for bad macro
                 reasons.extend(macro_adj['reasoning'])
            elif macro_adj['confidence_adjustment'] > 0:
                 # Bonus for good macro
                 pass 
                 
        # ------------------------------------------
        
        # Moving Average Crossovers
        ema_20_above_50 = indicators.get('ema_20_above_50', False)
        ema_50_above_200 = indicators.get('ema_50_above_200', False)
        golden_cross = indicators.get('golden_cross', False)
        death_cross = indicators.get('death_cross', False)
        
        if golden_cross:
            score += 2
            reasons.append("Golden Cross (EMA 20/50/200 bullish alignment)")
        elif death_cross:
            score -= 2
            reasons.append("Death Cross (EMA 20/50/200 bearish alignment)")
        elif ema_20_above_50:
            score += 1
            reasons.append("Price above key moving averages")
        elif not ema_20_above_50:
            score -= 1
            reasons.append("Price below key moving averages")
        
        # Price position relative to EMAs
        price_above_ema20 = indicators.get('price_above_ema20', False)
        price_above_ema50 = indicators.get('price_above_ema50', False)
        price_above_ema200 = indicators.get('price_above_ema200', False)
        
        if price_above_ema20 and price_above_ema50 and price_above_ema200:
            score += 1
            reasons.append("Price above all key EMAs (bullish)")
        elif not price_above_ema20 and not price_above_ema50:
            score -= 1
            reasons.append("Price below key EMAs (bearish)")
        
        # Advanced Pattern Recognition
        pattern_analysis = indicators.get('pattern_analysis', {})
        if pattern_analysis:
            patterns = pattern_analysis.get('patterns', [])
            pattern_signal = pattern_analysis.get('pattern_signal', 'neutral')
            primary_pattern = pattern_analysis.get('primary_pattern')
            
            if patterns:
                # Score based on pattern signals
                for pattern in patterns:
                    pattern_signal_type = pattern.get('signal', 'neutral')
                    pattern_strength = pattern.get('strength', 0.5)
                    pattern_name = pattern.get('name', 'unknown')
                    
                    if pattern_signal_type == 'bullish':
                        score += int(pattern_strength * 3)  # 0.7 strength = +2, 0.9 = +2
                        reasons.append(f"✅ {pattern.get('description', pattern_name)}")
                    elif pattern_signal_type == 'bearish':
                        score -= int(pattern_strength * 3)
                        reasons.append(f"⚠️ {pattern.get('description', pattern_name)}")
                
                # Add summary
                bullish_count = pattern_analysis.get('bullish_patterns', 0)
                bearish_count = pattern_analysis.get('bearish_patterns', 0)
                if bullish_count > bearish_count:
                    reasons.append(f"📊 {bullish_count} bullish pattern(s) detected")
                elif bearish_count > bullish_count:
                    reasons.append(f"📊 {bearish_count} bearish pattern(s) detected")
        
        # Fallback to basic candlestick pattern if no advanced patterns
        basic_pattern = indicators.get('basic_candlestick', 'unknown')
        if not pattern_analysis or not pattern_analysis.get('patterns'):
            if basic_pattern == 'bullish_engulfing':
                score += 2
                reasons.append("Bullish engulfing pattern detected")
            elif basic_pattern == 'bearish_engulfing':
                score -= 2
                reasons.append("Bearish engulfing pattern detected")
            elif basic_pattern == 'hammer':
                score += 1
                reasons.append("Hammer pattern (potential reversal)")
            elif basic_pattern == 'shooting_star':
                score -= 1
                reasons.append("Shooting star pattern (potential reversal)")
            elif basic_pattern == 'doji':
                reasons.append("Doji pattern (indecision)")
        
        # Money Flow Index (MFI) analysis
        mfi = indicators.get('mfi', 50)
        if mfi < 20:  # Oversold
            score += 2
            reasons.append(f"MFI indicates oversold condition ({mfi:.2f})")
        elif mfi > 80:  # Overbought
            score -= 3
            reasons.append(f"MFI indicates overbought condition ({mfi:.2f}) - strong sell signal")
        elif mfi > 70:  # Approaching overbought
            score -= 1
            reasons.append(f"MFI approaching overbought ({mfi:.2f})")
        
        # Multiple overbought indicators penalty (RSI + MFI both overbought = stronger signal)
        overbought_count = 0
        if rsi > self.rsi_overbought:
            overbought_count += 1
        if mfi > 80:
            overbought_count += 1
        
        if overbought_count >= 2:
            score -= 2  # Additional penalty for multiple overbought signals
            reasons.append("⚠️ Multiple overbought indicators (RSI + MFI) - high pullback risk")
        
        # Enhanced momentum signals
        momentum_score = momentum_analysis.get('momentum_score', 0)
        if momentum_score >= 3:
            score += 1
            reasons.append("Strong bullish momentum signals")
        elif momentum_score <= -3:
            score -= 1
            reasons.append("Strong bearish momentum signals")
        
        # Trend strength (ADX)
        trend_strength = momentum_analysis.get('trend_strength', 'weak')
        if trend_strength == 'strong' and momentum_score > 0:
            score += 1
            reasons.append("Strong trend with bullish momentum")
        elif trend_strength == 'strong' and momentum_score < 0:
            score -= 1
            reasons.append("Strong trend with bearish momentum")
        
        # Determine recommendation (with stricter overbought handling)
        # If multiple overbought indicators, cap confidence even if score suggests BUY
        overbought_penalty = 0
        if rsi > self.rsi_overbought and mfi > 80:
            overbought_penalty = 20  # Reduce confidence significantly
            # Force HOLD or SELL if extremely overbought
            if rsi > 75 and mfi > 85:
                # If excessively overbought, treat as negative factor
                score = min(score, 1) 
        
        # STANDARDIZED "SMART CONTRARIAN" LOGIC (User-Centric Update):
        # 1. Downtrend -> BUY ONLY if "Good Dip" (Long-Term Uptrend) OR "Strong Reversal" (Volume + Oversold)
        # 2. Uptrend -> BUY (Trend Follow) unless Extreme Overbought
        # 3. HOLD -> Default safe position
        
        # Determine strict signal states
        is_oversold = rsi < self.rsi_oversold or mfi < 20
        is_overbought = rsi > self.rsi_overbought or mfi > 80
        is_reversing_bearish = (
            (rsi > 65 and momentum_score < 0) or 
            'bearish' in str(pattern_analysis).lower() or
            death_cross
        )
        is_reversing_bullish = (
            (rsi < 35 and momentum_score > 0) or
            'bullish' in str(pattern_analysis).lower() or
            golden_cross
        )
        
        # Default Action
        action = "HOLD"
        confidence = 50
        
        # --- LOGIC TREE ---
        
        # Smart Contrarian Logic for Downtrends
        current_price = price_action.get('current_price', 0)
        ema_200 = indicators.get('ema_200', 0)
        
        # 1. 200 EMA Filter (The "Golden Rule")
        # If price > 200 EMA, the long-term trend is UP, so dips are buying opportunities.
        in_long_term_uptrend = current_price > ema_200 if ema_200 and not np.isnan(ema_200) else True 
        
        if trend == "downtrend" or trend == "sideways":
            # STRATEGIC VIEW: Price is low/falling.
            
            if in_long_term_uptrend:
                # "Good Dip" - Pullback in an uptrend
                action = "BUY"
                if is_oversold:
                    confidence = 90
                    reasons.append("💎 PERFECT SETUP: Oversold Dip in Long-Term Uptrend (>200 EMA)")
                elif score > 0:
                    confidence = 80
                    reasons.append("📉 Buying the Dip (Price > 200 EMA, Indicators Improving)")
                else:
                    confidence = 65
                    reasons.append("📉 Accumulating Dip in Long-Term Uptrend")
                    
            else:
                # "Bad Dip" - Price < 200 EMA (Long Term Downtrend)
                # Only buy if there is massive confirmation (Volume + Oversold)
                is_high_volume = volume_analysis.get('is_high_volume', False) or volume_analysis.get('volume_ratio', 1) > 1.2
                
                if is_oversold and is_high_volume:
                    action = "BUY"
                    confidence = 75 # Lower confidence than "Good Dip"
                    reasons.append("⚠️ Contrarian Play: Oversold + Volume Spike in Downtrend")
                elif is_oversold:
                    action = "HOLD" # Wait for volume
                    confidence = 55
                    reasons.append("✋ Oversold but in Downtrend - Waiting for Volume/Confirmation")
                else:
                    action = "HOLD" # Avoid
                    confidence = 50
                    reasons.append("⛔ AVOIDING: Price below 200 EMA (Long Term Downtrend)")
                 
            # Boost confidence if near support
            distance_to_support = support_resistance.get('current_distance_to_support', 100)
            if distance_to_support < 8:
                 if action == "BUY":
                     confidence += 5
                 reasons.append(f"✅ Near support zone ({distance_to_support:.1f}%)")
            
            # Check for severe weakness to override BUY (Stop Loss Prevention)
            if score < -5 and action == "BUY":
                 action = "HOLD"
                 confidence = 50
                 reasons.append("⚠️ Bearish momentum too strong (Score < -5) - Cancelling Buy")
                
        elif trend == "uptrend":
             # STRATEGIC VIEW: Price is high/rising.
             # User Intent: Ride trend (BUY) or Sell peak.
             
             # Check for potential SELL conditions first
             should_sell = (is_overbought and is_reversing_bearish) or score < -1
             
             if should_sell:
                 # Rocket Protection: Don't sell/short if trend is truly strong
                 if 'strong' in trend_strength and not is_reversing_bearish:
                     action = "HOLD"
                     confidence = 65
                     reasons.append("🚀 Strong Momentum - Ignoring SELL signal (Don't short the rocket)")
                 else:
                     # Valid Sell Signal
                     action = "SELL"
                     if is_overbought and is_reversing_bearish:
                         confidence = 90
                         reasons.append("🛑 Trend reversal detected at Overbought levels - SELL HIGHEST")
                     else:
                         confidence = 70
                         reasons.append("⚠️ Weakness detected in uptrend (Reversal indicated)")
                         
             elif is_overbought:
                 # Overbought but NO reversal yet -> HOLD (Wait for dip)
                 action = "HOLD"
                 confidence = 60
                 reasons.append("📈 Strong Uptrend but Overbought - Watch for pullback (Don't chase)")
                 
             else:
                 # Stable/Strong Uptrend + Not Overbought -> BUY (Join the move)
                 action = "BUY"
                 confidence = 85
                 if 'strong' in trend_strength:
                     reasons.append("🚀 Strong Uptrend Momentum - Joining the rocket")
                 else:
                     reasons.append("📈 Uptrend continues - Trend Following Entry")
        
        # --- HOLD OVERRIDE (For Unsure/Mixed Signals) ---
        # If signals are truly conflicting, force HOLD
        if (score > 2 and trend == "downtrend" and not is_oversold):
             # Weird case: Strong bullish signals in downtrend? Maybe just a bounce.
             action = "HOLD"
             confidence = 55
             reasons.append("⚠️ Conflicting signals: Indicators bullish but Trend bearish - Wait for confirmation")
        elif (score < -2 and trend == "uptrend" and not is_overbought):
             # Weird case: Strong bearish signals in uptrend?
             action = "HOLD"
             confidence = 55
             reasons.append("⚠️ Conflicting signals: Indicators bearish but Trend bullish - Wait for clarity")
             
        # Fallback / Safety
        if score <= -5: 
             # Catastrophic technicals
             if action == "BUY":
                 confidence = max(40, confidence - 20)
                 reasons.append("☠️ EXTREME NEGATIVE SIGNALS - High Risk Entry")

        # Adjust confidence based on multi-timeframe analysis
        if timeframe_analysis and timeframe_analysis.get('alignment') != 'insufficient_data':
            alignment = timeframe_analysis.get('alignment', 'mixed')
            
            if action == "BUY":
                if alignment == 'strong_bullish':
                    confidence = min(95, confidence + 10)
                    reasons.append(f"✅ Strong bullish alignment across timeframes")
                elif alignment == 'strong_bearish':
                    # Contrarian Buy
                    reasons.append(f"⚠️ Buying against major downtrend (Counter-trend)")
            
            elif action == "SELL":
                if alignment == 'strong_bearish':
                    confidence = min(95, confidence + 10)
                    reasons.append(f"✅ Strong bearish alignment - High probability sell")
                elif alignment == 'strong_bullish':
                     reasons.append(f"⚠️ Selling into major uptrend (Counter-trend)")
            
            # Add timeframe context
            daily = timeframe_analysis.get('daily_trend', 'unknown')
            weekly = timeframe_analysis.get('weekly_trend', 'unknown')
            reasons.append(f"   Daily: {daily}, Weekly: {weekly}")
        
        # Adjust confidence based on Relative Strength
        if relative_strength and relative_strength.get('available', False):
             outperformance = relative_strength.get('outperformance', 0)
             if outperformance > 5:
                 reasons.append(f"💪 Stock is outperforming market by {outperformance:.1f}%")
                 if action == "BUY": confidence = min(95, confidence + 5)
             elif outperformance < -5:
                 reasons.append(f"📉 Stock is underperforming market by {abs(outperformance):.1f}%")

        # Sector Momentum Analysis
        if sector_analysis and sector_analysis.get('available', False):
            sector_context = self.sector_analyzer.get_sector_context_for_recommendation(sector_analysis)
            score += sector_context.get('score_adjustment', 0)
            reasons.extend(sector_context.get('reasoning', []))
            
            # Additional confidence adjustment for sector alignment
            sector_signal = sector_analysis.get('sector_signal', 'neutral')
            if action == "BUY" and sector_signal in ['strong_outperform', 'outperform']:
                confidence = min(95, confidence + 5)
            elif action == "SELL" and sector_signal in ['strong_underperform', 'underperform']:
                confidence = min(95, confidence + 5)
        
        # Catalyst Detection and Confidence Adjustment
        if catalyst_info and catalyst_info.get('available', False):
            catalyst_adj = self.catalyst_detector.get_catalyst_adjustment(catalyst_info)
            confidence = max(30, confidence + catalyst_adj.get('confidence_adjustment', 0))
            reasons.extend(catalyst_adj.get('reasoning', []))

        # Estimate entry/exit based on User Intent
        entry_price = current_price
        
        # Get dynamic multipliers
        target_mult = 1.0
        stop_mult = 1.0
        if self.calibration_manager:
            multipliers = self.calibration_manager.get_multipliers('trading')
            target_mult = multipliers.get('target_mult', 1.0)
            stop_mult = multipliers.get('stop_loss_mult', 1.0)
        
        # --- ENTRY, TARGET, STOP LOSS LOGIC (REVERSAL STRATEGY) ---
        
        if action == "BUY":
            # REVERSAL BUY STRATEGY: Buying the dip / oversold bounce
            
            support = support_resistance.get('support', None)
            resistance = support_resistance.get('resistance', None)
            recent_low = price_action.get('recent_low', current_price * 0.95)
            
            # 1. STOP LOSS: Strict protection (Recent Swing Low)
            if support and support < current_price:
                # If support is unreasonably far (>15% drop), use a tighter stop
                if (current_price - support) / current_price < 0.15: 
                    stop_loss = support * 0.98 # 2% below support buffer
                else:
                    # Support too far, use recent low or volatility based stop
                    stop_loss = max(recent_low * 0.98, current_price * 0.95)
            else:
                 # Dynamic Volatility Stop (ATR)
                 atr = indicators.get('atr', 0)
                 if atr > 0:
                     # 2.0 ATR captures normal volatility noise
                     stop_loss = current_price - (2.0 * atr)
                 else:
                     stop_loss = current_price * 0.95 # Fallback to 5% fixed stop
            
            # Sanity check for stop loss
            if stop_loss >= current_price:
                 stop_loss = current_price * 0.95

            # 2. TARGET PRICE: Resistance (Potential Reversal Top)
            if resistance and resistance > current_price:
                target_price = resistance
                # If resistance is too close (<5% upside), aim higher (breakout)
                if (target_price - current_price) / current_price < 0.05:
                     target_price = current_price * 1.10
            else:
                 # No clear resistance, assume mean reversion momentum
                 risk = current_price - stop_loss
                 target_price = current_price + (risk * 3) # 3:1 Reward
                 
            # 3. ENTRY: Immediate
            entry_price = current_price

            # Enforce Minimum R:R Ratio of 2.0
            potential_reward = target_price - current_price
            potential_risk = current_price - stop_loss
            
            if potential_risk > 0:
                rr_ratio = potential_reward / potential_risk
                if rr_ratio < 2.0:
                    # Adjust target to reflect the minimum viable reversal trade (Greedy Target)
                    target_price = current_price + (potential_risk * 2.5)
                    reasons.append("🎯 Target adjusted for >2.0 R:R Ratio")

        elif action == "SELL":
            # SELL / SHORT STRATEGY
            
            support = support_resistance.get('support', None)
            resistance = support_resistance.get('resistance', None)
            recent_high = price_action.get('recent_high', current_price * 1.05)
            
            # 1. STOP LOSS (Invalidation)
            if resistance and resistance > current_price:
                 if (resistance - current_price) / current_price < 0.15:
                     stop_loss = resistance * 1.02
                 else:
                     stop_loss = min(recent_high * 1.02, current_price * 1.05)
            else:
                 # Dynamic Volatility Stop (ATR) for Shorts
                 atr = indicators.get('atr', 0)
                 if atr > 0:
                     stop_loss = current_price + (2.0 * atr)
                 else:
                     stop_loss = current_price * 1.05
                 
            # 2. TARGET PRICE (Downside / Support)
            if support and support < current_price:
                target_price = support
            else:
                risk = stop_loss - current_price
                target_price = current_price - (risk * 3)
            
            entry_price = current_price
        
        else: # HOLD
             entry_price = current_price
             target_price = current_price
             stop_loss = current_price
        
        # Calculate explicit Risk-to-Reward ratio
        if action == "BUY":
            potential_reward = target_price - entry_price
            potential_risk = entry_price - stop_loss
        elif action == "SELL":
            potential_reward = entry_price - target_price
            potential_risk = stop_loss - entry_price
        else:
            potential_reward = 0
            potential_risk = 0
        
        if potential_risk > 0:
            rr_ratio = potential_reward / potential_risk
        else:
            rr_ratio = 0
        
        # R:R quality assessment
        if rr_ratio >= 3.0:
            rr_quality = "Excellent"
        elif rr_ratio >= 2.0:
            rr_quality = "Good"
        elif rr_ratio >= 1.5:
            rr_quality = "Acceptable"
        elif rr_ratio >= 1.0:
            rr_quality = "Marginal"
            if action != "HOLD":
                confidence = max(30, confidence - 10)
                reasons.append(f"⚠️ Marginal R:R ratio ({rr_ratio:.1f}:1)")
        else:
            rr_quality = "Poor"
            if action != "HOLD":
                # Very poor R:R - consider forcing HOLD
                reasons.append(f"⚠️ Poor R:R ratio ({rr_ratio:.1f}:1) - trade may not be worth risk")
                confidence = max(25, confidence - 15)
        
        # Add R:R to reasons for good trades
        if rr_ratio >= 2.0 and action != "HOLD":
            reasons.append(f"✅ {rr_quality} Risk/Reward: {rr_ratio:.1f}:1")
        
        # Get liquidity grade from volume analysis
        liquidity_grade = volume_analysis.get('liquidity_grade', 'unknown')
        if liquidity_grade == 'low' and action != "HOLD":
            confidence = max(30, confidence - 10)
            reasons.append("⚠️ Low liquidity - increased execution risk")
        
        return {
            "action": action,
            "confidence": confidence,
            "entry_price": float(entry_price),
            "target_price": float(target_price),
            "stop_loss": float(stop_loss),
            "risk_reward_ratio": round(rr_ratio, 2),
            "rr_quality": rr_quality,
            "score": score,
            "reasons": reasons,
            "estimated_days": self._calculate_estimated_days(entry_price, target_price, indicators.get('atr', 0))
        }

    def _calculate_estimated_days(self, entry: float, target: float, atr: float) -> int:
        """
        Calculate estimated days to reach target based on ATR (Volatility).
        Formula: Days = (Distance / ATR) * Safety_Factor
        """
        if atr <= 0 or entry <= 0:
            return 10  # Fallback default
            
        distance = abs(target - entry)
        if distance == 0:
            return 10
            
        # Raw days needed if price moved perfectly linearly (impossible)
        raw_days = distance / atr
        
        # Safety factor: Price rarely moves in a straight line. 
        # 1.5 = Aggressive trend
        # 2.0 = Normal trend (zig-zag)
        # 3.0 = Slow/Choppy trend
        safety_factor = 2.0
        
        estimated_days = int(raw_days * safety_factor)
        
        # Clamp result to prevent absurd values, but give "full freedom" as requested
        # Allow anywhere from 1 day to 1 year based on pure math/volatility
        return max(1, min(365, estimated_days))
    
    def _generate_reasoning(self, recommendation: dict, indicators: dict,
                           price_action: dict, volume_analysis: dict,
                           momentum_analysis: dict, support_resistance: dict = None,
                           market_regime: dict = None, timeframe_analysis: dict = None,
                           relative_strength: dict = None, sector_analysis: dict = None,
                           catalyst_info: dict = None, sentiment_info: dict = None,
                           macro_info: dict = None) -> str:
        """Generate human-readable reasoning for the recommendation"""
        action = recommendation.get('action', 'HOLD')
        reasons = recommendation.get('reasons', [])
        
        reasoning = f"Trading Analysis Recommendation: {action}\n\n"
        reasoning += f"Confidence: {recommendation.get('confidence', 0)}%\n\n"
        
        # Add Catalyst Information (Earnings/Dividends)
        if catalyst_info and catalyst_info.get('available', False):
            reasoning += "📅 Upcoming Catalysts:\n"
            if catalyst_info.get('earnings_date'):
                days = catalyst_info.get('days_to_earnings')
                reasoning += f"   • Earnings in {days} days ({catalyst_info['earnings_date']})\n"
            if catalyst_info.get('dividend_date'):
                days = catalyst_info.get('days_to_dividend')
                reasoning += f"   • Ex-Dividend in {days} days ({catalyst_info['dividend_date']})\n"
            if catalyst_info.get('catalyst_warning'):
                reasoning += "   • ⚠️ CAUTION: Catalyst event imminent, volatility expected\n"
            reasoning += "\n"

        # Add Sentiment Information
        if sentiment_info and sentiment_info.get('available'):
            score = sentiment_info.get('sentiment_score', 0)
            rating = sentiment_info.get('sentiment_rating', 'neutral')
            emoji = "🐂" if "bullish" in rating else "🐻" if "bearish" in rating else "😐"
            reasoning += f"{emoji} News Sentiment: {rating.replace('_', ' ').upper()} (Score: {score:.1f})\n"
            
            headlines = sentiment_info.get('headlines', [])
            if headlines:
                reasoning += "   Recent Headlines:\n"
                for h in headlines[:2]: # Show top 2
                    reasoning += f"   • {h['title']} ({h['label']})\n"
            reasoning += "\n"

        # Add Macro Information
        if macro_info and macro_info.get('available'):
            risk = macro_info.get('risk_level', 'neutral')
            summary = macro_info.get('regime_summary', '')
            emoji = "🌪️" if risk == 'extreme' else "🌩️" if risk == 'high' else "🌥️" if risk == 'neutral' else "☀️"
            reasoning += f"{emoji} Macro Environment: {summary}\n"
            if macro_info.get('vix_status') != 'unknown':
                reasoning += f"   • VIX: {macro_info.get('vix_value', 0):.1f} ({macro_info.get('vix_status')})\n"
            if macro_info.get('yield_status') != 'unknown':
                reasoning += f"   • 10Y Yield: {macro_info.get('tnx_value', 0):.2f}%\n"
            reasoning += "\n"
            
        # Add Sector Analysis
        if sector_analysis and sector_analysis.get('available', False): # Changed from original to match diff's implied structure
            sector = sector_analysis.get('sector', 'Unknown')
            signal = sector_analysis.get('sector_signal', 'neutral')
            vs_sector = sector_analysis.get('stock_vs_sector', 0)
            
            signal_emoji = "🚀" if "outperform" in signal else "🐢" if "underperform" in signal else "⚖️"
            reasoning += f"{signal_emoji} Sector Analysis ({sector}): {signal.replace('_', ' ').title()}\n"
            reasoning += f"   • Sector Trend: {sector_analysis.get('sector_trend', 'unknown')}\n"
            reasoning += f"   • vs Sector: {vs_sector:+.1f}%\n"
            reasoning += "\n"
        
        # Add Multi-Timeframe Alignment (Moved from original position)
        if timeframe_analysis and timeframe_analysis.get('alignment') != 'insufficient_data':
            alignment = timeframe_analysis.get('alignment', 'mixed')
            timeframe_conf = timeframe_analysis.get('confidence', 50)
            daily_trend = timeframe_analysis.get('daily_trend', 'unknown')
            weekly_trend = timeframe_analysis.get('weekly_trend', 'unknown')
            monthly_trend = timeframe_analysis.get('monthly_trend', 'unknown')
            
            alignment_emoji = "✅" if 'strong' in alignment else "📈" if 'bullish' in alignment else "📉" if 'bearish' in alignment else "↔️"
            reasoning += f"{alignment_emoji} Multi-Timeframe Analysis: {alignment.replace('_', ' ').title()}\n"
            reasoning += f"   • Daily Trend (20 days): {daily_trend.upper()}\n"
            reasoning += f"   • Weekly Trend (35 days): {weekly_trend.upper()}\n"
            reasoning += f"   • Monthly Trend (90 days): {monthly_trend.upper()}\n"
            reasoning += f"   • Alignment Confidence: {timeframe_conf:.0f}%\n"
            
            if alignment == 'strong_bullish':
                reasoning += "   • ✅ All timeframes bullish - very strong signal\n"
            elif alignment == 'bullish':
                reasoning += "   • 📈 Majority bullish - good signal\n"
            elif alignment == 'strong_bearish':
                reasoning += "   • ⚠️ All timeframes bearish - very weak signal\n" # This line seems to be a copy-paste error from the diff, should be strong_bearish
            elif alignment == 'bearish':
                reasoning += "   • 📉 Majority bearish - weak signal\n"
            else:
                reasoning += "   • ↔️ Mixed signals - wait for clearer direction\n"
            reasoning += "\n"
        
        # Add relative strength information
        if relative_strength and relative_strength.get('available', False):
            rs_trend = relative_strength.get('trend', 'neutral')
            rs_ratio = relative_strength.get('rs_ratio', 1.0)
            outperformance = relative_strength.get('outperformance', 0)
            stock_return = relative_strength.get('stock_return', 0)
            market_return = relative_strength.get('market_return', 0)
            
            rs_emoji = "🚀" if rs_trend == 'strong_outperformance' else "📈" if rs_trend == 'outperformance' else "📉" if rs_trend == 'underperformance' else "↔️"
            reasoning += f"{rs_emoji} Relative Strength vs Market (SPY): {rs_trend.replace('_', ' ').title()}\n"
            reasoning += f"   • Stock Return (20 days): {stock_return:+.2f}%\n"
            reasoning += f"   • Market Return (20 days): {market_return:+.2f}%\n"
            reasoning += f"   • Outperformance: {outperformance:+.2f}%\n"
            reasoning += f"   • RS Ratio: {rs_ratio:.2f}\n"
            
            if rs_trend == 'strong_outperformance':
                reasoning += "   • 🚀 Stock significantly outperforming market - excellent opportunity\n"
            elif rs_trend == 'outperformance':
                reasoning += "   • 📈 Stock outperforming market - good opportunity\n"
            elif rs_trend == 'underperformance':
                reasoning += "   • 📉 Stock underperforming market - weaker signal\n"
            else:
                reasoning += "   • ↔️ Stock moving in line with market\n"
            reasoning += "\n"
        
        # Add market regime information
        if market_regime and market_regime.get('regime') != 'unknown':
            regime = market_regime.get('regime', 'unknown')
            strength = market_regime.get('strength', 50)
            regime_emoji = "📈" if regime == 'bull' else "📉" if regime == 'bear' else "↔️"
            reasoning += f"{regime_emoji} Market Regime: {regime.upper()} (Strength: {strength:.0f}%)\n"
            if regime == 'bull':
                reasoning += "   • Bull market conditions - favorable for long positions\n"
            elif regime == 'bear':
                reasoning += "   • Bear market conditions - more conservative approach recommended\n"
            reasoning += "\n"
        
        # Add support/resistance strength info
        if support_resistance and support_resistance.get('volume_weighted', False):
            support_strength = support_resistance.get('support_strength', 50)
            resistance_strength = support_resistance.get('resistance_strength', 50)
            reasoning += f"📊 Volume-Weighted Support/Resistance:\n"
            reasoning += f"   • Support: ${support_resistance.get('support', 0):.2f} (Strength: {support_strength:.0f}%)\n"
            reasoning += f"   • Resistance: ${support_resistance.get('resistance', 0):.2f} (Strength: {resistance_strength:.0f}%)\n"
            if support_resistance.get('support_levels'):
                reasoning += f"   • Additional Support Levels: {', '.join([f'${s:.2f}' for s in support_resistance['support_levels'][:2]])}\n"
            if support_resistance.get('resistance_levels'):
                reasoning += f"   • Additional Resistance Levels: {', '.join([f'${r:.2f}' for r in support_resistance['resistance_levels'][:2]])}\n"
            reasoning += "\n"
        
        reasoning += "Key Factors:\n"
        for reason in reasons:
            reasoning += f"• {reason}\n"
        
        reasoning += f"\nTechnical Indicators:\n"
        rsi = indicators.get('rsi', 0)
        rsi_status = momentum_analysis.get('rsi_status', 'neutral')
        if rsi > 70:
            rsi_status = f"OVERBOUGHT ⚠️"
        elif rsi < 30:
            rsi_status = f"OVERSOLD"
        reasoning += f"• RSI: {rsi:.2f} ({rsi_status})\n"
        
        macd_diff = indicators.get('macd_diff', 0)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        reasoning += f"• MACD: {macd_diff:.4f} ({'Bullish' if macd_diff > 0 else 'Bearish'}) [MACD: {macd:.2f}, Signal: {macd_signal:.2f}]\n"
        
        # Money Flow Index
        mfi = indicators.get('mfi', 50)
        if mfi > 80:
            mfi_status = "OVERBOUGHT ⚠️"
        elif mfi < 20:
            mfi_status = "OVERSOLD"
        else:
            mfi_status = "Neutral"
        reasoning += f"• Money Flow Index (MFI): {mfi:.2f} ({mfi_status})\n"
        
        # Additional momentum indicators
        stoch_k = indicators.get('stoch_k', 50)
        if stoch_k:
            reasoning += f"• Stochastic: {stoch_k:.2f} ({'Oversold' if stoch_k < 20 else 'Overbought' if stoch_k > 80 else 'Neutral'})\n"
        
        williams_r = indicators.get('williams_r', -50)
        if williams_r:
            reasoning += f"• Williams %R: {williams_r:.2f}\n"
        
        adx = indicators.get('adx', 25)
        if adx:
            reasoning += f"• ADX (Trend Strength): {adx:.2f} ({'Strong' if adx > 25 else 'Weak'})\n"
        
        # Moving averages
        if indicators.get('golden_cross'):
            reasoning += f"• Golden Cross detected (EMA 20/50/200 bullish alignment)\n"
        elif indicators.get('death_cross'):
            reasoning += f"• Death Cross detected (EMA 20/50/200 bearish alignment)\n"
        
        # Pattern Recognition
        pattern_analysis = indicators.get('pattern_analysis', {})
        if pattern_analysis and pattern_analysis.get('patterns'):
            patterns = pattern_analysis.get('patterns', [])
            primary_pattern = pattern_analysis.get('primary_pattern')
            pattern_signal = pattern_analysis.get('pattern_signal', 'neutral')
            
            reasoning += f"📊 Chart Pattern Analysis:\n"
            reasoning += f"   • Primary Pattern: {primary_pattern.replace('_', ' ').title() if primary_pattern else 'None'}\n"
            reasoning += f"   • Overall Signal: {pattern_signal.upper()}\n"
            reasoning += f"   • Patterns Detected: {len(patterns)}\n"
            
            # List top 3 patterns
            sorted_patterns = sorted(patterns, key=lambda x: x.get('strength', 0), reverse=True)
            for i, pattern in enumerate(sorted_patterns[:3], 1):
                pattern_name = pattern.get('name', 'unknown').replace('_', ' ').title()
                pattern_signal_type = pattern.get('signal', 'neutral')
                pattern_strength = pattern.get('strength', 0) * 100
                signal_emoji = "✅" if pattern_signal_type == 'bullish' else "⚠️" if pattern_signal_type == 'bearish' else "↔️"
                reasoning += f"   {i}. {signal_emoji} {pattern_name} ({pattern_strength:.0f}% strength) - {pattern.get('description', '')}\n"
            reasoning += "\n"
        
        # Fallback to basic candlestick pattern
        basic_pattern = indicators.get('basic_candlestick', 'unknown')
        if not pattern_analysis or not pattern_analysis.get('patterns'):
            if basic_pattern != 'unknown' and basic_pattern != 'bullish' and basic_pattern != 'bearish':
                reasoning += f"• Candlestick Pattern: {basic_pattern.replace('_', ' ').title()}\n"
        
        reasoning += f"• Trend: {price_action.get('trend', 'unknown')}\n"
        reasoning += f"• Volume: {volume_analysis.get('volume_ratio', 1):.2f}x average\n"
        reasoning += f"• OBV Trend: {indicators.get('obv_trend', 'neutral')}\n"
        
        if action == "BUY":
            reasoning += f"\nEntry Strategy:\n"
            reasoning += f"• Entry Price: ${recommendation.get('entry_price', 0):.2f}\n"
            reasoning += f"• Target Price: ${recommendation.get('target_price', 0):.2f}\n"
            reasoning += f"• Stop Loss: ${recommendation.get('stop_loss', 0):.2f}\n"
        
        return reasoning
