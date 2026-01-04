"""
Trading Strategy Analyzer
Focuses on technical analysis, price action, momentum, and short-term indicators
"""
import pandas as pd
import numpy as np
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import logging

logger = logging.getLogger(__name__)


class TradingAnalyzer:
    """Analyzes stocks using trading/technical analysis approach"""
    
    def __init__(self):
        self.rsi_oversold = 30
        self.rsi_overbought = 70
    
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
            
            # Calculate technical indicators
            indicators = self._calculate_indicators(df)
            
            # Analyze price action
            price_action = self._analyze_price_action(df)
            
            # Analyze volume
            volume_analysis = self._analyze_volume(df)
            
            # Analyze momentum
            momentum_analysis = self._analyze_momentum(indicators)
            
            # Identify support and resistance
            support_resistance = self._identify_levels(df)
            
            # Determine recommendation
            recommendation = self._make_recommendation(
                indicators, price_action, volume_analysis, 
                momentum_analysis, support_resistance, df
            )
            
            return {
                "strategy": "trading",
                "recommendation": recommendation,
                "indicators": indicators,
                "price_action": price_action,
                "volume_analysis": volume_analysis,
                "momentum_analysis": momentum_analysis,
                "support_resistance": support_resistance,
                "reasoning": self._generate_reasoning(
                    recommendation, indicators, price_action, 
                    volume_analysis, momentum_analysis
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
            indicators['rsi'] = float(rsi_indicator.rsi().iloc[-1])
            
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
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
        
        return indicators
    
    def _analyze_price_action(self, df: pd.DataFrame) -> dict:
        """Analyze price action and trends"""
        current_price = df['close'].iloc[-1]
        recent_prices = df['close'].tail(20)
        
        # Trend detection
        higher_highs = 0
        higher_lows = 0
        lower_highs = 0
        lower_lows = 0
        
        for i in range(1, len(recent_prices)):
            if recent_prices.iloc[i] > recent_prices.iloc[i-1]:
                higher_highs += 1
            else:
                lower_highs += 1
            
            if recent_prices.iloc[i] > recent_prices.iloc[i-1]:
                higher_lows += 1
            else:
                lower_lows += 1
        
        trend = "uptrend" if higher_highs > lower_highs else "downtrend" if lower_highs > higher_highs else "sideways"
        
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
            "higher_highs": higher_highs,
            "lower_lows": lower_lows
        }
    
    def _analyze_volume(self, df: pd.DataFrame) -> dict:
        """Analyze volume patterns"""
        current_volume = df['volume'].iloc[-1]
        avg_volume = df['volume'].tail(20).mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Volume trend
        recent_volumes = df['volume'].tail(10)
        volume_trend = "increasing" if recent_volumes.iloc[-1] > recent_volumes.iloc[0] else "decreasing"
        
        return {
            "current_volume": int(current_volume),
            "average_volume": int(avg_volume),
            "volume_ratio": float(volume_ratio),
            "volume_trend": volume_trend,
            "is_high_volume": volume_ratio > 1.5
        }
    
    def _analyze_momentum(self, indicators: dict) -> dict:
        """Analyze momentum indicators"""
        rsi = indicators.get('rsi', 50)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        macd_diff = indicators.get('macd_diff', 0)
        
        momentum_signals = []
        
        # RSI signals
        if rsi < self.rsi_oversold:
            momentum_signals.append("RSI oversold - potential buy")
        elif rsi > self.rsi_overbought:
            momentum_signals.append("RSI overbought - potential sell")
        
        # MACD signals
        if macd_diff > 0 and macd > macd_signal:
            momentum_signals.append("MACD bullish crossover")
        elif macd_diff < 0 and macd < macd_signal:
            momentum_signals.append("MACD bearish crossover")
        
        return {
            "rsi_status": "oversold" if rsi < self.rsi_oversold else "overbought" if rsi > self.rsi_overbought else "neutral",
            "macd_bullish": macd > macd_signal,
            "signals": momentum_signals
        }
    
    def _identify_levels(self, df: pd.DataFrame) -> dict:
        """Identify support and resistance levels"""
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
            "current_distance_to_support": float((df['close'].iloc[-1] - support) / support * 100),
            "current_distance_to_resistance": float((resistance - df['close'].iloc[-1]) / df['close'].iloc[-1] * 100)
        }
    
    def _make_recommendation(self, indicators: dict, price_action: dict, 
                           volume_analysis: dict, momentum_analysis: dict,
                           support_resistance: dict, df: pd.DataFrame) -> dict:
        """Make buy/sell recommendation based on analysis"""
        score = 0
        reasons = []
        
        current_price = df['close'].iloc[-1]
        rsi = indicators.get('rsi', 50)
        macd_diff = indicators.get('macd_diff', 0)
        trend = price_action.get('trend', 'sideways')
        volume_ratio = volume_analysis.get('volume_ratio', 1)
        
        # RSI analysis
        if rsi < self.rsi_oversold:
            score += 2
            reasons.append("RSI indicates oversold condition")
        elif rsi > self.rsi_overbought:
            score -= 2
            reasons.append("RSI indicates overbought condition")
        
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
        
        # Support/Resistance
        distance_to_support = support_resistance.get('current_distance_to_support', 0)
        if distance_to_support < 5:
            score += 1
            reasons.append("Near support level")
        
        # Determine recommendation
        if score >= 3:
            action = "BUY"
            confidence = min(90, 60 + (score * 5))
        elif score <= -3:
            action = "SELL"
            confidence = min(90, 60 + (abs(score) * 5))
        else:
            action = "HOLD"
            confidence = 50
        
        # Estimate entry/exit
        if action == "BUY":
            entry_price = current_price
            target_price = support_resistance.get('resistance', current_price * 1.1)
            stop_loss = support_resistance.get('support', current_price * 0.95)
        elif action == "SELL":
            entry_price = current_price
            target_price = support_resistance.get('support', current_price * 0.9)
            stop_loss = support_resistance.get('resistance', current_price * 1.05)
        else:
            entry_price = current_price
            target_price = current_price
            stop_loss = current_price
        
        return {
            "action": action,
            "confidence": confidence,
            "entry_price": float(entry_price),
            "target_price": float(target_price),
            "stop_loss": float(stop_loss),
            "score": score,
            "reasons": reasons
        }
    
    def _generate_reasoning(self, recommendation: dict, indicators: dict,
                           price_action: dict, volume_analysis: dict,
                           momentum_analysis: dict) -> str:
        """Generate human-readable reasoning for the recommendation"""
        action = recommendation.get('action', 'HOLD')
        reasons = recommendation.get('reasons', [])
        
        reasoning = f"Trading Analysis Recommendation: {action}\n\n"
        reasoning += f"Confidence: {recommendation.get('confidence', 0)}%\n\n"
        
        reasoning += "Key Factors:\n"
        for reason in reasons:
            reasoning += f"• {reason}\n"
        
        reasoning += f"\nTechnical Indicators:\n"
        reasoning += f"• RSI: {indicators.get('rsi', 0):.2f} ({momentum_analysis.get('rsi_status', 'neutral')})\n"
        reasoning += f"• MACD: {indicators.get('macd_diff', 0):.4f} ({'Bullish' if indicators.get('macd_diff', 0) > 0 else 'Bearish'})\n"
        reasoning += f"• Trend: {price_action.get('trend', 'unknown')}\n"
        reasoning += f"• Volume: {volume_analysis.get('volume_ratio', 1):.2f}x average\n"
        
        if action == "BUY":
            reasoning += f"\nEntry Strategy:\n"
            reasoning += f"• Entry Price: ${recommendation.get('entry_price', 0):.2f}\n"
            reasoning += f"• Target Price: ${recommendation.get('target_price', 0):.2f}\n"
            reasoning += f"• Stop Loss: ${recommendation.get('stop_loss', 0):.2f}\n"
        
        return reasoning

