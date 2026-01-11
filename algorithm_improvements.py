"""
Algorithm Accuracy Improvements
Additional features and enhancements to improve prediction accuracy
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MarketRegimeDetector:
    """Detects market regime (bull/bear/sideways) to adjust strategy"""
    
    def detect_regime(self, market_data: pd.DataFrame) -> Dict:
        """
        Detect current market regime
        
        Args:
            market_data: DataFrame with 'close' column (SPY or market index data)
            
        Returns:
            Dict with regime info: {'regime': 'bull'/'bear'/'sideways', 'strength': 0-100}
        """
        # Allow detection with less data (minimum 20 days instead of 50)
        if market_data.empty or len(market_data) < 20:
            return {'regime': 'unknown', 'strength': 50}
        
        # Handle both 'close' and 'Close' column names
        if 'close' in market_data.columns:
            prices = market_data['close'].values
        elif 'Close' in market_data.columns:
            prices = market_data['Close'].values
        else:
            # Try to find any price-like column
            price_cols = [c for c in market_data.columns if 'close' in c.lower() or 'price' in c.lower()]
            if price_cols:
                prices = market_data[price_cols[0]].values
            else:
                # Use first numeric column as fallback
                numeric_cols = market_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    prices = market_data[numeric_cols[0]].values
                else:
                    return {'regime': 'unknown', 'strength': 50}
        current_price = prices[-1]
        
        # Calculate moving averages
        sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else np.mean(prices)
        sma_200 = np.mean(prices[-200:]) if len(prices) >= 200 else sma_50
        
        # Trend strength - ensure arrays have matching lengths
        trend_strength = 0.0  # Initialize to prevent UnboundLocalError
        if len(prices) >= 20:
            # Get last 20 prices
            price_slice = prices[-20:]
            # Calculate differences (gives 19 elements: differences between consecutive pairs)
            price_changes = np.diff(price_slice)
            # Get base prices for division (first 19 prices from the slice, matching price_changes length)
            price_bases = price_slice[:-1]  # This gives 19 elements, matching price_changes
            
            # Handle division by zero and NaN/inf values
            with np.errstate(divide='ignore', invalid='ignore'):
                returns = np.divide(price_changes, price_bases, out=np.zeros_like(price_changes), where=price_bases!=0)
                # Remove any NaN or inf values
                returns = returns[np.isfinite(returns)]
            
            if len(returns) > 0:
                trend_strength = np.mean(returns) * 100
            else:
                trend_strength = 0.0
        
        # Determine regime - relaxed thresholds for better classification
        # Use trend_strength threshold of 0.02 (2%) instead of 0.1 (10%) for more realistic detection
        # Also consider price position relative to moving averages more flexibly
        
        # Calculate price position relative to MAs
        price_above_sma50 = current_price > sma_50
        price_above_sma200 = current_price > sma_200
        sma50_above_sma200 = sma_50 > sma_200
        
        # Bull market: Price above both MAs, SMA50 above SMA200, positive trend
        is_bull_aligned = price_above_sma50 and price_above_sma200 and sma50_above_sma200
        is_bull_trend = trend_strength > 0.005  # Further relaxed from 0.02 to 0.005 (0.5% avg daily return)
        
        # Bear market: Price below both MAs, SMA50 below SMA200, negative trend
        is_bear_aligned = not price_above_sma50 and not price_above_sma200 and not sma50_above_sma200
        is_bear_trend = trend_strength < -0.005  # Further relaxed from -0.02 to -0.005 (-0.5% avg daily return)
        
        # Determine regime with SIMPLIFIED and VERY relaxed criteria
        # PRIMARY SIGNAL: Price position relative to SMA50 (this is the key decision point)
        # SECONDARY SIGNAL: Trend strength reinforces the primary signal
        # Goal: Classify as bull/bear more often, only use sideways for truly neutral cases
        
        # Primary classification: Price above SMA50 = Bull, Price below/equal SMA50 = Bear
        # This ensures we ALWAYS get either bull or bear (never sideways) unless there's an error
        
        if price_above_sma50:
            # Price is above SMA50 - classify as BULL
            # Stronger if fully aligned and positive trend, but still bull even if not
            if is_bull_aligned and is_bull_trend:
                regime = 'bull'
                strength = min(100, 50 + abs(trend_strength) * 20)
            else:
                regime = 'bull'  # Still bull, just moderate
                strength = min(80, 40 + abs(trend_strength) * 15)
        else:
            # Price is at or below SMA50 - classify as BEAR
            # Stronger if fully aligned and negative trend, but still bear even if not
            if is_bear_aligned and is_bear_trend:
                regime = 'bear'
                strength = min(100, 50 + abs(trend_strength) * 20)
            else:
                regime = 'bear'  # Still bear, just moderate
                strength = min(80, 40 + abs(trend_strength) * 15)
        
        # Safety check: ensure we never return 'unknown' from this point (should be caught earlier)
        if regime not in ['bull', 'bear', 'sideways']:
            regime = 'sideways'  # Fallback safety
            strength = 50
        
        return {
            'regime': regime,
            'strength': strength,
            'sma_50': sma_50,
            'sma_200': sma_200,
            'trend_strength': trend_strength
        }


class VolumeWeightedLevels:
    """Calculate volume-weighted support and resistance levels"""
    
    def calculate_levels(self, df: pd.DataFrame, lookback: int = 100) -> Dict:
        """
        Calculate volume-weighted support and resistance
        
        Args:
            df: DataFrame with 'high', 'low', 'close', 'volume' columns
            lookback: Number of periods to analyze
            
        Returns:
            Dict with support/resistance levels and their strengths
        """
        if len(df) < lookback:
            lookback = len(df)
        
        data = df.tail(lookback)
        
        # Create price bins
        price_range = data['high'].max() - data['low'].min()
        num_bins = 20
        bin_size = price_range / num_bins
        
        # Volume profile
        volume_profile = {}
        for idx, row in data.iterrows():
            price = row['close']
            volume = row['volume']
            bin_price = round((price - data['low'].min()) / bin_size) * bin_size + data['low'].min()
            
            if bin_price not in volume_profile:
                volume_profile[bin_price] = 0
            volume_profile[bin_price] += volume
        
        # Find high-volume nodes (support/resistance)
        sorted_profile = sorted(volume_profile.items(), key=lambda x: x[1], reverse=True)
        
        # Support: lower high-volume areas
        support_levels = [price for price, vol in sorted_profile[:5] 
                          if price < data['close'].iloc[-1]]
        support_levels.sort(reverse=True)  # Highest first
        
        # Resistance: higher high-volume areas
        resistance_levels = [price for price, vol in sorted_profile[:5] 
                            if price > data['close'].iloc[-1]]
        resistance_levels.sort()  # Lowest first
        
        # Calculate strength based on volume and touches
        def calculate_strength(level, data):
            touches = sum(1 for _, row in data.iterrows() 
                         if abs(row['close'] - level) / level < 0.01)
            volume_at_level = volume_profile.get(level, 0)
            max_volume = max(volume_profile.values()) if volume_profile else 1
            return (touches * 20) + (volume_at_level / max_volume * 80)
        
        primary_support = support_levels[0] if support_levels else data['low'].min()
        primary_resistance = resistance_levels[0] if resistance_levels else data['high'].max()
        
        return {
            'support': primary_support,
            'resistance': primary_resistance,
            'support_strength': calculate_strength(primary_support, data),
            'resistance_strength': calculate_strength(primary_resistance, data),
            'support_levels': support_levels[:3],
            'resistance_levels': resistance_levels[:3],
            'volume_profile': dict(sorted_profile[:10])
        }


class MultiTimeframeAnalyzer:
    """Analyze multiple timeframes for trend confirmation"""
    
    def analyze_timeframes(self, df: pd.DataFrame) -> Dict:
        """
        Analyze daily, weekly, and monthly trends
        
        Args:
            df: Daily price DataFrame
            
        Returns:
            Dict with trend alignment across timeframes
        """
        if len(df) < 90:
            # Need at least 90 days for monthly analysis
            return {'alignment': 'insufficient_data', 'confidence': 50, 
                   'daily_trend': 'unknown', 'weekly_trend': 'unknown', 'monthly_trend': 'unknown'}
        
        # Daily trend (last 20 days)
        daily_prices = df['close'].tail(20)
        daily_trend = 'up' if daily_prices.iloc[-1] > daily_prices.iloc[0] else 'down'
        
        # Weekly trend (last 5 weeks = ~35 days)
        weekly_prices = df['close'].tail(35)
        weekly_trend = 'up' if weekly_prices.iloc[-1] > weekly_prices.iloc[0] else 'down'
        
        # Monthly trend (last 3 months = ~90 days)
        monthly_prices = df['close'].tail(90)
        monthly_trend = 'up' if monthly_prices.iloc[-1] > monthly_prices.iloc[0] else 'down'
        
        # Calculate alignment
        trends = [daily_trend, weekly_trend, monthly_trend]
        up_count = trends.count('up')
        down_count = trends.count('down')
        
        if up_count == 3:
            alignment = 'strong_bullish'
            confidence = 85
        elif up_count == 2:
            alignment = 'bullish'
            confidence = 65
        elif down_count == 3:
            alignment = 'strong_bearish'
            confidence = 85
        elif down_count == 2:
            alignment = 'bearish'
            confidence = 65
        else:
            alignment = 'mixed'
            confidence = 50
        
        return {
            'alignment': alignment,
            'confidence': confidence,
            'daily_trend': daily_trend,
            'weekly_trend': weekly_trend,
            'monthly_trend': monthly_trend,
            'trend_agreement': up_count if up_count >= 2 else down_count
        }


class RelativeStrengthAnalyzer:
    """Analyze stock performance relative to market"""
    
    def calculate_relative_strength(self, stock_prices: pd.Series, 
                                   market_prices: pd.Series) -> Dict:
        """
        Calculate relative strength vs market
        
        Args:
            stock_prices: Stock price series
            market_prices: Market index price series (SPY, QQQ, etc.)
            
        Returns:
            Dict with relative strength metrics
        """
        if len(stock_prices) < 20 or len(market_prices) < 20:
            return {'rs_ratio': 1.0, 'outperformance': 0, 'trend': 'neutral'}
        
        # Calculate returns
        stock_returns = (stock_prices.iloc[-1] - stock_prices.iloc[-20]) / stock_prices.iloc[-20]
        market_returns = (market_prices.iloc[-1] - market_prices.iloc[-20]) / market_prices.iloc[-20]
        
        # Relative strength ratio
        rs_ratio = stock_returns / market_returns if market_returns != 0 else 1.0
        
        # Outperformance
        outperformance = (stock_returns - market_returns) * 100
        
        # Trend
        if rs_ratio > 1.1:
            trend = 'strong_outperformance'
        elif rs_ratio > 1.0:
            trend = 'outperformance'
        elif rs_ratio < 0.9:
            trend = 'underperformance'
        else:
            trend = 'neutral'
        
        return {
            'rs_ratio': rs_ratio,
            'outperformance': outperformance,
            'trend': trend,
            'stock_return': stock_returns * 100,
            'market_return': market_returns * 100
        }


class VolatilityContext:
    """Analyze market volatility context"""
    
    def analyze_volatility(self, df: pd.DataFrame) -> Dict:
        """
        Analyze volatility context (high/low volatility periods)
        
        Args:
            df: DataFrame with price data
            
        Returns:
            Dict with volatility metrics
        """
        if len(df) < 20:
            return {'volatility_regime': 'unknown', 'atr_percent': 0}
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(14).mean().iloc[-1]
        
        # Volatility as percentage
        current_price = df['close'].iloc[-1]
        atr_percent = (atr / current_price * 100) if current_price > 0 else 0
        
        # Historical volatility
        returns = df['close'].pct_change().dropna()
        if len(returns) >= 20:
            historical_vol = returns.tail(20).std() * np.sqrt(252) * 100  # Annualized
        else:
            historical_vol = 0
        
        # Volatility regime
        if atr_percent > 3.0:
            regime = 'high_volatility'
        elif atr_percent < 1.0:
            regime = 'low_volatility'
        else:
            regime = 'normal_volatility'
        
        return {
            'volatility_regime': regime,
            'atr_percent': atr_percent,
            'historical_volatility': historical_vol,
            'atr': atr
        }


class DynamicConfidenceAdjuster:
    """Adjust confidence based on multiple factors"""
    
    def adjust_confidence(self, base_confidence: float, 
                         market_regime: Dict,
                         timeframe_alignment: Dict,
                         volatility: Dict,
                         signal_strength: float) -> float:
        """
        Adjust confidence based on market context
        
        Args:
            base_confidence: Base confidence from analysis
            market_regime: Market regime info
            timeframe_alignment: Multi-timeframe alignment
            volatility: Volatility context
            signal_strength: Strength of technical signals (0-1)
            
        Returns:
            Adjusted confidence (0-100)
        """
        confidence = base_confidence
        
        # Market regime adjustment
        regime = market_regime.get('regime', 'unknown')
        if regime == 'bull':
            confidence += 5  # Slightly more confident in bull markets
        elif regime == 'bear':
            confidence -= 10  # Less confident in bear markets
        
        # Timeframe alignment
        alignment_conf = timeframe_alignment.get('confidence', 50)
        if alignment_conf > 70:
            confidence += 10  # Strong alignment = higher confidence
        elif alignment_conf < 40:
            confidence -= 10  # Weak alignment = lower confidence
        
        # Volatility adjustment
        vol_regime = volatility.get('volatility_regime', 'normal')
        if vol_regime == 'high_volatility':
            confidence -= 5  # Less confident in high volatility
        elif vol_regime == 'low_volatility':
            confidence += 3  # More confident in stable markets
        
        # Signal strength
        confidence += (signal_strength - 0.5) * 20  # Adjust based on signal strength
        
        # Clamp to valid range
        return max(30, min(95, confidence))

