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
        Detect current market regime using Multi-Factor Analysis (SMA, Momentum, Drawdown)
        Provides high-fidelity detection to prevent 'Bull-Trap' entries in Bear markets.
        """
        if market_data.empty or len(market_data) < 50:
            return {'regime': 'unknown', 'strength': 50}
        
        # Standardize column name
        col = 'close' if 'close' in market_data.columns else 'Close'
        prices = market_data[col]
        current_price = float(prices.iloc[-1])
        
        # 1. Trend Analysis (Moving Averages)
        sma_50 = prices.rolling(50).mean().iloc[-1]
        sma_200 = prices.rolling(200).mean().iloc[-1] if len(prices) >= 200 else sma_50
        
        # 2. Risk Context (Drawdown from 1-year High)
        high_252 = prices.rolling(252).max().iloc[-1] if len(prices) >= 252 else prices.max()
        drawdown = (current_price - high_252) / high_252
        
        # 3. Momentum (Rate of Change)
        roc_20 = ((current_price - prices.iloc[-20]) / prices.iloc[-20]) * 100 if len(prices) >= 20 else 0
        
        # --- ELITE CLASSIFICATION LOGIC ---
        # A. CRASH DETECTION (Deep Drawdown or Extreme Velocity)
        if drawdown < -0.15 or roc_20 < -8:
            regime = 'crash'
            strength = min(100, abs(drawdown) * 400)
        
        # B. BEAR DETECTION (Price below 200 SMA or sustained negative slope)
        elif current_price < sma_200 or (current_price < sma_50 and roc_20 < -2):
            regime = 'bear'
            # If we are in a 'Bear Market Rally' (Price > SMA50 but < SMA200), still flag as Bear
            if current_price > sma_50:
                strength = 50 # Rally Strength
            else:
                strength = 75 # Solid Bear
        
        # C. BULL DETECTION
        elif current_price > sma_200 and current_price > sma_50 and roc_20 > -1:
            regime = 'bull'
            strength = 80 if roc_20 > 0 else 60
            
        # D. SIDEWAYS / UNCERTAIN
        else:
            regime = 'sideways'
            strength = 50
            
        return {
            'regime': regime,
            'strength': strength,
            'drawdown': drawdown,
            'sma_50': sma_50,
            'sma_200': sma_200,
            'roc_20': roc_20
        }


    def compute_regimes_vectorized(self, market_data: pd.DataFrame) -> pd.DataFrame:
        """
        Detect market regimes for the entire history vectorially
        
        Args:
            market_data: DataFrame with 'Close' or 'close' column
            
        Returns:
            DataFrame with 'Regime' column and other metrics
        """
        df = market_data.copy()
        
        # Standardize column name
        if 'Close' in df.columns:
            prices = df['Close']
        elif 'close' in df.columns:
            prices = df['close']
        else:
             # Try first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                prices = df[numeric_cols[0]]
            else:
                raise ValueError("No price column found in market data")
                
        # Calculate Moving Averages
        sma_50 = prices.rolling(window=50).mean()
        sma_200 = prices.rolling(window=200).mean()
        
        # Calculate Trend Strength (Vectorized)
        daily_returns = prices.pct_change()
        # Equivalent to original logic: mean return over 20 days * 100
        trend_strength = daily_returns.rolling(window=20).mean() * 100
        
        # Define Thresholds
        is_positive_trend = trend_strength > 0.02
        is_negative_trend = trend_strength < -0.02
        
        price_above_sma50 = prices > sma_50
        price_above_sma200 = prices > sma_200
        
        # Vectorized Regime Classification
        # Default to 'sideways'
        regimes = pd.Series('sideways', index=df.index)
        
        # Long-term BULL Context (Price > SMA200)
        bull_context = price_above_sma200
        
        # 1. Strong Bull: Price > SMA200 AND Price > SMA50
        mask_strong_bull = bull_context & price_above_sma50
        regimes[mask_strong_bull] = 'bull'
        
        # 2. Pullback/Sideways in Bull
        mask_bull_pullback = bull_context & (~price_above_sma50)
        regimes[mask_bull_pullback & is_negative_trend] = 'sideways'
        regimes[mask_bull_pullback & (~is_negative_trend)] = 'bull'
        
        # Long-term BEAR Context (Price <= SMA200)
        bear_context = ~price_above_sma200
        
        # 3. Strong Bear
        mask_strong_bear = bear_context & (~price_above_sma50)
        regimes[mask_strong_bear] = 'bear'
        
        # 4. Rally/Sideways in Bear
        mask_bear_rally = bear_context & price_above_sma50
        regimes[mask_bear_rally & is_positive_trend] = 'sideways'
        regimes[mask_bear_rally & (~is_positive_trend)] = 'bear'
        
        # Fill NaN at beginning with 'sideways'
        regimes = regimes.fillna('sideways')
        
        # Create result DataFrame
        result = pd.DataFrame(index=df.index)
        result['Regime'] = regimes
        result['TrendStrength'] = trend_strength
        result['SMA50'] = sma_50
        
        return result


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
        # Drop rows with NaNs in required columns
        data = df.tail(lookback).copy()
        data = data.dropna(subset=['high', 'low', 'close', 'volume'])
        
        if data.empty:
            return {
                'support': df['low'].min() if not df['low'].empty else 0,
                'resistance': df['high'].max() if not df['high'].empty else 0,
                'support_strength': 0,
                'resistance_strength': 0,
                'support_levels': [],
                'resistance_levels': [],
                'volume_profile': {}
            }
        
        # Create price bins
        low_min = data['low'].min()
        high_max = data['high'].max()
        price_range = high_max - low_min
        num_bins = 20
        
        if price_range <= 0 or np.isnan(price_range):
            bin_size = 0.01  # Minimum bin size
        else:
            bin_size = price_range / num_bins
        
        # Volume profile
        volume_profile = {}
        for idx, row in data.iterrows():
            price = row['close']
            # [FIX] Apply RECENCY BIAS to volume weighting
            # Recent volume is more relevant than volume from 100 days ago
            days_ago = len(data) - data.index.get_loc(idx) - 1
            recency_weight = 1.0 / (1.0 + (days_ago / 30.0)) # Decay by 50% every 30 days
            
            # Safely calculate bin index
            try:
                bin_idx = round((price - low_min) / bin_size)
                bin_price = bin_idx * bin_size + low_min
            except (ValueError, OverflowError, ZeroDivisionError):
                continue
            
            if bin_price not in volume_profile:
                volume_profile[bin_price] = 0
            volume_profile[bin_price] += row['volume'] * recency_weight
        
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


class StatisticalTargetCalibrator:
    """Validates and adjusts price targets using statistical probability (Z-Scores)"""
    
    def calibrate(self, current_price: float, target_price: float, 
                  estimated_days: int, atr: float, market_regime: str = 'unknown') -> Dict:
        """
        Calibrates the target price based on volatility and time
        
        Args:
            current_price: Current stock price
            target_price: Proposed target price
            estimated_days: Estimated time to reach target
            atr: Average True Range (Volatility)
            market_regime: 'bull', 'bear', or 'sideways'
            
        Returns:
            Dict with calibrated target and statistical metrics
        """
        if current_price <= 0 or estimated_days <= 0 or atr <= 0:
            return {'target_price': target_price, 'is_realistic': True, 'z_score': 0}
            
        # 1. Calculate Estimated Daily Volatility (Standard Deviation Proxy)
        # ATR is a range. Standard deviation of returns (sigma) is typically ~80% of ATR/Price ratio
        sigma_daily = (atr * 0.8) / current_price
        
        # 2. Square Root of Time rule for multi-day volatility (Sigma_T = Sigma * sqrt(T))
        # This handles the fact that price dispersion increases with the root of time
        sigma_t = sigma_daily * np.sqrt(estimated_days)
        
        # 3. Calculate Required Move as a Z-Score
        actual_move_pct = abs(target_price - current_price) / current_price
        z_score = actual_move_pct / sigma_t if sigma_t > 0 else 0
        
        # 4. Determine Dynamic Threshold based on Regime
        # Bull markets allow for higher momentum (2.8 sigma)
        # Bear/Sideways markets should be more conservative (1.8 sigma)
        threshold = 2.2 
        if market_regime == 'bull':
            threshold = 2.8
        elif market_regime == 'bear':
            threshold = 1.8
            
        calibrated_target = target_price
        is_realistic = True
        reason = ""
        
        # 5. Calibration Logic
        if z_score > threshold:
            # Over-extended target (Statistically improbable for the window)
            # Cap it at the threshold sigma
            max_realistic_move = current_price * (threshold * sigma_t)
            
            if target_price > current_price:
                calibrated_target = current_price + max_realistic_move
            else:
                calibrated_target = current_price - max_realistic_move
            
            is_realistic = False
            reason = f"Target was statistically extreme (Z={z_score:.1f}). Capped at {threshold} Sigma for {estimated_days}d."
            
        return {
            'target_price': float(calibrated_target),
            'z_score': float(z_score),
            'is_realistic': is_realistic,
            'reason': reason,
            'implied_sigma_t': float(sigma_t)
        }

