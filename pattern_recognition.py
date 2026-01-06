"""
Advanced Price Pattern Recognition
Detects chart patterns and candlestick formations for better trading signals
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PatternRecognizer:
    """Recognizes advanced chart patterns and candlestick formations"""
    
    def __init__(self):
        self.min_pattern_length = 20  # Minimum bars needed for pattern detection
    
    def detect_all_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Detect all chart patterns and candlestick formations
        
        Args:
            df: DataFrame with 'open', 'high', 'low', 'close', 'volume' columns
            
        Returns:
            Dict with detected patterns and their signals
        """
        if len(df) < self.min_pattern_length:
            return {'patterns': [], 'primary_pattern': None, 'pattern_signal': 'neutral'}
        
        patterns = []
        
        # 1. Advanced Candlestick Patterns
        candlestick_patterns = self._detect_advanced_candlestick_patterns(df)
        patterns.extend(candlestick_patterns)
        
        # 2. Chart Patterns
        chart_patterns = self._detect_chart_patterns(df)
        patterns.extend(chart_patterns)
        
        # 3. Determine primary pattern and signal
        primary_pattern, signal = self._get_primary_pattern_signal(patterns)
        
        return {
            'patterns': patterns,
            'primary_pattern': primary_pattern,
            'pattern_signal': signal,
            'pattern_count': len(patterns),
            'bullish_patterns': len([p for p in patterns if p.get('signal') == 'bullish']),
            'bearish_patterns': len([p for p in patterns if p.get('signal') == 'bearish'])
        }
    
    def _detect_advanced_candlestick_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect advanced candlestick patterns"""
        patterns = []
        
        if len(df) < 3:
            return patterns
        
        # Get recent candles
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else None
        prev2 = df.iloc[-3] if len(df) >= 3 else None
        prev3 = df.iloc[-4] if len(df) >= 4 else None
        
        # Calculate candle components
        def get_candle_info(candle):
            open_p = candle['open']
            close_p = candle['close']
            high_p = candle['high']
            low_p = candle['low']
            body = abs(close_p - open_p)
            upper_shadow = high_p - max(open_p, close_p)
            lower_shadow = min(open_p, close_p) - low_p
            total_range = high_p - low_p
            is_bullish = close_p > open_p
            return {
                'open': open_p, 'close': close_p, 'high': high_p, 'low': low_p,
                'body': body, 'upper_shadow': upper_shadow, 'lower_shadow': lower_shadow,
                'range': total_range, 'is_bullish': is_bullish
            }
        
        last_c = get_candle_info(last)
        prev_c = get_candle_info(prev) if prev is not None else None
        prev2_c = get_candle_info(prev2) if prev2 is not None else None
        prev3_c = get_candle_info(prev3) if prev3 is not None else None
        
        # Basic patterns (already detected, but add more detail)
        if last_c['range'] > 0:
            body_ratio = last_c['body'] / last_c['range']
            
            # Doji variations
            if body_ratio < 0.1:
                patterns.append({
                    'name': 'doji',
                    'type': 'candlestick',
                    'signal': 'neutral',
                    'strength': 0.3,
                    'description': 'Indecision - potential reversal'
                })
            
            # Hammer variations
            if body_ratio < 0.3 and last_c['lower_shadow'] > last_c['body'] * 2 and last_c['upper_shadow'] < last_c['body'] * 0.5:
                patterns.append({
                    'name': 'hammer',
                    'type': 'candlestick',
                    'signal': 'bullish' if not last_c['is_bullish'] else 'bullish',
                    'strength': 0.7,
                    'description': 'Hammer - potential bullish reversal'
                })
            
            # Shooting star
            if body_ratio < 0.3 and last_c['upper_shadow'] > last_c['body'] * 2 and last_c['lower_shadow'] < last_c['body'] * 0.5:
                patterns.append({
                    'name': 'shooting_star',
                    'type': 'candlestick',
                    'signal': 'bearish',
                    'strength': 0.7,
                    'description': 'Shooting star - potential bearish reversal'
                })
        
        # Multi-candle patterns
        if prev_c is not None:
            # Engulfing patterns
            if (not prev_c['is_bullish'] and last_c['is_bullish'] and
                last_c['open'] < prev_c['close'] and last_c['close'] > prev_c['open']):
                patterns.append({
                    'name': 'bullish_engulfing',
                    'type': 'candlestick',
                    'signal': 'bullish',
                    'strength': 0.8,
                    'description': 'Bullish engulfing - strong reversal signal'
                })
            
            if (prev_c['is_bullish'] and not last_c['is_bullish'] and
                last_c['open'] > prev_c['close'] and last_c['close'] < prev_c['open']):
                patterns.append({
                    'name': 'bearish_engulfing',
                    'type': 'candlestick',
                    'signal': 'bearish',
                    'strength': 0.8,
                    'description': 'Bearish engulfing - strong reversal signal'
                })
        
        # Three-candle patterns
        if prev2_c is not None and prev_c is not None:
            # Morning Star (bearish, bearish, bullish)
            if (not prev2_c['is_bullish'] and not prev_c['is_bullish'] and last_c['is_bullish'] and
                prev_c['body'] < prev2_c['body'] * 0.5 and last_c['close'] > (prev2_c['open'] + prev2_c['close']) / 2):
                patterns.append({
                    'name': 'morning_star',
                    'type': 'candlestick',
                    'signal': 'bullish',
                    'strength': 0.9,
                    'description': 'Morning star - very strong bullish reversal'
                })
            
            # Evening Star (bullish, small, bearish)
            if (prev2_c['is_bullish'] and prev_c['body'] < prev2_c['body'] * 0.5 and not last_c['is_bullish'] and
                last_c['close'] < (prev2_c['open'] + prev2_c['close']) / 2):
                patterns.append({
                    'name': 'evening_star',
                    'type': 'candlestick',
                    'signal': 'bearish',
                    'strength': 0.9,
                    'description': 'Evening star - very strong bearish reversal'
                })
            
            # Three White Soldiers (three consecutive bullish candles)
            if (prev2_c['is_bullish'] and prev_c['is_bullish'] and last_c['is_bullish'] and
                last_c['close'] > prev_c['close'] > prev2_c['close']):
                patterns.append({
                    'name': 'three_white_soldiers',
                    'type': 'candlestick',
                    'signal': 'bullish',
                    'strength': 0.8,
                    'description': 'Three white soldiers - strong bullish momentum'
                })
            
            # Three Black Crows (three consecutive bearish candles)
            if (not prev2_c['is_bullish'] and not prev_c['is_bullish'] and not last_c['is_bullish'] and
                last_c['close'] < prev_c['close'] < prev2_c['close']):
                patterns.append({
                    'name': 'three_black_crows',
                    'type': 'candlestick',
                    'signal': 'bearish',
                    'strength': 0.8,
                    'description': 'Three black crows - strong bearish momentum'
                })
        
        return patterns
    
    def _detect_chart_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Detect chart patterns (head & shoulders, triangles, etc.)"""
        patterns = []
        
        if len(df) < 30:
            return patterns
        
        # Use last 60 days for pattern detection
        lookback = min(60, len(df))
        data = df.tail(lookback)
        prices = data['close'].values
        highs = data['high'].values
        lows = data['low'].values
        
        # 1. Head and Shoulders / Inverse Head and Shoulders
        hns_pattern = self._detect_head_and_shoulders(highs, lows, prices)
        if hns_pattern:
            patterns.append(hns_pattern)
        
        # 2. Double Top / Double Bottom
        double_pattern = self._detect_double_top_bottom(highs, lows, prices)
        if double_pattern:
            patterns.append(double_pattern)
        
        # 3. Triangles
        triangle_pattern = self._detect_triangles(highs, lows, prices)
        if triangle_pattern:
            patterns.append(triangle_pattern)
        
        # 4. Flags and Pennants
        flag_pattern = self._detect_flags_pennants(highs, lows, prices, data['volume'].values)
        if flag_pattern:
            patterns.append(flag_pattern)
        
        # 5. Cup and Handle
        cup_pattern = self._detect_cup_and_handle(highs, lows, prices)
        if cup_pattern:
            patterns.append(cup_pattern)
        
        return patterns
    
    def _detect_head_and_shoulders(self, highs: np.ndarray, lows: np.ndarray, prices: np.ndarray) -> Optional[Dict]:
        """Detect Head and Shoulders pattern"""
        if len(highs) < 20:
            return None
        
        # Find local peaks
        peaks = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peaks.append((i, highs[i]))
        
        if len(peaks) < 3:
            return None
        
        # Get last 3 peaks
        recent_peaks = sorted(peaks[-3:], key=lambda x: x[0])
        
        # Check for Head and Shoulders (left shoulder, head, right shoulder)
        ls_idx, ls_price = recent_peaks[0]
        head_idx, head_price = recent_peaks[1]
        rs_idx, rs_price = recent_peaks[2]
        
        # Head should be highest, shoulders similar height
        if (head_price > ls_price * 1.02 and head_price > rs_price * 1.02 and
            abs(ls_price - rs_price) / max(ls_price, rs_price) < 0.05):
            
            # Check if right shoulder is declining (bearish signal)
            if rs_price < head_price:
                return {
                    'name': 'head_and_shoulders',
                    'type': 'chart',
                    'signal': 'bearish',
                    'strength': 0.85,
                    'description': 'Head and Shoulders - bearish reversal pattern'
                }
        
        # Check for Inverse Head and Shoulders (bottom pattern)
        # Find local troughs
        troughs = []
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                troughs.append((i, lows[i]))
        
        if len(troughs) < 3:
            return None
        
        recent_troughs = sorted(troughs[-3:], key=lambda x: x[0])
        
        if len(recent_troughs) >= 3:
            lt_idx, lt_price = recent_troughs[0]
            head_idx, head_price = recent_troughs[1]
            rt_idx, rt_price = recent_troughs[2]
            
            # Head should be lowest, shoulders similar height
            if (head_price < lt_price * 0.98 and head_price < rt_price * 0.98 and
                abs(lt_price - rt_price) / max(lt_price, rt_price) < 0.05):
                
                # Check if right shoulder is rising (bullish signal)
                if rt_price > head_price:
                    return {
                        'name': 'inverse_head_and_shoulders',
                        'type': 'chart',
                        'signal': 'bullish',
                        'strength': 0.85,
                        'description': 'Inverse Head and Shoulders - bullish reversal pattern'
                    }
        
        return None
    
    def _detect_double_top_bottom(self, highs: np.ndarray, lows: np.ndarray, prices: np.ndarray) -> Optional[Dict]:
        """Detect Double Top and Double Bottom patterns"""
        if len(highs) < 20:
            return None
        
        # Find local peaks
        peaks = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peaks.append((i, highs[i]))
        
        # Find local troughs
        troughs = []
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                troughs.append((i, lows[i]))
        
        # Double Top (bearish)
        if len(peaks) >= 2:
            recent_peaks = sorted(peaks[-2:], key=lambda x: x[0])
            p1_idx, p1_price = recent_peaks[0]
            p2_idx, p2_price = recent_peaks[1]
            
            # Peaks should be similar height and separated by a trough
            price_diff = abs(p1_price - p2_price) / max(p1_price, p2_price)
            if price_diff < 0.03 and p2_idx > p1_idx + 5:  # At least 5 bars apart
                # Check if second peak failed to break first (bearish)
                if p2_price <= p1_price * 1.01:
                    return {
                        'name': 'double_top',
                        'type': 'chart',
                        'signal': 'bearish',
                        'strength': 0.75,
                        'description': 'Double Top - bearish reversal pattern'
                    }
        
        # Double Bottom (bullish)
        if len(troughs) >= 2:
            recent_troughs = sorted(troughs[-2:], key=lambda x: x[0])
            t1_idx, t1_price = recent_troughs[0]
            t2_idx, t2_price = recent_troughs[1]
            
            # Troughs should be similar depth and separated by a peak
            price_diff = abs(t1_price - t2_price) / max(t1_price, t2_price)
            if price_diff < 0.03 and t2_idx > t1_idx + 5:
                # Check if second trough held support (bullish)
                if t2_price >= t1_price * 0.99:
                    return {
                        'name': 'double_bottom',
                        'type': 'chart',
                        'signal': 'bullish',
                        'strength': 0.75,
                        'description': 'Double Bottom - bullish reversal pattern'
                    }
        
        return None
    
    def _detect_triangles(self, highs: np.ndarray, lows: np.ndarray, prices: np.ndarray) -> Optional[Dict]:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        if len(highs) < 20:
            return None
        
        # Use last 30 bars for triangle detection
        lookback = min(30, len(highs))
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        
        # Calculate trend lines
        x = np.arange(lookback)
        
        # Upper trend line (resistance)
        high_slope = np.polyfit(x, recent_highs, 1)[0]
        
        # Lower trend line (support)
        low_slope = np.polyfit(x, recent_lows, 1)[0]
        
        # Ascending Triangle: horizontal resistance, rising support
        if abs(high_slope) < 0.01 and low_slope > 0.01:
            return {
                'name': 'ascending_triangle',
                'type': 'chart',
                'signal': 'bullish',
                'strength': 0.7,
                'description': 'Ascending Triangle - bullish continuation pattern'
            }
        
        # Descending Triangle: falling resistance, horizontal support
        if high_slope < -0.01 and abs(low_slope) < 0.01:
            return {
                'name': 'descending_triangle',
                'type': 'chart',
                'signal': 'bearish',
                'strength': 0.7,
                'description': 'Descending Triangle - bearish continuation pattern'
            }
        
        # Symmetrical Triangle: converging trend lines
        if abs(high_slope) > 0.01 and abs(low_slope) > 0.01:
            # Check if they're converging
            high_intercept = np.polyfit(x, recent_highs, 1)[1]
            low_intercept = np.polyfit(x, recent_lows, 1)[1]
            
            # Calculate where lines would meet
            if high_slope != low_slope:
                convergence_point = (low_intercept - high_intercept) / (high_slope - low_slope)
                
                # If convergence is within reasonable range (future bars)
                if 0 < convergence_point < lookback * 2:
                    # Determine direction based on recent price action
                    recent_trend = 'bullish' if prices[-1] > prices[-lookback//2] else 'bearish'
                    return {
                        'name': 'symmetrical_triangle',
                        'type': 'chart',
                        'signal': recent_trend,
                        'strength': 0.6,
                        'description': f'Symmetrical Triangle - {recent_trend} continuation pattern'
                    }
        
        return None
    
    def _detect_flags_pennants(self, highs: np.ndarray, lows: np.ndarray, prices: np.ndarray, volumes: np.ndarray) -> Optional[Dict]:
        """Detect Flag and Pennant patterns"""
        if len(highs) < 15:
            return None
        
        # Flag/Pennant: strong move followed by consolidation
        # Use last 20 bars
        lookback = min(20, len(highs))
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        recent_prices = prices[-lookback:]
        recent_volumes = volumes[-lookback:] if len(volumes) >= lookback else None
        
        # Check for strong initial move (first 5 bars)
        initial_move = (recent_prices[4] - recent_prices[0]) / recent_prices[0]
        
        if abs(initial_move) < 0.03:  # Need at least 3% move
            return None
        
        # Check for consolidation (last 10-15 bars)
        consolidation_highs = recent_highs[-10:]
        consolidation_lows = recent_lows[-10:]
        
        consolidation_range = (max(consolidation_highs) - min(consolidation_lows)) / recent_prices[-10]
        
        # Consolidation should be tight (less than 5% range)
        if consolidation_range > 0.05:
            return None
        
        # Volume should decrease during consolidation
        if recent_volumes is not None and len(recent_volumes) >= 15:
            initial_vol = np.mean(recent_volumes[:5])
            consolidation_vol = np.mean(recent_volumes[-10:])
            
            if consolidation_vol > initial_vol * 1.2:  # Volume shouldn't increase
                return None
        
        # Determine direction
        signal = 'bullish' if initial_move > 0 else 'bearish'
        pattern_name = 'bullish_flag' if signal == 'bullish' else 'bearish_flag'
        
        return {
            'name': pattern_name,
            'type': 'chart',
            'signal': signal,
            'strength': 0.7,
            'description': f'{pattern_name.replace("_", " ").title()} - {signal} continuation pattern'
        }
    
    def _detect_cup_and_handle(self, highs: np.ndarray, lows: np.ndarray, prices: np.ndarray) -> Optional[Dict]:
        """Detect Cup and Handle pattern"""
        if len(highs) < 40:
            return None
        
        # Cup and Handle: U-shaped cup followed by small handle
        # Use last 50 bars
        lookback = min(50, len(highs))
        recent_prices = prices[-lookback:]
        
        # Find the cup (U-shape in first 30-40 bars)
        cup_prices = recent_prices[:40]
        cup_start = cup_prices[0]
        cup_mid = cup_prices[len(cup_prices)//2]
        cup_end = cup_prices[-1]
        
        # Cup should have a dip (mid lower than start/end)
        cup_depth = (cup_start + cup_end) / 2 - cup_mid
        
        if cup_depth < cup_start * 0.05:  # Need at least 5% dip
            return None
        
        # Handle should be small consolidation (last 10 bars)
        handle_prices = recent_prices[-10:]
        handle_range = (max(handle_prices) - min(handle_prices)) / recent_prices[-10]
        
        if handle_range > 0.05:  # Handle should be tight
            return None
        
        # Price should be recovering (end higher than start)
        if cup_end > cup_start * 0.98:
            return {
                'name': 'cup_and_handle',
                'type': 'chart',
                'signal': 'bullish',
                'strength': 0.8,
                'description': 'Cup and Handle - bullish continuation pattern'
            }
        
        return None
    
    def _get_primary_pattern_signal(self, patterns: List[Dict]) -> Tuple[Optional[str], str]:
        """Determine primary pattern and overall signal"""
        if not patterns:
            return None, 'neutral'
        
        # Sort by strength
        sorted_patterns = sorted(patterns, key=lambda x: x.get('strength', 0), reverse=True)
        primary = sorted_patterns[0]
        
        # Calculate weighted signal
        bullish_score = sum(p.get('strength', 0) for p in patterns if p.get('signal') == 'bullish')
        bearish_score = sum(p.get('strength', 0) for p in patterns if p.get('signal') == 'bearish')
        
        if bullish_score > bearish_score * 1.2:
            signal = 'bullish'
        elif bearish_score > bullish_score * 1.2:
            signal = 'bearish'
        else:
            signal = 'neutral'
        
        return primary.get('name'), signal

