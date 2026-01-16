"""
Momentum and Trend Reversal Monitoring System
Detects changes in momentum and trend reversals for stocks
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MomentumMonitor:
    """Monitors stocks for momentum changes and trend reversals"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.momentum_file = self.data_dir / "momentum_states.json"
        self.history_file = self.data_dir / "momentum_history.json"
        self.momentum_states = self._load_states()
        self.history = self._load_history()
    
    def _load_states(self) -> Dict:
        """Load momentum states from file"""
        if self.momentum_file.exists():
            try:
                with open(self.momentum_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading momentum states: {e}")
        return {}
    
    def save(self):
        """Save momentum states to file"""
        try:
            with open(self.momentum_file, 'w') as f:
                json.dump(self.momentum_states, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving momentum states: {e}")
            
    def _load_history(self) -> List:
        """Load momentum change history from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading momentum history: {e}")
        return []
        
    def save_history(self):
        """Save momentum change history to file"""
        try:
            # Keep only last 100 entries to prevent file bloat
            if len(self.history) > 100:
                self.history = self.history[-100:]
                
            with open(self.history_file, 'w') as f:
                json.dump(self.history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving momentum history: {e}")
            
    def add_to_history(self, symbol: str, action: str, changes: dict, strategy: str = 'mixed', current_price: float = None):
        """Add a momentum change event to history"""
        
        # Check for duplicates (same symbol, same changes, same recommendation)
        # This prevents spamming the history with identical updates if the analyzer runs frequently
        for i in range(len(self.history) - 1, -1, -1):
            entry = self.history[i]
            if entry['symbol'] == symbol.upper():
                # Found last entry for this symbol
                last_changes = entry.get('changes', {})
                last_changes_list = last_changes.get('changes', [])
                new_changes_list = changes.get('changes', [])
                
                last_reco_action = last_changes.get('recommendation', {}).get('action')
                new_reco_action = changes.get('recommendation', {}).get('action')
                
                # Check if identical content
                if (last_changes_list == new_changes_list and 
                    last_reco_action == new_reco_action):
                    logger.debug(f"Skipping duplicate momentum history for {symbol}")
                    return
                break
        
        entry = {
            'symbol': symbol.upper(),
            'action': action,  # Previous action or signal
            'changes': changes,
            'strategy': strategy,
            'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'price_at_change': current_price,
            'status': 'pending',  # pending, correct, incorrect
            'verified': False
        }
        self.history.append(entry)
        self.save_history()

    def verify_history(self, data_fetcher) -> List[Dict]:
        """Verify past momentum changes"""
        verified_items = []
        now = datetime.now()
        
        for entry in self.history:
            if entry.get('verified') or entry.get('status') != 'pending':
                continue
                
            # Parse timestamp
            try:
                entry_time = datetime.strptime(entry.get('time', ''), "%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue
                
            # Verify after 24 hours
            if (now - entry_time).total_seconds() >= 24 * 3600:
                symbol = entry.get('symbol')
                price_at_change = entry.get('price_at_change')
                
                if not price_at_change:
                    # Can't verify without original price
                    entry['status'] = 'expired'
                    entry['verified'] = True
                    continue
                
                # Get current price
                try:
                    current_data = data_fetcher.fetch_stock_data(symbol)
                    current_price = current_data.get('price', 0)
                    
                    if current_price > 0:
                        recommendation = entry.get('changes', {}).get('recommendation', {})
                        action = recommendation.get('action', 'HOLD')
                        
                        entry['verification_price'] = current_price
                        entry['verification_time'] = now.strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Determine correctness
                        is_correct = False
                        
                        if action == 'BUY' or action == 'STRONG_BUY':
                            # Correct if price went up
                            if current_price > price_at_change:
                                is_correct = True
                        elif action == 'SELL' or action == 'STRONG_SELL':
                            # Correct if price went down (assuming looking for lower re-entry or taking profit)
                            # NOTE: Context matters, but generally SELL means price expected to drop or stay flat
                            if current_price < price_at_change:
                                is_correct = True
                        else:
                            # HOLD - Neutral, hard to verify automatically without more context
                            # For now, mark as expired/neutral
                            entry['status'] = 'neutral'
                            entry['verified'] = True
                            continue
                            
                        entry['status'] = 'correct' if is_correct else 'incorrect'
                        entry['verified'] = True
                        
                        verified_items.append(entry)
                        
                except Exception as e:
                    logger.error(f"Error verifying momentum history for {symbol}: {e}")
        
        if verified_items:
            self.save_history()
            
        return verified_items
        
    def clear_history(self):
        """Clear momentum history"""
        self.history = []
        if self.history_file.exists():
            try:
                self.history_file.unlink()
            except Exception as e:
                logger.error(f"Error deleting history file: {e}")
        self.save_history()
    
    def _get_stock_key(self, symbol: str) -> str:
        """Get normalized stock key"""
        return symbol.upper()
    
    def update_momentum_state(self, symbol: str, indicators: dict, price_action: dict, 
                             momentum_analysis: dict) -> Dict:
        """Update momentum state for a stock and detect changes
        
        Args:
            symbol: Stock symbol
            indicators: Current technical indicators
            price_action: Current price action analysis
            momentum_analysis: Current momentum analysis
            
        Returns:
            Dict with detected changes:
            {
                'momentum_changed': bool,
                'trend_reversed': bool,
                'changes': List[str],  # List of change descriptions
                'previous_state': dict,
                'current_state': dict
            }
        """
        symbol_key = self._get_stock_key(symbol)
        
        # Extract current state
        current_state = {
            'rsi': indicators.get('rsi', 50),
            'macd': indicators.get('macd', 0),
            'macd_signal': indicators.get('macd_signal', 0),
            'macd_diff': indicators.get('macd_diff', 0),
            'mfi': indicators.get('mfi', 50),
            'trend': price_action.get('trend', 'sideways'),
            'momentum_score': momentum_analysis.get('momentum_score', 0),
            'golden_cross': indicators.get('golden_cross', False),
            'death_cross': indicators.get('death_cross', False),
            'price_above_ema20': indicators.get('price_above_ema20', False),
            'price_above_ema50': indicators.get('price_above_ema50', False),
            'price_above_ema200': indicators.get('price_above_ema200', False),
            'timestamp': datetime.now().isoformat()
        }
        
        # Get previous state
        previous_state = self.momentum_states.get(symbol_key, {})
        
        # Detect changes
        changes = []
        momentum_changed = False
        trend_reversed = False
        
        if not previous_state:
            # First time tracking this stock
            self.momentum_states[symbol_key] = current_state
            self.save()
            return {
                'momentum_changed': False,
                'trend_reversed': False,
                'changes': [],
                'previous_state': {},
                'current_state': current_state
            }
        
        # Check for momentum changes
        prev_rsi = previous_state.get('rsi', 50)
        curr_rsi = current_state['rsi']
        
        # RSI momentum shift (crossing key levels)
        if prev_rsi < 30 and curr_rsi > 40:
            changes.append("📈 RSI recovered from oversold (momentum shift bullish)")
            momentum_changed = True
        elif prev_rsi > 70 and curr_rsi < 60:
            changes.append("📉 RSI dropped from overbought (momentum shift bearish)")
            momentum_changed = True
        elif (prev_rsi < 50 and curr_rsi > 50) or (prev_rsi > 50 and curr_rsi < 50):
            changes.append(f"🔄 RSI crossed 50 level ({prev_rsi:.1f} → {curr_rsi:.1f})")
            momentum_changed = True
        
        # MACD momentum change
        prev_macd_diff = previous_state.get('macd_diff', 0)
        curr_macd_diff = current_state['macd_diff']
        
        if prev_macd_diff <= 0 and curr_macd_diff > 0:
            changes.append("📈 MACD turned bullish (momentum shift)")
            momentum_changed = True
        elif prev_macd_diff >= 0 and curr_macd_diff < 0:
            changes.append("📉 MACD turned bearish (momentum shift)")
            momentum_changed = True
        
        # MFI momentum change
        prev_mfi = previous_state.get('mfi', 50)
        curr_mfi = current_state['mfi']
        
        if prev_mfi < 20 and curr_mfi > 30:
            changes.append("📈 MFI recovered from oversold (momentum shift bullish)")
            momentum_changed = True
        elif prev_mfi > 80 and curr_mfi < 70:
            changes.append("📉 MFI dropped from overbought (momentum shift bearish)")
            momentum_changed = True
        elif (prev_mfi < 50 and curr_mfi > 50) or (prev_mfi > 50 and curr_mfi < 50):
            changes.append(f"🔄 MFI crossed 50 level ({prev_mfi:.1f} → {curr_mfi:.1f})")
            momentum_changed = True
        
        # Momentum score change (significant shift)
        prev_momentum = previous_state.get('momentum_score', 0)
        curr_momentum = current_state['momentum_score']
        
        if abs(curr_momentum - prev_momentum) >= 3:
            direction = "bullish" if curr_momentum > prev_momentum else "bearish"
            changes.append(f"🔄 Momentum score shifted {direction} ({prev_momentum} → {curr_momentum})")
            momentum_changed = True
        
        # Check for trend reversals
        prev_trend = previous_state.get('trend', 'sideways')
        curr_trend = current_state['trend']
        
        if prev_trend != curr_trend:
            if (prev_trend == 'uptrend' and curr_trend == 'downtrend') or \
               (prev_trend == 'downtrend' and curr_trend == 'uptrend'):
                changes.append(f"⚠️ TREND REVERSAL: {prev_trend} → {curr_trend}")
                trend_reversed = True
            else:
                changes.append(f"🔄 Trend changed: {prev_trend} → {curr_trend}")
                momentum_changed = True
        
        # Golden Cross / Death Cross reversals
        prev_golden = previous_state.get('golden_cross', False)
        prev_death = previous_state.get('death_cross', False)
        curr_golden = current_state['golden_cross']
        curr_death = current_state['death_cross']
        
        if prev_death and curr_golden:
            changes.append("🟢 GOLDEN CROSS detected (major trend reversal to bullish)")
            trend_reversed = True
        elif prev_golden and curr_death:
            changes.append("🔴 DEATH CROSS detected (major trend reversal to bearish)")
            trend_reversed = True
        
        # EMA position reversals
        prev_above_20 = previous_state.get('price_above_ema20', False)
        prev_above_50 = previous_state.get('price_above_ema50', False)
        prev_above_200 = previous_state.get('price_above_ema200', False)
        
        curr_above_20 = current_state['price_above_ema20']
        curr_above_50 = current_state['price_above_ema50']
        curr_above_200 = current_state['price_above_ema200']
        
        # Price crossing key EMAs
        if prev_above_20 != curr_above_20:
            direction = "above" if curr_above_20 else "below"
            changes.append(f"🔄 Price crossed EMA 20 ({direction})")
            momentum_changed = True
        
        if prev_above_50 != curr_above_50:
            direction = "above" if curr_above_50 else "below"
            changes.append(f"🔄 Price crossed EMA 50 ({direction})")
            momentum_changed = True
        
        if prev_above_200 != curr_above_200:
            direction = "above" if curr_above_200 else "below"
            changes.append(f"⚠️ Price crossed EMA 200 ({direction}) - major trend change")
            trend_reversed = True
        
        # Determine recommendation based on trend and momentum changes
        recommendation = self._determine_recommendation(
            previous_state, current_state, trend_reversed, momentum_changed
        )
        
        # Update stored state
        self.momentum_states[symbol_key] = current_state
        self.save()
        
        return {
            'momentum_changed': momentum_changed,
            'trend_reversed': trend_reversed,
            'changes': changes,
            'previous_state': previous_state,
            'current_state': current_state,
            'recommendation': recommendation,
            'trend_change': self._get_trend_change_description(previous_state, current_state)
        }
    
    def _get_trend_change_description(self, previous_state: dict, current_state: dict) -> str:
        """Get human-readable description of trend change"""
        if not previous_state:
            return None
        
        prev_trend = previous_state.get('trend', 'sideways')
        curr_trend = current_state.get('trend', 'sideways')
        
        if prev_trend != curr_trend:
            return f"{prev_trend.title()} → {curr_trend.title()}"
        return None
    
    def _determine_recommendation(self, previous_state: dict, current_state: dict, 
                                  trend_reversed: bool, momentum_changed: bool) -> dict:
        """Determine recommendation based on trend and momentum changes (STANDARD LOGIC)"""
        if not previous_state:
            return {'action': 'HOLD', 'confidence': 50, 'reason': 'Initial state'}
        
        curr_trend = current_state.get('trend', 'sideways')
        prev_trend = previous_state.get('trend', 'sideways')
        curr_rsi = current_state.get('rsi', 50)
        curr_mfi = current_state.get('mfi', 50)
        curr_macd_diff = current_state.get('macd_diff', 0)
        curr_momentum = current_state.get('momentum_score', 0)
        golden_cross = current_state.get('golden_cross', False)
        death_cross = current_state.get('death_cross', False)
        
        action = 'HOLD'
        confidence = 50
        reasons = []
        
        # STANDARD LOGIC:
        # Bullish (Stock rising/recovering) -> BUY (Follow Trend / Dip Buy)
        # Bearish (Stock falling/overbought) -> SELL (Take Profit / Short)
        
        # Trend reversal analysis
        if trend_reversed:
            if prev_trend == 'downtrend' and curr_trend == 'uptrend':
                action = 'BUY'  # Bullish reversal
                confidence = 70
                reasons.append("Trend reversed from downtrend to uptrend (Bullish)")
            elif prev_trend == 'uptrend' and curr_trend == 'downtrend':
                action = 'SELL'  # Bearish reversal
                confidence = 70
                reasons.append("Trend reversed from uptrend to downtrend (Bearish)")
        
        # Golden Cross / Death Cross
        if golden_cross and not previous_state.get('golden_cross', False):
            action = 'BUY'  # Bullish
            confidence = max(confidence, 75)
            reasons.append("Golden Cross detected - strong bullish signal")
        elif death_cross and not previous_state.get('death_cross', False):
            action = 'SELL'  # Bearish
            confidence = max(confidence, 75)
            reasons.append("Death Cross detected - strong bearish signal")
        
        # RSI analysis
        if curr_rsi < 30:
            if action == 'HOLD':
                action = 'BUY'  # Bullish (Oversold bounce)
                confidence = 65
            reasons.append(f"RSI oversold ({curr_rsi:.1f}) - potential recovery")
        elif curr_rsi > 70:
            if action == 'HOLD':
                action = 'SELL'  # Bearish (Overbought pullback)
                confidence = 65
            reasons.append(f"RSI overbought ({curr_rsi:.1f}) - potential pullback")
        
        # MFI analysis
        if curr_mfi < 20:
            if action == 'HOLD':
                action = 'BUY'  # Bullish
                confidence = max(confidence, 60)
            reasons.append(f"MFI oversold ({curr_mfi:.1f}) - buying pressure")
        elif curr_mfi > 80:
            if action == 'HOLD':
                action = 'SELL'  # Bearish
                confidence = max(confidence, 60)
            reasons.append(f"MFI overbought ({curr_mfi:.1f}) - selling pressure")
        
        # MACD analysis
        if curr_macd_diff > 0:
            if action == 'HOLD' and curr_momentum > 0:
                action = 'BUY'  # Bullish
                confidence = max(confidence, 55)
            reasons.append("MACD bullish - upward momentum")
        elif curr_macd_diff < 0:
            if action == 'HOLD' and curr_momentum < 0:
                action = 'SELL'  # Bearish
                confidence = max(confidence, 55)
            reasons.append("MACD bearish - downward momentum")
        
        # Momentum score
        if curr_momentum >= 3:
            if action == 'HOLD':
                action = 'BUY'  # Bullish
                confidence = max(confidence, 60)
            reasons.append("Strong bullish momentum")
        elif curr_momentum <= -3:
            if action == 'HOLD':
                action = 'SELL'  # Bearish
                confidence = max(confidence, 60)
            reasons.append("Strong bearish momentum")
        
        # If conflicting signals, reduce confidence
        if (action == 'BUY' and (curr_rsi > 70 or curr_mfi > 80)) or \
           (action == 'SELL' and (curr_rsi < 30 or curr_mfi < 20)):
            confidence = max(50, confidence - 15)
            reasons.append("⚠️ Conflicting signals - reduced confidence")
        
        return {
            'action': action,
            'confidence': min(90, confidence),
            'reasons': reasons[:3]  # Top 3 reasons
        }
    
    def get_stock_state(self, symbol: str) -> Optional[Dict]:
        """Get current momentum state for a stock"""
        return self.momentum_states.get(self._get_stock_key(symbol))
    
    def remove_stock(self, symbol: str):
        """Remove a stock from momentum monitoring"""
        symbol_key = self._get_stock_key(symbol)
        if symbol_key in self.momentum_states:
            del self.momentum_states[symbol_key]
            self.save()
            logger.debug(f"Removed momentum monitoring for: {symbol_key}")
    
    def detect_peaks_bottoms(self, symbol: str, price_history: list, indicators: dict) -> Dict:
        """Detect if price is at a peak or bottom
        
        Args:
            symbol: Stock symbol
            price_history: List of price data points [{'date': str, 'close': float, 'high': float, 'low': float, 'volume': float}, ...]
            indicators: Current technical indicators
            
        Returns:
            Dict with peak/bottom detection:
            {
                'peak_detected': bool,
                'bottom_detected': bool,
                'confidence': float,
                'reasoning': str,
                'current_price': float,
                'peak_price': float (if peak detected),
                'bottom_price': float (if bottom detected)
            }
        """
        import pandas as pd
        import numpy as np
        
        if not price_history or len(price_history) < 20:
            return {'peak_detected': False, 'bottom_detected': False, 'confidence': 0.0}
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(price_history)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            df = df.sort_index()
            
            # Ensure required columns exist
            if 'close' not in df.columns or 'high' not in df.columns or 'low' not in df.columns:
                return {'peak_detected': False, 'bottom_detected': False, 'confidence': 0.0}
            
            current_price = float(df['close'].iloc[-1])
            current_high = float(df['high'].iloc[-1])
            current_low = float(df['low'].iloc[-1])
            
            # Recent price window (last 10-20 days)
            recent_window = min(20, len(df))
            recent_prices = df['close'].tail(recent_window)
            recent_highs = df['high'].tail(recent_window)
            recent_lows = df['low'].tail(recent_window)
            
            peak_detected = False
            bottom_detected = False
            confidence = 0.0
            peak_reasoning_parts = []
            bottom_reasoning_parts = []
            
            # 1. Detect actual local minima/maxima (price reversals)
            # Find the actual lowest/highest points in recent history
            max_high = float(recent_highs.max())
            min_low = float(recent_lows.min())
            
            # Find when the min/max occurred
            max_high_idx = recent_highs.idxmax() if len(recent_highs) > 0 else None
            min_low_idx = recent_lows.idxmin() if len(recent_lows) > 0 else None
            
            # Check if we're at or very close to the actual peak/bottom
            # A peak is detected if current price is near the max AND indicators suggest reversal
            # A bottom is detected if current price is near the min AND indicators suggest reversal
            is_at_peak = current_high >= max_high * 0.98  # Within 2% of peak
            is_at_bottom = current_low <= min_low * 1.02  # Within 2% of bottom
            
            # Additional check: if price has moved away from the min/max, we detected it late
            # This means the actual bottom/peak was in the recent past and we're seeing it now
            price_moved_from_bottom = current_price > min_low * 1.03  # Price is 3%+ above the low
            price_moved_from_peak = current_price < max_high * 0.97  # Price is 3%+ below the high
            
            # Get indicators once
            rsi = indicators.get('rsi', 50)
            mfi = indicators.get('mfi', 50)
            macd_diff = indicators.get('macd_diff', 0)
            
            # Peak detection signals
            peak_signals = 0
            if is_at_peak:
                peak_signals += 1
                peak_reasoning_parts.append(f"Price reached 20-day peak at ${max_high:.2f}")
            elif price_moved_from_peak:
                # Price was at peak recently and has moved down (reversal detected)
                # This handles cases where we detect the peak after price has already started falling
                peak_signals += 1
                price_decrease_pct = ((max_high - current_price) / max_high) * 100
                if price_decrease_pct >= 5.0:
                    peak_reasoning_parts.append(f"Peak detected at ${max_high:.2f}, price now reversing down ({price_decrease_pct:.1f}% decline)")
                else:
                    peak_reasoning_parts.append(f"Peak detected at ${max_high:.2f}, price now reversing down ({price_decrease_pct:.1f}% below peak)")
            
            # Check RSI for overbought (peak)
            if rsi >= 70:
                peak_signals += 2  # Strong signal
                peak_reasoning_parts.append(f"RSI overbought ({rsi:.1f})")
            elif rsi >= 65:
                peak_signals += 1
                peak_reasoning_parts.append(f"RSI approaching overbought ({rsi:.1f})")
            
            # Check MFI for overbought (peak)
            if mfi >= 80:
                peak_signals += 2
                peak_reasoning_parts.append(f"MFI overbought ({mfi:.1f})")
            elif mfi >= 75:
                peak_signals += 1
                peak_reasoning_parts.append(f"MFI approaching overbought ({mfi:.1f})")
            
            # Check MACD divergence (price making higher highs, MACD making lower highs = peak)
            if len(recent_prices) >= 10:
                recent_high_prices = recent_highs.tail(10)
                if len(recent_high_prices) >= 5:
                    price_trend = recent_high_prices.iloc[-1] > recent_high_prices.iloc[-5]
                    # If price is rising but MACD is falling, bearish divergence (peak forming)
                    if price_trend and macd_diff < 0:
                        peak_signals += 2
                        peak_reasoning_parts.append("Bearish MACD divergence detected")
            
            # Bottom detection signals
            bottom_signals = 0
            
            # Check volume confirmation (if available) - do this after bottom_signals is defined
            volume_ratio = 1.0
            if 'volume' in df.columns:
                recent_volumes = df['volume'].tail(recent_window)
                if len(recent_volumes) > 0:
                    avg_volume = recent_volumes.mean()
                    current_volume = recent_volumes.iloc[-1]
                    if avg_volume > 0:
                        volume_ratio = current_volume / avg_volume
                        if volume_ratio > 1.5:
                            if peak_signals > 0:
                                peak_signals += 1
                                peak_reasoning_parts.append("High volume confirms peak")
                            if bottom_signals > 0:
                                bottom_signals += 1
                                bottom_reasoning_parts.append("High volume confirms bottom")
            if is_at_bottom:
                bottom_signals += 1
                bottom_reasoning_parts.append(f"Price reached 20-day bottom at ${min_low:.2f}")
            elif price_moved_from_bottom:
                # Price was at bottom recently and has moved up (reversal detected)
                # This handles cases where we detect the bottom after price has already started rising
                bottom_signals += 1
                price_increase_pct = ((current_price - min_low) / min_low) * 100
                if price_increase_pct >= 5.0:
                    bottom_reasoning_parts.append(f"Bottom detected at ${min_low:.2f}, price now reversing up ({price_increase_pct:.1f}% recovery)")
                else:
                    bottom_reasoning_parts.append(f"Bottom detected at ${min_low:.2f}, price now reversing up ({price_increase_pct:.1f}% above bottom)")
            
            # Check RSI for oversold (bottom)
            if rsi <= 30:
                bottom_signals += 2  # Strong signal
                bottom_reasoning_parts.append(f"RSI oversold ({rsi:.1f})")
            elif rsi <= 35:
                bottom_signals += 1
                bottom_reasoning_parts.append(f"RSI approaching oversold ({rsi:.1f})")
            
            # Check MFI for oversold (bottom)
            if mfi <= 20:
                bottom_signals += 2
                bottom_reasoning_parts.append(f"MFI oversold ({mfi:.1f})")
            elif mfi <= 25:
                bottom_signals += 1
                bottom_reasoning_parts.append(f"MFI approaching oversold ({mfi:.1f})")
            
            # Check MACD divergence (price making lower lows, MACD making higher lows = bottom)
            if len(recent_prices) >= 10:
                recent_low_prices = recent_lows.tail(10)
                if len(recent_low_prices) >= 5:
                    price_trend = recent_low_prices.iloc[-1] < recent_low_prices.iloc[-5]
                    # If price is falling but MACD is rising, bullish divergence (bottom forming)
                    if price_trend and macd_diff > 0:
                        bottom_signals += 2
                        bottom_reasoning_parts.append("Bullish MACD divergence detected")
            
            # Determine if peak/bottom detected (need at least 3 signals)
            peak_detected = peak_signals >= 3
            bottom_detected = bottom_signals >= 3
            
            # FIX: Separate reasoning for peak vs bottom to avoid confusion
            # If both are detected, prioritize the stronger signal
            peak_reasoning_parts = []
            bottom_reasoning_parts = []
            
            # Separate peak reasoning
            if is_at_peak:
                peak_reasoning_parts.append(f"Price reached 20-day peak at ${max_high:.2f}")
            elif price_moved_from_peak:
                price_decrease_pct = ((max_high - current_price) / max_high) * 100
                if price_decrease_pct >= 5.0:
                    peak_reasoning_parts.append(f"Peak detected at ${max_high:.2f}, price now reversing down ({price_decrease_pct:.1f}% decline)")
                else:
                    peak_reasoning_parts.append(f"Peak detected at ${max_high:.2f}, price now reversing down ({price_decrease_pct:.1f}% below peak)")
            
            if rsi >= 70:
                peak_reasoning_parts.append(f"RSI overbought ({rsi:.1f})")
            elif rsi >= 65:
                peak_reasoning_parts.append(f"RSI approaching overbought ({rsi:.1f})")
            
            if mfi >= 80:
                peak_reasoning_parts.append(f"MFI overbought ({mfi:.1f})")
            elif mfi >= 75:
                peak_reasoning_parts.append(f"MFI approaching overbought ({mfi:.1f})")
            
            if len(recent_prices) >= 10:
                recent_high_prices = recent_highs.tail(10)
                if len(recent_high_prices) >= 5:
                    price_trend = recent_high_prices.iloc[-1] > recent_high_prices.iloc[-5]
                    if price_trend and macd_diff < 0:
                        peak_reasoning_parts.append("Bearish MACD divergence detected")
            
            # Separate bottom reasoning
            if is_at_bottom:
                bottom_reasoning_parts.append(f"Price reached 20-day bottom at ${min_low:.2f}")
            elif price_moved_from_bottom:
                price_increase_pct = ((current_price - min_low) / min_low) * 100
                if price_increase_pct >= 5.0:
                    bottom_reasoning_parts.append(f"Bottom detected at ${min_low:.2f}, price now reversing up ({price_increase_pct:.1f}% recovery)")
                else:
                    bottom_reasoning_parts.append(f"Bottom detected at ${min_low:.2f}, price now reversing up ({price_increase_pct:.1f}% above bottom)")
            
            if rsi <= 30:
                bottom_reasoning_parts.append(f"RSI oversold ({rsi:.1f})")
            elif rsi <= 35:
                bottom_reasoning_parts.append(f"RSI approaching oversold ({rsi:.1f})")
            
            if mfi <= 20:
                bottom_reasoning_parts.append(f"MFI oversold ({mfi:.1f})")
            elif mfi <= 25:
                bottom_reasoning_parts.append(f"MFI approaching oversold ({mfi:.1f})")
            
            if len(recent_prices) >= 10:
                recent_low_prices = recent_lows.tail(10)
                if len(recent_low_prices) >= 5:
                    price_trend = recent_low_prices.iloc[-1] < recent_low_prices.iloc[-5]
                    if price_trend and macd_diff > 0:
                        bottom_reasoning_parts.append("Bullish MACD divergence detected")
            
            # Determine final reasoning based on what was detected
            # If both detected, prioritize the stronger signal
            if peak_detected and bottom_detected:
                # Both detected - use the stronger signal
                if peak_signals >= bottom_signals:
                    final_reasoning = '; '.join(peak_reasoning_parts) if peak_reasoning_parts else 'Peak detected with multiple confirmations'
                    confidence = min(90, 50 + (peak_signals * 10))
                    # Only return peak detection
                    bottom_detected = False
                else:
                    final_reasoning = '; '.join(bottom_reasoning_parts) if bottom_reasoning_parts else 'Bottom detected with multiple confirmations'
                    confidence = min(90, 50 + (bottom_signals * 10))
                    # Only return bottom detection
                    peak_detected = False
            elif peak_detected:
                final_reasoning = '; '.join(peak_reasoning_parts) if peak_reasoning_parts else 'Peak detected with multiple confirmations'
                confidence = min(90, 50 + (peak_signals * 10))
            elif bottom_detected:
                final_reasoning = '; '.join(bottom_reasoning_parts) if bottom_reasoning_parts else 'Bottom detected with multiple confirmations'
                confidence = min(90, 50 + (bottom_signals * 10))
            else:
                final_reasoning = 'No significant signals'
                confidence = 0.0
            
            # Return the actual peak/bottom prices (the detected minimum/maximum)
            # These are the prices where reversal occurred, not necessarily current price
            return {
                'peak_detected': peak_detected,
                'bottom_detected': bottom_detected,
                'confidence': confidence,
                'reasoning': final_reasoning,
                'current_price': current_price,
                'peak_price': max_high if peak_detected else None,  # The actual peak price detected
                'bottom_price': min_low if bottom_detected else None,  # The actual bottom price detected
                'reversal_detected': peak_detected or bottom_detected,  # Indicates reversal is incoming
                'signals_count': {'peak': peak_signals, 'bottom': bottom_signals}
            }
        except Exception as e:
            logger.error(f"Error detecting peaks/bottoms for {symbol}: {e}")
            return {'peak_detected': False, 'bottom_detected': False, 'confidence': 0.0, 'reasoning': f'Error: {str(e)}'}

