"""
Momentum and Trend Reversal Monitoring System
Detects changes in momentum and trend reversals for stocks.

Key design principles:
- Only fire on MEANINGFUL momentum changes (not noise like RSI crossing 50)
- Require multi-signal confirmation for recommendations
- Cooldown per symbol to prevent duplicate alerts
- Verification uses minimum time window and price threshold
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MomentumMonitor:
    """Monitors stocks for momentum changes and trend reversals"""
    
    # Cooldown: don't fire same type of change for same stock within this many hours
    COOLDOWN_HOURS = 24
    
    # Verification settings
    VERIFY_MIN_HOURS = 72      # Wait at least 3 days before verifying
    VERIFY_MIN_PRICE_PCT = 2.0  # Minimum 2% price move to count as correct
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.momentum_file = self.data_dir / "momentum_states.json"
        self.history_file = self.data_dir / "momentum_history.json"
        self.cooldown_file = self.data_dir / "momentum_cooldowns.json"
        self.momentum_states = self._load_states()
        self.history = self._load_history()
        self.cooldowns = self._load_cooldowns()
    
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
    
    def _load_cooldowns(self) -> Dict:
        """Load cooldown timestamps per symbol+change_type"""
        if self.cooldown_file.exists():
            try:
                with open(self.cooldown_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading cooldowns: {e}")
        return {}
    
    def _save_cooldowns(self):
        """Save cooldown timestamps"""
        try:
            with open(self.cooldown_file, 'w') as f:
                json.dump(self.cooldowns, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cooldowns: {e}")
    
    def _is_on_cooldown(self, symbol: str, change_type: str) -> bool:
        """Check if a symbol+change_type is on cooldown"""
        key = f"{symbol.upper()}:{change_type}"
        last_fire = self.cooldowns.get(key)
        if not last_fire:
            return False
        try:
            last_time = datetime.fromisoformat(last_fire)
            return (datetime.now() - last_time).total_seconds() < self.COOLDOWN_HOURS * 3600
        except:
            return False
    
    def _set_cooldown(self, symbol: str, change_type: str):
        """Set cooldown for a symbol+change_type"""
        key = f"{symbol.upper()}:{change_type}"
        self.cooldowns[key] = datetime.now().isoformat()
        self._save_cooldowns()
        
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
            
    def add_to_history(self, symbol: str, action: str, changes: dict, strategy: str = 'mixed', current_price: float = None, current_date=None):
        """Add a momentum change event to history"""
        entry = {
            'symbol': symbol.upper(),
            'action': action,
            'changes': changes,
            'strategy': strategy,
            'time': current_date.strftime("%Y-%m-%d %H:%M:%S") if current_date else datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'price_at_change': current_price,
            'status': 'pending',
            'verified': False
        }
        self.history.append(entry)
        self.save_history()

    def verify_history(self, data_fetcher) -> List[Dict]:
        """Verify past momentum changes with proper time window and price threshold"""
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
            
            elapsed_hours = (now - entry_time).total_seconds() / 3600
                
            # Verify after minimum window (3 days), not 24 hours
            if elapsed_hours < self.VERIFY_MIN_HOURS:
                continue
            
            # Auto-expire very old entries (> 30 days)
            if elapsed_hours > 30 * 24:
                entry['status'] = 'expired'
                entry['verified'] = True
                verified_items.append(entry)
                continue
                
            symbol = entry.get('symbol')
            price_at_change = entry.get('price_at_change')
            
            if not price_at_change:
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
                    
                    # Calculate price change percentage
                    price_change_pct = ((current_price - price_at_change) / price_at_change) * 100
                    entry['price_change_pct'] = round(price_change_pct, 2)
                    
                    # Determine correctness with minimum threshold
                    is_correct = False
                    
                    if action in ('BUY', 'STRONG_BUY'):
                        # Correct only if price rose by at least VERIFY_MIN_PRICE_PCT
                        if price_change_pct >= self.VERIFY_MIN_PRICE_PCT:
                            is_correct = True
                    elif action in ('SELL', 'STRONG_SELL'):
                        # Correct only if price fell by at least VERIFY_MIN_PRICE_PCT
                        if price_change_pct <= -self.VERIFY_MIN_PRICE_PCT:
                            is_correct = True
                    else:
                        # HOLD - mark as neutral
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
                             momentum_analysis: dict, current_date=None) -> Dict:
        """Update momentum state for a stock and detect MEANINGFUL changes.
        
        Only fires on:
        - RSI extreme zone transitions (oversold recovery, overbought reversal) — NOT 50-crossings
        - MACD zero-line crossovers
        - MFI extreme zone transitions — NOT 50-crossings
        - Momentum score large shifts (≥5 points AND crossing zero)
        - Trend reversals (uptrend↔downtrend)
        - Golden Cross / Death Cross
        - EMA 50 and EMA 200 crossings (NOT EMA 20 — too noisy)
        
        All changes subject to per-symbol cooldown to prevent duplicate alerts.
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
            'timestamp': current_date.isoformat() if hasattr(current_date, 'isoformat') else (current_date if current_date else datetime.now().isoformat())
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
        
        # ── RSI extreme zone transitions ONLY ──
        prev_rsi = previous_state.get('rsi', 50)
        curr_rsi = current_state['rsi']
        
        # RSI recovery from oversold (< 30 → > 40): meaningful signal
        if prev_rsi < 30 and curr_rsi > 40:
            if not self._is_on_cooldown(symbol, 'rsi_recovery'):
                changes.append("📈 RSI recovered from oversold (momentum shift bullish)")
                momentum_changed = True
                self._set_cooldown(symbol, 'rsi_recovery')
        # RSI drop from overbought (> 70 → < 60): meaningful signal
        elif prev_rsi > 70 and curr_rsi < 60:
            if not self._is_on_cooldown(symbol, 'rsi_reversal'):
                changes.append("📉 RSI dropped from overbought (momentum shift bearish)")
                momentum_changed = True
                self._set_cooldown(symbol, 'rsi_reversal')
        # REMOVED: RSI 50-crossing — this is noise, not a signal
        
        # ── MACD zero-line crossover ──
        prev_macd_diff = previous_state.get('macd_diff', 0)
        curr_macd_diff = current_state['macd_diff']
        
        if prev_macd_diff <= 0 and curr_macd_diff > 0:
            if not self._is_on_cooldown(symbol, 'macd_bullish'):
                changes.append("📈 MACD turned bullish (momentum shift)")
                momentum_changed = True
                self._set_cooldown(symbol, 'macd_bullish')
        elif prev_macd_diff >= 0 and curr_macd_diff < 0:
            if not self._is_on_cooldown(symbol, 'macd_bearish'):
                changes.append("📉 MACD turned bearish (momentum shift)")
                momentum_changed = True
                self._set_cooldown(symbol, 'macd_bearish')
        
        # ── MFI extreme zone transitions ONLY ──
        prev_mfi = previous_state.get('mfi', 50)
        curr_mfi = current_state['mfi']
        
        if prev_mfi < 20 and curr_mfi > 30:
            if not self._is_on_cooldown(symbol, 'mfi_recovery'):
                changes.append("📈 MFI recovered from oversold (momentum shift bullish)")
                momentum_changed = True
                self._set_cooldown(symbol, 'mfi_recovery')
        elif prev_mfi > 80 and curr_mfi < 70:
            if not self._is_on_cooldown(symbol, 'mfi_reversal'):
                changes.append("📉 MFI dropped from overbought (momentum shift bearish)")
                momentum_changed = True
                self._set_cooldown(symbol, 'mfi_reversal')
        # REMOVED: MFI 50-crossing — same noise problem as RSI
        
        # ── Momentum score: require large shift AND zone crossing ──
        prev_momentum = previous_state.get('momentum_score', 0)
        curr_momentum = current_state['momentum_score']
        
        # Only fire if: shift ≥ 5 points AND crossed from negative to positive (or vice versa)
        score_shift = curr_momentum - prev_momentum
        crossed_zero = (prev_momentum < 0 and curr_momentum > 0) or (prev_momentum > 0 and curr_momentum < 0)
        
        if abs(score_shift) >= 5 and crossed_zero:
            direction = "bullish" if curr_momentum > prev_momentum else "bearish"
            if not self._is_on_cooldown(symbol, f'momentum_{direction}'):
                changes.append(f"🔄 Momentum score shifted {direction} ({prev_momentum} → {curr_momentum})")
                momentum_changed = True
                self._set_cooldown(symbol, f'momentum_{direction}')
        
        # ── Trend reversals ──
        prev_trend = previous_state.get('trend', 'sideways')
        curr_trend = current_state['trend']
        
        if prev_trend != curr_trend:
            if (prev_trend == 'uptrend' and curr_trend == 'downtrend') or \
               (prev_trend == 'downtrend' and curr_trend == 'uptrend'):
                if not self._is_on_cooldown(symbol, 'trend_reversal'):
                    changes.append(f"⚠️ TREND REVERSAL: {prev_trend} → {curr_trend}")
                    trend_reversed = True
                    self._set_cooldown(symbol, 'trend_reversal')
            else:
                # Sideways transitions are less significant
                if not self._is_on_cooldown(symbol, 'trend_shift'):
                    changes.append(f"🔄 Trend changed: {prev_trend} → {curr_trend}")
                    momentum_changed = True
                    self._set_cooldown(symbol, 'trend_shift')
        
        # ── Golden Cross / Death Cross ──
        prev_golden = previous_state.get('golden_cross', False)
        prev_death = previous_state.get('death_cross', False)
        curr_golden = current_state['golden_cross']
        curr_death = current_state['death_cross']
        
        if prev_death and curr_golden:
            if not self._is_on_cooldown(symbol, 'golden_cross'):
                changes.append("🟢 GOLDEN CROSS detected (major trend reversal to bullish)")
                trend_reversed = True
                self._set_cooldown(symbol, 'golden_cross')
        elif prev_golden and curr_death:
            if not self._is_on_cooldown(symbol, 'death_cross'):
                changes.append("🔴 DEATH CROSS detected (major trend reversal to bearish)")
                trend_reversed = True
                self._set_cooldown(symbol, 'death_cross')
        
        # ── EMA crossings: only EMA 50 and EMA 200 (EMA 20 is too noisy) ──
        prev_above_50 = previous_state.get('price_above_ema50', False)
        prev_above_200 = previous_state.get('price_above_ema200', False)
        curr_above_50 = current_state['price_above_ema50']
        curr_above_200 = current_state['price_above_ema200']
        
        # REMOVED: EMA 20 crossing — too noisy, fires on routine pullbacks
        
        if prev_above_50 != curr_above_50:
            direction = "above" if curr_above_50 else "below"
            if not self._is_on_cooldown(symbol, f'ema50_{direction}'):
                changes.append(f"🔄 Price crossed EMA 50 ({direction})")
                momentum_changed = True
                self._set_cooldown(symbol, f'ema50_{direction}')
        
        if prev_above_200 != curr_above_200:
            direction = "above" if curr_above_200 else "below"
            if not self._is_on_cooldown(symbol, f'ema200_{direction}'):
                changes.append(f"⚠️ Price crossed EMA 200 ({direction}) - major trend change")
                trend_reversed = True
                self._set_cooldown(symbol, f'ema200_{direction}')
        
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
        """Determine recommendation with MULTI-SIGNAL CONFIRMATION.
        
        Requires ≥2 confirming signals to set BUY or SELL.
        Conflicting signals = HOLD (not just reduced confidence).
        """
        if not previous_state:
            return {'action': 'HOLD', 'confidence': 50, 'reasons': ['Initial state']}
        
        curr_trend = current_state.get('trend', 'sideways')
        prev_trend = previous_state.get('trend', 'sideways')
        curr_rsi = current_state.get('rsi', 50)
        curr_mfi = current_state.get('mfi', 50)
        curr_macd_diff = current_state.get('macd_diff', 0)
        curr_momentum = current_state.get('momentum_score', 0)
        golden_cross = current_state.get('golden_cross', False)
        death_cross = current_state.get('death_cross', False)
        
        # Collect bullish and bearish signals separately
        bullish_signals = []
        bearish_signals = []
        
        # Trend reversal (strong signal, weight 2)
        if trend_reversed:
            if prev_trend == 'downtrend' and curr_trend == 'uptrend':
                bullish_signals.append(('trend_reversal', 70, "Trend reversed from downtrend to uptrend"))
            elif prev_trend == 'uptrend' and curr_trend == 'downtrend':
                bearish_signals.append(('trend_reversal', 70, "Trend reversed from uptrend to downtrend"))
        
        # Golden/Death Cross (strong signal)
        if golden_cross and not previous_state.get('golden_cross', False):
            bullish_signals.append(('golden_cross', 75, "Golden Cross detected - strong bullish signal"))
        if death_cross and not previous_state.get('death_cross', False):
            bearish_signals.append(('death_cross', 75, "Death Cross detected - strong bearish signal"))
        
        # RSI extremes
        if curr_rsi < 30:
            bullish_signals.append(('rsi_oversold', 65, f"RSI oversold ({curr_rsi:.1f}) - potential recovery"))
        elif curr_rsi > 70:
            bearish_signals.append(('rsi_overbought', 65, f"RSI overbought ({curr_rsi:.1f}) - potential pullback"))
        
        # MFI extremes
        if curr_mfi < 20:
            bullish_signals.append(('mfi_oversold', 60, f"MFI oversold ({curr_mfi:.1f}) - buying pressure"))
        elif curr_mfi > 80:
            bearish_signals.append(('mfi_overbought', 60, f"MFI overbought ({curr_mfi:.1f}) - selling pressure"))
        
        # MACD + momentum alignment (only if both agree)
        if curr_macd_diff > 0 and curr_momentum > 0:
            bullish_signals.append(('macd_momentum', 55, "MACD bullish with positive momentum"))
        elif curr_macd_diff < 0 and curr_momentum < 0:
            bearish_signals.append(('macd_momentum', 55, "MACD bearish with negative momentum"))
        
        # Strong momentum (must be significant, ≥5)
        if curr_momentum >= 5:
            bullish_signals.append(('strong_momentum', 60, "Strong bullish momentum"))
        elif curr_momentum <= -5:
            bearish_signals.append(('strong_momentum', 60, "Strong bearish momentum"))
        
        # ── Decision: require ≥2 confirming signals ──
        action = 'HOLD'
        confidence = 50
        reasons = []
        
        has_bullish = len(bullish_signals) >= 2
        has_bearish = len(bearish_signals) >= 2
        
        if has_bullish and has_bearish:
            # Conflicting confirmed signals → HOLD
            action = 'HOLD'
            confidence = 50
            reasons = ["⚠️ Conflicting signals — both bullish and bearish confirmed"]
        elif has_bullish:
            action = 'BUY'
            # Confidence = max of confirming signals, capped at 85
            confidence = min(85, max(s[1] for s in bullish_signals))
            # Bonus for more signals
            if len(bullish_signals) >= 3:
                confidence = min(85, confidence + 5)
            reasons = [s[2] for s in bullish_signals[:3]]
        elif has_bearish:
            action = 'SELL'
            confidence = min(85, max(s[1] for s in bearish_signals))
            if len(bearish_signals) >= 3:
                confidence = min(85, confidence + 5)
            reasons = [s[2] for s in bearish_signals[:3]]
        else:
            # Not enough signals to make a call
            action = 'HOLD'
            confidence = 50
            all_reasons = [s[2] for s in bullish_signals + bearish_signals]
            reasons = all_reasons[:2] if all_reasons else ["Insufficient confirming signals"]
        
        return {
            'action': action,
            'confidence': confidence,
            'reasons': reasons
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
        """Detect if price is at a peak or bottom.
        
        Uses multi-signal confirmation (≥3 signals needed).
        Fixed: no dead code, proper volume confirmation ordering.
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
            
            if 'close' not in df.columns or 'high' not in df.columns or 'low' not in df.columns:
                return {'peak_detected': False, 'bottom_detected': False, 'confidence': 0.0}
            
            current_price = float(df['close'].iloc[-1])
            current_high = float(df['high'].iloc[-1])
            current_low = float(df['low'].iloc[-1])
            
            # Recent price window
            recent_window = min(20, len(df))
            recent_highs = df['high'].tail(recent_window)
            recent_lows = df['low'].tail(recent_window)
            recent_prices = df['close'].tail(recent_window)
            
            max_high = float(recent_highs.max())
            min_low = float(recent_lows.min())
            
            # Proximity checks
            is_at_peak = current_high >= max_high * 0.98
            is_at_bottom = current_low <= min_low * 1.02
            price_moved_from_bottom = current_price > min_low * 1.03
            price_moved_from_peak = current_price < max_high * 0.97
            
            # Get indicators
            rsi = indicators.get('rsi', 50)
            mfi = indicators.get('mfi', 50)
            macd_diff = indicators.get('macd_diff', 0)
            
            # Volume ratio (compute once, use for both)
            volume_ratio = 1.0
            if 'volume' in df.columns:
                recent_volumes = df['volume'].tail(recent_window)
                if len(recent_volumes) > 0:
                    avg_volume = recent_volumes.mean()
                    current_volume = recent_volumes.iloc[-1]
                    if avg_volume > 0:
                        volume_ratio = current_volume / avg_volume
            
            # ── Peak detection signals ──
            peak_signals = 0
            peak_reasoning = []
            
            if is_at_peak:
                peak_signals += 1
                peak_reasoning.append(f"Price reached 20-day peak at ${max_high:.2f}")
            elif price_moved_from_peak:
                peak_signals += 1
                price_decrease_pct = ((max_high - current_price) / max_high) * 100
                peak_reasoning.append(f"Peak detected at ${max_high:.2f}, price reversing down ({price_decrease_pct:.1f}% decline)")
            
            if rsi >= 70:
                peak_signals += 2
                peak_reasoning.append(f"RSI overbought ({rsi:.1f})")
            elif rsi >= 65:
                peak_signals += 1
                peak_reasoning.append(f"RSI approaching overbought ({rsi:.1f})")
            
            if mfi >= 80:
                peak_signals += 2
                peak_reasoning.append(f"MFI overbought ({mfi:.1f})")
            elif mfi >= 75:
                peak_signals += 1
                peak_reasoning.append(f"MFI approaching overbought ({mfi:.1f})")
            
            # Bearish MACD divergence
            if len(recent_prices) >= 10:
                recent_high_prices = recent_highs.tail(10)
                if len(recent_high_prices) >= 5:
                    price_trend = recent_high_prices.iloc[-1] > recent_high_prices.iloc[-5]
                    if price_trend and macd_diff < 0:
                        peak_signals += 2
                        peak_reasoning.append("Bearish MACD divergence detected")
            
            # Volume confirmation for peak
            if volume_ratio > 1.5 and peak_signals > 0:
                peak_signals += 1
                peak_reasoning.append("High volume confirms peak")
            
            # ── Bottom detection signals ──
            bottom_signals = 0
            bottom_reasoning = []
            
            if is_at_bottom:
                bottom_signals += 1
                bottom_reasoning.append(f"Price reached 20-day bottom at ${min_low:.2f}")
            elif price_moved_from_bottom:
                bottom_signals += 1
                price_increase_pct = ((current_price - min_low) / min_low) * 100
                bottom_reasoning.append(f"Bottom detected at ${min_low:.2f}, price reversing up ({price_increase_pct:.1f}% recovery)")
            
            if rsi <= 30:
                bottom_signals += 2
                bottom_reasoning.append(f"RSI oversold ({rsi:.1f})")
            elif rsi <= 35:
                bottom_signals += 1
                bottom_reasoning.append(f"RSI approaching oversold ({rsi:.1f})")
            
            if mfi <= 20:
                bottom_signals += 2
                bottom_reasoning.append(f"MFI oversold ({mfi:.1f})")
            elif mfi <= 25:
                bottom_signals += 1
                bottom_reasoning.append(f"MFI approaching oversold ({mfi:.1f})")
            
            # Bullish MACD divergence
            if len(recent_prices) >= 10:
                recent_low_prices = recent_lows.tail(10)
                if len(recent_low_prices) >= 5:
                    price_trend = recent_low_prices.iloc[-1] < recent_low_prices.iloc[-5]
                    if price_trend and macd_diff > 0:
                        bottom_signals += 2
                        bottom_reasoning.append("Bullish MACD divergence detected")
            
            # Volume confirmation for bottom
            if volume_ratio > 1.5 and bottom_signals > 0:
                bottom_signals += 1
                bottom_reasoning.append("High volume confirms bottom")
            
            # ── Determine detection (≥3 signals) ──
            peak_detected = peak_signals >= 3
            bottom_detected = bottom_signals >= 3
            
            # If both detected, prioritize stronger
            if peak_detected and bottom_detected:
                if peak_signals >= bottom_signals:
                    bottom_detected = False
                else:
                    peak_detected = False
            
            # Build result
            if peak_detected:
                confidence = min(90, 50 + (peak_signals * 10))
                final_reasoning = '; '.join(peak_reasoning) if peak_reasoning else 'Peak detected with multiple confirmations'
            elif bottom_detected:
                confidence = min(90, 50 + (bottom_signals * 10))
                final_reasoning = '; '.join(bottom_reasoning) if bottom_reasoning else 'Bottom detected with multiple confirmations'
            else:
                confidence = 0.0
                final_reasoning = 'No significant signals'
            
            return {
                'peak_detected': peak_detected,
                'bottom_detected': bottom_detected,
                'confidence': confidence,
                'reasoning': final_reasoning,
                'current_price': current_price,
                'peak_price': max_high if peak_detected else None,
                'bottom_price': min_low if bottom_detected else None,
                'reversal_detected': peak_detected or bottom_detected,
                'signals_count': {'peak': peak_signals, 'bottom': bottom_signals}
            }
        except Exception as e:
            logger.error(f"Error detecting peaks/bottoms for {symbol}: {e}")
            return {'peak_detected': False, 'bottom_detected': False, 'confidence': 0.0, 'reasoning': f'Error: {str(e)}'}

    def get_statistics(self) -> Dict:
        """Get statistics about momentum change predictions"""
        total = len(self.history)
        verified_items = [p for p in self.history if p.get('verified', False)]
        verified = len(verified_items)
        correct = len([p for p in verified_items if p.get('status') == 'correct'])
        
        accuracy = (correct / verified * 100) if verified > 0 else 0
        
        return {
            'total_predictions': total,
            'verified': verified,
            'correct': correct,
            'incorrect': verified - correct,
            'accuracy': accuracy
        }
