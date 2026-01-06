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
        self.momentum_states = self._load_states()
    
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
        """Determine BUY/SELL/HOLD recommendation based on trend and momentum changes"""
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
        
        # Trend reversal analysis
        if trend_reversed:
            if prev_trend == 'downtrend' and curr_trend == 'uptrend':
                action = 'BUY'
                confidence = 70
                reasons.append("Trend reversed from downtrend to uptrend")
            elif prev_trend == 'uptrend' and curr_trend == 'downtrend':
                action = 'SELL'
                confidence = 70
                reasons.append("Trend reversed from uptrend to downtrend")
        
        # Golden Cross / Death Cross
        if golden_cross and not previous_state.get('golden_cross', False):
            action = 'BUY'
            confidence = max(confidence, 75)
            reasons.append("Golden Cross detected - strong bullish signal")
        elif death_cross and not previous_state.get('death_cross', False):
            action = 'SELL'
            confidence = max(confidence, 75)
            reasons.append("Death Cross detected - strong bearish signal")
        
        # RSI analysis
        if curr_rsi < 30:
            if action == 'HOLD':
                action = 'BUY'
                confidence = 65
            reasons.append(f"RSI oversold ({curr_rsi:.1f}) - potential buy opportunity")
        elif curr_rsi > 70:
            if action == 'HOLD':
                action = 'SELL'
                confidence = 65
            reasons.append(f"RSI overbought ({curr_rsi:.1f}) - potential sell signal")
        
        # MFI analysis
        if curr_mfi < 20:
            if action == 'HOLD':
                action = 'BUY'
                confidence = max(confidence, 60)
            reasons.append(f"MFI oversold ({curr_mfi:.1f}) - buying pressure")
        elif curr_mfi > 80:
            if action == 'HOLD':
                action = 'SELL'
                confidence = max(confidence, 60)
            reasons.append(f"MFI overbought ({curr_mfi:.1f}) - selling pressure")
        
        # MACD analysis
        if curr_macd_diff > 0:
            if action == 'HOLD' and curr_momentum > 0:
                action = 'BUY'
                confidence = max(confidence, 55)
            reasons.append("MACD bullish - upward momentum")
        elif curr_macd_diff < 0:
            if action == 'HOLD' and curr_momentum < 0:
                action = 'SELL'
                confidence = max(confidence, 55)
            reasons.append("MACD bearish - downward momentum")
        
        # Momentum score
        if curr_momentum >= 3:
            if action == 'HOLD':
                action = 'BUY'
                confidence = max(confidence, 60)
            reasons.append("Strong bullish momentum")
        elif curr_momentum <= -3:
            if action == 'HOLD':
                action = 'SELL'
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

