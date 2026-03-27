"""
Trend Change Prediction System
Predicts when trend changes might occur based on MULTI-SIGNAL CONFIRMATION.

Key design: Individual indicators produce SIGNALS (votes), not predictions.
A prediction is only emitted when ≥2 signals agree on direction.
This eliminates the single-indicator false positive problem.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class TrendChangePredictor:
    """Predicts when trend changes might occur using multi-signal confirmation"""
    
    def __init__(self, data_dir: Path = None):
        self.min_data_points = 50  # Minimum data points needed for prediction
        self.data_dir = data_dir
        self.learning_data_file = data_dir / "trend_change_learning.json" if data_dir else None
        
        # Load learned adjustments from verified predictions
        self.learned_adjustments = self._load_learned_adjustments()
        
    def predict_trend_changes(self, symbol: str, history_data: dict, 
                             indicators: dict, price_action: dict) -> List[Dict]:
        """
        Predict potential trend changes using multi-signal confirmation.
        
        Only emits a prediction when ≥2 independent signals agree on direction.
        
        Returns:
            List of trend change predictions (at most 1 per direction):
            [{
                'symbol': str,
                'current_trend': str,  # 'uptrend', 'downtrend', 'sideways'
                'predicted_change': str,  # 'bullish_reversal', 'bearish_reversal'
                'estimated_days': int,
                'estimated_date': str,
                'confidence': float,  # 0-100
                'reasoning': str,
                'key_indicators': dict,
                'confirming_signals': int  # Number of signals that agreed
            }]
        """
        try:
            data = history_data.get('data', [])
            if len(data) < self.min_data_points:
                return []
            
            df = pd.DataFrame(data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.sort_index()
            
            current_trend = price_action.get('trend', 'sideways')
            
            # Sideways trends are ambiguous — don't predict changes
            if current_trend == 'sideways':
                return []
            
            # ── Collect signals from each indicator ──
            signals = []
            
            rsi = indicators.get('rsi', 50)
            rsi_signal = self._get_rsi_signal(df, rsi, current_trend)
            if rsi_signal:
                signals.append(rsi_signal)
            
            macd_signal = self._get_macd_signal(df, indicators, current_trend)
            if macd_signal:
                signals.append(macd_signal)
            
            ema_signal = self._get_ema_signal(df, indicators, current_trend)
            if ema_signal:
                signals.append(ema_signal)
            
            divergence_signal = self._get_divergence_signal(df, indicators, current_trend)
            if divergence_signal:
                signals.append(divergence_signal)
            
            sr_signal = self._get_support_resistance_signal(df, price_action, current_trend)
            if sr_signal:
                signals.append(sr_signal)
            
            volume_signal = self._get_volume_signal(df, current_trend)
            if volume_signal:
                signals.append(volume_signal)
            
            adx_signal = self._get_adx_signal(df, current_trend)
            if adx_signal:
                signals.append(adx_signal)
            
            # ── Multi-signal confirmation ──
            # Group signals by direction
            bullish_signals = [s for s in signals if s['direction'] == 'bullish']
            bearish_signals = [s for s in signals if s['direction'] == 'bearish']
            
            predictions = []
            
            # Bullish reversal: need ≥3 confirming signals in a downtrend
            if current_trend == 'downtrend' and len(bullish_signals) >= 3:
                prediction = self._build_confirmed_prediction(
                    symbol, current_trend, 'bullish_reversal', bullish_signals
                )
                if prediction:
                    predictions.append(prediction)
            
            # Bearish reversal: need ≥3 confirming signals in an uptrend
            if current_trend == 'uptrend' and len(bearish_signals) >= 3:
                prediction = self._build_confirmed_prediction(
                    symbol, current_trend, 'bearish_reversal', bearish_signals
                )
                if prediction:
                    predictions.append(prediction)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting trend changes for {symbol}: {e}")
            return []
    
    def _build_confirmed_prediction(self, symbol: str, current_trend: str,
                                     predicted_change: str, 
                                     confirming_signals: List[Dict]) -> Optional[Dict]:
        """Build a prediction from multiple confirming signals."""
        if not confirming_signals or len(confirming_signals) < 2:
            return None
        
        # Weighted average of estimated days (weight by signal strength)
        total_weight = sum(s['strength'] for s in confirming_signals)
        if total_weight == 0:
            return None
        
        estimated_days = int(sum(
            s['estimated_days'] * s['strength'] for s in confirming_signals
        ) / total_weight)
        estimated_days = max(1, min(120, estimated_days))
        
        # Confidence = base from signal agreement + bonus for strength
        # 3 signals = base 55, 4 = base 60, 5 = base 65, 6 = base 68, 7 = base 70
        num_signals = len(confirming_signals)
        base_confidence = min(70, 40 + num_signals * 5)
        
        # Average strength bonus (0-15 points)
        avg_strength = sum(s['strength'] for s in confirming_signals) / num_signals
        strength_bonus = avg_strength * 15
        
        # Penalty if any signals disagree (not present in confirming list, but 
        # the absence of some signals = less certainty)
        confidence = min(85, base_confidence + strength_bonus)
        
        # Build combined reasoning
        signal_names = [s['source'] for s in confirming_signals]
        reasons = [s['reason'] for s in confirming_signals]
        combined_reasoning = f"Multi-signal confirmation ({', '.join(signal_names)}): " + "; ".join(reasons)
        
        # Build combined key indicators
        key_indicators = {}
        for s in confirming_signals:
            key_indicators.update(s.get('indicators', {}))
        key_indicators['confirming_signals'] = signal_names
        
        try:
            from utils.market_utils import add_trading_days
            estimated_date = add_trading_days(datetime.now(), estimated_days)
        except ImportError:
            estimated_date = datetime.now() + timedelta(days=estimated_days)
        
        return {
            'symbol': symbol,
            'current_trend': current_trend,
            'predicted_change': predicted_change,
            'estimated_days': estimated_days,
            'estimated_date': estimated_date.isoformat(),
            'confidence': float(confidence),
            'reasoning': combined_reasoning,
            'key_indicators': key_indicators,
            'confirming_signals': num_signals
        }
    
    # ══════════════════════════════════════════════════════════
    #  SIGNAL GENERATORS — return signal dicts, NOT predictions
    # ══════════════════════════════════════════════════════════
    
    def _get_rsi_signal(self, df: pd.DataFrame, current_rsi: float, 
                        current_trend: str) -> Optional[Dict]:
        """RSI-based signal — classic oversold/overbought thresholds."""
        try:
            from ta.momentum import RSIIndicator
            rsi_indicator = RSIIndicator(close=df['close'], window=14)
            rsi_values = rsi_indicator.rsi().dropna()
            
            if len(rsi_values) < 20:
                return None
            
            # Bullish signal: RSI < 30 in downtrend (textbook oversold)
            if current_trend == 'downtrend' and current_rsi < 30:
                oversold_periods = self._find_oversold_periods(rsi_values)
                avg_recovery_days = self._calculate_avg_recovery_time(oversold_periods, df)
                
                if avg_recovery_days and avg_recovery_days > 0:
                    # Strength: how deep the oversold (RSI 20 = very strong, RSI 29 = weak)
                    strength = min(1.0, (30 - current_rsi) / 15)
                    
                    return {
                        'direction': 'bullish',
                        'source': 'RSI',
                        'strength': strength,
                        'estimated_days': int(avg_recovery_days),
                        'reason': f"RSI oversold at {current_rsi:.1f}",
                        'indicators': {'rsi': float(current_rsi), 'rsi_status': 'oversold'}
                    }
            
            # Bearish signal: RSI > 70 in uptrend (textbook overbought)
            elif current_trend == 'uptrend' and current_rsi > 70:
                overbought_periods = self._find_overbought_periods(rsi_values)
                avg_reversal_days = self._calculate_avg_reversal_time(overbought_periods, df)
                
                if avg_reversal_days and avg_reversal_days > 0:
                    strength = min(1.0, (current_rsi - 70) / 15)
                    
                    return {
                        'direction': 'bearish',
                        'source': 'RSI',
                        'strength': strength,
                        'estimated_days': int(avg_reversal_days),
                        'reason': f"RSI overbought at {current_rsi:.1f}",
                        'indicators': {'rsi': float(current_rsi), 'rsi_status': 'overbought'}
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in RSI signal: {e}")
            return None
    
    def _get_macd_signal(self, df: pd.DataFrame, indicators: dict, 
                         current_trend: str) -> Optional[Dict]:
        """MACD crossover signal — must be converging AND close."""
        try:
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            macd_diff = indicators.get('macd_diff', 0)
            
            from ta.trend import MACD
            macd_indicator = MACD(close=df['close'])
            macd_line = macd_indicator.macd().dropna()
            signal_line = macd_indicator.macd_signal().dropna()
            
            if len(macd_line) < 20 or len(signal_line) < 20:
                return None
            
            macd_diff_abs = abs(macd_diff)
            
            # Check convergence (MACD and signal trending towards each other)
            macd_trend = macd_line.iloc[-1] - macd_line.iloc[-5] if len(macd_line) >= 5 else 0
            signal_trend = signal_line.iloc[-1] - signal_line.iloc[-5] if len(signal_line) >= 5 else 0
            is_converging = (macd_diff < 0 and macd_trend > signal_trend) or \
                           (macd_diff > 0 and macd_trend < signal_trend)
            
            # TIGHTENED: Must be converging AND within 0.2 (was 0.5 or just converging)
            if not (is_converging and macd_diff_abs < 0.2):
                return None
            
            convergence_rate = self._calculate_macd_convergence_rate(macd_line, signal_line)
            if not convergence_rate or convergence_rate <= 0:
                return None
            
            base_days = max(1, min(120, int(1 / convergence_rate * 5)))
            multiplier = self.learned_adjustments.get('macd_convergence_multiplier', 1.0)
            estimated_days = max(1, min(120, int(base_days * multiplier)))
            
            # Strength: closer to crossover = stronger
            strength = min(1.0, (0.2 - macd_diff_abs) / 0.2)
            
            if current_trend == 'downtrend' and macd_diff < 0:
                return {
                    'direction': 'bullish',
                    'source': 'MACD',
                    'strength': strength,
                    'estimated_days': estimated_days,
                    'reason': f"MACD converging with signal (diff: {macd_diff:.3f})",
                    'indicators': {'macd': float(macd), 'macd_signal': float(macd_signal), 
                                  'macd_diff': float(macd_diff)}
                }
            elif current_trend == 'uptrend' and macd_diff > 0:
                return {
                    'direction': 'bearish',
                    'source': 'MACD',
                    'strength': strength,
                    'estimated_days': estimated_days,
                    'reason': f"MACD converging with signal (diff: {macd_diff:.3f})",
                    'indicators': {'macd': float(macd), 'macd_signal': float(macd_signal), 
                                  'macd_diff': float(macd_diff)}
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in MACD signal: {e}")
            return None
    
    def _get_ema_signal(self, df: pd.DataFrame, indicators: dict, 
                        current_trend: str) -> Optional[Dict]:
        """EMA crossover signal — must be converging AND within 2%."""
        try:
            ema_20 = indicators.get('ema_20', 0)
            ema_50 = indicators.get('ema_50', 0)
            
            if ema_20 == 0 or ema_50 == 0:
                return None
            
            from ta.trend import EMAIndicator
            ema_20_ind = EMAIndicator(close=df['close'], window=20)
            ema_50_ind = EMAIndicator(close=df['close'], window=50)
            ema_20_vals = ema_20_ind.ema_indicator().dropna()
            ema_50_vals = ema_50_ind.ema_indicator().dropna()
            
            if len(ema_20_vals) < 20 or len(ema_50_vals) < 20:
                return None
            
            ema_diff = ema_20 - ema_50
            ema_diff_pct = abs((ema_diff / ema_50) * 100) if ema_50 > 0 else 0
            
            # Check convergence
            ema_20_trend = ema_20_vals.iloc[-1] - ema_20_vals.iloc[-5] if len(ema_20_vals) >= 5 else 0
            ema_50_trend = ema_50_vals.iloc[-1] - ema_50_vals.iloc[-5] if len(ema_50_vals) >= 5 else 0
            is_converging = (ema_diff < 0 and ema_20_trend > ema_50_trend) or \
                           (ema_diff > 0 and ema_20_trend < ema_50_trend)
            
            # TIGHTENED: Must be converging AND within 2% (was 5% or just converging)
            if not (is_converging and ema_diff_pct < 2):
                return None
            
            convergence_rate = self._calculate_ema_convergence_rate(ema_20_vals, ema_50_vals)
            if not convergence_rate or convergence_rate <= 0:
                return None
            
            base_days = max(1, min(120, int(1 / convergence_rate * 10)))
            multiplier = self.learned_adjustments.get('ema_convergence_multiplier', 1.0)
            estimated_days = max(1, min(120, int(base_days * multiplier)))
            
            # Strength: closer = stronger
            strength = min(1.0, (2 - ema_diff_pct) / 2)
            
            if current_trend == 'downtrend' and ema_diff < 0:
                return {
                    'direction': 'bullish',
                    'source': 'EMA',
                    'strength': strength,
                    'estimated_days': estimated_days,
                    'reason': f"EMA 20/50 converging ({ema_diff_pct:.1f}% apart, Golden Cross potential)",
                    'indicators': {'ema_20': float(ema_20), 'ema_50': float(ema_50), 
                                  'ema_diff_pct': float(ema_diff_pct)}
                }
            elif current_trend == 'uptrend' and ema_diff > 0:
                return {
                    'direction': 'bearish',
                    'source': 'EMA',
                    'strength': strength,
                    'estimated_days': estimated_days,
                    'reason': f"EMA 20/50 converging ({ema_diff_pct:.1f}% apart, Death Cross potential)",
                    'indicators': {'ema_20': float(ema_20), 'ema_50': float(ema_50), 
                                  'ema_diff_pct': float(ema_diff_pct)}
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in EMA signal: {e}")
            return None
    
    def _get_divergence_signal(self, df: pd.DataFrame, indicators: dict, 
                               current_trend: str) -> Optional[Dict]:
        """Momentum divergence signal — price vs momentum disagreement."""
        try:
            prices = df['close'].tail(20).values
            if len(prices) < 10:
                return None
            
            momentum = np.diff(prices) / prices[:-1] * 100
            if len(momentum) < 10:
                return None
            
            # Bearish divergence: price up but momentum weakening
            if current_trend == 'uptrend':
                recent_price_trend = np.mean(prices[-5:]) > np.mean(prices[-10:-5])
                recent_momentum = np.mean(momentum[-5:]) < np.mean(momentum[-10:-5])
                
                if recent_price_trend and recent_momentum:
                    divergence_strength = abs(np.mean(momentum[-5:]) - np.mean(momentum[-10:-5]))
                    estimated_days = max(3, min(60, int(15 - divergence_strength * 2)))
                    strength = min(1.0, divergence_strength / 2)
                    
                    return {
                        'direction': 'bearish',
                        'source': 'Divergence',
                        'strength': strength,
                        'estimated_days': estimated_days,
                        'reason': "Bearish divergence — price rising but momentum weakening",
                        'indicators': {'divergence_type': 'bearish', 
                                      'divergence_strength': float(divergence_strength)}
                    }
            
            # Bullish divergence: price down but momentum improving
            elif current_trend == 'downtrend':
                recent_price_trend = np.mean(prices[-5:]) < np.mean(prices[-10:-5])
                recent_momentum = np.mean(momentum[-5:]) > np.mean(momentum[-10:-5])
                
                if recent_price_trend and recent_momentum:
                    divergence_strength = abs(np.mean(momentum[-5:]) - np.mean(momentum[-10:-5]))
                    estimated_days = max(3, min(60, int(15 - divergence_strength * 2)))
                    strength = min(1.0, divergence_strength / 2)
                    
                    return {
                        'direction': 'bullish',
                        'source': 'Divergence',
                        'strength': strength,
                        'estimated_days': estimated_days,
                        'reason': "Bullish divergence — price falling but momentum improving",
                        'indicators': {'divergence_type': 'bullish', 
                                      'divergence_strength': float(divergence_strength)}
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in divergence signal: {e}")
            return None
    
    def _get_support_resistance_signal(self, df: pd.DataFrame, price_action: dict, 
                                        current_trend: str) -> Optional[Dict]:
        """Support/resistance proximity signal — must be within 2%."""
        try:
            current_price = df['close'].iloc[-1]
            support = price_action.get('recent_low', current_price * 0.95)
            resistance = price_action.get('recent_high', current_price * 1.05)
            
            distance_to_support = ((current_price - support) / support) * 100
            distance_to_resistance = ((resistance - current_price) / current_price) * 100
            
            # TIGHTENED: Must be within 2% (was 5%)
            if current_trend == 'downtrend' and distance_to_support < 2:
                adjustment = self.learned_adjustments.get('support_resistance_days_adjustment', 0)
                estimated_days = max(1, 3 + adjustment)
                strength = min(1.0, (2 - distance_to_support) / 2)
                
                return {
                    'direction': 'bullish',
                    'source': 'Support',
                    'strength': strength,
                    'estimated_days': estimated_days,
                    'reason': f"Price near support ({distance_to_support:.1f}% away)",
                    'indicators': {'distance_to_support': float(distance_to_support)}
                }
            
            elif current_trend == 'uptrend' and distance_to_resistance < 2:
                adjustment = self.learned_adjustments.get('support_resistance_days_adjustment', 0)
                estimated_days = max(1, 3 + adjustment)
                strength = min(1.0, (2 - distance_to_resistance) / 2)
                
                return {
                    'direction': 'bearish',
                    'source': 'Resistance',
                    'strength': strength,
                    'estimated_days': estimated_days,
                    'reason': f"Price near resistance ({distance_to_resistance:.1f}% away)",
                    'indicators': {'distance_to_resistance': float(distance_to_resistance)}
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in S/R signal: {e}")
            return None
    
    def _get_volume_signal(self, df: pd.DataFrame, current_trend: str) -> Optional[Dict]:
        """Volume spike signal — above-average volume supports trend reversal.
        
        Rationale: Reversals on low volume are often fake-outs.
        High volume at potential reversal points = institutional participation.
        """
        try:
            if 'volume' not in df.columns or len(df) < 25:
                return None
            
            volumes = df['volume'].tail(25)
            if volumes.isna().all() or volumes.sum() == 0:
                return None
            
            avg_volume_20 = volumes.iloc[:-5].mean()  # 20-day average (excluding last 5)
            recent_volume = volumes.iloc[-5:].mean()   # Last 5 days average
            
            if avg_volume_20 == 0:
                return None
            
            volume_ratio = recent_volume / avg_volume_20
            
            # Volume must be above 1.5x average to be meaningful
            if volume_ratio < 1.5:
                return None
            
            # Strength: higher volume ratio = stronger signal
            strength = min(1.0, (volume_ratio - 1.5) / 1.5)  # 1.5x = 0, 3x = 1.0
            
            # Volume supports the reversal direction based on current trend
            if current_trend == 'downtrend':
                return {
                    'direction': 'bullish',
                    'source': 'Volume',
                    'strength': strength,
                    'estimated_days': 5,  # Volume signals are typically short-term
                    'reason': f"Volume spike ({volume_ratio:.1f}x avg) — potential capitulation/accumulation",
                    'indicators': {'volume_ratio': float(volume_ratio)}
                }
            elif current_trend == 'uptrend':
                return {
                    'direction': 'bearish',
                    'source': 'Volume',
                    'strength': strength,
                    'estimated_days': 5,
                    'reason': f"Volume spike ({volume_ratio:.1f}x avg) — potential distribution/climax",
                    'indicators': {'volume_ratio': float(volume_ratio)}
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in volume signal: {e}")
            return None
    
    def _get_adx_signal(self, df: pd.DataFrame, current_trend: str) -> Optional[Dict]:
        """ADX trend weakness signal — weakening trend supports reversal.
        
        Rationale: ADX < 25 means the trend is losing strength.
        A weakening trend + other reversal signals = higher probability of actual reversal.
        ADX > 25 = strong trend, reversals less likely.
        """
        try:
            from ta.trend import ADXIndicator
            
            if len(df) < 30 or 'high' not in df.columns or 'low' not in df.columns:
                return None
            
            adx_indicator = ADXIndicator(
                high=df['high'], low=df['low'], close=df['close'], window=14
            )
            adx_values = adx_indicator.adx().dropna()
            
            if len(adx_values) < 5:
                return None
            
            current_adx = adx_values.iloc[-1]
            prev_adx = adx_values.iloc[-5]
            adx_declining = current_adx < prev_adx  # ADX is weakening
            
            # Only signal when ADX < 25 (weak/weakening trend)
            if current_adx >= 25:
                return None
            
            # Stronger signal when ADX is both low AND declining
            if not adx_declining:
                return None
            
            # Strength: lower ADX = stronger reversal signal
            strength = min(1.0, (25 - current_adx) / 15)  # ADX 10 = strong, ADX 24 = weak
            
            if current_trend == 'downtrend':
                return {
                    'direction': 'bullish',
                    'source': 'ADX',
                    'strength': strength,
                    'estimated_days': 7,
                    'reason': f"Trend losing strength (ADX {current_adx:.1f}, declining) — reversal more likely",
                    'indicators': {'adx': float(current_adx), 'adx_declining': True}
                }
            elif current_trend == 'uptrend':
                return {
                    'direction': 'bearish',
                    'source': 'ADX',
                    'strength': strength,
                    'estimated_days': 7,
                    'reason': f"Trend losing strength (ADX {current_adx:.1f}, declining) — reversal more likely",
                    'indicators': {'adx': float(current_adx), 'adx_declining': True}
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in ADX signal: {e}")
            return None
    
    # ══════════════════════════════════════
    #  HELPER METHODS
    # ══════════════════════════════════════
    
    def _find_oversold_periods(self, rsi_values: pd.Series) -> List[int]:
        """Find periods where RSI was oversold"""
        oversold_periods = []
        in_oversold = False
        start_idx = None
        
        for i, rsi in enumerate(rsi_values):
            if rsi < 30 and not in_oversold:
                in_oversold = True
                start_idx = i
            elif rsi >= 30 and in_oversold:
                in_oversold = False
                if start_idx is not None:
                    oversold_periods.append(i - start_idx)
        
        return oversold_periods
    
    def _find_overbought_periods(self, rsi_values: pd.Series) -> List[int]:
        """Find periods where RSI was overbought"""
        overbought_periods = []
        in_overbought = False
        start_idx = None
        
        for i, rsi in enumerate(rsi_values):
            if rsi > 70 and not in_overbought:
                in_overbought = True
                start_idx = i
            elif rsi <= 70 and in_overbought:
                in_overbought = False
                if start_idx is not None:
                    overbought_periods.append(i - start_idx)
        
        return overbought_periods
    
    def _load_learned_adjustments(self) -> Dict:
        """Load learned adjustments from verified predictions"""
        if not self.learning_data_file or not self.learning_data_file.exists():
            return {
                'rsi_recovery_multiplier': 1.0,
                'rsi_reversal_multiplier': 1.0,
                'macd_convergence_multiplier': 1.0,
                'ema_convergence_multiplier': 1.0,
                'divergence_days_adjustment': 0,
                'support_resistance_days_adjustment': 0,
                'price_prediction_adjustment': 1.0,
                'verified_count': 0
            }
        
        try:
            with open(self.learning_data_file, 'r') as f:
                data = json.load(f)
                if 'price_prediction_adjustment' not in data:
                    data['price_prediction_adjustment'] = 1.0
                return data
        except Exception as e:
            logger.error(f"Error loading learned adjustments: {e}")
            return {
                'rsi_recovery_multiplier': 1.0,
                'rsi_reversal_multiplier': 1.0,
                'macd_convergence_multiplier': 1.0,
                'ema_convergence_multiplier': 1.0,
                'divergence_days_adjustment': 0,
                'support_resistance_days_adjustment': 0,
                'price_prediction_adjustment': 1.0,
                'verified_count': 0
            }
    
    def _save_learned_adjustments(self):
        """Save learned adjustments to file"""
        if not self.learning_data_file:
            return
        
        try:
            self.learning_data_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.learning_data_file, 'w') as f:
                json.dump(self.learned_adjustments, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving learned adjustments: {e}")
    
    def learn_from_verified_predictions(self, verified_predictions: List[Dict]):
        """Learn from verified predictions to improve future estimates"""
        if not verified_predictions:
            return
        
        # Group by signal source (from reasoning text)
        rsi_predictions = []
        macd_predictions = []
        ema_predictions = []
        
        for pred in verified_predictions:
            if not pred.get('verified') or pred.get('was_correct') is None:
                continue
            
            reasoning = pred.get('reasoning', '')
            estimated_days = pred.get('estimated_days', 0)
            
            try:
                estimated_date = datetime.fromisoformat(pred.get('estimated_date', ''))
                verification_date = datetime.fromisoformat(pred.get('verification_date', ''))
                actual_days = (verification_date - estimated_date).days
            except:
                actual_days = None
            
            pred_data = {
                'estimated_days': estimated_days,
                'actual_days': actual_days,
                'was_correct': pred.get('was_correct', False)
            }
            
            if 'RSI' in reasoning:
                rsi_predictions.append(pred_data)
            elif 'MACD' in reasoning:
                macd_predictions.append(pred_data)
            elif 'EMA' in reasoning:
                ema_predictions.append(pred_data)
        
        # RSI adjustments
        if rsi_predictions:
            correct_rsi = [p for p in rsi_predictions if p['was_correct']]
            if correct_rsi and len(correct_rsi) >= 3:
                avg_estimated = np.mean([p['estimated_days'] for p in correct_rsi])
                avg_actual = np.mean([p['actual_days'] for p in correct_rsi if p['actual_days'] is not None])
                if avg_actual and avg_estimated > 0:
                    ratio = avg_actual / avg_estimated
                    self.learned_adjustments['rsi_recovery_multiplier'] = (
                        0.7 * self.learned_adjustments['rsi_recovery_multiplier'] + 0.3 * ratio
                    )
        
        # MACD adjustments
        if macd_predictions:
            correct_macd = [p for p in macd_predictions if p['was_correct']]
            if correct_macd and len(correct_macd) >= 3:
                avg_estimated = np.mean([p['estimated_days'] for p in correct_macd])
                avg_actual = np.mean([p['actual_days'] for p in correct_macd if p['actual_days'] is not None])
                if avg_actual and avg_estimated > 0:
                    ratio = avg_actual / avg_estimated
                    self.learned_adjustments['macd_convergence_multiplier'] = (
                        0.7 * self.learned_adjustments['macd_convergence_multiplier'] + 0.3 * ratio
                    )
        
        # EMA adjustments
        if ema_predictions:
            correct_ema = [p for p in ema_predictions if p['was_correct']]
            if correct_ema and len(correct_ema) >= 3:
                avg_estimated = np.mean([p['estimated_days'] for p in correct_ema])
                avg_actual = np.mean([p['actual_days'] for p in correct_ema if p['actual_days'] is not None])
                if avg_actual and avg_estimated > 0:
                    ratio = avg_actual / avg_estimated
                    self.learned_adjustments['ema_convergence_multiplier'] = (
                        0.7 * self.learned_adjustments['ema_convergence_multiplier'] + 0.3 * ratio
                    )
        
        self.learned_adjustments['verified_count'] += len(verified_predictions)
        self._save_learned_adjustments()
        logger.info(f"🧠 Learned from {len(verified_predictions)} verified trend change predictions")
    
    def _calculate_avg_recovery_time(self, periods: List[int], df: pd.DataFrame) -> Optional[float]:
        """Calculate average recovery time from oversold periods"""
        if not periods:
            return None
        
        avg_period = np.mean(periods) if periods else 10
        base_estimate = min(120, max(1, avg_period))
        
        multiplier = self.learned_adjustments.get('rsi_recovery_multiplier', 1.0)
        adjusted = base_estimate * multiplier
        return min(120, max(1, adjusted))
    
    def _calculate_avg_reversal_time(self, periods: List[int], df: pd.DataFrame) -> Optional[float]:
        """Calculate average reversal time from overbought periods"""
        if not periods:
            return None
        
        avg_period = np.mean(periods) if periods else 10
        base_estimate = min(120, max(1, avg_period))
        
        multiplier = self.learned_adjustments.get('rsi_reversal_multiplier', 1.0)
        adjusted = base_estimate * multiplier
        return min(120, max(1, adjusted))
    
    def _calculate_macd_convergence_rate(self, macd_line: pd.Series, 
                                         signal_line: pd.Series) -> Optional[float]:
        """Calculate how fast MACD is converging with signal line"""
        if len(macd_line) < 5 or len(signal_line) < 5:
            return None
        
        recent_diff = abs(macd_line.iloc[-1] - signal_line.iloc[-1])
        prev_diff = abs(macd_line.iloc[-5] - signal_line.iloc[-5])
        
        if prev_diff == 0:
            return None
        
        convergence_rate = (prev_diff - recent_diff) / prev_diff / 5
        return max(0.01, min(0.5, convergence_rate))
    
    def _calculate_ema_convergence_rate(self, ema_20: pd.Series, ema_50: pd.Series) -> Optional[float]:
        """Calculate how fast EMA 20 is converging with EMA 50"""
        if len(ema_20) < 5 or len(ema_50) < 5:
            return None
        
        recent_diff_pct = abs((ema_20.iloc[-1] - ema_50.iloc[-1]) / ema_50.iloc[-1]) * 100
        prev_diff_pct = abs((ema_20.iloc[-5] - ema_50.iloc[-5]) / ema_50.iloc[-5]) * 100
        
        if prev_diff_pct == 0:
            return None
        
        convergence_rate = (prev_diff_pct - recent_diff_pct) / prev_diff_pct / 5
        return max(0.01, min(0.3, convergence_rate))
