"""
Trend Change Prediction System
Predicts when trend changes might occur based on technical indicators and historical patterns
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
    """Predicts when trend changes might occur"""
    
    def __init__(self, data_dir: Path = None):
        self.min_data_points = 50  # Minimum data points needed for prediction
        self.data_dir = data_dir
        self.learning_data_file = data_dir / "trend_change_learning.json" if data_dir else None
        
        # Load learned adjustments from verified predictions
        self.learned_adjustments = self._load_learned_adjustments()
        
    def predict_trend_changes(self, symbol: str, history_data: dict, 
                             indicators: dict, price_action: dict) -> List[Dict]:
        """
        Predict potential trend changes and estimate when they might occur
        
        Returns:
            List of trend change predictions:
            [{
                'symbol': str,
                'current_trend': str,  # 'uptrend', 'downtrend', 'sideways'
                'predicted_change': str,  # 'bullish_reversal', 'bearish_reversal', 'continuation'
                'estimated_days': int,  # Estimated days until change
                'estimated_date': str,  # ISO format date
                'confidence': float,  # 0-100
                'reasoning': str,
                'key_indicators': dict
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
            predictions = []
            
            # Analyze RSI for potential reversals
            rsi = indicators.get('rsi', 50)
            rsi_prediction = self._predict_rsi_reversal(df, rsi, current_trend)
            if rsi_prediction:
                predictions.append(rsi_prediction)
            
            # Analyze MACD for trend changes
            macd_prediction = self._predict_macd_crossover(df, indicators, current_trend)
            if macd_prediction:
                predictions.append(macd_prediction)
            
            # Analyze moving average crossovers
            ema_prediction = self._predict_ema_crossover(df, indicators, current_trend)
            if ema_prediction:
                predictions.append(ema_prediction)
            
            # Analyze momentum divergence
            divergence_prediction = self._predict_momentum_divergence(df, indicators, current_trend)
            if divergence_prediction:
                predictions.append(divergence_prediction)
            
            # Analyze support/resistance breaks
            support_resistance_prediction = self._predict_support_resistance_break(
                df, price_action, current_trend
            )
            if support_resistance_prediction:
                predictions.append(support_resistance_prediction)
            
            # Add continuation predictions for strong trends
            continuation_prediction = self._predict_trend_continuation(df, indicators, current_trend)
            if continuation_prediction:
                predictions.append(continuation_prediction)
            
            # Add symbol to all predictions
            for pred in predictions:
                pred['symbol'] = symbol
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting trend changes for {symbol}: {e}")
            return []
    
    def _predict_rsi_reversal(self, df: pd.DataFrame, current_rsi: float, 
                              current_trend: str) -> Optional[Dict]:
        """Predict RSI-based trend reversal"""
        try:
            # Calculate RSI history
            from ta.momentum import RSIIndicator
            rsi_indicator = RSIIndicator(close=df['close'], window=14)
            rsi_values = rsi_indicator.rsi().dropna()
            
            if len(rsi_values) < 20:
                return None
            
            # Check for oversold/overbought conditions that typically lead to reversals
            # More lenient thresholds: RSI < 35 (approaching oversold) or > 65 (approaching overbought)
            if current_trend == 'downtrend' and current_rsi < 35:
                # Oversold or approaching oversold in downtrend - potential bullish reversal
                oversold_periods = self._find_oversold_periods(rsi_values)
                avg_recovery_days = self._calculate_avg_recovery_time(oversold_periods, df)
                
                if avg_recovery_days and avg_recovery_days > 0:
                    estimated_date = datetime.now() + timedelta(days=int(avg_recovery_days))
                    # Adjust confidence based on how oversold (lower RSI = higher confidence)
                    if current_rsi < 30:
                        confidence = max(0, min(85, 50 + (30 - current_rsi) * 2))  # Very oversold
                    else:
                        confidence = max(0, min(70, 40 + (35 - current_rsi) * 2))  # Approaching oversold
                    
                    return {
                        'current_trend': current_trend,
                        'predicted_change': 'bullish_reversal',
                        'estimated_days': int(avg_recovery_days),
                        'estimated_date': estimated_date.isoformat(),
                        'confidence': float(confidence),
                        'reasoning': f"RSI {'oversold' if current_rsi < 30 else 'approaching oversold'} ({current_rsi:.1f}) in downtrend - historical patterns suggest bullish reversal in ~{int(avg_recovery_days)} days",
                        'key_indicators': {'rsi': float(current_rsi), 'rsi_status': 'oversold' if current_rsi < 30 else 'approaching_oversold'}
                    }
            
            elif current_trend == 'uptrend' and current_rsi > 65:
                # Overbought or approaching overbought in uptrend - potential bearish reversal
                overbought_periods = self._find_overbought_periods(rsi_values)
                avg_reversal_days = self._calculate_avg_reversal_time(overbought_periods, df)
                
                if avg_reversal_days and avg_reversal_days > 0:
                    estimated_date = datetime.now() + timedelta(days=int(avg_reversal_days))
                    # Adjust confidence based on how overbought (higher RSI = higher confidence)
                    if current_rsi > 70:
                        confidence = max(0, min(85, 50 + (current_rsi - 70) * 2))  # Very overbought
                    else:
                        confidence = max(0, min(70, 40 + (current_rsi - 65) * 2))  # Approaching overbought
                    
                    return {
                        'current_trend': current_trend,
                        'predicted_change': 'bearish_reversal',
                        'estimated_days': int(avg_reversal_days),
                        'estimated_date': estimated_date.isoformat(),
                        'confidence': float(confidence),
                        'reasoning': f"RSI {'overbought' if current_rsi > 70 else 'approaching overbought'} ({current_rsi:.1f}) in uptrend - historical patterns suggest bearish reversal in ~{int(avg_reversal_days)} days",
                        'key_indicators': {'rsi': float(current_rsi), 'rsi_status': 'overbought' if current_rsi > 70 else 'approaching_overbought'}
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in RSI reversal prediction: {e}")
            return None
    
    def _predict_macd_crossover(self, df: pd.DataFrame, indicators: dict, 
                               current_trend: str) -> Optional[Dict]:
        """Predict MACD crossover trend change"""
        try:
            macd = indicators.get('macd', 0)
            macd_signal = indicators.get('macd_signal', 0)
            macd_diff = indicators.get('macd_diff', 0)
            
            # Calculate MACD history
            from ta.trend import MACD
            macd_indicator = MACD(close=df['close'])
            macd_line = macd_indicator.macd().dropna()
            signal_line = macd_indicator.macd_signal().dropna()
            
            if len(macd_line) < 20 or len(signal_line) < 20:
                return None
            
            # Check if MACD is approaching signal line (potential crossover)
            # More lenient: check if MACD is converging (diff getting smaller) or already close
            macd_diff_abs = abs(macd_diff)
            macd_trend = macd_line.iloc[-1] - macd_line.iloc[-5] if len(macd_line) >= 5 else 0
            signal_trend = signal_line.iloc[-1] - signal_line.iloc[-5] if len(signal_line) >= 5 else 0
            is_converging = (macd_diff < 0 and macd_trend > signal_trend) or (macd_diff > 0 and macd_trend < signal_trend)
            
            if current_trend == 'downtrend' and macd_diff < 0 and (macd_diff_abs < 0.5 or is_converging):
                # MACD approaching signal from below - potential bullish crossover
                convergence_rate = self._calculate_macd_convergence_rate(macd_line, signal_line)
                
                if convergence_rate and convergence_rate > 0:
                    base_days = max(1, min(180, int(1 / convergence_rate * 5)))  # 1-180 days range
                    # Apply learned adjustment
                    multiplier = self.learned_adjustments.get('macd_convergence_multiplier', 1.0)
                    estimated_days = int(base_days * multiplier)
                    estimated_days = max(1, min(180, estimated_days))  # Keep in reasonable range but allow freedom
                    estimated_date = datetime.now() + timedelta(days=estimated_days)
                    # Higher confidence if closer to crossover
                    if macd_diff_abs < 0.1:
                        confidence = max(0, min(75, 40 + (0.1 - macd_diff_abs) * 350))
                    else:
                        confidence = max(0, min(65, 35 + (0.5 - macd_diff_abs) * 60))
                    
                    return {
                        'current_trend': current_trend,
                        'predicted_change': 'bullish_reversal',
                        'estimated_days': int(estimated_days),
                        'estimated_date': estimated_date.isoformat(),
                        'confidence': float(confidence),
                        'reasoning': f"MACD {'converging with' if is_converging else 'near'} signal line (diff: {macd_diff:.3f}) - potential bullish crossover in ~{estimated_days} days",
                        'key_indicators': {'macd': float(macd), 'macd_signal': float(macd_signal), 'macd_diff': float(macd_diff)}
                    }
            
            elif current_trend == 'uptrend' and macd_diff > 0 and (macd_diff_abs < 0.5 or is_converging):
                # MACD approaching signal from above - potential bearish crossover
                convergence_rate = self._calculate_macd_convergence_rate(macd_line, signal_line)
                
                if convergence_rate and convergence_rate > 0:
                    base_days = max(1, min(180, int(1 / convergence_rate * 5)))
                    # Apply learned adjustment
                    multiplier = self.learned_adjustments.get('macd_convergence_multiplier', 1.0)
                    estimated_days = int(base_days * multiplier)
                    estimated_days = max(1, min(180, estimated_days))  # Keep in reasonable range but allow freedom
                    estimated_date = datetime.now() + timedelta(days=estimated_days)
                    # Higher confidence if closer to crossover
                    if macd_diff_abs < 0.1:
                        confidence = max(0, min(75, 40 + (0.1 - macd_diff_abs) * 350))
                    else:
                        confidence = max(0, min(65, 35 + (0.5 - macd_diff_abs) * 60))
                    
                    return {
                        'current_trend': current_trend,
                        'predicted_change': 'bearish_reversal',
                        'estimated_days': int(estimated_days),
                        'estimated_date': estimated_date.isoformat(),
                        'confidence': float(confidence),
                        'reasoning': f"MACD {'converging with' if is_converging else 'near'} signal line (diff: {macd_diff:.3f}) - potential bearish crossover in ~{estimated_days} days",
                        'key_indicators': {'macd': float(macd), 'macd_signal': float(macd_signal), 'macd_diff': float(macd_diff)}
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in MACD crossover prediction: {e}")
            return None
    
    def _predict_ema_crossover(self, df: pd.DataFrame, indicators: dict, 
                              current_trend: str) -> Optional[Dict]:
        """Predict EMA crossover trend change"""
        try:
            ema_20 = indicators.get('ema_20', 0)
            ema_50 = indicators.get('ema_50', 0)
            current_price = df['close'].iloc[-1]
            
            if ema_20 == 0 or ema_50 == 0:
                return None
            
            # Calculate EMA history
            from ta.trend import EMAIndicator
            ema_20_ind = EMAIndicator(close=df['close'], window=20)
            ema_50_ind = EMAIndicator(close=df['close'], window=50)
            ema_20_vals = ema_20_ind.ema_indicator().dropna()
            ema_50_vals = ema_50_ind.ema_indicator().dropna()
            
            if len(ema_20_vals) < 20 or len(ema_50_vals) < 20:
                return None
            
            # Check for approaching crossover
            ema_diff = ema_20 - ema_50
            ema_diff_pct = (ema_diff / ema_50) * 100 if ema_50 > 0 else 0
            
            # Check if EMAs are converging (trending towards each other)
            ema_20_trend = ema_20_vals.iloc[-1] - ema_20_vals.iloc[-5] if len(ema_20_vals) >= 5 else 0
            ema_50_trend = ema_50_vals.iloc[-1] - ema_50_vals.iloc[-5] if len(ema_50_vals) >= 5 else 0
            is_ema_converging = (ema_diff < 0 and ema_20_trend > ema_50_trend) or (ema_diff > 0 and ema_20_trend < ema_50_trend)
            
            # Golden Cross potential (EMA 20 crossing above EMA 50)
            # More lenient: check if converging or within 5% of each other
            if current_trend == 'downtrend' and ema_diff < 0 and (abs(ema_diff_pct) < 5 or is_ema_converging):
                # Estimate based on convergence rate
                convergence_rate = self._calculate_ema_convergence_rate(ema_20_vals, ema_50_vals)
                
                if convergence_rate and convergence_rate > 0:
                    base_days = max(1, min(180, int(1 / convergence_rate * 10)))
                    # Apply learned adjustment
                    multiplier = self.learned_adjustments.get('ema_convergence_multiplier', 1.0)
                    estimated_days = int(base_days * multiplier)
                    estimated_days = max(1, min(180, estimated_days))  # Keep in reasonable range but allow freedom
                    estimated_date = datetime.now() + timedelta(days=estimated_days)
                    # Higher confidence if closer together
                    if abs(ema_diff_pct) < 2:
                        confidence = max(0, min(80, 50 + (2 - abs(ema_diff_pct)) * 15))
                    else:
                        confidence = max(0, min(65, 40 + (5 - abs(ema_diff_pct)) * 5))
                    
                    return {
                        'current_trend': current_trend,
                        'predicted_change': 'bullish_reversal',
                        'estimated_days': int(estimated_days),
                        'estimated_date': estimated_date.isoformat(),
                        'confidence': float(confidence),
                        'reasoning': f"EMA 20 {'approaching' if abs(ema_diff_pct) < 2 else 'converging towards'} EMA 50 (Golden Cross potential, {abs(ema_diff_pct):.1f}% apart) - estimated ~{estimated_days} days",
                        'key_indicators': {'ema_20': float(ema_20), 'ema_50': float(ema_50), 'ema_diff_pct': float(ema_diff_pct)}
                    }
            
            # Death Cross potential (EMA 20 crossing below EMA 50)
            elif current_trend == 'uptrend' and ema_diff > 0 and (abs(ema_diff_pct) < 5 or is_ema_converging):
                convergence_rate = self._calculate_ema_convergence_rate(ema_20_vals, ema_50_vals)
                
                if convergence_rate and convergence_rate > 0:
                    base_days = max(1, min(180, int(1 / convergence_rate * 10)))
                    # Apply learned adjustment
                    multiplier = self.learned_adjustments.get('ema_convergence_multiplier', 1.0)
                    estimated_days = int(base_days * multiplier)
                    estimated_days = max(1, min(180, estimated_days))  # Keep in reasonable range but allow freedom
                    estimated_date = datetime.now() + timedelta(days=estimated_days)
                    # Higher confidence if closer together
                    if abs(ema_diff_pct) < 2:
                        confidence = max(0, min(80, 50 + (2 - abs(ema_diff_pct)) * 15))
                    else:
                        confidence = max(0, min(65, 40 + (5 - abs(ema_diff_pct)) * 5))
                    
                    return {
                        'current_trend': current_trend,
                        'predicted_change': 'bearish_reversal',
                        'estimated_days': int(estimated_days),
                        'estimated_date': estimated_date.isoformat(),
                        'confidence': float(confidence),
                        'reasoning': f"EMA 20 {'approaching' if abs(ema_diff_pct) < 2 else 'converging towards'} EMA 50 (Death Cross potential, {abs(ema_diff_pct):.1f}% apart) - estimated ~{estimated_days} days",
                        'key_indicators': {'ema_20': float(ema_20), 'ema_50': float(ema_50), 'ema_diff_pct': float(ema_diff_pct)}
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in EMA crossover prediction: {e}")
            return None
    
    def _predict_momentum_divergence(self, df: pd.DataFrame, indicators: dict, 
                                    current_trend: str) -> Optional[Dict]:
        """Predict trend change based on momentum divergence"""
        try:
            # Look for price vs momentum divergence (price making new highs/lows but momentum weakening)
            prices = df['close'].tail(20).values
            if len(prices) < 10:
                return None
            
            # Calculate momentum (rate of change)
            momentum = np.diff(prices) / prices[:-1] * 100
            
            # Check for bearish divergence (price up, momentum down)
            if current_trend == 'uptrend':
                recent_price_trend = np.mean(prices[-5:]) > np.mean(prices[-10:-5])
                recent_momentum = np.mean(momentum[-5:]) < np.mean(momentum[-10:-5]) if len(momentum) >= 10 else False
                
                if recent_price_trend and recent_momentum:
                    # Bearish divergence detected
                    # Bearish divergence detected
                    # Calculate estimated days based on divergence strength (dynamic)
                    divergence_strength = abs(np.mean(momentum[-5:]) - np.mean(momentum[-10:-5]))
                    estimated_days = max(3, min(60, int(15 - divergence_strength * 2)))  # 3-60 days dynamic range
                    estimated_date = datetime.now() + timedelta(days=estimated_days)
                    
                    return {
                        'current_trend': current_trend,
                        'predicted_change': 'bearish_reversal',
                        'estimated_days': int(estimated_days),
                        'estimated_date': estimated_date.isoformat(),
                        'confidence': float(65),
                        'reasoning': "Bearish divergence detected - price rising but momentum weakening, potential reversal in ~10 days",
                        'key_indicators': {'divergence_type': 'bearish'}
                    }
            
            # Check for bullish divergence (price down, momentum up)
            elif current_trend == 'downtrend':
                recent_price_trend = np.mean(prices[-5:]) < np.mean(prices[-10:-5])
                recent_momentum = np.mean(momentum[-5:]) > np.mean(momentum[-10:-5]) if len(momentum) >= 10 else False
                
                if recent_price_trend and recent_momentum:
                    # Bullish divergence detected
                    # Bullish divergence detected
                    # Calculate estimated days based on divergence strength (dynamic)
                    divergence_strength = abs(np.mean(momentum[-5:]) - np.mean(momentum[-10:-5]))
                    estimated_days = max(3, min(60, int(15 - divergence_strength * 2)))  # 3-60 days dynamic range
                    estimated_date = datetime.now() + timedelta(days=estimated_days)
                    
                    return {
                        'current_trend': current_trend,
                        'predicted_change': 'bullish_reversal',
                        'estimated_days': int(estimated_days),
                        'estimated_date': estimated_date.isoformat(),
                        'confidence': float(65),
                        'reasoning': "Bullish divergence detected - price falling but momentum improving, potential reversal in ~10 days",
                        'key_indicators': {'divergence_type': 'bullish'}
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in momentum divergence prediction: {e}")
            return None
    
    def _predict_support_resistance_break(self, df: pd.DataFrame, price_action: dict, 
                                          current_trend: str) -> Optional[Dict]:
        """Predict trend change based on support/resistance break"""
        try:
            current_price = df['close'].iloc[-1]
            support = price_action.get('recent_low', current_price * 0.95)
            resistance = price_action.get('recent_high', current_price * 1.05)
            
            # Check distance to support/resistance
            distance_to_support = ((current_price - support) / support) * 100
            distance_to_resistance = ((resistance - current_price) / current_price) * 100
            
            # More lenient: check if price is approaching support/resistance (within 5%)
            # If very close to support in downtrend, potential bounce
            if current_trend == 'downtrend' and distance_to_support < 5:
                base_days = 3 if distance_to_support < 2 else 7  # Quick bounce if very close, slower if approaching
                adjustment = self.learned_adjustments.get('support_resistance_days_adjustment', 0)
                estimated_days = max(1, base_days + adjustment)
                estimated_date = datetime.now() + timedelta(days=estimated_days)
                # Higher confidence if closer to support
                confidence = 70 if distance_to_support < 2 else max(55, 70 - (distance_to_support - 2) * 5)
                
                return {
                    'current_trend': current_trend,
                    'predicted_change': 'bullish_reversal',
                    'estimated_days': int(estimated_days),
                    'estimated_date': estimated_date.isoformat(),
                    'confidence': float(confidence),
                    'reasoning': f"Price {'near' if distance_to_support < 2 else 'approaching'} support level ({distance_to_support:.1f}% away) - potential bounce in ~{estimated_days} days",
                    'key_indicators': {'distance_to_support': float(distance_to_support)}
                }
            
            # If very close to resistance in uptrend, potential rejection
            elif current_trend == 'uptrend' and distance_to_resistance < 5:
                base_days = 3 if distance_to_resistance < 2 else 7  # Quick rejection if very close, slower if approaching
                adjustment = self.learned_adjustments.get('support_resistance_days_adjustment', 0)
                estimated_days = max(1, base_days + adjustment)
                estimated_date = datetime.now() + timedelta(days=estimated_days)
                # Higher confidence if closer to resistance
                confidence = 70 if distance_to_resistance < 2 else max(55, 70 - (distance_to_resistance - 2) * 5)
                
                return {
                    'current_trend': current_trend,
                    'predicted_change': 'bearish_reversal',
                    'estimated_days': int(estimated_days),
                    'estimated_date': estimated_date.isoformat(),
                    'confidence': float(confidence),
                    'reasoning': f"Price {'near' if distance_to_resistance < 2 else 'approaching'} resistance level ({distance_to_resistance:.1f}% away) - potential rejection in ~{estimated_days} days",
                    'key_indicators': {'distance_to_resistance': float(distance_to_resistance)}
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in support/resistance prediction: {e}")
            return None
    
    def _predict_trend_continuation(self, df: pd.DataFrame, indicators: dict, 
                                   current_trend: str) -> Optional[Dict]:
        """Predict trend continuation for strong trends"""
        try:
            rsi = indicators.get('rsi', 50)
            macd_diff = indicators.get('macd_diff', 0)
            ema_20 = indicators.get('ema_20', 0)
            ema_50 = indicators.get('ema_50', 0)
            
            # Strong uptrend continuation
            if current_trend == 'uptrend':
                # Check for strong bullish momentum
                if (rsi > 50 and rsi < 70 and  # Healthy RSI (not overbought)
                    macd_diff > 0 and  # MACD bullish
                    ema_20 > ema_50):  # Price above EMAs
                    
                    # Estimate continuation based on momentum strength
                    momentum_strength = (rsi - 50) / 20  # 0-1 scale
                    estimated_days = max(5, min(20, int(10 + momentum_strength * 10)))
                    estimated_date = datetime.now() + timedelta(days=estimated_days)
                    confidence = min(70, 50 + (rsi - 50) * 0.4)
                    
                    return {
                        'current_trend': current_trend,
                        'predicted_change': 'continuation',
                        'estimated_days': int(estimated_days),
                        'estimated_date': estimated_date.isoformat(),
                        'confidence': float(confidence),
                        'reasoning': f"Strong uptrend continuation expected - RSI {rsi:.1f}, MACD bullish, price above EMAs - estimated ~{estimated_days} days",
                        'key_indicators': {'rsi': float(rsi), 'macd_diff': float(macd_diff), 'trend_strength': 'strong'}
                    }
            
            # Strong downtrend continuation
            elif current_trend == 'downtrend':
                # Check for strong bearish momentum
                if (rsi < 50 and rsi > 30 and  # Healthy RSI (not oversold)
                    macd_diff < 0 and  # MACD bearish
                    ema_20 < ema_50):  # Price below EMAs
                    
                    # Estimate continuation based on momentum strength
                    momentum_strength = (50 - rsi) / 20  # 0-1 scale
                    estimated_days = max(5, min(20, int(10 + momentum_strength * 10)))
                    estimated_date = datetime.now() + timedelta(days=estimated_days)
                    confidence = min(70, 50 + (50 - rsi) * 0.4)
                    
                    return {
                        'current_trend': current_trend,
                        'predicted_change': 'continuation',
                        'estimated_days': int(estimated_days),
                        'estimated_date': estimated_date.isoformat(),
                        'confidence': float(confidence),
                        'reasoning': f"Strong downtrend continuation expected - RSI {rsi:.1f}, MACD bearish, price below EMAs - estimated ~{estimated_days} days",
                        'key_indicators': {'rsi': float(rsi), 'macd_diff': float(macd_diff), 'trend_strength': 'strong'}
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in trend continuation prediction: {e}")
            return None
    
    # Helper methods
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
                'price_prediction_adjustment': 1.0,  # Adjustment factor for price predictions
                'verified_count': 0
            }
        
        try:
            with open(self.learning_data_file, 'r') as f:
                data = json.load(f)
                # Ensure price_prediction_adjustment exists
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
                'price_prediction_adjustment': 1.0,  # Adjustment factor for price predictions
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
        """Learn from verified predictions to improve future estimates (days and price accuracy)"""
        if not verified_predictions:
            return
        
        # Group by prediction type
        rsi_predictions = []
        macd_predictions = []
        ema_predictions = []
        divergence_predictions = []
        support_resistance_predictions = []
        
        for pred in verified_predictions:
            if not pred.get('verified') or pred.get('was_correct') is None:
                continue
            
            reasoning = pred.get('reasoning', '')
            estimated_days = pred.get('estimated_days', 0)
            
            # Calculate actual days (if available)
            try:
                estimated_date = datetime.fromisoformat(pred.get('estimated_date', ''))
                verification_date = datetime.fromisoformat(pred.get('verification_date', ''))
                actual_days = (verification_date - estimated_date).days
            except:
                actual_days = None
            
            # Get price accuracy data
            actual_low_price = pred.get('actual_low_price')
            actual_high_price = pred.get('actual_high_price')
            current_price_at_verification = pred.get('current_price_at_verification')
            
            # Get predicted price from key indicators or calculate from reasoning
            predicted_low_price = None
            predicted_high_price = None
            
            # Try to extract predicted price from key_indicators or calculate from current price
            key_indicators = pred.get('key_indicators', {})
            predicted_change = pred.get('predicted_change', '')
            
            # Calculate predicted price drop/rise based on prediction type
            if current_price_at_verification and predicted_change:
                if predicted_change == 'bearish_reversal':
                    # Predicted price drop - estimate based on confidence
                    confidence = pred.get('confidence', 50)
                    drop_pct = min(15, max(5, confidence / 5))  # 5-15% drop
                    predicted_low_price = current_price_at_verification * (1 - drop_pct / 100)
                elif predicted_change == 'bullish_reversal':
                    # Price drops before bounce - estimate drop
                    confidence = pred.get('confidence', 50)
                    drop_pct = min(10, max(3, confidence / 7))  # 3-10% drop
                    predicted_low_price = current_price_at_verification * (1 - drop_pct / 100)
            
            # Categorize by prediction type (include price accuracy data)
            pred_data = {
                'estimated_days': estimated_days,
                'actual_days': actual_days,
                'was_correct': pred.get('was_correct', False),
                'predicted_low_price': predicted_low_price,
                'actual_low_price': actual_low_price,
                'predicted_high_price': predicted_high_price,
                'actual_high_price': actual_high_price,
                'current_price': current_price_at_verification
            }
            
            if 'RSI' in reasoning:
                rsi_predictions.append(pred_data)
            elif 'MACD' in reasoning:
                macd_predictions.append(pred_data)
            elif 'EMA' in reasoning:
                ema_predictions.append(pred_data)
            elif 'divergence' in reasoning.lower():
                divergence_predictions.append(pred_data)
            elif 'support' in reasoning.lower() or 'resistance' in reasoning.lower():
                support_resistance_predictions.append(pred_data)
        
        # Calculate adjustments based on accuracy
        # If predictions were consistently too fast/slow, adjust multipliers
        
        # RSI adjustments
        if rsi_predictions:
            correct_rsi = [p for p in rsi_predictions if p['was_correct']]
            if correct_rsi and len(correct_rsi) >= 3:
                avg_estimated = np.mean([p['estimated_days'] for p in correct_rsi])
                avg_actual = np.mean([p['actual_days'] for p in correct_rsi if p['actual_days'] is not None])
                if avg_actual and avg_estimated > 0:
                    ratio = avg_actual / avg_estimated
                    # Smooth adjustment (weighted average with existing multiplier)
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
        
        # Learn from price accuracy (adjust price prediction percentages)
        # Calculate average price accuracy for all prediction types
        all_price_predictions = rsi_predictions + macd_predictions + ema_predictions + divergence_predictions + support_resistance_predictions
        correct_price_predictions = [p for p in all_price_predictions if p['was_correct'] and p.get('predicted_low_price') and p.get('actual_low_price')]
        
        if correct_price_predictions and len(correct_price_predictions) >= 3:
            # Calculate average price prediction accuracy
            price_errors = []
            for p in correct_price_predictions:
                predicted = p['predicted_low_price']
                actual = p['actual_low_price']
                if predicted > 0 and actual > 0:
                    error_pct = abs((predicted - actual) / actual) * 100
                    price_errors.append(error_pct)
            
            if price_errors:
                avg_price_error = np.mean(price_errors)
                # If average error is high, adjust price prediction percentages
                # Store adjustment factor (lower = more conservative predictions)
                if 'price_prediction_adjustment' not in self.learned_adjustments:
                    self.learned_adjustments['price_prediction_adjustment'] = 1.0
                
                # If error is > 20%, make predictions more conservative
                if avg_price_error > 20:
                    adjustment_factor = 0.9  # Reduce predicted drops by 10%
                elif avg_price_error < 10:
                    adjustment_factor = 1.05  # Increase predicted drops by 5% (more aggressive)
                else:
                    adjustment_factor = 1.0  # No adjustment needed
                
                # Smooth adjustment
                self.learned_adjustments['price_prediction_adjustment'] = (
                    0.8 * self.learned_adjustments['price_prediction_adjustment'] + 0.2 * adjustment_factor
                )
                
                logger.info(f"📊 Price prediction accuracy: {avg_price_error:.1f}% average error. Adjustment factor: {self.learned_adjustments['price_prediction_adjustment']:.3f}")
        
        # Update verified count
        self.learned_adjustments['verified_count'] += len(verified_predictions)
        
        # Save learned adjustments
        self._save_learned_adjustments()
        logger.info(f"🧠 Learned from {len(verified_predictions)} verified trend change predictions (days + price accuracy)")
    
    def _calculate_avg_recovery_time(self, periods: List[int], df: pd.DataFrame) -> Optional[float]:
        """Calculate average recovery time from oversold periods"""
        if not periods:
            return None
        
        # Use historical patterns - typically 5-15 days for RSI recovery, but allow freedom
        avg_period = np.mean(periods) if periods else 10
        base_estimate = min(120, max(1, avg_period))
        
        # Apply learned adjustment
        multiplier = self.learned_adjustments.get('rsi_recovery_multiplier', 1.0)
        adjusted = base_estimate * multiplier
        return min(120, max(1, adjusted))
    
    def _calculate_avg_reversal_time(self, periods: List[int], df: pd.DataFrame) -> Optional[float]:
        """Calculate average reversal time from overbought periods"""
        if not periods:
            return None
        
        avg_period = np.mean(periods) if periods else 10
        base_estimate = min(120, max(1, avg_period))
        
        # Apply learned adjustment
        multiplier = self.learned_adjustments.get('rsi_reversal_multiplier', 1.0)
        adjusted = base_estimate * multiplier
        return min(120, max(1, adjusted))
    
    def _calculate_macd_convergence_rate(self, macd_line: pd.Series, 
                                         signal_line: pd.Series) -> Optional[float]:
        """Calculate how fast MACD is converging with signal line"""
        if len(macd_line) < 5 or len(signal_line) < 5:
            return None
        
        # Calculate recent convergence rate
        recent_diff = abs(macd_line.iloc[-1] - signal_line.iloc[-1])
        prev_diff = abs(macd_line.iloc[-5] - signal_line.iloc[-5]) if len(macd_line) >= 5 else recent_diff
        
        if prev_diff == 0:
            return None
        
        convergence_rate = (prev_diff - recent_diff) / prev_diff / 5  # Per day rate
        return max(0.01, min(0.5, convergence_rate))  # Clamp between 1% and 50% per day
    
    def _calculate_ema_convergence_rate(self, ema_20: pd.Series, ema_50: pd.Series) -> Optional[float]:
        """Calculate how fast EMA 20 is converging with EMA 50"""
        if len(ema_20) < 5 or len(ema_50) < 5:
            return None
        
        recent_diff_pct = abs((ema_20.iloc[-1] - ema_50.iloc[-1]) / ema_50.iloc[-1]) * 100
        prev_diff_pct = abs((ema_20.iloc[-5] - ema_50.iloc[-5]) / ema_50.iloc[-5]) * 100 if len(ema_20) >= 5 else recent_diff_pct
        
        if prev_diff_pct == 0:
            return None
        
        convergence_rate = (prev_diff_pct - recent_diff_pct) / prev_diff_pct / 5
        return max(0.01, min(0.3, convergence_rate))

