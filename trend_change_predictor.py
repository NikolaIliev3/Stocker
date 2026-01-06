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
            if current_trend == 'downtrend' and current_rsi < 30:
                # Oversold in downtrend - potential bullish reversal
                # Estimate based on historical RSI recovery patterns
                oversold_periods = self._find_oversold_periods(rsi_values)
                avg_recovery_days = self._calculate_avg_recovery_time(oversold_periods, df)
                
                if avg_recovery_days and avg_recovery_days > 0:
                    estimated_date = datetime.now() + timedelta(days=int(avg_recovery_days))
                    confidence = min(85, 50 + (30 - current_rsi) * 2)  # Lower RSI = higher confidence
                    
                    return {
                        'current_trend': current_trend,
                        'predicted_change': 'bullish_reversal',
                        'estimated_days': int(avg_recovery_days),
                        'estimated_date': estimated_date.isoformat(),
                        'confidence': confidence,
                        'reasoning': f"RSI oversold ({current_rsi:.1f}) in downtrend - historical patterns suggest bullish reversal in ~{int(avg_recovery_days)} days",
                        'key_indicators': {'rsi': current_rsi, 'rsi_status': 'oversold'}
                    }
            
            elif current_trend == 'uptrend' and current_rsi > 70:
                # Overbought in uptrend - potential bearish reversal
                overbought_periods = self._find_overbought_periods(rsi_values)
                avg_reversal_days = self._calculate_avg_reversal_time(overbought_periods, df)
                
                if avg_reversal_days and avg_reversal_days > 0:
                    estimated_date = datetime.now() + timedelta(days=int(avg_reversal_days))
                    confidence = min(85, 50 + (current_rsi - 70) * 2)  # Higher RSI = higher confidence
                    
                    return {
                        'current_trend': current_trend,
                        'predicted_change': 'bearish_reversal',
                        'estimated_days': int(avg_reversal_days),
                        'estimated_date': estimated_date.isoformat(),
                        'confidence': confidence,
                        'reasoning': f"RSI overbought ({current_rsi:.1f}) in uptrend - historical patterns suggest bearish reversal in ~{int(avg_reversal_days)} days",
                        'key_indicators': {'rsi': current_rsi, 'rsi_status': 'overbought'}
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
            if current_trend == 'downtrend' and macd_diff < 0 and abs(macd_diff) < 0.1:
                # MACD approaching signal from below - potential bullish crossover
                # Estimate based on convergence rate
                convergence_rate = self._calculate_macd_convergence_rate(macd_line, signal_line)
                
                if convergence_rate and convergence_rate > 0:
                    base_days = max(3, min(30, int(1 / convergence_rate * 5)))  # 3-30 days range
                    # Apply learned adjustment
                    multiplier = self.learned_adjustments.get('macd_convergence_multiplier', 1.0)
                    estimated_days = int(base_days * multiplier)
                    estimated_days = max(3, min(30, estimated_days))  # Keep in range
                    estimated_date = datetime.now() + timedelta(days=estimated_days)
                    confidence = min(75, 40 + abs(macd_diff) * 100)  # Closer to crossover = higher confidence
                    
                    return {
                        'current_trend': current_trend,
                        'predicted_change': 'bullish_reversal',
                        'estimated_days': estimated_days,
                        'estimated_date': estimated_date.isoformat(),
                        'confidence': confidence,
                        'reasoning': f"MACD converging with signal line (diff: {macd_diff:.3f}) - potential bullish crossover in ~{estimated_days} days",
                        'key_indicators': {'macd': macd, 'macd_signal': macd_signal, 'macd_diff': macd_diff}
                    }
            
            elif current_trend == 'uptrend' and macd_diff > 0 and abs(macd_diff) < 0.1:
                # MACD approaching signal from above - potential bearish crossover
                convergence_rate = self._calculate_macd_convergence_rate(macd_line, signal_line)
                
                if convergence_rate and convergence_rate > 0:
                    base_days = max(3, min(30, int(1 / convergence_rate * 5)))
                    # Apply learned adjustment
                    multiplier = self.learned_adjustments.get('macd_convergence_multiplier', 1.0)
                    estimated_days = int(base_days * multiplier)
                    estimated_days = max(3, min(30, estimated_days))  # Keep in range
                    estimated_date = datetime.now() + timedelta(days=estimated_days)
                    confidence = min(75, 40 + abs(macd_diff) * 100)
                    
                    return {
                        'current_trend': current_trend,
                        'predicted_change': 'bearish_reversal',
                        'estimated_days': estimated_days,
                        'estimated_date': estimated_date.isoformat(),
                        'confidence': confidence,
                        'reasoning': f"MACD converging with signal line (diff: {macd_diff:.3f}) - potential bearish crossover in ~{estimated_days} days",
                        'key_indicators': {'macd': macd, 'macd_signal': macd_signal, 'macd_diff': macd_diff}
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
            
            # Golden Cross potential (EMA 20 crossing above EMA 50)
            if current_trend == 'downtrend' and ema_diff < 0 and abs(ema_diff_pct) < 2:
                # Estimate based on convergence rate
                convergence_rate = self._calculate_ema_convergence_rate(ema_20_vals, ema_50_vals)
                
                if convergence_rate and convergence_rate > 0:
                    base_days = max(5, min(40, int(1 / convergence_rate * 10)))
                    # Apply learned adjustment
                    multiplier = self.learned_adjustments.get('ema_convergence_multiplier', 1.0)
                    estimated_days = int(base_days * multiplier)
                    estimated_days = max(5, min(40, estimated_days))  # Keep in range
                    estimated_date = datetime.now() + timedelta(days=estimated_days)
                    confidence = min(80, 50 + abs(ema_diff_pct) * 10)
                    
                    return {
                        'current_trend': current_trend,
                        'predicted_change': 'bullish_reversal',
                        'estimated_days': estimated_days,
                        'estimated_date': estimated_date.isoformat(),
                        'confidence': confidence,
                        'reasoning': f"EMA 20 approaching EMA 50 (Golden Cross potential) - estimated ~{estimated_days} days",
                        'key_indicators': {'ema_20': ema_20, 'ema_50': ema_50, 'ema_diff_pct': ema_diff_pct}
                    }
            
            # Death Cross potential (EMA 20 crossing below EMA 50)
            elif current_trend == 'uptrend' and ema_diff > 0 and abs(ema_diff_pct) < 2:
                convergence_rate = self._calculate_ema_convergence_rate(ema_20_vals, ema_50_vals)
                
                if convergence_rate and convergence_rate > 0:
                    base_days = max(5, min(40, int(1 / convergence_rate * 10)))
                    # Apply learned adjustment
                    multiplier = self.learned_adjustments.get('ema_convergence_multiplier', 1.0)
                    estimated_days = int(base_days * multiplier)
                    estimated_days = max(5, min(40, estimated_days))  # Keep in range
                    estimated_date = datetime.now() + timedelta(days=estimated_days)
                    confidence = min(80, 50 + abs(ema_diff_pct) * 10)
                    
                    return {
                        'current_trend': current_trend,
                        'predicted_change': 'bearish_reversal',
                        'estimated_days': estimated_days,
                        'estimated_date': estimated_date.isoformat(),
                        'confidence': confidence,
                        'reasoning': f"EMA 20 approaching EMA 50 (Death Cross potential) - estimated ~{estimated_days} days",
                        'key_indicators': {'ema_20': ema_20, 'ema_50': ema_50, 'ema_diff_pct': ema_diff_pct}
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
                    estimated_days = 10  # Typical divergence to reversal time
                    estimated_date = datetime.now() + timedelta(days=estimated_days)
                    
                    return {
                        'current_trend': current_trend,
                        'predicted_change': 'bearish_reversal',
                        'estimated_days': estimated_days,
                        'estimated_date': estimated_date.isoformat(),
                        'confidence': 65,
                        'reasoning': "Bearish divergence detected - price rising but momentum weakening, potential reversal in ~10 days",
                        'key_indicators': {'divergence_type': 'bearish'}
                    }
            
            # Check for bullish divergence (price down, momentum up)
            elif current_trend == 'downtrend':
                recent_price_trend = np.mean(prices[-5:]) < np.mean(prices[-10:-5])
                recent_momentum = np.mean(momentum[-5:]) > np.mean(momentum[-10:-5]) if len(momentum) >= 10 else False
                
                if recent_price_trend and recent_momentum:
                    # Bullish divergence detected
                    estimated_days = 10
                    estimated_date = datetime.now() + timedelta(days=estimated_days)
                    
                    return {
                        'current_trend': current_trend,
                        'predicted_change': 'bullish_reversal',
                        'estimated_days': estimated_days,
                        'estimated_date': estimated_date.isoformat(),
                        'confidence': 65,
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
            
            # If very close to support in downtrend, potential bounce
            if current_trend == 'downtrend' and distance_to_support < 2:
                base_days = 3  # Quick bounce if support holds
                adjustment = self.learned_adjustments.get('support_resistance_days_adjustment', 0)
                estimated_days = max(1, base_days + adjustment)
                estimated_date = datetime.now() + timedelta(days=estimated_days)
                
                return {
                    'current_trend': current_trend,
                    'predicted_change': 'bullish_reversal',
                    'estimated_days': estimated_days,
                    'estimated_date': estimated_date.isoformat(),
                    'confidence': 70,
                    'reasoning': f"Price near support level ({distance_to_support:.1f}% away) - potential bounce in ~{estimated_days} days",
                    'key_indicators': {'distance_to_support': distance_to_support}
                }
            
            # If very close to resistance in uptrend, potential rejection
            elif current_trend == 'uptrend' and distance_to_resistance < 2:
                base_days = 3
                adjustment = self.learned_adjustments.get('support_resistance_days_adjustment', 0)
                estimated_days = max(1, base_days + adjustment)
                estimated_date = datetime.now() + timedelta(days=estimated_days)
                
                return {
                    'current_trend': current_trend,
                    'predicted_change': 'bearish_reversal',
                    'estimated_days': estimated_days,
                    'estimated_date': estimated_date.isoformat(),
                    'confidence': 70,
                    'reasoning': f"Price near resistance level ({distance_to_resistance:.1f}% away) - potential rejection in ~{estimated_days} days",
                    'key_indicators': {'distance_to_resistance': distance_to_resistance}
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in support/resistance prediction: {e}")
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
    
    def _calculate_avg_recovery_time(self, periods: List[int], df: pd.DataFrame) -> Optional[float]:
        """Calculate average recovery time from oversold periods"""
        if not periods:
            return None
        
        # Use historical patterns - typically 5-15 days for RSI recovery
        avg_period = np.mean(periods) if periods else 10
        return min(30, max(3, avg_period))
    
    def _calculate_avg_reversal_time(self, periods: List[int], df: pd.DataFrame) -> Optional[float]:
        """Calculate average reversal time from overbought periods"""
        if not periods:
            return None
        
        avg_period = np.mean(periods) if periods else 10
        return min(30, max(3, avg_period))
    
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

