"""
Machine Learning Training System
Trains ML models on historical stock data to improve predictions
"""
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif

# Suppress harmless feature name warnings from LightGBM/XGBoost
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')

# Try to import advanced ML libraries
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger = logging.getLogger(__name__)
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    logger = logging.getLogger(__name__)
    logger.warning("LightGBM not available. Install with: pip install lightgbm")

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    logger = logging.getLogger(__name__)
    logger.warning("Optuna not available. Install with: pip install optuna")
import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

from secure_ml_storage import SecureMLStorage
try:
    from config import ML_TRAINING_CPU_CORES, HYPERPARAMETER_TUNING_PARALLEL_TRIALS
except ImportError:
    # Fallback if not in config
    import multiprocessing
    import sys
    # Windows-specific: limit to 4 cores to prevent deadlocks
    if sys.platform == 'win32':
        ML_TRAINING_CPU_CORES = min(4, multiprocessing.cpu_count())
    else:
        ML_TRAINING_CPU_CORES = -1  # Use all cores on non-Windows
    HYPERPARAMETER_TUNING_PARALLEL_TRIALS = None

# Helper function to safely get n_jobs value (Windows-specific fix)
import sys
import multiprocessing

def get_safe_n_jobs(requested_cores):
    """Get safe n_jobs value, accounting for Windows deadlock issues"""
    if sys.platform == 'win32':
        # Windows: limit to max 4 cores to prevent multiprocessing deadlocks
        if requested_cores == -1:
            return min(4, multiprocessing.cpu_count())
        else:
            return min(4, requested_cores, multiprocessing.cpu_count())
    else:
        # Non-Windows: use requested value
        if requested_cores == -1:
            return -1  # Use all cores
        else:
            return min(requested_cores, multiprocessing.cpu_count())

# Import enhanced features if available
try:
    from enhanced_features import EnhancedFeatureExtractor as EnhancedExtractor
    HAS_ENHANCED_FEATURES = True
except ImportError:
    HAS_ENHANCED_FEATURES = False
    logger = logging.getLogger(__name__)
    logger.debug("Enhanced features module not available")

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts ML features from stock data and indicators"""
    
    def __init__(self, data_fetcher=None):
        """Initialize feature extractor with optional data fetcher for enhanced features"""
        self.data_fetcher = data_fetcher
        if HAS_ENHANCED_FEATURES and data_fetcher:
            try:
                self.enhanced_extractor = EnhancedExtractor(data_fetcher)
            except Exception as e:
                logger.warning(f"Could not initialize enhanced extractor: {e}")
                self.enhanced_extractor = None
        else:
            self.enhanced_extractor = None
    
    def extract_features(self, stock_data: dict, history_data: dict, 
                        financials_data: dict = None, indicators: dict = None,
                        market_regime: dict = None, timeframe_analysis: dict = None,
                        relative_strength: dict = None, support_resistance: dict = None,
                        news_data: list = None) -> np.ndarray:
        """Extract comprehensive feature vector for ML model with enhanced features"""
        features = []
        
        # Technical Indicators (26 features)
        if indicators:
            features.extend([
                indicators.get('rsi', 50),
                indicators.get('macd', 0),
                indicators.get('macd_signal', 0),
                indicators.get('macd_diff', 0),
                indicators.get('ema_20', 0),
                indicators.get('ema_50', 0),
                indicators.get('ema_200', 0),
                indicators.get('bb_upper', 0),
                indicators.get('bb_lower', 0),
                indicators.get('bb_middle', 0),
                indicators.get('atr', 0),
                indicators.get('atr_percent', 0),
                indicators.get('stoch_k', 50),
                indicators.get('stoch_d', 50),
                indicators.get('williams_r', -50),
                indicators.get('adx', 25),
                indicators.get('adx_pos', 0),
                indicators.get('adx_neg', 0),
                indicators.get('obv', 0),
            ])
            
            # Boolean features (convert to 0/1)
            features.extend([
                1 if indicators.get('ema_20_above_50', False) else 0,
                1 if indicators.get('ema_50_above_200', False) else 0,
                1 if indicators.get('golden_cross', False) else 0,
                1 if indicators.get('death_cross', False) else 0,
                1 if indicators.get('price_above_ema20', False) else 0,
                1 if indicators.get('price_above_ema50', False) else 0,
                1 if indicators.get('price_above_ema200', False) else 0,
            ])
        else:
            features.extend([0] * 26)
        
        # Price Action Features (8 features)
        if history_data and history_data.get('data'):
            try:
                df = pd.DataFrame(history_data['data'])
                if len(df) > 0 and 'close' in df.columns:
                    prices = df['close'].values
                    volumes = df['volume'].values if 'volume' in df.columns else []
                    
                    # Price momentum
                    if len(prices) >= 5:
                        features.append((prices[-1] - prices[-5]) / prices[-5] if prices[-5] > 0 else 0)
                    else:
                        features.append(0)
                    
                    if len(prices) >= 10:
                        features.append((prices[-1] - prices[-10]) / prices[-10] if prices[-10] > 0 else 0)
                    else:
                        features.append(0)
                    
                    if len(prices) >= 20:
                        features.append((prices[-1] - prices[-20]) / prices[-20] if prices[-20] > 0 else 0)
                    else:
                        features.append(0)
                    
                    # Volatility
                    if len(prices) >= 20:
                        features.append(np.std(prices[-20:]) / np.mean(prices[-20:]) if np.mean(prices[-20:]) > 0 else 0)
                    else:
                        features.append(0)
                    
                    # Volume features
                    if len(volumes) >= 20:
                        avg_volume = np.mean(volumes[-20:])
                        current_volume = volumes[-1] if len(volumes) > 0 else 0
                        features.append(current_volume / avg_volume if avg_volume > 0 else 1)
                        features.append(np.std(volumes[-20:]) / avg_volume if avg_volume > 0 else 0)
                    else:
                        features.extend([1, 0])
                    
                    # Price position in range
                    if len(prices) >= 20:
                        high_20 = np.max(prices[-20:])
                        low_20 = np.min(prices[-20:])
                        features.append((prices[-1] - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5)
                    else:
                        features.append(0.5)
                    
                    # Trend strength
                    if len(prices) >= 20:
                        x = np.arange(len(prices[-20:]))
                        slope = np.polyfit(x, prices[-20:], 1)[0]
                        features.append(slope / prices[-1] if prices[-1] > 0 else 0)
                    else:
                        features.append(0)
                else:
                    features.extend([0] * 8)
            except Exception as e:
                logger.warning(f"Error extracting price features: {e}")
                features.extend([0] * 8)
        else:
            features.extend([0] * 8)
        
        # Fundamental Features (10 features)
        if financials_data:
            info = financials_data.get('info', {})
            features.extend([
                info.get('trailingPE', 0) or info.get('pe_ratio', 0) or 0,
                info.get('forwardPE', 0) or 0,
                info.get('pegRatio', 0) or 0,
                info.get('priceToBook', 0) or 0,
                info.get('priceToSalesTrailing12Months', 0) or 0,
                info.get('dividendYield', 0) or 0,
                info.get('debtToEquity', 0) or 0,
                info.get('returnOnEquity', 0) or 0,
                info.get('returnOnAssets', 0) or 0,
                info.get('profitMargins', 0) or 0,
            ])
        else:
            features.extend([0] * 10)
        
        # Market cap (log scale)
        market_cap = stock_data.get('market_cap', 0) or stock_data.get('info', {}).get('marketCap', 0) or 0
        features.append(np.log10(market_cap + 1))
        
        # Current price
        current_price = stock_data.get('price', 0)
        features.append(current_price)
        
        # ===== ENHANCED FEATURES =====
        # Extract additional enhanced features if available
        if self.enhanced_extractor:
            try:
                enhanced_features = self.enhanced_extractor.extract_all_enhanced_features(
                    stock_data, history_data, news_data
                )
                # Add enhanced features to feature vector
                for key, value in enhanced_features.items():
                    if isinstance(value, (int, float)):
                        features.append(float(value))
                    elif isinstance(value, bool):
                        features.append(1.0 if value else 0.0)
                    else:
                        features.append(0.0)
            except Exception as e:
                logger.debug(f"Could not extract enhanced features: {e}")
        
        # 1. Market Regime Indicator (3 features: bull, bear, sideways as one-hot encoded)
        if market_regime:
            regime = market_regime.get('regime', 'unknown')
            regime_strength = market_regime.get('strength', 50) / 100.0  # Normalize to 0-1
            features.extend([
                1.0 if regime == 'bull' else 0.0,
                1.0 if regime == 'bear' else 0.0,
                1.0 if regime == 'sideways' else 0.0,
                regime_strength
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.5])  # Default to neutral
        
        # 2. Relative Strength Ratio (1 feature)
        if relative_strength and relative_strength.get('available', False):
            rs_ratio = relative_strength.get('rs_ratio', 1.0)
            outperformance = relative_strength.get('outperformance', 0) / 100.0  # Normalize to percentage
            features.extend([rs_ratio, outperformance])
        else:
            features.extend([1.0, 0.0])  # Default: neutral (ratio = 1.0, no outperformance)
        
        # 3. Multi-Timeframe Alignment Score (2 features)
        if timeframe_analysis and timeframe_analysis.get('alignment') != 'insufficient_data':
            alignment = timeframe_analysis.get('alignment', 'mixed')
            timeframe_conf = timeframe_analysis.get('confidence', 50) / 100.0  # Normalize to 0-1
            
            # Encode alignment: strong_bullish=1.0, bullish=0.66, mixed=0.5, bearish=0.33, strong_bearish=0.0
            alignment_score = 0.5  # Default: mixed
            if alignment == 'strong_bullish':
                alignment_score = 1.0
            elif alignment == 'bullish':
                alignment_score = 0.66
            elif alignment == 'bearish':
                alignment_score = 0.33
            elif alignment == 'strong_bearish':
                alignment_score = 0.0
            
            features.extend([alignment_score, timeframe_conf])
        else:
            features.extend([0.5, 0.5])  # Default: mixed alignment, 50% confidence
        
        # 4. Volume Profile Features (2 features)
        if support_resistance and support_resistance.get('volume_weighted', False):
            support_strength = support_resistance.get('support_strength', 50) / 100.0  # Normalize to 0-1
            resistance_strength = support_resistance.get('resistance_strength', 50) / 100.0
            features.extend([support_strength, resistance_strength])
        else:
            features.extend([0.5, 0.5])  # Default: medium strength
        
        # 5. Volatility Regime (3 features: high, normal, low as one-hot encoded)
        if history_data and history_data.get('data'):
            try:
                df = pd.DataFrame(history_data['data'])
                if len(df) >= 20 and 'close' in df.columns:
                    prices = df['close'].values[-20:]
                    # Calculate ATR percentage
                    if 'high' in df.columns and 'low' in df.columns:
                        highs = df['high'].values[-20:]
                        lows = df['low'].values[-20:]
                        high_low = highs - lows
                        high_close = np.abs(highs - np.roll(prices, 1))
                        low_close = np.abs(lows - np.roll(prices, 1))
                        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                        atr = np.mean(true_range)
                        atr_percent = (atr / prices[-1] * 100) if prices[-1] > 0 else 0
                        
                        # Classify volatility regime
                        if atr_percent > 3.0:
                            vol_regime = 'high'
                        elif atr_percent < 1.0:
                            vol_regime = 'low'
                        else:
                            vol_regime = 'normal'
                        
                        features.extend([
                            1.0 if vol_regime == 'high' else 0.0,
                            1.0 if vol_regime == 'normal' else 0.0,
                            1.0 if vol_regime == 'low' else 0.0,
                            atr_percent / 10.0  # Normalize (assuming max ~10%)
                        ])
                    else:
                        features.extend([0.0, 1.0, 0.0, 0.0])  # Default: normal volatility
                else:
                    features.extend([0.0, 1.0, 0.0, 0.0])  # Default: normal volatility
            except Exception as e:
                logger.warning(f"Error extracting volatility features: {e}")
                features.extend([0.0, 1.0, 0.0, 0.0])  # Default: normal volatility
        else:
            features.extend([0.0, 1.0, 0.0, 0.0])  # Default: normal volatility
        
        # 6. Support/Resistance Distance Features (2 features)
        if support_resistance:
            current_price = stock_data.get('price', 0)
            support = support_resistance.get('support', 0)
            resistance = support_resistance.get('resistance', 0)
            
            if current_price > 0:
                # Distance to support as percentage
                if support > 0:
                    dist_to_support = abs(current_price - support) / current_price
                else:
                    dist_to_support = 0.0
                
                # Distance to resistance as percentage
                if resistance > 0:
                    dist_to_resistance = abs(resistance - current_price) / current_price
                else:
                    dist_to_resistance = 0.0
                
                features.extend([dist_to_support, dist_to_resistance])
            else:
                features.extend([0.0, 0.0])
        else:
            features.extend([0.0, 0.0])
        
        # ===== ADVANCED TIME-SERIES FEATURES =====
        if history_data and history_data.get('data'):
            try:
                df = pd.DataFrame(history_data['data'])
                # Ensure date column exists
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                else:
                    df['date'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq='D')
                
                # Ensure column names are lowercase (required by extraction methods)
                df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]
                
                if len(df) > 0 and 'close' in df.columns:
                    ts_features = self._extract_time_series_features(df)
                    ts_feature_list = list(ts_features.values())
                    if len(ts_feature_list) > 0:
                        features.extend(ts_feature_list)
                        logger.debug(f"Extracted {len(ts_feature_list)} time-series features")
                    else:
                        # Add default values if extraction returned empty
                        features.extend([0.0] * 58)  # Actual count: 58 TS features
                else:
                    # Add default values for time-series features
                    features.extend([0.0] * 58)  # Actual count: 58 TS features
            except Exception as e:
                logger.warning(f"Error extracting time-series features: {e}")
                features.extend([0.0] * 58)  # Actual count: 58 TS features
        else:
            features.extend([0.0] * 58)  # Actual count: 58 TS features
        
        # ===== PATTERN DETECTION FEATURES =====
        if history_data and history_data.get('data'):
            try:
                df = pd.DataFrame(history_data['data'])
                # Ensure column names are lowercase
                df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]
                
                if len(df) >= 2 and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    pattern_features = self._extract_pattern_features(df)
                    pattern_feature_list = list(pattern_features.values())
                    if len(pattern_feature_list) > 0:
                        features.extend(pattern_feature_list)
                        logger.debug(f"Extracted {len(pattern_feature_list)} pattern features")
                    else:
                        features.extend([0.0] * 10)  # Actual count: 10 pattern features
                else:
                    features.extend([0.0] * 10)  # Actual count: 10 pattern features
            except Exception as e:
                logger.warning(f"Error extracting pattern features: {e}")
                features.extend([0.0] * 10)  # Actual count: 10 pattern features
        else:
            features.extend([0.0] * 10)  # Actual count: 10 pattern features
        
        return np.array(features, dtype=np.float32)
    
    def _extract_time_series_features(self, df: pd.DataFrame) -> Dict:
        """Extract advanced time-series features"""
        features = {}
        
        if len(df) < 2:
            return features
        
        # Ensure we have required columns
        if 'close' not in df.columns:
            return features
        
        close = df['close']
        volume = df['volume'] if 'volume' in df.columns else pd.Series([0] * len(df))
        
        # 1. Lag Features
        for lag in [1, 2, 3, 5, 10, 20]:
            if len(close) > lag:
                features[f'price_lag_{lag}'] = float(close.shift(lag).iloc[-1]) if not pd.isna(close.shift(lag).iloc[-1]) else 0.0
                features[f'volume_lag_{lag}'] = float(volume.shift(lag).iloc[-1]) if len(volume) > lag and not pd.isna(volume.shift(lag).iloc[-1]) else 0.0
                if lag < len(close):
                    pct_change = close.pct_change(lag).iloc[-1]
                    features[f'price_change_lag_{lag}'] = float(pct_change) if not pd.isna(pct_change) else 0.0
            else:
                features[f'price_lag_{lag}'] = 0.0
                features[f'volume_lag_{lag}'] = 0.0
                features[f'price_change_lag_{lag}'] = 0.0
        
        # 2. Rolling Statistics
        for window in [5, 10, 20, 50]:
            if len(close) >= window:
                ma = close.rolling(window).mean().iloc[-1]
                std = close.rolling(window).std().iloc[-1]
                features[f'price_ma_{window}'] = float(ma) if not pd.isna(ma) else 0.0
                features[f'price_std_{window}'] = float(std) if not pd.isna(std) else 0.0
                
                if len(volume) >= window:
                    vol_ma = volume.rolling(window).mean().iloc[-1]
                    features[f'volume_ma_{window}'] = float(vol_ma) if not pd.isna(vol_ma) else 0.0
                else:
                    features[f'volume_ma_{window}'] = 0.0
                
                # Price relative to MA
                current_price = close.iloc[-1]
                if ma > 0:
                    features[f'price_vs_ma_{window}'] = float((current_price - ma) / ma)
                else:
                    features[f'price_vs_ma_{window}'] = 0.0
            else:
                features[f'price_ma_{window}'] = 0.0
                features[f'price_std_{window}'] = 0.0
                features[f'volume_ma_{window}'] = 0.0
                features[f'price_vs_ma_{window}'] = 0.0
        
        # 3. Momentum Features
        for period in [5, 10, 20]:
            if len(close) > period:
                momentum = close.pct_change(period).iloc[-1]
                features[f'momentum_{period}'] = float(momentum) if not pd.isna(momentum) else 0.0
                if period < len(close):
                    roc = ((close.iloc[-1] - close.iloc[-period-1]) / close.iloc[-period-1]) if close.iloc[-period-1] > 0 else 0.0
                    features[f'roc_{period}'] = float(roc)
                else:
                    features[f'roc_{period}'] = 0.0
            else:
                features[f'momentum_{period}'] = 0.0
                features[f'roc_{period}'] = 0.0
        
        # 4. Volatility Features
        returns = close.pct_change().dropna()
        for window in [5, 10, 20]:
            if len(returns) >= window:
                vol = returns.rolling(window).std().iloc[-1]
                features[f'volatility_{window}'] = float(vol) if not pd.isna(vol) else 0.0
                features[f'realized_vol_{window}'] = float(vol * np.sqrt(252)) if not pd.isna(vol) else 0.0
            else:
                features[f'volatility_{window}'] = 0.0
                features[f'realized_vol_{window}'] = 0.0
        
        # 5. Autocorrelation
        for lag in [1, 5, 10]:
            if len(close) > lag * 2:
                try:
                    autocorr = close.autocorr(lag=lag)
                    features[f'autocorr_{lag}'] = float(autocorr) if not pd.isna(autocorr) else 0.0
                except:
                    features[f'autocorr_{lag}'] = 0.0
            else:
                features[f'autocorr_{lag}'] = 0.0
        
        # 6. Price Position in Range
        for window in [10, 20, 50]:
            if len(close) >= window:
                rolling_max = close.rolling(window).max().iloc[-1]
                rolling_min = close.rolling(window).min().iloc[-1]
                current_price = close.iloc[-1]
                
                if rolling_max > rolling_min and not (pd.isna(rolling_max) or pd.isna(rolling_min)):
                    features[f'price_position_{window}'] = float((current_price - rolling_min) / (rolling_max - rolling_min))
                else:
                    features[f'price_position_{window}'] = 0.5
            else:
                features[f'price_position_{window}'] = 0.5
        
        # 7. Trend Strength
        for window in [10, 20]:
            if len(close) >= window:
                window_data = close.tail(window)
                x = np.arange(len(window_data))
                try:
                    slope = np.polyfit(x, window_data.values, 1)[0]
                    features[f'trend_strength_{window}'] = float(slope / window_data.iloc[-1]) if window_data.iloc[-1] > 0 else 0.0
                except:
                    features[f'trend_strength_{window}'] = 0.0
            else:
                features[f'trend_strength_{window}'] = 0.0
        
        # 8. Volume-Price Correlation
        if len(close) > 5 and len(volume) > 5:
            price_changes = close.pct_change().dropna()
            volume_changes = volume.pct_change().dropna()
            aligned = pd.concat([price_changes, volume_changes], axis=1).dropna()
            if len(aligned) > 1:
                try:
                    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                    features['price_volume_correlation'] = float(corr) if not pd.isna(corr) else 0.0
                except:
                    features['price_volume_correlation'] = 0.0
            else:
                features['price_volume_correlation'] = 0.0
        else:
            features['price_volume_correlation'] = 0.0
        
        # 9. Momentum Acceleration
        if len(close) >= 10:
            try:
                roc_5 = close.pct_change(5).iloc[-1]
                roc_10 = close.pct_change(10).iloc[-1]
                if not (pd.isna(roc_5) or pd.isna(roc_10)):
                    features['momentum_acceleration'] = float(roc_5 - roc_10)
                else:
                    features['momentum_acceleration'] = 0.0
            except:
                features['momentum_acceleration'] = 0.0
        else:
            features['momentum_acceleration'] = 0.0
        
        # 10. Support/Resistance Distance (if high/low available)
        if all(col in df.columns for col in ['high', 'low']):
            current_price = close.iloc[-1]
            if len(df) >= 20:
                recent_high = df['high'].tail(20).max()
                recent_low = df['low'].tail(20).min()
                
                if recent_high > recent_low and current_price > 0:
                    features['distance_to_resistance'] = float((recent_high - current_price) / current_price)
                    features['distance_to_support'] = float((current_price - recent_low) / current_price)
                else:
                    features['distance_to_resistance'] = 0.0
                    features['distance_to_support'] = 0.0
            else:
                features['distance_to_resistance'] = 0.0
                features['distance_to_support'] = 0.0
        else:
            features['distance_to_resistance'] = 0.0
            features['distance_to_support'] = 0.0
        
        return features
    
    def _extract_pattern_features(self, df: pd.DataFrame) -> Dict:
        """Extract pattern detection features"""
        features = {}
        
        if len(df) < 2 or not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return features
        
        # Candlestick patterns
        open_price = df['open'].iloc[-1]
        close_price = df['close'].iloc[-1]
        high_price = df['high'].iloc[-1]
        low_price = df['low'].iloc[-1]
        
        body = abs(close_price - open_price)
        upper_shadow = high_price - max(open_price, close_price)
        lower_shadow = min(open_price, close_price) - low_price
        total_range = high_price - low_price
        
        # Doji
        features['doji'] = 1.0 if total_range > 0 and body / total_range < 0.1 else 0.0
        
        # Hammer
        features['hammer'] = 1.0 if (lower_shadow > 2 * body and upper_shadow < body and close_price > open_price) else 0.0
        
        # Shooting Star
        features['shooting_star'] = 1.0 if (upper_shadow > 2 * body and lower_shadow < body and close_price < open_price) else 0.0
        
        # Engulfing patterns
        if len(df) >= 2:
            prev_open = df['open'].iloc[-2]
            prev_close = df['close'].iloc[-2]
            
            # Bullish engulfing
            features['bullish_engulfing'] = 1.0 if (
                prev_close < prev_open and
                close_price > open_price and
                open_price < prev_close and
                close_price > prev_open
            ) else 0.0
            
            # Bearish engulfing
            features['bearish_engulfing'] = 1.0 if (
                prev_close > prev_open and
                close_price < open_price and
                open_price > prev_close and
                close_price < prev_open
            ) else 0.0
        else:
            features['bullish_engulfing'] = 0.0
            features['bearish_engulfing'] = 0.0
        
        # Chart patterns
        if len(df) >= 40:
            prices = df['close'].tail(40).values
            
            # Double Top
            peaks = []
            for i in range(1, len(prices) - 1):
                if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                    peaks.append((i, prices[i]))
            
            if len(peaks) >= 2:
                last_two_peaks = sorted(peaks, key=lambda x: x[0])[-2:]
                peak_diff = abs(last_two_peaks[0][1] - last_two_peaks[1][1]) / last_two_peaks[0][1] if last_two_peaks[0][1] > 0 else 1.0
                features['double_top'] = 1.0 if peak_diff < 0.02 else 0.0
            else:
                features['double_top'] = 0.0
            
            # Double Bottom
            troughs = []
            for i in range(1, len(prices) - 1):
                if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                    troughs.append((i, prices[i]))
            
            if len(troughs) >= 2:
                last_two_troughs = sorted(troughs, key=lambda x: x[0])[-2:]
                trough_diff = abs(last_two_troughs[0][1] - last_two_troughs[1][1]) / last_two_troughs[0][1] if last_two_troughs[0][1] > 0 else 1.0
                features['double_bottom'] = 1.0 if trough_diff < 0.02 else 0.0
            else:
                features['double_bottom'] = 0.0
            
            # Triangle (volatility decreasing)
            recent_volatility = df['close'].tail(20).std()
            earlier_volatility = df['close'].tail(40).head(20).std()
            features['triangle'] = 1.0 if (recent_volatility < earlier_volatility * 0.7 and earlier_volatility > 0) else 0.0
        else:
            features['double_top'] = 0.0
            features['double_bottom'] = 0.0
            features['triangle'] = 0.0
        
        # Support/Resistance breaks
        if len(df) >= 40:
            current_price = df['close'].iloc[-1]
            recent_high = df['high'].tail(20).max()
            recent_low = df['low'].tail(20).min()
            prev_high = df['high'].tail(40).head(20).max()
            prev_low = df['low'].tail(40).head(20).min()
            
            features['resistance_break'] = 1.0 if current_price > prev_high * 1.01 else 0.0
            features['support_break'] = 1.0 if current_price < prev_low * 0.99 else 0.0
        else:
            features['resistance_break'] = 0.0
            features['support_break'] = 0.0
        
        return features
    
    def _add_feature_interactions(self, X: np.ndarray, feature_names: List[str], fit: bool = True) -> Tuple[np.ndarray, List[str]]:
        """Add important feature interactions"""
        interactions = []
        interaction_names = []
        
        # Map feature names to indices (handle partial matches)
        name_to_idx = {name: idx for idx, name in enumerate(feature_names)}
        
        # Find important feature indices (try multiple name variations)
        rsi_idx = name_to_idx.get('rsi', -1)
        volume_ratio_idx = name_to_idx.get('volume_ratio', -1)
        macd_idx = name_to_idx.get('macd', -1)
        
        # Try to find momentum feature
        momentum_idx = -1
        for name in ['momentum_5', 'return_5d', 'roc_5']:
            if name in name_to_idx:
                momentum_idx = name_to_idx[name]
                break
        
        # Try to find price position feature
        price_pos_idx = -1
        for name in ['price_position_in_range', 'price_position_20', 'price_position_10']:
            if name in name_to_idx:
                price_pos_idx = name_to_idx[name]
                break
        
        # Try to find volatility feature
        volatility_idx = -1
        for name in ['volatility_20d', 'volatility_20', 'volatility_10', 'price_std_20']:
            if name in name_to_idx:
                volatility_idx = name_to_idx[name]
                break
        
        # Create interactions (only if both features exist)
        if rsi_idx >= 0 and volume_ratio_idx >= 0 and rsi_idx < X.shape[1] and volume_ratio_idx < X.shape[1]:
            interactions.append(X[:, rsi_idx] * X[:, volume_ratio_idx])
            interaction_names.append('rsi_x_volume_ratio')
        
        if macd_idx >= 0 and momentum_idx >= 0 and macd_idx < X.shape[1] and momentum_idx < X.shape[1]:
            interactions.append(X[:, macd_idx] * X[:, momentum_idx])
            interaction_names.append('macd_x_momentum')
        
        if price_pos_idx >= 0 and volatility_idx >= 0 and price_pos_idx < X.shape[1] and volatility_idx < X.shape[1]:
            interactions.append(X[:, price_pos_idx] * X[:, volatility_idx])
            interaction_names.append('price_pos_x_volatility')
        
        if rsi_idx >= 0 and price_pos_idx >= 0 and rsi_idx < X.shape[1] and price_pos_idx < X.shape[1]:
            interactions.append(X[:, rsi_idx] * X[:, price_pos_idx])
            interaction_names.append('rsi_x_price_pos')
        
        if volume_ratio_idx >= 0 and momentum_idx >= 0 and volume_ratio_idx < X.shape[1] and momentum_idx < X.shape[1]:
            interactions.append(X[:, volume_ratio_idx] * X[:, momentum_idx])
            interaction_names.append('volume_ratio_x_momentum')
        
        if len(interactions) > 0:
            X_with_interactions = np.column_stack([X, np.array(interactions).T])
            return X_with_interactions, interaction_names
        else:
            return X, []
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features including time-series and pattern features"""
        base_features = [
            'rsi', 'macd', 'macd_signal', 'macd_diff',
            'ema_20', 'ema_50', 'ema_200',
            'bb_upper', 'bb_lower', 'bb_middle',
            'atr', 'atr_percent',
            'stoch_k', 'stoch_d', 'williams_r',
            'adx', 'adx_pos', 'adx_neg', 'obv',
            'ema_20_above_50', 'ema_50_above_200',
            'golden_cross', 'death_cross',
            'price_above_ema20', 'price_above_ema50', 'price_above_ema200',
            'return_5d', 'return_10d', 'return_20d',
            'volatility_20d', 'volume_ratio', 'volume_volatility',
            'price_position_in_range', 'trend_slope',
            'pe_ratio', 'forward_pe', 'peg_ratio', 'price_to_book',
            'price_to_sales', 'dividend_yield', 'debt_to_equity',
            'roe', 'roa', 'profit_margin',
            'log_market_cap', 'current_price',
            # Enhanced features
            'market_regime_bull', 'market_regime_bear', 'market_regime_sideways', 'market_regime_strength',
            'relative_strength_ratio', 'relative_strength_outperformance',
            'timeframe_alignment_score', 'timeframe_confidence',
            'support_strength', 'resistance_strength',
            'volatility_regime_high', 'volatility_regime_normal', 'volatility_regime_low', 'volatility_atr_percent',
            'distance_to_support', 'distance_to_resistance'
        ]
        
        # Add time-series feature names
        time_series_features = []
        for lag in [1, 2, 3, 5, 10, 20]:
            time_series_features.extend([f'price_lag_{lag}', f'volume_lag_{lag}', f'price_change_lag_{lag}'])
        for window in [5, 10, 20, 50]:
            time_series_features.extend([f'price_ma_{window}', f'price_std_{window}', f'volume_ma_{window}', f'price_vs_ma_{window}'])
        for period in [5, 10, 20]:
            time_series_features.extend([f'momentum_{period}', f'roc_{period}'])
        for window in [5, 10, 20]:
            time_series_features.extend([f'volatility_{window}', f'realized_vol_{window}'])
        for lag in [1, 5, 10]:
            time_series_features.append(f'autocorr_{lag}')
        for window in [10, 20, 50]:
            time_series_features.append(f'price_position_{window}')
        for window in [10, 20]:
            time_series_features.append(f'trend_strength_{window}')
        time_series_features.extend(['price_volume_correlation', 'momentum_acceleration', 
                                    'distance_to_resistance', 'distance_to_support'])
        
        # Add pattern feature names
        pattern_features = [
            'doji', 'hammer', 'shooting_star', 'bullish_engulfing', 'bearish_engulfing',
            'double_top', 'double_bottom', 'triangle', 'resistance_break', 'support_break'
        ]
        
        return base_features + time_series_features + pattern_features


class StockPredictionML:
    """Machine Learning model for stock prediction"""
    
    def __init__(self, data_dir: Path, strategy: str):
        self.data_dir = data_dir
        self.strategy = strategy
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.model_file = f"ml_model_{strategy}.pkl"
        self.scaler_file = f"scaler_{strategy}.pkl"
        self.metadata_file = self.data_dir / f"ml_metadata_{strategy}.json"
        
        self.storage = SecureMLStorage(data_dir)
        self.feature_extractor = FeatureExtractor()
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = {}
        self.selected_features = None  # Indices of selected features
        self.feature_selection_threshold = 0.005  # Reduced from 0.01 - keep more features but still filter noise
        self.metadata = {}
        self.train_accuracy = 0.0
        self.test_accuracy = 0.0
        self.use_hyperparameter_tuning = False  # Can be enabled for better accuracy
        self.use_regime_models = False  # Can be enabled for regime-specific models
        self.feature_selection_method = 'importance'  # 'importance', 'rfe', 'mutual_info'
        
        self._load_model()
    
    def _load_model(self):
        """Load trained model if exists"""
        if self.storage.model_exists(self.model_file):
            try:
                self.model = self.storage.load_model(self.model_file)
                self.scaler = self.storage.load_model(self.scaler_file)
                
                # Load label encoder if it exists
                label_encoder_file = self.data_dir / f"label_encoder_{self.strategy}.pkl"
                if self.storage.model_exists(label_encoder_file):
                    self.label_encoder = self.storage.load_model(label_encoder_file)
                    logger.debug(f"Loaded label encoder for {self.strategy}")
                else:
                    # Label encoder file missing - try to recover from metadata or use defaults
                    logger.debug(f"Label encoder file not found for {self.strategy}. Attempting recovery.")
                    self.label_encoder = LabelEncoder()
                    
                    # Try to get class order from metadata first (if available)
                    classes_to_use = None
                    if self.metadata_file.exists():
                        try:
                            with open(self.metadata_file, 'r') as f:
                                metadata = json.load(f)
                                classes_to_use = metadata.get('label_classes')
                                if classes_to_use:
                                    logger.debug(f"Found class order in metadata: {classes_to_use}")
                        except Exception as e:
                            logger.debug(f"Could not read metadata for class recovery: {e}")
                    
                    # If no metadata, use default expected classes
                    if not classes_to_use:
                        classes_to_use = ['BUY', 'HOLD', 'SELL']  # Standard expected classes
                        logger.debug(f"Using default class order: {classes_to_use}")
                    
                    try:
                        self.label_encoder.fit(classes_to_use)
                        logger.debug(f"Created label encoder with classes: {list(self.label_encoder.classes_)}")
                        logger.debug(f"Note: If class order differs from original training, predictions may be incorrect. Consider retraining.")
                    except Exception as e:
                        logger.debug(f"Could not create label encoder: {e}. Model will need retraining.")
                        # Leave encoder unfitted - is_trained() will return False and predict() will handle gracefully
                
                # Load metadata
                if self.metadata_file.exists():
                    try:
                        with open(self.metadata_file, 'r') as f:
                            self.metadata = json.load(f)
                            self.feature_importance = self.metadata.get('feature_importance', {})
                            # Load selected features if available
                            self.selected_features = self.metadata.get('selected_features', None)
                            if self.selected_features:
                                self.selected_features = np.array(self.selected_features)
                            # Also load accuracy values
                            self.train_accuracy = self.metadata.get('train_accuracy', 0)
                            self.test_accuracy = self.metadata.get('test_accuracy', 0)
                            logger.info(f"Loaded ML model for {self.strategy} - Train: {self.train_accuracy*100:.1f}%, Test: {self.test_accuracy*100:.1f}%")
                            if self.selected_features is not None:
                                logger.info(f"Using {len(self.selected_features)} selected features (out of {len(self.feature_extractor.get_feature_names())} total)")
                    except Exception as meta_error:
                        logger.error(f"Error loading metadata: {meta_error}")
                        self.metadata = {}
                        self.train_accuracy = 0
                        self.test_accuracy = 0
                else:
                    logger.warning(f"Metadata file not found for {self.strategy}")
                    self.metadata = {}
                    self.train_accuracy = 0
                    self.test_accuracy = 0
                
                logger.info(f"Loaded ML model for {self.strategy}")
            except Exception as e:
                logger.error(f"Error loading ML model: {e}")
                self.metadata = {}
                self.train_accuracy = 0
                self.test_accuracy = 0
    
    def train(self, training_samples: List[Dict], test_size: float = 0.2, random_seed: int = None,
              use_hyperparameter_tuning: bool = False, use_regime_models: bool = False,
              feature_selection_method: str = 'importance',
              # Optional hyperparameter overrides for testing
              rf_n_estimators: int = None, rf_max_depth: int = None,
              rf_min_samples_split: int = None, rf_min_samples_leaf: int = None,
              gb_n_estimators: int = None, gb_max_depth: int = None,
              disable_feature_selection: bool = False, use_smote: bool = False,
              use_interactions: bool = None,
              # Simplification knobs (default: simpler + more generalizable)
              model_family: str = 'voting',  # 'rf' | 'voting' - voting is default for better accuracy
              hold_confidence_threshold: float = 0.0,
              # Binary classification mode (UP/DOWN only, no HOLD) - achieves ~65% accuracy
              use_binary_classification: bool = False) -> Dict:
        """Train ML model on historical data"""
        if len(training_samples) < 100:
            return {"error": f"Insufficient training samples: {len(training_samples)} < 100"}

        # --- Backward compatibility shim ---
        # Older scripts used: train(samples, disable_feature_selection, use_smote, feature_selection_method)
        # Example: train(samples, False, False, 'importance')
        if isinstance(test_size, bool):
            legacy_disable = bool(test_size)
            legacy_smote = bool(random_seed) if isinstance(random_seed, bool) else False
            legacy_method = use_hyperparameter_tuning if isinstance(use_hyperparameter_tuning, str) else feature_selection_method
            disable_feature_selection = legacy_disable
            use_smote = legacy_smote
            feature_selection_method = legacy_method
            test_size = 0.2
            random_seed = None
            use_hyperparameter_tuning = False
            use_regime_models = False
        
        # Force retraining by resetting model components
        logger.info("Resetting model for fresh training...")
        self.model = None
        self.scaler = StandardScaler()  # Reset scaler
        self.label_encoder = LabelEncoder()  # Reset encoder
        
        # Use variable random seed if not provided (based on current time)
        if random_seed is None:
            import time
            random_seed = int(time.time() * 1000) % 10000  # Use milliseconds for seed
        
        # Set training options
        self.use_hyperparameter_tuning = use_hyperparameter_tuning
        self.use_regime_models = use_regime_models
        self.feature_selection_method = feature_selection_method
        
        logger.info(f"Training ML model on {len(training_samples)} samples with random_seed={random_seed}...")
        logger.info(f"  • Hyperparameter tuning: {use_hyperparameter_tuning}")
        logger.info(f"  • Regime-specific models: {use_regime_models}")
        logger.info(f"  • Feature selection method: {feature_selection_method}")
        logger.info(f"  • Model family: {model_family}")
        logger.info(f"  • HOLD confidence threshold: {hold_confidence_threshold}")
        logger.info(f"  • Binary classification: {use_binary_classification}")
        
        # Prepare data
        X = np.array([s['features'] for s in training_samples])
        y = np.array([s['label'] for s in training_samples])
        
        # Binary classification mode: convert BUY/SELL/HOLD to UP/DOWN (skip HOLD)
        # This achieves ~65% accuracy vs ~45% for 3-class
        if use_binary_classification:
            logger.info("Binary classification mode: converting labels to UP/DOWN")
            # Filter out HOLD samples and convert labels
            binary_mask = []
            binary_labels = []
            for i, label in enumerate(y):
                if label == 'BUY':
                    binary_mask.append(i)
                    binary_labels.append('DOWN')  # BUY signal = price went down (inverted logic)
                elif label == 'SELL':
                    binary_mask.append(i)
                    binary_labels.append('UP')  # SELL signal = price went up (inverted logic)
                # Skip HOLD
            
            if len(binary_mask) < 100:
                return {"error": f"Not enough binary samples after filtering HOLD: {len(binary_mask)} < 100"}
            
            X = X[binary_mask]
            y = np.array(binary_labels)
            logger.info(f"Binary samples: {len(X)} (removed {len(training_samples) - len(X)} HOLD samples)")
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Check for and remove duplicate samples (can cause overfitting)
        from collections import Counter
        import hashlib
        
        # Create hash of each sample to detect duplicates
        sample_hashes = []
        for x in X:
            # Round to 6 decimal places to catch near-duplicates
            x_rounded = np.round(x, 6)
            sample_hash = hashlib.md5(x_rounded.tobytes()).hexdigest()
            sample_hashes.append(sample_hash)
        
        unique_indices = []
        seen_hashes = set()
        for i, h in enumerate(sample_hashes):
            if h not in seen_hashes:
                unique_indices.append(i)
                seen_hashes.add(h)
        
        if len(unique_indices) < len(X):
            logger.warning(f"Removed {len(X) - len(unique_indices)} duplicate samples ({len(unique_indices)}/{len(X)} unique) to prevent overfitting")
            X = X[unique_indices]
            y = y[unique_indices]
        
        # Check class distribution
        label_counts = Counter(y)
        logger.info(f"Training data distribution: {dict(label_counts)}")
        
        # Warn if severe class imbalance
        if len(label_counts) > 1:
            min_class = min(label_counts.values())
            max_class = max(label_counts.values())
            if min_class < max_class * 0.1:  # Less than 10% of majority class
                logger.warning(f"Severe class imbalance detected ({min_class} vs {max_class}). This may affect model performance.")
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data (using random split - original method that gave 47% accuracy)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_seed, stratify=y_encoded
        )
        
        # ===== FEATURE SELECTION =====
        # Scale features first
        X_train_scaled_initial = self.scaler.fit_transform(X_train)
        X_test_scaled_initial = self.scaler.transform(X_test)
        
        # Initialize feature_importance_dict for later use
        feature_importance_dict = {}
        
        # Skip feature selection if disabled
        if disable_feature_selection:
            logger.info("Feature selection disabled - using all features")
            X_train_selected = X_train_scaled_initial
            X_test_selected = X_test_scaled_initial
            self.selected_features = np.arange(len(X_train[0]))
            selected_feature_names = self.feature_extractor.get_feature_names()
            selected_feature_indices = list(range(len(selected_feature_names)))
            # Create dummy importance dict for all features
            feature_importance_dict = {name: 1.0 / len(selected_feature_names) for name in selected_feature_names}
        else:
            # Determine number of features to select (40-60% of total - more aggressive filtering)
            # Reduced from 60% to 50% to remove more noise features
            n_features_to_select = max(30, min(70, int(len(X_train[0]) * 0.5)))
            
            if self.feature_selection_method == 'rfe' and len(X_train) > n_features_to_select:
                # Use RFE for feature selection
                logger.info(f"Step 1: Using RFE to select top {n_features_to_select} features...")
                estimator = RandomForestClassifier(n_estimators=50, random_state=random_seed, n_jobs=get_safe_n_jobs(ML_TRAINING_CPU_CORES))
                rfe = RFE(estimator, n_features_to_select=n_features_to_select)
                X_train_selected = rfe.fit_transform(X_train_scaled_initial, y_train)
                X_test_selected = rfe.transform(X_test_scaled_initial)
                self.selected_features = np.array(rfe.get_support(indices=True))
                selected_feature_names = [self.feature_extractor.get_feature_names()[i] for i in self.selected_features]
                logger.info(f"RFE selected {len(self.selected_features)} features")
            elif self.feature_selection_method == 'mutual_info':
                # Use mutual information
                logger.info(f"Step 1: Using Mutual Information to select top {n_features_to_select} features...")
                selector = SelectKBest(score_func=mutual_info_classif, k=n_features_to_select)
                X_train_selected = selector.fit_transform(X_train_scaled_initial, y_train)
                X_test_selected = selector.transform(X_test_scaled_initial)
                self.selected_features = np.array(selector.get_support(indices=True))
                selected_feature_names = [self.feature_extractor.get_feature_names()[i] for i in self.selected_features]
                logger.info(f"Mutual Information selected {len(self.selected_features)} features")
            else:
                # Use importance-based selection (default)
                logger.info("Step 1: Training initial model to calculate feature importances...")
                initial_rf = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=10,  # Original value
                    min_samples_split=10,  # Original value
                    min_samples_leaf=5,  # Original value
                    max_features='sqrt',
                    class_weight='balanced',
                    random_state=random_seed,
                    n_jobs=get_safe_n_jobs(ML_TRAINING_CPU_CORES)
                )
                
                initial_rf.fit(X_train_scaled_initial, y_train)
                
                # Get feature importances
                feature_importances = initial_rf.feature_importances_
                feature_names = self.feature_extractor.get_feature_names()
                
                # Create feature importance dictionary
                feature_importance_dict = dict(zip(feature_names, feature_importances))
                feature_importance_dict = dict(sorted(
                    feature_importance_dict.items(),
                    key=lambda x: x[1],
                    reverse=True
                ))
                
                # Select top K features
                top_features = list(feature_importance_dict.items())[:n_features_to_select]
                all_feature_names = self.feature_extractor.get_feature_names()
                name_to_index = {name: idx for idx, name in enumerate(all_feature_names)}
                
                selected_feature_indices = []
                selected_feature_names = []
                removed_features = []
                
                for name, importance in feature_importance_dict.items():
                    if len(selected_feature_indices) < n_features_to_select:
                        idx = name_to_index[name]
                        selected_feature_indices.append(idx)
                        selected_feature_names.append(name)
                    else:
                        removed_features.append((name, importance))
                
                # Sort indices to maintain feature order
                selected_feature_indices.sort()
                self.selected_features = np.array(selected_feature_indices)
                
                logger.info(f"Feature Selection Results:")
                logger.info(f"  • Total features: {len(feature_names)}")
                logger.info(f"  • Selected features: {len(selected_feature_indices)}")
                logger.info(f"  • Removed features: {len(removed_features)}")
                
                if removed_features:
                    logger.info(f"  • Top 5 removed features: {[(n, f'{imp:.4f}') for n, imp in removed_features[:5]]}")
                
                # Filter features
                X_train_selected = X_train_scaled_initial[:, self.selected_features]
                X_test_selected = X_test_scaled_initial[:, self.selected_features]
                selected_feature_indices = list(self.selected_features)
        
        # Step 4: Add feature interactions (optional, can be disabled)
        # Default OFF: interactions explode feature space and tend to overfit on small/noisy datasets.
        if use_interactions is None:
            use_interactions = False
        if use_interactions and len(selected_feature_indices) > 10:
            try:
                X_train_selected, interaction_names = self.feature_extractor._add_feature_interactions(
                    X_train_selected, selected_feature_names
                )
                X_test_selected, _ = self.feature_extractor._add_feature_interactions(
                    X_test_selected, selected_feature_names, fit=False
                )
                logger.info(f"Added {len(interaction_names)} feature interactions")
            except Exception as e:
                logger.warning(f"Could not add feature interactions: {e}")
        
        logger.info(f"Step 2: Retraining model with {X_train_selected.shape[1]} features (including interactions)...")
        
        # Update scaler to work with selected features + interactions
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # Store feature importance for selected features only
        self.feature_importance = {
            name: feature_importance_dict[name] 
            for name in selected_feature_names
        }
        
        # ===== HYPERPARAMETER TUNING (Optional) =====
        if self.use_hyperparameter_tuning and HAS_OPTUNA and len(X_train_scaled) > 500:
            logger.info("Step 3: Tuning hyperparameters with Optuna...")
            try:
                best_params = self._tune_hyperparameters(X_train_scaled, y_train, random_seed)
                logger.info(f"Best hyperparameters found: {best_params}")
            except Exception as e:
                logger.warning(f"Hyperparameter tuning failed: {e}. Using default parameters.")
                best_params = None
        else:
            best_params = None
        
        # Create model - OPTIMIZED configuration based on systematic testing
        # Binary mode achieves ~65% accuracy, 3-class achieves ~45-48%
        rf_params = {
            'n_estimators': rf_n_estimators if rf_n_estimators is not None else 100,
            'max_depth': rf_max_depth if rf_max_depth is not None else 5,  # OPTIMIZED (was 8)
            'min_samples_split': rf_min_samples_split if rf_min_samples_split is not None else 20,
            'min_samples_leaf': rf_min_samples_leaf if rf_min_samples_leaf is not None else 10,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': random_seed,
            'n_jobs': get_safe_n_jobs(ML_TRAINING_CPU_CORES)
        }
        
        # Apply tuned parameters if available (only if not overridden)
        if best_params and 'rf' in best_params:
            for key, value in best_params['rf'].items():
                if key not in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
                    rf_params[key] = value
        
        rf = RandomForestClassifier(**rf_params)

        # OPTIMIZED: Voting ensemble with deeper GB (achieves best validation accuracy)
        if model_family == 'voting':
            gb = GradientBoostingClassifier(
                n_estimators=gb_n_estimators if gb_n_estimators is not None else 150,
                max_depth=gb_max_depth if gb_max_depth is not None else 4,  # OPTIMIZED (was 3)
                learning_rate=0.05,
                min_samples_split=20,
                min_samples_leaf=10,
                subsample=0.8,
                random_state=random_seed
            )
            lr = LogisticRegression(
                max_iter=1000,
                C=0.2,
                class_weight='balanced',
                random_state=random_seed
            )
            self.model = VotingClassifier(
                estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
                voting='soft',
                weights=[2, 1, 1]
            )
            logger.info("Using simplified Voting ensemble (sklearn-only)")
        else:
            self.model = rf
            logger.info("Using single RandomForest (simplified default)")
        
        # Train (suppress feature name warnings)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')
            self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        # Key trick for noisy 3-class tasks: if the model isn't confident, predict HOLD.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')
            # Raw accuracy (argmax)
            raw_train_score = self.model.score(X_train_scaled, y_train)
            raw_test_score = self.model.score(X_test_scaled, y_test)

            train_pred = self.model.predict(X_train_scaled)
            test_pred = self.model.predict(X_test_scaled)

            train_proba = self.model.predict_proba(X_train_scaled)
            test_proba = self.model.predict_proba(X_test_scaled)

        classes = list(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else []
        hold_idx = classes.index('HOLD') if 'HOLD' in classes else None

        def apply_hold_threshold(pred, proba):
            if hold_idx is None:
                return pred
            if hold_confidence_threshold is None or hold_confidence_threshold <= 0:
                return pred
            maxp = np.max(proba, axis=1)
            pred_adj = pred.copy()
            pred_adj[maxp < float(hold_confidence_threshold)] = hold_idx
            return pred_adj

        train_pred_adj = apply_hold_threshold(train_pred, train_proba)
        test_pred_adj = apply_hold_threshold(test_pred, test_proba)

        train_score = accuracy_score(y_train, train_pred_adj)
        test_score = accuracy_score(y_test, test_pred_adj)
        
        # Warn if overfitting detected
        if train_score > 0.95 and test_score < train_score - 0.1:
            logger.warning(f"⚠️ Overfitting detected! Train: {train_score*100:.1f}%, Test: {test_score*100:.1f}%")
            logger.warning("   Model may be too complex. Consider reducing model complexity or adding more training data.")
        
        # Log training details
        logger.info(
            f"Training accuracy: {train_score*100:.1f}%, Validation accuracy: {test_score*100:.1f}% "
            f"(raw: {raw_train_score*100:.1f}% / {raw_test_score*100:.1f}%)"
        )
        if self.selected_features is not None:
            total_features = len(self.feature_extractor.get_feature_names())
            selected_count = len(self.selected_features)
            logger.info(f"✅ Feature selection: Using {selected_count}/{total_features} features "
                       f"({selected_count/total_features*100:.1f}% retained)")
        
        y_pred = self.model.predict(X_test_scaled)
        
        # Feature importance: handle both RF and voting ensemble safely
        try:
            rf_model = None
            if hasattr(self.model, 'feature_importances_'):
                rf_model = self.model
            elif hasattr(self.model, 'named_estimators_') and 'rf' in getattr(self.model, 'named_estimators_', {}):
                rf_model = self.model.named_estimators_.get('rf')

            if rf_model is not None and hasattr(rf_model, 'feature_importances_'):
                final_importances = rf_model.feature_importances_
                selected_feature_names = [self.feature_extractor.get_feature_names()[idx]
                                          for idx in self.selected_features]
                final_importance_dict = dict(zip(selected_feature_names, final_importances))
                for name in final_importance_dict:
                    if name in self.feature_importance:
                        self.feature_importance[name] = max(self.feature_importance[name], final_importance_dict[name])
                    else:
                        self.feature_importance[name] = final_importance_dict[name]

                self.feature_importance = dict(sorted(
                    self.feature_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:20])
        except Exception as e:
            logger.debug(f"Could not compute final feature importances: {e}")
        
        # Cross-validation (suppress feature name warnings)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        # Save model securely
        self.storage.save_model(self.model, self.model_file)
        self.storage.save_model(self.scaler, self.scaler_file)
        # Save label encoder (needed for prediction)
        label_encoder_file = self.data_dir / f"label_encoder_{self.strategy}.pkl"
        self.storage.save_model(self.label_encoder, label_encoder_file)
        
        # Save metadata
        self.metadata = {
            'strategy': self.strategy,
            'samples': len(training_samples),
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'trained_date': datetime.now().isoformat(),
            'feature_importance': self.feature_importance,
            'selected_features': self.selected_features.tolist() if self.selected_features is not None else None,
            'feature_selection_threshold': self.feature_selection_threshold,
            'feature_selection_method': self.feature_selection_method,
            'total_features': len(self.feature_extractor.get_feature_names()),
            'selected_features_count': len(self.selected_features) if self.selected_features is not None else None,
            'label_classes': self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else None,
            'model_type': model_family,
            'hold_confidence_threshold': float(hold_confidence_threshold) if hold_confidence_threshold is not None else 0.0,
            'has_optuna': HAS_OPTUNA,
            'used_hyperparameter_tuning': self.use_hyperparameter_tuning,
            'use_binary_classification': use_binary_classification  # Store binary mode flag
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"Trained and saved ML model: {test_score*100:.1f}% accuracy")
        
        return self.metadata
    
    def predict(self, features: np.ndarray) -> Dict:
        """Make prediction using trained model"""
        if self.model is None:
            return {
                'action': 'HOLD',
                'confidence': 50,
                'error': 'Model not trained'
            }
        
        # Handle missing values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = features.reshape(1, -1)
        
        # Check if label encoder is fitted
        if not hasattr(self.label_encoder, 'classes_') or len(self.label_encoder.classes_) == 0:
            logger.debug(f"LabelEncoder for {self.strategy} is not fitted. Model needs to be retrained. Falling back to rule-based prediction.")
            # Fallback to rule-based prediction (don't log as error, just debug)
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'error': 'LabelEncoder not fitted - model needs retraining',
                'probabilities': {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 100.0}
            }
        
        # Apply feature selection if available
        if self.selected_features is not None:
            if len(features[0]) > len(self.selected_features):
                # Select only the features that were used during training
                features = features[:, self.selected_features]
            elif len(features[0]) < len(self.selected_features):
                # Features don't match - model may need retraining
                logger.warning(f"Feature count mismatch: got {len(features[0])}, expected {len(self.selected_features)}")
                return {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'error': 'Feature count mismatch - model may need retraining',
                    'probabilities': {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 100.0}
                }
        
        # Add feature interactions if they were used during training
        # (This is a simplified version - in production, you'd store interaction info)
        # For now, we'll skip interactions in prediction to avoid complexity
        
        # Scale
        features_scaled = self.scaler.transform(features)
        
        # Predict (suppress feature name warnings - they're harmless)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Optional: low-confidence -> HOLD (stored in metadata after training)
        hold_threshold = 0.0
        try:
            hold_threshold = float(self.metadata.get('hold_confidence_threshold', 0.0)) if self.metadata else 0.0
        except Exception:
            hold_threshold = 0.0

        classes = list(self.label_encoder.classes_)
        hold_idx = classes.index('HOLD') if 'HOLD' in classes else None
        maxp = float(np.max(probabilities))
        if hold_idx is not None and hold_threshold > 0.0 and maxp < hold_threshold:
            prediction = hold_idx

        # Decode label
        action = self.label_encoder.inverse_transform([prediction])[0]
        original_action = action  # Store for logging

        # Convert binary classification (UP/DOWN) back to BUY/SELL for compatibility
        # Binary mode: UP = price goes up = SELL signal, DOWN = price goes down = BUY signal
        # Check if model was trained in binary mode
        is_binary_mode = False
        if self.metadata and self.metadata.get('use_binary_classification', False):
            is_binary_mode = True
        elif hasattr(self.label_encoder, 'classes_'):
            classes = list(self.label_encoder.classes_)
            # If model only has UP/DOWN classes, it was trained in binary mode
            if set(classes) == {'UP', 'DOWN'} or (len(classes) == 2 and 'UP' in classes and 'DOWN' in classes):
                is_binary_mode = True
        
        if is_binary_mode:
            if action == 'UP':
                action = 'SELL'  # UP prediction = price goes up = sell signal
            elif action == 'DOWN':
                action = 'BUY'  # DOWN prediction = price goes down = buy signal
            # Note: Binary mode doesn't predict HOLD, so we use the action directly
            logger.debug(f"Binary mode conversion: {original_action} → {action}")

        # Calculate confidence for the chosen class
        confidence = float(probabilities[prediction] * 100)
        
        # Get probabilities for all classes
        prob_dict = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            # Convert binary class names to BUY/SELL for compatibility
            display_name = class_name
            if class_name == 'UP':
                display_name = 'SELL'
            elif class_name == 'DOWN':
                display_name = 'BUY'
            prob_dict[display_name] = float(probabilities[i] * 100)
        
        # Ensure all expected actions are in prob_dict (for 3-class compatibility)
        if 'BUY' not in prob_dict:
            prob_dict['BUY'] = 0.0
        if 'SELL' not in prob_dict:
            prob_dict['SELL'] = 0.0
        if 'HOLD' not in prob_dict:
            prob_dict['HOLD'] = 0.0
        
        return {
            'action': action,
            'confidence': confidence,
            'probabilities': prob_dict
        }
    
    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, random_seed: int) -> Dict:
        """Tune hyperparameters using Optuna"""
        if not HAS_OPTUNA:
            return {}
        
        def objective(trial):
            # Suggest hyperparameters for XGBoost/LightGBM
            params = {}
            
            if HAS_LIGHTGBM:
                params['lgb'] = {
                    'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('lgb_max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('lgb_subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.6, 1.0),
                    'min_child_samples': trial.suggest_int('lgb_min_child_samples', 10, 50),
                    'reg_alpha': trial.suggest_float('lgb_reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('lgb_reg_lambda', 0, 2),
                }
            
            if HAS_XGBOOST:
                params['xgb'] = {
                    'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 500),
                    'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
                    'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
                    'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
                    'gamma': trial.suggest_float('xgb_gamma', 0, 0.5),
                    'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0, 1),
                    'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0, 2),
                }
            
            # Create model with suggested parameters
            if HAS_LIGHTGBM:
                model = LGBMClassifier(**params['lgb'], random_state=random_seed, verbose=-1)
            elif HAS_XGBOOST:
                model = XGBClassifier(**params['xgb'], random_state=random_seed, eval_metric='mlogloss')
            else:
                return 0.0
            
            # Evaluate using cross-validation
            scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy', n_jobs=get_safe_n_jobs(ML_TRAINING_CPU_CORES))
            return scores.mean()
        
        # Run optimization (limited trials for speed)
        # Use parallel trials for faster hyperparameter tuning
        import multiprocessing
        
        # Determine number of parallel workers
        if HYPERPARAMETER_TUNING_PARALLEL_TRIALS is None:
            # Auto: use all cores minus 1 (leave one for system)
            n_workers = max(1, multiprocessing.cpu_count() - 1)
        else:
            n_workers = HYPERPARAMETER_TUNING_PARALLEL_TRIALS
        
        # Use in-memory study for parallel optimization (faster than SQLite for parallel)
        study = optuna.create_study(
            direction='maximize', 
            study_name='ml_optimization'
        )
        
        # INCREASED from 20 to 50 trials for better optimization
        # More trials = better chance of finding optimal parameters for YOUR data
        logger.info(f"Starting hyperparameter tuning with {n_workers} parallel workers (using {multiprocessing.cpu_count()} CPU cores)...")
        logger.info("Running 50 trials (increased from 20 for better optimization)...")
        study.optimize(objective, n_trials=50, timeout=1200, n_jobs=n_workers)  # 50 trials, 20 min timeout
        
        # Extract best parameters
        best_params = {}
        if HAS_LIGHTGBM and study.best_params:
            best_params['lgb'] = {k.replace('lgb_', ''): v for k, v in study.best_params.items() if k.startswith('lgb_')}
        if HAS_XGBOOST and study.best_params:
            best_params['xgb'] = {k.replace('xgb_', ''): v for k, v in study.best_params.items() if k.startswith('xgb_')}
        
        logger.info(f"Hyperparameter tuning complete. Best CV score: {study.best_value:.4f}")
        return best_params
    
    def is_trained(self) -> bool:
        """Check if model is trained and ready to use"""
        if self.model is None:
            return False
        # Also check if LabelEncoder is fitted (needed for predictions)
        if not hasattr(self.label_encoder, 'classes_') or len(self.label_encoder.classes_) == 0:
            return False
        return True

