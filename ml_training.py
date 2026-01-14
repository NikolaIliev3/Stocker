"""
Machine Learning Training System
Trains ML models on historical stock data to improve predictions
"""
import pandas as pd
import numpy as np
import warnings
import logging
import sys
import multiprocessing
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

from secure_ml_storage import SecureMLStorage
try:
    from model_versioning import ModelVersionManager
    HAS_VERSIONING = True
except ImportError:
    HAS_VERSIONING = False
    logger = logging.getLogger(__name__)
    logger.warning("Model versioning not available")

try:
    from production_monitor import ProductionMonitor
    HAS_MONITORING = True
except ImportError:
    HAS_MONITORING = False
    logger = logging.getLogger(__name__)
    logger.warning("Production monitoring not available")

try:
    from model_explainability import ModelExplainer
    HAS_EXPLAINABILITY = True
except ImportError:
    HAS_EXPLAINABILITY = False
    logger = logging.getLogger(__name__)
    logger.warning("Model explainability not available")

try:
    from training_data_validator import TrainingDataValidator
    HAS_DATA_VALIDATION = True
except ImportError:
    HAS_DATA_VALIDATION = False
    logger = logging.getLogger(__name__)
    logger.warning("Training data validation not available")

try:
    from prediction_quality_scorer import PredictionQualityScorer
    HAS_QUALITY_SCORER = True
except ImportError:
    HAS_QUALITY_SCORER = False
    logger = logging.getLogger(__name__)
    logger.warning("Prediction quality scorer not available")

try:
    from config import (ML_TRAINING_CPU_CORES, HYPERPARAMETER_TUNING_PARALLEL_TRIALS,
                       ML_FEATURE_SELECTION_THRESHOLD, ML_RF_N_ESTIMATORS, ML_RF_MAX_DEPTH,
                       ML_RF_MIN_SAMPLES_SPLIT, ML_RF_MIN_SAMPLES_LEAF, ML_GB_N_ESTIMATORS,
                       ML_GB_MAX_DEPTH, ML_GB_LEARNING_RATE, ML_MIN_TRAINING_SAMPLES,
                       ML_MIN_BINARY_SAMPLES)
except ImportError:
    # Fallback if not in config
    # Windows-specific: limit to 4 cores to prevent deadlocks
    if sys.platform == 'win32':
        ML_TRAINING_CPU_CORES = min(4, multiprocessing.cpu_count())
    else:
        ML_TRAINING_CPU_CORES = -1  # Use all cores on non-Windows
    HYPERPARAMETER_TUNING_PARALLEL_TRIALS = None
    ML_FEATURE_SELECTION_THRESHOLD = 0.005
    ML_RF_N_ESTIMATORS = 80
    ML_RF_MAX_DEPTH = 4
    ML_RF_MIN_SAMPLES_SPLIT = 30
    ML_RF_MIN_SAMPLES_LEAF = 15
    ML_GB_N_ESTIMATORS = 150
    ML_GB_MAX_DEPTH = 4
    ML_GB_LEARNING_RATE = 0.03
    ML_MIN_TRAINING_SAMPLES = 100
    ML_MIN_BINARY_SAMPLES = 30

# Helper function to safely get n_jobs value (Windows-specific fix)
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

        # ===== ENHANCED FEATURES (Moved to end to preserve index alignment) =====
        # Extract additional enhanced features if available
        # Always add exactly 15 enhanced features to maintain consistency
        enhanced_feature_names = [
            'avg_spread', 'spread_volatility', 'price_volume_correlation', 'liquidity_proxy',
            'intraday_volatility', 'volume_trend', 'volume_ma_ratio',
            'sector_relative_strength', 'sector_correlation',
            'news_sentiment', 'news_volume', 'market_cap_category',
            'sector_type', 'beta', 'dividend_yield'
        ]
        
        if self.enhanced_extractor:
            try:
                enhanced_features = self.enhanced_extractor.extract_all_enhanced_features(
                    stock_data, history_data, news_data
                )
                # Add enhanced features in fixed order
                for name in enhanced_feature_names:
                    value = enhanced_features.get(name, 0.0)
                    if isinstance(value, (int, float)):
                        features.append(float(value))
                    elif isinstance(value, bool):
                        features.append(1.0 if value else 0.0)
                    else:
                        features.append(0.0)
            except Exception as e:
                logger.debug(f"Could not extract enhanced features: {e}")
                # Add zeros for all enhanced features if extraction fails
                features.extend([0.0] * len(enhanced_feature_names))
        else:
            # Add zeros for all enhanced features if extractor not available
            features.extend([0.0] * len(enhanced_feature_names))
        
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
                except (ValueError, AttributeError, IndexError) as e:
                    logger.debug(f"Could not calculate autocorrelation for lag {lag}: {e}")
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
                except (ValueError, np.linalg.LinAlgError, IndexError) as e:
                    logger.debug(f"Could not calculate trend strength for window {window}: {e}")
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
                except (ValueError, AttributeError, IndexError) as e:
                    logger.debug(f"Could not calculate price-volume correlation: {e}")
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
            except (ValueError, IndexError, AttributeError) as e:
                logger.debug(f"Could not calculate momentum acceleration: {e}")
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
        ]
        
        # New Features (Moved to end to preserve index alignment)
        momentum_enhanced_features = [
            # Enhanced features from enhanced_extractor (always 15 features)
            'avg_spread', 'spread_volatility', 'price_volume_correlation', 'liquidity_proxy',
            'intraday_volatility', 'volume_trend', 'volume_ma_ratio',
            'sector_relative_strength', 'sector_correlation',
            'news_sentiment', 'news_volume', 'market_cap_category',
            'sector_type', 'beta', 'dividend_yield',
            # Market regime and other enhanced features
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
        
        return base_features + time_series_features + pattern_features + momentum_enhanced_features


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
        self.regime_models = {}  # Dict of regime-specific models: {'bull': model, 'bear': model, 'sideways': model}
        self.regime_scalers = {}  # Dict of regime-specific scalers
        self.regime_label_encoders = {}  # Dict of regime-specific label encoders
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = {}
        self.selected_features = None  # Indices of selected features
        self.feature_selection_threshold = ML_FEATURE_SELECTION_THRESHOLD
        self.metadata = {}
        self.train_accuracy = 0.0
        self.test_accuracy = 0.0
        self.use_hyperparameter_tuning = True  # Enabled by default for better accuracy
        self.use_regime_models = True  # Enabled by default for better accuracy (+3-5%)
        self.feature_selection_method = 'rfe'  # Changed to 'rfe' for better feature selection
        self.regime_detector = None  # Will be initialized when needed
        
        # Initialize version manager
        if HAS_VERSIONING:
            self.version_manager = ModelVersionManager(data_dir, strategy)
        else:
            self.version_manager = None
        
        # Initialize production monitor
        if HAS_MONITORING:
            self.production_monitor = ProductionMonitor(data_dir, strategy)
        else:
            self.production_monitor = None
        
        # Model explainer will be initialized after model is loaded
        self.model_explainer = None
        
        # Initialize quality scorer
        if HAS_QUALITY_SCORER:
            self.quality_scorer = PredictionQualityScorer()
        else:
            self.quality_scorer = None
        
        self._load_model()
        
        # Initialize explainer after model is loaded
        if HAS_EXPLAINABILITY and self.model is not None:
            try:
                feature_names = self.feature_extractor.get_feature_names()
                self.model_explainer = ModelExplainer(
                    model=self.model,
                    feature_names=feature_names,
                    scaler=self.scaler
                )
            except Exception as e:
                logger.debug(f"Could not initialize model explainer: {e}")
    
    def _load_model(self):
        """Load trained model if exists"""
        # First, try to load metadata to check if regime models are used
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
                    
                    # Check if regime models were used
                    if self.metadata.get('use_regime_models', False):
                        self.use_regime_models = True
                        self._load_regime_models()
                        logger.info(f"Loaded regime-specific models for {self.strategy}")
                        if 'regime_metadata' in self.metadata:
                            regime_info = []
                            for regime, meta in self.metadata['regime_metadata'].items():
                                regime_info.append(f"{regime}: {meta.get('test_accuracy', 0)*100:.1f}%")
                            logger.info(f"  Regime accuracies: {', '.join(regime_info)}")
                        # If regime models loaded successfully, we're done
                        if len(self.regime_models) > 0:
                            return
            except Exception as meta_error:
                logger.error(f"Error loading metadata: {meta_error}")
                self.metadata = {}
                self.train_accuracy = 0
                self.test_accuracy = 0
        
        # If metadata doesn't exist or regime models weren't loaded, check if regime model files exist directly
        # This handles the case where metadata was deleted but model files still exist
        regimes = ['bull', 'bear', 'sideways']
        regime_files_exist = False
        for regime in regimes:
            regime_model_file = f"ml_model_{self.strategy}_{regime}.pkl"
            regime_scaler_file = f"scaler_{self.strategy}_{regime}.pkl"
            regime_encoder_file = f"label_encoder_{self.strategy}_{regime}.pkl"
            if (self.storage.model_exists(regime_model_file) and 
                self.storage.model_exists(regime_scaler_file) and
                self.storage.model_exists(regime_encoder_file)):
                regime_files_exist = True
                break
        
        if regime_files_exist and not self.regime_models:
            logger.info(f"Found regime model files but no metadata. Attempting to load regime models directly...")
            self.use_regime_models = True
            # Create minimal metadata if it doesn't exist
            if not self.metadata:
                self.metadata = {'use_regime_models': True, 'regime_metadata': {}}
            self._load_regime_models()
            if len(self.regime_models) > 0:
                logger.info(f"Successfully loaded {len(self.regime_models)} regime model(s) without metadata")
                return
        
        # Load single model (non-regime)
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
                
                # Load metadata if not already loaded
                if not self.metadata and self.metadata_file.exists():
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
                            
                            if self.selected_features is not None:
                                logger.info(f"Using {len(self.selected_features)} selected features (out of {len(self.feature_extractor.get_feature_names())} total)")
                    except Exception as meta_error:
                        logger.error(f"Error loading metadata: {meta_error}")
                        self.metadata = {}
                        self.train_accuracy = 0
                        self.test_accuracy = 0
                elif self.metadata:
                    logger.info(f"Loaded ML model for {self.strategy} - Train: {self.train_accuracy*100:.1f}%, Test: {self.test_accuracy*100:.1f}%")
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
              use_hyperparameter_tuning: bool = False, use_regime_models: bool = True,  # Enabled by default
              feature_selection_method: str = 'importance',
              # Optional hyperparameter overrides for testing
              rf_n_estimators: int = None, rf_max_depth: int = None,
              rf_min_samples_split: int = None, rf_min_samples_leaf: int = None,
              gb_n_estimators: int = None, gb_max_depth: int = None,
              disable_feature_selection: bool = False, use_smote: bool = None,  # None = auto-enable for binary classification
              use_interactions: bool = None,
              # Simplification knobs (default: simpler + more generalizable)
              model_family: str = 'voting',  # 'rf' | 'voting' - voting is default for better accuracy
              hold_confidence_threshold: float = 0.0,
              # Binary classification mode (UP/DOWN only, no HOLD) - achieves ~65% accuracy
              use_binary_classification: bool = False,
              # Regularization parameters (for config manager)
              lgb_reg_alpha: float = None, lgb_reg_lambda: float = None,
              lgb_subsample: float = None, lgb_colsample_bytree: float = None,
              lgb_min_child_samples: int = None,
              xgb_reg_alpha: float = None, xgb_reg_lambda: float = None,
              xgb_subsample: float = None, xgb_colsample_bytree: float = None,
              xgb_min_child_weight: int = None, xgb_gamma: float = None) -> Dict:
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
        
        # Auto-enable SMOTE for binary classification if not explicitly set
        if use_smote is None:
            use_smote = use_binary_classification  # Auto-enable SMOTE for binary classification to reduce bias
        
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
        logger.info(f"  • SMOTE balancing: {use_smote}")
        
        # Validate training data before training
        if HAS_DATA_VALIDATION:
            try:
                validator = TrainingDataValidator()
                validation_report = validator.validate_training_samples(training_samples)
                
                if not validation_report['valid']:
                    logger.error(f"❌ Training data validation failed:")
                    for issue in validation_report['issues']:
                        logger.error(f"   - {issue}")
                    return {
                        "error": "Training data validation failed",
                        "validation_report": validation_report
                    }
                
                if validation_report.get('warnings'):
                    logger.warning(f"⚠️ Training data validation warnings:")
                    for warning in validation_report['warnings']:
                        logger.warning(f"   - {warning}")
                
                # Clean training samples if needed
                if validation_report.get('missing_percentage', 0) > 0 or validation_report.get('outlier_percentage', 0) > 10:
                    training_samples = validator.clean_training_samples(training_samples, validation_report)
                    logger.info(f"Cleaned training samples: {len(training_samples)} samples after cleaning")
                
                logger.info(f"✅ Training data validation passed")
            except Exception as e:
                logger.warning(f"Training data validation error: {e}. Proceeding with training anyway.")
        
        # Create version backup before training (if versioning enabled)
        version_number = None
        if self.version_manager:
            try:
                # Get current metadata for backup
                current_metadata = {}
                if self.metadata_file.exists():
                    try:
                        with open(self.metadata_file, 'r') as f:
                            current_metadata = json.load(f)
                    except (IOError, OSError, json.JSONDecodeError) as e:
                        logger.debug(f"Could not load current metadata for backup: {e}")
                        current_metadata = {}
                
                version_number = self.version_manager.generate_version_number()
                backup_created = self.version_manager.create_version_backup(version_number, current_metadata)
                if backup_created:
                    logger.info(f"📦 Created version backup: {version_number}")
            except Exception as e:
                logger.warning(f"Could not create version backup: {e}")
        
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
            
            # Reduced minimum from 100 to 30 for incremental training scenarios (backtest results)
            if len(binary_mask) < ML_MIN_BINARY_SAMPLES:
                return {"error": f"Not enough binary samples after filtering HOLD: {len(binary_mask)} < {ML_MIN_BINARY_SAMPLES}"}
            
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
        
        # Track which samples were kept after filtering (for regime detection alignment)
        kept_sample_indices = None
        
        # Binary classification mode: convert BUY/SELL/HOLD to UP/DOWN (skip HOLD)
        # This achieves ~65% accuracy vs ~45% for 3-class
        if use_binary_classification:
            # Check if labels are already binary (UP/DOWN) - this can happen in regime model training
            from collections import Counter
            label_dist = Counter(y)
            unique_labels = set(y)
            
            # Check if labels are already binary (UP/DOWN instead of BUY/SELL/HOLD)
            if unique_labels.issubset({'UP', 'DOWN'}):
                # Labels are already binary - skip conversion
                logger.info("Labels are already in binary format (UP/DOWN). Skipping conversion.")
                if len(y) < ML_MIN_BINARY_SAMPLES:
                    logger.warning(f"Not enough binary samples ({len(y)} < {ML_MIN_BINARY_SAMPLES}). "
                                 f"Label distribution: {dict(label_dist)}. "
                                 f"Falling back to 3-class classification.")
                    use_binary_classification = False
                    kept_sample_indices = None
                else:
                    kept_sample_indices = list(range(len(y)))  # Keep all samples
                    logger.info(f"✅ Using existing binary labels: {len(y)} samples")
                    logger.info(f"Binary label distribution: {dict(label_dist)}")
            else:
                # Labels are in BUY/SELL/HOLD format - convert to binary
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
                
                # Check label distribution BEFORE filtering (from original y array)
                original_label_dist = Counter(y)
                logger.info(f"Label distribution before binary filtering: {dict(original_label_dist)}")
                
                # Reduced minimum from 100 to 30 for incremental training scenarios (backtest results)
                if len(binary_mask) < 30:
                    # Auto-disable binary classification if not enough samples
                    logger.warning(f"Not enough binary samples ({len(binary_mask)} < 30). "
                                 f"Original label distribution: {dict(original_label_dist)}. "
                                 f"Falling back to 3-class classification (BUY/SELL/HOLD).")
                    logger.warning("💡 Tip: Use stocks with more volatility or longer time periods to get more BUY/SELL signals.")
                    use_binary_classification = False  # Fall back to 3-class
                    kept_sample_indices = None  # Keep all samples
                else:
                    kept_sample_indices = binary_mask  # Track which samples were kept
                    X = X[binary_mask]
                    y = np.array(binary_labels)
                    binary_label_dist = Counter(y)
                    logger.info(f"✅ Binary classification successful: {len(X)} samples (removed {len(training_samples) - len(X)} HOLD samples)")
                    logger.info(f"Binary label distribution (UP/DOWN): {dict(binary_label_dist)}")
        
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
            # Update kept_sample_indices to reflect duplicate removal
            if kept_sample_indices is not None:
                kept_sample_indices = [kept_sample_indices[i] for i in unique_indices]
            else:
                kept_sample_indices = unique_indices
            X = X[unique_indices]
            y = y[unique_indices]
        else:
            # No duplicates removed, but ensure kept_sample_indices is set
            if kept_sample_indices is None:
                kept_sample_indices = list(range(len(X)))
        
        # Check class distribution
        label_counts = Counter(y)
        logger.info(f"Training data distribution: {dict(label_counts)}")
        
        # Warn if severe class imbalance
        if len(label_counts) > 1:
            min_class = min(label_counts.values())
            max_class = max(label_counts.values())
            if min_class < max_class * 0.1:  # Less than 10% of majority class
                logger.warning(f"Severe class imbalance detected ({min_class} vs {max_class}). This may affect model performance.")
        
        # ===== REGIME-SPECIFIC MODEL TRAINING =====
        if use_regime_models:
            logger.info("Training regime-specific models (bull/bear/sideways)...")
            # Filter training_samples to match the kept indices (after binary filtering and duplicate removal)
            # kept_sample_indices now contains the original indices of samples that are in X
            filtered_training_samples = [training_samples[i] for i in kept_sample_indices]
            
            # Verify alignment
            if len(filtered_training_samples) != len(X):
                logger.error(f"Sample count mismatch: {len(filtered_training_samples)} training samples vs {len(X)} features")
                return {"error": f"Sample alignment error: {len(filtered_training_samples)} != {len(X)}"}
            
            regime_result = self._train_regime_models(
                X, y, filtered_training_samples, test_size, random_seed,
                use_hyperparameter_tuning, feature_selection_method,
                disable_feature_selection, use_smote, use_interactions,
                model_family, hold_confidence_threshold, use_binary_classification,
                rf_n_estimators, rf_max_depth, rf_min_samples_split, rf_min_samples_leaf,
                gb_n_estimators, gb_max_depth,
                lgb_reg_alpha, lgb_reg_lambda, lgb_subsample, lgb_colsample_bytree, lgb_min_child_samples,
                xgb_reg_alpha, xgb_reg_lambda, xgb_subsample, xgb_colsample_bytree, xgb_min_child_weight, xgb_gamma
            )
            
            # If regime training failed, fall back to single model
            if regime_result is None:
                logger.warning("Regime-specific training failed. Falling back to single model training.")
                use_regime_models = False  # Disable regime models and continue with normal training
            else:
                return regime_result
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # TEMPORAL SPLIT: Use date-based split instead of random to prevent data leakage
        # This ensures model doesn't see "future" patterns during training
        use_temporal_split = True
        if use_temporal_split and training_samples and len(training_samples) > 0:
            # Extract dates from training samples
            try:
                from datetime import datetime
                sample_dates = []
                for sample in training_samples:
                    date_str = sample.get('date', '')
                    if date_str:
                        try:
                            sample_dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
                        except:
                            sample_dates.append(None)
                    else:
                        sample_dates.append(None)
                
                # If we have dates for most samples, use temporal split
                valid_dates = [d for d in sample_dates if d is not None]
                if len(valid_dates) >= len(training_samples) * 0.8:  # At least 80% have dates
                    # Sort by date
                    sorted_indices = sorted(range(len(sample_dates)), 
                                          key=lambda i: sample_dates[i] if sample_dates[i] else datetime.min)
                    
                    # Split temporally: use first (1-test_size) for training, last test_size for testing
                    split_idx = int(len(sorted_indices) * (1 - test_size))
                    train_indices = sorted_indices[:split_idx]
                    test_indices = sorted_indices[split_idx:]
                    
                    X_train = X[train_indices]
                    X_test = X[test_indices]
                    y_train = y_encoded[train_indices]
                    y_test = y_encoded[test_indices]
                    
                    # Log temporal split info
                    if valid_dates:
                        train_dates = [sample_dates[i] for i in train_indices if sample_dates[i]]
                        test_dates = [sample_dates[i] for i in test_indices if sample_dates[i]]
                        if train_dates and test_dates:
                            logger.info(f"📅 Temporal split: Train={min(train_dates).date()} to {max(train_dates).date()} "
                                      f"({len(train_indices)} samples), "
                                      f"Test={min(test_dates).date()} to {max(test_dates).date()} ({len(test_indices)} samples)")
                else:
                    # Fallback to random split if dates not available
                    logger.warning("⚠️ Less than 80% of samples have dates, using random split (may cause data leakage)")
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y_encoded, test_size=test_size, random_state=random_seed, stratify=y_encoded
                    )
            except Exception as e:
                logger.warning(f"⚠️ Temporal split failed: {e}. Falling back to random split.")
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y_encoded, test_size=test_size, random_state=random_seed, stratify=y_encoded
                )
        else:
            # Fallback to random split if temporal split disabled or no samples
            logger.warning("⚠️ Using random split (temporal split disabled or no samples). This may cause data leakage.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=random_seed, stratify=y_encoded
            )
        
        # Apply SMOTE for class balancing if enabled and binary classification is used
        if use_smote and use_binary_classification:
            try:
                from imblearn.over_sampling import SMOTE
                # Check class imbalance
                unique, counts = np.unique(y_train, return_counts=True)
                class_dist = dict(zip(unique, counts))
                if len(class_dist) == 2:
                    max_count = max(class_dist.values())
                    min_count = min(class_dist.values())
                    imbalance_ratio = max_count / min_count if min_count > 0 else 1.0
                    
                    if imbalance_ratio > 1.3:  # More than 30% imbalance
                        logger.info(f"Applying SMOTE to balance classes (ratio: {imbalance_ratio:.2f})")
                        logger.info(f"  Before SMOTE: {class_dist}")
                        smote = SMOTE(random_state=random_seed, k_neighbors=min(5, min_count - 1))
                        X_train, y_train = smote.fit_resample(X_train, y_train)
                        unique_after, counts_after = np.unique(y_train, return_counts=True)
                        class_dist_after = dict(zip(unique_after, counts_after))
                        logger.info(f"  After SMOTE: {class_dist_after}")
                    else:
                        logger.debug(f"Class imbalance ratio ({imbalance_ratio:.2f}) is acceptable, skipping SMOTE")
            except ImportError as e:
                logger.warning(f"SMOTE requested but imbalanced-learn not available: {e}. Install with: pip install --upgrade scikit-learn imbalanced-learn")
            except Exception as e:
                logger.warning(f"SMOTE application failed: {e}. Continuing without SMOTE.")
        
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
            # Determine number of features to select (top 50-70 features to reduce overfitting)
            # Reduced feature count to prevent overfitting while maintaining performance
            total_features = len(X_train[0])
            n_features_to_select = max(50, min(70, int(total_features * 0.5)))  # 50% of features, clamped to 50-70 range
            
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
        
        # Step 4: Add feature interactions (enabled by default for better accuracy)
        # Feature interactions capture non-linear relationships between features
        if use_interactions is None:
            use_interactions = True  # Enabled by default
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
        
        # ===== HYPERPARAMETER TUNING (Enabled by default) =====
        if self.use_hyperparameter_tuning and HAS_OPTUNA and len(X_train_scaled) > 500:
            logger.info("Step 3: Tuning hyperparameters with Optuna...")
            try:
                best_params = self._tune_hyperparameters(X_train_scaled, y_train, random_seed)
                if best_params:
                    logger.info(f"Best hyperparameters found: {best_params}")
                else:
                    logger.info("Hyperparameter tuning completed but no optimal parameters found - using defaults")
            except Exception as e:
                logger.warning(f"Hyperparameter tuning failed: {e}. Using default parameters.")
                best_params = None
        else:
            best_params = None
            if not HAS_OPTUNA:
                logger.debug("Optuna not available - skipping hyperparameter tuning")
            elif len(X_train_scaled) <= 500:
                logger.debug(f"Not enough samples ({len(X_train_scaled)}) for hyperparameter tuning - need >500")
        
        # Create model - ANTI-OVERFITTING configuration
        # Reduced complexity and increased regularization to prevent overfitting
        rf_params = {
            'n_estimators': rf_n_estimators if rf_n_estimators is not None else ML_RF_N_ESTIMATORS,
            'max_depth': rf_max_depth if rf_max_depth is not None else ML_RF_MAX_DEPTH,
            'min_samples_split': rf_min_samples_split if rf_min_samples_split is not None else ML_RF_MIN_SAMPLES_SPLIT,
            'min_samples_leaf': rf_min_samples_leaf if rf_min_samples_leaf is not None else ML_RF_MIN_SAMPLES_LEAF,
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': random_seed,
            'n_jobs': get_safe_n_jobs(ML_TRAINING_CPU_CORES)
        }
        # Add max_samples if sklearn version supports it (0.22+)
        try:
            from sklearn import __version__ as sklearn_version
            if tuple(map(int, sklearn_version.split('.')[:2])) >= (0, 22):
                rf_params['max_samples'] = 0.8  # Use only 80% of samples per tree (bootstrap sampling)
        except (ValueError, AttributeError, ImportError) as e:
            logger.debug(f"Could not check sklearn version for max_samples: {e}")
        
        # Apply tuned parameters if available (only if not overridden)
        if best_params and 'rf' in best_params:
            for key, value in best_params['rf'].items():
                if key not in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']:
                    rf_params[key] = value
        
        rf = RandomForestClassifier(**rf_params)

        # OPTIMIZED: Stacking ensemble with LightGBM/XGBoost (achieves best validation accuracy)
        if model_family == 'voting':
            # Use LightGBM if available, otherwise XGBoost, otherwise fallback to GradientBoosting
            if HAS_LIGHTGBM:
                gb_params = {
                    'n_estimators': gb_n_estimators if gb_n_estimators is not None else ML_GB_N_ESTIMATORS,
                    'max_depth': gb_max_depth if gb_max_depth is not None else ML_GB_MAX_DEPTH,
                    'learning_rate': ML_GB_LEARNING_RATE,
                    'subsample': lgb_subsample if lgb_subsample is not None else 0.6,  # Configurable
                    'colsample_bytree': lgb_colsample_bytree if lgb_colsample_bytree is not None else 0.6,  # Configurable
                    'min_child_samples': lgb_min_child_samples if lgb_min_child_samples is not None else 50,  # Configurable
                    'reg_alpha': lgb_reg_alpha if lgb_reg_alpha is not None else 1.0,  # Configurable
                    'reg_lambda': lgb_reg_lambda if lgb_reg_lambda is not None else 3.0,  # Configurable
                    'random_state': random_seed,
                    'verbose': -1
                }
                # Apply tuned parameters if available
                if best_params and 'lgb' in best_params:
                    gb_params.update(best_params['lgb'])
                gb = LGBMClassifier(**gb_params)
                logger.info("Using LightGBM in ensemble")
            elif HAS_XGBOOST:
                gb_params = {
                    'n_estimators': gb_n_estimators if gb_n_estimators is not None else ML_GB_N_ESTIMATORS,
                    'max_depth': gb_max_depth if gb_max_depth is not None else ML_GB_MAX_DEPTH,
                    'learning_rate': ML_GB_LEARNING_RATE,
                    'subsample': xgb_subsample if xgb_subsample is not None else 0.6,  # Configurable
                    'colsample_bytree': xgb_colsample_bytree if xgb_colsample_bytree is not None else 0.6,  # Configurable
                    'min_child_weight': xgb_min_child_weight if xgb_min_child_weight is not None else 5,  # Configurable
                    'gamma': xgb_gamma if xgb_gamma is not None else 0.2,  # Configurable
                    'reg_alpha': xgb_reg_alpha if xgb_reg_alpha is not None else 1.0,  # Configurable
                    'reg_lambda': xgb_reg_lambda if xgb_reg_lambda is not None else 3.0,  # Configurable
                    'random_state': random_seed,
                    'eval_metric': 'mlogloss'
                }
                # Apply tuned parameters if available
                if best_params and 'xgb' in best_params:
                    gb_params.update(best_params['xgb'])
                gb = XGBClassifier(**gb_params)
                logger.info("Using XGBoost in ensemble")
            else:
                # Fallback to sklearn GradientBoosting
                gb = GradientBoostingClassifier(
                    n_estimators=gb_n_estimators if gb_n_estimators is not None else ML_GB_N_ESTIMATORS,
                    max_depth=gb_max_depth if gb_max_depth is not None else ML_GB_MAX_DEPTH,
                    learning_rate=0.05,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    subsample=0.8,
                    random_state=random_seed
                )
                logger.info("Using sklearn GradientBoosting (LightGBM/XGBoost not available)")
            
            lr = LogisticRegression(
                max_iter=1000,
                C=0.1,  # Reduced from 0.2 for stronger regularization
                class_weight='balanced',
                random_state=random_seed,
                penalty='l2'  # Explicit L2 regularization
            )
            
            # Use StackingClassifier instead of VotingClassifier for better performance
            if HAS_LIGHTGBM or HAS_XGBOOST:
                # Stacking ensemble: meta-learner learns optimal combination
                self.model = StackingClassifier(
                    estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
                    final_estimator=LogisticRegression(max_iter=1000, C=0.2, class_weight='balanced', random_state=random_seed, penalty='l2'),  # Reduced C from 0.5 for stronger regularization
                    cv=5,  # 5-fold cross-validation for meta-learner
                    stack_method='predict_proba',  # Use probabilities for better combination
                    n_jobs=get_safe_n_jobs(ML_TRAINING_CPU_CORES)
                )
                logger.info("Using Stacking ensemble with LightGBM/XGBoost (optimal performance)")
            else:
                # Fallback to VotingClassifier if advanced libraries not available
                self.model = VotingClassifier(
                    estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
                    voting='soft',
                    weights=[2, 1, 1]
                )
                logger.info("Using Voting ensemble (sklearn-only)")
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
        train_test_gap = train_score - test_score
        if train_score > 0.90 and train_test_gap > 0.15:
            logger.warning(f"⚠️ SEVERE OVERFITTING DETECTED!")
            logger.warning(f"   Training: {train_score*100:.1f}%, Test: {test_score*100:.1f}%, Gap: {train_test_gap*100:.1f}%")
            logger.warning("   Model is memorizing training data. Consider:")
            logger.warning("   - Increasing regularization (already applied)")
            logger.warning("   - Using more diverse training data")
            logger.warning("   - Reducing model complexity further")
        elif train_test_gap > 0.10:
            logger.warning(f"⚠️ Moderate overfitting detected. Train: {train_score*100:.1f}%, Test: {test_score*100:.1f}%, Gap: {train_test_gap*100:.1f}%")
        
        # Log generalization quality
        if train_test_gap < 0.05:
            logger.info(f"✅ Excellent generalization! Train-test gap: {train_test_gap*100:.1f}%")
        elif train_test_gap < 0.10:
            logger.info(f"✓ Good generalization. Train-test gap: {train_test_gap*100:.1f}%")
        
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
        # NOTE: CV runs on training set (X_train_scaled, y_train) which is correct
        # The temporal split above ensures no data leakage between train/test
        # CV scores may still be optimistic due to overfitting, but test_score is the real metric
        cv_scores = np.array([test_score])  # Default to test score
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')
                logger.info("Running cross-validation on training set...")
                # Use fewer folds for StackingClassifier to avoid nested CV issues
                cv_folds = 3 if isinstance(self.model, StackingClassifier) else 5
                cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv_folds, scoring='accuracy', n_jobs=get_safe_n_jobs(ML_TRAINING_CPU_CORES))
                if len(cv_scores) > 0 and not np.isnan(cv_scores).any():
                    logger.info(f"Cross-validation complete: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}% (folds: {cv_folds}, scores: {[f'{s*100:.1f}%' for s in cv_scores]})")
                else:
                    logger.warning("Cross-validation returned invalid scores. Using test score as fallback.")
                    cv_scores = np.array([test_score])
        except Exception as e:
            logger.warning(f"Cross-validation failed: {e}. Using test score as fallback.")
            import traceback
            logger.debug(traceback.format_exc())
            cv_scores = np.array([test_score])  # Fallback to test score if CV fails
        
        # Save model securely
        self.storage.save_model(self.model, self.model_file)
        self.storage.save_model(self.scaler, self.scaler_file)
        # Save label encoder (needed for prediction)
        label_encoder_file = self.data_dir / f"label_encoder_{self.strategy}.pkl"
        self.storage.save_model(self.label_encoder, label_encoder_file)
        
        # Store accuracy values for access by regime model training
        self.train_accuracy = float(train_score)
        self.test_accuracy = float(test_score)
        
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
        
        # Register new version and check if it should be activated
        should_activate = True
        if self.version_manager:
            try:
                # Generate version number if not already created
                if not version_number:
                    version_number = self.version_manager.generate_version_number()
                
                registered_version, should_activate = self.version_manager.register_new_version(
                    metadata=self.metadata,
                    test_accuracy=test_score,
                    train_accuracy=train_score,
                    cv_mean=cv_scores.mean() if len(cv_scores) > 0 else None
                )
                
                if not should_activate:
                    # New model is worse, rollback to previous version
                    logger.warning(f"⚠️ New model performs worse than current. Rolling back...")
                    current_version = self.version_manager.get_current_version()
                    if current_version:
                        rollback_success = self.version_manager.rollback_to_version(current_version.get('version'))
                        if rollback_success:
                            logger.info(f"✅ Rolled back to previous version {current_version.get('version')}")
                            # Reload the previous model
                            self._load_model()
                            # Return previous metadata
                            return self.metadata
                        else:
                            logger.error("❌ Rollback failed! New model may be active despite being worse.")
                    else:
                        logger.warning("⚠️ No previous version to rollback to. Keeping new model.")
                else:
                    logger.info(f"✅ New model version {registered_version} activated (accuracy: {test_score*100:.1f}%)")
            except Exception as e:
                logger.error(f"Error in version management: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        # Only save if versioning says we should activate, or if versioning is disabled
        if should_activate or not self.version_manager:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)

            logger.info(f"Trained and saved ML model: {test_score*100:.1f}% accuracy")
        else:
            # Don't save if we rolled back
            logger.info(f"Model training complete but not saved (rolled back to previous version)")
            # Don't save if we rolled back
            logger.info(f"Model training complete but not saved (rolled back to previous version)")
        
        return self.metadata
    
    def _train_regime_models(self, X: np.ndarray, y: np.ndarray, training_samples: List[Dict],
                            test_size: float, random_seed: int, use_hyperparameter_tuning: bool,
                            feature_selection_method: str, disable_feature_selection: bool,
                            use_smote: bool, use_interactions: bool, model_family: str,
                            hold_confidence_threshold: float, use_binary_classification: bool,
                            rf_n_estimators: int, rf_max_depth: int, rf_min_samples_split: int,
                            rf_min_samples_leaf: int, gb_n_estimators: int, gb_max_depth: int,
                            lgb_reg_alpha: float = None, lgb_reg_lambda: float = None,
                            lgb_subsample: float = None, lgb_colsample_bytree: float = None,
                            lgb_min_child_samples: int = None,
                            xgb_reg_alpha: float = None, xgb_reg_lambda: float = None,
                            xgb_subsample: float = None, xgb_colsample_bytree: float = None,
                            xgb_min_child_weight: int = None, xgb_gamma: float = None) -> Dict:
        """Train separate models for each market regime"""
        from algorithm_improvements import MarketRegimeDetector
        import yfinance as yf
        
        # Initialize regime detector
        regime_detector = MarketRegimeDetector()
        
        # Detect regime for each training sample
        logger.info("Detecting market regime for each training sample...")
        regime_labels = []
        spy_cache = {}  # Cache SPY data by date to avoid repeated fetches
        
        for i, sample in enumerate(training_samples):
            regime = 'sideways'  # Default
            sample_date = sample.get('date')
            
            try:
                    # Try to fetch SPY data for regime detection
                if sample_date:
                    # Use cached SPY data if available
                    if sample_date not in spy_cache:
                        try:
                            # Fetch SPY data up to the sample date
                            # Use a wider date range and more robust fetching
                            end_date = pd.to_datetime(sample_date) + pd.Timedelta(days=1)
                            start_date = end_date - pd.Timedelta(days=365)  # Get 1 year of data
                            
                            # Try multiple methods to fetch SPY data
                            hist = None
                            
                            # Method 1: Direct yfinance (without progress parameter for compatibility)
                            try:
                                ticker = yf.Ticker('SPY')
                                # Try with start/end dates first
                                try:
                                    hist = ticker.history(start=start_date.strftime('%Y-%m-%d'), 
                                                         end=end_date.strftime('%Y-%m-%d'))
                                except TypeError:
                                    # Some yfinance versions don't support start/end
                                    hist = ticker.history(period="1y")
                                    if not hist.empty:
                                        hist = hist[(hist.index >= start_date) & (hist.index <= end_date)]
                                
                                if hist.empty or len(hist) < 20:  # Relaxed from 50 to 20
                                    hist = None
                            except Exception as e1:
                                hist = None
                            
                            # Method 2: Try with period and filter
                            if hist is None or (isinstance(hist, pd.DataFrame) and hist.empty):
                                try:
                                    ticker = yf.Ticker('SPY')
                                    hist = ticker.history(period="1y")
                                    if not hist.empty:
                                        # Filter to date range
                                        hist = hist[hist.index <= end_date]
                                        if len(hist) < 20:  # Relaxed from 50 to 20
                                            hist = None
                                except Exception as e2:
                                    hist = None
                            
                            if hist is not None and not hist.empty and len(hist) >= 20:  # Relaxed from 50 to 20
                                # Handle MultiIndex columns if present (normalize to simple columns)
                                if isinstance(hist.columns, pd.MultiIndex):
                                    # Extract Close column and flatten
                                    if 'Close' in hist.columns.get_level_values(1):
                                        hist = hist.xs('Close', level=1, axis=1).to_frame()
                                        hist.columns = ['Close']
                                    elif 'close' in hist.columns.get_level_values(1):
                                        hist = hist.xs('close', level=1, axis=1).to_frame()
                                        hist.columns = ['Close']
                                spy_cache[sample_date] = hist
                            else:
                                spy_cache[sample_date] = None
                                if i < 10:  # Log first 10 failures for debugging
                                    logger.debug(f"Could not fetch sufficient SPY data for {sample_date} (got {len(hist) if hist is not None else 0} rows, need 20)")
                        except Exception as e:
                            logger.debug(f"Error fetching SPY data for {sample_date}: {e}")
                            spy_cache[sample_date] = None
                    
                    spy_data = spy_cache.get(sample_date)
                    if spy_data is not None and not spy_data.empty and len(spy_data) >= 20:  # Relaxed from 50 to 20
                        try:
                            # Handle MultiIndex columns from yfinance (e.g., ('SPY', 'Close'))
                            spy_data_for_regime = spy_data.copy()
                            
                            # Check for MultiIndex columns
                            if isinstance(spy_data_for_regime.columns, pd.MultiIndex):
                                # Extract Close column from MultiIndex
                                if 'Close' in spy_data_for_regime.columns.get_level_values(1):
                                    spy_data_for_regime = spy_data_for_regime.xs('Close', level=1, axis=1).to_frame()
                                    spy_data_for_regime.columns = ['close']
                                elif 'close' in spy_data_for_regime.columns.get_level_values(1):
                                    spy_data_for_regime = spy_data_for_regime.xs('close', level=1, axis=1).to_frame()
                                    spy_data_for_regime.columns = ['close']
                                else:
                                    # Try to get first numeric column
                                    numeric_cols = spy_data_for_regime.select_dtypes(include=[np.number])
                                    if not numeric_cols.empty:
                                        spy_data_for_regime = numeric_cols.iloc[:, [0]].copy()
                                        spy_data_for_regime.columns = ['close']
                                    else:
                                        spy_data_for_regime = None
                            else:
                                # Normal columns - handle 'Close' or 'close'
                                if 'Close' in spy_data_for_regime.columns:
                                    spy_data_for_regime = spy_data_for_regime[['Close']].rename(columns={'Close': 'close'})
                                elif 'close' in spy_data_for_regime.columns:
                                    spy_data_for_regime = spy_data_for_regime[['close']].copy()
                                else:
                                    # Try to infer close column from numeric columns
                                    numeric_cols = spy_data_for_regime.select_dtypes(include=[np.number])
                                    if not numeric_cols.empty:
                                        spy_data_for_regime = numeric_cols.iloc[:, [0]].copy()
                                        spy_data_for_regime.columns = ['close']
                                    else:
                                        spy_data_for_regime = None
                            
                            # Detect regime using SPY data
                            if spy_data_for_regime is not None and not spy_data_for_regime.empty and 'close' in spy_data_for_regime.columns:
                                # Ensure we have enough data points
                                if len(spy_data_for_regime) >= 20:
                                    try:
                                        regime_info = regime_detector.detect_regime(spy_data_for_regime)
                                        regime = regime_info.get('regime', 'sideways')
                                        if regime == 'unknown':
                                            regime = 'sideways'
                                        
                                        # Log all detections for first 10 samples to debug
                                        if i < 10:
                                            logger.info(f"Sample {i+1}: Regime={regime}, trend={regime_info.get('trend_strength', 0):.4f}, price={spy_data_for_regime['close'].iloc[-1]:.2f}, sma50={regime_info.get('sma_50', 0):.2f}")
                                        elif regime != 'sideways' and i % 500 == 0:
                                            logger.info(f"Detected regime {regime} from SPY data for {sample_date} (sample {i+1}), trend_strength={regime_info.get('trend_strength', 0):.4f}")
                                    except Exception as detect_e:
                                        if i < 5:
                                            logger.warning(f"Error in detect_regime for sample {i+1}: {detect_e}")
                                        regime = 'sideways'
                                else:
                                    if i < 5:
                                        logger.debug(f"SPY data has insufficient rows: {len(spy_data_for_regime)} (need 20)")
                                    regime = 'sideways'
                            else:
                                if i < 5:
                                    cols_list = list(spy_data.columns) if hasattr(spy_data, 'columns') else 'unknown'
                                    if isinstance(cols_list, pd.MultiIndex):
                                        cols_list = [str(c) for c in cols_list]
                                    logger.debug(f"SPY data missing 'close' column for {sample_date} (sample {i+1}), columns={cols_list}, data_for_regime={spy_data_for_regime is not None}")
                                regime = 'sideways'
                        except Exception as e:
                            if i < 5:
                                import traceback
                                logger.warning(f"Error detecting regime from SPY data for {sample_date} (sample {i+1}): {e}\n{traceback.format_exc()}")
                            regime = 'sideways'
                    else:
                        # Fallback: Use stock's own price data for regime detection if SPY unavailable
                        sample_symbol = sample.get('symbol', '')
                        if sample_symbol:
                            # Try to fetch stock's own historical data for regime detection
                            try:
                                ticker = yf.Ticker(sample_symbol)
                                # Fetch recent history (1 year) - remove progress parameter for compatibility
                                stock_hist = ticker.history(period="1y")
                                if not stock_hist.empty and len(stock_hist) >= 20:
                                    # Filter to date range if sample_date available
                                    if sample_date:
                                        sample_datetime = pd.to_datetime(sample_date)
                                        stock_hist = stock_hist[stock_hist.index <= sample_datetime]
                                    
                                    if len(stock_hist) >= 20:
                                        # Handle MultiIndex columns from yfinance
                                        stock_hist_for_regime = None
                                        if isinstance(stock_hist.columns, pd.MultiIndex):
                                            # Extract Close column from MultiIndex
                                            if 'Close' in stock_hist.columns.get_level_values(1):
                                                stock_hist_for_regime = stock_hist.xs('Close', level=1, axis=1).to_frame()
                                                stock_hist_for_regime.columns = ['close']
                                            elif 'close' in stock_hist.columns.get_level_values(1):
                                                stock_hist_for_regime = stock_hist.xs('close', level=1, axis=1).to_frame()
                                                stock_hist_for_regime.columns = ['close']
                                        elif 'Close' in stock_hist.columns:
                                            stock_hist_for_regime = stock_hist[['Close']].rename(columns={'Close': 'close'})
                                        elif 'close' in stock_hist.columns:
                                            stock_hist_for_regime = stock_hist[['close']].copy()
                                        
                                        if stock_hist_for_regime is not None and not stock_hist_for_regime.empty and 'close' in stock_hist_for_regime.columns:
                                            if len(stock_hist_for_regime) >= 20:
                                                try:
                                                    regime_info = regime_detector.detect_regime(stock_hist_for_regime)
                                                    regime = regime_info.get('regime', 'sideways')
                                                    if regime == 'unknown':
                                                        regime = 'sideways'
                                                    
                                                    # Log all detections for first 10 samples to debug
                                                    if i < 10:
                                                        logger.info(f"Sample {i+1} ({sample_symbol}): Regime={regime}, trend={regime_info.get('trend_strength', 0):.4f}, price={stock_hist_for_regime['close'].iloc[-1]:.2f}, sma50={regime_info.get('sma_50', 0):.2f}")
                                                    elif regime != 'sideways' and i % 500 == 0:
                                                        logger.info(f"Used {sample_symbol}'s own data for regime detection: {regime} (sample {i+1}), trend_strength={regime_info.get('trend_strength', 0):.4f}")
                                                except Exception as detect_e:
                                                    if i < 5:
                                                        logger.warning(f"Error in detect_regime for {sample_symbol} (sample {i+1}): {detect_e}")
                                                    regime = 'sideways'
                                            else:
                                                regime = 'sideways'
                                        else:
                                            if i < 5:
                                                logger.debug(f"{sample_symbol} stock data format issue: hist_for_regime={stock_hist_for_regime is not None}, empty={stock_hist_for_regime.empty if stock_hist_for_regime is not None else 'N/A'}, has_close={'close' in stock_hist_for_regime.columns if stock_hist_for_regime is not None else False}")
                                            regime = 'sideways'
                                    else:
                                        regime = 'sideways'
                                else:
                                    regime = 'sideways'
                            except Exception as e:
                                if i < 5:
                                    logger.debug(f"Could not fetch {sample_symbol} data for regime detection (sample {i+1}): {e}")
                                regime = 'sideways'
                        else:
                            # Last resort: default to sideways
                            regime = 'sideways'
                else:
                    regime = 'sideways'  # Default if no date
            except Exception as e:
                logger.debug(f"Error detecting regime for sample {i}: {e}")
                regime = 'sideways'  # Default on error
            
            regime_labels.append(regime)
            
            # Progress logging every 100 samples
            if (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(training_samples)} samples for regime detection...")
        
        # Count samples per regime
        from collections import Counter
        regime_counts = Counter(regime_labels)
        logger.info(f"Regime distribution: {dict(regime_counts)}")
        
        # Check if we have enough samples for each regime
        # Lowered from 100 to 50 to allow training with smaller datasets
        min_samples_per_regime = 50  # Minimum samples needed per regime (relaxed from 100)
        regimes_to_train = []
        for regime in ['bull', 'bear', 'sideways']:
            count = regime_counts.get(regime, 0)
            if count >= min_samples_per_regime:
                regimes_to_train.append(regime)
                logger.info(f"  {regime}: {count} samples (sufficient)")
            else:
                logger.warning(f"  {regime}: {count} samples (insufficient, need {min_samples_per_regime})")
        
        if len(regimes_to_train) == 0:
            logger.warning("Not enough samples for any regime. Falling back to single model.")
            # Continue with normal training by returning None (caller will handle)
            return None
        
        # Train models for each regime
        regime_models = {}
        regime_metadata = {}
        overall_train_acc = 0.0
        overall_test_acc = 0.0
        overall_cv_mean = 0.0
        overall_cv_std = 0.0
        total_samples = 0
        
        for regime in regimes_to_train:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training {regime.upper()} regime model")
            logger.info(f"{'='*60}")
            
            # Filter samples for this regime
            regime_mask = np.array([r == regime for r in regime_labels])
            regime_X = X[regime_mask]
            regime_y = y[regime_mask]
            regime_samples = [s for i, s in enumerate(training_samples) if regime_mask[i]]
            
            logger.info(f"Training on {len(regime_X)} {regime} samples...")
            
            # Check label distribution in regime samples
            from collections import Counter
            regime_label_dist = Counter([s.get('label', 'HOLD') for s in regime_samples])
            logger.info(f"Regime {regime} label distribution: {dict(regime_label_dist)}")
            
            # Create a temporary ML instance for this regime
            # We'll train it using the same logic as the main model
            regime_ml = StockPredictionML(self.data_dir, f"{self.strategy}_{regime}")
            regime_ml.use_hyperparameter_tuning = use_hyperparameter_tuning
            regime_ml.feature_selection_method = feature_selection_method
            
            # Train the regime model (reuse existing train logic)
            # Note: regime_samples have original BUY/SELL/HOLD labels, so binary classification will work correctly
            try:
                regime_result = regime_ml.train(
                    regime_samples,
                    test_size=test_size,
                    random_seed=random_seed,
                    use_hyperparameter_tuning=use_hyperparameter_tuning,
                    use_regime_models=False,  # Don't recurse!
                    feature_selection_method=feature_selection_method,
                    disable_feature_selection=disable_feature_selection,
                    use_smote=use_smote,
                    use_interactions=use_interactions,
                    model_family=model_family,
                    hold_confidence_threshold=hold_confidence_threshold,
                    use_binary_classification=use_binary_classification,  # This will convert BUY/SELL to UP/DOWN
                    rf_n_estimators=rf_n_estimators,
                    rf_max_depth=rf_max_depth,
                    rf_min_samples_split=rf_min_samples_split,
                    rf_min_samples_leaf=rf_min_samples_leaf,
                    gb_n_estimators=gb_n_estimators,
                    gb_max_depth=gb_max_depth,
                    lgb_reg_alpha=lgb_reg_alpha, lgb_reg_lambda=lgb_reg_lambda,
                    lgb_subsample=lgb_subsample, lgb_colsample_bytree=lgb_colsample_bytree,
                    lgb_min_child_samples=lgb_min_child_samples,
                    xgb_reg_alpha=xgb_reg_alpha, xgb_reg_lambda=xgb_reg_lambda,
                    xgb_subsample=xgb_subsample, xgb_colsample_bytree=xgb_colsample_bytree,
                    xgb_min_child_weight=xgb_min_child_weight, xgb_gamma=xgb_gamma
                )
                
                if 'error' in regime_result:
                    logger.error(f"Error training {regime} model: {regime_result['error']}")
                    continue
                
                # Get accuracy from result or from regime_ml attributes
                train_acc = regime_result.get('train_accuracy', regime_ml.train_accuracy if hasattr(regime_ml, 'train_accuracy') else 0.0)
                test_acc = regime_result.get('test_accuracy', regime_ml.test_accuracy if hasattr(regime_ml, 'test_accuracy') else 0.0)
                cv_mean = regime_result.get('cv_mean', 0.0)
                cv_std = regime_result.get('cv_std', 0.0)
                
                # Store the regime model
                regime_models[regime] = regime_ml
                regime_metadata[regime] = {
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'samples': len(regime_X),
                    'selected_features': regime_ml.selected_features.tolist() if regime_ml.selected_features is not None else None,
                    'num_features': len(regime_ml.selected_features) if regime_ml.selected_features is not None else None
                }
                
                # Weighted average for overall metrics
                overall_train_acc += train_acc * len(regime_X)
                overall_test_acc += test_acc * len(regime_X)
                overall_cv_mean += cv_mean * len(regime_X)
                # For CV std, we'll use a weighted average (simplified approach)
                overall_cv_std += cv_std * len(regime_X)
                total_samples += len(regime_X)
                
                logger.info(f"✓ {regime} model trained: {test_acc*100:.1f}% test accuracy")
                
            except Exception as e:
                logger.error(f"Error training {regime} model: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
        
        if len(regime_models) == 0:
            logger.error("Failed to train any regime models. Falling back to single model.")
            return None
        
        # Calculate weighted average accuracy and CV scores
        if total_samples > 0:
            overall_train_acc /= total_samples
            overall_test_acc /= total_samples
            overall_cv_mean /= total_samples
            overall_cv_std /= total_samples
        
        # Store regime models
        self.regime_models = regime_models
        self.use_regime_models = True
        
        # Update metadata
        self.metadata = {
            'strategy': self.strategy,
            'train_accuracy': overall_train_acc,
            'test_accuracy': overall_test_acc,
            'cv_mean': overall_cv_mean,
            'cv_std': overall_cv_std,
            'use_regime_models': True,
            'regime_metadata': regime_metadata,
            'regime_distribution': dict(regime_counts),
            'trained_regimes': list(regime_models.keys()),
            'training_date': datetime.now().isoformat(),
            'use_binary_classification': use_binary_classification
        }
        
        # Save metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"REGIME-SPECIFIC MODELS TRAINING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Overall Training Accuracy: {overall_train_acc*100:.1f}%")
        logger.info(f"Overall Test Accuracy: {overall_test_acc*100:.1f}%")
        logger.info(f"Trained {len(regime_models)} regime models: {', '.join(regime_models.keys())}")
        
        return self.metadata
    
    def _detect_current_regime(self, history_data: dict = None, stock_data: dict = None) -> str:
        """Detect current market regime for prediction"""
        try:
            from algorithm_improvements import MarketRegimeDetector
            import yfinance as yf
            
            if self.regime_detector is None:
                self.regime_detector = MarketRegimeDetector()
            
            # Try to get SPY data for regime detection
            spy_df = None
            
            # Check cache first
            if hasattr(self, '_spy_data_cache') and hasattr(self, '_spy_data_cache_time'):
                if self._spy_data_cache_time and (datetime.now() - self._spy_data_cache_time).days < 1:
                    spy_df = self._spy_data_cache
            
            # Fetch SPY data if not cached
            if spy_df is None or spy_df.empty:
                try:
                    spy_hist = yf.download('SPY', period="1y", interval="1d", progress=False, show_errors=False)
                    if not spy_hist.empty:
                        spy_df = spy_hist
                        self._spy_data_cache = spy_df
                        self._spy_data_cache_time = datetime.now()
                except Exception as e:
                    logger.debug(f"Could not fetch SPY data for regime detection: {e}")
            
            # If SPY data available, use it
            if spy_df is not None and not spy_df.empty:
                # Handle yfinance MultiIndex columns
                if isinstance(spy_df.columns, pd.MultiIndex):
                    if 'Close' in spy_df.columns.get_level_values(1):
                        spy_df = spy_df['Close'].to_frame()
                        spy_df.columns = ['close']
                    elif 'close' in spy_df.columns.get_level_values(1):
                        spy_df = spy_df['close'].to_frame()
                        spy_df.columns = ['close']
                elif 'Close' in spy_df.columns:
                    spy_df = spy_df.rename(columns={'Close': 'close'})
                elif 'close' not in spy_df.columns and len(spy_df.columns) > 0:
                    # Assume first column is close price
                    spy_df = spy_df.rename(columns={spy_df.columns[0]: 'close'})
                
                if 'close' in spy_df.columns:
                    regime_info = self.regime_detector.detect_regime(spy_df)
                    regime = regime_info.get('regime', 'sideways')
                    # Map 'unknown' to 'sideways'
                    if regime == 'unknown':
                        regime = 'sideways'
                    return regime
            
            # Fallback: try to detect from history_data if available
            if history_data and history_data.get('data'):
                try:
                    import pandas as pd
                    df = pd.DataFrame(history_data['data'])
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df.set_index('date', inplace=True)
                    # Ensure 'close' column exists
                    if 'close' not in df.columns and 'Close' in df.columns:
                        df = df.rename(columns={'Close': 'close'})
                    if 'close' in df.columns and len(df) >= 50:
                        regime_info = self.regime_detector.detect_regime(df)
                        regime = regime_info.get('regime', 'sideways')
                        if regime == 'unknown':
                            regime = 'sideways'
                        return regime
                except Exception as e:
                    logger.debug(f"Could not detect regime from history_data: {e}")
            
            # Default to sideways if we can't detect
            return 'sideways'
            
        except Exception as e:
            logger.warning(f"Error detecting market regime: {e}. Defaulting to sideways.")
            return 'sideways'
    
    def predict(self, features: np.ndarray, stock_data: dict = None, history_data: dict = None) -> Dict:
        """Make prediction using trained model (or regime-specific model if enabled)"""
        # Check if we should use regime-specific models
        if self.use_regime_models and len(self.regime_models) > 0:
            # Detect current market regime
            current_regime = self._detect_current_regime(history_data, stock_data)
            
            if current_regime in self.regime_models:
                regime_ml = self.regime_models[current_regime]
                logger.debug(f"Using {current_regime} regime model for prediction")
                # Regime models also support the same signature
                return regime_ml.predict(features, stock_data, history_data)
            else:
                # Fallback to sideways or default model
                fallback_regime = 'sideways' if 'sideways' in self.regime_models else list(self.regime_models.keys())[0]
                logger.debug(f"{current_regime} regime model not found, using {fallback_regime} model")
                regime_ml = self.regime_models[fallback_regime]
                return regime_ml.predict(features, stock_data, history_data)
        
        # Original single model prediction
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
        scaler_expected_features = self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else None
        input_features = len(features[0])
        
        logger.debug(f"ML prediction for {self.strategy}: input_features={input_features}, scaler_expects={scaler_expected_features}, selected_features={'None' if self.selected_features is None else f'{len(self.selected_features)} indices'}")
        
        # If selected_features is None or empty, try to infer from scaler
        if (self.selected_features is None or (isinstance(self.selected_features, np.ndarray) and len(self.selected_features) == 0)) and scaler_expected_features is not None:
            if scaler_expected_features < input_features:
                # Feature selection was used - estimate number of selected features
                # If scaler expects 75 and we have 130, interactions likely add ~5, so selected = ~70
                estimated_selected = max(1, scaler_expected_features - 5)  # Assume ~5 interactions
                logger.warning(f"selected_features is None/empty but scaler expects {scaler_expected_features} < {input_features} input. Inferring {estimated_selected} selected features (estimated, may be incorrect).")
                self.selected_features = np.arange(min(estimated_selected, input_features))
            elif scaler_expected_features == input_features:
                # No feature selection needed
                logger.debug(f"selected_features is None/empty, but scaler expects {scaler_expected_features} = input {input_features}. No feature selection needed.")
                self.selected_features = None  # Keep as None
        
        # Apply feature selection if available
        if self.selected_features is not None and len(self.selected_features) > 0:
            if input_features > len(self.selected_features):
                # Select only the features that were used during training
                features = features[:, self.selected_features]
                logger.debug(f"Applied feature selection: {input_features} -> {len(features[0])} features")
            elif input_features < len(self.selected_features):
                # Features don't match - model may need retraining
                logger.warning(f"Feature count mismatch: got {input_features}, expected {len(self.selected_features)} selected features")
                return {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'error': 'Feature count mismatch - model may need retraining',
                    'probabilities': {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 100.0}
                }
        elif scaler_expected_features is not None and scaler_expected_features < input_features:
            # Fallback: if we still have more features than scaler expects, use first N
            logger.warning(f"selected_features is None but scaler expects {scaler_expected_features} < {input_features}. Using first {scaler_expected_features} features as emergency fallback.")
            features = features[:, :scaler_expected_features]
        
        # Add feature interactions if they were used during training
        # Check if interactions were used (scaler expects more features than selected)
        num_selected = len(features[0])  # Current number of features after selection
        
        use_interactions = False
        if scaler_expected_features is not None and scaler_expected_features > num_selected:
            # Scaler expects more features than we have after selection - interactions were used
            use_interactions = True
            logger.debug(f"Detected interactions: scaler expects {scaler_expected_features} features, we have {num_selected} after selection")
        
        if use_interactions:
            # Get feature names for selected features
            all_feature_names = self.feature_extractor.get_feature_names()
            if self.selected_features is not None and len(self.selected_features) > 0:
                selected_feature_names = [all_feature_names[i] for i in self.selected_features if i < len(all_feature_names)]
            else:
                selected_feature_names = all_feature_names[:num_selected]
            
            # Add interactions using the same logic as training (method is on FeatureExtractor)
            try:
                features, _ = self.feature_extractor._add_feature_interactions(features, selected_feature_names, fit=False)
                logger.debug(f"Added feature interactions: {len(features[0])} features total (was {num_selected})")
            except Exception as e:
                logger.warning(f"Could not add feature interactions during prediction: {e}. Using features without interactions.")
                # If interactions fail, we can't proceed - return error
                return {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'error': f'Feature interaction error: {e}',
                    'probabilities': {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 100.0}
                }
        
        # Final check: ensure feature count matches scaler expectations
        if scaler_expected_features is not None and len(features[0]) != scaler_expected_features:
            logger.error(f"Feature count mismatch after selection/interactions: got {len(features[0])}, scaler expects {scaler_expected_features}")
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'error': f'Feature count mismatch: got {len(features[0])}, expected {scaler_expected_features}',
                'probabilities': {'BUY': 0.0, 'SELL': 0.0, 'HOLD': 100.0}
            }
        
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
        
        # For binary classification, add confidence threshold to reduce bias
        # If confidence is too low, return HOLD instead of making a biased prediction
        is_binary_mode = (set(classes) == {'UP', 'DOWN'} or 
                         (len(classes) == 2 and 'UP' in classes and 'DOWN' in classes))
        
        # Binary confidence threshold: if max probability < 60%, return HOLD
        # This helps reduce bias toward majority class
        # Increased from 55% to 60% to be more aggressive in filtering uncertain predictions
        binary_confidence_threshold = 0.60  # 60% minimum confidence for binary predictions
        if is_binary_mode and maxp < binary_confidence_threshold:
            logger.debug(f"Low confidence binary prediction ({maxp*100:.1f}% < {binary_confidence_threshold*100}%), returning HOLD")
            return {
                'action': 'HOLD',
                'confidence': float(maxp * 100),
                'reasoning': f'Low confidence prediction ({maxp*100:.1f}%) - avoiding biased prediction',
                'probabilities': {'BUY': float(probabilities[classes.index('DOWN')] * 100) if 'DOWN' in classes else 0.0,
                                 'SELL': float(probabilities[classes.index('UP')] * 100) if 'UP' in classes else 0.0,
                                 'HOLD': 100.0 - float(maxp * 100)}
            }
        
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
        
        # Record prediction in production monitor
        if hasattr(self, 'production_monitor') and self.production_monitor:
            try:
                symbol = stock_data.get('symbol', 'UNKNOWN') if stock_data else 'UNKNOWN'
                self.production_monitor.record_prediction(
                    symbol=symbol,
                    action=action,
                    confidence=confidence,
                    features=features_scaled[0] if len(features_scaled) > 0 else None,
                    metadata={'strategy': self.strategy, 'probabilities': prob_dict}
                )
            except Exception as e:
                logger.debug(f"Error recording prediction in production monitor: {e}")
        
        # Generate explanation if explainer is available
        explanation = None
        if hasattr(self, 'model_explainer') and self.model_explainer:
            try:
                explanation = self.model_explainer.explain_prediction(
                    features=features_scaled[0] if len(features_scaled) > 0 else features[0],
                    prediction={'action': action, 'confidence': confidence, 'probabilities': prob_dict}
                )
            except Exception as e:
                logger.debug(f"Error generating explanation: {e}")
        
        result = {
            'action': action,
            'confidence': confidence,
            'probabilities': prob_dict
        }
        
        # Add explanation if available
        if explanation and explanation.get('has_explanation'):
            result['explanation'] = explanation
        
        # Score prediction quality
        if hasattr(self, 'quality_scorer') and self.quality_scorer:
            try:
                quality_score = self.quality_scorer.score_prediction(
                    prediction=result,
                    features=features_scaled[0] if len(features_scaled) > 0 else features[0]
                )
                result['quality_score'] = quality_score
                
                # Flag if quality is very low
                if quality_score.get('should_flag', False):
                    result['flagged'] = True
                    result['flag_reason'] = 'Low quality prediction detected'
                    logger.warning(f"⚠️ Low quality prediction flagged: {quality_score.get('quality_level')} "
                                 f"(score: {quality_score.get('quality_score', 0):.2f})")
            except Exception as e:
                logger.debug(f"Error scoring prediction quality: {e}")
        
        return result
    
    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, random_seed: int) -> Dict:
        """Tune hyperparameters using Optuna"""
        if not HAS_OPTUNA:
            return {}
        
        def objective(trial):
            # Suggest hyperparameters for XGBoost/LightGBM
            params = {}
            
            if HAS_LIGHTGBM:
                params['lgb'] = {
                    'n_estimators': trial.suggest_int('lgb_n_estimators', 100, 300),  # Reduced max from 500
                    'max_depth': trial.suggest_int('lgb_max_depth', 3, 5),  # Reduced max from 10 to prevent overfitting
                    'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.1, log=True),  # Reduced max from 0.3
                    'subsample': trial.suggest_float('lgb_subsample', 0.6, 0.8),  # Reduced max from 1.0
                    'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.6, 0.8),  # Reduced max from 1.0
                    'min_child_samples': trial.suggest_int('lgb_min_child_samples', 20, 50),  # Increased min from 10
                    'reg_alpha': trial.suggest_float('lgb_reg_alpha', 0.1, 1.0),  # Increased min from 0
                    'reg_lambda': trial.suggest_float('lgb_reg_lambda', 1.0, 3.0),  # Increased min from 0, max from 2
                }
            
            if HAS_XGBOOST:
                params['xgb'] = {
                    'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 300),  # Reduced max from 500
                    'max_depth': trial.suggest_int('xgb_max_depth', 3, 5),  # Reduced max from 10 to prevent overfitting
                    'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1, log=True),  # Reduced max from 0.3
                    'subsample': trial.suggest_float('xgb_subsample', 0.6, 0.8),  # Reduced max from 1.0
                    'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 0.8),  # Reduced max from 1.0
                    'min_child_weight': trial.suggest_int('xgb_min_child_weight', 2, 10),  # Increased min from 1
                    'gamma': trial.suggest_float('xgb_gamma', 0.1, 0.5),  # Increased min from 0
                    'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0.1, 1.0),  # Increased min from 0
                    'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1.0, 3.0),  # Increased min from 0, max from 2
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
    
    def _load_regime_models(self):
        """Load regime-specific models if they exist"""
        if not self.metadata.get('use_regime_models', False):
            return
        
        self.regime_models = {}
        regimes = ['bull', 'bear', 'sideways']
        
        for regime in regimes:
            regime_model_file = f"ml_model_{self.strategy}_{regime}.pkl"
            regime_scaler_file = f"scaler_{self.strategy}_{regime}.pkl"
            regime_encoder_file = f"label_encoder_{self.strategy}_{regime}.pkl"
            
            if (self.storage.model_exists(regime_model_file) and 
                self.storage.model_exists(regime_scaler_file) and
                self.storage.model_exists(regime_encoder_file)):
                try:
                    # Create a minimal regime model object without calling __init__ (which would try to load again)
                    # Use object.__new__ to create instance without calling __init__
                    regime_ml = object.__new__(StockPredictionML)
                    # Set ALL required attributes (from __init__)
                    regime_ml.data_dir = self.data_dir
                    regime_ml.strategy = f"{self.strategy}_{regime}"
                    regime_ml.model_file = f"ml_model_{self.strategy}_{regime}.pkl"
                    regime_ml.scaler_file = f"scaler_{self.strategy}_{regime}.pkl"
                    regime_ml.metadata_file = self.data_dir / f"ml_metadata_{self.strategy}_{regime}.json"
                    regime_ml.storage = self.storage
                    regime_ml.feature_extractor = self.feature_extractor
                    regime_ml.use_regime_models = False  # Regime models don't use nested regime models
                    regime_ml.regime_models = {}  # Regime models don't have nested regime models
                    regime_ml.regime_detector = None
                    regime_ml.feature_importance = {}
                    regime_ml.use_hyperparameter_tuning = False
                    regime_ml.feature_selection_method = 'importance'
                    
                    # Initialize production monitor (same as main model)
                    if HAS_MONITORING:
                        regime_ml.production_monitor = ProductionMonitor(self.data_dir, f"{self.strategy}_{regime}")
                    else:
                        regime_ml.production_monitor = None
                    
                    # Initialize other optional components
                    regime_ml.model_explainer = None
                    if HAS_QUALITY_SCORER:
                        from prediction_quality_scorer import PredictionQualityScorer
                        regime_ml.quality_scorer = PredictionQualityScorer()
                    else:
                        regime_ml.quality_scorer = None
                    
                    # Initialize version manager (optional, regime models don't need versioning)
                    regime_ml.version_manager = None
                    
                    # Load the actual model components
                    regime_ml.model = self.storage.load_model(regime_model_file)
                    regime_ml.scaler = self.storage.load_model(regime_scaler_file)
                    regime_ml.label_encoder = self.storage.load_model(regime_encoder_file)
                    
                    # Load metadata for this regime if available
                    # Try multiple sources: main metadata -> regime's own metadata file
                    regime_metadata = {}
                    if self.metadata and 'regime_metadata' in self.metadata:
                        regime_metadata = self.metadata['regime_metadata'].get(regime, {})
                    
                    # Also try loading from the regime model's own metadata file
                    if not regime_metadata and regime_ml.metadata_file.exists():
                        try:
                            with open(regime_ml.metadata_file, 'r') as f:
                                regime_own_metadata = json.load(f)
                                # Extract selected_features and other info from regime's own metadata
                                if 'selected_features' in regime_own_metadata:
                                    regime_metadata['selected_features'] = regime_own_metadata['selected_features']
                                if 'train_accuracy' in regime_own_metadata:
                                    regime_metadata['train_accuracy'] = regime_own_metadata['train_accuracy']
                                if 'test_accuracy' in regime_own_metadata:
                                    regime_metadata['test_accuracy'] = regime_own_metadata['test_accuracy']
                                logger.debug(f"Loaded regime metadata from {regime_ml.metadata_file.name}")
                        except Exception as e:
                            logger.debug(f"Could not load regime metadata from {regime_ml.metadata_file.name}: {e}")
                    
                    regime_ml.metadata = regime_metadata if regime_metadata else {}
                    regime_ml.train_accuracy = regime_metadata.get('train_accuracy', 0) if regime_metadata else 0
                    regime_ml.test_accuracy = regime_metadata.get('test_accuracy', 0) if regime_metadata else 0
                    
                    # Load selected features if available
                    # First try from regime-specific metadata (from main metadata or regime's own file)
                    if regime_metadata and 'selected_features' in regime_metadata:
                        regime_ml.selected_features = np.array(regime_metadata['selected_features'])
                        logger.info(f"Loaded selected_features for {regime} regime: {len(regime_ml.selected_features)} features")
                    # Then try from main metadata's regime_metadata section
                    elif self.metadata and 'regime_metadata' in self.metadata:
                        regime_meta = self.metadata['regime_metadata'].get(regime, {})
                        if 'selected_features' in regime_meta:
                            regime_ml.selected_features = np.array(regime_meta['selected_features'])
                            logger.info(f"Loaded selected_features for {regime} regime from main metadata: {len(regime_ml.selected_features)} features")
                        else:
                            # Fallback to main model's selected features
                            regime_ml.selected_features = self.selected_features if hasattr(self, 'selected_features') and self.selected_features is not None else None
                            if regime_ml.selected_features is not None:
                                logger.debug(f"Using main model's selected_features for {regime} regime: {len(regime_ml.selected_features)} features")
                            else:
                                logger.warning(f"No selected_features found for {regime} regime. Feature selection may not work correctly.")
                    else:
                        # Fallback to main model's selected features
                        regime_ml.selected_features = self.selected_features if hasattr(self, 'selected_features') and self.selected_features is not None else None
                        if regime_ml.selected_features is not None:
                            logger.debug(f"Using main model's selected_features for {regime} regime: {len(regime_ml.selected_features)} features")
                        else:
                            logger.warning(f"No selected_features found for {regime} regime. Feature selection may not work correctly.")
                    
                    # Verify the model loaded correctly
                    if regime_ml.model is None:
                        logger.warning(f"Regime model {regime}: model is None")
                        continue
                    if not hasattr(regime_ml.label_encoder, 'classes_'):
                        logger.warning(f"Regime model {regime}: label_encoder has no classes_ attribute")
                        continue
                    if len(regime_ml.label_encoder.classes_) == 0:
                        logger.warning(f"Regime model {regime}: label_encoder has no classes")
                        continue
                    
                    self.regime_models[regime] = regime_ml
                    logger.info(f"✓ Loaded {regime} regime model for {self.strategy} (model={regime_ml.model is not None}, encoder={len(regime_ml.label_encoder.classes_)} classes)")
                except Exception as e:
                    logger.error(f"Error loading {regime} regime model for {self.strategy}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
        
        if not self.regime_models:
            logger.warning(f"No regime models loaded for {self.strategy}. Falling back to single model logic.")
            # Try to load the main model as fallback
            if self.storage.model_exists(self.model_file):
                try:
                    self.model = self.storage.load_model(self.model_file)
                    logger.info(f"Loaded fallback single model for {self.strategy}")
                except Exception as e:
                    logger.error(f"Error loading fallback model: {e}")
        else:
            logger.info(f"Successfully loaded {len(self.regime_models)} regime model(s) for {self.strategy}")
    
    def is_trained(self) -> bool:
        """Check if model is trained and ready to use. Reloads model if files exist but model isn't loaded."""
        # First, check if model files exist but model isn't loaded - try to reload
        if self.model is None and len(self.regime_models) == 0:
            # Check if regime model files exist
            regimes = ['bull', 'bear', 'sideways']
            regime_files_exist = False
            for regime in regimes:
                regime_model_file = f"ml_model_{self.strategy}_{regime}.pkl"
                regime_scaler_file = f"scaler_{self.strategy}_{regime}.pkl"
                regime_encoder_file = f"label_encoder_{self.strategy}_{regime}.pkl"
                if (self.storage.model_exists(regime_model_file) and 
                    self.storage.model_exists(regime_scaler_file) and
                    self.storage.model_exists(regime_encoder_file)):
                    regime_files_exist = True
                    break
            
            # Check if single model files exist
            single_model_exists = (self.storage.model_exists(self.model_file) and 
                                  self.storage.model_exists(self.scaler_file))
            
            # If model files exist but model isn't loaded, try to reload
            if regime_files_exist or single_model_exists:
                logger.debug(f"Model files exist but model not loaded for {self.strategy}. Attempting to reload...")
                try:
                    self._load_model()
                except Exception as e:
                    logger.debug(f"Could not reload model: {e}")
        
        # Check for regime models first
        if self.use_regime_models and len(self.regime_models) > 0:
            # At least one regime model should have a fitted label encoder
            for regime, regime_ml in self.regime_models.items():
                if (regime_ml.model is not None and 
                    hasattr(regime_ml.label_encoder, 'classes_') and 
                    len(regime_ml.label_encoder.classes_) > 0):
                    return True
            return False
        
        # Check single model
        if self.model is None:
            return False
        # Also check if LabelEncoder is fitted (needed for predictions)
        if not hasattr(self.label_encoder, 'classes_') or len(self.label_encoder.classes_) == 0:
            return False
        return True

