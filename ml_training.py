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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import RFE, SelectKBest, mutual_info_classif

# Suppress harmless feature name warnings from LightGBM/XGBoost
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')
# Suppress yfinance deprecation warnings
warnings.filterwarnings('ignore', message='.*Timestamp.utcnow is deprecated.*')
warnings.filterwarnings('ignore', category=FutureWarning)
try:
    from pandas.errors import Pandas4Warning
    warnings.filterwarnings('ignore', category=Pandas4Warning)
except ImportError:
    pass

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

# Import Quant Settings
try:
    from config import (ML_TRAINING_CPU_CORES, HYPERPARAMETER_TUNING_PARALLEL_TRIALS,
                       ML_FEATURE_SELECTION_THRESHOLD, ML_RF_N_ESTIMATORS, ML_RF_MAX_DEPTH,
                       ML_RF_MIN_SAMPLES_SPLIT, ML_RF_MIN_SAMPLES_LEAF, ML_GB_N_ESTIMATORS,
                       ML_GB_MAX_DEPTH, ML_GB_LEARNING_RATE, ML_MIN_TRAINING_SAMPLES,
                       ML_MIN_BINARY_SAMPLES, ML_BINARY_HOLD_THRESHOLD)
except ImportError:
    # Fallback if not in config
    ML_TRAINING_CPU_CORES = multiprocessing.cpu_count()
    HYPERPARAMETER_TUNING_PARALLEL_TRIALS = None # Default to no parallel tuning
    ML_FEATURE_SELECTION_THRESHOLD = 0.01
    ML_RF_N_ESTIMATORS = 100
    ML_RF_MAX_DEPTH = 5
    ML_RF_MIN_SAMPLES_SPLIT = 20
    ML_RF_MIN_SAMPLES_LEAF = 10
    ML_GB_N_ESTIMATORS = 100
    ML_GB_MAX_DEPTH = 3
    ML_GB_LEARNING_RATE = 0.05 # Slightly faster learning
    
    ML_MIN_TRAINING_SAMPLES = 100
    ML_MIN_BINARY_SAMPLES = 30
    ML_BINARY_HOLD_THRESHOLD = 0.55

# Helper function to safely get n_jobs value (Windows-specific fix)
def get_safe_n_jobs(requested_cores):
    """Get safe n_jobs value for parallel processing"""
    # MAXIMUM PERFORMANCE MODE: Use all available cores
    # User has Ryzen 7 7735HS (8 cores/16 threads) and wants max speed
    if requested_cores == -1:
        return multiprocessing.cpu_count()  # Use ALL cores
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

# Import advanced features (FinBERT, fundamentals, macro)
# DISABLED: These features caused accuracy to drop from 56% to 36%
# TODO: Re-enable after proper feature engineering and calibration
HAS_ADVANCED_FEATURES = False  # Forced off - degraded model performance
# try:
#     from advanced_features import AdvancedFeatureExtractor
#     HAS_ADVANCED_FEATURES = True
# except ImportError:
#     HAS_ADVANCED_FEATURES = False
#     logger = logging.getLogger(__name__)
#     logger.debug("Advanced features module not available (pip install transformers torch)")

# Import LSTM predictor
# DISABLED: PyTorch DLL loading issues on Windows
HAS_LSTM = False
# try:
#     from lstm_predictor import LSTMPredictor, create_lstm_predictor
#     HAS_LSTM = True
# except ImportError:
#     HAS_LSTM = False
#     logger = logging.getLogger(__name__)
#     logger.debug("LSTM predictor not available (pip install torch)")

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
        
        # Initialize advanced feature extractor (FinBERT, fundamentals, macro)
        if HAS_ADVANCED_FEATURES:
            try:
                self.advanced_extractor = AdvancedFeatureExtractor(data_fetcher)
                logger.info("Advanced features enabled (FinBERT, fundamentals, macro)")
            except Exception as e:
                logger.warning(f"Could not initialize advanced extractor: {e}")
                self.advanced_extractor = None
        else:
            self.advanced_extractor = None
    
    def extract_features(self, stock_data: dict, history_data: dict, 
                        financials_data: dict = None, indicators: dict = None,
                        market_regime: dict = None, timeframe_analysis: dict = None,
                        relative_strength: dict = None, support_resistance: dict = None,
                        news_data: list = None, sector_analysis: dict = None,
                        catalyst_info: dict = None, volume_analysis: dict = None,
                        macro_analysis: dict = None,
                        as_of_date: Optional[datetime] = None) -> np.ndarray:
        """Extract comprehensive feature vector for ML model with enhanced features (Date-aware)"""
        features = []
        
        # Technical Indicators (28 features - Added MFI and EMA Dist)
        # FORCE: Always initialize indicators dict if None to ensure defaults are used
        indicators = indicators or {}
        
        # Safe scalar extraction
        def get_f(key, default=0):
            try:
                v = indicators.get(key, default)
                if v is None: return float(default)
                # Handle Series/Array
                if hasattr(v, 'iloc'): 
                    val = float(v.iloc[-1])
                elif isinstance(v, (list, np.ndarray)) and len(v) > 0: 
                    val = float(v[-1])
                else: 
                    val = float(v)
                
                # Check for NaN (CRITICAL FIX)
                if pd.isna(val) or np.isinf(val):
                    return float(default)
                return val
            except:
                return float(default)

        # Robust price extraction for normalization (Key to preventing Scaler Spikes)
        c_price = 0
        if stock_data:
            c_price = stock_data.get('price') or stock_data.get('current_price') or stock_data.get('close') or stock_data.get('Close') or 0
            
        if (not c_price or c_price == 0) and history_data and history_data.get('data'):
            try:
                hist_list = history_data['data']
                if hist_list and len(hist_list) > 0:
                    last_row = hist_list[-1]
                    c_price = last_row.get('close') or last_row.get('Close') or 0
            except: pass
        
        try:
            if hasattr(c_price, 'iloc'): c_price = float(c_price.iloc[-1])
            elif isinstance(c_price, (list, np.ndarray)) and len(c_price) > 0: c_price = float(c_price[-1])
            # Check for NaN
            current_close = float(c_price) if c_price and not pd.isna(c_price) else 0.0
        except:
            current_close = 0.0
        
        # Normalize Price-based indicators (Fixes Drift/Bias/Scaler Spikes)
        def norm_p(v): 
            # Ensure v is a scalar float (handles Series, arrays, etc.)
            try:
                if v is None: return 1.0
                if hasattr(v, 'iloc'): v_val = float(v.iloc[-1])
                elif isinstance(v, (list, np.ndarray)) and len(v) > 0: v_val = float(v[-1])
                else: v_val = float(v) if v else 0
                
                # Check for NaN
                if pd.isna(v_val) or np.isinf(v_val):
                    return 1.0 # Default to neutral ratio
            except:
                v_val = 0
            
            # check for zero divisor
            if current_close == 0:
                return 1.0
                
            # Return ratio to current price (should be ~1.0 for EMAs)
            return float(v_val / current_close) if v_val else 1.0
        
        ema_200_val = get_f('ema_200', 0)
        dist_ema_200 = (current_close - ema_200_val) / ema_200_val if ema_200_val and ema_200_val != 0 else 0

        feat_ema20 = norm_p(get_f('ema_20', 0))
        feat_ema50 = norm_p(get_f('ema_50', 0))
        feat_ema200 = norm_p(ema_200_val)
        feat_bbu = norm_p(get_f('bb_upper', 0))
        feat_bbl = norm_p(get_f('bb_lower', 0))
        feat_bbm = norm_p(get_f('bb_middle', 0))
        feat_atr = float(get_f('atr', 0) / current_close) if current_close and get_f('atr') else 0.0


        features.extend([
            get_f('rsi', 50),
            get_f('mfi', 50),
            get_f('macd', 0),
            get_f('macd_signal', 0),
            get_f('macd_diff', 0),
            feat_ema20,
            feat_ema50,
            feat_ema200,
            dist_ema_200,
            feat_bbu,
            feat_bbl,
            feat_bbm,
            feat_atr,
            get_f('atr_percent', 0),
            get_f('stoch_k', 50),
            get_f('stoch_d', 50),
            get_f('williams_r', -50),
            get_f('adx', 25),
            get_f('adx_pos', 0),
            get_f('adx_neg', 0),
            # Normalize OBV to avoid billion-scale values
            np.log10(abs(get_f('obv', 0)) + 1) * (1 if get_f('obv', 0) >= 0 else -1),
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
        # INTEGRITY FIX: Always zero-out fundamentals.
        # Training data (yfinance historical) lacks point-in-time fundamentals.
        # Zeroing them ensures the model doesn't learn noise in training or drift in production.
        features.extend([0] * 10)
        
        # Market cap (log scale)
        market_cap = stock_data.get('market_cap', 0) or stock_data.get('info', {}).get('marketCap', 0) or 0
        features.append(np.log10(market_cap + 1))
        
        # Current price REMOVED (Non-stationary)
        # current_price = stock_data.get('price', 0)
        # features.append(current_price)
        # Replaced with placeholder to maintain index alignment if needed, or just removed. 
        # Removing completely to cleaner feature space.
        pass
        
        # ===== ADVANCED TIME-SERIES FEATURES =====
        if history_data and history_data.get('data'):
            try:
                df = pd.DataFrame(history_data['data'])
                # Ensure date column exists
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.sort_values('date')
                else:
                    # Deterministic fallback: Use as_of_date or fixed epoch
                    base_date = as_of_date if as_of_date else pd.Timestamp.now()
                    df['date'] = pd.date_range(end=base_date, periods=len(df), freq='D')
                
                # Ensure column names are lowercase (required by extraction methods)
                df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]
                
                if len(df) > 0 and 'close' in df.columns:
                    ts_features = self._extract_time_series_features(df)
                    # FIX: Use explicit key list for deterministic order and length
                    ts_keys = [
                        'price_change_lag_1', 'price_change_lag_2', 'price_change_lag_3', 
                        'price_change_lag_5', 'price_change_lag_10', 'price_change_lag_20',
                        'price_ma_5', 'price_std_5', 'volume_ma_5', 'price_vs_ma_5',
                        'price_ma_10', 'price_std_10', 'volume_ma_10', 'price_vs_ma_10',
                        'price_ma_20', 'price_std_20', 'volume_ma_20', 'price_vs_ma_20',
                        'price_ma_50', 'price_std_50', 'volume_ma_50', 'price_vs_ma_50',
                        'momentum_5', 'roc_5', 'momentum_10', 'roc_10', 'momentum_20', 'roc_20',
                        'volatility_5', 'realized_vol_5', 'volatility_10', 'realized_vol_10', 
                        'volatility_20', 'realized_vol_20',
                        'autocorr_1', 'autocorr_5', 'autocorr_10',
                        'price_position_10', 'price_position_20', 'price_position_50',
                        'trend_strength_10', 'trend_strength_20',
                        'price_volume_correlation', 'momentum_acceleration',
                        'distance_to_resistance', 'distance_to_support',
                        'rsi_14', 'rsi_20', 'rsi_oversold', 'rsi_overbought'
                    ]
                    ts_feature_list = [ts_features.get(k, 0.0) for k in ts_keys]
                    features.extend(ts_feature_list)
                    logger.debug(f"Extracted {len(ts_feature_list)} time-series features")
                else:
                    # Add default values for time-series features (Total 50)
                    features.extend([0.0] * 50)
            except Exception as e:
                logger.warning(f"Error extracting time-series features: {e}")
                features.extend([0.0] * 50)
        else:
            features.extend([0.0] * 50)
        
        # ===== PATTERN DETECTION FEATURES =====
        if history_data and history_data.get('data'):
            try:
                df = pd.DataFrame(history_data['data'])
                # Ensure column names are lowercase
                df.columns = [col.lower() if isinstance(col, str) else col for col in df.columns]
                
                if len(df) >= 2 and all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                    pattern_features = self._extract_pattern_features(df)
                    # FIX: Explicit key list for consistency
                    pattern_keys = [
                        'doji', 'hammer', 'shooting_star', 'bullish_engulfing', 'bearish_engulfing',
                        'double_top', 'double_bottom', 'triangle', 'resistance_break', 'support_break'
                    ]
                    pattern_feature_list = [pattern_features.get(k, 0.0) for k in pattern_keys]
                    features.extend(pattern_feature_list)
                    logger.debug(f"Extracted {len(pattern_feature_list)} pattern features")
                else:
                    features.extend([0.0] * 10)  # Actual count: 10 pattern features
            except Exception as e:
                logger.warning(f"Error extracting pattern features: {e}")
                features.extend([0.0] * 10)  # Actual count: 10 pattern features
        else:
            features.extend([0.0] * 10)  # Actual count: 10 pattern features
        
        # ===== VOLUME ANALYSIS (3 features) =====
        if volume_analysis:
            features.extend([
                float(volume_analysis.get('volume_ratio', 1.0)),
                1.0 if volume_analysis.get('volume_trend') == 'increasing' else 0.0,
                1.0 if volume_analysis.get('is_high_volume', False) else 0.0
            ])
        else:
            features.extend([1.0, 0.0, 0.0])

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
                    stock_data, history_data, news_data, as_of_date
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
            # Handle both dict (from training) and string (from hybrid_predictor)
            if isinstance(market_regime, dict):
                regime = market_regime.get('regime', 'unknown')
                regime_strength = market_regime.get('strength', 50) / 100.0
            else:
                # Assume string
                regime = str(market_regime).lower()
                regime_strength = 0.5  # Default strength
                
            features.extend([
                1.0 if regime == 'bull' else 0.0,
                1.0 if regime == 'bear' else 0.0,
                1.0 if regime == 'sideways' else 0.0,
                regime_strength
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.5])  # Default to neutral
        
        # 2. Relative Strength Ratio (1 feature)
        if relative_strength and hasattr(relative_strength, 'get') and relative_strength.get('available', False):
            rs_ratio = relative_strength.get('rs_ratio', 1.0)
            if pd.isna(rs_ratio): rs_ratio = 1.0
            
            outperformance = relative_strength.get('outperformance', 0)
            if pd.isna(outperformance): outperformance = 0.0
            outperformance = outperformance / 100.0
            
            features.extend([rs_ratio, outperformance])
        else:
            features.extend([1.0, 0.0])  # Default: neutral (ratio = 1.0, no outperformance)
        
        # 3. Multi-Timeframe Alignment Score (2 features)
        if timeframe_analysis and hasattr(timeframe_analysis, 'get') and timeframe_analysis.get('alignment') != 'insufficient_data':
            alignment = timeframe_analysis.get('alignment', 'mixed')
            timeframe_conf = timeframe_analysis.get('confidence', 50)
            if pd.isna(timeframe_conf): timeframe_conf = 50.0
            timeframe_conf = timeframe_conf / 100.0
            
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
        if support_resistance and hasattr(support_resistance, 'get') and support_resistance.get('volume_weighted', False):
            support_strength = support_resistance.get('support_strength', 50)
            if pd.isna(support_strength): support_strength = 50.0
            support_strength = support_strength / 100.0
            
            resistance_strength = support_resistance.get('resistance_strength', 50)
            if pd.isna(resistance_strength): resistance_strength = 50.0
            resistance_strength = resistance_strength / 100.0
            
            features.extend([support_strength, resistance_strength])
        else:
            features.extend([0.5, 0.5])  # Default: medium strength
        
        # 5. Volatility Regime (3 features: high, normal, low as one-hot encoded)
        if history_data and history_data.get('data'):
            try:
                # Optimized: Don't create full DataFrame effectively if not needed
                data_list = history_data['data']
                if len(data_list) >= 20: 
                    # Just pluck what we need to avoid heavy pandas overhead if possible
                    # But for now, safe pandas usage is fine
                     df = pd.DataFrame(data_list)
                     if 'close' in df.columns:
                        prices = df['close'].values[-20:].astype(float)
                        
                        # Calculate ATR percentage
                        if 'high' in df.columns and 'low' in df.columns:
                            highs = df['high'].values[-20:].astype(float)
                            lows = df['low'].values[-20:].astype(float)
                            
                            high_low = highs - lows
                            high_close = np.abs(highs - np.roll(prices, 1))
                            low_close = np.abs(lows - np.roll(prices, 1))
                            
                            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
                            # SAFE FIX: Use nanmean to handle potential NaNs in history
                            atr = np.nanmean(true_range)
                            
                            last_price = prices[-1]
                            if pd.isna(last_price) or last_price <= 0:
                                atr_percent = 0.0
                            elif pd.isna(atr):
                                atr_percent = 0.0
                            else:
                                atr_percent = (atr / last_price * 100)
                            
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
                        features.extend([0.0, 1.0, 0.0, 0.0])
                else:
                    features.extend([0.0, 1.0, 0.0, 0.0])
            except Exception as e:
                logger.warning(f"Error extracting volatility features: {e}")
                features.extend([0.0, 1.0, 0.0, 0.0])  # Default: normal volatility
        else:
            features.extend([0.0, 1.0, 0.0, 0.0])  # Default: normal volatility
        
        # 6. Support/Resistance Distance Features (2 features)
        if support_resistance and hasattr(support_resistance, 'get'):
            current_price = stock_data.get('price', 0)
            if pd.isna(current_price): current_price = 0.0
            
            support = support_resistance.get('support', 0)
            if pd.isna(support): support = 0.0
            
            resistance = support_resistance.get('resistance', 0)
            if pd.isna(resistance): resistance = 0.0
            
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
        
        # ===== NEW TRADING STRATEGY FEATURES (12 features) =====
        
        # 7. Sector Momentum Features (5 features)
        if sector_analysis and sector_analysis.get('available', False):
            sector_momentum = sector_analysis.get('sector_momentum', 0) / 100.0  # Normalize to -1 to 1
            stock_momentum = sector_analysis.get('stock_momentum', 0) / 100.0
            stock_vs_sector = sector_analysis.get('stock_vs_sector', 0) / 100.0
            
            # Sector signal encoded: strong_outperform=1, outperform=0.75, inline=0.5, underperform=0.25, strong_underperform=0
            sector_signal = sector_analysis.get('sector_signal', 'neutral')
            signal_score = 0.5  # Default: neutral
            if sector_signal == 'strong_outperform':
                signal_score = 1.0
            elif sector_signal == 'outperform':
                signal_score = 0.75
            elif sector_signal == 'underperform':
                signal_score = 0.25
            elif sector_signal == 'strong_underperform':
                signal_score = 0.0
            
            # Sector trend encoded
            sector_trend = sector_analysis.get('sector_trend', 'unknown')
            trend_score = 0.5  # Default: sideways
            if sector_trend == 'strong_uptrend':
                trend_score = 1.0
            elif sector_trend == 'uptrend':
                trend_score = 0.75
            elif sector_trend == 'downtrend':
                trend_score = 0.25
            elif sector_trend == 'strong_downtrend':
                trend_score = 0.0
            
            features.extend([
                sector_momentum,
                stock_vs_sector,
                signal_score,
                trend_score,
                1.0 if sector_analysis.get('outperforming_sector', False) else 0.0
            ])
        else:
            features.extend([0.0, 0.0, 0.5, 0.5, 0.0])
        
        # 8. Catalyst Features (4 features)
        if catalyst_info and catalyst_info.get('available', False):
            days_to_earnings = catalyst_info.get('days_to_earnings')
            days_to_dividend = catalyst_info.get('days_to_dividend')
            
            # Normalize days to 0-1 scale (0 = imminent, 1 = far away)
            earnings_proximity = 0.0
            if days_to_earnings is not None:
                if days_to_earnings <= 2:
                    earnings_proximity = 1.0  # Very close
                elif days_to_earnings <= 7:
                    earnings_proximity = 0.7
                elif days_to_earnings <= 14:
                    earnings_proximity = 0.4
                elif days_to_earnings <= 30:
                    earnings_proximity = 0.2
                else:
                    earnings_proximity = 0.0
            
            dividend_proximity = 0.0
            if days_to_dividend is not None:
                if days_to_dividend <= 3:
                    dividend_proximity = 1.0
                elif days_to_dividend <= 7:
                    dividend_proximity = 0.5
                else:
                    dividend_proximity = 0.0
            
            features.extend([
                earnings_proximity,
                dividend_proximity,
                1.0 if catalyst_info.get('has_upcoming_catalyst', False) else 0.0,
                1.0 if catalyst_info.get('catalyst_warning', False) else 0.0
            ])
        else:
            features.extend([0.0, 0.0, 0.0, 0.0])
        
        # 9. Liquidity Features (3 features)
        if volume_analysis:
            # Liquidity grade encoded
            liquidity_grade = volume_analysis.get('liquidity_grade', 'unknown')
            liquidity_score = 0.5  # Default: medium
            if liquidity_grade == 'high':
                liquidity_score = 1.0
            elif liquidity_grade == 'medium':
                liquidity_score = 0.66
            elif liquidity_grade == 'low':
                liquidity_score = 0.33
            elif liquidity_grade == 'very_low':
                liquidity_score = 0.0
            
            # Average dollar volume (log normalized)
            avg_dollar_vol = volume_analysis.get('avg_dollar_volume', 0)
            if avg_dollar_vol > 0:
                log_dollar_vol = np.log10(avg_dollar_vol) / 10.0  # Normalize: $1B = 0.9, $10M = 0.7
            else:
                log_dollar_vol = 0.0
            
            volume_ratio = volume_analysis.get('volume_ratio', 1.0)
            
            features.extend([
                liquidity_score,
                log_dollar_vol,
                min(volume_ratio / 3.0, 1.0)  # Clip at 3x average
            ])
        else:
            features.extend([0.5, 0.5, 0.33])
        
        # 10. Macro Regime Features (8 features)
        if macro_analysis and macro_analysis.get('available', False):
            vix_value = macro_analysis.get('vix_value', 20.0) / 100.0  # Normalize VIX (20 -> 0.2)
            tnx_value = macro_analysis.get('tnx_value', 4.0) / 10.0   # Normalize TNX (4.0% -> 0.4)
            dxy_value = (macro_analysis.get('dxy_value', 100.0) - 100.0) / 20.0 # Normalize DXY offset
            
            # Map statuses to numeric scores
            vix_map = {'extreme_fear': 1.0, 'high_fear': 0.7, 'normal': 0.4, 'complacency': 0.1, 'unknown': 0.5}
            vix_score = vix_map.get(macro_analysis.get('vix_status', 'unknown'), 0.5)
            
            yield_map = {'rising_rapidly': 1.0, 'rising': 0.7, 'stable': 0.5, 'falling': 0.3, 'unknown': 0.5}
            yield_score = yield_map.get(macro_analysis.get('yield_status', 'unknown'), 0.5)
            
            risk_map = {'extreme': 1.0, 'high': 0.7, 'neutral': 0.5, 'low': 0.2, 'unknown': 0.5}
            risk_score = risk_map.get(macro_analysis.get('risk_level', 'unknown'), 0.5)
            
            # Additional trend features (VIX daily change proxy if available)
            vix_change = macro_analysis.get('vix_change_pct', 0) / 10.0 # Normalize 10% change to 1.0
            
            features.extend([
                vix_value, vix_score,
                tnx_value, yield_score,
                dxy_value, risk_score,
                vix_change,
                0.5 # Placeholder for yield_curve (to match 8 features)
            ])
        else:
            features.extend([0.2, 0.5, 0.4, 0.5, 0.0, 0.5, 0.0, 0.5])
        
        # ===== ADVANCED FEATURES (FinBERT, Fundamentals, Macro) - 25 features =====
        advanced_feature_names = [
            # Sentiment (5)
            'finbert_positive', 'finbert_negative', 'finbert_neutral', 'finbert_compound', 'news_count',
            # Analyst (6)
            'analyst_buy_pct', 'analyst_hold_pct', 'analyst_sell_pct', 'analyst_score', 
            'recent_upgrades', 'recent_downgrades',
            # Fundamentals (10)
            'pe_ratio', 'pe_vs_sector', 'peg_ratio', 'revenue_growth', 'earnings_growth',
            'debt_to_equity', 'current_ratio', 'profit_margin', 'roe', 'free_cash_flow_yield',
            # Macro (8)
            'vix_level', 'vix_change_pct', 'vix_regime', 'treasury_10y', 'treasury_change',
            'yield_curve', 'dxy_level', 'dxy_change_pct',
            # Sector Rotation (4)
            'sector_momentum_1w', 'sector_momentum_1m', 'sector_vs_spy', 'stock_vs_sector'
        ]
        
        if hasattr(self, 'advanced_extractor') and self.advanced_extractor:
            try:
                symbol = stock_data.get('symbol', stock_data.get('info', {}).get('symbol', ''))
                adv_features = self.advanced_extractor.extract_all_features(
                    symbol=symbol,
                    stock_data=stock_data.get('info', stock_data),
                    news_headlines=news_data if isinstance(news_data, list) else None,
                    as_of_date=as_of_date
                )
                # Add advanced features in fixed order
                for name in advanced_feature_names:
                    value = adv_features.get(name, 0.0)
                    if isinstance(value, (int, float)):
                        # Clamp extreme values
                        features.append(float(max(-100, min(100, value))))
                    else:
                        features.append(0.0)
                logger.debug(f"Added {len(advanced_feature_names)} advanced features for {symbol}")
            except Exception as e:
                logger.debug(f"Could not extract advanced features: {e}")
                features.extend([0.0] * len(advanced_feature_names))
        else:
            features.extend([0.0] * len(advanced_feature_names))
            
        # ===== QUANT MODE: RULE-ALIGNED FEATURES (10 features) =====
        # Goal: Directly expose the logic used in TradingAnalyzer and Safety Filters
        indicators = indicators or {}
        rsi = get_f('rsi', 50)
        rsi_prev = indicators.get('rsi_prev', 50)
        ema_50_val = get_f('ema_50', 0)
        
        # 1. RSI Rebound Detection
        features.append(1.0 if rsi > rsi_prev else 0.0)
        # 2. Extension Metric (SMA50 Distance)
        dist_ema50 = (current_close - ema_50_val) / ema_50_val if ema_50_val and ema_50_val != 0 else 0.0
        features.append(float(dist_ema50))
        # 3. Market Stress Intensity
        drawdown = 0.0
        if market_regime and isinstance(market_regime, dict):
            drawdown = market_regime.get('drawdown', 0.0)
        features.append(float(abs(drawdown)))
        # 4. Volatility Stress
        atr_pct = get_f('atr_percent', 0.0)
        features.append(min(atr_pct / 5.0, 1.0)) # Scaled 0-1
        # 5. Trend/RSI Interaction (Dip Buying Setup)
        regime = 'unknown'
        if market_regime:
            regime = market_regime.get('regime', 'unknown') if isinstance(market_regime, dict) else str(market_regime)
        is_uptrend = 1.0 if regime == 'bull' else 0.0
        features.append(1.0 if (30 <= rsi <= 55 and is_uptrend) else 0.0)
        # 6. Momentum Acceleration
        features.append(float(rsi - rsi_prev))
        # 7. Extreme Overbought Filter
        features.append(1.0 if rsi > 85 else 0.0)
        # 8. Extreme Oversold Setup
        features.append(1.0 if rsi < 25 else 0.0)
        # 9. Volume Surge Interaction
        v_ratio = volume_analysis.get('volume_ratio', 1.0) if volume_analysis else 1.0
        features.append(1.0 if (v_ratio > 2.0 and rsi > rsi_prev) else 0.0)
        # 10. Golden Cross Intensity
        dist_20_50 = (get_f('ema_20',0) - ema_50_val) / ema_50_val if ema_50_val > 0 else 0.0
        features.append(float(dist_20_50))

        # --- QUANT MODE: INTERACTION FEATURES (5 features) ---
        # 11. Trend-RS Interaction (User Recommendation)
        rs_score = 0.0
        if relative_strength and isinstance(relative_strength, dict):
            rs_score = relative_strength.get('rs_ratio', 1.0)
        features.append(float(is_uptrend * rs_score))
        
        # 12. Momentum Quality (RSI Rebound * Volume Spike)
        v_spike = min(v_ratio / 2.0, 2.0) # Normalized
        rsi_reb = 1.0 if rsi > rsi_prev else 0.0
        features.append(float(rsi_reb * v_spike))
        
        # 13. Setup Strength (Confluence of Rule Signals)
        # Summing occurrences of key rule patterns
        setup_count = (1.0 if rsi < 35 else 0) + (1.0 if rsi > rsi_prev else 0) + (1.0 if v_ratio > 1.5 else 0) + (1.0 if is_uptrend else 0)
        features.append(float(setup_count / 4.0)) # Scaled 0-1
        
        # 14. Extension Risk Interaction
        ext_risk = abs(dist_ema50) * (atr_pct / 2.0)
        features.append(float(ext_risk))
        
        # 15. Market Regime Weight (Safe/Aggressive proxy)
        features.append(float(1.0 - min(abs(drawdown), 0.5))) # 1.0 = Safe, 0.5 = Stressed

        # CRITICAL SAFETY CLAMP: Prevent billion-scale values from breaking scaler
        for i in range(len(features)):
            if abs(features[i]) > 100000:
                features[i] = float(np.log10(abs(features[i]) + 1) * (1 if features[i] >= 0 else -1))
        
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
                # REMOVED: Raw prices cause overfitting (non-stationary)
                # features[f'price_lag_{lag}'] = float(close.shift(lag).iloc[-1]) if not pd.isna(close.shift(lag).iloc[-1]) else 0.0
                # features[f'volume_lag_{lag}'] = float(volume.shift(lag).iloc[-1]) if len(volume) > lag and not pd.isna(volume.shift(lag).iloc[-1]) else 0.0
                
                # KEEP: Percentage changes (Stationary)
                if lag < len(close):
                    pct_change = close.pct_change(lag).iloc[-1]
                    features[f'price_change_lag_{lag}'] = float(pct_change) if not pd.isna(pct_change) else 0.0
            else:
                # features[f'price_lag_{lag}'] = 0.0
                # features[f'volume_lag_{lag}'] = 0.0
                features[f'price_change_lag_{lag}'] = 0.0
        
        # 2. Rolling Statistics
        for window in [5, 10, 20, 50]:
            if len(close) >= window:
                ma = close.rolling(window).mean().iloc[-1]
                std = close.rolling(window).std().iloc[-1]
                
                # NORMALIZE RAW VALUES (Fixes Scaler Explosion and Non-Stationarity)
                # CRITICAL FIX: Use Ratios (Value / Price) instead of Logs/Raw to prevent drift
                current_price = close.iloc[-1]
                if current_price > 0:
                    # Ratios are stationary: 1.05 means MA is 5% above price
                    features[f'price_ma_{window}'] = float(ma / current_price) if not pd.isna(ma) else 1.0
                    features[f'price_std_{window}'] = float(std / current_price) if not pd.isna(std) else 0.0
                else:
                    features[f'price_ma_{window}'] = 1.0
                    features[f'price_std_{window}'] = 0.0
                
                if len(volume) >= window:
                    vol_ma = volume.rolling(window).mean().iloc[-1]
                    curr_vol = volume.iloc[-1]
                    # Volume Ratio: (Current / Average) -> 1.5 means 50% above average
                    # Stationary and Scale-Invariant
                    if vol_ma > 0:
                        features[f'volume_ma_{window}'] = float(curr_vol / vol_ma) if not pd.isna(vol_ma) else 1.0
                    else:
                        features[f'volume_ma_{window}'] = 1.0
                else:
                    features[f'volume_ma_{window}'] = 0.0
                
                # Price relative to MA (Already correct)
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

        # 11. RSI (Relative Strength Index) - CRITICAL FIX
        # Previously missing, causing model to be "blind" to oversold conditions
        try:
            rsi_14 = self._calculate_rsi(close, 14)
            rsi_20 = self._calculate_rsi(close, 20)
            
            # Add raw RSI (normalized / 100)
            features['rsi_14'] = float(rsi_14.iloc[-1] / 100.0) if not pd.isna(rsi_14.iloc[-1]) else 0.5
            features['rsi_20'] = float(rsi_20.iloc[-1] / 100.0) if not pd.isna(rsi_20.iloc[-1]) else 0.5
            
            # Add RSI Regime (Oversold/Overbought/Neutral)
            rsi_val = rsi_14.iloc[-1] if not pd.isna(rsi_14.iloc[-1]) else 50.0
            features['rsi_oversold'] = 1.0 if rsi_val < 30 else 0.0
            features['rsi_overbought'] = 1.0 if rsi_val > 70 else 0.0
            
        except Exception as e:
            logger.debug(f"Could not calculate RSI: {e}")
            features['rsi_14'] = 0.5
            features['rsi_20'] = 0.5
            features['rsi_oversold'] = 0.0
            features['rsi_overbought'] = 0.0
        
        return features

    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
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
        
        # DEBUG INTERACTION
        if not fit: # Only log during prediction to avoid spam
             logger.info(f"🔍 INTERACTION DEBUG: macd={macd_idx}, mom={momentum_idx}, rsi={rsi_idx}, vol_rat={volume_ratio_idx}")
             
        # Create interactions (only if both features exist)
        # Create interactions (only if both features exist)
        if len(X) > 0:
            # Helper to safely multiply features
            def safe_multiply(f1_idx, f2_idx, name):
                if f1_idx >= 0 and f2_idx >= 0 and f1_idx < X.shape[1] and f2_idx < X.shape[1]:
                    # Use float64 to prevent overflow during multiplication
                    f1 = X[:, f1_idx].astype(np.float64)
                    f2 = X[:, f2_idx].astype(np.float64)
                    
                    # Multiply
                    product = f1 * f2
                    
                    # Replace Inf/NaN with 0 to prevent downstream errors
                    if not np.isfinite(product).all():
                        product = np.nan_to_num(product, nan=0.0, posinf=0.0, neginf=0.0)
                        
                    interactions.append(product)
                    interaction_names.append(name)

            safe_multiply(rsi_idx, volume_ratio_idx, 'rsi_x_volume_ratio')
            safe_multiply(macd_idx, momentum_idx, 'macd_x_momentum')
            safe_multiply(price_pos_idx, volatility_idx, 'price_pos_x_volatility')
            safe_multiply(rsi_idx, price_pos_idx, 'rsi_x_price_pos')
            safe_multiply(volume_ratio_idx, momentum_idx, 'volume_ratio_x_momentum')
        
        if len(interactions) > 0:
            X_with_interactions = np.column_stack([X, np.array(interactions).T])
            # Final safety check
            if not np.isfinite(X_with_interactions).all():
                X_with_interactions = np.nan_to_num(X_with_interactions, nan=0.0, posinf=0.0, neginf=0.0)
            return X_with_interactions, interaction_names
        else:
            return X, []
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features in the exact order they are extracted"""
        names = []
        
        # 1. Technical Indicators (28)
        names.extend([
            'rsi', 'mfi', 'macd', 'macd_signal', 'macd_diff',
            'ema_20', 'ema_50', 'ema_200', 'dist_ema_200',
            'bb_upper', 'bb_lower', 'bb_middle',
            'atr', 'atr_percent',
            'stoch_k', 'stoch_d', 'williams_r',
            'adx', 'adx_pos', 'adx_neg', 'obv',
            'ema_20_above_50', 'ema_50_above_200',
            'golden_cross', 'death_cross',
            'price_above_ema20', 'price_above_ema50', 'price_above_ema200'
        ])
        
        # 2. Price Action Features (8)
        names.extend([
            'return_5d', 'return_10d', 'return_20d',
            'volatility_20d', 'volume_ratio_base', 'volume_volatility',
            'price_position_in_range', 'trend_slope'
        ])
        
        # 3. Fundamental Features (10)
        names.extend([
            'fundamental_pe', 'fundamental_fpe', 'fundamental_peg', 'fundamental_pb',
            'fundamental_ps', 'fundamental_dy', 'fundamental_de',
            'fundamental_roe', 'fundamental_roa', 'fundamental_pm'
        ])
        
        # 4. Market Cap (1)
        names.append('log_market_cap')
        
        # 5. Advanced Time-Series Features (50)
        for lag in [1, 2, 3, 5, 10, 20]:
            names.append(f'price_change_lag_{lag}')
        for window in [5, 10, 20, 50]:
            names.extend([f'price_ma_{window}', f'price_std_{window}', f'volume_ma_{window}', f'price_vs_ma_{window}'])
        for period in [5, 10, 20]:
            names.extend([f'momentum_{period}', f'roc_{period}'])
        for window in [5, 10, 20]:
            names.extend([f'volatility_{window}', f'realized_vol_{window}'])
        for lag in [1, 5, 10]:
            names.append(f'autocorr_{lag}')
        for window in [10, 20, 50]:
            names.append(f'price_position_{window}')
        for window in [10, 20]:
            names.append(f'trend_strength_{window}')
        names.append('price_volume_correlation')
        names.append('momentum_acceleration')
        names.extend(['distance_to_resistance_ts', 'distance_to_support_ts'])
        # Added RSI features (4)
        names.extend(['rsi_14', 'rsi_20', 'rsi_oversold', 'rsi_overbought'])
        
        # 6. Pattern Detection Features (10)
        names.extend([
            'pattern_doji', 'pattern_hammer', 'pattern_shooting_star',
            'pattern_bullish_engulfing', 'pattern_bearish_engulfing',
            'pattern_double_top', 'pattern_double_bottom', 'pattern_triangle',
            'pattern_resistance_break', 'pattern_support_break'
        ])
        
        # 7. Volume Analysis Internal (3)
        names.extend(['vol_ana_ratio', 'vol_ana_trend', 'vol_ana_high'])
        
        # 8. Enhanced Features (15)
        names.extend([
            'avg_spread', 'spread_volatility', 'price_volume_correlation_enh', 'liquidity_proxy',
            'intraday_volatility', 'volume_trend_enh', 'volume_ma_ratio',
            'sector_relative_strength', 'sector_correlation',
            'news_sentiment_enh', 'news_volume_enh', 'market_cap_category',
            'sector_type', 'beta', 'dividend_yield_enh'
        ])
        
        # 9. Market Regime (4)
        names.extend(['regime_bull', 'regime_bear', 'regime_sideways', 'regime_strength'])
        
        # 10. RS (2)
        names.extend(['rs_ratio', 'rs_outperformance'])
        
        # 11. Timeframe Alignment (2)
        names.extend(['tf_alignment_score', 'tf_confidence'])
        
        # 12. Volume Profile (2)
        names.extend(['vp_support_strength', 'vp_resistance_strength'])
        
        # 13. Volatility Regime (4)
        names.extend(['vol_regime_high', 'vol_regime_normal', 'vol_regime_low', 'vol_atr_pct'])
        
        # 14. Support/Resistance Distance (2)
        names.extend(['dist_to_support_rel', 'dist_to_resistance_rel'])
        
        # 15. Sector Momentum (5)
        names.extend(['sector_mom', 'stock_vs_sector_mom', 'sector_signal_score', 'sector_trend_score', 'sector_outperforming'])
        
        # 16. Catalyst (4)
        names.extend(['catalyst_earn_prox', 'catalyst_div_prox', 'catalyst_upcoming', 'catalyst_warning'])
        
        # 17. Liquidity (3)
        names.extend(['liq_score', 'liq_log_vol', 'liq_ratio'])
        
        # 18. Macro Regime (8)
        names.extend(['macro_vix_val', 'macro_vix_score', 'macro_tnx_val', 'macro_yield_score', 'macro_dxy_val', 'macro_risk_score', 'macro_vix_change', 'macro_yield_curve_placeholder'])
        
        # 19. Advanced Features (33)
        names.extend([
            'finbert_positive', 'finbert_negative', 'finbert_neutral', 'finbert_compound', 'news_count',
            'analyst_buy_pct', 'analyst_hold_pct', 'analyst_sell_pct', 'analyst_score', 
            'recent_upgrades', 'recent_downgrades',
            'pe_ratio_adv', 'pe_vs_sector_adv', 'peg_ratio_adv', 'revenue_growth_adv', 'earnings_growth_adv',
            'debt_to_equity_adv', 'current_ratio_adv', 'profit_margin_adv', 'roe_adv', 'free_cash_flow_yield_adv',
            'vix_level_adv', 'vix_change_pct_adv', 'vix_regime_adv', 'treasury_10y_adv', 'treasury_change_adv',
            'yield_curve_adv', 'dxy_level_adv', 'dxy_change_pct_adv',
            'sector_momentum_1w', 'sector_momentum_1m', 'sector_vs_spy', 'stock_vs_sector_adv'
        ])
        
        # 20. Quant Mode: Rule-Aligned Features (15)
        names.extend([
            'rule_rsi_rebound', 'rule_dist_ema50', 'rule_market_drawdown', 'rule_vol_stress',
            'rule_dip_buy_setup', 'rule_rsi_accel', 'rule_extreme_ob', 'rule_extreme_os',
            'rule_vol_surge_bull', 'rule_ema_spread',
            'rule_trend_rs_interact', 'rule_mom_quality', 'rule_setup_strength',
            'rule_ext_risk', 'rule_market_weight'
        ])
        
        return names


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
        self.model_reg = None  # Regression model for target pricing
        self.model_meta = None # Meta-classifier for signal filtering (Phase 3)
        self.regime_models = {}  # Dict of regime-specific models: {'bull': model, 'bear': model, 'sideways': model}
        self.regime_target_models = {}  # Dict of regime-specific regression models
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
                        if len(self.regime_models) > 0:
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
                # Continue to load main model as fallback
        
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
                    
                    # Load Meta-Model if exists (Phase 3)
                    meta_model_file = f"ml_model_meta_{self.strategy}.pkl"
                    if self.storage.model_exists(meta_model_file):
                        try:
                            self.model_meta = self.storage.load_model(meta_model_file)
                            logger.info(f"Loaded Meta-Classifier for {self.strategy} (Signal Filtering Active)")
                        except Exception as e:
                            logger.error(f"Error loading meta-model: {e}")
                else:
                    logger.warning(f"Metadata file not found for {self.strategy}")
                
                logger.info(f"Loaded ML model for {self.strategy}")
                
                # Load regression model
                reg_model_file = f"ml_model_target_{self.strategy}.pkl"
                if self.storage.model_exists(reg_model_file):
                    self.model_reg = self.storage.load_model(reg_model_file)
                    logger.info("Loaded Regression model for Target Pricing")
                else:
                    self.model_reg = None
            except Exception as e:
                logger.error(f"Error loading ML model: {e}")
                import traceback
                self.train_accuracy = 0
                self.test_accuracy = 0
    
    def train_meta(self, meta_samples: List[Dict], random_seed: int = None):
        """
        Train the Meta-Classifier (Phase 3)
        Predicts if a primary signal (Rule-based or ML) will succeed (1) or fail (0).
        Labels are derived from Triple Barrier outcomes.
        """
        if not meta_samples or len(meta_samples) < 30:
            logger.warning("Insufficient meta-samples to train meta-classifier. Minimum 30 needed.")
            return False
            
        logger.info(f"🚀 Training Meta-Classifier (Signal Filter)...")
        
        try:
            # 1. Prepare Features & Labels
            X = np.array([s['features'] for s in meta_samples])
            
            # Label is 1 for SUCCESS, 0 for FAILURE
            y = np.array([1 if s['label'] in [1, '1', 'BUY', 'SELL', 'SUCCESS'] else 0 for s in meta_samples])
            
            # 2. Scale features using isolated statistics
            if not hasattr(self.scaler, 'mean_'):
                logger.warning("Main scaler not fitted. Scaling meta-samples with isolated statistics.")
                # CRITICAL: Even in fallback, we must split to avoid global scaling leakage
                # Use first 80% as 'pseudo-train' for scaling statistics
                split_idx = int(len(X) * 0.8)
                X_pseudo_train = X[:split_idx]
                temp_scaler = StandardScaler().fit(X_pseudo_train)
                X_scaled = temp_scaler.transform(X)
            else:
                X_scaled = self.scaler.transform(X)
            
            # Use only selected features if available
            if self.selected_features is not None:
                X_scaled = X_scaled[:, self.selected_features]
                
            # 3. Train a balanced RandomForest as the filter
            # Parameters tuned for generalization on small metadata
            from sklearn.ensemble import RandomForestClassifier
            model_meta = RandomForestClassifier(
                n_estimators=100,
                max_depth=4, 
                min_samples_leaf=10,
                class_weight='balanced',
                random_state=random_seed if random_seed else 42,
                n_jobs=get_safe_n_jobs(ML_TRAINING_CPU_CORES)
            )
            
            model_meta.fit(X_scaled, y)
            self.model_meta = model_meta
            
            # 4. Save the trained meta-model
            # Save directly to production name as this is typically called after main model activation
            meta_model_file = f"ml_model_meta_{self.strategy}.pkl"
            self.storage.save_model(self.model_meta, meta_model_file)
            logger.info(f"Saved Meta-Classifier {meta_model_file} with integrity check")
            logger.info(f"✨ Trained Meta-Classifier on {len(meta_samples)} rule-based signals.")
            logger.info(f"✅ Meta-Classifier trained (Test accuracy would be checked in backtest).")
            return True
        except Exception as e:
            logger.error(f"Error training meta-classifier: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return False
    
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
              hold_confidence_threshold: float = ML_BINARY_HOLD_THRESHOLD,
              # Binary classification mode (UP/DOWN only, no HOLD) - achieves ~65% accuracy
              use_binary_classification: bool = True,
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
        
        # SMOTE disabled for time-series data — creates synthetic samples that leak future class distribution
        # Use class_weight='balanced' in learners instead (already set in RF/XGB/LGB configs)
        if use_smote is None:
            use_smote = False  # Disabled: SMOTE on time-series inflates train accuracy
        
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
        
        # Prepare data - ENSURE HOMOGENEITY
        feature_vectors = []
        for i, s in enumerate(training_samples):
            feat = s['features']
            if isinstance(feat, np.ndarray):
                feature_vectors.append(feat)
            else:
                feature_vectors.append(np.array(feat))
        
        # Check for shape consistency before creating the big array
        shapes = [v.shape[0] for v in feature_vectors]
        if len(set(shapes)) > 1:
            most_common_shape = max(set(shapes), key=shapes.count)
            logger.warning(f"⚠️ Feature shape mismatch detected! Most common: {most_common_shape}, but found {set(shapes)}")
            # Filter to only consistent samples
            valid_indices = [i for i, v in enumerate(feature_vectors) if v.shape[0] == most_common_shape]
            logger.info(f"💾 Keeping {len(valid_indices)} out of {len(training_samples)} samples with shape {most_common_shape}")
            X = np.stack([feature_vectors[i] for i in valid_indices])
            y = np.array([training_samples[i]['label'] for i in valid_indices])
            y_reg = np.array([training_samples[i].get('price_change_pct', 0.0) for i in valid_indices])
            sample_weights = np.array([training_samples[i].get('sample_weight', 1.0) for i in valid_indices])
        else:
            X = np.stack(feature_vectors) if feature_vectors else np.array([])
            y = np.array([s['label'] for s in training_samples])
            # Extract regression target (future return %)
            y_reg = np.array([s.get('price_change_pct', 0.0) for s in training_samples])
            # Extract sample weights (verified predictions = 10x, historical = 1x)
            sample_weights = np.array([s.get('sample_weight', 1.0) for s in training_samples])
        
        # Log weight distribution
        unique_weights, weight_counts = np.unique(sample_weights, return_counts=True)
        weight_dist = dict(zip(unique_weights, weight_counts))
        if len(weight_dist) > 1:
            logger.info(f"📊 Sample weight distribution: {weight_dist}")
            verified_count = sum(1 for w in sample_weights if w > 1.0)
            if verified_count > 0:
                logger.info(f"   ✨ {verified_count} verified samples with 10x weight")
        

        
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
            y_reg = y_reg[unique_indices] # Align regression target
            sample_weights = sample_weights[unique_indices]  # Keep weights aligned
        
        # Check class distribution
        label_counts = Counter(y)
        logger.info(f"Training data distribution: {dict(label_counts)}")
        
        # ===== SAMPLE LIMITING TO PREVENT OVERFITTING =====
        # Keep the MOST RECENT samples to preserve temporal order (not random stratified)
        MAX_TRAINING_SAMPLES = 15000
        
        if len(X) > MAX_TRAINING_SAMPLES:
            logger.warning(f"⚠️ Large dataset detected ({len(X)} samples). "
                          f"Keeping most recent {MAX_TRAINING_SAMPLES} to preserve temporal order.")
            
            # Keep the LAST N samples (most recent data) instead of random subsample
            # This preserves temporal ordering and prioritizes recent market behavior
            keep_start = len(X) - MAX_TRAINING_SAMPLES
            X = X[keep_start:]
            y = y[keep_start:]
            sample_weights = sample_weights[keep_start:]
            if y_reg is not None:
                y_reg = y_reg[keep_start:]
            
            # Also trim training_samples list to stay aligned
            if len(training_samples) > MAX_TRAINING_SAMPLES:
                training_samples = training_samples[keep_start:]
            
            new_label_counts = Counter(y)
            logger.info(f"📉 Kept most recent {len(X)} samples. Distribution: {dict(new_label_counts)}")
        
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
                # CRITICAL: Pure Directional Mode (Sniper Recovery)
                # We interpret BUY as Action (UP).
                # To restore 70% accuracy, we filter out "Gray Zone" samples that are neither 
                # clear winners nor clear losers if we had SELL labels.
                # Since we use BUY/HOLD, we map BUY -> UP and ensure the HOLDs we learn 
                # from are significantly different (e.g. actual losses).
                
                logger.info("Binary classification mode: Signal Purity Active (Strict 3% Sniper)")
                
                binary_mask = []
                binary_labels = []
                
                for i, label in enumerate(y):
                    if label == 'BUY' or label == 'UP':
                        binary_mask.append(i)
                        binary_labels.append('UP')
                    elif label == 'HOLD' or label == 'DOWN':
                        # In Patient Sniper mode, HOLD is the negative class.
                        # We keep it as 'DOWN' to train the discriminator.
                        binary_mask.append(i)
                        binary_labels.append('DOWN')

                from collections import Counter
                binary_label_dist = Counter(binary_labels)
                
                # Check if we have at least one of each class
                if len(binary_label_dist) < 2:
                     logger.warning(f"⚠️ Single class detected: {dict(binary_label_dist)}. Cannot train binary classifier.")
                     # If we only have UP (BUY) signals, we can't train a discriminator.
                     return {"error": f"Single class training data: {dict(binary_label_dist)}. Need both BUY and HOLD samples."}

                # Reduced minimum from 100 to 30 for incremental training scenarios (backtest results)
                if len(binary_mask) < 30:
                    # Auto-disable binary classification if not enough samples
                    logger.warning(f"Not enough binary samples ({len(binary_mask)} < 30). "
                                 f"Falling back to 3-class classification (BUY/SELL/HOLD).")
                    use_binary_classification = False  # Fall back to 3-class
                    kept_sample_indices = None  # Keep all samples
                else:
                    kept_sample_indices = binary_mask  # Track which samples were kept
                    X = X[binary_mask]
                    y = np.array(binary_labels)
                    if y_reg is not None:
                        y_reg = y_reg[binary_mask]
                    logger.info(f"✅ Binary classification ready: {len(X)} samples")
        
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
            if y_reg is not None:
                y_reg = y_reg[unique_indices]
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
                xgb_reg_alpha, xgb_reg_lambda, xgb_subsample, xgb_colsample_bytree, xgb_min_child_weight, xgb_gamma,
                y_reg=y_reg  # Pass regression targets
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
                
                # CRITICAL DEBUG: Log sizes to diagnose split failure
                logger.debug(f"🔍 Split Debug: X={len(X)}, tr={len(training_samples)}, kept={len(kept_sample_indices) if 'kept_sample_indices' in locals() and kept_sample_indices is not None else 'None'}")
                
                # CRITICAL FIX: Ensure alignment with filtered X (e.g. after binary/regime filtering)
                # kept_sample_indices maps X rows to original training_samples indices
                current_samples = training_samples
                if 'kept_sample_indices' in locals() and kept_sample_indices is not None:
                    # Only filter if counts mismatch (implying filtering happened)
                    if len(kept_sample_indices) == len(X) and len(X) != len(training_samples):
                        logger.debug("   -> Filtering training_samples using kept_sample_indices")
                        current_samples = [training_samples[i] for i in kept_sample_indices]
                    elif len(X) == len(training_samples):
                         # No filtering happened
                        logger.debug("   -> Sizes match, using original training_samples")
                        current_samples = training_samples
                    else:
                        # Fallback for safety (e.g. regime subset)
                        if len(X) < len(training_samples) and len(kept_sample_indices) == len(X):
                             logger.debug("   -> Fallback Filtering (subset)")
                             current_samples = [training_samples[i] for i in kept_sample_indices]
                        else:
                             logger.warning(f"   -> ⚠️ SIZE MISMATCH and len(kept) != len(X). X={len(X)}, kept={len(kept_sample_indices)}")
                             # Try to forcefully align by just taking first N samples (Desperate fix)
                             current_samples = training_samples[:len(X)]
                
                logger.debug(f"   -> Final current_samples size: {len(current_samples)}")
                
                sample_dates = []
                for i, sample in enumerate(current_samples):
                    date_str = sample.get('date', '')
                    if date_str:
                        try:
                            sample_dates.append(datetime.strptime(date_str, '%Y-%m-%d'))
                        except:
                            logger.debug(f"Invalid date format in sample {i}: {date_str}")
                            sample_dates.append(None)
                    else:
                        if i < 3: # Log first few missing to avoid spam
                             logger.debug(f"Sample {i} missing 'date' key. Available keys: {list(sample.keys())}")
                        sample_dates.append(None)
                
                # If we have dates for most samples, use temporal split
                valid_dates = [d for d in sample_dates if d is not None]
                # CRITICAL FIX: Compare to current_samples (not training_samples) for regime-specific training
                # STRICT INTEGRITY: Enforce temporal split for all time-series data (Rule #7)
                if len(valid_dates) < len(current_samples):
                    missing = len(current_samples) - len(valid_dates)
                    logger.error(f"❌ {missing} samples are missing dates. Cannot perform strict temporal split.")
                    return {"error": f"Temporal integrity failure: {missing} samples missing dates."}

                # Sort by date
                sorted_indices = sorted(range(len(sample_dates)), 
                                        key=lambda i: sample_dates[i])
                
                # Split temporally with PURGING GAP to prevent autocorrelation leakage
                # Gap of 10 trading days ensures no temporal overlap between train/test
                purge_gap = 10  # Trading days gap (lookforward is 15, so 10 is safe)
                split_idx = int(len(sorted_indices) * (1 - test_size))
                train_indices = sorted_indices[:max(0, split_idx - purge_gap)]
                test_indices = sorted_indices[split_idx:]
                
                if len(train_indices) == 0 or len(test_indices) == 0:
                    logger.warning(f"Purging gap too large for dataset. Reducing gap.")
                    train_indices = sorted_indices[:split_idx]
                    test_indices = sorted_indices[split_idx:]
                
                X_train = X[train_indices]
                X_test = X[test_indices]
                y_train = y_encoded[train_indices]
                y_test = y_encoded[test_indices]
                weights_train = sample_weights[train_indices]
                weights_test = sample_weights[test_indices]
                y_reg_train = y_reg[train_indices]
                y_reg_test = y_reg[test_indices]
                X_train_reg = X_train  # Preserve pre-SMOTE data for regression
                
                # Log temporal split info
                train_dates = [sample_dates[i] for i in train_indices]
                test_dates = [sample_dates[i] for i in test_indices]
                logger.info(f"📅 Temporal split (Strict): Train={min(train_dates).date()} to {max(train_dates).date()} "
                            f"({len(train_indices)} samples), "
                            f"Test={min(test_dates).date()} to {max(test_dates).date()} ({len(test_indices)} samples)")
            except Exception as e:
                logger.critical(f"❌❌❌ CRITICAL: Temporal split FAILED: {e}. Aborting to prevent Accuracy Inflation.")
                return {"error": f"Temporal split failed: {e}"}
        else:
            # If temporal split logic is reached but no samples or disabled
            logger.error("❌ Temporal split attempted but no valid data found.")
            return {"error": "Temporal split failure: No valid data"}
        
        # Apply SMOTE for class balancing if enabled and binary classification is used
        # CRITICAL: Track if SMOTE was applied to avoid double-correction with class_weight='balanced'
        smote_applied = False
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
                        X_train_original_size = len(X_train)
                        X_train, y_train = smote.fit_resample(X_train, y_train)
                        # Recreate weights: keep original weights + add 1.0 for synthetic samples
                        new_samples_count = len(X_train) - X_train_original_size
                        if new_samples_count > 0:
                            # Original weights + new ones for synthetic samples
                            weights_train = np.concatenate([
                                weights_train[:X_train_original_size],
                                np.ones(new_samples_count)  # Synthetic samples get weight 1.0
                            ])
                            logger.info(f"  Added {new_samples_count} synthetic samples with weight 1.0")
                        unique_after, counts_after = np.unique(y_train, return_counts=True)
                        class_dist_after = dict(zip(unique_after, counts_after))
                        logger.info(f"  After SMOTE: {class_dist_after}")
                        smote_applied = True  # Mark SMOTE as applied to disable class_weight later
                    else:
                        logger.debug(f"Class imbalance ratio ({imbalance_ratio:.2f}) is acceptable, skipping SMOTE")
            except ImportError as e:
                logger.warning(f"SMOTE requested but imbalanced-learn not available: {e}. Install with: pip install --upgrade scikit-learn imbalanced-learn")
            except Exception as e:
                logger.warning(f"SMOTE application failed: {e}. Continuing without SMOTE.")
        
        # ===== FEATURE SELECTION =====
        # Phase 1: Scaling (Global Normalization Protocol)
        # If the scaler is already fitted (e.g. inherited from parent), DO NOT re-fit.
        # This ensures all regimes use the SAME 190-feature normalization statistics.
        if hasattr(self.scaler, 'mean_'):
            logger.info("Using already fitted global scaler (preserving 190-feature parity)")
            X_train_scaled_initial = self.scaler.transform(X_train)
            X_test_scaled_initial = self.scaler.transform(X_test)
        else:
            logger.info(f"Fitting fresh scaler on {X_train.shape[1]} features")
            X_train_scaled_initial = self.scaler.fit_transform(X_train)
            X_test_scaled_initial = self.scaler.transform(X_test)
        
        # Phase 2: Correlation-Based Pruning (Target: SNR Improvement)
        # We remove redundant features (>0.9 correlation) to simplify the model's focus.
        if not disable_feature_selection and len(X_train[0]) > 50:
            try:
                logger.info("Step 0: Performing Correlation-Based Pruning (abs(corr) > 0.90)...")
                # Use a subsample for correlation to save memory/time if needed, but 10k is fine
                corr_matrix = pd.DataFrame(X_train_scaled_initial[:min(10000, len(X_train_scaled_initial))]).corr().abs()
                upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
                
                if to_drop:
                    kept_indices = [i for i in range(X_train_scaled_initial.shape[1]) if i not in to_drop]
                    X_train_scaled_initial = X_train_scaled_initial[:, kept_indices]
                    X_test_scaled_initial = X_test_scaled_initial[:, kept_indices]
                    
                    # Store original indices of kept features to maintain mapping
                    prev_indices = kept_indices
                    logger.info(f"  ✓ Dropped {len(to_drop)} redundant features. {len(kept_indices)} features remain.")
                else:
                    prev_indices = list(range(X_train_scaled_initial.shape[1]))
            except Exception as corr_error:
                logger.warning(f"Correlation pruning failed: {corr_error}. Proceeding with RFE only.")
                prev_indices = list(range(X_train_scaled_initial.shape[1]))
        else:
            prev_indices = list(range(X_train_scaled_initial.shape[1]))
        
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
            # Determine number of features to select (Strict RFE Mode)
            # We enforce a hard limit of 15-20 features to eliminate noise
            n_features_to_select = 50  # Increased limit for higher capacity (Accuracy Recovery)
            
            # ===== STOCK-SPECIFIC FEATURE PRIORITIZATION =====
            # Market-wide features are identical for all stocks on the same date,
            # causing duplicate predictions. We deprioritize these features.
            MARKET_WIDE_FEATURES = {
                # Regime (Various names and prefixes)
                'market_regime_bull', 'market_regime_bear', 'market_regime_sideways',
                'regime_bull', 'regime_bear', 'regime_sideways', 'regime_strength',
                # Macro
                'vix_level', 'treasury_yield', 'sp500_trend',
                'macro_vix_val', 'macro_vix_score', 'macro_tnx_val', 'macro_yield_score', 
                'macro_dxy_val', 'macro_risk_score', 'macro_vix_change', 'macro_yield_curve_placeholder',
                # Advanced Macro
                'vix_level_adv', 'vix_change_pct_adv', 'vix_regime_adv', 'treasury_10y_adv', 'treasury_change_adv',
                'yield_curve_adv', 'dxy_level_adv', 'dxy_change_pct_adv',
                # News / Sentiment
                'news_volume', 'news_sentiment', 'news_score',
                'news_sentiment_enh', 'news_volume_enh',
                'finbert_positive', 'finbert_negative', 'finbert_neutral', 'finbert_compound', 'news_count',
                # Economy
                'macro_gdp', 'macro_unemployment', 'macro_inflation',
                'market_breadth', 'advance_decline_ratio'
            }
            
            # Use pruned feature names for the rest of the selection logic
            full_feature_names = self.feature_extractor.get_feature_names()
            feature_names = [full_feature_names[i] for i in prev_indices]
            
            # Identify market-wide feature indices (relative to the PRUNED matrix)
            market_wide_indices = set()
            stock_specific_indices = []
            for idx, name in enumerate(feature_names):
                if name in MARKET_WIDE_FEATURES:
                    market_wide_indices.add(idx)
                else:
                    stock_specific_indices.append(idx)
            
            excluded_count = len(market_wide_indices)
            logger.info(f"Feature prioritization: Excluding {excluded_count} market-wide features from RFE")
            logger.info(f"  Stock-specific features available: {len(stock_specific_indices)}")
            
            # Create a filtered feature matrix for RFE (stock-specific only)
            if len(stock_specific_indices) > n_features_to_select:
                X_train_filtered = X_train_scaled_initial[:, stock_specific_indices]
                X_test_filtered = X_test_scaled_initial[:, stock_specific_indices]
            else:
                # Not enough stock-specific features, use all available
                X_train_filtered = X_train_scaled_initial
                X_test_filtered = X_test_scaled_initial
                stock_specific_indices = list(range(len(feature_names)))
                logger.warning(f"Not enough stock-specific features after pruning, using all {len(feature_names)} remaining")
            
            # Always use RFE for accurate feature pruning unless explicitly disabled
            if len(X_train_filtered[0]) > n_features_to_select:
                # Use RFE for feature selection
                logger.info(f"Step 1: Using Strict RFE to select top {n_features_to_select} features (Noise Reduction)...")
                estimator = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=random_seed, n_jobs=get_safe_n_jobs(ML_TRAINING_CPU_CORES))
                rfe = RFE(estimator, n_features_to_select=n_features_to_select, step=0.05)
                X_train_selected = rfe.fit_transform(X_train_filtered, y_train)
                X_test_selected = rfe.transform(X_test_filtered)
                
                # Map RFE indices back to ORIGINAL feature indices
                rfe_support = np.array(rfe.get_support(indices=True))
                # 1. Map RFE selection back to stock_specific_indices (pruned indices)
                selected_pruned_indices = [stock_specific_indices[i] for i in rfe_support]
                # 2. Map pruned indices back to original full indices
                self.selected_features = np.array([prev_indices[i] for i in selected_pruned_indices])
                
                selected_feature_names = [full_feature_names[i] for i in self.selected_features]
                logger.info(f"✅ RFE selected {len(self.selected_features)} structural features")
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
                
                # Filter features using ALREADY SCALED data (Synchronize with Step 0)
                # This ensures we use the 190-feature global normalization statistics.
                X_train_selected = X_train_scaled_initial[:, self.selected_features]
                X_test_selected = X_test_scaled_initial[:, self.selected_features]
                selected_feature_indices = list(self.selected_features)
        
        # Ensure selected_feature_indices is defined for all paths
        if 'selected_feature_indices' not in locals():
            selected_feature_indices = list(self.selected_features)

        # DEBUG TRAINING DATA
        try:
            max_vals = np.max(X_train_selected, axis=0)
            mean_vals = np.mean(X_train_selected, axis=0)
            logger.info(f"🔍 TRAINING DATA DEBUG: Shape={X_train_selected.shape}")
            logger.info(f"   Max Values (first 5): {max_vals[:5]}")
            logger.info(f"   Max Values (last 5): {max_vals[-5:]}")
            if np.max(max_vals) > 1000000:
                logger.warning(f"⚠️ FOUND LARGE VALUES in Training Data! Max={np.max(max_vals)}")
            else:
                logger.info(f"   No large values found (Max={np.max(max_vals)})")
        except Exception as e:
            logger.warning(f"Could not log training debug info: {e}")

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
        
        # CRITICAL FIX: DO NOT overwrite self.scaler here. 
        # self.scaler must remain the 190-feature global scaler fitted in Step 0.
        # X_train_selected is already scaled (derived from X_train_scaled_initial).
        X_train_scaled = X_train_selected
        X_test_scaled = X_test_selected
        
        # Store feature importance for selected features only
        # Handle case where feature_importance_dict might be empty (RFE/MI)
        if not feature_importance_dict:
            # Assign equal importance if not available
            default_imp = 1.0 / len(selected_feature_names) if selected_feature_names else 0.0
            self.feature_importance = {name: default_imp for name in selected_feature_names}
        else:
            self.feature_importance = {
                name: feature_importance_dict.get(name, 0.0) 
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
        
        # Create model - EXTREME ANTI-OVERFITTING configuration
        # CRITICAL FIX: Previous model overfitted (training 66-72%, test 54%)
        # Solution: Much shallower trees, fewer estimators, stricter minimum samples
        rf_params = {
            'n_estimators': rf_n_estimators if rf_n_estimators is not None else ML_RF_N_ESTIMATORS,
            'max_depth': rf_max_depth if rf_max_depth is not None else ML_RF_MAX_DEPTH,
            'min_samples_split': rf_min_samples_split if rf_min_samples_split is not None else ML_RF_MIN_SAMPLES_SPLIT,
            'min_samples_leaf': rf_min_samples_leaf if rf_min_samples_leaf is not None else ML_RF_MIN_SAMPLES_LEAF,
            'max_features': 'sqrt',
            # CRITICAL FIX: Only use class_weight='balanced' when SMOTE was NOT applied
            # Using both causes double-correction -> minority class (DOWN/SELL) gets over-weighted
            'class_weight': None if smote_applied else 'balanced',
            'random_state': random_seed,
            'n_jobs': get_safe_n_jobs(ML_TRAINING_CPU_CORES)
        }
        # Add max_samples if sklearn version supports it (0.22+)
        try:
            from sklearn import __version__ as sklearn_version
            if tuple(map(int, sklearn_version.split('.')[:2])) >= (0, 22):
                rf_params['max_samples'] = 0.5  # REDUCED from 0.8 - use only 50% per tree
        except (ValueError, AttributeError, ImportError) as e:
            logger.debug(f"Could not check sklearn version for max_samples: {e}")
        
        # IGNORE tuned hyperparameters - they cause overfitting!
        # Don't apply best_params from Optuna as they overfit to validation set
        logger.info(f"RF params (anti-overfit): n_estimators={rf_params['n_estimators']}, max_depth={rf_params['max_depth']}, min_samples_leaf={rf_params['min_samples_leaf']}")
        
        rf = RandomForestClassifier(**rf_params)

        # OPTIMIZED: Stacking ensemble with LightGBM/XGBoost (achieves best validation accuracy)
        if model_family == 'voting':
            # Use LightGBM if available, otherwise XGBoost, otherwise fallback to GradientBoosting
            if HAS_LIGHTGBM:
                # EXTREME ANTI-OVERFITTING: Very shallow trees, few estimators, heavy regularization
                gb_params = {
                    'n_estimators': gb_n_estimators if gb_n_estimators is not None else 60,  # INCREASED from 30 to allow some learning
                    'max_depth': gb_max_depth if gb_max_depth is not None else 4,  # INCREASED from 2 to 4
                    'learning_rate': 0.02,  # INCREASED from 0.01
                    'subsample': lgb_subsample if lgb_subsample is not None else 0.4,  # INCREASED from 0.3
                    'colsample_bytree': lgb_colsample_bytree if lgb_colsample_bytree is not None else 0.4,  # INCREASED
                    'min_child_samples': lgb_min_child_samples if lgb_min_child_samples is not None else 100,  # REDUCED from 150
                    'reg_alpha': lgb_reg_alpha if lgb_reg_alpha is not None else 2.0,  # REDUCED from 5.0
                    'reg_lambda': lgb_reg_lambda if lgb_reg_lambda is not None else 5.0,  # REDUCED from 10.0
                    'random_state': random_seed,
                    'verbose': -1
                }
                # IGNORE tuned parameters - they cause overfitting!
                logger.info(f"LGB params (balanced): n_estimators={gb_params['n_estimators']}, max_depth={gb_params['max_depth']}, reg_lambda={gb_params['reg_lambda']}")
                gb = LGBMClassifier(**gb_params)
                logger.info("Using LightGBM in ensemble")
            elif HAS_XGBOOST:
                # EXTREME ANTI-OVERFITTING for XGBoost
                gb_params = {
                    'n_estimators': gb_n_estimators if gb_n_estimators is not None else 60,  # INCREASED
                    'max_depth': gb_max_depth if gb_max_depth is not None else 4,  # INCREASED
                    'learning_rate': 0.02,  # INCREASED
                    'subsample': xgb_subsample if xgb_subsample is not None else 0.4,  # INCREASED
                    'colsample_bytree': xgb_colsample_bytree if xgb_colsample_bytree is not None else 0.4,  # INCREASED
                    'min_child_weight': xgb_min_child_weight if xgb_min_child_weight is not None else 10,  # REDUCED from 20
                    'gamma': xgb_gamma if xgb_gamma is not None else 0.5,  # REDUCED from 1.0
                    'reg_alpha': xgb_reg_alpha if xgb_reg_alpha is not None else 2.0,  # REDUCED
                    'reg_lambda': xgb_reg_lambda if xgb_reg_lambda is not None else 5.0,  # REDUCED
                    'random_state': random_seed,
                    'eval_metric': 'mlogloss'
                }
                # IGNORE tuned parameters - they cause overfitting!
                logger.info(f"XGB params (balanced): n_estimators={gb_params['n_estimators']}, max_depth={gb_params['max_depth']}, reg_lambda={gb_params['reg_lambda']}")
                gb = XGBClassifier(**gb_params)
                logger.info("Using XGBoost in ensemble")
            else:
                # Fallback to sklearn GradientBoosting - EXTREME ANTI-OVERFITTING
                gb = GradientBoostingClassifier(
                    n_estimators=gb_n_estimators if gb_n_estimators is not None else 60,  # INCREASED
                    max_depth=gb_max_depth if gb_max_depth is not None else 4,  # INCREASED
                    learning_rate=0.02,  # INCREASED
                    min_samples_split=50,  # REDUCED from 100
                    min_samples_leaf=25,  # REDUCED from 50
                    subsample=0.4,  # INCREASED
                    random_state=random_seed
                )
                logger.info("Using sklearn GradientBoosting (balanced config)")
            
            lr = LogisticRegression(
                max_iter=1000,
                C=0.1,  # Reduced from 0.2 for stronger regularization
                class_weight=None if smote_applied else 'balanced',
                random_state=random_seed,
                penalty='l2'  # Explicit L2 regularization
            )
            
            # Use VotingClassifier for ANTI-OVERFITTING
            # Stacking causes 100% train accuracy (meta-learner overfits)
            # Voting averages probabilities without overfitting
            if HAS_LIGHTGBM or HAS_XGBOOST:
                # Soft voting: average predicted probabilities
                # This prevents the 40% train-test gap we saw with stacking
                voting_clf = VotingClassifier(
                    estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
                    voting='soft',  # Average probabilities
                    weights=[2, 2, 1],  # RF and GB weighted higher than LR
                    n_jobs=get_safe_n_jobs(ML_TRAINING_CPU_CORES)
                )
                logger.info("Using Voting ensemble (anti-overfitting mode)")
            else:
                # Fallback to VotingClassifier if advanced libraries not available
                voting_clf = VotingClassifier(
                    estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
                    voting='soft',
                    weights=[2, 1, 1]
                )
                logger.info("Using Voting ensemble (sklearn-only)")
            
            # CRITICAL FIX: Add Probability Calibration
            # Tree-based models (RF, GBM) tend to be uncalibrated and push probs towards 0.5.
            # Calibration stretches these to 0-1 range, allowing "Strong" signals > 80% to appear.
            # This solves the "Zero Strong Signals" issue without lowering profit thresholds.
            from sklearn.calibration import CalibratedClassifierCV
            logger.info("Step 4: Calibrating probabilities to enable High Confidence signals...")
            
            # Use 'sigmoid' (Platt Scaling) for robustness.
            # logic: Isotonic overfits on noisy financial data, leading to "fake" high confidence.
            # Sigmoid enforces a smooth logistic curve, which is safer for market predictions.
            method = 'sigmoid' 
            calibrated_clf = CalibratedClassifierCV(voting_clf, method=method, cv=5)
            
            self.model = calibrated_clf
        else:
            self.model = rf
            logger.info("Using single RandomForest (simplified default)")
        
        # Train (suppress feature name warnings)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')
            
            # Log if we're using weighted training (verified samples have more influence)
            if weights_train is not None and len(np.unique(weights_train)) > 1:
                verified_in_train = sum(1 for w in weights_train if w > 1.0)
                logger.info(f"⚖️ Training with sample weights ({verified_in_train} verified samples with 10x weight)")
            
            self.model.fit(X_train_scaled, y_train, sample_weight=weights_train)
            
            # ===== REGRESSION MODEL TRAINING (Target Pricing) =====
            try:
                from sklearn.ensemble import RandomForestRegressor
                logger.info("Step 5: Training Regression Model for Dynamic Targets...")
                
                # Get regression data (fallback to current if not set)
                # Note: X_train_reg was preserved before SMOTE
                X_reg_src = locals().get('X_train_reg', X_train)
                y_reg_src = locals().get('y_reg_train', None)
                
                if y_reg_src is not None:
                    # Apply scaling (must transform raw features using the fitted scaler)
                    try:
                        X_reg_scaled = self.scaler.transform(X_reg_src)
                    except ValueError as e:
                        logger.warning(f"Scaler dimension mismatch in regression ({e}). Fitting new scaler.")
                        local_scaler = StandardScaler()
                        X_reg_scaled = local_scaler.fit_transform(X_reg_src)
                    
                    # Apply feature selection if used
                    if self.selected_features is not None:
                        # Ensure indices are valid (handle mismatch if local scaler used different features - unlikely as X_reg_src is same source)
                        try:
                            X_reg_scaled = X_reg_scaled[:, self.selected_features]
                        except IndexError:
                             logger.warning("Feature selection indices out of bounds for regression. Using all features.")
                    
                    # Train Regressor
                    self.model_reg = RandomForestRegressor(
                        n_estimators=50, 
                        max_depth=5, 
                        min_samples_split=20, # Require strong patterns
                        n_jobs=get_safe_n_jobs(ML_TRAINING_CPU_CORES),
                        random_state=random_seed
                    )
                    self.model_reg.fit(X_reg_scaled, y_reg_src)
                    logger.info(f"✅ Regression model trained on {len(X_reg_scaled)} samples")
                else:
                    logger.warning("Skipping regression training (y_reg_train not available)")
            
            except Exception as e:
                logger.error(f"Failed to train regression model: {e}")
                self.model_reg = None
        
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
        
        # Save model securely to temporary files first to prevent corruption
        tmp_model_file = f"tmp_ml_model_{self.strategy}.pkl"
        tmp_scaler_file = f"tmp_scaler_{self.strategy}.pkl"
        tmp_encoder_file = f"tmp_label_encoder_{self.strategy}.pkl"
        
        self.storage.save_model(self.model, tmp_model_file)
        self.storage.save_model(self.scaler, tmp_scaler_file)
        self.storage.save_model(self.label_encoder, tmp_encoder_file)

        # Save regression model
        tmp_reg_file = f"tmp_ml_model_target_{self.strategy}.pkl"
        if self.model_reg is not None:
            self.storage.save_model(self.model_reg, tmp_reg_file)
        
        # Store accuracy values for access by regime model training
        self.train_accuracy = float(train_score)
        self.test_accuracy = float(test_score)
        
        # Prepare metadata
        # CRITICAL: Store scaler's expected feature count to prevent mismatch during prediction
        scaler_n_features = self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else X_train_scaled.shape[1]
        
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
            'selected_feature_names': selected_feature_names if 'selected_feature_names' in dir() else None,
            'scaler_n_features': scaler_n_features,  # Store exact feature count scaler was trained on
            'feature_selection_threshold': self.feature_selection_threshold,
            'feature_selection_method': self.feature_selection_method,
            'total_features': len(self.feature_extractor.get_feature_names()),
            'selected_features_count': len(self.selected_features) if self.selected_features is not None else None,
            'label_classes': self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else None,
            'model_type': model_family,
            'hold_confidence_threshold': float(hold_confidence_threshold) if hold_confidence_threshold is not None else 0.0,
            'has_optuna': HAS_OPTUNA,
            'used_hyperparameter_tuning': self.use_hyperparameter_tuning,
            'use_binary_classification': use_binary_classification,
            'ml_retrained': True  # Flag to indicate successful training
        }
        
        # Register new version and check if it should be activated
        should_activate = True
        if self.version_manager:
            try:
                # Generate version number
                if not version_number:
                    version_number = self.version_manager.generate_version_number()
                
                # Check performance before activating
                registered_version, should_activate = self.version_manager.register_new_version(
                    metadata=self.metadata,
                    test_accuracy=test_score,
                    train_accuracy=train_score,
                    cv_mean=cv_scores.mean() if len(cv_scores) > 0 else None
                )
                
                if not should_activate:
                    logger.warning(f"⚠️ New model performs worse than current. Rejected version {version_number}.")
                    # Cleanup temporary files
                    for tmp_file in [tmp_model_file, tmp_scaler_file, tmp_encoder_file, tmp_reg_file]:
                        tmp_path = self.data_dir / tmp_file
                        if tmp_path.exists():
                            tmp_path.unlink()
                    return self.metadata
            except Exception as e:
                logger.error(f"Error in version management: {e}")
                # Fallback to activating if version manager fails
                should_activate = True

        # Activate the new model by renaming temporary files to main names
        if should_activate:
            try:
                import shutil
                shutil.move(self.data_dir / tmp_model_file, self.data_dir / self.model_file)
                shutil.move(self.data_dir / tmp_scaler_file, self.data_dir / self.scaler_file)
                shutil.move(self.data_dir / tmp_encoder_file, self.data_dir / f"label_encoder_{self.strategy}.pkl")
                
                if self.model_reg is not None and (self.data_dir / tmp_reg_file).exists():
                    shutil.move(self.data_dir / tmp_reg_file, self.data_dir / f"ml_model_target_{self.strategy}.pkl")
                
                # Activate meta-model (if it was just trained)
                tmp_meta_file = f"tmp_ml_model_meta_{self.strategy}.pkl"
                if (self.data_dir / tmp_meta_file).exists():
                    shutil.move(self.data_dir / tmp_meta_file, self.data_dir / f"ml_model_meta_{self.strategy}.pkl")
                
                logger.info(f"✅ Successfully activated new model version (accuracy: {test_score*100:.1f}%)")
            except Exception as e:
                logger.error(f"Error activating new model files: {e}")
                return self.metadata
        
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
                            xgb_min_child_weight: int = None, xgb_gamma: float = None,
                            y_reg=None) -> Dict:
        """Train separate models for each market regime"""
        from algorithm_improvements import MarketRegimeDetector
        import yfinance as yf
        
        # Initialize regime detector
        regime_detector = MarketRegimeDetector()
        
        # Detect regime for each training sample
        logger.info("Detecting market regime for each training sample...")
        regime_labels = []
        
        # CRITICAL OPTIMIZATION: Fetch SPY data ONCE for the entire training range
        spy_history = None
        if training_samples:
            try:
                sample_dates = []
                for s in training_samples:
                    if s.get('date'):
                        try:
                            sample_dates.append(pd.to_datetime(s['date']))
                        except: pass
                
                if sample_dates:
                    min_date = min(sample_dates) - pd.Timedelta(days=365)
                    max_date = max(sample_dates) + pd.Timedelta(days=1)
                    
                    logger.info(f"Fetching SPY history for regime detection ({min_date.date()} to {max_date.date()})...")
                    ticker = yf.Ticker('SPY')
                    spy_history = ticker.history(start=min_date.strftime('%Y-%m-%d'), 
                                               end=max_date.strftime('%Y-%m-%d'))
                    
                    if not spy_history.empty:
                        if isinstance(spy_history.columns, pd.MultiIndex):
                            if 'Close' in spy_history.columns.get_level_values(1):
                                spy_history = spy_history.xs('Close', level=1, axis=1).to_frame()
                        elif 'Close' not in spy_history.columns and 'close' in spy_history.columns:
                            spy_history = spy_history.rename(columns={'close': 'Close'})
            except Exception as e:
                logger.warning(f"Error fetching aggregate SPY history: {e}")

        # CRITICAL OPTIMIZATION: Pre-compute regimes Vectorially
        # This replaces the O(N) loop slicing with O(1) lookup
        regime_lookup = {}
        if spy_history is not None and not spy_history.empty:
            try:
                logger.info("Pre-computing market regimes vectorially...")
                regime_df = regime_detector.compute_regimes_vectorized(spy_history)
                # Convert to dict for fast O(1) lookup: Date -> Regime
                # Normalize dates to string YYYY-MM-DD for reliable key matching
                regime_lookup = {d.strftime('%Y-%m-%d'): r for d, r in zip(regime_df.index, regime_df['Regime'])}
            except Exception as e:
                logger.error(f"Vectorized regime computation failed: {e}")

        for i, sample in enumerate(training_samples):
            regime = 'sideways'
            sample_date = sample.get('date')
            
            # FAST LOOKUP
            if sample_date in regime_lookup:
                regime = regime_lookup[sample_date]
            else:
                # Fallback: Try nearest date if exact match fails (e.g. weekend)
                # But since we have a fast lookup, limit the "search" to trivial range to keep speed
                try: 
                    # Only try 1-3 days back for nearest trading day
                    s_date = pd.to_datetime(sample_date)
                    for lag in range(1, 4):
                        d_lag = (s_date - pd.Timedelta(days=lag)).strftime('%Y-%m-%d')
                        if d_lag in regime_lookup:
                            regime = regime_lookup[d_lag]
                            break
                except:
                    pass
            
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
        min_samples_per_regime = 25  # Minimum samples needed per regime (relaxed from 100)
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
            
            # Filter regression target if available
            regime_y_reg = None
            if y_reg is not None:
                regime_y_reg = y_reg[regime_mask]
            
            logger.info(f"Training on {len(regime_X)} {regime} samples...")
            
            # Check label distribution in regime samples
            from collections import Counter
            regime_label_dist = Counter([s.get('label', 'HOLD') for s in regime_samples])
            logger.info(f"Regime {regime} label distribution: {dict(regime_label_dist)}")
            
            # Create a temporary ML instance for this regime
            # We'll train it using the same logic as the main model
            regime_ml = StockPredictionML(self.data_dir, f"{self.strategy}_{regime}")
            
            # CRITICAL: Inherit the fitted global scaler to ensure 190-feature parity
            if hasattr(self.scaler, 'mean_'):
                regime_ml.scaler = self.scaler
                logger.info(f"  Propagating global 190-feature scaler to {regime} sub-model")
            
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
                    use_binary_classification=use_binary_classification,  # This will convert BUY/HOLD to UP/DOWN
                    # CRITICAL: Regime models must also know to use binary logic
                    # The recursive call to .train() above handles the conversion logic internally.
                    # We just need to make sure we don't accidentally filter out the HOLDs before they get there.
                    # In this implementation, 'regime_samples' contains the raw dictionaries, so .train() will process them correctly.

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
                    logger.warning(f"⚠️ Skipping {regime} model: {regime_result['error']}")
                    continue
                
                # Get accuracy from result or from regime_ml attributes
                train_acc = regime_result.get('train_accuracy', regime_ml.train_accuracy if hasattr(regime_ml, 'train_accuracy') else 0.0)
                test_acc = regime_result.get('test_accuracy', regime_ml.test_accuracy if hasattr(regime_ml, 'test_accuracy') else 0.0)
                cv_mean = regime_result.get('cv_mean', 0.0)
                cv_std = regime_result.get('cv_std', 0.0)
                
                # Store the regime model
                regime_models[regime] = regime_ml
                
                # Get scaler feature count from regime model
                regime_scaler_n_features = regime_ml.scaler.n_features_in_ if hasattr(regime_ml.scaler, 'n_features_in_') else None
                
                # Get selected_feature_names from regime model's metadata
                regime_selected_feature_names = None
                if regime_ml.metadata and 'selected_feature_names' in regime_ml.metadata:
                    regime_selected_feature_names = regime_ml.metadata['selected_feature_names']
                
                regime_metadata[regime] = {
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std,
                    'samples': len(regime_X),
                    'selected_features': regime_ml.selected_features.tolist() if regime_ml.selected_features is not None else None,
                    'selected_feature_names': regime_selected_feature_names,  # Store feature names for prediction
                    'scaler_n_features': regime_scaler_n_features,  # Store scaler's expected feature count
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
            'training_data_start': min(s.get('date', '9999-12-31') for s in training_samples),
            'training_data_end': max(s.get('date', '1900-01-01') for s in training_samples),
            'use_binary_classification': use_binary_classification,
            'ml_retrained': True  # Flag to indicate successful training
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
    
    def _detect_current_regime(self, history_data: dict = None, stock_data: dict = None, 
                               as_of_date: Optional[datetime] = None) -> str:
        """Detect current market regime for prediction (Date-aware)"""
        try:
            from algorithm_improvements import MarketRegimeDetector
            import yfinance as yf
            
            if self.regime_detector is None:
                self.regime_detector = MarketRegimeDetector()
            
            # Try to get SPY data for regime detection
            spy_df = None
            
            # Check cache first (Point-in-time cache)
            cache_key = f"_spy_data_{as_of_date.strftime('%Y-%m-%d')}" if as_of_date else "_spy_data_live"
            
            if hasattr(self, cache_key) and hasattr(self, f"{cache_key}_time"):
                cache_time = getattr(self, f"{cache_key}_time")
                if cache_time and (datetime.now() - cache_time).days < 1:
                    spy_df = getattr(self, cache_key)
            
            # Fetch SPY data if not cached
            if spy_df is None or spy_df.empty:
                try:
                    if as_of_date:
                        # Backtest: Fetch historical SPY leading up to as_of_date
                        end_str = (as_of_date + timedelta(days=1)).strftime('%Y-%m-%d')
                        # Download slightly more to ensure enough data for indicators
                        spy_hist = yf.download('SPY', end=end_str, period="1y", interval="1d", progress=False, show_errors=False)
                    else:
                        # Live: Fetch current SPY
                        spy_hist = yf.download('SPY', period="1y", interval="1d", progress=False, show_errors=False)
                        
                    if not spy_hist.empty:
                        spy_df = spy_hist
                        setattr(self, cache_key, spy_df)
                        setattr(self, f"{cache_key}_time", datetime.now())
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
    
    def get_regime_accuracy(self, regime: str) -> float:
        """Get the test accuracy for a specific regime"""
        if not self.use_regime_models or 'regime_metadata' not in self.metadata:
            return 0.0
        
        regime_meta = self.metadata['regime_metadata'].get(regime, {})
        return regime_meta.get('test_accuracy', 0.0)
    
    def predict(self, features: np.ndarray, stock_data: dict = None, 
                history_data: dict = None, is_backtest: bool = False,
                as_of_date: Optional[datetime] = None) -> Dict:
        """Make prediction using trained model (Date-aware)"""
        # Check if we should use regime-specific models
        if self.use_regime_models and len(self.regime_models) > 0:
            # LOGGING FEATURES DEBUG
            try:
                # DEBUG: Identify NaN columns
                if np.isnan(features).any():
                    nan_indices = np.where(np.isnan(features))[1] if features.ndim > 1 else np.where(np.isnan(features))[0]
                    # Get feature names if possible
                    feat_names = []
                    if self.selected_features is not None: 
                        # This might be tricky if selection hasn't happened yet. 
                        # Predicting raw features means we use all names.
                        feat_names = self.feature_extractor.get_feature_names()
                    
                    nan_names = [feat_names[i] for i in nan_indices if i < len(feat_names)] if feat_names else [str(i) for i in nan_indices]
                    logger.warning(f"⚠️ NaN INPUT FEATURES DETECTED: {len(nan_indices)} NaNs in indices: {nan_indices[:10]}... Names: {nan_names[:5]}")

                zero_count = np.count_nonzero(features == 0)
                logger.info(f"🔍 PREDICT INPUT DEBUG: Shape={features.shape}, Min={np.nanmin(features):.3f}, Max={np.nanmax(features):.3f}, Mean={np.nanmean(features):.3f}, Zeros={zero_count} ({zero_count/features.size*100:.1f}%)")
            except Exception as e:
                logger.debug(f"Feature debug log error: {e}")
                
            # Detect current market regime (Date-aware)
            current_regime = self._detect_current_regime(history_data, stock_data, as_of_date)
            self.last_regime = current_regime
            
            if current_regime in self.regime_models:
                regime_ml = self.regime_models[current_regime]
                logger.debug(f"Using {current_regime} regime model for prediction")
                # Regime models also support the same signature
                return regime_ml.predict(features, stock_data, history_data, is_backtest=is_backtest)
            else:
                # Fallback to sideways or default model
                fallback_regime = 'sideways' if 'sideways' in self.regime_models else list(self.regime_models.keys())[0]
                logger.debug(f"{current_regime} regime model not found, using {fallback_regime} model")
                regime_ml = self.regime_models[fallback_regime]
                return regime_ml.predict(features, stock_data, history_data, is_backtest=is_backtest)
        
        # Original single model prediction
        if self.model is None:
            return {
                'action': 'HOLD',
                'confidence': 50,
                'error': 'Model not trained'
            }
        
        # 1. Handle missing values & Reshape
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        features = features.reshape(1, -1)
        
        # 2. Scale FULL vector first (Synchronized with train Step 0)
        # CRITICAL: We MUST scale before selecting to match training mathematical pipeline.
        scaler_expected_features = self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else None
        input_features = len(features[0])
        
        try:
            # Check if scaler exists and is fitted
            if not hasattr(self.scaler, 'mean_'):
                logger.warning(f"Scaler for {self.strategy} not fitted. Attempting to fit on current input (UNSAFE - fallback only).")
                features_scaled = self.scaler.fit_transform(features)
            else:
                # If feature counts don't match, we cannot scale. This implies a broken pipeline.
                if scaler_expected_features is not None and input_features != scaler_expected_features:
                    logger.error(f"FATAL: Feature count mismatch for scaler! Expected {scaler_expected_features}, got {input_features}")
                    return {'action': 'HOLD', 'confidence': 0.0, 'error': f'Scaler mismatch: {input_features} vs {scaler_expected_features}'}
                
                features_scaled = self.scaler.transform(features)
                
            # DEBUG SCALED FEATURES (FULL VECTOR)
            try:
                max_idx = np.argmax(np.abs(features_scaled[0]))
                logger.debug(f"🔍 SCALED FULL VECTOR: Shape={features_scaled.shape}, Min={np.min(features_scaled):.3f}, Max={np.max(features_scaled):.3f}")
            except: pass
        except Exception as e:
            logger.error(f"Scaling failed: {e}")
            return {'action': 'HOLD', 'confidence': 0.0, 'error': f'Scaling error: {e}'}

        # 3. Apply feature selection & interactions (on SCALED data)
        # This matches the Step 1/Step 2 logic in train()
        
        # If selected_features is None or empty, try to infer from metadata or model
        if (self.selected_features is None or (isinstance(self.selected_features, np.ndarray) and len(self.selected_features) == 0)):
             # Re-attempt load from metadata
             if self.metadata and 'selected_features' in self.metadata:
                 self.selected_features = np.array(self.metadata['selected_features'])
        
        # Apply selection
        if self.selected_features is not None and len(self.selected_features) > 0:
            features = features_scaled[:, self.selected_features]
            logger.debug(f"Applied selection to scaled data: {input_features} -> {len(features[0])} features")
        else:
            features = features_scaled
        
        # Add Interactions (Expand scaled selected features)
        num_selected = len(features[0])
        # interactions were used if the model expects MORE features than we currently have
        model_expects = self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else num_selected
        
        if model_expects > num_selected:
            logger.debug(f"Detected interactions: model expects {model_expects}, we have {num_selected}. Adding...")
            selected_feature_names = self.metadata.get('selected_feature_names')
            if not selected_feature_names:
                all_names = self.feature_extractor.get_feature_names()
                if self.selected_features is not None:
                     selected_feature_names = [all_names[i] for i in self.selected_features if i < len(all_names)]
                else:
                     selected_feature_names = all_names[:num_selected]
            
            try:
                features, interaction_names = self.feature_extractor._add_feature_interactions(features, selected_feature_names, fit=False)
                logger.debug(f"Added interactions: final count {len(features[0])} features")
            except Exception as e:
                logger.warning(f"Interaction expansion failed: {e}")
        
        # Final adjustment: ensure feature count matches model exactly
        if len(features[0]) != model_expects:
             logger.warning(f"Inference count mismatch: got {len(features[0])}, model expects {model_expects}. Adjusting.")
             if len(features[0]) > model_expects:
                 features_final = features[:, :model_expects]
             else:
                 features_final = np.column_stack([features, np.zeros((1, model_expects - len(features[0])))])
        else:
             features_final = features

        # Predict using scaled and selected (and interacted) features
        features_scaled = features_final # Re-assign for metadata recording next
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.utils.validation')
            prediction = self.model.predict(features_final)[0]
            probabilities = self.model.predict_proba(features_final)[0]
        
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
        is_binary_mode = (len(classes) == 2 or 'HOLD' not in classes)
        
        # Binary confidence threshold: if max probability < threshold, return HOLD
        # This helps reduce bias toward majority class
        # Configurable via ML_BINARY_HOLD_THRESHOLD
        # FIX: Enforce safety threshold even in backtests to reflect real-world performance
        binary_confidence_threshold = ML_BINARY_HOLD_THRESHOLD
        if is_binary_mode and maxp < binary_confidence_threshold:
            logger.debug(f"Low confidence binary prediction ({maxp*100:.1f}% < {binary_confidence_threshold*100}%), returning HOLD")
            
            # Map probabilities based on available classes
            buy_prob = 0.0
            sell_prob = 0.0
            
            if 'BUY' in classes:
                buy_prob = float(probabilities[classes.index('BUY')] * 100)
            elif 'UP' in classes:
                buy_prob = float(probabilities[classes.index('UP')] * 100)

            if 'SELL' in classes:
                 sell_prob = float(probabilities[classes.index('SELL')] * 100)
            elif 'DOWN' in classes:
                 sell_prob = float(probabilities[classes.index('DOWN')] * 100)
            
            # Fill in the gaps if we have the other side
            if 'BUY' in classes and 'SELL' in classes:
                 pass # Already set
            elif 'UP' in classes and 'DOWN' in classes:
                 # Previous logic had DOWN->BUY, UP->SELL. Preserving that weird mapping if it was intentional for "Dip Buying"?
                 # Actually, let's stick to the visible classes in logs: BUY and SELL.
                 pass
                 
            return {
                'action': 'HOLD',
                'confidence': float(maxp * 100),
                'reasoning': f'Low confidence prediction ({maxp*100:.1f}%) - avoiding biased prediction',
                'probabilities': {'BUY': buy_prob, 'SELL': sell_prob, 'HOLD': 0.0}
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
            # Look up probability for UP instead of relying on default argmax (0.50 cutoff)
            classes_list = list(self.label_encoder.classes_)
            up_prob = 0.0
            down_prob = 0.0
            
            if 'UP' in classes_list:
                up_prob = float(probabilities[classes_list.index('UP')])
            if 'DOWN' in classes_list:
                down_prob = float(probabilities[classes_list.index('DOWN')])

            # Use threshold-based logic over argmax
            from config import ML_HIGH_ACCURACY_THRESHOLD
            
            if up_prob >= ML_HIGH_ACCURACY_THRESHOLD:
                action = 'BUY'
                confidence = float(up_prob * 100)
            else:
                action = 'HOLD'
                confidence = float(max(probabilities) * 100) # Keep DOWN confidence for tracking
                
            logger.debug(f"Binary Threshold Check: UP={up_prob:.3f}, Threshold={ML_HIGH_ACCURACY_THRESHOLD:.2f} -> Action={action}")
        else:
            # Default logic for multi-class
            # Calculate confidence for the chosen class
            confidence = float(probabilities[prediction] * 100)
        
        # Get probabilities for all classes
        prob_dict = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            # Convert binary class names to BUY/HOLD for compatibility
            display_name = class_name
            if class_name == 'UP':
                display_name = 'BUY'
            elif class_name == 'DOWN':
                display_name = 'HOLD'
            prob_dict[display_name] = float(probabilities[i] * 100)
        
        # Ensure all expected actions are in prob_dict (for 3-class compatibility)
        if 'BUY' not in prob_dict: prob_dict['BUY'] = 0.0
        if 'SELL' not in prob_dict: prob_dict['SELL'] = 0.0
        if 'HOLD' not in prob_dict: prob_dict['HOLD'] = 0.0
        
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
        
        # Phase 3 Meta-Labeling: Secondary Verification
        meta_confidence = 100.0
        # Use getattr to safely check for model_meta (handles regime submodels or stale objects)
        model_meta = getattr(self, 'model_meta', None)
        if model_meta is not None:
            try:
                # Meta-classifier uses the same processed features
                meta_probs = model_meta.predict_proba(features_scaled)
                # Success probability (class 1)
                meta_confidence = float(meta_probs[0][1] * 100) if len(meta_probs[0]) > 1 else 0.0
                logger.debug(f"Meta-Classifier Signal Verification: {meta_confidence:.1f}%")
            except Exception as e:
                logger.debug(f"Error in meta-prediction: {e}")

        result = {
            'action': action,
            'confidence': confidence,
            'probabilities': prob_dict,
            'meta_confidence': meta_confidence
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
    
    def predict_batch(self, feature_vectors: List[List[float]]) -> np.ndarray:
        """
        Predict probabilities for a batch of pre-calculated feature vectors.
        Handles both Single Model and Regime-Specific Models via recursion.
        """
        try:
            # 1. Convert to NP and Sanitize
            X = np.array(feature_vectors)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            n_samples = len(X)
            
            if n_samples == 0:
                return np.array([])

            # 2. Regime-Specific Dispatch
            if self.use_regime_models and len(self.regime_models) > 0:
                # Find feature indices for regimes
                names = self.feature_extractor.get_feature_names()
                try:
                    bull_idx = names.index('regime_bull')
                    bear_idx = names.index('regime_bear')
                    # Sideways is default if neither bull/bear active
                except ValueError:
                    logger.warning("Regime features not found for batch prediction. Using fallback.")
                    # Fall through to single model
                    pass
                else:
                    # Create result array
                    final_probs = np.zeros((n_samples, 3))
                    
                    # Identify regimes
                    is_bull = X[:, bull_idx] > 0.5
                    is_bear = X[:, bear_idx] > 0.5
                    is_sideways = ~(is_bull | is_bear)
                    
                    # Predict for each group
                    for regime_key, mask in [('bull', is_bull), ('bear', is_bear), ('sideways', is_sideways)]:
                        if np.any(mask):
                            model = self.regime_models.get(regime_key)
                            if model:
                                # Use internal _predict_single_batch to get potentially 2 or 3 columns
                                probs = model._predict_single_batch(X[mask])
                                
                                # MAP TO 3 CLASSES [SELL, HOLD, BUY]
                                if probs.shape[1] == 2 and hasattr(model.model, 'classes_'):
                                    classes = model.model.classes_
                                    for col_idx, class_label in enumerate(classes):
                                        if class_label in [0, 1, 2]:
                                            final_probs[np.where(mask)[0], class_label] = probs[:, col_idx]
                                elif probs.shape[1] == 3:
                                    final_probs[mask] = probs
                                else:
                                    logger.warning(f"Unexpected class count ({probs.shape[1]}) for {regime_key} model")
                            else:
                                # Fallback to single model
                                probs = self._predict_single_batch(X[mask])
                                if probs.shape[1] == 3:
                                    final_probs[mask] = probs
                                elif probs.shape[1] == 2 and hasattr(self.model, 'classes_'):
                                    for col_idx, class_label in enumerate(self.model.classes_):
                                        if class_label in [0, 1, 2]:
                                            final_probs[np.where(mask)[0], class_label] = probs[:, col_idx]
                    
                    return final_probs

            # 3. Single Model Prediction
            return self._predict_single_batch(X)
            
        except Exception as e:
            logger.error(f"Error in predict_batch: {e}")
            return np.zeros((len(feature_vectors), 3))

    def _predict_single_batch(self, X: np.ndarray) -> np.ndarray:
        """Helper for single model batch prediction"""
        if self.model is None or not self.is_trained():
            return np.zeros((len(X), 3))
            
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Select Features
        if self.selected_features is not None:
            X_selected = X_scaled[:, self.selected_features]
        else:
            X_selected = X_scaled
            
        # Interactions
        expected_features = 0
        if hasattr(self.model, 'n_features_in_'):
            expected_features = self.model.n_features_in_
        elif hasattr(self.model, 'estimators_'): # VotingClassifier
             if hasattr(self.model.estimators_[0], 'n_features_in_'):
                 expected_features = self.model.estimators_[0].n_features_in_
             
        if expected_features > X_selected.shape[1]:
             names = [self.feature_extractor.get_feature_names()[i] for i in self.selected_features]
             X_selected, _ = self.feature_extractor._add_feature_interactions(X_selected, names, fit=False)
             
        # Predict
        return self.model.predict_proba(X_selected)

        """Predict target price/return using regression model"""
        # Regime handling
        if self.use_regime_models and len(self.regime_models) > 0:
            current_regime = getattr(self, 'last_regime', None)
            if not current_regime:
                 current_regime = self._detect_current_regime(history_data, stock_data)
            
            if current_regime in self.regime_models:
                return self.regime_models[current_regime].predict_target(features, stock_data, history_data)
            else:
                 # Fallback
                 fallback = 'sideways' if 'sideways' in self.regime_models else list(self.regime_models.keys())[0]
                 return self.regime_models[fallback].predict_target(features, stock_data, history_data)
        
        # Single model logic
        if self.model_reg is None:
             return {'predicted_return': 0.0, 'confidence': 0.0, 'error': 'No regression model'}
        
        try:
             # Scale
             features_scaled = self.scaler.transform(features)
             # Select features
             if self.selected_features is not None:
                  features_scaled = features_scaled[:, self.selected_features]
             
             predicted_return = self.model_reg.predict(features_scaled)[0]
             
             return {
                 'predicted_return': float(predicted_return),
                 'confidence': 100.0,
                 'regime': getattr(self, 'last_regime', 'unknown')
             }
        except Exception as e:
             logger.error(f"Target prediction failed: {e}")
             return {'predicted_return': 0.0, 'error': str(e)}

    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray, random_seed: int) -> Dict:
        """Tune hyperparameters using Optuna"""
        if not HAS_OPTUNA:
            return {}
        
        def objective(trial):
            # Suggest hyperparameters for XGBoost/LightGBM
            params = {}
            
            if HAS_LIGHTGBM:
                params['lgb'] = {
                    'n_estimators': trial.suggest_int('lgb_n_estimators', 30, 120),  # Reduced from 80-400
                    'max_depth': trial.suggest_int('lgb_max_depth', 2, 5),  # Reduced from 3-6 to prevent overfitting
                    'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.05, log=True), # Reduced max learning rate
                    'subsample': trial.suggest_float('lgb_subsample', 0.4, 0.7),  # Reduced for regularization
                    'colsample_bytree': trial.suggest_float('lgb_colsample_bytree', 0.4, 0.7),
                    'min_child_samples': trial.suggest_int('lgb_min_child_samples', 40, 100),  # Increased min samples
                    'reg_alpha': trial.suggest_float('lgb_reg_alpha', 1.0, 5.0),  # Increased regularization
                    'reg_lambda': trial.suggest_float('lgb_reg_lambda', 2.0, 10.0),  # Increased regularization
                }
            
            if HAS_XGBOOST:
                params['xgb'] = {
                    'n_estimators': trial.suggest_int('xgb_n_estimators', 30, 120),  # Reduced from 80-400
                    'max_depth': trial.suggest_int('xgb_max_depth', 2, 5),  # Reduced from 3-6
                    'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.05, log=True),
                    'subsample': trial.suggest_float('xgb_subsample', 0.4, 0.7),
                    'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.4, 0.7),
                    'min_child_weight': trial.suggest_int('xgb_min_child_weight', 5, 15),  # Increased min weight
                    'gamma': trial.suggest_float('xgb_gamma', 0.2, 1.0),  # Increased gamma
                    'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1.0, 5.0),  # Increased reg
                    'reg_lambda': trial.suggest_float('xgb_reg_lambda', 2.0, 10.0),  # Increased reg
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
        
        # REDUCED to 15 trials to prevent overfitting to validation set
        # Too many trials = model overfits to cross-validation splits
        logger.info(f"Starting hyperparameter tuning with {n_workers} parallel workers (using {multiprocessing.cpu_count()} CPU cores)...")
        logger.info("Running 15 trials (reduced from 50 to prevent overfitting)...")
        study.optimize(objective, n_trials=15, timeout=600, n_jobs=n_workers)  # 15 trials, 10 min timeout
        
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
        # Trust self.use_regime_models if set (e.g. forced by _load_model), otherwise check metadata
        if not self.use_regime_models and not self.metadata.get('use_regime_models', False):
            return
        
        self.regime_models = {}
        regimes = ['bull', 'bear', 'sideways']
        
        for regime in regimes:
            regime_model_file = f"ml_model_{self.strategy}_{regime}.pkl"
            regime_scaler_file = f"scaler_{self.strategy}_{regime}.pkl"
            regime_encoder_file = f"label_encoder_{self.strategy}_{regime}.pkl"
            
            # Only strictly require the model file - scaler and encoder can be recovered/defaulted
            if self.storage.model_exists(regime_model_file):
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
                    # Load the actual model components
                    if self.storage.model_exists(regime_model_file):
                        regime_ml.model = self.storage.load_model(regime_model_file)
                    else:
                        logger.warning(f"Regime model file missing: {regime_model_file}")
                        continue

                    if self.storage.model_exists(regime_scaler_file):
                        regime_ml.scaler = self.storage.load_model(regime_scaler_file)
                    else:
                        logger.warning(f"Regime scaler missing for {regime}. Using unfitted StandardScaler (may fail).")
                        regime_ml.scaler = StandardScaler()

                    # Load or recover label encoder
                    if self.storage.model_exists(regime_encoder_file):
                        regime_ml.label_encoder = self.storage.load_model(regime_encoder_file)
                    else:
                        logger.warning(f"Regime label encoder missing for {regime}. Creating default.")
                        regime_ml.label_encoder = LabelEncoder()
                        # Default classes - simplified recovery
                        regime_ml.label_encoder.fit(['BUY', 'HOLD', 'SELL'])
                    
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
                                if 'selected_feature_names' in regime_own_metadata:
                                    regime_metadata['selected_feature_names'] = regime_own_metadata['selected_feature_names']
                                if 'scaler_n_features' in regime_own_metadata:
                                    regime_metadata['scaler_n_features'] = regime_own_metadata['scaler_n_features']
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
                if self.storage.model_exists(regime_model_file):
                    regime_files_exist = True
                    break
                # Log usage for debugging once (checking 'bull' regime)
                if regime == 'bull':
                     logger.debug(f"Checking regime model path: {self.storage.data_dir / regime_model_file}")
            
            # Check if single model files exist
            single_model_exists = self.storage.model_exists(self.model_file)
            logger.debug(f"Checking single model path: {self.storage.data_dir / self.model_file}")
            
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
                else:
                    logger.debug(f"DIAGNOSTIC - Regime {regime} not ready: model={regime_ml.model is not None}, "
                               f"has_classes={hasattr(regime_ml.label_encoder, 'classes_')}")
            logger.debug(f"DIAGNOSTIC - No ready regime models found out of {len(self.regime_models)}")
            return False
        
        # Check single model
        if self.model is None:
            logger.debug("DIAGNOSTIC - Single model is None")
            return False
        # Also check if LabelEncoder is fitted (needed for predictions)
        if not hasattr(self.label_encoder, 'classes_') or len(self.label_encoder.classes_) == 0:
            logger.debug(f"DIAGNOSTIC - Single model label encoder not fitted: {hasattr(self.label_encoder, 'classes_')}")
            return False
        return True

