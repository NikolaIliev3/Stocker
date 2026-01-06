"""
Machine Learning Training System
Trains ML models on historical stock data to improve predictions
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score
import joblib
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

from secure_ml_storage import SecureMLStorage

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extracts ML features from stock data and indicators"""
    
    def extract_features(self, stock_data: dict, history_data: dict, 
                        financials_data: dict = None, indicators: dict = None,
                        market_regime: dict = None, timeframe_analysis: dict = None,
                        relative_strength: dict = None, support_resistance: dict = None) -> np.ndarray:
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
    
    def get_feature_names(self) -> List[str]:
        """Get names of all features"""
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
        return base_features


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
        self.feature_selection_threshold = 0.01  # Minimum importance to keep feature
        self.metadata = {}
        self.train_accuracy = 0.0
        self.test_accuracy = 0.0
        
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
    
    def train(self, training_samples: List[Dict], test_size: float = 0.2) -> Dict:
        """Train ML model on historical data"""
        if len(training_samples) < 100:
            return {"error": f"Insufficient training samples: {len(training_samples)} < 100"}
        
        logger.info(f"Training ML model on {len(training_samples)} samples...")
        
        # Prepare data
        X = np.array([s['features'] for s in training_samples])
        y = np.array([s['label'] for s in training_samples])
        
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
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # ===== FEATURE SELECTION =====
        # Step 1: Train initial model to get feature importances
        logger.info("Step 1: Training initial model to calculate feature importances...")
        initial_rf = RandomForestClassifier(
            n_estimators=50,  # Smaller for faster initial training
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        # Scale features for initial training
        X_train_scaled_initial = self.scaler.fit_transform(X_train)
        X_test_scaled_initial = self.scaler.transform(X_test)
        
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
        
        # Step 2: Select features based on importance threshold
        # Map feature names back to original indices
        all_feature_names = self.feature_extractor.get_feature_names()
        name_to_index = {name: idx for idx, name in enumerate(all_feature_names)}
        
        selected_feature_indices = []
        selected_feature_names = []
        removed_features = []
        
        for name, importance in feature_importance_dict.items():
            if importance >= self.feature_selection_threshold:
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
        
        # Step 3: Filter features
        X_train_selected = X_train_scaled_initial[:, self.selected_features]
        X_test_selected = X_test_scaled_initial[:, self.selected_features]
        
        logger.info(f"Step 2: Retraining model with {len(selected_feature_indices)} selected features...")
        
        # Update scaler to work with selected features only
        # Create new scaler for selected features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # Store feature importance for selected features only
        self.feature_importance = {
            name: feature_importance_dict[name] 
            for name in selected_feature_names
        }
        
        # Create ensemble model with regularization to prevent overfitting
        # Reduced complexity to improve generalization
        rf = RandomForestClassifier(
            n_estimators=100,  # Reduced from 200
            max_depth=8,  # Reduced from 15 to prevent overfitting
            min_samples_split=20,  # Increased from 10
            min_samples_leaf=10,  # Increased from 5
            max_features='sqrt',  # Limit features per split
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=100,  # Reduced from 150
            max_depth=5,  # Reduced from 10
            learning_rate=0.05,  # Reduced from 0.1 for slower learning
            min_samples_split=20,  # Added regularization
            min_samples_leaf=10,  # Added regularization
            subsample=0.8,  # Use 80% of samples per tree (prevents overfitting)
            random_state=42
        )
        
        lr = LogisticRegression(
            max_iter=1000,
            C=0.1,  # Added regularization (lower C = more regularization)
            class_weight='balanced',
            random_state=42
            # L2 regularization is the default, no need to specify penalty='l2'
        )
        
        # Ensemble voting classifier
        self.model = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb), ('lr', lr)],
            voting='soft',
            weights=[2, 2, 1]
        )
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Warn if overfitting detected
        if train_score > 0.95 and test_score < train_score - 0.1:
            logger.warning(f"⚠️ Overfitting detected! Train: {train_score*100:.1f}%, Test: {test_score*100:.1f}%")
            logger.warning("   Model may be too complex. Consider reducing model complexity or adding more training data.")
        
        # Log training details
        logger.info(f"Training accuracy: {train_score*100:.1f}%, Validation accuracy: {test_score*100:.1f}%")
        if self.selected_features is not None:
            total_features = len(self.feature_extractor.get_feature_names())
            selected_count = len(self.selected_features)
            logger.info(f"✅ Feature selection: Using {selected_count}/{total_features} features "
                       f"({selected_count/total_features*100:.1f}% retained)")
        
        y_pred = self.model.predict(X_test_scaled)
        
        # Feature importance (already calculated during feature selection)
        # Update with final model importances for selected features only
        if hasattr(self.model.named_estimators_['rf'], 'feature_importances_'):
            final_importances = self.model.named_estimators_['rf'].feature_importances_
            # Map back to feature names (only selected features)
            selected_feature_names = [self.feature_extractor.get_feature_names()[idx] 
                                     for idx in self.selected_features]
            final_importance_dict = dict(zip(selected_feature_names, final_importances))
            # Merge with initial importances (keep the higher value)
            for name in final_importance_dict:
                if name in self.feature_importance:
                    self.feature_importance[name] = max(
                        self.feature_importance[name],
                        final_importance_dict[name]
                    )
                else:
                    self.feature_importance[name] = final_importance_dict[name]
            
            # Sort and keep top 20
            self.feature_importance = dict(sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:20])
        
        # Cross-validation
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
            'total_features': len(self.feature_extractor.get_feature_names()),
            'selected_features_count': len(self.selected_features) if self.selected_features is not None else None,
            'label_classes': self.label_encoder.classes_.tolist() if hasattr(self.label_encoder, 'classes_') else None
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
        
        # Scale
        features_scaled = self.scaler.transform(features)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        probabilities = self.model.predict_proba(features_scaled)[0]
        
        # Decode label
        action = self.label_encoder.inverse_transform([prediction])[0]
        
        # Calculate confidence
        confidence = float(probabilities[prediction] * 100)
        
        # Get probabilities for all classes
        prob_dict = {}
        for i, class_name in enumerate(self.label_encoder.classes_):
            prob_dict[class_name] = float(probabilities[i] * 100)
        
        return {
            'action': action,
            'confidence': confidence,
            'probabilities': prob_dict
        }
    
    def is_trained(self) -> bool:
        """Check if model is trained and ready to use"""
        if self.model is None:
            return False
        # Also check if LabelEncoder is fitted (needed for predictions)
        if not hasattr(self.label_encoder, 'classes_') or len(self.label_encoder.classes_) == 0:
            return False
        return True

