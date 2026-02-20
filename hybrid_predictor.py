"""
Hybrid Stock Predictor
Combines rule-based weight adjustment with ML for optimal predictions
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

from adaptive_weight_adjuster import AdaptiveWeightAdjuster
from ml_training import StockPredictionML, FeatureExtractor
from trading_analyzer import TradingAnalyzer
from investing_analyzer import InvestingAnalyzer
from mixed_analyzer import MixedAnalyzer
from prediction_cache import get_cache
from prediction_cache import get_cache
from algorithm_improvements import StatisticalTargetCalibrator
from pattern_matcher import PatternMatcher

# Import Sector ML if available
try:
    from sector_ml import SectorMLManager, SECTOR_DISPLAY_NAMES
    from sector_mapper import SectorMapper
    HAS_SECTOR_ML = True
except ImportError:
    HAS_SECTOR_ML = False

# Import configuration
# Logger must be initialized before using it in try/except block
logger = logging.getLogger(__name__)

# Import Quant Mode Overrides (High Accuracy)
try:
    from quant_config import ML_HIGH_ACCURACY_THRESHOLD, ML_LABEL_THRESHOLD, ML_RF_N_ESTIMATORS
    IS_QUANT_MODE = True
    logger.info("🎯 Quant Mode Active: High Accuracy Thresholds Loaded")
except ImportError:
    # Standard Configuration
    from config import (ML_HIGH_ACCURACY_THRESHOLD, ML_VERY_HIGH_ACCURACY_THRESHOLD,
                   ML_PERFORMANCE_ACCURACY_DIFF_THRESHOLD, ML_CONFIDENCE_DIFF_THRESHOLD_NORMAL,
                   ML_PERFORMANCE_HISTORY_LIMIT, ML_RECENT_PERFORMANCE_WINDOW,
                   ML_ENSEMBLE_WEIGHT_UPDATE_THRESHOLD, ML_ENSEMBLE_WEIGHT_MAX,
                   AI_PREDICTOR_ENABLED, LLM_AI_PREDICTOR_ENABLED)
    IS_QUANT_MODE = False


class HybridStockPredictor:
    """Hybrid predictor combining rule-based and ML approaches"""
    
    def __init__(self, data_dir: Path, strategy: str, 
                 trading_analyzer: TradingAnalyzer = None,
                 investing_analyzer: InvestingAnalyzer = None,
                 mixed_analyzer: MixedAnalyzer = None,
                 data_fetcher = None,
                 seeker_ai = None):
        self.data_dir = data_dir
        self.strategy = strategy
        
        # Initialize components
        self.weight_adjuster = AdaptiveWeightAdjuster(data_dir, strategy)
        self.ml_predictor = StockPredictionML(data_dir, strategy)
        self.feature_extractor = FeatureExtractor()
        self._ml_warning_logged = False
        self._ml_success_logged = False
        
        # Base analyzers with data fetcher injection for regime detection
        self.trading_analyzer = trading_analyzer or TradingAnalyzer(data_fetcher=data_fetcher)
        self.investing_analyzer = investing_analyzer or InvestingAnalyzer()
        self.mixed_analyzer = mixed_analyzer or MixedAnalyzer()
        self.mixed_analyzer = mixed_analyzer or MixedAnalyzer()
        self.statistical_calibrator = StatisticalTargetCalibrator()
        self.pattern_matcher = PatternMatcher(data_dir)
        
        # AI components removed
        self.ai_predictor = None
        self.llm_ai_predictor = None
        
        # Ensemble weights (now includes AI)
        self.ensemble_weights_file = data_dir / f"ensemble_weights_{strategy}.json"
        self.ensemble_weights = self._load_ensemble_weights()
        
        # Performance tracking
        self.performance_file = data_dir / "performance_history.json"
        self.performance_history = self._load_performance()
        
        # Sector ML Manager (for sector-specific specialized models)
        self.sector_manager = None
        self.sector_mapper = None
        if HAS_SECTOR_ML:
            try:
                self.sector_manager = SectorMLManager(data_dir, strategy)
                self.sector_mapper = SectorMapper(data_dir)
                sector_count = len(self.sector_manager.list_models())
                if sector_count > 0:
                    logger.info(f"🏭 Loaded {sector_count} Sector model(s) for {strategy}")
            except Exception as e:
                logger.warning(f"Could not initialize Sector ML Manager: {e}")
    
    def _load_ensemble_weights(self) -> Dict:
        """Load ensemble weights"""
        if self.ensemble_weights_file.exists():
            try:
                with open(self.ensemble_weights_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.debug(f"Error loading ensemble weights: {e}")
        # Ensemble weights: Rules and ML only
        default_weights = {'rule': 0.5, 'ml': 0.5, 'ai': 0.0}
        return default_weights
    
    def _save_ensemble_weights(self) -> None:
        """Save ensemble weights"""
        try:
            with open(self.ensemble_weights_file, 'w') as f:
                json.dump(self.ensemble_weights, f, indent=2)
        except (IOError, OSError, json.JSONEncodeError) as e:
            logger.error(f"Error saving ensemble weights: {e}")
    
    def _load_performance(self) -> Dict:
        """Load performance history"""
        if self.performance_file.exists():
            try:
                with open(self.performance_file, 'r') as f:
                    all_performance = json.load(f)
                    return all_performance.get(self.strategy, {
                        'weight_based': [],
                        'ml_based': [],
                        'hybrid': []
                    })
            except Exception as e:
                logger.debug(f"Error loading performance history: {e}")
        return {
            'weight_based': [],
            'ml_based': [],
            'hybrid': []
        }
    
    def _save_performance(self) -> None:
        """Save performance history"""
        try:
            all_performance = {}
            if self.performance_file.exists():
                try:
                    with open(self.performance_file, 'r') as f:
                        all_performance = json.load(f)
                except (IOError, OSError, json.JSONDecodeError):
                    logger.debug("Could not load existing performance data, starting fresh")
            
            all_performance[self.strategy] = self.performance_history
            
            # Keep only last N entries (from config)
            for method in ['weight_based', 'ml_based', 'hybrid']:
                if len(all_performance[self.strategy][method]) > ML_PERFORMANCE_HISTORY_LIMIT:
                    all_performance[self.strategy][method] = \
                        all_performance[self.strategy][method][-ML_PERFORMANCE_HISTORY_LIMIT:]
            
            with open(self.performance_file, 'w') as f:
                json.dump(all_performance, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance: {e}")
    
    def predict(self, stock_data: dict, history_data: dict, 
                financials_data: dict = None, is_backtest: bool = False, current_date=None) -> Dict:
        """Make prediction using hybrid approach"""
        # Check cache first (only if not backtesting)
        symbol = stock_data.get('symbol', '')
        if symbol and not is_backtest:
            cache = get_cache()
            cached_prediction = cache.get(symbol, self.strategy)
            if cached_prediction:
                logger.debug(f"⚡ Using cached prediction for {symbol} ({self.strategy})")
                return cached_prediction
        
        # --- CRITICAL FIX: TEMPORAL FIREWALL (Anti-Lookahead) ---
        if current_date:
            try:
                if isinstance(current_date, str):
                    current_date = datetime.strptime(current_date, '%Y-%m-%d')
                
                # Slicing history_data ensures ALL downstream components (Analyzers, Features, ML)
                # are blind to any data beyond the simulated "today".
                if history_data and 'data' in history_data:
                    orig_len = len(history_data['data'])
                    history_data['data'] = [
                        row for row in history_data['data'] 
                        if datetime.strptime(row['date'], '%Y-%m-%d') <= current_date
                    ]
                    if len(history_data['data']) < orig_len:
                        logger.debug(f"🛡️ Temporal Firewall: Sliced history ({orig_len} -> {len(history_data['data'])} rows)")
            except Exception as e:
                logger.warning(f"Temporal Firewall error: {e}")
        # ---------------------------------------------------------
        
        # 0. ENSURE FRESH DATA (Real-time Entry Price Realism)
        # The user noted that entry prices can be stale. We force a fresh price check here.
        # SKIP this if we are backtesting (prevent data leakage/time travel)
        if not is_backtest and self.trading_analyzer and hasattr(self.trading_analyzer, 'data_fetcher') and self.trading_analyzer.data_fetcher:
            try:
                # Quick fetch of just the current price (force_refresh=True)
                fresh_data = self.trading_analyzer.data_fetcher.fetch_stock_data(symbol, force_refresh=True)
                if fresh_data and fresh_data.get('price'):
                    current_real_time_price = fresh_data.get('price')
                    old_price = stock_data.get('price', 0)
                    
                    # Log if there was a significant discrepancy (stale data detected)
                    if abs(current_real_time_price - old_price) / old_price > 0.005: # > 0.5% diff
                        logger.warning(f"⚠️ Stale price detected for {symbol}! Old: {old_price}, Fresh: {current_real_time_price}. Updating...")
                    
                    # Update stock_data with fresh price
                    stock_data['price'] = current_real_time_price
                    # Also update history last close if it's today (to avoid gap)
                    # (Optional: logic to update history_data.iloc[-1] if needed, but price is most important)
            except Exception as e:
                logger.debug(f"Could not force refresh price for {symbol}: {e}")

        # 1. Get rule-based prediction
        if self.strategy == "trading":
            base_analysis = self.trading_analyzer.analyze(stock_data, history_data, current_date=current_date)
        elif self.strategy == "investing":
            base_analysis = self.investing_analyzer.analyze(
                stock_data, financials_data, history_data, current_date=current_date
            )
        else:  # mixed
            base_analysis = self.mixed_analyzer.analyze(
                stock_data, financials_data, history_data, current_date=current_date
            )
        
        if 'error' in base_analysis:
            return base_analysis
            
        # Preserver extra fields from stock_data (like laplacian_score)
        if 'laplacian_score' in stock_data:
            base_analysis['laplacian_score'] = stock_data['laplacian_score']
        
        # Apply learned weights to rule-based prediction
        rule_prediction = self._apply_learned_weights(base_analysis)
        
        # AI prediction removed
        ai_prediction = None
        ai_available = False
        
        # 3. Get ML prediction (if available)
        ml_prediction = None
        ml_available = self.ml_predictor.is_trained()
        
        # Diagnostic: Log ML model status (use INFO level for visibility)
        if not ml_available:
            # Only log warning once to avoid spamming
            if not getattr(self, '_ml_warning_logged', False):
                logger.warning(f"⚠️ ML model not available for {self.strategy} strategy")
                # Try to diagnose why
                if hasattr(self.ml_predictor, 'model'):
                    logger.warning(f"   - Main model is None: {self.ml_predictor.model is None}")
                if hasattr(self.ml_predictor, 'regime_models'):
                    logger.warning(f"   - Regime models count: {len(self.ml_predictor.regime_models)}")
                    if self.ml_predictor.regime_models:
                        for regime, regime_ml in self.ml_predictor.regime_models.items():
                            logger.warning(f"     - {regime}: model={regime_ml.model is not None if hasattr(regime_ml, 'model') else 'N/A'}")
                self._ml_warning_logged = True
        else:
            # Only log success once
            if not getattr(self, '_ml_success_logged', False):
                logger.info(f"✓ ML model available for {self.strategy} strategy")
                self._ml_success_logged = True
        
        # Check for Sector-specific model (sector-specialized ML)
        sector_result = None
        used_sector_model = False
        sector_fallback_reason = None
        detected_sector = None
        
        if self.sector_manager and self.sector_mapper and symbol:
            try:
                # Detect sector for this stock
                detected_sector = self.sector_mapper.get_sector(symbol)
                
                if detected_sector and self.sector_manager.has_model(detected_sector):
                    # Get sector model accuracy
                    sector_accuracy = self.sector_manager.get_accuracy(detected_sector)
                    
                    # Get general model accuracy
                    general_accuracy = 0
                    if hasattr(self.ml_predictor, 'metadata') and self.ml_predictor.metadata:
                        general_accuracy = self.ml_predictor.metadata.get('test_accuracy', 0)
                    
                    sector_display = SECTOR_DISPLAY_NAMES.get(detected_sector, detected_sector)
                    
                    # Only use sector model if it's more accurate than general
                    if sector_accuracy > general_accuracy:
                        indicators = base_analysis.get('indicators', {})
                        market_regime = base_analysis.get('market_regime', None)
                        timeframe_analysis = base_analysis.get('timeframe_analysis', None)
                        relative_strength = base_analysis.get('relative_strength', None)
                        support_resistance = base_analysis.get('support_resistance', None)
                        
                        features = self.feature_extractor.extract_features(
                            stock_data, history_data, financials_data, indicators,
                            market_regime, timeframe_analysis, relative_strength, support_resistance,
                            None, None, None, volume_analysis, macro_info,
                            as_of_date=current_date
                        )
                        
                        sector_result = self.sector_manager.predict(detected_sector, features, self.ml_predictor)
                        used_sector_model = sector_result.get('used_sector_model', False)
                        
                        if used_sector_model:
                            logger.info(f"🏭 Using {sector_display} model for {symbol}: {sector_result.get('action')} ({sector_result.get('confidence', 0):.0%}) [Sector: {sector_accuracy*100:.1f}% > General: {general_accuracy*100:.1f}%]")
                    else:
                        # Sector model exists but general is better
                        sector_fallback_reason = f"General model ({general_accuracy*100:.1f}%) outperforms {sector_display} ({sector_accuracy*100:.1f}%)"
                        logger.info(f"📊 {symbol}: {sector_fallback_reason} - Using general model")
                elif detected_sector:
                    sector_fallback_reason = f"No trained model for sector: {detected_sector}"
                else:
                    sector_fallback_reason = f"Unknown sector for {symbol}"
                    
            except Exception as e:
                logger.warning(f"Sector ML prediction failed for {symbol}: {e}")
        
        if ml_available:
            try:
                indicators = base_analysis.get('indicators', {})
                # Get enhanced analysis data for feature extraction
                market_regime = base_analysis.get('market_regime', None)
                timeframe_analysis = base_analysis.get('timeframe_analysis', None)
                relative_strength = base_analysis.get('relative_strength', None)
                support_resistance = base_analysis.get('support_resistance', None)
                
                # Calculate volume analysis for liquidity features (matching training pipeline)
                volume_analysis = None
                if history_data and history_data.get('data') and len(history_data['data']) >= 20:
                    try:
                        hist_df = pd.DataFrame(history_data['data'])
                        # Ensure column names are lowercase
                        hist_df.columns = [col.lower() if isinstance(col, str) else col for col in hist_df.columns]
                        
                        if 'close' in hist_df.columns and 'volume' in hist_df.columns:
                            avg_volume = hist_df['volume'].tail(20).mean()
                            avg_price = hist_df['close'].tail(20).mean()
                            avg_dollar_volume = avg_volume * avg_price
                            current_volume = hist_df['volume'].iloc[-1] if len(hist_df) > 0 else 0
                            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                            
                            # Calculate volume trend (last vs 10 days ago)
                            if len(hist_df) >= 10:
                                volume_trend = "increasing" if hist_df['volume'].iloc[-1] > hist_df['volume'].iloc[-10] else "decreasing"
                            else:
                                volume_trend = "neutral"
                                
                            is_high_volume = volume_ratio > 1.5
                            
                            # Liquidity grade based on daily dollar volume
                            if avg_dollar_volume >= 50_000_000:
                                liquidity_grade = 'high'
                            elif avg_dollar_volume >= 10_000_000:
                                liquidity_grade = 'medium'
                            elif avg_dollar_volume >= 1_000_000:
                                liquidity_grade = 'low'
                            else:
                                liquidity_grade = 'very_low'
                            
                            volume_analysis = {
                                'volume_ratio': float(volume_ratio),
                                'avg_dollar_volume': float(avg_dollar_volume),
                                'liquidity_grade': liquidity_grade,
                                'volume_trend': volume_trend,
                                'is_high_volume': is_high_volume
                            }
                    except Exception as vol_error:
                        pass  # Use default if calculation fails

                features = self.feature_extractor.extract_features(
                    stock_data, history_data, financials_data, indicators,
                    market_regime, timeframe_analysis, relative_strength, support_resistance,
                    None, None, None, volume_analysis, base_analysis.get('macro_info'),
                    as_of_date=current_date
                )
                
                # DEBUG: Log all input features to diagnose SELL bias
                if features is not None:
                    is_empty = False
                    if isinstance(features, pd.DataFrame):
                        if features.empty:
                            is_empty = True
                    elif isinstance(features, np.ndarray):
                        if features.size == 0:
                            is_empty = True
                    elif len(features) == 0:
                         is_empty = True
                         
                    if not is_empty:
                        # Safe logging for DataFrame or other types
                        log_features = features.iloc[0].to_dict() if isinstance(features, pd.DataFrame) else "Numpy Array/List"
                        logger.info(f"🔍 ML Input Features for {symbol} (Full): {log_features}")
                # Pass stock_data and history_data for regime-specific model detection
                ml_prediction = self.ml_predictor.predict(
                    features, stock_data, history_data, 
                    is_backtest=is_backtest, 
                    as_of_date=current_date
                )
                # Check if prediction has an error (e.g., LabelEncoder not fitted)
                if ml_prediction and 'error' in ml_prediction:
                    logger.warning(f"⚠️ ML prediction unavailable: {ml_prediction.get('error')}. Using rule-based only.")
                    ml_available = False
                    ml_prediction = None
                else:
                    # Diagnostic logging for backtesting (use INFO for visibility)
                    if ml_prediction:
                        # NEW: Get dynamic target price
                        try:
                            target_res = self.ml_predictor.predict_target(features, stock_data, history_data)
                            if target_res and 'predicted_return' in target_res:
                                ml_prediction['predicted_return'] = target_res['predicted_return']
                                # Calculate absolute target price
                                current_price = stock_data.get('price', 0)
                                if current_price > 0:
                                     ml_prediction['target_price'] = current_price * (1 + target_res['predicted_return']/100)
                        except Exception as e:
                            logger.debug(f"Failed to get target prediction: {e}")

                        logger.info(f"TRAIL_MARKER ML prediction: action={ml_prediction.get('action')}, confidence={ml_prediction.get('confidence', 0):.1f}%, "
                                   f"probabilities={ml_prediction.get('probabilities', {})}")
            except Exception as e:
                logger.error(f"❌ ML prediction failed: {e}. Using rule-based only.")
                import traceback
                logger.error(traceback.format_exc())
                ml_available = False
                ml_prediction = None
        
        
        # CRITICAL FIX: Use Sector Model if it was determined to be better
        if used_sector_model and sector_result:
            try:
                # Use the Sector Result as the ML Prediction
                ml_prediction = {
                    'action': sector_result.get('action', 'HOLD'),
                    'confidence': sector_result.get('confidence', 0),
                    'probabilities': sector_result.get('probabilities', {}),
                    'source': 'sector_model',
                    'sector': detected_sector
                }
                ml_available = True
                
                # Log the switch
                general_acc_disp = f"{general_accuracy*100:.1f}%" if 'general_accuracy' in locals() else "Lower"
                sector_acc_disp = f"{sector_result.get('sector_accuracy', 0)*100:.1f}%"
                
                logger.info(f"🔄 Switched to SECTOR MODEL for {symbol}: {ml_prediction['action']} "
                           f"({sector_acc_disp} acc > {general_acc_disp} acc)")
            except Exception as e:
                logger.error(f"Error switching to sector model: {e}")
        
        # Consensus and AI updates removed
        
        # 4. Calculate dynamic ensemble weights (Rule and ML only)
        if ml_available:
            ensemble_weights = self._calculate_dynamic_weights(rule_prediction, ml_prediction)
        else:
            ensemble_weights = {'rule': 1.0, 'ml': 0.0, 'ai': 0.0}
        
        # 5. Combine predictions (Rules and ML)
        final_prediction = self._ensemble_predictions(
            rule_prediction, ml_prediction, ensemble_weights, base_analysis,
            stock_data, history_data,
            is_backtest=is_backtest
        )
        
        # Diagnostic logging for backtesting
        if ml_available and ml_prediction:
            log_parts = [f"Hybrid prediction: rule={rule_prediction.get('action')} ({rule_prediction.get('confidence', 0):.1f}%)"]
            if ml_available and ml_prediction:
                log_parts.append(f"ml={ml_prediction.get('action')} ({ml_prediction.get('confidence', 0):.1f}%)")
            log_parts.append(f"final={final_prediction.get('recommendation', {}).get('action')} "
                           f"(weights: rule={ensemble_weights.get('rule', 0):.2f}, ml={ensemble_weights.get('ml', 0):.2f})")
            logger.debug(", ".join(log_parts))
        
        # Add Sector ML metadata to final prediction
        if sector_result and used_sector_model:
            final_prediction['used_sector_model'] = True
            final_prediction['sector'] = detected_sector
            final_prediction['sector_display'] = SECTOR_DISPLAY_NAMES.get(detected_sector, detected_sector) if HAS_SECTOR_ML else detected_sector
            final_prediction['sector_accuracy'] = sector_result.get('sector_accuracy', 0)
        elif detected_sector:
            final_prediction['used_sector_model'] = False
            final_prediction['sector'] = detected_sector
            final_prediction['sector_fallback_reason'] = sector_fallback_reason
        else:
            final_prediction['used_sector_model'] = False
            final_prediction['sector'] = None
            final_prediction['sector_fallback_reason'] = sector_fallback_reason
        
        # 6. Add Historical Upside Analysis (Phase 5)
        self._add_historical_upside(final_prediction, base_analysis)

        # Store in cache for future use
        if symbol:
            cache = get_cache()
            cache.set(symbol, self.strategy, final_prediction)
            logger.debug(f"💾 Cached prediction for {symbol} ({self.strategy})")
        
        return final_prediction
    
    def _determine_signal_context(self, stock_data: dict, history_data: dict, base_analysis: dict) -> str:
        """Classifies the signal into specific market scenarios for transparency."""
        try:
            if not history_data or 'data' not in history_data or len(history_data['data']) < 50:
                return "Unknown Context"
                
            df = pd.DataFrame(history_data['data'])
            # Standardize columns
            df.columns = [col.lower() for col in df.columns]
            if 'close' not in df.columns: return "Unknown Context"
            
            close = df['close']
            current_price = stock_data.get('price', close.iloc[-1])
            sma50 = close.rolling(50).mean()
            
            curr_sma50 = sma50.iloc[-1]
            prev_sma50 = sma50.iloc[-5] if len(sma50) > 5 else curr_sma50
            
            # --- Scenario 1: Trend Reversal (Highest Accuracy) ---
            # Definition: Price recently crossed above SMA50 or RSI was oversold
            was_below = any(close.iloc[-10:-2] < sma50.iloc[-10:-2]) if len(close) > 10 else False
            if was_below and current_price > curr_sma50:
                return "Trend Reversal (High Accuracy)"
            
            # --- Scenario 2: Strong Uptrend (Momentum) ---
            # Definition: Price > SMA50 and SMA50 is rising
            if current_price > curr_sma50 and curr_sma50 > prev_sma50:
                return "Strong Uptrend (Momentum)"
                
            # --- Scenario 3: Sector Impulse (New) ---
            # Definition: Stock is outperforming sector even if technicals are mixed
            indicators = base_analysis.get('indicators', {})
            sector_analysis = base_analysis.get('sector_analysis') # Might be passed in
            if not sector_analysis and 'sector_analysis' in indicators:
                sector_analysis = indicators['sector_analysis']
                
            if sector_analysis and sector_analysis.get('outperforming_sector'):
                return "Sector Impulse (Market Alpha)"
                
            # Fallback
            if current_price > curr_sma50:
                return "Technical Breakout"
            else:
                return "Counter-Trend / Mean Reversion"
                
        except Exception as e:
            logger.debug(f"Failed to determine signal context: {e}")
            return "Standard Signal"

    def _add_historical_upside(self, prediction: Dict, base_analysis: Dict) -> None:
        """Enrich prediction with historical upside analysis"""
        try:
            if not prediction or prediction.get('action') != 'BUY':
                return
                
            # Extract features for pattern matching
            indicators = base_analysis.get('indicators', {})
            current_price = base_analysis.get('price', 0) or base_analysis.get('current_price', 0)
            ema_50 = indicators.get('ema_50', current_price)
            
            # Safe feature extraction
            rsi = indicators.get('rsi', 50)
            atr_percent = base_analysis.get('volatility', {}).get('atr_percent', 0)
            if atr_percent == 0 and 'atr' in indicators and current_price > 0:
                atr_percent = (indicators['atr'] / current_price) * 100
                
            sma50_dist = 0.0
            if ema_50 > 0:
                sma50_dist = ((current_price / ema_50) - 1) * 100
                
            confidence = prediction.get('confidence', 0)
            
            # Query Pattern Matcher
            feature_dict = {
                'rsi': rsi,
                'atr_percent': atr_percent,
                'sma50_dist': sma50_dist
            }
            
            # Re-init if needed
            if not hasattr(self, 'pattern_matcher'):
                 self.pattern_matcher = PatternMatcher(self.data_dir)
                 
            history = self.pattern_matcher.find_similar_patterns(
                feature_dict=feature_dict,
                confidence=confidence
            )
            
            # Add to prediction
            if history:
                 prediction['history_upside'] = history
                 if history['sample_size'] >= 20:
                     logger.info(f"📜 History: {history['note']}")
                     
        except Exception as e:
            logger.debug(f"Failed to add historical upside: {e}")

    def _apply_learned_weights(self, base_analysis: dict) -> dict:
        """Apply learned weights to rule-based analysis"""
        # This is simplified - in practice, you'd modify the analyzer
        # to use learned weights. For now, we'll use the base analysis.
        recommendation = base_analysis.get('recommendation', {}).copy()
        return {
            'action': recommendation.get('action', 'HOLD'),
            'confidence': float(recommendation.get('confidence', 50)),
            'entry_price': float(recommendation.get('entry_price', 0)),
            'target_price': float(recommendation.get('target_price', 0)),
            'stop_loss': float(recommendation.get('stop_loss', 0)),
            'estimated_days': int(recommendation.get('estimated_days', 7))
        }
    
    def _calculate_dynamic_weights(self, rule_pred: Dict, ml_pred: Optional[Dict]) -> Dict:
        """Calculate dynamic ensemble weights (Rule and ML only)
        
        CRITICAL FIX: ML weight is now based on PROVEN backtest accuracy,
        not just training accuracy. If backtest accuracy is poor (< 55%),
        ML weight is significantly reduced.
        """
        has_ml = ml_pred is not None
        
        if not has_ml:
            return {'rule': 1.0, 'ml': 0.0, 'ai': 0.0}
        
        # 1. Start with a balanced 50/50 baseline (Safe)
        rule_weight = 0.50
        ml_weight = 0.50
        
        # 2. Check REGIME Accuracy (Smart)
        regime_accuracy = 0
        current_regime = 'unknown'
        
        # Check if we have regime info from the last prediction
        if hasattr(self.ml_predictor, 'last_regime') and self.ml_predictor.last_regime:
             current_regime = self.ml_predictor.last_regime
             # Get accuracy for this specific regime
             if hasattr(self.ml_predictor, 'get_regime_accuracy'):
                 regime_accuracy = self.ml_predictor.get_regime_accuracy(current_regime) * 100
                 if regime_accuracy > 0:
                     logger.debug(f"🔍 Regime Context: {current_regime.upper()} (Acc: {regime_accuracy:.1f}%)")
        
        # Check PROVEN backtest performance first (most important)
        ml_perf = self._get_recent_performance('ml_based')
        rule_perf = self._get_recent_performance('weight_based')
        
        # 1. Check Training Accuracy (Potential)
        trained_accuracy = 0
        if hasattr(self.ml_predictor, 'metadata') and self.ml_predictor.metadata:
            # CRITICAL FIX: Use CV Mean if available, as it's more robust than single-split Test Acc
            cv_mean = self.ml_predictor.metadata.get('cv_mean', 0)
            test_acc = self.ml_predictor.metadata.get('test_accuracy', 0)
            # Trust the established stability (CV) over the specific test split noise
            trained_accuracy = max(cv_mean, test_acc) * 100
            if cv_mean > test_acc:
                logger.debug(f"📊 Trusting CV Score ({cv_mean:.1%}) over Test Score ({test_acc:.1%}) for weighting")
            
        # 2. Check Real Performance (Reality)
        proven_ml_accuracy = 50
        has_history = False
        if ml_perf and ml_perf.get('total', 0) >= 10:
            proven_ml_accuracy = ml_perf.get('accuracy', 50)
            has_history = True

        # DYNAMIC ADJUSTMENT
        # Priority: Regime Accuracy > Proven History > Global Training Accuracy
        
        effective_accuracy = 50
        source = "default"
        
        if regime_accuracy > 0:
            effective_accuracy = regime_accuracy
            source = f"regime_{current_regime}"
        elif has_history:
            effective_accuracy = proven_ml_accuracy
            source = "history"
        else:
            effective_accuracy = trained_accuracy
            source = "training"
            
        # Adjust weight based on proven competence
        # If model is barely better than random (50-55%), give it almost zero weight
        if effective_accuracy < 55:
            ml_weight = 0.1 # Reduced from 0.2 for stricter filter
        elif effective_accuracy < 60:
            ml_weight = 0.3 # Reduced from 0.4
        elif effective_accuracy > 70:
            ml_weight = 0.8  # Higher trust
        elif effective_accuracy > 80:
             ml_weight = 0.9 # QUANT MODE dominant trust
             
        # Normalize
        total = rule_weight + ml_weight
        return {'rule': rule_weight / total, 'ml': ml_weight / total, 'ai': 0.0}

    def _calculate_market_stress_threshold(self, regime: str, market_data: dict) -> float:
        """
        Dynamic threshold based on regime and recovery signals (Conservative Logic)
        """
        base_threshold = -0.10  # -10.0% base
        
        # Regime adjustments
        if regime == "bull":
            regime_adj = -0.03  # Allow -13.0% in bull
        elif regime == "sideways":
            regime_adj = -0.01  # Allow -11.0% in sideways
        else:  # bear / crash
            regime_adj = 0.02   # Tighten to -8.0% in bear
            
        # Recovery bonus (only if substantial)
        roc_20 = market_data.get('roc_20', 0)
        recovery_adj = -0.02 if roc_20 > 3.0 else 0.0  # Allow 2% more if 3% recovery
        
        final_threshold = base_threshold + regime_adj + recovery_adj
        
        # Safety net: Don't allow beyond absolute -15% or above -8%
        return max(-0.15, min(-0.08, final_threshold))

    def _apply_market_stress_filter(self, regime: str, market_data: dict) -> tuple:
        """
        Adaptive market stress veto with regime awareness
        """
        current_drawdown = market_data.get('drawdown', 0)
        regime = regime.lower()
        
        threshold = self._calculate_market_stress_threshold(regime, market_data)
        
        if current_drawdown < threshold:
            msg = (
                f"🛑 MARKET STRESS VETO: "
                f"Drawdown {current_drawdown*100:.1f}% exceeds {threshold*100:.1f}% threshold "
                f"(Regime: {regime}, Recovery: {market_data.get('roc_20', 0):.1f}%)"
            )
            return True, msg
        
        return False, None

    def _apply_pro_filters(self, final_action: str, final_confidence: float, 
                      stock_data: dict, history_data: dict, base_analysis: dict = None, reasoning: str = "",
                      ml_prediction: dict = None, is_backtest: bool = False) -> tuple:
        """
        ELITE LONG-ONLY FILTERS (Quant Mode):
        - Goal: Extreme Accuracy (>60%) by filtering out 'noise' and shorting.
        - Rules:
            1. No SELL signals (Global Shorting Veto).
            2. Global BUY threshold: 75% confidence.
            3. SMA200 / Bear Market BUY threshold: 85% confidence.
        """
        if not base_analysis or final_action == 'HOLD':
            return final_action, final_confidence, reasoning
            
        symbol = stock_data.get('symbol', 'Unknown')
        regime_info = base_analysis.get('market_regime', {})
        regime = regime_info.get('regime', 'unknown')
        
        indicators = base_analysis.get('indicators', {})
        price_above_ema200 = indicators.get('price_above_ema200', True)
        
        # 0. VOLATILITY VETO (Global Hardening)
        # Filter out "Dead" stocks (<0.5%) and "Choppy" stocks (>3.0%)
        if final_action != 'HOLD':
             volatility = base_analysis.get('volatility', {})
             atr_percent = volatility.get('atr_percent', 0)
             
             if atr_percent > 4.0:
                 msg = f"🛑 VOLATILITY VETO: ATR {atr_percent:.1f}% > 4.0% (Too Volatile)"
                 logger.info(msg + ". Forced HOLD.")
                 reasoning += msg + "\n"
                 return 'HOLD', 50.0, reasoning
             elif atr_percent > 0 and atr_percent < 0.5:
                 msg = f"🛑 VOLATILITY VETO: ATR {atr_percent:.1f}% < 0.5% (Too Dead)"
                 logger.info(msg + ". Forced HOLD.")
                 reasoning += msg + "\n"
                 return 'HOLD', 50.0, reasoning

        # 1. HARD VETO ON SHORTS
        if final_action == 'SELL':
            msg = f"🛑 ELITE VETO: Suppressing {symbol} SELL ({final_confidence:.1f}%) - Long-Only Strategy"
            logger.info(msg)
            reasoning += msg + "\n"
            return 'HOLD', 50.0, reasoning
            
        # 2. MARKET STRESS VETO (Adaptive)
        is_vetoed, veto_reason = self._apply_market_stress_filter(regime, regime_info)
        if final_action == 'BUY' and is_vetoed:
            logger.info(veto_reason + " Missing trade for capital preservation.")
            reasoning += veto_reason + "\n"
            return 'HOLD', 50.0, reasoning

        # 3. ELITE BUY FILTERS (Uptrend Only)
        if final_action == 'BUY':
            # --- PHASE 3 HARDENING (Post-Audit) ---
            indicators = base_analysis.get('indicators', {})
            rsi = indicators.get('rsi', 50)
            ema_50 = indicators.get('ema_50', 0)
            current_price = base_analysis.get('current_price', 0) or base_analysis.get('price', 0)
            
            # Rule 1: Extension Veto (Targeting 70-80% Losers)
            # Losers in this zone are typically overextended (> 5.0% from SMA50).
            if final_confidence < 80.0 and ema_50 > 0:
                dist_sma50 = ((current_price / ema_50) - 1) * 100
                if dist_sma50 > 5.0:
                     msg = f"🛑 EXTENSION VETO: {symbol} is too extended ({dist_sma50:.1f}% > 5.0%) for moderate confidence."
                     logger.info(msg)
                     reasoning += msg + "\n"
                     return 'HOLD', 50.0, reasoning

            # --- PHASE 3: GOLDEN SIGNAL & HARD ML FILTER ---
            # Goal: Maximize precision by enforcing Rule+ML confluence
            if ml_prediction is None:
                ml_prediction = base_analysis.get('ml_prediction', {})
            
            ml_prob_buy = 0.0
            if ml_prediction and 'probabilities' in ml_prediction:
                probs = ml_prediction['probabilities']
                if isinstance(probs, dict):
                    # Handle dictionary format (percentages 0-100 or ratios 0-1)
                    ml_prob_buy = probs.get('BUY', probs.get('buy', 0.0))
                    if ml_prob_buy > 1.0:
                        ml_prob_buy /= 100.0
                elif isinstance(probs, (list, np.ndarray)) and len(probs) > 2:
                    # Handle array format (0: SELL, 1: HOLD, 2: BUY)
                    ml_prob_buy = probs[2]
                    if ml_prob_buy > 1.0:
                        ml_prob_buy /= 100.0
            
            # Rule A: THE GOLDEN SIGNAL (Confluence)
            # If Rules are very strong AND ML agrees, it's a nearly "guaranteed" winner
            if final_confidence >= 80.0 and ml_prob_buy >= 0.60:
                msg = f"✨ GOLDEN SIGNAL: Rule ({final_confidence:.1f}%) and ML ({ml_prob_buy*100:.1f}%) in high agreement."
                logger.info(msg)
                reasoning += msg + "\n"
                # We do not return here, we let it pass through with potentially boosted confidence
                final_confidence = max(final_confidence, 90.0)
            
            # Rule B: THE HARD ML FILTER (Secondary Veto)
            # If Rules approve but ML thinks it's likely a loser/neutral, we VETO
            # Except for "Ultra High Confidence" rules (>95%) which are allowed as safety overrides
            elif final_confidence < 95.0 and ml_prob_buy < 0.40:
                msg = f"🛑 ML UTILITY VETO: ML probability {ml_prob_buy*100:.1f}% too low for Rule {final_confidence:.1f}%."
                logger.info(msg + " Forced HOLD.")
                reasoning += msg + "\n"
                return 'HOLD', 50.0, reasoning
            
            # --------------------------------------

            # Backtest (2025): Even at 90% confidence, win rate is low in stress.
            # We only allow BUYs if the market is stable (Bull/Sideways and low drawdown).
            # [USER-VERIFIED] 75% Confidence has ~70% Accuracy. Locking this in.
            min_threshold = 75.0 # Lowered from 85.0 based on audit
            
            if final_confidence < min_threshold:
                msg = f"🛡️ PROTECTION: Suppressing weak {symbol} BUY ({final_confidence:.1f}% < {min_threshold}%)"
                logger.info(msg)
                reasoning += msg + "\n"
                return 'HOLD', 50.0, reasoning
            else:
                logger.info(f"💎 ELITE TRADE: Executing High-Conviction {symbol} BUY ({final_confidence:.1f}% >= {min_threshold}%)")
                    
        if final_action == 'BUY' and final_confidence > 80.0 and not is_backtest:
            logger.debug(f"⚠️ CALIBRATION CAP: Reducing {final_confidence:.1f}% -> 80.0% (Inversion Safeguard)")
            final_confidence = 80.0
            
        # 4. FINAL PASS
        if final_action != 'HOLD':
            pass_msg = "✅ ALL PRO FILTERS PASSED"
            logger.info(f"{pass_msg} for {symbol}")
            reasoning += pass_msg + "\n"
            
        return final_action, final_confidence, reasoning
    
    
    def _ensemble_predictions(self, rule_pred: Dict, ml_pred: Optional[Dict],
                             weights: Dict, base_analysis: Dict,
                             stock_data: dict = None, history_data: dict = None,
                             is_backtest: bool = False) -> Dict:
        """Combine rule-based and ML predictions with Volatility-Adjusted Targeting"""
        
        # --- QUANT MODE: CALCULATE ATR (Average True Range) ---
        # We calculate this here so it's available for target setting
        atr = 0.0
        atr_target_dist = 0.0
        atr_stop_dist = 0.0
        # Try to get current price from all possible keys
        current_price = (
            rule_pred.get('entry_price', 0) or 
            stock_data.get('price', 0) or 
            stock_data.get('current_price', 0) or 
            stock_data.get('close', 0)
        )
        
        if stock_data and history_data and 'data' in history_data:
            try:
                # Need historical dataframe for ATR
                df = pd.DataFrame(history_data['data'])
                if not df.empty and 'high' in df.columns:
                    df['tr1'] = df['high'] - df['low']
                    df['tr2'] = abs(df['high'] - df['close'].shift())
                    df['tr3'] = abs(df['low'] - df['close'].shift())
                    df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
                    atr = df['tr'].rolling(14).mean().iloc[-1]
                    
                    # --- REGIME-AWARE ATR SCALING ---
                    # Physics (Volatility) dictates reality.
                    # Bull/Sideways: 2.0 * ATR (Standard capture)
                    # Bear/Crash: 1.2 * ATR (Tighter bounces, capturing high-frequency alpha)
                    regime_info = base_analysis.get('market_regime', {})
                    regime = regime_info.get('regime', 'unknown')
                    
                    target_multiplier = 2.0
                    if regime in ['bear', 'crash']:
                        target_multiplier = 1.2
                        logger.debug(f"📉 BEAR ALPHA ACTIVE: Reducing Target Multiplier to {target_multiplier}x ATR")
                    
                    atr_target_dist = target_multiplier * atr
                    atr_stop_dist = 1.5 * atr
                    
                    # Sanity check: If ATR is < 1% of price (Zombie stock), boost min
                    if atr < current_price * 0.01:
                        atr_target_dist = current_price * 0.02
                        atr_stop_dist = current_price * 0.015
            except Exception as e:
                logger.debug(f"ATR Calc Failed: {e}")
        
        # Initialize reasoning from base analysis to avoid UnboundLocalError
        reasoning = base_analysis.get('reasoning', "")
        
        # Determine if ML is available
        has_ml = ml_pred is not None
        
        if not has_ml:
            res = {
                **base_analysis,
                'method': 'rule_based_only',
                'ensemble_weights': {'rule': 1.0, 'ml': 0.0, 'ai': 0.0}
            }
            # Ensure estimated_days is also at top-level for early return
            if 'recommendation' in base_analysis:
                res['estimated_days'] = base_analysis['recommendation'].get('estimated_days', 7)
            return res
        
        rule_action = rule_pred.get('action', 'HOLD')
        rule_confidence = rule_pred.get('confidence', 50)
        ml_action = ml_pred.get('action', 'HOLD')
        ml_confidence = ml_pred.get('confidence', 50)
        
        # Get symbol for logging
        symbol = stock_data.get('symbol', 'Unknown') if stock_data else 'Unknown'
        
        # --- REGIME-AWARE CONTEXT LOGGING ---
        # Logic moves to _apply_pro_filters for final decision making.
        regime_info = base_analysis.get('market_regime', {})
        regime = regime_info.get('regime', 'unknown')
        indicators = base_analysis.get('indicators', {})
        price_above_ema200 = indicators.get('price_above_ema200', True)
        
        logger.info(f"🔍 SIGNAL CONTEXT ({symbol}): ML={ml_action} ({ml_confidence:.1f}%), Regime={regime}, AboveSMA200={price_above_ema200}")
        


        # --- REGIME VETO REMOVED ---
        # Logic moved to _apply_pro_filters for smarter Inversion (Shorting)
        # instead of just downgrading to HOLD.

        
        # Weighted confidence
        # AGGRESSIVE LOGIC UPDATE: Action > Inaction
        # If Rule Analyzer gives an Aggressive Signal (BUY/SELL) and ML gives HOLD (or low conf),
        # we trust the specific rule-based logic (Accumulation/Scale Out) over the generic ML indecision.
        
        rule_weight = weights.get('rule', 0)
        ml_weight = weights.get('ml', 0)
        
        # AGGRESSIVE STRATEGY PIVOT: DISABLE SELL SIGNALS
        # User Instruction: "If SELLs hinder accuracy, remove them."
        # Backtest Result: BUY Alpha = 77.8%, SELL Alpha = 50.6%.
        # Conclusion: SELLs are effectively random. Disabling them to protect capital.
        if ml_action == 'SELL':
             logger.debug(f"🛑 ML SELL Suppressed (Buy-Only High Precision Mode): Converting {ml_confidence:.1f}% SELL to HOLD")
             ml_action = 'HOLD'
             ml_pred['action'] = 'HOLD'
             ml_confidence = 50.0 
             ml_pred['confidence'] = 50.0

        # --- SMART COMPROMISE: REGIME-ADAPTIVE THRESHOLDS ---
        # User Instruction: "In Bull regime, accept ML BUY if > 40% AND Rule Confidence > 80%"
        # This prevents the "Permabear" ML from blocking obvious strong technical setups.
        # PLACEMENT: Must be AFTER "Disable Sell" to catch converted HOLDs.
        # PLACEMENT: Must be AFTER "Disable Sell" to catch converted HOLDs.
        if isinstance(ml_pred, dict):
             probabilities = ml_pred.get('probabilities', {})
             ml_buy_prob = probabilities.get('BUY', 0.0)
             
             # Logic: If ML is "On the fence" (HOLD) but leaning BUY (>40%), and Rules are SURE (BUY > 80%),
             # we nudge the ML to agree.
             if regime == 'bull' and ml_action == 'HOLD' and ml_buy_prob > 40.0:
                 if rule_action == 'BUY' and rule_confidence >= 80.0:
                     logger.info(f"🐂 BULL REGIME OVERRIDE: Promoting ML HOLD ({ml_buy_prob:.1f}%) to BUY due to Strong Rule Signal ({rule_confidence:.1f}%)")
                     ml_action = 'BUY'
                     ml_confidence = ml_buy_prob
                     # Update current local state (ml_pred dict update is optional but good for debugging)
                     ml_pred['action'] = 'BUY' 
                     ml_pred['confidence'] = ml_confidence

        # AGGRESSIVE LOGIC UPDATE: Action > Inaction
        # If Rule Analyzer gives an Aggressive Signal (BUY/SELL) and ML gives HOLD,
        # we normally trust the specific rule-based logic.
        # EXCEPTION: If ML is dominant (> 0.90 weight), we respect its decision to HOLD.
        is_ml_dominant = ml_weight > 0.90
        is_ml_dominant = ml_weight > 0.90
        
        # Flag to force HOLD in stalemate situations
        force_hold = False
        
        # --- BULL MARKET SAFETY LOCK (REMOVED) ---
        # With Triple Barrier models, we trust the model's directional conviction.
        
        # Scenario 1: Consensus (Both agree)
        if rule_action == ml_action:
             if rule_action != 'HOLD':
                 logger.debug(f"✅ Consensus Strong Signal: {rule_action}")
             pass # Standard weights apply
             
        # Scenario 2: Major Conflict (BUY vs SELL)
        elif (rule_action == 'BUY' and ml_action == 'SELL') or (rule_action == 'SELL' and ml_action == 'BUY'):
             # If they fight, it's risky. Only allow if one is SUPER confident.
             # Otherwise, force a STALEMATE (HOLD) to save capital.
             if ml_confidence > 80 and is_ml_dominant:
                 logger.debug(f"⚔️ Conflict (R:{rule_action} vs M:{ml_action}) -> ML Wins (High Conf {ml_confidence:.1f}%)")
             elif rule_confidence > 80 and not is_ml_dominant:
                 logger.debug(f"⚔️ Conflict (R:{rule_action} vs M:{ml_action}) -> Rule Wins (High Conf {rule_confidence:.1f}%)")
             else:
                 # Stalemate
                 # CRITICAL EXCEPTION: Reversal Detection
                 # If Rule says SELL (Downtrend) but ML says BUY (Reversal),
                 # check if we have technical confirmation for a bottom.
                 is_reversal_setup = False
                 reversal_reason = ""
                 
                 rsi = base_analysis.get('indicators', {}).get('rsi_14', 50)
                 patterns = base_analysis.get('indicators', {}).get('pattern_analysis', {})
                 candlestick = base_analysis.get('indicators', {}).get('candlestick_pattern') or 'unknown'
                 
                 # 0. TOPOLOGY CHECK (New Feature)
                 # Check for "Mispricing" via Laplacian Residual
                 # Negative Residual = Lagging behind cluster = Potential BUY (Reversion/Catch-up)
                 laplacian_score = base_analysis.get('price_data', {}).get('laplacian_score', 0)
                 # Sometimes stock_data is merged into analysis differently, let's check input
                 # (In practice, stock_data features often don't persist to here unless explicitly passed)
                 # Wait, 'base_analysis' comes from TradingAnalyzer.analyze.
                 # We need to make sure 'laplacian_score' survives. 
                 # The 'stock_data' passed to predict() had it. 
                 # We didn't modify TradingAnalyzer to pass it through.
                 # Workaround: We can't easily access stack_data here without changing signatures.
                 # Let's assume it might not be here.
                 
                 # ... wait, we can modify 'predict' to attach it to 'base_analysis' before calling this.
                 
                 # 1. RSI Oversold Check
                 if ml_action == 'BUY' and rsi < 35: 
                     is_reversal_setup = True
                     reversal_reason = f"RSI Oversold ({rsi:.1f})"
                     
                 # 2. Pattern Check
                 if ml_action == 'BUY' and ('engulfing' in candlestick or 'hammer' in candlestick or 'double_bottom' in str(patterns)):
                     is_reversal_setup = True
                     reversal_reason = f"Bullish Pattern ({candlestick})"
                 
                 if is_reversal_setup and ml_confidence > 60:
                      logger.debug(f"🔄 Reversal Exception: Trusting ML BUY ({ml_confidence:.1f}%) due to {reversal_reason}")
                      # Boost ML weight to override Rule
                      ml_weight = max(ml_weight, 0.65)
                      rule_weight = 1.0 - ml_weight
                      force_hold = False
                      reasoning += f"🔄 REVERSAL DETECTED: {reversal_reason}. Trusting ML Signal.\n"
                 else:
                      force_hold = True
                      logger.debug(f"⚔️ Conflict (R:{rule_action} vs M:{ml_action}) -> FORCE HOLD (Stalemate)")
                 
        # Scenario 3: Action vs HOLD (One wants to move, one wants to wait)
        elif rule_action != ml_action:
             # Calculate Risk Environment
             volatility = base_analysis.get('volatility', {})
             rsi = base_analysis.get('indicators', {}).get('rsi_14', 50)
             
             # Determine Risk Level
             is_high_risk = False
             risk_reason = ""
             
             # Determine Risk Level
             is_high_risk = False
             risk_reason = ""
             
             # Risk Factor 1: Extreme RSI (Overbought/Oversold Reversal Risk)
             if ml_action == 'BUY' and rsi > 75: # Loose limit for Quant mode
                 is_high_risk = True
                 risk_reason = f"Extreme RSI ({rsi:.1f})"
             elif ml_action == 'SELL' and rsi < 35:
                 is_high_risk = True
                 risk_reason = f"Low RSI ({rsi:.1f})"
                 
             # Risk Factor 2: High Volatility (ATR Check)
             atr_percent = volatility.get('atr_percent', 0)
             if atr_percent > 4.0: # >4% daily move is very volatile
                 is_high_risk = True
                 risk_reason = f"High Volatility ({atr_percent:.1f}%)"
                 
             if is_high_risk:
                 # Safety First: In high risk, if ANYONE says HOLD, we HOLD.
                 if rule_action == 'HOLD' or ml_action == 'HOLD':
                     logger.debug(f"🛑 Risk Veto: {risk_reason} -> Forcing HOLD despite signal")
                     force_hold = True
             else:
                 # Low Risk Environment: Allow Action to override Hold ("Smart Aggression")
                 # This resolves the "Stalemate" problem in safe markets
                 if ml_action != 'HOLD' and ml_confidence > 55:
                      logger.debug(f"🚀 Smart Aggression: Low Risk (RSI {rsi:.0f}) -> Trusting ML {ml_action}")
                      # Allow ML to proceed (don't force hold)
                 elif rule_action != 'HOLD' and rule_confidence > 60:
                      logger.debug(f"🚀 Smart Aggression: Low Risk -> Trusting Rule {rule_action}")

             if is_ml_dominant and ml_confidence > 85: # Require HIGH confidence to force HOLD
                 # ML is the boss. If it says HOLD, we HOLD.
                 logger.debug(f"🛡️ ML Dominant ({ml_weight:.2f}): FORCE HOLD - Zeroing Rule Weight")
                 rule_weight = 0.0
                 ml_weight = 1.0
                 force_hold = True 
             else:
                 # CONVICTION ALIGNMENT: If Rules say Go and ML says Meh -> Depend on Weight.
                logger.debug(f"🚀 Consensus Check: Rule says {rule_action}, ML says HOLD.")
                # Respect the original weights (don't force a bias)
             
         # Scenario 4: Safety Veto (Rule says HOLD, ML says Action)
        elif rule_action == 'HOLD' and ml_action != 'HOLD':
             # SAFETY VETO: Prevent ML from blindly buying into extreme overbought conditions
             rsi = base_analysis.get('indicators', {}).get('rsi', 50)
             # Relaxed threshold: Only block if RSI is > 82 (Extreme)
             if ml_action == 'BUY' and rsi > 82 and ml_confidence < 90:
                 logger.info(f"🛡️ SAFETY VETO: Blocking ML BUY because RSI is {rsi:.2f} (Extreme Overbought). Forced HOLD.")
                 # Force Rule weight to dominate (1.0) to enforce the HOLD
                 rule_weight = 1.0
                 ml_weight = 0.0
                 force_hold = True
             
             elif ml_confidence > 60:  # If ML is somewhat confident (>60%), let it speak
                 # If ML is reasonably confident, let it swing the vote
                 pass 
             else:
                 # If ML is weak and Rule is Unsure, stay Safe (HOLD)
                 pass
                 logger.debug(f"⚖️ Boosting Rule weight for Safety (Rule: HOLD, ML: Weak {ml_action})")
                 rule_weight = max(rule_weight, 0.6)
                 ml_weight = 1.0 - rule_weight
            
        # CONFIDENCE LEAK FIX:
        # If one model says HOLD and other says ACTION, treat the HOLD
        # confidence as 50.0 (Neutral) for the averaging.
        # This prevents "High Confidence HOLD" from artificially boosting a BUY signal.
        calc_rule_conf = rule_confidence
        calc_ml_conf = ml_confidence
        
        if rule_action != 'HOLD' and ml_action == 'HOLD':
             calc_ml_conf = 50.0
        elif rule_action == 'HOLD' and ml_action != 'HOLD':
             calc_rule_conf = 50.0

        final_confidence = (
            calc_rule_conf * rule_weight + 
            calc_ml_conf * ml_weight
        )
        
        # Action selection (weighted confidence wins)
        action_counts = {}
        action_counts[rule_action] = action_counts.get(rule_action, 0) + rule_weight
        action_counts[ml_action] = action_counts.get(ml_action, 0) + ml_weight
        
        # Determine ensemble action first to avoid UnboundLocalError in veto layer
        ensemble_action = max(action_counts.items(), key=lambda x: x[1])[0]
        
        # Phase 3 Meta-Labeling: Secondary Signal Veto Layer
        # ---------------------------------------------------
        # Even if the ensemble agrees, we consult a second 'brain' (trained on path-dependent correctness).
        meta_confidence = ml_pred.get('meta_confidence', 100.0)
        
        if not force_hold and ensemble_action != 'HOLD':
            # Block if success probability is critically low (<50%)
            if meta_confidence < 50.0:
                logger.info(f"🛡️ META-VETO: Blocking {ensemble_action} for {stock_data.get('symbol')} (Success Prob: {meta_confidence:.1f}%)")
                force_hold = True
                reasoning += f"🛡️ META-VERIFIER: Signal blocked (Success probability: {meta_confidence:.1f}%)\n"
            elif meta_confidence > 80.0:
                reasoning += f"✨ META-VERIFIER: Signal Verified (Success probability: {meta_confidence:.1f}%)\n"
            else:
                reasoning += f"✓ META-VERIFIER: Success probability: {meta_confidence:.1f}%\n"

        # Final action selection
        if force_hold:
            final_action = 'HOLD'
            final_confidence = 50.0 
        else:
            final_action = ensemble_action
        
        # Boost confidence if they agree and not forced hold
        if rule_action == ml_action and not force_hold:
            final_confidence = min(95, final_confidence * 1.1)
            
        # TOPOLOGY BOOST
        # If Laplacian Score confirms the action, boost confidence
        laplacian_score = base_analysis.get('laplacian_score', 0)
        topology_boost = False
        if laplacian_score != 0:
            # Negative Score = "Lagging" = Bullish Reversion
            if final_action == 'BUY' and laplacian_score < -0.01:
                boost_mult = 1.0 + (abs(laplacian_score) * 5) # e.g. -0.02 -> +10%
                boost_mult = min(1.2, boost_mult)
                final_confidence = min(98, final_confidence * boost_mult)
                topology_boost = True
                reasoning += f"🌐 TOPOLOGY SIGNAL: Validated by Laplacian Residual ({laplacian_score:.4f} - Lagging Peers)\n"
            
            # Positive Score = "Running Hot" = Bearish Reversion
            elif final_action == 'SELL' and laplacian_score > 0.01:
                boost_mult = 1.0 + (abs(laplacian_score) * 5)
                boost_mult = min(1.2, boost_mult)
                final_confidence = min(98, final_confidence * boost_mult)
                topology_boost = True
                reasoning += f"🌐 TOPOLOGY SIGNAL: Validated by Laplacian Residual ({laplacian_score:.4f} - Overheating)\n"
             
            # Conflict check: If buying into specific peer-pressure (Buying a Stock that is already Running Hot relative to peers)
            elif final_action == 'BUY' and laplacian_score > 0.02:
                reasoning += f"⚠️ TOPOLOGY WARNING: Buying against Laplacian Pressure (Score {laplacian_score:.4f})\n"
                final_confidence *= 0.9 # Penalize
        
        # Use rule-based prices (they're more detailed)
        recommendation = base_analysis.get('recommendation', {})
        
        # Generate reasoning
        reasoning = base_analysis.get('reasoning', '')
        reasoning += f"\n\n🤖 HYBRID ANALYSIS (Rules + ML):\n"
        reasoning += f"Rule-based: {rule_action} ({rule_confidence:.1f}%)\n"
        reasoning += f"ML Model: {ml_action} ({ml_confidence:.1f}%)\n"
        
        weight_parts = [f"Rule {weights.get('rule', 0)*100:.0f}%"]
        if weights.get('ml', 0) > 0:
            weight_parts.append(f"ML {weights.get('ml', 0)*100:.0f}%")
        reasoning += f"Ensemble Weights: {' / '.join(weight_parts)}\n"
        reasoning += f"Final Prediction: {final_action} ({final_confidence:.1f}% confidence)\n"
        
        # Agreement analysis
        if rule_action == ml_action:
            reasoning += "✓ Both methods agree - high confidence\n"
        else:
            reasoning += "⚠ Methods differ - using weighted consensus\n"
        
        result = base_analysis.copy()
        
        # --- CRITICAL FIX: RECALCULATE TARGET/STOP IF ACTION FLIPPED ---
        # If the final_action (weighted consensus) differs from the rule-based action,
        # we MUST flip the target and stop loss to avoid "Accuracy Fraud" (instant verification).
        final_target = recommendation.get('target_price', 0)
        final_stop = recommendation.get('stop_loss', 0)
        current_price = (
            base_analysis.get('price', 0) or 
            stock_data.get('price', 0) or 
            stock_data.get('current_price', 0) or 
            stock_data.get('close', 0)
        )
        
        if final_action != rule_action:
            # Detect ATR for dynamic offset
            atr = base_analysis.get('indicators', {}).get('atr', 0)
            
            # Fallback if ATR is missing or invalid (prevent 0 target)
            if atr <= 0 and current_price > 0:
                atr = current_price * 0.02  # Assume 2% daily volatility as fallback
                logger.warning(f"⚠️ ATR missing/zero for {symbol}, using 2% fallback: {atr:.2f}")
            if current_price > 0:
                if final_action == 'BUY':
                    # Flip to BUY levels
                    offset = atr * 3 if atr > 0 else current_price * 0.05
                    final_target = current_price + offset
                    final_stop = current_price - (offset / 2)
                    reasoning += f"🎯 Adjusted Target/Stop for BUY flip (ATR offset)\n"
                elif final_action == 'SELL':
                    # Flip to SELL levels
                    offset = atr * 3 if atr > 0 else current_price * 0.05
                    final_target = current_price - offset
                    final_stop = current_price + (offset / 2)
                    reasoning += f"🎯 Adjusted Target/Stop for SELL flip (ATR offset)\n"
                elif final_action == 'HOLD':
                    final_target = current_price
                    final_stop = current_price
        
        # === DYNAMIC TARGET PRICING Integration ===
        # GOAL: Maximize Profit % (Conservative but realistic, NOT capped at 3%)
        # Minimum: 3.0% (Hard Floor)
        # Maximum: Determined by Volatility (ATR), Technical Levels, and ML Confluence
        # Strategy: Use CONSERVATIVE MEDIAN of all valid targets
        
        # 1. Start with the Base Technical Target from Rule Analysis
        base_target = final_target # From support/resistance
        
        # 2. Calculate Volatility-Based Potentials
        # Conservative: 2.0x ATR
        # Moderate: 3.0x ATR  
        # Sanity Cap: 5.0x ATR (absolute max)
        atr_target_conservative = current_price + (2.0 * atr) if final_action == 'BUY' else current_price - (2.0 * atr)
        atr_target_moderate = current_price + (3.0 * atr) if final_action == 'BUY' else current_price - (3.0 * atr)
        atr_target_max = current_price + (5.0 * atr) if final_action == 'BUY' else current_price - (5.0 * atr)
        
        # 3. Get ML Target (Regression)
        ml_target_price = 0.0
        if ml_pred and 'target_price' in ml_pred:
             ml_target_price = ml_pred['target_price']

        # 4. Determine Dynamic Target
        # Collect ALL realistic target candidates
        
        potential_targets = []
        
        # A. Technical Resistance (Often the most reliable)
        if base_target > 0:
            potential_targets.append(('Technical', base_target))
            
        # B. Volatility Extension (Physics)
        potential_targets.append(('ATR_2x', atr_target_conservative))
        potential_targets.append(('ATR_3x', atr_target_moderate))
        
        # C. ML Regression (Statistical)
        if final_action == 'BUY' and ml_target_price > current_price:
            potential_targets.append(('ML_Regression', ml_target_price))
        elif final_action == 'SELL' and ml_target_price < current_price:
            potential_targets.append(('ML_Regression', ml_target_price))
            
        # Select target using CONSERVATIVE MEDIAN approach:
        # - Use all targets (no arbitrary 3x ATR cap)
        # - Pick the MEDIAN target (conservative: not min, not max)
        # - Sanity cap: must be within 5x ATR of current price
        
        best_target = base_target
        best_source = "Technical"
        
        from config import MIN_PROFIT_TARGET_PCT
        
        if final_action == 'BUY':
             # Sort ascending to find median
             sorted_vals = sorted(potential_targets, key=lambda x: x[1])
             
             # Sanity filter: remove anything beyond 5x ATR (unrealistic)
             sane_targets = [(s, t) for s, t in sorted_vals if t <= atr_target_max]
             if not sane_targets:
                 sane_targets = [('ATR_Max_Clamp', atr_target_max)]
             
             # Pick the MEDIAN (conservative middle ground)
             mid_idx = len(sane_targets) // 2
             best_source, best_target = sane_targets[mid_idx]
              
             # Enforce 3% FLOOR (never below, but allow above)
             min_threshold = current_price * (1 + (MIN_PROFIT_TARGET_PCT / 100.0))
             
             if best_target < min_threshold:
                 # If 3% is achievable within 5x ATR, enforce it as floor
                 if min_threshold <= atr_target_max:
                     best_target = min_threshold
                     best_source = "Min_3pct_Floor"
                 else:
                     # 3% is not realistic for this stock's current volatility
                     logger.info(f"🛡️ SNIPER VETO: {symbol} cannot realistically hit {MIN_PROFIT_TARGET_PCT}% target (MaxATR={atr_target_max:.2f})")
                     final_action = 'HOLD'
                     final_confidence = 50.0
                     reasoning += f"🛡️ SNIPER VETO: Max realistic target ({atr_target_max:.2f}) is below the {MIN_PROFIT_TARGET_PCT}% mandatory threshold.\n"

             # Finalize Patient Sniper BUY parameters
             final_target = best_target
             final_stop = 0.0  # PATIENT SNIPER: No stop-loss interference
             
             # Calculate achieved profit for logging
             achieved_pct = ((best_target - current_price) / current_price) * 100 if current_price > 0 else 0
             reasoning += f"🎯 PATIENT SNIPER: Target {final_target:.2f} ({best_source}, +{achieved_pct:.1f}%), No Stop-Loss.\n"
              
        elif final_action == 'SELL':
             # Keep existing SELL logic (inverted, conservative)
             sorted_targets = sorted(potential_targets, key=lambda x: x[1], reverse=False)
             sane_targets = [(s, t) for s, t in sorted_targets if t >= (current_price - (5.0 * atr))]
             if not sane_targets:
                 sane_targets = [('ATR_Max_Clamp', atr_target_max)]
             
             mid_idx = len(sane_targets) // 2
             best_source, best_target = sane_targets[mid_idx]
             final_target = best_target
             final_stop = current_price + (abs(current_price - best_target) / 2)
        
        # DEBUG TRACE
        logger.info(f"🔍 TGT TRACE {symbol}: ATR={atr:.2f} | Tech={base_target:.2f} | ML={ml_target_price:.2f}")
        logger.info(f"🔍 TGT TRACE {symbol}: Candidates={potential_targets}")
        logger.info(f"🔍 TGT TRACE {symbol}: Selected={best_source} @ {best_target:.2f}")
        
        # 5. VERIFY MINIMUM PROFIT THRESHOLD (Final safety net)
        calculated_profit_pct = 0.0
        if current_price > 0:
            if final_action == 'BUY':
                calculated_profit_pct = ((best_target - current_price) / current_price) * 100
            elif final_action == 'SELL':
                calculated_profit_pct = ((current_price - best_target) / current_price) * 100
        
        # If the BEST realistic target is still less than our minimum 3%, we discard the trade
        if calculated_profit_pct < MIN_PROFIT_TARGET_PCT and final_action != 'HOLD':
            logger.info(f"📉 REJECTING {symbol}: Best Target {calculated_profit_pct:.2f}% < {MIN_PROFIT_TARGET_PCT}% Minimum")
            final_action = 'HOLD'
            reasoning += f"📉 Target Too Low: Best calculated target ({best_source}: {calculated_profit_pct:.2f}%) is below {MIN_PROFIT_TARGET_PCT}% minimum.\n"
        else:
            reasoning += f"🎯 Target Selected: {best_source} (+{calculated_profit_pct:.1f}%)\n"
            
        final_target = best_target
        
        # Recalculate Stop Loss to maintain reasonable R:R (at least 1:3 if possible, min 1:1)
        # Users hate tight stops getting hit on volatility noise.
        # We ensure stop is at least 1.5% away or 1.0x ATR.
        min_stop_dist = current_price * 0.015
        if atr > 0:
            volatility_stop_dist = 1.0 * atr # Tighter trailing stop start
            actual_stop_dist = max(min_stop_dist, volatility_stop_dist)
        else:
            actual_stop_dist = min_stop_dist
            
        if final_action == 'BUY':
            final_stop = current_price - actual_stop_dist
        elif final_action == 'SELL':
            final_stop = current_price + actual_stop_dist

        # RELAXED SNIPER MODE: Lower threshold for path-dependent models
        # QUANT MODE UPDATE: Trust the probability. IF > 50%, it's an edge.
        SNIPER_THRESHOLD = 50.0
        
        if final_action == 'BUY' and final_confidence < SNIPER_THRESHOLD:
            logger.info(f"🎯 SNIPER MODE: BUY at {final_confidence:.1f}% below {SNIPER_THRESHOLD}% threshold → HOLD")
            final_action = 'HOLD'
            reasoning += f"🎯 SNIPER MODE: Only taking trades with ≥{SNIPER_THRESHOLD}% confidence\n"
        elif final_action == 'BUY':
             logger.info(f"🎯 SNIPER MODE: Trade Accepted ({final_confidence:.1f}% > {SNIPER_THRESHOLD}%)")

        # === SIGNAL CONTEXT TAGGING ===
        # Help the user identify high-accuracy scenarios (Reversal vs Momentum)
        # Apply even if sniped to HOLD, so user knows what the "potential" trade was.
        context = "Standard Signal"
        if final_action == 'BUY' or (rule_action == 'BUY' and final_action == 'HOLD'):
            context = self._determine_signal_context(stock_data, history_data, base_analysis)
            reasoning = f"[{context}] " + reasoning

        # Fetch est days from wherever it ended up
        est_days = result.get('estimated_days') or recommendation.get('estimated_days') or 7

        # APPLY STATISTICAL CALIBRATION (Final Realism Filter)
        if final_action != 'HOLD':
            atr = base_analysis.get('indicators', {}).get('atr', 0)
            
            calibration = self.statistical_calibrator.calibrate(
                current_price, final_target, est_days, atr, market_regime=base_analysis.get('market_regime', 'unknown')
            )
            
            if not calibration['is_realistic']:
                final_target = calibration['target_price']
                reasoning += f"⚖️ Statistical Realism: {calibration['reason']}\n"

        # QUANT MODE: Apply Elite Long-Only Filters (Veto weak signals and all shorts)
        # Goal: Extreme Accuracy (>60%) by accepting lower trade frequency.
        final_action, final_confidence, reasoning = self._apply_pro_filters(
            final_action, final_confidence, stock_data, history_data, base_analysis, reasoning, ml_pred,
            is_backtest=is_backtest
        )

        result['recommendation'] = {
            **recommendation,
            'action': final_action,
            'confidence': float(final_confidence),
            'target_price': float(final_target),
            'stop_loss': float(final_stop),
            'estimated_days': int(est_days), # Use the dynamic est_days from calibration check a few lines above
            'context': context
        }
        result['reasoning'] = reasoning
        result['method'] = 'hybrid_ensemble'
        result['ml_enhanced'] = True
        # CRITICAL FIX: Ensure dictionary vs object consistency for regime
        regime_data = base_analysis.get('market_regime') or base_analysis.get('macro_info') or 'unknown'
        result['market_regime'] = regime_data
        
        # --- TIME HORIZON ENHANCEMENT ---
        # Ensure estimated_days and estimated_target_date are in the top-level
        target_rec = result.get('recommendation', {})
        est_days = target_rec.get('estimated_days') or result.get('estimated_days') or base_analysis.get('estimated_days') or 7
        result['estimated_days'] = est_days
        
        # Ensure it's inside the recommendation dict too (for Scanner/UI)
        if 'recommendation' in result:
             result['recommendation']['estimated_days'] = est_days

        if 'estimated_target_date' not in result or not result['estimated_target_date']:
            from datetime import timedelta
            # Use provided current_date (backtest) or now (live)
            start_date = base_analysis.get('current_date') or datetime.now()
            # If start_date is string, convert it
            if isinstance(start_date, str):
                try:
                    start_date = datetime.fromisoformat(start_date)
                except:
                    start_date = datetime.now()
            
            target_date = start_date + timedelta(days=int(est_days))
            result['estimated_target_date'] = target_date.isoformat()
            
        # Ensure target date is also in recommendation for dashboard symmetry
        if 'recommendation' in result:
             result['recommendation']['estimated_target_date'] = result['estimated_target_date']

        result['ensemble_weights'] = weights
        result['rule_prediction'] = {
            'action': rule_action,
            'confidence': rule_confidence
        }
        result['ml_prediction'] = {
            'action': ml_action,
            'confidence': ml_confidence,
            'probabilities': ml_pred.get('probabilities', {})
        }
        
        # Standardize reasoning into a list for the UI
        if 'recommendation' in result:
            result['recommendation']['reasoning'] = [line.strip() for line in reasoning.split('\n') if line.strip()]
        
        return result
    
    def _get_recent_performance(self, method: str) -> Optional[Dict]:
        """Get recent performance metrics"""
        history = self.performance_history.get(method, [])
        if not history:
            return None
        
        recent = history[-ML_RECENT_PERFORMANCE_WINDOW:] if len(history) > ML_RECENT_PERFORMANCE_WINDOW else history
        correct = sum(1 for h in recent if h.get('was_correct') is True)
        total = len(recent)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'total': total,
            'correct': correct
        }
    
    def update_performance(self, prediction: Dict, actual_outcome: bool) -> None:
        """Update performance tracking after prediction is verified"""
        method = prediction.get('method', 'hybrid')
        
        if method == 'hybrid_ensemble':
            rule_pred = prediction.get('rule_prediction', {})
            ml_pred = prediction.get('ml_prediction', {})
            
            if rule_pred:
                self.performance_history['weight_based'].append({
                    'was_correct': actual_outcome,
                    'confidence': rule_pred.get('confidence', 50),
                    'date': datetime.now().isoformat()
                })
            
            if ml_pred:
                self.performance_history['ml_based'].append({
                    'was_correct': actual_outcome,
                    'confidence': ml_pred.get('confidence', 50),
                    'date': datetime.now().isoformat()
                })
            
            self.performance_history['hybrid'].append({
                'was_correct': actual_outcome,
                'confidence': prediction.get('recommendation', {}).get('confidence', 50),
                'date': datetime.now().isoformat()
            })
            
            self._save_performance()
            
            # Update ensemble weights based on performance
            self._update_ensemble_weights()
    
    def _update_ensemble_weights(self) -> None:
        """Update ensemble weights based on recent performance"""
        rule_perf = self._get_recent_performance('weight_based')
        ml_perf = self._get_recent_performance('ml_based')
        
        # Two-way comparison (rule vs ML)
        if rule_perf and ml_perf and rule_perf.get('total', 0) >= 10:
            rule_acc = rule_perf.get('accuracy', 50)
            ml_acc = ml_perf.get('accuracy', 50)
            
            if abs(rule_acc - ml_acc) > ML_ENSEMBLE_WEIGHT_UPDATE_THRESHOLD:
                if rule_acc > ml_acc:
                    self.ensemble_weights['rule'] = min(ML_ENSEMBLE_WEIGHT_MAX, self.ensemble_weights.get('rule', 0.5) + 0.05)
                    self.ensemble_weights['ml'] = 1.0 - self.ensemble_weights['rule']
                    self.ensemble_weights['ai'] = 0.0
                else:
                    self.ensemble_weights['ml'] = min(ML_ENSEMBLE_WEIGHT_MAX, self.ensemble_weights.get('ml', 0.5) + 0.05)
                    self.ensemble_weights['rule'] = 1.0 - self.ensemble_weights['ml']
                    self.ensemble_weights['ai'] = 0.0
                
                self._save_ensemble_weights()

