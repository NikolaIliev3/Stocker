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

# Import Sector ML if available
try:
    from sector_ml import SectorMLManager, SECTOR_DISPLAY_NAMES
    from sector_mapper import SectorMapper
    HAS_SECTOR_ML = True
except ImportError:
    HAS_SECTOR_ML = False

# Import configuration
from config import (ML_HIGH_ACCURACY_THRESHOLD, ML_VERY_HIGH_ACCURACY_THRESHOLD,
                   ML_PERFORMANCE_ACCURACY_DIFF_THRESHOLD, ML_CONFIDENCE_DIFF_THRESHOLD_NORMAL,
                   ML_PERFORMANCE_HISTORY_LIMIT, ML_RECENT_PERFORMANCE_WINDOW,
                   ML_ENSEMBLE_WEIGHT_UPDATE_THRESHOLD, ML_ENSEMBLE_WEIGHT_MAX,
                   AI_PREDICTOR_ENABLED, LLM_AI_PREDICTOR_ENABLED)

logger = logging.getLogger(__name__)


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
                            market_regime, timeframe_analysis, relative_strength, support_resistance
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
                    None, None, None, volume_analysis
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
                ml_prediction = self.ml_predictor.predict(features, stock_data, history_data, is_backtest=is_backtest)
                # Check if prediction has an error (e.g., LabelEncoder not fitted)
                if ml_prediction and 'error' in ml_prediction:
                    logger.warning(f"⚠️ ML prediction unavailable: {ml_prediction.get('error')}. Using rule-based only.")
                    ml_available = False
                    ml_prediction = None
                else:
                    # Diagnostic logging for backtesting (use INFO for visibility)
                    if ml_prediction:
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
            rule_prediction, ml_prediction, ensemble_weights, base_analysis
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
        
        # Store in cache for future use
        if symbol:
            cache = get_cache()
            cache.set(symbol, self.strategy, final_prediction)
            logger.debug(f"💾 Cached prediction for {symbol} ({self.strategy})")
        
        return final_prediction
    
    def _apply_learned_weights(self, analysis: Dict) -> Dict:
        """Apply learned weights to rule-based analysis"""
        # This is simplified - in practice, you'd modify the analyzer
        # to use learned weights. For now, we'll use the base analysis.
        recommendation = analysis.get('recommendation', {}).copy()
        return {
            'action': recommendation.get('action', 'HOLD'),
            'confidence': recommendation.get('confidence', 50),
            'entry_price': recommendation.get('entry_price', 0),
            'target_price': recommendation.get('target_price', 0),
            'stop_loss': recommendation.get('stop_loss', 0)
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
            trained_accuracy = self.ml_predictor.metadata.get('test_accuracy', 0) * 100
            
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
            source = "backtest_history"
        else:
            effective_accuracy = trained_accuracy
            source = "global_training"
            
        # Apply Logic based on Effective Accuracy
        if effective_accuracy > 60:
            ml_weight = 0.65 # High trust
            rule_weight = 0.35
            logger.debug(f"⚖️ Boosting ML Weight (0.65) due to strong {source} ({effective_accuracy:.1f}%)")
        elif effective_accuracy > 55:
            ml_weight = 0.60 # Moderate trust
            rule_weight = 0.40
        elif effective_accuracy < 50:
            ml_weight = 0.35 # Penalize
            rule_weight = 0.65
            logger.debug(f"⚖️ Reducing ML Weight (0.35) due to weak {source} ({effective_accuracy:.1f}%)")
        
        # If no strong signal, stick to 50/50 default set above
        # If no strong signal, stick to 50/50 default set above
            # No history yet - trust training accuracy
            # CRITICAL ADJUSTMENT: Trust ML Training initially to break "Permabear" cycle
            # CRITICAL ADJUSTMENT: Trust ML Training initially to break "Permabear" cycle
            # Lowered threshold to 53% (better than random) and increased weight to 0.45 (near parity)
            if trained_accuracy > 53: 
                 logger.debug(f"⚖️ Trusting New Brain: No history, but Training Acc {trained_accuracy:.1f}% -> ML Weight 0.55")
                 ml_weight = 0.55
                 rule_weight = 0.45
            else:
                 # Even if weak, don't crush it completely (0.35 -> 0.40)
                 ml_weight = 0.40
                 rule_weight = 0.60
        
        # DISAGREEMENT PENALTY (Disabled for Strong Models)
        # Only apply disagreement penalty if the model is NOT Strong (accuracy > 60%)
        rule_action = rule_pred.get('action', 'HOLD')
        ml_action = ml_pred.get('action', 'HOLD')
        
        is_strong_model = effective_accuracy > 60
        
        if rule_action != ml_action and not is_strong_model:
            # If model is weak/moderate and disagrees with rules, limit its influence
            if effective_accuracy < 52:
                ml_weight = min(ml_weight, 0.40)
                rule_weight = 1.0 - ml_weight
        
        return {'rule': rule_weight, 'ml': ml_weight, 'ai': 0.0}
    
    def _ensemble_predictions(self, rule_pred: Dict, ml_pred: Optional[Dict],
                             weights: Dict, base_analysis: Dict) -> Dict:
        """Combine rule-based and ML predictions"""
        # Determine if ML is available
        has_ml = ml_pred is not None
        
        if not has_ml:
            return {
                **base_analysis,
                'method': 'rule_based_only',
                'ensemble_weights': {'rule': 1.0, 'ml': 0.0, 'ai': 0.0}
            }
        
        rule_action = rule_pred.get('action', 'HOLD')
        rule_confidence = rule_pred.get('confidence', 50)
        ml_action = ml_pred.get('action', 'HOLD')
        ml_confidence = ml_pred.get('confidence', 50)
        
        # Weighted confidence
        # AGGRESSIVE LOGIC UPDATE: Action > Inaction
        # If Rule Analyzer gives an Aggressive Signal (BUY/SELL) and ML gives HOLD (or low conf),
        # we trust the specific rule-based logic (Accumulation/Scale Out) over the generic ML indecision.
        
        rule_weight = weights.get('rule', 0)
        ml_weight = weights.get('ml', 0)
        
        # AGGRESSIVE LOGIC UPDATE: Action > Inaction
        # If Rule Analyzer gives an Aggressive Signal (BUY/SELL) and ML gives HOLD,
        # we normally trust the specific rule-based logic.
        # EXCEPTION: If ML is dominant (> 0.90 weight), we respect its decision to HOLD.
        is_ml_dominant = ml_weight > 0.90
        
        # Flag to force HOLD in stalemate situations
        force_hold = False
        
        # --- BULL MARKET SAFETY LOCK ---
        # Prevent "Inversions" (Selling winners) by downgrading weak ML Sells in confirmed Uptrends.
        market_regime = base_analysis.get('market_regime', {}).get('regime', 'unknown')
        trend = base_analysis.get('price_action', {}).get('trend', 'sideways')
        
        if market_regime == 'bull' and trend == 'uptrend' and ml_action == 'SELL':
             if ml_confidence < 85.0: # If ML is not SUPER sure (85%), don't sell a winner
                 logger.info(f"🛡️ Safety Lock: Downgrading ML SELL ({ml_confidence:.1f}%) to HOLD in Bull Uptrend")
                 ml_action = 'HOLD'
                 ml_pred['action'] = 'HOLD'
                 # If Rule Action is BUY, this allows the BUY to win (BUY vs HOLD = BUY)
                 # If Rule Action is HOLD, this results in HOLD (HOLD vs HOLD = HOLD)
                 # This effectively blocks the "Inversion" error.
        
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
             
             # Risk Factor 1: Extreme RSI (Overbought/Oversold Reversal Risk)
             if ml_action == 'BUY' and rsi > 65:
                 is_high_risk = True
                 risk_reason = f"High RSI ({rsi:.1f})"
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
                 # ACTION BIAS: If Rules say Go and ML says Meh -> GO.
                 logger.debug(f"🚀 ACTION BIAS: Rule says {rule_action}, ML says HOLD. Trusting Rule.")
                 # Force Rule to have majority to pass the signal through
                 rule_weight = max(rule_weight, 0.65) 
                 ml_weight = 1.0 - rule_weight
             
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
            
        final_confidence = (
            rule_confidence * rule_weight + 
            ml_confidence * ml_weight
        )
        
        # Action selection (weighted confidence wins)
        action_counts = {}
        action_counts[rule_action] = action_counts.get(rule_action, 0) + rule_weight
        action_counts[ml_action] = action_counts.get(ml_action, 0) + ml_weight
        
        # Get action with highest weighted count
        if force_hold:
            final_action = 'HOLD'
            # Ensure confidence reflects the safety decision (neutral but firm)
            final_confidence = 50.0 
        else:
            final_action = max(action_counts.items(), key=lambda x: x[1])[0]
        
        # Boost confidence if they agree and not forced hold
        if rule_action == ml_action and not force_hold:
            final_confidence = min(95, final_confidence * 1.1)
        
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
        current_price = base_analysis.get('price', 0)
        
        if final_action != rule_action:
            # Detect ATR for dynamic offset
            atr = base_analysis.get('indicators', {}).get('atr', 0)
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

        result['recommendation'] = {
            **recommendation,
            'action': final_action,
            'confidence': final_confidence,
            'target_price': final_target,
            'stop_loss': final_stop
        }
        result['reasoning'] = reasoning
        result['method'] = 'hybrid_ensemble'
        result['ml_enhanced'] = True
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

