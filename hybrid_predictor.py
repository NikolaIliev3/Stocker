"""
Hybrid Stock Predictor
Combines rule-based weight adjustment with ML for optimal predictions
"""
import json
import numpy as np
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
                 seeker_ai = None):
        self.data_dir = data_dir
        self.strategy = strategy
        
        # Initialize components
        self.weight_adjuster = AdaptiveWeightAdjuster(data_dir, strategy)
        self.ml_predictor = StockPredictionML(data_dir, strategy)
        self.feature_extractor = FeatureExtractor()
        
        # Base analyzers
        self.trading_analyzer = trading_analyzer or TradingAnalyzer()
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
                financials_data: dict = None) -> Dict:
        """Make prediction using hybrid approach"""
        # Check cache first
        symbol = stock_data.get('symbol', '')
        if symbol:
            cache = get_cache()
            cached_prediction = cache.get(symbol, self.strategy)
            if cached_prediction:
                logger.debug(f"⚡ Using cached prediction for {symbol} ({self.strategy})")
                return cached_prediction
        
        # 1. Get rule-based prediction
        if self.strategy == "trading":
            base_analysis = self.trading_analyzer.analyze(stock_data, history_data)
        elif self.strategy == "investing":
            base_analysis = self.investing_analyzer.analyze(
                stock_data, financials_data, history_data
            )
        else:  # mixed
            base_analysis = self.mixed_analyzer.analyze(
                stock_data, financials_data, history_data
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
            logger.warning(f"⚠️ ML model not available for {self.strategy} strategy")
            # Try to diagnose why
            if hasattr(self.ml_predictor, 'model'):
                logger.warning(f"   - Main model is None: {self.ml_predictor.model is None}")
            if hasattr(self.ml_predictor, 'regime_models'):
                logger.warning(f"   - Regime models count: {len(self.ml_predictor.regime_models)}")
                if self.ml_predictor.regime_models:
                    for regime, regime_ml in self.ml_predictor.regime_models.items():
                        logger.warning(f"     - {regime}: model={regime_ml.model is not None if hasattr(regime_ml, 'model') else 'N/A'}")
            if hasattr(self.ml_predictor, 'label_encoder'):
                has_classes = hasattr(self.ml_predictor.label_encoder, 'classes_')
                logger.warning(f"   - Label encoder has classes: {has_classes}")
                if has_classes:
                    logger.warning(f"     - Classes: {list(self.ml_predictor.label_encoder.classes_)}")
        else:
            logger.info(f"✓ ML model available for {self.strategy} strategy")
        
        if ml_available:
            try:
                indicators = base_analysis.get('indicators', {})
                # Get enhanced analysis data for feature extraction
                market_regime = base_analysis.get('market_regime', None)
                timeframe_analysis = base_analysis.get('timeframe_analysis', None)
                relative_strength = base_analysis.get('relative_strength', None)
                support_resistance = base_analysis.get('support_resistance', None)
                
                features = self.feature_extractor.extract_features(
                    stock_data, history_data, financials_data, indicators,
                    market_regime, timeframe_analysis, relative_strength, support_resistance
                )
                # Pass stock_data and history_data for regime-specific model detection
                ml_prediction = self.ml_predictor.predict(features, stock_data, history_data)
                # Check if prediction has an error (e.g., LabelEncoder not fitted)
                if ml_prediction and 'error' in ml_prediction:
                    logger.warning(f"⚠️ ML prediction unavailable: {ml_prediction.get('error')}. Using rule-based only.")
                    ml_available = False
                    ml_prediction = None
                else:
                    # Diagnostic logging for backtesting (use INFO for visibility)
                    if ml_prediction:
                        logger.info(f"✓ ML prediction: action={ml_prediction.get('action')}, confidence={ml_prediction.get('confidence', 0):.1f}%, "
                                   f"probabilities={ml_prediction.get('probabilities', {})}")
            except Exception as e:
                logger.error(f"❌ ML prediction failed: {e}. Using rule-based only.")
                import traceback
                logger.error(traceback.format_exc())
                ml_available = False
                ml_prediction = None
        
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
        
        # Start with rules dominant (proven approach)
        rule_weight = 0.65
        ml_weight = 0.35
        
        # Check PROVEN backtest performance first (most important)
        ml_perf = self._get_recent_performance('ml_based')
        rule_perf = self._get_recent_performance('weight_based')
        
        proven_ml_accuracy = 50  # Default to random chance
        if ml_perf and ml_perf.get('total', 0) >= 10:
            proven_ml_accuracy = ml_perf.get('accuracy', 50)
            
            # If ML has proven itself in backtests (> 55%), increase its weight
            if proven_ml_accuracy > 60:
                ml_weight = 0.70
                rule_weight = 0.30
                logger.debug(f"ML proven in backtest ({proven_ml_accuracy:.1f}%) - increasing weight")
            elif proven_ml_accuracy > 55:
                ml_weight = 0.55
                rule_weight = 0.45
            elif proven_ml_accuracy < 52:
                # ML is basically random chance - heavily favor rules
                ml_weight = 0.20
                rule_weight = 0.80
                logger.debug(f"ML poor in backtest ({proven_ml_accuracy:.1f}%) - favoring rules")
        
        # If rules have proven better, always favor rules
        if rule_perf and ml_perf:
            if rule_perf.get('total', 0) >= 10 and ml_perf.get('total', 0) >= 10:
                rule_acc = rule_perf.get('accuracy', 50)
                ml_acc = ml_perf.get('accuracy', 50)
                if rule_acc > ml_acc + 5:  # Rules are significantly better
                    rule_weight = max(rule_weight, 0.70)
                    ml_weight = 1.0 - rule_weight
        
        # DISAGREEMENT PENALTY: When ML and rules disagree, favor rules
        # (since rules have domain knowledge while ML may be overconfident)
        rule_action = rule_pred.get('action', 'HOLD')
        ml_action = ml_pred.get('action', 'HOLD')
        
        if rule_action != ml_action:
            # They disagree - reduce ML weight since it hasn't proven reliable
            if proven_ml_accuracy < 55:
                ml_weight = min(ml_weight, 0.35)
                rule_weight = 1.0 - ml_weight
                logger.debug(f"ML/Rules disagree and ML unproven - favoring rules")
        
        # Confidence calibration: reduce ML confidence impact if overconfident
        ml_confidence = ml_pred.get('confidence', 50)
        if ml_confidence > 65 and proven_ml_accuracy < 55:
            # ML is overconfident - it claims high confidence but backtest is poor
            ml_weight = min(ml_weight, 0.30)
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
        
        # Scenario 1: Rule says Action, ML says HOLD (or unsure)
        if rule_action != 'HOLD' and (ml_action == 'HOLD' or ml_confidence < 60):
             logger.debug(f"⚖️ Boosting Rule weight for Aggressive Signal (Rule: {rule_action}, ML: {ml_action})")
             # Force Rule to have majority to pass the signal through
             rule_weight = max(rule_weight, 0.75) 
             ml_weight = 1.0 - rule_weight
             
            
         # Scenario 2: Rule says HOLD (Unsure), ML says Action
        elif rule_action == 'HOLD' and ml_action != 'HOLD':
             # SAFETY VETO: Prevent ML from blindly buying into extreme overbought conditions
             rsi = base_analysis.get('indicators', {}).get('rsi', 50)
             if ml_action == 'BUY' and rsi > 78 and ml_confidence < 85:
                 logger.info(f"🛡️ SAFETY VETO: Blocking ML BUY because RSI is {rsi:.2f} (Extreme Overbought). Forced HOLD.")
                 # Force Rule weight to dominate (1.0) to enforce the HOLD
                 rule_weight = 1.0
                 ml_weight = 0.0
             
             elif ml_confidence > 60:  # Lowered from 70 to allow ML (56-62% acc) to act more often
                 # If ML is reasonably confident, let it swing the vote
                 pass 
             else:
                 # If ML is weak and Rule is Unsure, stay Safe (HOLD)
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
        final_action = max(action_counts.items(), key=lambda x: x[1])[0]
        
        # Boost confidence if they agree
        if rule_action == ml_action:
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
        result['recommendation'] = {
            **recommendation,
            'action': final_action,
            'confidence': final_confidence
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

