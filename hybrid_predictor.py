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

logger = logging.getLogger(__name__)


class HybridStockPredictor:
    """Hybrid predictor combining rule-based and ML approaches"""
    
    def __init__(self, data_dir: Path, strategy: str, 
                 trading_analyzer: TradingAnalyzer = None,
                 investing_analyzer: InvestingAnalyzer = None,
                 mixed_analyzer: MixedAnalyzer = None):
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
        
        # Ensemble weights
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
            except:
                pass
        return {'rule': 0.5, 'ml': 0.5}
    
    def _save_ensemble_weights(self):
        """Save ensemble weights"""
        try:
            with open(self.ensemble_weights_file, 'w') as f:
                json.dump(self.ensemble_weights, f, indent=2)
        except Exception as e:
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
            except:
                pass
        return {
            'weight_based': [],
            'ml_based': [],
            'hybrid': []
        }
    
    def _save_performance(self):
        """Save performance history"""
        try:
            all_performance = {}
            if self.performance_file.exists():
                try:
                    with open(self.performance_file, 'r') as f:
                        all_performance = json.load(f)
                except:
                    pass
            
            all_performance[self.strategy] = self.performance_history
            
            # Keep only last 1000 entries
            for method in ['weight_based', 'ml_based', 'hybrid']:
                if len(all_performance[self.strategy][method]) > 1000:
                    all_performance[self.strategy][method] = \
                        all_performance[self.strategy][method][-1000:]
            
            with open(self.performance_file, 'w') as f:
                json.dump(all_performance, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving performance: {e}")
    
    def predict(self, stock_data: dict, history_data: dict, 
                financials_data: dict = None) -> Dict:
        """Make prediction using hybrid approach"""
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
        
        # 2. Get ML prediction (if available)
        ml_prediction = None
        ml_available = self.ml_predictor.is_trained()
        
        # Diagnostic: Log ML model status
        if not ml_available:
            logger.debug(f"ML model not available for {self.strategy} strategy")
        else:
            logger.debug(f"ML model available for {self.strategy} strategy")
        
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
                ml_prediction = self.ml_predictor.predict(features)
                # Check if prediction has an error (e.g., LabelEncoder not fitted)
                if ml_prediction and 'error' in ml_prediction:
                    logger.debug(f"ML prediction unavailable: {ml_prediction.get('error')}. Using rule-based only.")
                    ml_available = False
                    ml_prediction = None
                else:
                    # Diagnostic logging for backtesting
                    if ml_prediction:
                        logger.debug(f"ML prediction: action={ml_prediction.get('action')}, confidence={ml_prediction.get('confidence', 0):.1f}%, "
                                   f"probabilities={ml_prediction.get('probabilities', {})}")
            except Exception as e:
                logger.debug(f"ML prediction failed: {e}. Using rule-based only.")
                ml_available = False
                ml_prediction = None
        
        # 3. Calculate dynamic ensemble weights
        if ml_available:
            ensemble_weights = self._calculate_dynamic_weights(
                rule_prediction, ml_prediction
            )
        else:
            ensemble_weights = {'rule': 1.0, 'ml': 0.0}
        
        # 4. Combine predictions
        final_prediction = self._ensemble_predictions(
            rule_prediction, ml_prediction, ensemble_weights, base_analysis
        )
        
        # Diagnostic logging for backtesting
        if ml_available and ml_prediction:
            logger.debug(f"Hybrid prediction: rule={rule_prediction.get('action')} ({rule_prediction.get('confidence', 0):.1f}%), "
                        f"ml={ml_prediction.get('action')} ({ml_prediction.get('confidence', 0):.1f}%), "
                        f"final={final_prediction.get('recommendation', {}).get('action')} "
                        f"(weights: rule={ensemble_weights.get('rule', 0):.2f}, ml={ensemble_weights.get('ml', 0):.2f})")
        
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
    
    def _calculate_dynamic_weights(self, rule_pred: Dict, ml_pred: Dict) -> Dict:
        """Calculate dynamic ensemble weights"""
        rule_weight = 0.5
        ml_weight = 0.5
        
        # Check if ML model has high training accuracy (from metadata)
        ml_high_accuracy = False
        ml_test_acc = 0
        try:
            if self.ml_predictor and self.ml_predictor.metadata:
                ml_test_acc = self.ml_predictor.metadata.get('test_accuracy', 0)
                if ml_test_acc >= 0.60:  # 60%+ validation accuracy
                    ml_high_accuracy = True
                    # When ML has high accuracy, favor it MUCH more aggressively
                    if ml_test_acc >= 0.62:  # 62%+ = very high accuracy
                        ml_weight = 0.85  # Strongly favor ML
                        rule_weight = 0.15
                    else:  # 60-62%
                        ml_weight = 0.75
                        rule_weight = 0.25
                    logger.info(f"🎯 ML model has high accuracy ({ml_test_acc*100:.1f}%) - strongly favoring ML (weight: {ml_weight:.2f})")
        except:
            pass
        
        # Adjust based on historical performance (only if ML doesn't have high accuracy)
        if not ml_high_accuracy:
            rule_perf = self._get_recent_performance('weight_based')
            ml_perf = self._get_recent_performance('ml_based')
            
            if rule_perf and ml_perf:
                rule_acc = rule_perf.get('accuracy', 50)
                ml_acc = ml_perf.get('accuracy', 50)
                
                if abs(rule_acc - ml_acc) > 10:
                    if rule_acc > ml_acc:
                        rule_weight = 0.7
                        ml_weight = 0.3
                    else:
                        rule_weight = 0.3
                        ml_weight = 0.7
        
        # When ML has high accuracy, be more conservative about confidence adjustments
        if ml_high_accuracy:
            # Only make small adjustments based on confidence, don't override ML preference
            rule_confidence = rule_pred.get('confidence', 50)
            ml_confidence = ml_pred.get('confidence', 50)
            
            confidence_diff = abs(rule_confidence - ml_confidence)
            if confidence_diff > 30:  # Only adjust if confidence difference is very large
                if rule_confidence > ml_confidence:
                    rule_weight += 0.05  # Small adjustment
                    ml_weight -= 0.05
                else:
                    rule_weight -= 0.05
                    ml_weight += 0.05
            
            # When predictions disagree and ML has high accuracy, favor ML unless rule confidence is MUCH higher
            if rule_pred.get('action') != ml_pred.get('action'):
                if rule_confidence > ml_confidence + 25:  # Rule must be 25%+ more confident
                    rule_weight = 0.6
                    ml_weight = 0.4
                else:
                    # Default to ML when it has high accuracy
                    rule_weight = 0.3
                    ml_weight = 0.7
        else:
            # Original logic for when ML doesn't have high accuracy
            rule_confidence = rule_pred.get('confidence', 50)
            ml_confidence = ml_pred.get('confidence', 50)
            
            confidence_diff = abs(rule_confidence - ml_confidence)
            if confidence_diff > 20:
                if rule_confidence > ml_confidence:
                    rule_weight += 0.1
                    ml_weight -= 0.1
                else:
                    rule_weight -= 0.1
                    ml_weight += 0.1
            
            # Adjust based on agreement
            if rule_pred.get('action') != ml_pred.get('action'):
                if rule_confidence > ml_confidence:
                    rule_weight = 0.7
                    ml_weight = 0.3
                else:
                    rule_weight = 0.3
                    ml_weight = 0.7
        
        # Clamp weights (but preserve ML preference when it has high accuracy)
        if ml_high_accuracy:
            rule_weight = max(0.1, min(0.4, rule_weight))  # Keep rule weight low
            ml_weight = 1.0 - rule_weight
        else:
            rule_weight = max(0.2, min(0.8, rule_weight))
            ml_weight = 1.0 - rule_weight
        
        return {'rule': rule_weight, 'ml': ml_weight}
    
    def _ensemble_predictions(self, rule_pred: Dict, ml_pred: Optional[Dict],
                             weights: Dict, base_analysis: Dict) -> Dict:
        """Combine rule-based and ML predictions"""
        if ml_pred is None:
            return {
                **base_analysis,
                'method': 'rule_based_only',
                'ensemble_weights': {'rule': 1.0, 'ml': 0.0}
            }
        
        rule_action = rule_pred.get('action', 'HOLD')
        rule_confidence = rule_pred.get('confidence', 50)
        ml_action = ml_pred.get('action', 'HOLD')
        ml_confidence = ml_pred.get('confidence', 50)
        
        # Weighted confidence
        final_confidence = (
            rule_confidence * weights['rule'] + 
            ml_confidence * weights['ml']
        )
        
        # Action selection
        if rule_action == ml_action:
            final_action = rule_action
            final_confidence = min(95, final_confidence * 1.1)
        else:
            rule_weighted_conf = rule_confidence * weights['rule']
            ml_weighted_conf = ml_confidence * weights['ml']
            final_action = rule_action if rule_weighted_conf > ml_weighted_conf else ml_action
        
        # Use rule-based prices (they're more detailed)
        recommendation = base_analysis.get('recommendation', {})
        
        # Generate reasoning
        reasoning = base_analysis.get('reasoning', '')
        reasoning += f"\n\n🤖 HYBRID AI ANALYSIS:\n"
        reasoning += f"Rule-based: {rule_action} ({rule_confidence:.1f}%)\n"
        reasoning += f"ML Model: {ml_action} ({ml_confidence:.1f}%)\n"
        reasoning += f"Ensemble Weights: Rule {weights['rule']*100:.0f}% / ML {weights['ml']*100:.0f}%\n"
        reasoning += f"Final Prediction: {final_action} ({final_confidence:.1f}% confidence)\n"
        
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
        
        recent = history[-50:] if len(history) > 50 else history
        correct = sum(1 for h in recent if h.get('was_correct') is True)
        total = len(recent)
        accuracy = (correct / total * 100) if total > 0 else 0
        
        return {
            'accuracy': accuracy,
            'total': total,
            'correct': correct
        }
    
    def update_performance(self, prediction: Dict, actual_outcome: bool):
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
    
    def _update_ensemble_weights(self):
        """Update ensemble weights based on recent performance"""
        rule_perf = self._get_recent_performance('weight_based')
        ml_perf = self._get_recent_performance('ml_based')
        
        if rule_perf and ml_perf and rule_perf.get('total', 0) >= 10:
            rule_acc = rule_perf.get('accuracy', 50)
            ml_acc = ml_perf.get('accuracy', 50)
            
            if abs(rule_acc - ml_acc) > 5:
                if rule_acc > ml_acc:
                    self.ensemble_weights['rule'] = min(0.8, self.ensemble_weights['rule'] + 0.05)
                    self.ensemble_weights['ml'] = 1.0 - self.ensemble_weights['rule']
                else:
                    self.ensemble_weights['ml'] = min(0.8, self.ensemble_weights['ml'] + 0.05)
                    self.ensemble_weights['rule'] = 1.0 - self.ensemble_weights['ml']
                
                self._save_ensemble_weights()

