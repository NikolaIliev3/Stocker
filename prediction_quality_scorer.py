"""
Real-time Prediction Quality Scorer
Scores predictions in real-time and flags unusual/low-quality predictions
"""
import numpy as np
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class PredictionQualityScorer:
    """Scores prediction quality in real-time"""
    
    def __init__(self):
        self.quality_threshold = 0.5  # Below this = low quality
        self.unusual_threshold = 0.3  # Below this = unusual/flagged
    
    def score_prediction(self, prediction: Dict, features: np.ndarray = None,
                        historical_predictions: list = None) -> Dict:
        """Score a prediction's quality
        
        Args:
            prediction: Prediction dict with action, confidence, probabilities
            features: Feature vector used for prediction
            historical_predictions: List of recent predictions for comparison
        
        Returns:
            Quality score dict with score, flags, and reasoning
        """
        score = 1.0  # Start with perfect score
        flags = []
        reasoning = []
        
        confidence = prediction.get('confidence', 0)
        probabilities = prediction.get('probabilities', {})
        action = prediction.get('action', 'HOLD')
        
        # Factor 1: Confidence level (higher = better quality)
        if confidence < 50:
            score -= 0.3
            flags.append('low_confidence')
            reasoning.append(f"Low confidence ({confidence:.1f}%)")
        elif confidence < 60:
            score -= 0.15
            flags.append('moderate_confidence')
            reasoning.append(f"Moderate confidence ({confidence:.1f}%)")
        
        # Factor 2: Probability distribution (more decisive = better)
        if probabilities:
            max_prob = max(probabilities.values())
            prob_entropy = self._calculate_entropy(probabilities)
            
            if max_prob < 0.6:
                score -= 0.2
                flags.append('uncertain_probabilities')
                reasoning.append(f"Uncertain probabilities (max: {max_prob*100:.1f}%)")
            
            if prob_entropy > 1.0:  # High entropy = uncertain
                score -= 0.15
                flags.append('high_entropy')
                reasoning.append(f"High probability entropy ({prob_entropy:.2f})")
        
        # Factor 3: Feature quality (if features provided)
        if features is not None:
            feature_quality = self._assess_feature_quality(features)
            if feature_quality < 0.7:
                score -= 0.2
                flags.append('poor_features')
                reasoning.append("Poor feature quality detected")
        
        # Factor 4: Unusual prediction (compared to historical)
        if historical_predictions:
            unusual_score = self._check_unusual_prediction(prediction, historical_predictions)
            if unusual_score < 0.5:
                score -= 0.25
                flags.append('unusual_prediction')
                reasoning.append("Unusual prediction compared to recent history")
        
        # Clamp score
        score = max(0.0, min(1.0, score))
        
        # Determine quality level
        if score >= 0.7:
            quality_level = 'high'
        elif score >= 0.5:
            quality_level = 'medium'
        elif score >= 0.3:
            quality_level = 'low'
        else:
            quality_level = 'very_low'
        
        # Should flag?
        should_flag = score < self.unusual_threshold
        
        return {
            'quality_score': score,
            'quality_level': quality_level,
            'flags': flags,
            'reasoning': reasoning,
            'should_flag': should_flag,
            'confidence': confidence,
            'action': action
        }
    
    def _calculate_entropy(self, probabilities: Dict) -> float:
        """Calculate entropy of probability distribution"""
        probs = [p / 100.0 for p in probabilities.values() if p > 0]  # Convert to 0-1
        if not probs:
            return 0.0
        
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        return entropy
    
    def _assess_feature_quality(self, features: np.ndarray) -> float:
        """Assess quality of feature vector"""
        if len(features) == 0:
            return 0.0
        
        # Check for NaN/Inf
        if np.isnan(features).any() or np.isinf(features).any():
            return 0.0
        
        # Check for zero variance (constant features)
        if np.std(features) == 0:
            return 0.5
        
        # Check for extreme values (outliers)
        if np.max(np.abs(features)) > 100:  # Arbitrary threshold
            return 0.7
        
        return 1.0
    
    def _check_unusual_prediction(self, prediction: Dict, 
                                  historical_predictions: List[Dict]) -> float:
        """Check if prediction is unusual compared to historical"""
        if not historical_predictions:
            return 1.0  # Can't compare, assume normal
        
        # Get recent actions
        recent_actions = [p.get('action') for p in historical_predictions[-20:]]
        current_action = prediction.get('action')
        
        # Check if action is common
        action_frequency = recent_actions.count(current_action) / len(recent_actions) if recent_actions else 0
        
        # If action is very rare (<10%), it's unusual
        if action_frequency < 0.1:
            return 0.3
        
        # Get recent confidence levels
        recent_confidences = [p.get('confidence', 50) for p in historical_predictions[-20:]]
        if recent_confidences:
            avg_confidence = np.mean(recent_confidences)
            current_confidence = prediction.get('confidence', 50)
            
            # If confidence is very different from average, it's unusual
            confidence_diff = abs(current_confidence - avg_confidence)
            if confidence_diff > 30:
                return 0.5
        
        return 1.0  # Normal prediction
