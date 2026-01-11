"""
Model Explainability System
Provides SHAP values and feature contributions to explain predictions
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

# Try to import SHAP
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    logging.getLogger(__name__).warning("SHAP not available. Install with: pip install shap")

logger = logging.getLogger(__name__)


class ModelExplainer:
    """Explains model predictions using SHAP values and feature importance"""
    
    def __init__(self, model, feature_names: List[str], scaler=None):
        self.model = model
        self.feature_names = feature_names
        self.scaler = scaler
        self.shap_explainer = None
        self.shap_values_cache = None
        
        # Initialize SHAP explainer if available
        if HAS_SHAP and model is not None:
            try:
                # Use TreeExplainer for tree-based models (faster)
                if hasattr(model, 'predict_proba'):
                    # Try to get base estimator for ensemble models
                    base_estimator = self._get_base_estimator(model)
                    if base_estimator is not None:
                        try:
                            self.shap_explainer = shap.TreeExplainer(base_estimator)
                            logger.debug("Initialized SHAP TreeExplainer")
                        except:
                            # Fallback to KernelExplainer (slower but works for any model)
                            logger.debug("TreeExplainer failed, will use KernelExplainer when needed")
                    else:
                        logger.debug("No base estimator found, will use KernelExplainer when needed")
            except Exception as e:
                logger.debug(f"Could not initialize SHAP explainer: {e}")
    
    def _get_base_estimator(self, model):
        """Extract base estimator from ensemble models"""
        # Try VotingClassifier
        if hasattr(model, 'named_estimators_'):
            # Get RandomForest if available (best for SHAP)
            if 'rf' in model.named_estimators_:
                return model.named_estimators_['rf']
            # Otherwise return first estimator
            estimators = list(model.named_estimators_.values())
            if estimators:
                return estimators[0]
        
        # Try StackingClassifier
        if hasattr(model, 'estimators_'):
            estimators = model.estimators_
            if estimators:
                # Prefer RandomForest
                for est in estimators:
                    if hasattr(est, 'feature_importances_'):
                        return est
                return estimators[0]
        
        # If model itself is tree-based
        if hasattr(model, 'feature_importances_'):
            return model
        
        return None
    
    def explain_prediction(self, features: np.ndarray, 
                          prediction: Dict = None) -> Dict:
        """Explain a single prediction using SHAP values
        
        Args:
            features: Feature vector (1D or 2D array)
            prediction: Optional prediction dict with action, confidence, etc.
        
        Returns:
            Explanation dict with feature contributions
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features if scaler is provided
        if self.scaler is not None:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
        
        explanation = {
            'has_explanation': False,
            'method': None,
            'feature_contributions': {},
            'top_contributors': [],
            'reasoning': []
        }
        
        # Method 1: SHAP values (if available)
        if HAS_SHAP and self.model is not None:
            try:
                shap_values = self._get_shap_values(features_scaled)
                if shap_values is not None:
                    explanation['has_explanation'] = True
                    explanation['method'] = 'shap'
                    explanation['feature_contributions'] = self._format_shap_contributions(
                        shap_values, features_scaled[0]
                    )
                    explanation['top_contributors'] = self._get_top_contributors(
                        explanation['feature_contributions'], top_n=10
                    )
                    explanation['reasoning'] = self._generate_reasoning_from_shap(
                        explanation['top_contributors'], prediction
                    )
                    return explanation
            except Exception as e:
                logger.debug(f"SHAP explanation failed: {e}")
        
        # Method 2: Feature importance (fallback)
        if self.model is not None:
            try:
                feature_importance = self._get_feature_importance()
                if feature_importance:
                    explanation['has_explanation'] = True
                    explanation['method'] = 'feature_importance'
                    explanation['feature_contributions'] = self._format_importance_contributions(
                        feature_importance, features_scaled[0]
                    )
                    explanation['top_contributors'] = self._get_top_contributors(
                        explanation['feature_contributions'], top_n=10
                    )
                    explanation['reasoning'] = self._generate_reasoning_from_importance(
                        explanation['top_contributors'], prediction
                    )
                    return explanation
            except Exception as e:
                logger.debug(f"Feature importance explanation failed: {e}")
        
        return explanation
    
    def _get_shap_values(self, features: np.ndarray) -> Optional[np.ndarray]:
        """Get SHAP values for features"""
        if not HAS_SHAP or self.model is None:
            return None
        
        try:
            # Use TreeExplainer if available
            if self.shap_explainer is not None:
                shap_values = self.shap_explainer.shap_values(features)
                # Handle multi-class output (returns list)
                if isinstance(shap_values, list):
                    # For classification, get values for predicted class
                    prediction = self.model.predict(features)[0]
                    if isinstance(prediction, (np.integer, int)):
                        shap_values = shap_values[prediction]
                    else:
                        shap_values = shap_values[0]  # Default to first class
                return shap_values[0] if shap_values.ndim > 1 else shap_values
            
            # Fallback: KernelExplainer (slower but works for any model)
            # Use a sample of training data as background
            # For now, use a simple background (zeros or mean)
            background = np.zeros((1, features.shape[1]))
            explainer = shap.KernelExplainer(self.model.predict_proba, background)
            shap_values = explainer.shap_values(features[0])
            
            # Handle multi-class
            if isinstance(shap_values, list):
                prediction = self.model.predict(features)[0]
                if isinstance(prediction, (np.integer, int)) and prediction < len(shap_values):
                    shap_values = shap_values[prediction]
                else:
                    shap_values = shap_values[0]
            
            return shap_values
            
        except Exception as e:
            logger.debug(f"Error computing SHAP values: {e}")
            return None
    
    def _get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from model"""
        if self.model is None:
            return None
        
        try:
            # Try to get from model directly
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                return dict(zip(self.feature_names[:len(importances)], importances))
            
            # Try from base estimator
            base_estimator = self._get_base_estimator(self.model)
            if base_estimator and hasattr(base_estimator, 'feature_importances_'):
                importances = base_estimator.feature_importances_
                return dict(zip(self.feature_names[:len(importances)], importances))
            
            # Try from named estimators
            if hasattr(self.model, 'named_estimators_'):
                for name, estimator in self.model.named_estimators_.items():
                    if hasattr(estimator, 'feature_importances_'):
                        importances = estimator.feature_importances_
                        return dict(zip(self.feature_names[:len(importances)], importances))
            
        except Exception as e:
            logger.debug(f"Error getting feature importance: {e}")
        
        return None
    
    def _format_shap_contributions(self, shap_values: np.ndarray, 
                                   feature_values: np.ndarray) -> Dict[str, float]:
        """Format SHAP values as feature contributions"""
        contributions = {}
        for i, (name, shap_val, feat_val) in enumerate(zip(
            self.feature_names[:len(shap_values)],
            shap_values,
            feature_values
        )):
            # Contribution is SHAP value weighted by feature value
            contributions[name] = float(shap_val * feat_val)
        
        return contributions
    
    def _format_importance_contributions(self, importance: Dict[str, float],
                                        feature_values: np.ndarray) -> Dict[str, float]:
        """Format feature importance as contributions (weighted by feature values)"""
        contributions = {}
        for i, (name, feat_val) in enumerate(zip(self.feature_names[:len(feature_values)], feature_values)):
            imp = importance.get(name, 0)
            # Contribution is importance weighted by normalized feature value
            contributions[name] = float(imp * abs(feat_val))
        
        return contributions
    
    def _get_top_contributors(self, contributions: Dict[str, float], 
                             top_n: int = 10) -> List[Tuple[str, float]]:
        """Get top N contributing features"""
        sorted_contribs = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return sorted_contribs[:top_n]
    
    def _generate_reasoning_from_shap(self, top_contributors: List[Tuple[str, float]],
                                     prediction: Dict = None) -> List[str]:
        """Generate human-readable reasoning from SHAP contributions"""
        reasoning = []
        
        if not top_contributors:
            return ["No feature contributions available"]
        
        # Get top positive and negative contributors
        positive = [(name, val) for name, val in top_contributors if val > 0]
        negative = [(name, val) for name, val in top_contributors if val < 0]
        
        if positive:
            top_positive = positive[0]
            reasoning.append(f"Strongest positive signal: {top_positive[0]} (contribution: {top_positive[1]:.4f})")
        
        if negative:
            top_negative = negative[0]
            reasoning.append(f"Strongest negative signal: {top_negative[0]} (contribution: {top_negative[1]:.4f})")
        
        # Add top 3 contributors
        if len(top_contributors) >= 3:
            reasoning.append(f"Top 3 factors: {', '.join([name for name, _ in top_contributors[:3]])}")
        
        return reasoning
    
    def _generate_reasoning_from_importance(self, top_contributors: List[Tuple[str, float]],
                                          prediction: Dict = None) -> List[str]:
        """Generate human-readable reasoning from feature importance"""
        reasoning = []
        
        if not top_contributors:
            return ["No feature importance available"]
        
        # Top contributing features
        top_features = [name for name, _ in top_contributors[:5]]
        reasoning.append(f"Most influential features: {', '.join(top_features)}")
        
        return reasoning
    
    def get_feature_importance_summary(self) -> Dict:
        """Get summary of feature importance"""
        importance = self._get_feature_importance()
        if not importance:
            return {'available': False}
        
        sorted_importance = sorted(
            importance.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return {
            'available': True,
            'top_features': sorted_importance[:20],
            'total_features': len(importance)
        }
