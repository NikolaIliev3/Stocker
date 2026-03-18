"""
Production Monitoring and Alerting System
Tracks model performance in real-time, detects degradation, and sends alerts
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Tuple
from collections import deque
import logging
import pandas as pd

# Try to import drift detector
# Note: Use broad Exception catch because data_drift_detector imports evidently
# which can fail with pydantic ConfigError on Python 3.14+ (not just ImportError)
try:
    from data_drift_detector import DataDriftDetector
    HAS_DRIFT_DETECTOR = True
except Exception:
    HAS_DRIFT_DETECTOR = False
    logging.getLogger(__name__).debug("Data drift detector not available")

logger = logging.getLogger(__name__)


class ProductionMonitor:
    """Monitors model performance in production and alerts on issues"""
    
    def __init__(self, data_dir: Path, strategy: str):
        self.data_dir = data_dir
        self.strategy = strategy
        self.monitoring_file = data_dir / f"production_monitoring_{strategy}.json"
        
        # Configuration
        self.accuracy_threshold = 0.40  # Alert if accuracy drops below 40%
        self.degradation_threshold = 0.15  # Alert if accuracy drops by 15%
        self.min_samples_for_alert = 20  # Need at least 20 samples before alerting
        self.window_size = 100  # Track last 100 predictions
        
        # Performance tracking
        self.recent_predictions = deque(maxlen=self.window_size)
        self.performance_history = []
        self.alerts_history = []
        
        # Initialize drift detector if available
        if HAS_DRIFT_DETECTOR:
            self.drift_detector = DataDriftDetector(data_dir, strategy)
        else:
            self.drift_detector = None
        
        # Load existing data
        self._load_monitoring_data()
    
    def _load_monitoring_data(self):
        """Load monitoring data from file"""
        if self.monitoring_file.exists():
            try:
                with open(self.monitoring_file, 'r') as f:
                    data = json.load(f)
                    self.performance_history = data.get('performance_history', [])
                    self.alerts_history = data.get('alerts_history', [])
                    
                    # Restore recent predictions (last N)
                    recent = data.get('recent_predictions', [])
                    self.recent_predictions = deque(recent[-self.window_size:], maxlen=self.window_size)
            except Exception as e:
                logger.error(f"Error loading monitoring data: {e}")
                self.performance_history = []
                self.alerts_history = []
                self.recent_predictions = deque(maxlen=self.window_size)
        else:
            self.performance_history = []
            self.alerts_history = []
            self.recent_predictions = deque(maxlen=self.window_size)
    
    def _save_monitoring_data(self):
        """Save monitoring data to file"""
        try:
            data = {
                'strategy': self.strategy,
                'last_updated': datetime.now().isoformat(),
                'recent_predictions': list(self.recent_predictions),
                'performance_history': self.performance_history[-100:],  # Keep last 100
                'alerts_history': self.alerts_history[-50:]  # Keep last 50 alerts
            }
            with open(self.monitoring_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving monitoring data: {e}")
    
    def record_prediction(self, symbol: str, action: str, confidence: float, 
                         features: np.ndarray = None, metadata: Dict = None):
        """Record a prediction made in production"""
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'action': action,
            'confidence': float(confidence),
            'features_hash': None,  # Can store feature hash for drift detection
            'metadata': metadata or {}
        }
        
        if features is not None:
            # Store feature hash for drift detection
            feature_hash = hash(tuple(np.round(features, 4)))
            prediction_record['features_hash'] = feature_hash
        
        self.recent_predictions.append(prediction_record)
        self._save_monitoring_data()
    
    def record_outcome(self, symbol: str, action: str, was_correct: bool, 
                      actual_price_change: float = None):
        """Record the outcome of a prediction"""
        # Find matching prediction
        matching_pred = None
        for pred in reversed(self.recent_predictions):
            if pred.get('symbol') == symbol and pred.get('action') == action:
                # Check if it's recent (within last 30 days)
                pred_time = datetime.fromisoformat(pred.get('timestamp', ''))
                if (datetime.now() - pred_time).days <= 30:
                    matching_pred = pred
                    break
        
        if matching_pred:
            matching_pred['outcome'] = was_correct
            matching_pred['outcome_recorded_at'] = datetime.now().isoformat()
            if actual_price_change is not None:
                matching_pred['actual_price_change'] = float(actual_price_change)
            
            # Update performance metrics
            self._update_performance_metrics()
            self._check_for_alerts()
            self._save_monitoring_data()
    
    def _update_performance_metrics(self):
        """Update performance metrics from recent predictions"""
        # Get predictions with outcomes
        predictions_with_outcomes = [
            p for p in self.recent_predictions 
            if 'outcome' in p
        ]
        
        if len(predictions_with_outcomes) < self.min_samples_for_alert:
            return
        
        # Calculate current accuracy
        correct = sum(1 for p in predictions_with_outcomes if p.get('outcome') is True)
        total = len(predictions_with_outcomes)
        current_accuracy = correct / total if total > 0 else 0
        
        # Get previous accuracy
        previous_accuracy = 0
        if self.performance_history:
            previous_accuracy = self.performance_history[-1].get('accuracy', 0)
        
        # Calculate accuracy change
        accuracy_change = current_accuracy - previous_accuracy
        
        # Store performance snapshot
        performance_snapshot = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': current_accuracy,
            'correct': correct,
            'total': total,
            'accuracy_change': accuracy_change,
            'previous_accuracy': previous_accuracy
        }
        
        self.performance_history.append(performance_snapshot)
        
        logger.debug(f"Production monitoring ({self.strategy}): Accuracy={current_accuracy*100:.1f}% "
                    f"({correct}/{total}), Change={accuracy_change*100:+.1f}%")
    
    def _check_for_alerts(self):
        """Check for performance issues and generate alerts"""
        if len(self.performance_history) < 2:
            return
        
        current_perf = self.performance_history[-1]
        current_accuracy = current_perf.get('accuracy', 0)
        total_samples = current_perf.get('total', 0)
        
        # Need minimum samples before alerting
        if total_samples < self.min_samples_for_alert:
            return
        
        alerts = []
        
        # Check 1: Low absolute accuracy
        if current_accuracy < self.accuracy_threshold:
            alert = {
                'type': 'low_accuracy',
                'severity': 'critical',
                'message': f"Production accuracy ({current_accuracy*100:.1f}%) below threshold ({self.accuracy_threshold*100:.0f}%)",
                'current_accuracy': current_accuracy,
                'threshold': self.accuracy_threshold,
                'samples': total_samples,
                'timestamp': datetime.now().isoformat()
            }
            alerts.append(alert)
        
        # Check 2: Significant degradation
        if len(self.performance_history) >= 2:
            previous_perf = self.performance_history[-2]
            previous_accuracy = previous_perf.get('accuracy', 0)
            accuracy_drop = previous_accuracy - current_accuracy
            
            if accuracy_drop > self.degradation_threshold:
                alert = {
                    'type': 'degradation',
                    'severity': 'warning',
                    'message': f"Accuracy dropped by {accuracy_drop*100:.1f}% (from {previous_accuracy*100:.1f}% to {current_accuracy*100:.1f}%)",
                    'previous_accuracy': previous_accuracy,
                    'current_accuracy': current_accuracy,
                    'degradation': accuracy_drop,
                    'threshold': self.degradation_threshold,
                    'samples': total_samples,
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)
        
        # Check 3: Declining trend (3 consecutive drops)
        if len(self.performance_history) >= 3:
            recent = self.performance_history[-3:]
            accuracies = [p.get('accuracy', 0) for p in recent]
            if all(accuracies[i] > accuracies[i+1] for i in range(len(accuracies)-1)):
                alert = {
                    'type': 'declining_trend',
                    'severity': 'warning',
                    'message': f"Declining trend detected: {accuracies[0]*100:.1f}% → {accuracies[-1]*100:.1f}% over 3 periods",
                    'accuracies': accuracies,
                    'samples': total_samples,
                    'timestamp': datetime.now().isoformat()
                }
                alerts.append(alert)
        
        # Record alerts
        for alert in alerts:
            # Check if similar alert was already sent recently (avoid spam)
            recent_alerts = [a for a in self.alerts_history 
                           if a.get('type') == alert['type'] and 
                           (datetime.now() - datetime.fromisoformat(a.get('timestamp', ''))).hours < 24]
            
            if not recent_alerts:  # No similar alert in last 24 hours
                self.alerts_history.append(alert)
                logger.warning(f"🚨 PRODUCTION ALERT ({self.strategy}): {alert['message']}")
    
    def get_current_performance(self) -> Dict:
        """Get current performance metrics"""
        if not self.performance_history:
            return {
                'accuracy': 0,
                'total_predictions': 0,
                'predictions_with_outcomes': 0,
                'status': 'insufficient_data'
            }
        
        current = self.performance_history[-1]
        return {
            'accuracy': current.get('accuracy', 0),
            'total_predictions': len(self.recent_predictions),
            'predictions_with_outcomes': current.get('total', 0),
            'correct': current.get('correct', 0),
            'accuracy_change': current.get('accuracy_change', 0),
            'status': 'monitoring' if current.get('total', 0) >= self.min_samples_for_alert else 'insufficient_data'
        }
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alerts_history
            if datetime.fromisoformat(alert.get('timestamp', '')) >= cutoff
        ]
    
    def detect_data_drift(self, current_features: np.ndarray, 
                         reference_features: Optional[np.ndarray] = None,
                         feature_names: Optional[List[str]] = None) -> Dict:
        """Detect data drift by comparing feature distributions"""
        # Use Evidently-based detection if available
        if self.drift_detector is not None and HAS_DRIFT_DETECTOR:
            try:
                # Convert to DataFrame for Evidently
                if isinstance(current_features, np.ndarray):
                    if feature_names is not None:
                        current_df = pd.DataFrame(current_features, columns=feature_names)
                    else:
                        # Generate generic column names
                        n_features = current_features.shape[1] if len(current_features.shape) > 1 else len(current_features)
                        current_df = pd.DataFrame(current_features, columns=[f'feature_{i}' for i in range(n_features)])
                else:
                    current_df = current_features  # Already a DataFrame
                
                # Check drift using Evidently
                drift_result = self.drift_detector.check_drift(current_df)
                
                # Log if significant drift
                if drift_result.get('drift_detected'):
                    logger.warning(f"Data drift detected: {drift_result.get('recommendation')}")
                
                return drift_result
            except Exception as e:
                logger.error(f"Evidently drift detection failed: {e}, falling back to hash-based")
                # Fall through to hash-based backup
        
        # Fallback: Hash-based drift detection (original implementation)
        if len(self.recent_predictions) < 20:
            return {'drift_detected': False, 'reason': 'insufficient_data'}
        
        # Get recent feature hashes
        recent_hashes = [
            p.get('features_hash') for p in self.recent_predictions
            if p.get('features_hash') is not None
        ]
        
        if len(recent_hashes) < 20:
            return {'drift_detected': False, 'reason': 'insufficient_feature_data'}
        
        # Calculate hash distribution
        hash_counts = {}
        for h in recent_hashes:
            hash_counts[h] = hash_counts.get(h, 0) + 1
        
        # If too many unique hashes (high diversity), might indicate drift
        unique_ratio = len(hash_counts) / len(recent_hashes)
        
        drift_detected = unique_ratio > 0.8  # More than 80% unique hashes
        
        return {
            'drift_detected': drift_detected,
            'unique_hash_ratio': unique_ratio,
            'total_samples': len(recent_hashes),
            'unique_hashes': len(hash_counts),
            'method': 'hash_based_fallback'
        }
    
    def should_trigger_retraining(self) -> Tuple[bool, str]:
        """Determine if retraining should be triggered based on monitoring"""
        current_perf = self.get_current_performance()
        
        if current_perf['status'] == 'insufficient_data':
            return False, "Insufficient data for monitoring"
        
        accuracy = current_perf['accuracy']
        
        # Trigger retraining if accuracy is very low
        if accuracy < 0.35:
            return True, f"Critical: Accuracy {accuracy*100:.1f}% is below 35%"
        
        # Trigger if accuracy dropped significantly
        if current_perf['accuracy_change'] < -0.20:  # 20% drop
            return True, f"Degradation: Accuracy dropped by {abs(current_perf['accuracy_change']*100):.1f}%"
        
        # Check for recent critical alerts
        recent_critical = [
            a for a in self.get_recent_alerts(hours=6)
            if a.get('severity') == 'critical'
        ]
        
        if recent_critical:
            return True, f"Critical alerts detected: {len(recent_critical)} in last 6 hours"
        
        return False, "Performance within acceptable range"
