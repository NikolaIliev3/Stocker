"""
Data Drift Detection Module
Uses scipy KS-test to detect when live market data has drifted from training data.
This indicates the model may need retraining because market conditions have changed.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json
import logging
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


class DataDriftDetector:
    """
    Detects data drift between training data and live/current data.
    Uses Kolmogorov-Smirnov test per feature to detect distribution shifts.
    """
    
    def __init__(self, data_dir: Path, strategy: str = 'trading'):
        """
        Initialize the drift detector.
        """
        self.data_dir = data_dir
        self.strategy = strategy
        self.drift_dir = data_dir / "drift_detection"
        self.drift_dir.mkdir(parents=True, exist_ok=True)
        
        self.reference_file = self.drift_dir / f"reference_data_{strategy}.parquet"
        self.drift_history_file = self.drift_dir / f"drift_history_{strategy}.json"
        
        # Drift detection thresholds
        self.dataset_drift_threshold = 0.5
        self.feature_drift_threshold = 0.1
        self.ks_p_value_threshold = 0.05  # p < 0.05 = significant drift
        
        # Load reference data if exists
        self.reference_data = self._load_reference_data()
        self.drift_history = self._load_drift_history()
        
        # Critical features to monitor
        self.critical_features = [
            'rsi', 'macd', 'atr_percent', 'momentum_5', 'momentum_10',
            'volatility_20', 'price_vs_ma_20', 'price_vs_ma_50',
            'volume_ratio', 'trend_strength_20'
        ]
    
    def _load_reference_data(self) -> Optional[pd.DataFrame]:
        """Load reference (training) data distribution"""
        if self.reference_file.exists():
            try:
                return pd.read_parquet(self.reference_file)
            except Exception as e:
                logger.error(f"Error loading reference data: {e}")
        return None
    
    def _load_drift_history(self) -> List[Dict]:
        """Load drift detection history"""
        if self.drift_history_file.exists():
            try:
                with open(self.drift_history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading drift history: {e}")
        return []
    
    def _save_drift_history(self):
        """Save drift detection history"""
        try:
            # Keep last 100 entries
            history_to_save = self.drift_history[-100:]
            with open(self.drift_history_file, 'w') as f:
                json.dump(history_to_save, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving drift history: {e}")
    
    def set_reference_data(self, training_features: pd.DataFrame):
        """
        Set reference data from training features.
        Call this after training a new model.
        
        Args:
            training_features: DataFrame of features used for training
        """
        try:
            # Store reference data
            self.reference_data = training_features.copy()
            self.reference_data.to_parquet(self.reference_file)
            logger.info(f"Reference data saved: {len(training_features)} samples, "
                       f"{len(training_features.columns)} features")
        except Exception as e:
            logger.error(f"Error saving reference data: {e}")
    
    def check_drift(self, current_data: pd.DataFrame) -> Dict:
        """
        Check for data drift between reference and current data using KS-test.
        
        For each numeric feature, runs a two-sample Kolmogorov-Smirnov test
        comparing the reference distribution to the current distribution.
        A feature is considered drifted if p-value < 0.05.
        
        Args:
            current_data: DataFrame of current/live features
            
        Returns:
            Dict with drift detection results:
            - drift_detected: bool
            - drift_score: float (0-1, fraction of features that drifted)
            - drifted_features: list of feature names that drifted
            - recommendation: str
        """
        if self.reference_data is None:
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'drifted_features': [],
                'recommendation': 'No reference data. Train a model first.',
                'error': 'No reference data available'
            }
        
        if len(current_data) < 10:
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'drifted_features': [],
                'recommendation': 'Need at least 10 samples for drift detection',
                'error': 'Insufficient data'
            }
        
        try:
            # Align columns between reference and current data
            common_columns = list(set(self.reference_data.columns) & set(current_data.columns))
            
            if len(common_columns) < 5:
                return {
                    'drift_detected': False,
                    'drift_score': 0.0,
                    'drifted_features': [],
                    'recommendation': 'Feature columns do not match reference data',
                    'error': 'Column mismatch'
                }
            
            ref_aligned = self.reference_data[common_columns].copy()
            curr_aligned = current_data[common_columns].copy()
            
            # Handle any NaN/inf values
            ref_aligned = ref_aligned.replace([np.inf, -np.inf], np.nan).fillna(0)
            curr_aligned = curr_aligned.replace([np.inf, -np.inf], np.nan).fillna(0)
            
            # Run KS-test per feature
            drifted_features = []
            tested_features = 0
            
            for col in common_columns:
                try:
                    ref_values = ref_aligned[col].values.astype(float)
                    curr_values = curr_aligned[col].values.astype(float)
                    
                    # Skip constant columns (no variance to test)
                    if np.std(ref_values) == 0 and np.std(curr_values) == 0:
                        continue
                    
                    tested_features += 1
                    statistic, p_value = ks_2samp(ref_values, curr_values)
                    
                    if p_value < self.ks_p_value_threshold:
                        drifted_features.append(col)
                        
                except (ValueError, TypeError):
                    # Skip non-numeric or problematic columns
                    continue
            
            # Calculate drift score (fraction of tested features that drifted)
            drift_share = len(drifted_features) / tested_features if tested_features > 0 else 0.0
            dataset_drift = drift_share > self.dataset_drift_threshold
            
            # Check if critical features drifted
            critical_drifted = [f for f in drifted_features if f in self.critical_features]
            
            # Generate recommendation
            if dataset_drift or drift_share > 0.5:
                recommendation = "⚠️ SIGNIFICANT DRIFT DETECTED - Consider retraining the model"
            elif len(critical_drifted) > 2:
                recommendation = f"⚠️ Critical features drifted: {critical_drifted[:3]} - Monitor closely"
            elif drift_share > 0.2:
                recommendation = "Minor drift detected - Continue monitoring"
            else:
                recommendation = "✓ No significant drift - Model inputs are stable"
            
            # Record in history
            drift_record = {
                'timestamp': datetime.now().isoformat(),
                'strategy': self.strategy,
                'drift_detected': dataset_drift,
                'drift_score': float(drift_share),
                'num_drifted_features': len(drifted_features),
                'critical_features_drifted': critical_drifted,
                'samples_analyzed': len(current_data)
            }
            self.drift_history.append(drift_record)
            self._save_drift_history()
            
            return {
                'drift_detected': dataset_drift,
                'drift_score': float(drift_share),
                'drifted_features': drifted_features,
                'critical_features_drifted': critical_drifted,
                'recommendation': recommendation,
                'samples_analyzed': len(current_data),
                'reference_samples': len(self.reference_data)
            }
            
        except Exception as e:
            logger.error(f"Error checking drift: {e}")
            return {
                'drift_detected': False,
                'drift_score': 0.0,
                'drifted_features': [],
                'recommendation': 'Error during drift detection',
                'error': str(e)
            }
    
    def get_drifted_features(self, current_data: pd.DataFrame) -> List[str]:
        """Get list of features that have drifted."""
        result = self.check_drift(current_data)
        return result.get('drifted_features', [])
    
    def should_retrain(self, current_data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Determine if model should be retrained based on drift.
        
        Returns:
            Tuple of (should_retrain: bool, reason: str)
        """
        result = self.check_drift(current_data)
        
        if result.get('error'):
            return False, f"Cannot determine: {result['error']}"
        
        drift_score = result.get('drift_score', 0.0)
        critical_drifted = result.get('critical_features_drifted', [])
        
        # Retrain if significant drift
        if drift_score > 0.5:
            return True, f"High drift detected ({drift_score*100:.1f}% of features)"
        
        # Retrain if multiple critical features drifted
        if len(critical_drifted) >= 3:
            return True, f"Multiple critical features drifted: {critical_drifted}"
        
        # Check for persistent drift (drift in 3+ consecutive checks)
        recent_history = self.drift_history[-5:]
        if len(recent_history) >= 3:
            consecutive_drift = sum(1 for h in recent_history[-3:] if h.get('drift_detected', False))
            if consecutive_drift >= 3:
                return True, "Persistent drift detected over multiple checks"
        
        return False, result.get('recommendation', 'No retraining needed')
    
    def get_drift_summary(self) -> Dict:
        """Get summary of drift detection status."""
        if not self.drift_history:
            return {
                'status': 'no_data',
                'message': 'No drift checks performed yet'
            }
        
        recent = self.drift_history[-10:]
        avg_drift_score = np.mean([h.get('drift_score', 0) for h in recent])
        drift_count = sum(1 for h in recent if h.get('drift_detected', False))
        
        return {
            'status': 'monitoring',
            'checks_performed': len(self.drift_history),
            'recent_drift_rate': drift_count / len(recent) if recent else 0,
            'average_drift_score': float(avg_drift_score),
            'last_check': self.drift_history[-1] if self.drift_history else None,
            'reference_data_available': self.reference_data is not None,
            'reference_samples': len(self.reference_data) if self.reference_data is not None else 0
        }
    
    def generate_drift_report(self, current_data: pd.DataFrame) -> str:
        """Generate a human-readable drift report."""
        result = self.check_drift(current_data)
        
        report_lines = [
            "=" * 50,
            "DATA DRIFT DETECTION REPORT",
            f"Strategy: {self.strategy}",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 50,
            "",
            f"Samples Analyzed: {result.get('samples_analyzed', 0)}",
            f"Reference Samples: {result.get('reference_samples', 0)}",
            "",
            f"DRIFT DETECTED: {'YES ⚠️' if result.get('drift_detected') else 'NO ✓'}",
            f"Drift Score: {result.get('drift_score', 0)*100:.1f}%",
            "",
        ]
        
        drifted = result.get('drifted_features', [])
        if drifted:
            report_lines.append(f"Drifted Features ({len(drifted)}):")
            for feat in drifted[:10]:  # Show top 10
                is_critical = "⚠️ CRITICAL" if feat in self.critical_features else ""
                report_lines.append(f"  - {feat} {is_critical}")
            if len(drifted) > 10:
                report_lines.append(f"  ... and {len(drifted) - 10} more")
        else:
            report_lines.append("No features drifted significantly.")
        
        report_lines.extend([
            "",
            "RECOMMENDATION:",
            result.get('recommendation', 'No recommendation'),
            "=" * 50
        ])
        
        return "\n".join(report_lines)
