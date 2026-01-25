"""
Training Data Validation Pipeline
Validates training samples before model training to ensure data quality
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import Counter
import logging

# Try to import label quality analyzer
try:
    from label_quality_analyzer import LabelQualityAnalyzer
    HAS_LABEL_QUALITY = True
except ImportError:
    HAS_LABEL_QUALITY = False
    logging.getLogger(__name__).debug("Label quality analyzer not available")

logger = logging.getLogger(__name__)


class TrainingDataValidator:
    """Validates training data before model training"""
    
    def __init__(self, data_dir=None):
        self.validation_results = {}
        self.issues_found = []
        self.warnings = []
        
        # Initialize label quality analyzer if available
        if HAS_LABEL_QUALITY and data_dir is not None:
            try:
                from pathlib import Path
                self.label_quality_analyzer = LabelQualityAnalyzer(Path(data_dir))
            except Exception as e:
                logger.debug(f"Could not initialize label quality analyzer: {e}")
                self.label_quality_analyzer = None
        else:
            self.label_quality_analyzer = None
    
    def validate_training_samples(self, training_samples: List[Dict]) -> Dict:
        """Comprehensive validation of training samples
        
        Args:
            training_samples: List of training sample dicts with 'features' and 'label' keys
        
        Returns:
            Validation report with pass/fail status and issues
        """
        self.issues_found = []
        self.warnings = []
        
        if not training_samples:
            return {
                'valid': False,
                'reason': 'No training samples provided',
                'issues': ['Empty training dataset']
            }
        
        # Check 1: Minimum sample size
        if len(training_samples) < 30:
            self.issues_found.append(f"Insufficient samples: {len(training_samples)} < 30 minimum")
            return {
                'valid': False,
                'reason': 'Insufficient training samples',
                'issues': self.issues_found
            }
        
        # Check 2: Feature consistency
        feature_validation = self._validate_features(training_samples)
        if not feature_validation['valid']:
            self.issues_found.extend(feature_validation['issues'])
        
        # Check 3: Label distribution
        label_validation = self._validate_labels(training_samples)
        if not label_validation['valid']:
            self.issues_found.extend(label_validation['issues'])
        else:
            self.warnings.extend(label_validation['warnings'])
        
        # Check 4: Missing values
        missing_validation = self._validate_missing_values(training_samples)
        if not missing_validation['valid']:
            self.issues_found.extend(missing_validation['issues'])
        
        # Check 5: Outliers
        outlier_validation = self._validate_outliers(training_samples)
        if outlier_validation['outlier_percentage'] > 10:
            self.warnings.append(f"High outlier percentage: {outlier_validation['outlier_percentage']:.1f}%")
        
        # Check 6: Class balance
        balance_validation = self._validate_class_balance(training_samples)
        if not balance_validation['balanced']:
            self.warnings.append(f"Class imbalance detected: {balance_validation['ratio']:.2f}:1")
        
        # Check 7: Feature variance
        variance_validation = self._validate_feature_variance(training_samples)
        if variance_validation['low_variance_features'] > 0:
            self.warnings.append(f"{variance_validation['low_variance_features']} features have low variance")
        
        # Check 8: Label quality (using Cleanlab if available)
        label_quality_info = {}
        if self.label_quality_analyzer is not None:
            try:
                # Extract features and labels
                X = np.array([s['features'] for s in training_samples if 'features' in s])
                
                # Convert labels to numeric
                labels_raw = [s.get('label') for s in training_samples if 'label' in s]
                label_map = {'SELL': 0, 'HOLD': 1, 'BUY': 2, 'DOWN': 0, 'UP': 2, 'AVOID': 0}
                y = np.array([label_map.get(l, 1) for l in labels_raw])
                
                # Only run if we have enough samples
                if len(X) >= 50 and len(y) >= 50:
                    # Run label quality analysis
                    logger.info("Running label quality analysis with Cleanlab...")
                    label_quality_result = self.label_quality_analyzer.analyze_labels(
                        X, y, pred_probs=None, model=None
                    )
                    
                    if label_quality_result.get('issues_found'):
                        num_issues = label_quality_result.get('num_high_confidence_issues', 0)
                        if num_issues > 0:
                            self.warnings.append(
                                f"Label quality check: {num_issues} potentially mislabeled samples detected"
                            )
                    
                    label_quality_info = {
                        'analyzed': True,
                        'num_issues': label_quality_result.get('num_high_confidence_issues', 0),
                        'mean_quality_score': label_quality_result.get('quality_scores_summary', {}).get('mean', 0)
                    }
            except Exception as e:
                logger.warning(f"Label quality analysis failed: {e}")
                label_quality_info = {'analyzed': False, 'error': str(e)}
        
        # Determine overall validity
        is_valid = len(self.issues_found) == 0
        
        return {
            'valid': is_valid,
            'issues': self.issues_found,
            'warnings': self.warnings,
            'sample_count': len(training_samples),
            'feature_count': feature_validation.get('feature_count', 0),
            'label_distribution': label_validation.get('distribution', {}),
            'missing_percentage': missing_validation.get('missing_percentage', 0),
            'outlier_percentage': outlier_validation.get('outlier_percentage', 0),
            'class_balance_ratio': balance_validation.get('ratio', 1.0),
            'low_variance_features': variance_validation.get('low_variance_features', 0),
            'label_quality': label_quality_info
        }
    
    def _validate_features(self, training_samples: List[Dict]) -> Dict:
        """Validate feature consistency"""
        issues = []
        
        # Check all samples have features
        samples_without_features = [i for i, s in enumerate(training_samples) if 'features' not in s]
        if samples_without_features:
            issues.append(f"{len(samples_without_features)} samples missing 'features' key")
        
        # Check feature dimensions are consistent
        feature_lengths = []
        for sample in training_samples:
            if 'features' in sample:
                features = sample['features']
                if isinstance(features, (list, np.ndarray)):
                    feature_lengths.append(len(features))
                else:
                    issues.append(f"Sample has invalid features type: {type(features)}")
        
        if feature_lengths:
            unique_lengths = set(feature_lengths)
            if len(unique_lengths) > 1:
                issues.append(f"Inconsistent feature dimensions: {unique_lengths}")
            
            feature_count = feature_lengths[0] if feature_lengths else 0
        else:
            feature_count = 0
        
        # Check for NaN or Inf in features
        nan_samples = []
        for i, sample in enumerate(training_samples):
            if 'features' in sample:
                features = np.array(sample['features'])
                if np.isnan(features).any() or np.isinf(features).any():
                    nan_samples.append(i)
        
        if nan_samples:
            issues.append(f"{len(nan_samples)} samples contain NaN or Inf values")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'feature_count': feature_count
        }
    
    def _validate_labels(self, training_samples: List[Dict]) -> Dict:
        """Validate label distribution"""
        issues = []
        warnings = []
        
        # Check all samples have labels
        samples_without_labels = [i for i, s in enumerate(training_samples) if 'label' not in s]
        if samples_without_labels:
            issues.append(f"{len(samples_without_labels)} samples missing 'label' key")
            return {'valid': False, 'issues': issues}
        
        # Get label distribution
        labels = [s.get('label') for s in training_samples if 'label' in s]
        label_counts = Counter(labels)
        
        # Check for valid labels
        valid_labels = {'BUY', 'SELL', 'HOLD', 'UP', 'DOWN', 'AVOID'}
        invalid_labels = [l for l in label_counts.keys() if l not in valid_labels]
        if invalid_labels:
            issues.append(f"Invalid labels found: {invalid_labels}")
        
        # Check minimum samples per class
        min_samples_per_class = min(label_counts.values()) if label_counts else 0
        if min_samples_per_class < 10:
            issues.append(f"Some classes have <10 samples (minimum: {min_samples_per_class})")
        
        # Check for single class (can't train classifier)
        if len(label_counts) < 2:
            issues.append(f"Only {len(label_counts)} unique label(s) found - need at least 2 for classification")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'distribution': dict(label_counts),
            'unique_labels': len(label_counts),
            'min_samples_per_class': min_samples_per_class
        }
    
    def _validate_missing_values(self, training_samples: List[Dict]) -> Dict:
        """Validate missing values in features"""
        issues = []
        
        total_features = 0
        missing_count = 0
        
        for sample in training_samples:
            if 'features' in sample:
                features = np.array(sample['features'])
                total_features += len(features)
                missing_count += np.isnan(features).sum() + np.isinf(features).sum()
        
        missing_percentage = (missing_count / total_features * 100) if total_features > 0 else 0
        
        if missing_percentage > 5:
            issues.append(f"High missing value percentage: {missing_percentage:.2f}%")
        
        return {
            'valid': missing_percentage <= 5,
            'issues': issues,
            'missing_percentage': missing_percentage,
            'missing_count': missing_count,
            'total_features': total_features
        }
    
    def _validate_outliers(self, training_samples: List[Dict]) -> Dict:
        """Validate outliers in features"""
        if not training_samples or 'features' not in training_samples[0]:
            return {'outlier_percentage': 0}
        
        # Extract all features
        feature_matrix = np.array([s['features'] for s in training_samples if 'features' in s])
        
        if len(feature_matrix) == 0:
            return {'outlier_percentage': 0}
        
        # Detect outliers using IQR method
        outlier_count = 0
        for col_idx in range(feature_matrix.shape[1]):
            column = feature_matrix[:, col_idx]
            if len(column) > 0 and np.std(column) > 0:
                Q1 = np.percentile(column, 25)
                Q3 = np.percentile(column, 75)
                IQR = Q3 - Q1
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = np.sum((column < lower_bound) | (column > upper_bound))
                    outlier_count += outliers
        
        total_values = feature_matrix.size
        outlier_percentage = (outlier_count / total_values * 100) if total_values > 0 else 0
        
        return {
            'outlier_percentage': outlier_percentage,
            'outlier_count': outlier_count,
            'total_values': total_values
        }
    
    def _validate_class_balance(self, training_samples: List[Dict]) -> Dict:
        """Validate class balance"""
        labels = [s.get('label') for s in training_samples if 'label' in s]
        if not labels:
            return {'balanced': False, 'ratio': 0}
        
        label_counts = Counter(labels)
        if len(label_counts) < 2:
            return {'balanced': True, 'ratio': 1.0}  # Single class, technically balanced
        
        counts = list(label_counts.values())
        max_count = max(counts)
        min_count = min(counts)
        
        ratio = max_count / min_count if min_count > 0 else float('inf')
        balanced = ratio <= 1.5  # Consider balanced if ratio <= 1.5:1
        
        return {
            'balanced': balanced,
            'ratio': ratio,
            'max_count': max_count,
            'min_count': min_count
        }
    
    def _validate_feature_variance(self, training_samples: List[Dict]) -> Dict:
        """Validate feature variance (low variance features are not useful)"""
        if not training_samples or 'features' not in training_samples[0]:
            return {'low_variance_features': 0}
        
        feature_matrix = np.array([s['features'] for s in training_samples if 'features' in s])
        
        if len(feature_matrix) == 0:
            return {'low_variance_features': 0}
        
        # Calculate variance for each feature
        variances = np.var(feature_matrix, axis=0)
        
        # Features with variance < 0.001 are considered low variance
        low_variance_threshold = 0.001
        low_variance_count = np.sum(variances < low_variance_threshold)
        
        return {
            'low_variance_features': int(low_variance_count),
            'total_features': len(variances),
            'low_variance_percentage': (low_variance_count / len(variances) * 100) if len(variances) > 0 else 0
        }
    
    def clean_training_samples(self, training_samples: List[Dict], 
                              validation_report: Dict) -> List[Dict]:
        """Clean training samples based on validation report"""
        cleaned_samples = []
        removed_count = 0
        
        for sample in training_samples:
            # Remove samples with missing features or labels
            if 'features' not in sample or 'label' not in sample:
                removed_count += 1
                continue
            
            features = np.array(sample['features'])
            
            # Remove samples with NaN or Inf
            if np.isnan(features).any() or np.isinf(features).any():
                removed_count += 1
                continue
            
            # Fill remaining missing values with 0
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            sample['features'] = features.tolist()
            
            cleaned_samples.append(sample)
        
        logger.info(f"Cleaned training samples: {len(cleaned_samples)} kept, {removed_count} removed")
        
        return cleaned_samples
