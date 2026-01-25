"""
Label Quality Analyzer Module
Uses Cleanlab to find mislabeled training examples that could hurt model accuracy.

In stock market ML, labels can be noisy because:
- What looked like a "correct" BUY might have been lucky timing
- Market conditions during verification period were unusual
- The outcome was borderline (small gain/loss near threshold)

Finding and removing these noisy labels can improve model accuracy.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)

# Try to import Cleanlab
try:
    from cleanlab.filter import find_label_issues
    from cleanlab.rank import get_label_quality_scores
    from cleanlab.count import estimate_cv_predicted_probabilities
    HAS_CLEANLAB = True
except ImportError:
    HAS_CLEANLAB = False
    logger.warning("Cleanlab not available. Install with: pip install cleanlab")


class LabelQualityAnalyzer:
    """
    Analyzes training label quality using Cleanlab.
    
    Cleanlab uses confident learning to find samples where the model's
    predictions strongly disagree with the given label. These are likely
    mislabeled examples or edge cases that confuse the model.
    
    For stock predictions:
    - A "BUY" label where the model consistently predicts "SELL" might be mislabeled
    - These noisy labels hurt model performance
    - Removing them can improve accuracy by 2-5%
    """
    
    def __init__(self, data_dir: Path, strategy: str = 'trading'):
        """
        Initialize the label quality analyzer.
        
        Args:
            data_dir: Directory to store analysis results
            strategy: Trading strategy name
        """
        self.data_dir = data_dir
        self.strategy = strategy
        self.quality_dir = data_dir / "label_quality"
        self.quality_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_file = self.quality_dir / f"label_quality_{strategy}.json"
        self.issues_file = self.quality_dir / f"label_issues_{strategy}.json"
        
        # Thresholds
        self.issue_threshold = 0.3  # Samples with quality score < 0.3 are potential issues
        self.max_remove_ratio = 0.15  # Don't remove more than 15% of training data
        
        # Load previous results
        self.analysis_history = self._load_history()
    
    def _load_history(self) -> List[Dict]:
        """Load analysis history"""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading label quality history: {e}")
        return []
    
    def _save_results(self, results: Dict):
        """Save analysis results"""
        try:
            self.analysis_history.append(results)
            # Keep last 50 analyses
            history_to_save = self.analysis_history[-50:]
            with open(self.results_file, 'w') as f:
                json.dump(history_to_save, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving label quality results: {e}")
    
    def analyze_labels(self, X: np.ndarray, y: np.ndarray, 
                       pred_probs: Optional[np.ndarray] = None,
                       model=None) -> Dict:
        """
        Analyze label quality in training data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels array (n_samples,)
            pred_probs: Optional predicted probabilities from cross-validation
                       Shape: (n_samples, n_classes)
            model: Optional model to use for generating pred_probs if not provided
        
        Returns:
            Dict with analysis results:
            - issues_found: bool
            - num_issues: int
            - issue_indices: list of sample indices with potential label issues
            - quality_scores: quality score for each sample
            - recommendation: str
        """
        if not HAS_CLEANLAB:
            return {
                'issues_found': False,
                'num_issues': 0,
                'issue_indices': [],
                'quality_scores': [],
                'recommendation': 'Install cleanlab to enable label quality analysis',
                'error': 'Cleanlab not installed'
            }
        
        if len(X) < 50:
            return {
                'issues_found': False,
                'num_issues': 0,
                'issue_indices': [],
                'quality_scores': [],
                'recommendation': 'Need at least 50 samples for label quality analysis',
                'error': 'Insufficient samples'
            }
        
        try:
            # Ensure labels are integers
            y = np.array(y).astype(int)
            
            # Get number of classes
            n_classes = len(np.unique(y))
            if n_classes < 2:
                return {
                    'issues_found': False,
                    'num_issues': 0,
                    'issue_indices': [],
                    'quality_scores': [],
                    'recommendation': 'Need at least 2 classes for analysis',
                    'error': 'Single class in labels'
                }
            
            # Generate predicted probabilities if not provided
            if pred_probs is None:
                if model is not None:
                    # Use cross-validation to get out-of-sample predictions
                    try:
                        # Cleanlab 2.x API - doesn't use cv keyword
                        from sklearn.model_selection import cross_val_predict
                        
                        # Clone the model for cross-validation
                        from sklearn.base import clone
                        model_clone = clone(model)
                        
                        # Use sklearn's cross_val_predict for probability predictions
                        pred_probs = cross_val_predict(
                            model_clone, X, y, cv=5, method='predict_proba'
                        )
                    except Exception as e:
                        logger.warning(f"CV prediction failed: {e}, trying to fit and use model.predict_proba")
                        try:
                            # Fit the model first if not fitted
                            from sklearn.base import clone
                            model_clone = clone(model)
                            model_clone.fit(X, y)
                            pred_probs = model_clone.predict_proba(X)
                        except Exception as e2:
                            return {
                                'issues_found': False,
                                'num_issues': 0,
                                'issue_indices': [],
                                'quality_scores': [],
                                'recommendation': f'Could not generate predictions: {e2}',
                                'error': str(e2)
                            }
                else:
                    return {
                        'issues_found': False,
                        'num_issues': 0,
                        'issue_indices': [],
                        'quality_scores': [],
                        'recommendation': 'Provide pred_probs or model for analysis',
                        'error': 'No predictions available'
                    }
            
            # Ensure pred_probs has correct shape
            if len(pred_probs.shape) == 1:
                # Binary classification with single column - expand to 2 columns
                pred_probs = np.column_stack([1 - pred_probs, pred_probs])
            
            # Find label issues
            issue_indices = find_label_issues(
                labels=y,
                pred_probs=pred_probs,
                return_indices_ranked_by='self_confidence'
            )
            
            # Get quality scores for all samples
            quality_scores = get_label_quality_scores(
                labels=y,
                pred_probs=pred_probs
            )
            
            # Identify high-confidence issues (low quality scores)
            high_confidence_issues = [
                idx for idx in issue_indices 
                if quality_scores[idx] < self.issue_threshold
            ]
            
            # Calculate statistics
            num_issues = len(issue_indices)
            num_high_conf_issues = len(high_confidence_issues)
            issue_ratio = num_issues / len(y)
            
            # Generate recommendation
            if num_high_conf_issues > len(y) * 0.1:
                recommendation = (
                    f"⚠️ HIGH NOISE: {num_high_conf_issues} samples ({num_high_conf_issues/len(y)*100:.1f}%) "
                    f"have low label quality. Consider reviewing or removing them."
                )
            elif num_high_conf_issues > 0:
                recommendation = (
                    f"Found {num_high_conf_issues} potentially mislabeled samples. "
                    f"Removing them may improve accuracy."
                )
            else:
                recommendation = "✓ Label quality looks good. No obvious issues found."
            
            # Analyze label distribution of issues
            issue_label_dist = {}
            for idx in issue_indices[:50]:  # Analyze top 50 issues
                label = int(y[idx])
                issue_label_dist[label] = issue_label_dist.get(label, 0) + 1
            
            # Prepare results
            results = {
                'issues_found': num_issues > 0,
                'num_issues': num_issues,
                'num_high_confidence_issues': num_high_conf_issues,
                'issue_ratio': float(issue_ratio),
                'issue_indices': [int(i) for i in issue_indices[:100]],  # Top 100
                'high_confidence_issue_indices': [int(i) for i in high_confidence_issues[:50]],
                'quality_scores_summary': {
                    'mean': float(np.mean(quality_scores)),
                    'std': float(np.std(quality_scores)),
                    'min': float(np.min(quality_scores)),
                    'median': float(np.median(quality_scores)),
                    'below_threshold': int(np.sum(quality_scores < self.issue_threshold))
                },
                'issue_label_distribution': issue_label_dist,
                'recommendation': recommendation,
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(y),
                'strategy': self.strategy
            }
            
            # Save results
            self._save_results(results)
            
            # Also save detailed issue indices for later use
            try:
                issue_details = {
                    'timestamp': datetime.now().isoformat(),
                    'issue_indices': [int(i) for i in issue_indices],
                    'quality_scores': [float(s) for s in quality_scores]
                }
                with open(self.issues_file, 'w') as f:
                    json.dump(issue_details, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not save detailed issue indices: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing label quality: {e}")
            import traceback
            traceback.print_exc()
            return {
                'issues_found': False,
                'num_issues': 0,
                'issue_indices': [],
                'quality_scores': [],
                'recommendation': 'Error during analysis',
                'error': str(e)
            }
    
    def clean_training_data(self, X: np.ndarray, y: np.ndarray,
                           pred_probs: Optional[np.ndarray] = None,
                           model=None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Remove potentially mislabeled samples from training data.
        
        Args:
            X: Feature matrix
            y: Labels array
            pred_probs: Optional predicted probabilities
            model: Optional model for generating predictions
            
        Returns:
            Tuple of (X_clean, y_clean, removal_info)
        """
        # First analyze the labels
        analysis = self.analyze_labels(X, y, pred_probs, model)
        
        if analysis.get('error'):
            logger.warning(f"Label analysis failed: {analysis['error']}")
            return X, y, {'removed': 0, 'reason': 'Analysis failed'}
        
        # Get high-confidence issues to remove
        issues_to_remove = analysis.get('high_confidence_issue_indices', [])
        
        # Don't remove too many samples
        max_to_remove = int(len(y) * self.max_remove_ratio)
        if len(issues_to_remove) > max_to_remove:
            logger.warning(f"Limiting removal from {len(issues_to_remove)} to {max_to_remove} samples")
            issues_to_remove = issues_to_remove[:max_to_remove]
        
        if not issues_to_remove:
            return X, y, {'removed': 0, 'reason': 'No issues to remove'}
        
        # Create mask for keeping samples
        keep_mask = np.ones(len(y), dtype=bool)
        keep_mask[issues_to_remove] = False
        
        X_clean = X[keep_mask]
        y_clean = y[keep_mask]
        
        removal_info = {
            'removed': len(issues_to_remove),
            'original_size': len(y),
            'new_size': len(y_clean),
            'removal_ratio': len(issues_to_remove) / len(y),
            'removed_indices': issues_to_remove
        }
        
        logger.info(f"Removed {removal_info['removed']} potentially mislabeled samples "
                   f"({removal_info['removal_ratio']*100:.1f}%)")
        
        return X_clean, y_clean, removal_info
    
    def get_quality_report(self) -> Dict:
        """Get summary of label quality status."""
        if not self.analysis_history:
            return {
                'status': 'no_data',
                'message': 'No label quality analysis performed yet'
            }
        
        recent = self.analysis_history[-1]
        
        return {
            'status': 'analyzed',
            'last_analysis': recent.get('timestamp'),
            'total_samples': recent.get('total_samples', 0),
            'issues_found': recent.get('num_issues', 0),
            'high_confidence_issues': recent.get('num_high_confidence_issues', 0),
            'mean_quality_score': recent.get('quality_scores_summary', {}).get('mean', 0),
            'recommendation': recent.get('recommendation', ''),
            'analyses_performed': len(self.analysis_history)
        }
    
    def generate_quality_report(self) -> str:
        """Generate a human-readable label quality report."""
        if not self.analysis_history:
            return "No label quality analysis has been performed yet."
        
        recent = self.analysis_history[-1]
        quality_summary = recent.get('quality_scores_summary', {})
        
        report_lines = [
            "=" * 50,
            "LABEL QUALITY ANALYSIS REPORT",
            f"Strategy: {self.strategy}",
            f"Timestamp: {recent.get('timestamp', 'Unknown')}",
            "=" * 50,
            "",
            f"Total Samples Analyzed: {recent.get('total_samples', 0)}",
            f"Potential Label Issues: {recent.get('num_issues', 0)}",
            f"High-Confidence Issues: {recent.get('num_high_confidence_issues', 0)}",
            "",
            "Quality Score Statistics:",
            f"  Mean: {quality_summary.get('mean', 0):.3f}",
            f"  Median: {quality_summary.get('median', 0):.3f}",
            f"  Std Dev: {quality_summary.get('std', 0):.3f}",
            f"  Min: {quality_summary.get('min', 0):.3f}",
            f"  Below Threshold ({self.issue_threshold}): {quality_summary.get('below_threshold', 0)}",
            "",
            "Issue Distribution by Label:",
        ]
        
        label_dist = recent.get('issue_label_distribution', {})
        label_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        for label, count in label_dist.items():
            label_name = label_names.get(int(label), f'Label {label}')
            report_lines.append(f"  {label_name}: {count} issues")
        
        report_lines.extend([
            "",
            "RECOMMENDATION:",
            recent.get('recommendation', 'No recommendation'),
            "=" * 50
        ])
        
        return "\n".join(report_lines)
