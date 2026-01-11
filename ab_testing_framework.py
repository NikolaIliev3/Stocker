"""
A/B Testing Framework for Model Versions
Allows testing new model versions alongside current model with traffic splitting
"""
import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ABTestingFramework:
    """Manages A/B testing of model versions"""
    
    def __init__(self, data_dir: Path, strategy: str):
        self.data_dir = data_dir
        self.strategy = strategy
        self.ab_tests_file = data_dir / f"ab_tests_{strategy}.json"
        self.ab_tests = self._load_ab_tests()
    
    def _load_ab_tests(self) -> Dict:
        """Load A/B test configurations"""
        if self.ab_tests_file.exists():
            try:
                with open(self.ab_tests_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading A/B tests: {e}")
        
        return {
            'active_tests': {},
            'completed_tests': {},
            'test_history': []
        }
    
    def _save_ab_tests(self):
        """Save A/B test configurations"""
        try:
            with open(self.ab_tests_file, 'w') as f:
                json.dump(self.ab_tests, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving A/B tests: {e}")
    
    def create_ab_test(self, test_name: str, control_version: str, 
                      treatment_version: str, traffic_split: float = 0.5,
                      min_samples: int = 100) -> str:
        """Create a new A/B test
        
        Args:
            test_name: Name of the test
            control_version: Version name for control group (current model)
            treatment_version: Version name for treatment group (new model)
            traffic_split: Fraction of traffic to treatment (0.0-1.0)
            min_samples: Minimum samples per group before evaluation
        
        Returns:
            Test ID
        """
        test_id = f"{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        test_config = {
            'test_id': test_id,
            'test_name': test_name,
            'strategy': self.strategy,
            'control_version': control_version,
            'treatment_version': treatment_version,
            'traffic_split': traffic_split,
            'min_samples': min_samples,
            'created_at': datetime.now().isoformat(),
            'status': 'active',
            'control_results': {'total': 0, 'correct': 0, 'accuracy': 0},
            'treatment_results': {'total': 0, 'correct': 0, 'accuracy': 0},
            'assignments': {}  # Track which predictions go to which group
        }
        
        self.ab_tests['active_tests'][test_id] = test_config
        self._save_ab_tests()
        
        logger.info(f"Created A/B test {test_id}: {control_version} vs {treatment_version} "
                   f"(split: {traffic_split*100:.0f}% treatment)")
        
        return test_id
    
    def assign_to_group(self, test_id: str, prediction_id: str) -> str:
        """Assign a prediction to control or treatment group
        
        Args:
            test_id: A/B test ID
            prediction_id: Unique prediction identifier
        
        Returns:
            'control' or 'treatment'
        """
        if test_id not in self.ab_tests['active_tests']:
            return 'control'  # Default to control if test not found
        
        test = self.ab_tests['active_tests'][test_id]
        
        # Check if already assigned
        if prediction_id in test['assignments']:
            return test['assignments'][prediction_id]
        
        # Random assignment based on traffic split
        traffic_split = test.get('traffic_split', 0.5)
        assignment = 'treatment' if random.random() < traffic_split else 'control'
        
        test['assignments'][prediction_id] = assignment
        self._save_ab_tests()
        
        return assignment
    
    def record_outcome(self, test_id: str, prediction_id: str, 
                      was_correct: bool, group: str = None):
        """Record outcome for A/B test
        
        Args:
            test_id: A/B test ID
            prediction_id: Prediction identifier
            was_correct: Whether prediction was correct
            group: 'control' or 'treatment' (if None, looks up from assignments)
        """
        if test_id not in self.ab_tests['active_tests']:
            return
        
        test = self.ab_tests['active_tests'][test_id]
        
        # Get group if not provided
        if group is None:
            group = test['assignments'].get(prediction_id, 'control')
        
        # Update results
        if group == 'control':
            results = test['control_results']
        else:
            results = test['treatment_results']
        
        results['total'] += 1
        if was_correct:
            results['correct'] += 1
        
        results['accuracy'] = (results['correct'] / results['total'] * 100) if results['total'] > 0 else 0
        
        self._save_ab_tests()
        
        # Check if test should be evaluated
        self._evaluate_test(test_id)
    
    def _evaluate_test(self, test_id: str):
        """Evaluate A/B test and determine winner"""
        if test_id not in self.ab_tests['active_tests']:
            return
        
        test = self.ab_tests['active_tests'][test_id]
        min_samples = test.get('min_samples', 100)
        
        control_results = test['control_results']
        treatment_results = test['treatment_results']
        
        # Need minimum samples in both groups
        if control_results['total'] < min_samples or treatment_results['total'] < min_samples:
            return
        
        control_acc = control_results['accuracy']
        treatment_acc = treatment_results['accuracy']
        accuracy_diff = treatment_acc - control_acc
        
        # Statistical significance check (simple version)
        # For production, use proper statistical tests (chi-square, t-test, etc.)
        is_significant = abs(accuracy_diff) > 5.0  # 5% difference threshold
        
        # Determine winner
        if is_significant:
            if treatment_acc > control_acc:
                winner = 'treatment'
                improvement = accuracy_diff
            else:
                winner = 'control'
                improvement = -accuracy_diff
        else:
            winner = 'inconclusive'
            improvement = 0
        
        # Store evaluation
        test['evaluation'] = {
            'control_accuracy': control_acc,
            'treatment_accuracy': treatment_acc,
            'difference': accuracy_diff,
            'is_significant': is_significant,
            'winner': winner,
            'improvement': improvement,
            'evaluated_at': datetime.now().isoformat()
        }
        
        self._save_ab_tests()
        
        logger.info(f"A/B test {test_id} evaluation: "
                   f"Control {control_acc:.1f}% vs Treatment {treatment_acc:.1f}% "
                   f"(diff: {accuracy_diff:+.1f}%, winner: {winner})")
    
    def get_active_tests(self) -> List[Dict]:
        """Get all active A/B tests"""
        return list(self.ab_tests['active_tests'].values())
    
    def complete_test(self, test_id: str, promote_winner: bool = True) -> Dict:
        """Complete an A/B test and optionally promote winner
        
        Args:
            test_id: Test ID to complete
            promote_winner: If True, promote winning version to production
        
        Returns:
            Test results
        """
        if test_id not in self.ab_tests['active_tests']:
            return {'error': 'Test not found'}
        
        test = self.ab_tests['active_tests'][test_id]
        test['status'] = 'completed'
        test['completed_at'] = datetime.now().isoformat()
        
        # Move to completed
        self.ab_tests['completed_tests'][test_id] = test
        del self.ab_tests['active_tests'][test_id]
        
        self._save_ab_tests()
        
        evaluation = test.get('evaluation', {})
        winner = evaluation.get('winner', 'inconclusive')
        
        result = {
            'test_id': test_id,
            'winner': winner,
            'control_accuracy': evaluation.get('control_accuracy', 0),
            'treatment_accuracy': evaluation.get('treatment_accuracy', 0),
            'improvement': evaluation.get('improvement', 0),
            'promote_version': None
        }
        
        if promote_winner and winner != 'inconclusive':
            if winner == 'treatment':
                result['promote_version'] = test['treatment_version']
                logger.info(f"✅ A/B test {test_id} complete: Promoting {test['treatment_version']} to production")
            else:
                result['promote_version'] = test['control_version']
                logger.info(f"✅ A/B test {test_id} complete: Keeping {test['control_version']} in production")
        
        return result
    
    def get_test_results(self, test_id: str) -> Optional[Dict]:
        """Get results for a specific test"""
        if test_id in self.ab_tests['active_tests']:
            return self.ab_tests['active_tests'][test_id]
        elif test_id in self.ab_tests['completed_tests']:
            return self.ab_tests['completed_tests'][test_id]
        return None
