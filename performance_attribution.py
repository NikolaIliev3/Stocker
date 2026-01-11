"""
Performance Attribution Analysis
Breaks down model performance by strategy, regime, sector, confidence levels, etc.
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class PerformanceAttribution:
    """Analyzes performance by different dimensions"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.attribution_file = data_dir / "performance_attribution.json"
        self.attribution_data = self._load_attribution_data()
    
    def _load_attribution_data(self) -> Dict:
        """Load attribution data from file"""
        if self.attribution_file.exists():
            try:
                with open(self.attribution_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading attribution data: {e}")
        
        return {
            'by_strategy': {},
            'by_regime': {},
            'by_confidence': {},
            'by_sector': {},
            'by_action': {},
            'by_time_period': {},
            'last_updated': None
        }
    
    def _save_attribution_data(self):
        """Save attribution data to file"""
        try:
            self.attribution_data['last_updated'] = datetime.now().isoformat()
            with open(self.attribution_file, 'w') as f:
                json.dump(self.attribution_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving attribution data: {e}")
    
    def record_prediction_outcome(self, prediction: Dict, outcome: Dict):
        """Record a prediction and its outcome for attribution analysis
        
        Args:
            prediction: Prediction dict with action, confidence, strategy, etc.
            outcome: Outcome dict with was_correct, actual_price_change, etc.
        """
        strategy = prediction.get('strategy', 'unknown')
        action = prediction.get('action', 'UNKNOWN')
        confidence = prediction.get('confidence', 50)
        regime = prediction.get('market_regime', 'unknown')
        sector = prediction.get('sector', 'unknown')
        was_correct = outcome.get('was_correct', False)
        timestamp = prediction.get('timestamp', datetime.now().isoformat())
        
        # Update by strategy
        if strategy not in self.attribution_data['by_strategy']:
            self.attribution_data['by_strategy'][strategy] = {
                'total': 0, 'correct': 0, 'incorrect': 0, 'accuracy': 0
            }
        self.attribution_data['by_strategy'][strategy]['total'] += 1
        if was_correct:
            self.attribution_data['by_strategy'][strategy]['correct'] += 1
        else:
            self.attribution_data['by_strategy'][strategy]['incorrect'] += 1
        self._update_accuracy(self.attribution_data['by_strategy'][strategy])
        
        # Update by regime
        if regime not in self.attribution_data['by_regime']:
            self.attribution_data['by_regime'][regime] = {
                'total': 0, 'correct': 0, 'incorrect': 0, 'accuracy': 0
            }
        self.attribution_data['by_regime'][regime]['total'] += 1
        if was_correct:
            self.attribution_data['by_regime'][regime]['correct'] += 1
        else:
            self.attribution_data['by_regime'][regime]['incorrect'] += 1
        self._update_accuracy(self.attribution_data['by_regime'][regime])
        
        # Update by confidence level (buckets: 0-50, 50-70, 70-85, 85-100)
        confidence_bucket = self._get_confidence_bucket(confidence)
        if confidence_bucket not in self.attribution_data['by_confidence']:
            self.attribution_data['by_confidence'][confidence_bucket] = {
                'total': 0, 'correct': 0, 'incorrect': 0, 'accuracy': 0
            }
        self.attribution_data['by_confidence'][confidence_bucket]['total'] += 1
        if was_correct:
            self.attribution_data['by_confidence'][confidence_bucket]['correct'] += 1
        else:
            self.attribution_data['by_confidence'][confidence_bucket]['incorrect'] += 1
        self._update_accuracy(self.attribution_data['by_confidence'][confidence_bucket])
        
        # Update by action
        if action not in self.attribution_data['by_action']:
            self.attribution_data['by_action'][action] = {
                'total': 0, 'correct': 0, 'incorrect': 0, 'accuracy': 0
            }
        self.attribution_data['by_action'][action]['total'] += 1
        if was_correct:
            self.attribution_data['by_action'][action]['correct'] += 1
        else:
            self.attribution_data['by_action'][action]['incorrect'] += 1
        self._update_accuracy(self.attribution_data['by_action'][action])
        
        # Update by sector
        if sector != 'unknown':
            if sector not in self.attribution_data['by_sector']:
                self.attribution_data['by_sector'][sector] = {
                    'total': 0, 'correct': 0, 'incorrect': 0, 'accuracy': 0
                }
            self.attribution_data['by_sector'][sector]['total'] += 1
            if was_correct:
                self.attribution_data['by_sector'][sector]['correct'] += 1
            else:
                self.attribution_data['by_sector'][sector]['incorrect'] += 1
            self._update_accuracy(self.attribution_data['by_sector'][sector])
        
        # Update by time period (month)
        try:
            pred_date = datetime.fromisoformat(timestamp)
            period_key = pred_date.strftime("%Y-%m")
            if period_key not in self.attribution_data['by_time_period']:
                self.attribution_data['by_time_period'][period_key] = {
                    'total': 0, 'correct': 0, 'incorrect': 0, 'accuracy': 0
                }
            self.attribution_data['by_time_period'][period_key]['total'] += 1
            if was_correct:
                self.attribution_data['by_time_period'][period_key]['correct'] += 1
            else:
                self.attribution_data['by_time_period'][period_key]['incorrect'] += 1
            self._update_accuracy(self.attribution_data['by_time_period'][period_key])
        except:
            pass
        
        self._save_attribution_data()
    
    def _get_confidence_bucket(self, confidence: float) -> str:
        """Get confidence bucket for grouping"""
        if confidence < 50:
            return "0-50"
        elif confidence < 70:
            return "50-70"
        elif confidence < 85:
            return "70-85"
        else:
            return "85-100"
    
    def _update_accuracy(self, stats: Dict):
        """Update accuracy in stats dict"""
        total = stats.get('total', 0)
        correct = stats.get('correct', 0)
        stats['accuracy'] = (correct / total * 100) if total > 0 else 0
    
    def get_attribution_report(self) -> Dict:
        """Get comprehensive attribution report"""
        report = {
            'by_strategy': self._format_attribution(self.attribution_data['by_strategy']),
            'by_regime': self._format_attribution(self.attribution_data['by_regime']),
            'by_confidence': self._format_attribution(self.attribution_data['by_confidence']),
            'by_action': self._format_attribution(self.attribution_data['by_action']),
            'by_sector': self._format_attribution(self.attribution_data['by_sector']),
            'by_time_period': self._format_attribution(self.attribution_data['by_time_period']),
            'insights': self._generate_insights()
        }
        
        return report
    
    def _format_attribution(self, data: Dict) -> Dict:
        """Format attribution data for display"""
        formatted = {}
        for key, stats in data.items():
            formatted[key] = {
                'accuracy': stats.get('accuracy', 0),
                'total': stats.get('total', 0),
                'correct': stats.get('correct', 0),
                'incorrect': stats.get('incorrect', 0)
            }
        return formatted
    
    def _generate_insights(self) -> List[str]:
        """Generate insights from attribution data"""
        insights = []
        
        # Best/worst strategy
        strategy_data = self.attribution_data['by_strategy']
        if strategy_data:
            best_strategy = max(strategy_data.items(), key=lambda x: x[1].get('accuracy', 0))
            worst_strategy = min(strategy_data.items(), key=lambda x: x[1].get('accuracy', 0))
            if best_strategy[1].get('total', 0) >= 10:
                insights.append(f"Best performing strategy: {best_strategy[0]} ({best_strategy[1].get('accuracy', 0):.1f}% accuracy)")
            if worst_strategy[1].get('total', 0) >= 10:
                insights.append(f"Worst performing strategy: {worst_strategy[0]} ({worst_strategy[1].get('accuracy', 0):.1f}% accuracy)")
        
        # Best/worst regime
        regime_data = self.attribution_data['by_regime']
        if regime_data:
            best_regime = max(regime_data.items(), key=lambda x: x[1].get('accuracy', 0))
            if best_regime[1].get('total', 0) >= 10:
                insights.append(f"Best performing regime: {best_regime[0]} ({best_regime[1].get('accuracy', 0):.1f}% accuracy)")
        
        # Confidence calibration
        confidence_data = self.attribution_data['by_confidence']
        if confidence_data:
            high_conf = confidence_data.get('85-100', {})
            low_conf = confidence_data.get('0-50', {})
            if high_conf.get('total', 0) >= 10 and low_conf.get('total', 0) >= 10:
                high_acc = high_conf.get('accuracy', 0)
                low_acc = low_conf.get('accuracy', 0)
                if high_acc > low_acc + 10:
                    insights.append(f"Confidence calibration working: High confidence ({high_acc:.1f}%) > Low confidence ({low_acc:.1f}%)")
                else:
                    insights.append(f"⚠️ Confidence calibration issue: High confidence ({high_acc:.1f}%) not much better than low ({low_acc:.1f}%)")
        
        # Best/worst action
        action_data = self.attribution_data['by_action']
        if action_data:
            best_action = max(action_data.items(), key=lambda x: x[1].get('accuracy', 0))
            worst_action = min(action_data.items(), key=lambda x: x[1].get('accuracy', 0))
            if best_action[1].get('total', 0) >= 10:
                insights.append(f"Most accurate action: {best_action[0]} ({best_action[1].get('accuracy', 0):.1f}% accuracy)")
            if worst_action[1].get('total', 0) >= 10 and worst_action[0] != best_action[0]:
                insights.append(f"Least accurate action: {worst_action[0]} ({worst_action[1].get('accuracy', 0):.1f}% accuracy)")
        
        return insights
    
    def get_performance_trends(self, days: int = 30) -> Dict:
        """Get performance trends over time"""
        cutoff = datetime.now() - timedelta(days=days)
        
        recent_periods = {}
        for period, stats in self.attribution_data['by_time_period'].items():
            try:
                period_date = datetime.strptime(period, "%Y-%m")
                if period_date >= cutoff:
                    recent_periods[period] = stats
            except:
                continue
        
        # Calculate trend
        if len(recent_periods) >= 2:
            periods_sorted = sorted(recent_periods.items())
            accuracies = [p[1].get('accuracy', 0) for p in periods_sorted]
            
            # Simple linear trend
            if len(accuracies) >= 2:
                trend = accuracies[-1] - accuracies[0]
                trend_direction = 'improving' if trend > 0 else 'declining' if trend < 0 else 'stable'
            else:
                trend = 0
                trend_direction = 'stable'
        else:
            trend = 0
            trend_direction = 'insufficient_data'
        
        return {
            'trend': trend,
            'trend_direction': trend_direction,
            'recent_periods': {k: v.get('accuracy', 0) for k, v in recent_periods.items()},
            'period_count': len(recent_periods)
        }
