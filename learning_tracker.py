"""
Learning Tracker - Tracks all data sources and learning progress
Makes the AI system feel like a "megamind" that continuously learns
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class LearningTracker:
    """Tracks all learning sources and knowledge accumulation"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.learning_file = self.data_dir / "learning_tracker.json"
        self.data = self._load_data()
    
    def _load_data(self) -> Dict:
        """Load learning data from file"""
        if self.learning_file.exists():
            try:
                with open(self.learning_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading learning data: {e}")
        
        # Initialize default structure
        return {
            'data_sources': {
                'manual_analyses': 0,
                'auto_scans': 0,
                'verified_predictions': 0,
                'background_trainings': 0,
                'trend_change_predictions': 0,
                'verified_trend_changes': 0,
                'stocks_analyzed': set(),
                'unique_symbols': []
            },
            'knowledge': {
                'total_training_samples': 0,
                'total_predictions_made': 0,
                'total_predictions_verified': 0,
                'total_stocks_learned': 0,
                'model_retrainings': 0,
                'trend_change_accuracy': [],
                'accuracy_improvements': []
            },
            'learning_history': [],
            'first_learning_date': None,
            'last_learning_date': None
        }
    
    def save(self):
        """Save learning data to file"""
        try:
            # Convert sets to lists for JSON serialization
            data_to_save = self.data.copy()
            if 'stocks_analyzed' in data_to_save.get('data_sources', {}):
                data_to_save['data_sources']['stocks_analyzed'] = list(data_to_save['data_sources']['stocks_analyzed'])
            
            with open(self.learning_file, 'w') as f:
                json.dump(data_to_save, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving learning data: {e}")
    
    def record_manual_analysis(self, symbol: str):
        """Record a manual stock analysis"""
        if not self.data['data_sources'].get('first_learning_date'):
            self.data['data_sources']['first_learning_date'] = datetime.now().isoformat()
        
        self.data['data_sources']['manual_analyses'] += 1
        self._add_symbol(symbol)
        self._add_learning_event('manual_analysis', {'symbol': symbol})
        self.save()
        logger.info(f"📊 Learning: Manual analysis of {symbol} recorded")
    
    def record_auto_scan(self, predictions_made: int, strategy: str):
        """Record an automatic market scan"""
        if not self.data['data_sources'].get('first_learning_date'):
            self.data['data_sources']['first_learning_date'] = datetime.now().isoformat()
        
        self.data['data_sources']['auto_scans'] += 1
        self.data['knowledge']['total_predictions_made'] += predictions_made
        self._add_learning_event('auto_scan', {
            'predictions_made': predictions_made,
            'strategy': strategy
        })
        self.save()
        logger.info(f"🔍 Learning: Auto-scan recorded ({predictions_made} predictions)")
    
    def record_verified_prediction(self, was_correct: bool, strategy: str):
        """Record a verified prediction"""
        self.data['data_sources']['verified_predictions'] += 1
        self.data['knowledge']['total_predictions_verified'] += 1
        
        # Track accuracy improvements
        if was_correct:
            self.data['knowledge']['accuracy_improvements'].append({
                'timestamp': datetime.now().isoformat(),
                'type': 'correct_prediction',
                'strategy': strategy
            })
        
        self._add_learning_event('verified_prediction', {
            'was_correct': was_correct,
            'strategy': strategy
        })
        self.save()
    
    def record_background_training(self, symbol: str, samples_generated: int, strategy: str, success: bool):
        """Record background training on a stock"""
        self.data['data_sources']['background_trainings'] += 1
        self.data['knowledge']['total_training_samples'] += samples_generated
        
        if success:
            self.data['knowledge']['model_retrainings'] += 1
        
        self._add_symbol(symbol)
        self._add_learning_event('background_training', {
            'symbol': symbol,
            'samples': samples_generated,
            'strategy': strategy,
            'success': success
        })
        self.save()
        logger.info(f"🧠 Learning: Background training on {symbol} recorded ({samples_generated} samples)")
    
    def record_model_training(self, strategy: str, samples_used: int, accuracy: float):
        """Record a full model training session"""
        self.data['knowledge']['total_training_samples'] += samples_used
        self.data['knowledge']['model_retrainings'] += 1
        
        self._add_learning_event('model_training', {
            'strategy': strategy,
            'samples': samples_used,
            'accuracy': accuracy
        })
        self.save()
    
    def record_trend_change_prediction(self, symbol: str, predicted_change: str, 
                                      estimated_days: int, confidence: float):
        """Record a trend change prediction"""
        if not self.data['data_sources'].get('first_learning_date'):
            self.data['data_sources']['first_learning_date'] = datetime.now().isoformat()
        
        self.data['data_sources']['trend_change_predictions'] += 1
        self.data['knowledge']['total_predictions_made'] += 1
        self._add_symbol(symbol)
        self._add_learning_event('trend_change_prediction', {
            'symbol': symbol,
            'predicted_change': predicted_change,
            'estimated_days': estimated_days,
            'confidence': confidence
        })
        self.save()
        logger.info(f"📊 Learning: Trend change prediction for {symbol} recorded ({predicted_change})")
    
    def record_verified_trend_change(self, was_correct: bool, predicted_change: str, 
                                    actual_change: str = None, confidence: float = None):
        """Record a verified trend change prediction"""
        self.data['data_sources']['verified_trend_changes'] += 1
        self.data['knowledge']['total_predictions_verified'] += 1
        
        # Track accuracy
        self.data['knowledge']['trend_change_accuracy'].append({
            'timestamp': datetime.now().isoformat(),
            'was_correct': was_correct,
            'predicted_change': predicted_change,
            'actual_change': actual_change,
            'confidence': confidence
        })
        
        # Keep only last 1000 accuracy records
        if len(self.data['knowledge']['trend_change_accuracy']) > 1000:
            self.data['knowledge']['trend_change_accuracy'] = self.data['knowledge']['trend_change_accuracy'][-1000:]
        
        # Track accuracy improvements
        if was_correct:
            self.data['knowledge']['accuracy_improvements'].append({
                'timestamp': datetime.now().isoformat(),
                'type': 'correct_trend_change',
                'predicted_change': predicted_change
            })
        
        self._add_learning_event('verified_trend_change', {
            'was_correct': was_correct,
            'predicted_change': predicted_change,
            'actual_change': actual_change,
            'confidence': confidence
        })
        self.save()
        logger.info(f"✅ Learning: Trend change prediction verified ({'correct' if was_correct else 'incorrect'})")
    
    def _add_symbol(self, symbol: str):
        """Add a symbol to the learned stocks list"""
        if 'stocks_analyzed' not in self.data['data_sources']:
            self.data['data_sources']['stocks_analyzed'] = set()
        
        if isinstance(self.data['data_sources']['stocks_analyzed'], list):
            self.data['data_sources']['stocks_analyzed'] = set(self.data['data_sources']['stocks_analyzed'])
        
        self.data['data_sources']['stocks_analyzed'].add(symbol.upper())
        self.data['data_sources']['unique_symbols'] = sorted(list(self.data['data_sources']['stocks_analyzed']))
        self.data['knowledge']['total_stocks_learned'] = len(self.data['data_sources']['stocks_analyzed'])
    
    def _add_learning_event(self, event_type: str, details: Dict):
        """Add a learning event to history"""
        event = {
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'details': details
        }
        self.data['learning_history'].append(event)
        
        # Keep only last 1000 events
        if len(self.data['learning_history']) > 1000:
            self.data['learning_history'] = self.data['learning_history'][-1000:]
        
        self.data['data_sources']['last_learning_date'] = datetime.now().isoformat()
    
    def get_statistics(self) -> Dict:
        """Get comprehensive learning statistics"""
        stats = {
            'data_sources': {
                'manual_analyses': self.data['data_sources'].get('manual_analyses', 0),
                'auto_scans': self.data['data_sources'].get('auto_scans', 0),
                'verified_predictions': self.data['data_sources'].get('verified_predictions', 0),
                'background_trainings': self.data['data_sources'].get('background_trainings', 0),
                'unique_stocks': self.data['knowledge'].get('total_stocks_learned', 0)
            },
            'knowledge': {
                'total_training_samples': self.data['knowledge'].get('total_training_samples', 0),
                'total_predictions_made': self.data['knowledge'].get('total_predictions_made', 0),
                'total_predictions_verified': self.data['knowledge'].get('total_predictions_verified', 0),
                'model_retrainings': self.data['knowledge'].get('model_retrainings', 0),
                'accuracy_improvements': len(self.data['knowledge'].get('accuracy_improvements', []))
            },
            'learning_rate': self._calculate_learning_rate(),
            'total_learning_events': len(self.data.get('learning_history', [])),
            'first_learning': self.data['data_sources'].get('first_learning_date'),
            'last_learning': self.data['data_sources'].get('last_learning_date')
        }
        
        return stats
    
    def _calculate_learning_rate(self) -> float:
        """Calculate learning rate (events per day)"""
        first_date = self.data['data_sources'].get('first_learning_date')
        if not first_date:
            return 0.0
        
        try:
            first = datetime.fromisoformat(first_date)
            now = datetime.now()
            days = (now - first).days
            if days == 0:
                days = 1
            
            total_events = len(self.data.get('learning_history', []))
            return total_events / days
        except:
            return 0.0
    
    def get_recent_learning_activity(self, hours: int = 24) -> List[Dict]:
        """Get recent learning activity"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = []
        
        for event in self.data.get('learning_history', []):
            try:
                event_time = datetime.fromisoformat(event['timestamp'])
                if event_time >= cutoff:
                    recent.append(event)
            except:
                continue
        
        return recent
    
    def get_learning_summary(self) -> str:
        """Get a human-readable learning summary"""
        stats = self.get_statistics()
        
        summary = f"🧠 MEGAMIND LEARNING SUMMARY 🧠\n"
        summary += f"{'='*50}\n\n"
        
        summary += f"📊 DATA SOURCES:\n"
        summary += f"   • Manual Analyses: {stats['data_sources']['manual_analyses']}\n"
        summary += f"   • Auto Scans: {stats['data_sources']['auto_scans']}\n"
        summary += f"   • Verified Predictions: {stats['data_sources']['verified_predictions']}\n"
        summary += f"   • Background Trainings: {stats['data_sources']['background_trainings']}\n"
        summary += f"   • Unique Stocks Learned: {stats['data_sources']['unique_stocks']}\n\n"
        
        summary += f"💡 KNOWLEDGE ACCUMULATED:\n"
        summary += f"   • Training Samples: {stats['knowledge']['total_training_samples']:,}\n"
        summary += f"   • Predictions Made: {stats['knowledge']['total_predictions_made']:,}\n"
        summary += f"   • Predictions Verified: {stats['knowledge']['total_predictions_verified']:,}\n"
        summary += f"   • Model Retrainings: {stats['knowledge']['model_retrainings']}\n"
        summary += f"   • Accuracy Improvements: {stats['knowledge']['accuracy_improvements']}\n\n"
        
        if stats['learning_rate'] > 0:
            summary += f"⚡ LEARNING RATE: {stats['learning_rate']:.1f} events/day\n"
        
        if stats['first_learning']:
            try:
                first = datetime.fromisoformat(stats['first_learning'])
                days_active = (datetime.now() - first).days
                summary += f"📅 Active for {days_active} days\n"
            except:
                pass
        
        return summary

