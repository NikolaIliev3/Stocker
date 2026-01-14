"""
Learning Tracker - Tracks all data sources and learning progress
Makes the AI system feel like a "megamind" that continuously learns
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from calibration_manager import CalibrationManager

logger = logging.getLogger(__name__)


class LearningTracker:
    """Tracks all learning sources and knowledge accumulation"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.learning_file = self.data_dir / "learning_tracker.json"
        self.calibration_manager = CalibrationManager(data_dir)
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
                'peak_detections': 0,
                'bottom_detections': 0,
                'trend_change_detections': 0,
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
                'peak_bottom_accuracy': [],
                'accuracy_improvements': []
            },
            'learning_history': [],
            'pending_peak_verifications': [],  # Peak detections waiting to be verified
            'pending_bottom_verifications': [],  # Bottom detections waiting to be verified
            'pending_buy_opportunities': [],  # Buy opportunities waiting to be verified
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
    
    def record_verified_prediction(self, was_correct: bool, strategy: str, 
                                  prediction_data: Dict = None, actual_outcome: Dict = None):
        """Record a verified prediction"""
        self.data['data_sources']['verified_predictions'] += 1
        self.data['knowledge']['total_predictions_verified'] += 1
        
        # Update calibration if data is available
        if prediction_data and actual_outcome:
            self.calibration_manager.update_from_verified_prediction(prediction_data, actual_outcome)
        
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
    
    def record_peak_detection(self, symbol: str, price: float, confidence: float, reasoning: str, current_price: float = None):
        """Record a peak detection event for Megamind learning and add to pending verifications
        
        Args:
            symbol: Stock symbol
            price: Peak price detected
            confidence: Detection confidence
            reasoning: Detection reasoning
            current_price: Current price when detection was made (for tracking)
        """
        if not self.data['data_sources'].get('first_learning_date'):
            self.data['data_sources']['first_learning_date'] = datetime.now().isoformat()
        
        # Ensure pending_peak_verifications exists
        if 'pending_peak_verifications' not in self.data:
            self.data['pending_peak_verifications'] = []
        
        self.data['data_sources']['peak_detections'] = self.data['data_sources'].get('peak_detections', 0) + 1
        self._add_symbol(symbol)
        
        # Add to pending verifications (will be checked in 7-10 days)
        detection_id = len(self.data['pending_peak_verifications']) + 1
        verification_record = {
            'id': detection_id,
            'symbol': symbol.upper(),
            'peak_price': price,
            'current_price_at_detection': current_price or price,
            'confidence': confidence,
            'reasoning': reasoning,
            'detection_date': datetime.now().isoformat(),
            'verification_due_date': (datetime.now() + timedelta(days=7)).isoformat(),  # Verify after 7 days
            'verified': False
        }
        self.data['pending_peak_verifications'].append(verification_record)
        
        # Keep only last 500 pending verifications
        if len(self.data['pending_peak_verifications']) > 500:
            self.data['pending_peak_verifications'] = self.data['pending_peak_verifications'][-500:]
        
        self._add_learning_event('peak_detection', {
            'symbol': symbol,
            'price': price,
            'confidence': confidence,
            'reasoning': reasoning,
            'verification_id': detection_id
        })
        self.save()
        logger.info(f"🧠 Learning: Peak detection for {symbol} recorded (${price:.2f}, {confidence:.1f}% confidence) - will verify in 7 days")
    
    def record_bottom_detection(self, symbol: str, price: float, confidence: float, reasoning: str, current_price: float = None):
        """Record a bottom detection event for Megamind learning and add to pending verifications
        
        Args:
            symbol: Stock symbol
            price: Bottom price detected
            confidence: Detection confidence
            reasoning: Detection reasoning
            current_price: Current price when detection was made (for tracking)
        """
        if not self.data['data_sources'].get('first_learning_date'):
            self.data['data_sources']['first_learning_date'] = datetime.now().isoformat()
        
        # Ensure pending_bottom_verifications exists
        if 'pending_bottom_verifications' not in self.data:
            self.data['pending_bottom_verifications'] = []
        
        self.data['data_sources']['bottom_detections'] = self.data['data_sources'].get('bottom_detections', 0) + 1
        self._add_symbol(symbol)
        
        # Add to pending verifications (will be checked in 7-10 days)
        detection_id = len(self.data['pending_bottom_verifications']) + 1
        verification_record = {
            'id': detection_id,
            'symbol': symbol.upper(),
            'bottom_price': price,
            'current_price_at_detection': current_price or price,
            'confidence': confidence,
            'reasoning': reasoning,
            'detection_date': datetime.now().isoformat(),
            'verification_due_date': (datetime.now() + timedelta(days=7)).isoformat(),  # Verify after 7 days
            'verified': False
        }
        self.data['pending_bottom_verifications'].append(verification_record)
        
        # Keep only last 500 pending verifications
        if len(self.data['pending_bottom_verifications']) > 500:
            self.data['pending_bottom_verifications'] = self.data['pending_bottom_verifications'][-500:]
        
        self._add_learning_event('bottom_detection', {
            'symbol': symbol,
            'price': price,
            'confidence': confidence,
            'reasoning': reasoning,
            'verification_id': detection_id
        })
        self.save()
        logger.info(f"🧠 Learning: Bottom detection for {symbol} recorded (${price:.2f}, {confidence:.1f}% confidence) - will verify in 7 days")
    
    def record_trend_change_detection(self, symbol: str, old_trend: str, new_trend: str, price: float, confidence: float):
        """Record a trend change detection event for Megamind learning"""
        if not self.data['data_sources'].get('first_learning_date'):
            self.data['data_sources']['first_learning_date'] = datetime.now().isoformat()
        
        self.data['data_sources']['trend_change_detections'] = self.data['data_sources'].get('trend_change_detections', 0) + 1
        self._add_symbol(symbol)
        self._add_learning_event('trend_change_detection', {
            'symbol': symbol,
            'old_trend': old_trend,
            'new_trend': new_trend,
            'price': price,
            'confidence': confidence
        })
        self.save()
        logger.info(f"🧠 Learning: Trend change detection for {symbol} recorded ({old_trend} → {new_trend} at ${price:.2f})")
    
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
                'buy_opportunities': self.data['data_sources'].get('buy_opportunity_predictions', 0),
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
    
    def verify_peak_detection(self, detection_record: Dict, current_price: float, current_date: datetime = None) -> Dict:
        """Verify if a peak detection was correct by checking if price actually reversed
        
        Args:
            detection_record: The peak detection record to verify
            current_price: Current price of the stock
            current_date: Current date (defaults to now)
            
        Returns:
            Dict with verification results:
            {
                'was_correct': bool,
                'reversal_confirmed': bool,  # Price actually declined from peak
                'false_positive': bool,  # Price continued rising above peak
                'peak_price': float,
                'current_price': float,
                'price_change_pct': float,
                'verification_date': str
            }
        """
        if current_date is None:
            current_date = datetime.now()
        
        peak_price = detection_record.get('peak_price', 0)
        if peak_price == 0:
            return {'was_correct': False, 'error': 'Invalid peak price'}
        
        price_change_pct = ((current_price - peak_price) / peak_price) * 100
        
        # Peak detection is CORRECT if:
        # 1. Price declined significantly from peak (3%+ decrease) OR
        # 2. Price is at least 1.5% below peak (smaller threshold for confirmation)
        reversal_confirmed = price_change_pct <= -1.5
        
        # Peak detection is FALSE POSITIVE if:
        # Price exceeded the peak price by 2%+ (price continued rising)
        false_positive = price_change_pct >= 2.0
        
        # Determine if detection was correct
        was_correct = reversal_confirmed and not false_positive
        
        result = {
            'was_correct': was_correct,
            'reversal_confirmed': reversal_confirmed,
            'false_positive': false_positive,
            'peak_price': peak_price,
            'current_price': current_price,
            'price_change_pct': price_change_pct,
            'verification_date': current_date.isoformat()
        }
        
        return result
    
    def verify_bottom_detection(self, detection_record: Dict, current_price: float, current_date: datetime = None) -> Dict:
        """Verify if a bottom detection was correct by checking if price actually reversed
        
        Args:
            detection_record: The bottom detection record to verify
            current_price: Current price of the stock
            current_date: Current date (defaults to now)
            
        Returns:
            Dict with verification results:
            {
                'was_correct': bool,
                'reversal_confirmed': bool,  # Price actually rose from bottom
                'false_positive': bool,  # Price continued falling below bottom
                'bottom_price': float,
                'current_price': float,
                'price_change_pct': float,
                'verification_date': str
            }
        """
        if current_date is None:
            current_date = datetime.now()
        
        bottom_price = detection_record.get('bottom_price', 0)
        if bottom_price == 0:
            return {'was_correct': False, 'error': 'Invalid bottom price'}
        
        price_change_pct = ((current_price - bottom_price) / bottom_price) * 100
        
        # Bottom detection is CORRECT if:
        # 1. Price rose significantly from bottom (3%+ increase) OR
        # 2. Price is at least 1.5% above bottom (smaller threshold for confirmation)
        reversal_confirmed = price_change_pct >= 1.5
        
        # Bottom detection is FALSE POSITIVE if:
        # Price fell below the bottom price by 2%+ (price continued falling)
        false_positive = price_change_pct <= -2.0
        
        # Determine if detection was correct
        was_correct = reversal_confirmed and not false_positive
        
        result = {
            'was_correct': was_correct,
            'reversal_confirmed': reversal_confirmed,
            'false_positive': false_positive,
            'bottom_price': bottom_price,
            'current_price': current_price,
            'price_change_pct': price_change_pct,
            'verification_date': current_date.isoformat()
        }
        
        return result
    
    def record_verified_peak_detection(self, detection_record: Dict, verification_result: Dict):
        """Record a verified peak detection for learning"""
        if 'peak_bottom_accuracy' not in self.data['knowledge']:
            self.data['knowledge']['peak_bottom_accuracy'] = []
        
        was_correct = verification_result.get('was_correct', False)
        confidence = detection_record.get('confidence', 0)
        
        self.data['knowledge']['peak_bottom_accuracy'].append({
            'timestamp': verification_result.get('verification_date', datetime.now().isoformat()),
            'type': 'peak',
            'was_correct': was_correct,
            'peak_price': verification_result.get('peak_price', 0),
            'current_price': verification_result.get('current_price', 0),
            'price_change_pct': verification_result.get('price_change_pct', 0),
            'reversal_confirmed': verification_result.get('reversal_confirmed', False),
            'false_positive': verification_result.get('false_positive', False),
            'confidence': confidence,
            'symbol': detection_record.get('symbol', '')
        })
        
        if len(self.data['knowledge']['peak_bottom_accuracy']) > 1000:
            self.data['knowledge']['peak_bottom_accuracy'] = self.data['knowledge']['peak_bottom_accuracy'][-1000:]
        
        if was_correct:
            self.data['knowledge']['accuracy_improvements'].append({
                'timestamp': datetime.now().isoformat(),
                'type': 'correct_peak_detection',
                'symbol': detection_record.get('symbol', ''),
                'confidence': confidence
            })
        
        detection_record['verified'] = True
        detection_record['verification_result'] = verification_result
        
        self._add_learning_event('verified_peak_detection', {
            'symbol': detection_record.get('symbol', ''),
            'was_correct': was_correct,
            'peak_price': verification_result.get('peak_price', 0),
            'current_price': verification_result.get('current_price', 0),
            'confidence': confidence
        })
        self.save()
        
        status = 'CORRECT' if was_correct else 'FALSE POSITIVE'
        logger.info(f"Learning: Peak detection verified for {detection_record.get('symbol', '')} - {status}")
    
    def record_verified_bottom_detection(self, detection_record: Dict, verification_result: Dict):
        """Record a verified bottom detection for learning"""
        if 'peak_bottom_accuracy' not in self.data['knowledge']:
            self.data['knowledge']['peak_bottom_accuracy'] = []
        
        was_correct = verification_result.get('was_correct', False)
        confidence = detection_record.get('confidence', 0)
        
        self.data['knowledge']['peak_bottom_accuracy'].append({
            'timestamp': verification_result.get('verification_date', datetime.now().isoformat()),
            'type': 'bottom',
            'was_correct': was_correct,
            'bottom_price': verification_result.get('bottom_price', 0),
            'current_price': verification_result.get('current_price', 0),
            'price_change_pct': verification_result.get('price_change_pct', 0),
            'reversal_confirmed': verification_result.get('reversal_confirmed', False),
            'false_positive': verification_result.get('false_positive', False),
            'confidence': confidence,
            'symbol': detection_record.get('symbol', '')
        })
        
        if len(self.data['knowledge']['peak_bottom_accuracy']) > 1000:
            self.data['knowledge']['peak_bottom_accuracy'] = self.data['knowledge']['peak_bottom_accuracy'][-1000:]
        
        if was_correct:
            self.data['knowledge']['accuracy_improvements'].append({
                'timestamp': datetime.now().isoformat(),
                'type': 'correct_bottom_detection',
                'symbol': detection_record.get('symbol', ''),
                'confidence': confidence
            })
        
        detection_record['verified'] = True
        detection_record['verification_result'] = verification_result
        
        self._add_learning_event('verified_bottom_detection', {
            'symbol': detection_record.get('symbol', ''),
            'was_correct': was_correct,
            'bottom_price': verification_result.get('bottom_price', 0),
            'current_price': verification_result.get('current_price', 0),
            'confidence': confidence
        })
        self.save()
        
        status = 'CORRECT' if was_correct else 'FALSE POSITIVE'
        logger.info(f"Learning: Bottom detection verified for {detection_record.get('symbol', '')} - {status}")
    
    def get_pending_peak_verifications(self, symbol: str = None) -> List[Dict]:
        """Get pending peak detections that need verification"""
        if 'pending_peak_verifications' not in self.data:
            self.data['pending_peak_verifications'] = []
        
        pending = [d for d in self.data['pending_peak_verifications'] 
                   if not d.get('verified', False)]
        
        if symbol:
            pending = [d for d in pending if d.get('symbol', '').upper() == symbol.upper()]
        
        now = datetime.now()
        ready_for_verification = []
        for detection in pending:
            try:
                due_date_str = detection.get('verification_due_date')
                if due_date_str:
                    due_date = datetime.fromisoformat(due_date_str)
                    if now >= due_date:
                        ready_for_verification.append(detection)
            except:
                continue
        
        return ready_for_verification
    
    def get_pending_bottom_verifications(self, symbol: str = None) -> List[Dict]:
        """Get pending bottom detections that need verification"""
        if 'pending_bottom_verifications' not in self.data:
            self.data['pending_bottom_verifications'] = []
        
        pending = [d for d in self.data['pending_bottom_verifications'] 
                   if not d.get('verified', False)]
        
        if symbol:
            pending = [d for d in pending if d.get('symbol', '').upper() == symbol.upper()]
        
        now = datetime.now()
        ready_for_verification = []
        for detection in pending:
            try:
                due_date_str = detection.get('verification_due_date')
                if due_date_str:
                    due_date = datetime.fromisoformat(due_date_str)
                    if now >= due_date:
                        ready_for_verification.append(detection)
            except:
                continue
        
        return ready_for_verification
    
    def get_peak_bottom_accuracy_stats(self) -> Dict:
        """Get accuracy statistics for peak/bottom detections"""
        if 'peak_bottom_accuracy' not in self.data['knowledge']:
            return {
                'total_peaks_verified': 0,
                'correct_peaks': 0,
                'false_positive_peaks': 0,
                'peak_accuracy': 0.0,
                'total_bottoms_verified': 0,
                'correct_bottoms': 0,
                'false_positive_bottoms': 0,
                'bottom_accuracy': 0.0,
                'overall_accuracy': 0.0
            }
        
        peak_records = [r for r in self.data['knowledge']['peak_bottom_accuracy'] if r.get('type') == 'peak']
        bottom_records = [r for r in self.data['knowledge']['peak_bottom_accuracy'] if r.get('type') == 'bottom']
        
        total_peaks = len(peak_records)
        correct_peaks = sum(1 for r in peak_records if r.get('was_correct', False))
        false_positive_peaks = sum(1 for r in peak_records if r.get('false_positive', False))
        
        total_bottoms = len(bottom_records)
        correct_bottoms = sum(1 for r in bottom_records if r.get('was_correct', False))
        false_positive_bottoms = sum(1 for r in bottom_records if r.get('false_positive', False))
        
        peak_accuracy = (correct_peaks / total_peaks * 100) if total_peaks > 0 else 0.0
        bottom_accuracy = (correct_bottoms / total_bottoms * 100) if total_bottoms > 0 else 0.0
        
        total_verified = total_peaks + total_bottoms
        total_correct = correct_peaks + correct_bottoms
        overall_accuracy = (total_correct / total_verified * 100) if total_verified > 0 else 0.0
        
        return {
            'total_peaks_verified': total_peaks,
            'correct_peaks': correct_peaks,
            'false_positive_peaks': false_positive_peaks,
            'peak_accuracy': peak_accuracy,
            'total_bottoms_verified': total_bottoms,
            'correct_bottoms': correct_bottoms,
            'false_positive_bottoms': false_positive_bottoms,
            'bottom_accuracy': bottom_accuracy,
            'overall_accuracy': overall_accuracy
        }
    
    def record_buy_opportunity_prediction(self, symbol: str, current_price: float, predicted_price: float, 
                                          estimated_date: str, confidence: float, reasoning: str):
        """Record a predicted future buy opportunity for Megamind learning
        
        Args:
            symbol: Stock symbol
            current_price: Current price when prediction was made
            predicted_price: Predicted future better entry price
            estimated_date: Estimated date for the opportunity (ISO format)
            confidence: Confidence in the prediction (0-100)
            reasoning: Reasoning behind the prediction
        """
        if not self.data['data_sources'].get('first_learning_date'):
            self.data['data_sources']['first_learning_date'] = datetime.now().isoformat()
        
        # Ensure pending_buy_opportunities exists
        if 'pending_buy_opportunities' not in self.data:
            self.data['pending_buy_opportunities'] = []
            
        self.data['data_sources']['buy_opportunity_predictions'] = self.data['data_sources'].get('buy_opportunity_predictions', 0) + 1
        self._add_symbol(symbol)
        
        opportunity_id = len(self.data['pending_buy_opportunities']) + 1
        prediction_record = {
            'id': opportunity_id,
            'symbol': symbol.upper(),
            'current_price_at_prediction': current_price,
            'predicted_price': predicted_price,
            'estimated_date': estimated_date,
            'confidence': confidence,
            'reasoning': reasoning,
            'prediction_date': datetime.now().isoformat(),
            'verified': False
        }
        self.data['pending_buy_opportunities'].append(prediction_record)
        
        # Keep only last 500 pending
        if len(self.data['pending_buy_opportunities']) > 500:
            self.data['pending_buy_opportunities'] = self.data['pending_buy_opportunities'][-500:]
            
        self._add_learning_event('buy_opportunity_prediction', {
            'symbol': symbol,
            'predicted_price': predicted_price,
            'estimated_date': estimated_date,
            'confidence': confidence
        })
        self.save()
        logger.info(f"🧠 Learning: Buy opportunity for {symbol} recorded (${predicted_price:.2f} by {estimated_date})")

    def verify_buy_opportunity(self, prediction_record: Dict, current_price: float) -> Optional[Dict]:
        """Verify if a predicted buy opportunity was correct (did price reach the target?)"""
        target_price = prediction_record.get('predicted_price', 0)
        if target_price == 0:
            return None
            
        # Target is reached if current price <= target_price (since it's a better buying opportunity)
        # Note: We consider it "correct" if it hit the target price at ANY point, 
        # but here we check against CURRENT price when verification runs.
        hit_target = current_price <= target_price
        
        # We also check if the date has passed
        try:
            est_date = datetime.fromisoformat(prediction_record.get('estimated_date', ''))
            now = datetime.now()
            date_passed = now >= est_date
        except:
            date_passed = False
            
        # If it hit the target, it's correct regardless of date (early hit is good)
        if hit_target:
            return {
                'was_correct': True,
                'target_hit': True,
                'target_price': target_price,
                'actual_price': current_price,
                'verification_date': datetime.now().isoformat()
            }
        
        # If date passed and target NOT hit, it's incorrect
        if date_passed:
            return {
                'was_correct': False,
                'target_hit': False,
                'target_price': target_price,
                'actual_price': current_price,
                'verification_date': datetime.now().isoformat()
            }
            
        # Otherwise wait
        return None

    def record_verified_buy_opportunity(self, prediction_record: Dict, result: Dict):
        """Record verified buy opportunity for learning"""
        if 'buy_opportunity_accuracy' not in self.data['knowledge']:
            self.data['knowledge']['buy_opportunity_accuracy'] = []
            
        was_correct = result.get('was_correct', False)
        
        self.data['knowledge']['buy_opportunity_accuracy'].append({
            'timestamp': result.get('verification_date', datetime.now().isoformat()),
            'symbol': prediction_record.get('symbol', ''),
            'was_correct': was_correct,
            'predicted_price': prediction_record.get('predicted_price', 0),
            'actual_price': result.get('actual_price', 0),
            'confidence': prediction_record.get('confidence', 0)
        })
        
        prediction_record['verified'] = True
        prediction_record['verification_result'] = result
        
        self._add_learning_event('verified_buy_opportunity', {
            'symbol': prediction_record.get('symbol', ''),
            'was_correct': was_correct
        })
        self.save()
        logger.info(f"✅ Learning: Buy opportunity verified for {prediction_record.get('symbol', '')} - {'CORRECT' if was_correct else 'INCORRECT'}")

