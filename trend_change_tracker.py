"""
Trend Change Predictions Tracker
Stores and verifies trend change predictions for learning system
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class TrendChangeTracker:
    """Tracks trend change predictions and verifies their accuracy"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_file = self.data_dir / "trend_change_predictions.json"
        self.predictions = []
        self.load()
    
    def load(self):
        """Load trend change predictions from file"""
        if self.predictions_file.exists():
            try:
                with open(self.predictions_file, 'r') as f:
                    data = json.load(f)
                    self.predictions = data.get('predictions', [])
            except Exception as e:
                logger.error(f"Error loading trend change predictions: {e}")
                self.predictions = []
        else:
            self.predictions = []
    
    def save(self):
        """Save trend change predictions to file"""
        try:
            data = {
                'predictions': self.predictions,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.predictions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving trend change predictions: {e}")
    
    def add_prediction(self, symbol: str, current_trend: str, predicted_change: str,
                      estimated_days: int, estimated_date: str, confidence: float,
                      reasoning: str, key_indicators: dict) -> Dict:
        """Add a trend change prediction"""
        prediction = {
            "id": len(self.predictions) + 1,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol.upper(),
            "current_trend": current_trend,
            "predicted_change": predicted_change,
            "estimated_days": estimated_days,
            "estimated_date": estimated_date,
            "confidence": confidence,
            "reasoning": reasoning,
            "key_indicators": key_indicators,
            "status": "active",  # active, verified, expired
            "verified": False,
            "was_correct": None,
            "actual_change": None,
            "verification_date": None
        }
        
        self.predictions.append(prediction)
        self.save()
        return prediction
    
    def get_active_predictions(self) -> List[Dict]:
        """Get all active trend change predictions"""
        return [p for p in self.predictions if p.get('status') == 'active']
    
    def get_prediction_by_id(self, prediction_id: int) -> Optional[Dict]:
        """Get a prediction by ID"""
        for pred in self.predictions:
            if pred.get('id') == prediction_id:
                return pred
        return None
    
    def verify_prediction(self, prediction_id: int, actual_change: str, 
                        was_correct: bool) -> Optional[Dict]:
        """Verify a trend change prediction"""
        prediction = self.get_prediction_by_id(prediction_id)
        if not prediction or prediction.get('status') != 'active':
            return None
        
        prediction['status'] = 'verified'
        prediction['verified'] = True
        prediction['was_correct'] = was_correct
        prediction['actual_change'] = actual_change
        prediction['verification_date'] = datetime.now().isoformat()
        
        self.save()
        return prediction
    
    def delete_prediction(self, prediction_id: int) -> bool:
        """Delete a trend change prediction"""
        prediction = self.get_prediction_by_id(prediction_id)
        if prediction:
            self.predictions.remove(prediction)
            self.save()
            return True
        return False
    
    def get_statistics(self) -> Dict:
        """Get statistics about trend change predictions"""
        total = len(self.predictions)
        active = len([p for p in self.predictions if p.get('status') == 'active'])
        verified = len([p for p in self.predictions if p.get('verified', False)])
        correct = len([p for p in self.predictions if p.get('was_correct') is True])
        
        accuracy = (correct / verified * 100) if verified > 0 else 0
        
        return {
            'total_predictions': total,
            'active': active,
            'verified': verified,
            'correct': correct,
            'incorrect': verified - correct,
            'accuracy': accuracy
        }

