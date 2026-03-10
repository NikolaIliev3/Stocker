"""
Trend Change Predictions Tracker
Stores and verifies trend change predictions for learning system.
Now includes entry price tracking and deduplication.
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
    
    def has_active_prediction(self, symbol: str, direction: str) -> bool:
        """Check if an active prediction already exists for this symbol + direction."""
        for p in self.predictions:
            if (p.get('status') == 'active' and 
                p.get('symbol', '').upper() == symbol.upper() and
                p.get('predicted_change') == direction):
                return True
        return False
    
    def add_prediction(self, symbol: str, current_trend: str, predicted_change: str,
                      estimated_days: int, estimated_date: str, confidence: float,
                      reasoning: str, key_indicators: dict,
                      entry_price: float = None) -> Optional[Dict]:
        """Add a trend change prediction with deduplication and entry price tracking.
        
        Returns None if a duplicate active prediction exists for this symbol + direction.
        """
        symbol = symbol.upper()
        
        # Deduplication: reject if same symbol + direction already active
        if self.has_active_prediction(symbol, predicted_change):
            logger.info(f"Skipping duplicate trend prediction for {symbol} ({predicted_change}) — already active")
            return None
        
        prediction = {
            "id": len(self.predictions) + 1,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "current_trend": current_trend,
            "predicted_change": predicted_change,
            "estimated_days": estimated_days,
            "estimated_date": estimated_date,
            "confidence": confidence,
            "reasoning": reasoning,
            "key_indicators": key_indicators,
            "entry_price": entry_price,  # Price at prediction time
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
                        was_correct: bool, actual_low_price: float = None,
                        actual_high_price: float = None, current_price: float = None) -> Optional[Dict]:
        """Verify a trend change prediction with price accuracy tracking"""
        prediction = self.get_prediction_by_id(prediction_id)
        if not prediction or prediction.get('status') != 'active':
            return None
        
        prediction['status'] = 'verified'
        prediction['verified'] = True
        prediction['was_correct'] = was_correct
        prediction['actual_change'] = actual_change
        prediction['verification_date'] = datetime.now().isoformat()
        
        # Store price accuracy data for learning
        if actual_low_price is not None:
            prediction['actual_low_price'] = actual_low_price
        if actual_high_price is not None:
            prediction['actual_high_price'] = actual_high_price
        if current_price is not None:
            prediction['current_price_at_verification'] = current_price
        
        # Calculate price change from entry if available
        entry_price = prediction.get('entry_price')
        if entry_price and current_price and entry_price > 0:
            prediction['price_change_pct'] = ((current_price - entry_price) / entry_price) * 100
        
        # Calculate days accuracy
        try:
            estimated_date = datetime.fromisoformat(prediction.get('estimated_date', ''))
            verification_date = datetime.now()
            actual_days = (verification_date - estimated_date).days
            estimated_days = prediction.get('estimated_days', 0)
            prediction['days_accuracy'] = {
                'estimated': estimated_days,
                'actual': actual_days,
                'difference': actual_days - estimated_days
            }
        except:
            pass
        
        self.save()
        return prediction
    
    def check_for_verification(self, data_fetcher) -> List[Dict]:
        """Check active trend predictions for verification.
        
        Note: Main verification logic is in main.py (_start_trend_change_verification).
        This method handles simple time-based expiry.
        """
        verified_items = []
        active_preds = self.get_active_predictions()
        
        for pred in active_preds:
            try:
                # Check if prediction is very old (> 6 months) — expire it
                try:
                    timestamp = datetime.fromisoformat(pred.get('timestamp', ''))
                    if (datetime.now() - timestamp).days > 180:
                        pred['status'] = 'expired'
                        pred['verification_date'] = datetime.now().isoformat()
                        verified_items.append(pred)
                        logger.info(f"Expired old trend prediction #{pred.get('id')} for {pred.get('symbol')}")
                except:
                    pass
                    
            except Exception as e:
                logger.error(f"Error checking verification for prediction {pred.get('id')}: {e}")
        
        if verified_items:
            self.save()
                
        return verified_items

    def delete_prediction(self, prediction_id: int) -> bool:
        """Delete a trend change prediction"""
        prediction = self.get_prediction_by_id(prediction_id)
        if prediction:
            self.predictions.remove(prediction)
            self.save()
            return True
        return False
    
    def clear_active_predictions(self) -> int:
        """Clear all active predictions (used after major logic changes).
        Returns count of cleared predictions."""
        count = 0
        for pred in self.predictions:
            if pred.get('status') == 'active':
                pred['status'] = 'expired'
                pred['verification_date'] = datetime.now().isoformat()
                pred['was_correct'] = None  # Unknown — expired due to logic change
                pred['actual_change'] = 'expired_logic_change'
                count += 1
        if count > 0:
            self.save()
            logger.info(f"Cleared {count} active trend change predictions due to logic change")
        return count
    
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
