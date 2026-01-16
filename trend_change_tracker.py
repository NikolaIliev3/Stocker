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
        """Check active trend predictions for verification"""
        verified_items = []
        active_preds = self.get_active_predictions()
        
        for pred in active_preds:
            try:
                symbol = pred.get('symbol')
                if not symbol:
                    continue
                    
                # Get current data
                current_data = data_fetcher.fetch_stock_data(symbol)
                current_price = current_data.get('price', 0)
                
                if current_price <= 0:
                    continue
                    
                # Check 1: Has estimated date passed?
                estimated_date_str = pred.get('estimated_date', '')
                try:
                    estimated_date = datetime.fromisoformat(estimated_date_str)
                    has_time_passed = datetime.now() > estimated_date
                except:
                    has_time_passed = False
                    
                # Check 2: Has significant movement happened in predicted direction?
                # We need historical context or price at prediction time, but trend change only records "current trend"
                # So we largely rely on time-based verification OR major reversals
                
                should_verify = has_time_passed
                
                if should_verify:
                    # Determine what happened
                    predicted_change = pred.get('predicted_change', '')
                    
                    # Logic to determine if prediction was correct
                    # This is simplified - ideally we'd look at price history since prediction
                    
                    # For now, we will verify based on what the trend LOOKS like now compared to prediction
                    # However, determining "trend" instantly is hard. 
                    # We will use price movement as a proxy if we can't do full analysis
                    
                    # If we don't have entry price, we can't know for sure.
                    # BUT, we can mark it as "expired - waiting for manual verification" 
                    # OR we can try to deduce it.
                    
                    # Since we can't easily auto-verify purely on price without start price,
                    # we will mostly rely on MANUAL verification for now, unless we update the tracker
                    # to store start price.
                    
                    # Let's check if we CAN auto-verify. 
                    # If we can't auto-verify safely, we leave it active for manual verification.
                    pass 

            except Exception as e:
                logger.error(f"Error checking verification for prediction {pred.get('id')}: {e}")
                
        return verified_items

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

