"""
Predictions tracking module for Stocker App
Tracks predictions and verifies their accuracy
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class PredictionsTracker:
    """Tracks stock predictions and verifies their accuracy"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_file = self.data_dir / "predictions.json"
        self.predictions = []
        self.load()
    
    def load(self):
        """Load predictions from file"""
        if self.predictions_file.exists():
            try:
                with open(self.predictions_file, 'r') as f:
                    data = json.load(f)
                    self.predictions = data.get('predictions', [])
            except Exception as e:
                logger.error(f"Error loading predictions: {e}")
                self.predictions = []
        else:
            self.predictions = []
    
    def save(self):
        """Save predictions to file"""
        try:
            data = {
                'predictions': self.predictions,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.predictions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
    
    def add_prediction(self, symbol: str, strategy: str, action: str, 
                      entry_price: float, target_price: float, 
                      stop_loss: float, confidence: float, reasoning: str) -> Dict:
        """Add a new prediction"""
        prediction = {
            "id": len(self.predictions) + 1,
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol.upper(),
            "strategy": strategy,
            "action": action,  # BUY, SELL, HOLD, AVOID
            "entry_price": entry_price,
            "target_price": target_price,
            "stop_loss": stop_loss,
            "confidence": confidence,
            "reasoning": reasoning,
            "status": "active",  # active, verified, expired
            "verified": False,
            "was_correct": None,
            "actual_price_at_target": None,
            "verification_date": None,
            "days_to_target": None
        }
        
        self.predictions.append(prediction)
        self.save()
        return prediction
    
    def verify_prediction(self, prediction_id: int, current_price: float) -> Optional[Dict]:
        """Verify if a prediction came true"""
        prediction = self.get_prediction_by_id(prediction_id)
        if not prediction or prediction['status'] != 'active':
            return None
        
        action = prediction['action']
        entry_price = prediction['entry_price']
        target_price = prediction['target_price']
        stop_loss = prediction['stop_loss']
        
        was_correct = None
        
        if action == "BUY":
            # BUY prediction is correct if price reached target before stop loss
            if current_price >= target_price:
                was_correct = True
            elif current_price <= stop_loss:
                was_correct = False
        elif action == "SELL":
            # SELL prediction is correct if price reached target (went down) before stop loss (went up)
            if current_price <= target_price:
                was_correct = True
            elif current_price >= stop_loss:
                was_correct = False
        elif action in ["HOLD", "AVOID"]:
            # For HOLD/AVOID, we check if price moved in expected direction
            price_change = ((current_price - entry_price) / entry_price) * 100
            if action == "HOLD" and abs(price_change) < 5:  # Price stayed relatively stable
                was_correct = True
            elif action == "AVOID" and price_change < -5:  # Price went down (good to avoid)
                was_correct = True
            else:
                was_correct = False
        
        # Update prediction
        prediction['verified'] = True
        prediction['status'] = 'verified'
        prediction['was_correct'] = was_correct
        prediction['actual_price_at_target'] = current_price
        prediction['verification_date'] = datetime.now().isoformat()
        
        # Calculate days since prediction
        pred_date = datetime.fromisoformat(prediction['timestamp'])
        days_diff = (datetime.now() - pred_date).days
        prediction['days_to_target'] = days_diff
        
        self.save()
        return prediction
    
    def verify_all_active_predictions(self, data_fetcher) -> Dict:
        """Verify all active predictions by fetching current prices"""
        verified_count = 0
        correct_count = 0
        
        for prediction in self.predictions:
            if prediction['status'] == 'active':
                try:
                    # Fetch current price
                    stock_data = data_fetcher.fetch_stock_data(prediction['symbol'])
                    current_price = stock_data.get('price', 0)
                    
                    if current_price > 0:
                        result = self.verify_prediction(prediction['id'], current_price)
                        if result:
                            verified_count += 1
                            if result['was_correct']:
                                correct_count += 1
                except Exception as e:
                    logger.warning(f"Could not verify prediction {prediction['id']} for {prediction['symbol']}: {e}")
                    continue
        
        return {
            "verified": verified_count,
            "correct": correct_count,
            "accuracy": (correct_count / verified_count * 100) if verified_count > 0 else 0
        }
    
    def get_prediction_by_id(self, prediction_id: int) -> Optional[Dict]:
        """Get a prediction by ID"""
        for pred in self.predictions:
            if pred['id'] == prediction_id:
                return pred
        return None
    
    def get_active_predictions(self) -> List[Dict]:
        """Get all active predictions"""
        return [p for p in self.predictions if p['status'] == 'active']
    
    def get_verified_predictions(self) -> List[Dict]:
        """Get all verified predictions"""
        return [p for p in self.predictions if p['status'] == 'verified']
    
    def get_statistics(self) -> Dict:
        """Get prediction statistics"""
        total = len(self.predictions)
        active = len(self.get_active_predictions())
        verified = len(self.get_verified_predictions())
        
        verified_preds = self.get_verified_predictions()
        correct = sum(1 for p in verified_preds if p.get('was_correct') == True)
        incorrect = sum(1 for p in verified_preds if p.get('was_correct') == False)
        
        accuracy = (correct / verified * 100) if verified > 0 else 0
        
        return {
            "total_predictions": total,
            "active": active,
            "verified": verified,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": accuracy
        }

