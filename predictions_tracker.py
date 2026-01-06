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
                      stop_loss: float, confidence: float, reasoning: str,
                      estimated_days: int = None) -> Dict:
        """Add a new prediction with estimated target date
        
        Args:
            estimated_days: Estimated number of days until target should be reached.
                          If None, calculated based on strategy.
        """
        # Calculate estimated target date based on strategy if not provided
        if estimated_days is None:
            if strategy == "trading":
                estimated_days = 10  # Short-term: ~10 days
            elif strategy == "mixed":
                estimated_days = 21  # Medium-term: ~3 weeks
            else:  # investing
                estimated_days = 547  # Long-term: ~1.5 years (547 days)
        
        estimated_target_date = (datetime.now() + timedelta(days=estimated_days)).isoformat()
        
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
            "days_to_target": None,
            "estimated_target_date": estimated_target_date,
            "estimated_days": estimated_days
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
            strategy = prediction.get('strategy', 'trading')
            
            if action == "HOLD":
                # Strategy-specific thresholds for HOLD (same as backtest evaluation)
                if strategy == "trading":
                    was_correct = abs(price_change) < 3
                elif strategy == "mixed":
                    was_correct = abs(price_change) < 5
                else:  # investing
                    # For investing (1.5 year timeframe), allow up to 20% movement
                    was_correct = abs(price_change) < 20
            elif action == "AVOID":
                # AVOID (legacy) is correct if price went down
                was_correct = price_change < -5
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
    
    def verify_all_active_predictions(self, data_fetcher, check_all: bool = False) -> Dict:
        """Verify all active predictions by fetching current prices
        
        Args:
            data_fetcher: Data fetcher instance
            check_all: If True, verify all active predictions. If False, only verify
                      predictions that have reached their estimated target date.
        """
        verified_count = 0
        correct_count = 0
        newly_verified = []  # Track newly verified predictions
        now = datetime.now()
        
        for prediction in self.predictions:
            if prediction['status'] == 'active':
                # Check if target date has been reached (or check_all is True)
                should_verify = check_all
                
                if not should_verify:
                    # Check if estimated target date has been reached
                    estimated_date_str = prediction.get('estimated_target_date')
                    if estimated_date_str:
                        try:
                            estimated_date = datetime.fromisoformat(estimated_date_str)
                            should_verify = now >= estimated_date
                        except:
                            # If date parsing fails, verify anyway (backward compatibility)
                            should_verify = True
                    else:
                        # No estimated date - verify if prediction is old enough
                        # Default to verifying if older than strategy-appropriate timeframe
                        pred_date = datetime.fromisoformat(prediction['timestamp'])
                        days_old = (now - pred_date).days
                        strategy = prediction.get('strategy', 'trading')
                        
                        if strategy == "trading" and days_old >= 10:
                            should_verify = True
                        elif strategy == "mixed" and days_old >= 21:
                            should_verify = True
                        elif strategy == "investing" and days_old >= 365:
                            should_verify = True
                
                if should_verify:
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
                                # Add to newly verified list
                                newly_verified.append(result)
                    except Exception as e:
                        logger.warning(f"Could not verify prediction {prediction['id']} for {prediction['symbol']}: {e}")
                        continue
        
        return {
            "verified": verified_count,
            "correct": correct_count,
            "accuracy": (correct_count / verified_count * 100) if verified_count > 0 else 0,
            "newly_verified": newly_verified  # Include list of newly verified predictions
        }
    
    def get_prediction_by_id(self, prediction_id: int) -> Optional[Dict]:
        """Get a prediction by ID"""
        for pred in self.predictions:
            if pred['id'] == prediction_id:
                return pred
        return None
    
    def update_prediction_action(self, prediction_id: int, new_action: str, 
                                new_entry_price: float, new_target_price: float,
                                new_stop_loss: float, new_confidence: float,
                                new_reasoning: str) -> bool:
        """Update a prediction's action and prices
        
        Args:
            prediction_id: ID of the prediction to update
            new_action: New action (BUY/SELL/HOLD)
            new_entry_price: New entry price
            new_target_price: New target price
            new_stop_loss: New stop loss
            new_confidence: New confidence level
            new_reasoning: Updated reasoning
            
        Returns:
            True if updated, False if not found
        """
        prediction = self.get_prediction_by_id(prediction_id)
        if not prediction:
            logger.warning(f"Prediction #{prediction_id} not found for update")
            return False
        
        # Update prediction fields
        prediction['action'] = new_action
        prediction['entry_price'] = new_entry_price
        prediction['target_price'] = new_target_price
        prediction['stop_loss'] = new_stop_loss
        prediction['confidence'] = new_confidence
        prediction['reasoning'] = new_reasoning
        prediction['timestamp'] = datetime.now().isoformat()
        
        # Recalculate estimated target date based on strategy
        strategy = prediction.get('strategy', 'trading')
        if strategy == "trading":
            estimated_days = 10
        elif strategy == "mixed":
            estimated_days = 21
        else:  # investing
            estimated_days = 547
        
        prediction['estimated_days'] = estimated_days
        prediction['estimated_target_date'] = (datetime.now() + timedelta(days=estimated_days)).isoformat()
        
        self.save()
        logger.info(f"Updated prediction #{prediction_id}: action changed to {new_action}")
        return True
    
    def get_active_predictions(self) -> List[Dict]:
        """Get all active predictions"""
        return [p for p in self.predictions if p['status'] == 'active']
    
    def get_verified_predictions(self) -> List[Dict]:
        """Get all verified predictions"""
        return [p for p in self.predictions if p['status'] == 'verified']
    
    def delete_prediction(self, prediction_id: int) -> bool:
        """Delete a prediction by ID
        
        Args:
            prediction_id: ID of the prediction to delete
            
        Returns:
            True if deleted, False if not found
        """
        for i, pred in enumerate(self.predictions):
            if pred['id'] == prediction_id:
                del self.predictions[i]
                # Reassign IDs to maintain sequential order
                for j, p in enumerate(self.predictions):
                    p['id'] = j + 1
                self.save()
                logger.info(f"Deleted prediction #{prediction_id}")
                return True
        logger.warning(f"Prediction #{prediction_id} not found for deletion")
        return False
    
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

