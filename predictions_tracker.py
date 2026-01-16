"""
Predictions tracking module for Stocker App
Tracks predictions and verifies their accuracy
"""
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Import production monitor if available
try:
    from production_monitor import ProductionMonitor
    HAS_MONITORING = True
except ImportError:
    HAS_MONITORING = False

# Import performance attribution if available
try:
    from performance_attribution import PerformanceAttribution
    HAS_ATTRIBUTION = True
except ImportError:
    HAS_ATTRIBUTION = False

logger = logging.getLogger(__name__)


class PredictionsTracker:
    """Tracks stock predictions and verifies their accuracy"""
    
    
    def __init__(self, data_dir: Path, price_move_predictor=None):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_file = self.data_dir / "predictions.json"
        self.predictions = []
        self.price_move_predictor = price_move_predictor
        
        # Initialize production monitors for each strategy
        self.production_monitors = {}
        if HAS_MONITORING:
            for strategy in ['trading', 'mixed', 'investing']:
                try:
                    self.production_monitors[strategy] = ProductionMonitor(data_dir, strategy)
                except Exception as e:
                    logger.warning(f"Could not initialize production monitor for {strategy}: {e}")
        
        # Initialize performance attribution
        if HAS_ATTRIBUTION:
            try:
                self.performance_attribution = PerformanceAttribution(data_dir)
            except Exception as e:
                logger.warning(f"Could not initialize performance attribution: {e}")
                self.performance_attribution = None
        else:
            self.performance_attribution = None
        
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
                      estimated_days: int = None, predicted_move_pct: float = None) -> Dict:
        """Add a new prediction with estimated target date
        
        Args:
            estimated_days: Estimated number of days until target should be reached.
                          If None, calculated based on strategy.
            predicted_move_pct: Predicated magnitude of the move in percentage (absolute)
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
        
        # Store predicted move percentage if provided (for Megamind learning)
        if predicted_move_pct is not None:
            prediction['predicted_move_pct'] = float(predicted_move_pct)
        
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
            # SELL prediction (standard logic: bearish, expecting price to go DOWN)
            # Correct if price reached target (went DOWN) before stop loss (went UP)
            if current_price <= target_price:
                was_correct = True
            elif current_price >= stop_loss:
                was_correct = False
        elif action in ["HOLD", "AVOID"]:
            # For HOLD/AVOID, we check if price moved in expected direction
            price_change = ((current_price - entry_price) / entry_price) * 100
            strategy = prediction.get('strategy', 'trading')
            
            # Check for time-based expiry for HOLD/AVOID since they don't have explicit targets
            estimated_date_str = prediction.get('estimated_target_date')
            is_expired = False
            if estimated_date_str:
                try:
                    is_expired = datetime.now() >= datetime.fromisoformat(estimated_date_str)
                except:
                    pass
            
            if is_expired:
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
            else:
                # Not expired yet, keep active
                return None
        
        # KEY CHANGE: If was_correct is still None (neither Target nor Stop hit), 
        # do NOT verify. Keep it active until it hits a level.
        if was_correct is None:
            return None
            
        # Calculate price change
        price_change = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
        
        # Update prediction
        prediction['verified'] = True
        prediction['status'] = 'verified'
        prediction['was_correct'] = was_correct
        prediction['actual_price_at_target'] = current_price
        prediction['verification_date'] = datetime.now().isoformat()
        prediction['actual_price_change'] = price_change
        
        # Calculate days since prediction
        pred_date = datetime.fromisoformat(prediction['timestamp'])
        days_diff = (datetime.now() - pred_date).days
        prediction['days_to_target'] = days_diff

        # -------------------------------------------------------------------------
        # MEGAMIND LEARNING INJECTION
        # -------------------------------------------------------------------------
        # If we have a price_move_predictor and this prediction has a predicted move,
        # teach the system!
        if getattr(self, 'price_move_predictor', None) and 'predicted_move_pct' in prediction:
            try:
                predicted_pct = prediction['predicted_move_pct']
                actual_abs_pct = abs(price_change)
                # Learn from this specific outcome
                self.price_move_predictor.learn_from_outcome(
                    predicted_pct=predicted_pct,
                    actual_pct=actual_abs_pct,
                    signal_type=action
                )
                logger.info(f"🧠 Fed prediction result to PriceMovePredictor: Pred {predicted_pct}%, Actual {actual_abs_pct}%")
            except Exception as e:
                logger.error(f"Failed to feed data to PriceMovePredictor: {e}")
        # -------------------------------------------------------------------------
        
        # Record outcome in production monitor
        strategy = prediction.get('strategy', 'trading')
        if strategy in self.production_monitors:
            try:
                self.production_monitors[strategy].record_outcome(
                    symbol=prediction['symbol'],
                    action=action,
                    was_correct=was_correct,
                    actual_price_change=price_change
                )
            except Exception as e:
                logger.debug(f"Error recording outcome in production monitor: {e}")
        
        # Record outcome in performance attribution
        if self.performance_attribution:
            try:
                outcome_dict = {
                    'was_correct': was_correct,
                    'actual_price_change': price_change
                }
                # Add market regime and sector if available
                prediction_with_metadata = prediction.copy()
                self.performance_attribution.record_prediction_outcome(
                    prediction=prediction_with_metadata,
                    outcome=outcome_dict
                )
            except Exception as e:
                logger.debug(f"Error recording outcome in performance attribution: {e}")
        
        self.save()
        return prediction
    
    def verify_all_active_predictions(self, data_fetcher, check_all: bool = False) -> Dict:
        """Verify all active predictions by fetching current prices
        
        Args:
            data_fetcher: Data fetcher instance
            check_all: Legacy parameter, no longer restricts verification. 
                      Now consistently checks ALL active predictions for Price Targets/Stops.
        """
        verified_count = 0
        correct_count = 0
        newly_verified = []  # Track newly verified predictions
        now = datetime.now()
        
        for prediction in self.predictions:
            if prediction['status'] == 'active':
                # ALWAYS check price conditions for active predictions.
                # We want to know if Target or Stop was hit immediately.
                # We no longer wait for the 'date' to expire to check for success.
                # We also don't auto-fail if date expires but levels aren't hit (handled in verify_prediction).
                
                try:
                    # Fetch current price
                    stock_data = data_fetcher.fetch_stock_data(prediction['symbol'])
                    current_price = stock_data.get('price', 0)
                    
                    if current_price > 0:
                        result = self.verify_prediction(prediction['id'], current_price)
                        if result:
                            # If verify_prediction returns a result, it means it was verified (Target/Stop hit)
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
        """Get prediction statistics with enhanced tracking"""
        total = len(self.predictions)
        active = len(self.get_active_predictions())
        verified = len(self.get_verified_predictions())
        
        verified_preds = self.get_verified_predictions()
        correct = sum(1 for p in verified_preds if p.get('was_correct') == True)
        incorrect = sum(1 for p in verified_preds if p.get('was_correct') == False)
        
        accuracy = (correct / verified * 100) if verified > 0 else 0
        
        # Calculate accuracy by strategy
        strategy_stats = {}
        for strategy in ['trading', 'mixed', 'investing']:
            strategy_preds = [p for p in verified_preds if p.get('strategy') == strategy]
            if strategy_preds:
                strategy_correct = sum(1 for p in strategy_preds if p.get('was_correct') == True)
                strategy_accuracy = (strategy_correct / len(strategy_preds) * 100) if strategy_preds else 0
                strategy_stats[strategy] = {
                    'total': len(strategy_preds),
                    'correct': strategy_correct,
                    'accuracy': strategy_accuracy
                }
        
        # Calculate accuracy over time (last 30 days)
        from datetime import datetime, timedelta
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_preds = [
            p for p in verified_preds 
            if datetime.fromisoformat(p.get('verification_date', '2000-01-01')) >= thirty_days_ago
        ]
        recent_accuracy = 0
        if recent_preds:
            recent_correct = sum(1 for p in recent_preds if p.get('was_correct') == True)
            recent_accuracy = (recent_correct / len(recent_preds) * 100) if recent_preds else 0
        
        # Calculate average days to target for correct predictions
        correct_preds = [p for p in verified_preds if p.get('was_correct') == True]
        avg_days_to_target = 0
        if correct_preds:
            days_list = [p.get('days_to_target', 0) for p in correct_preds if p.get('days_to_target') is not None]
            if days_list:
                avg_days_to_target = sum(days_list) / len(days_list)
        
        return {
            "total_predictions": total,
            "active": active,
            "verified": verified,
            "correct": correct,
            "incorrect": incorrect,
            "accuracy": accuracy,
            "recent_accuracy": recent_accuracy,  # Last 30 days
            "strategy_stats": strategy_stats,
            "avg_days_to_target": avg_days_to_target,
            "recent_verified_count": len(recent_preds)
        }

    def get_all_tracked_symbols(self) -> List[str]:
        """Get list of unique symbols from all predictions"""
        symbols = set()
        for p in self.predictions:
            if 'symbol' in p:
                symbols.add(p['symbol'])
        return list(symbols)

