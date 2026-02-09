"""
Prediction Service
Decouples prediction verification logic from the main GUI (God Object Refactoring).
Follows Principle 5: Single Responsibility Principle.
"""
import threading
import logging
from typing import Callable, Optional, Dict, List

logger = logging.getLogger(__name__)

class PredictionService:
    """
    Manages background verification of predictions.
    Decoupled from Tkinter - uses callbacks for UI updates.
    """
    
    def __init__(self, predictions_tracker, data_fetcher, learning_tracker):
        self.predictions_tracker = predictions_tracker
        self.data_fetcher = data_fetcher
        self.learning_tracker = learning_tracker
        self.running = False
        
        # Callbacks (to be set by the UI)
        self.on_verification_complete: Optional[Callable[[List[Dict]], None]] = None
        self.on_auto_train_trigger: Optional[Callable[[], None]] = None
        self.on_status_update: Optional[Callable[[], None]] = None

    def start_periodic_verification(self, interval_ms: int = 3600000, scheduler_func: Optional[Callable] = None):
        """
        Start the periodic verification loop.
        
        Args:
            interval_ms: Interval in milliseconds (default 1 hour)
            scheduler_func: Function to schedule next run (e.g., root.after). 
                            If None, user must manually handle scheduling or simple recursion.
                            Using `scheduler_func` allows integration with Tkinter's event loop without coupling.
        """
        if scheduler_func:
            # Tkinter style scheduling
            self._schedule_next(interval_ms, scheduler_func)
        else:
            # Independent thread loop (if needed in future)
            pass

    def _schedule_next(self, interval_ms, scheduler_func):
        """Internal scheduler wrapper"""
        self.run_verification_task()
        scheduler_func(interval_ms, lambda: self._schedule_next(interval_ms, scheduler_func))

    def run_verification_task(self):
        """Run verification in a background thread"""
        thread = threading.Thread(target=self._verify_in_background)
        thread.daemon = True
        thread.start()

    def _verify_in_background(self):
        """Core logic extracted from main.py"""
        try:
            logger.info("Service: Running periodic verification...")
            result = self.predictions_tracker.verify_all_active_predictions(self.data_fetcher)
            
            if result['verified'] > 0:
                newly_verified = result.get('newly_verified', [])
                
                # Record learning events
                self._record_learning(newly_verified)
                
                # Trigger Callbacks (Must typically be scheduled back to main thread by the caller,
                # but here we just invoke them. The CALLER is responsible for thread safety via .after if needed,
                # OR we assume these callbacks are thread-safe wrappers).
                # NOTE: For Tkinter, these raw calls from a thread are unsafe if they touch GUI directly.
                # Ideally, we pass a `dispatcher` function to __init__.
                
                if self.on_verification_complete:
                    self.on_verification_complete(newly_verified)
                
                if self.on_status_update:
                    self.on_status_update()
                    
                if self.on_auto_train_trigger:
                    self.on_auto_train_trigger()
                    
        except Exception as e:
            logger.error(f"Error in PredictionService: {e}")

    def _record_learning(self, verified_preds: List[Dict]):
        """Isolate learning recording logic"""
        for pred in verified_preds:
            if pred.get('was_correct') is not None:
                outcome = {
                    'was_correct': pred.get('was_correct', False),
                    'actual_price_change': pred.get('actual_price_change', 0.0),
                    'target_hit': pred.get('was_correct', False) and pred.get('action') in ['BUY', 'SELL'],
                }
                self.learning_tracker.record_verified_prediction(
                    pred.get('was_correct', False),
                    pred.get('strategy', 'trading'),
                    prediction_data=pred,
                    actual_outcome=outcome
                )
