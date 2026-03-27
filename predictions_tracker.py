"""
Predictions tracking module for Stocker App
Tracks predictions and verifies their accuracy
"""
import json
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

try:
    from safe_storage import LockingJSONStorage
except ImportError:
    # Fallback/Circular import handler (should not happen if safe_storage is base)
    pass

# Import production monitor if available
# Note: Use broad Exception catch because production_monitor imports evidently
# which can fail with pydantic ConfigError on Python 3.14+ (not just ImportError)
try:
    from production_monitor import ProductionMonitor
    HAS_MONITORING = True
except Exception:
    HAS_MONITORING = False

# Import performance attribution if available
try:
    from performance_attribution import PerformanceAttribution
    HAS_ATTRIBUTION = True
except ImportError:
    HAS_ATTRIBUTION = False

from config import PREDICTIONS_FILE, STATE_FILE
from sector_mapper import SectorMapper

logger = logging.getLogger(__name__)


class PredictionsTracker:
    """Tracks stock predictions and verifies their accuracy"""
    
    def __init__(self, data_dir: Path, price_move_predictor=None):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_file = PREDICTIONS_FILE
        
        # Initialize Safe Storage
        self.storage = LockingJSONStorage(self.predictions_file)
        self.lock = threading.RLock() # Re-entrant lock for thread safety
        
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
        
        # Initialize Sector Mapper for alpha benchmarking
        self.sector_mapper = SectorMapper(self.data_dir)
        
        self.load()
    
    def load(self):
        """Load predictions from file safely"""
        with self.lock:
            data = self.storage.load(default={})
            # Handle both dict format {"predictions": [...]} and raw list format [...]
            if isinstance(data, list):
                self.predictions = data
            else:
                self.predictions = data.get('predictions', [])
    
    def save(self):
        """Save predictions to file safely"""
        with self.lock:
            data = {
                'predictions': self.predictions,
                'last_updated': datetime.now().isoformat()
            }
            # Use explicit file locking for writes
            self.storage.write_locked(data)
    
    def add_prediction(self, symbol: str, strategy: str, action: str, 
                      entry_price: float, target_price: float, 
                      stop_loss: float, confidence: float, reasoning: str,
                      estimated_days: int = None, predicted_move_pct: float = None,
                      market_regime: str = "unknown", atr_percent: float = 0.0) -> Optional[Dict]:
        """Add a new prediction with estimated target date
        
        Args:
            estimated_days: Estimated number of days until target should be reached.
                          If None, calculated based on strategy.
            predicted_move_pct: Predicated magnitude of the move in percentage (absolute)
            market_regime: Market regime at the time of prediction (bull, bear, etc.)
            
        Returns:
            Dict with the prediction data, or None if action is HOLD/AVOID (veto).
        """
        # HOLD/AVOID are vetoes — they are not actionable predictions.
        # Do not store, track, or verify them.
        if action in ("HOLD", "AVOID"):
            logger.info(f"⏭️ Skipping {action} for {symbol} — veto signal, not stored.")
            return None
            
        # --- QUANT MODE HARD ENFORCEMENT ---
        # Catch any predictions that bypassed the hybrid_predictor's filters
        # (e.g., from main.py's consensus upgrades)
        if action == "SELL":
             logger.info(f"🛑 ELITE VETO (Tracker): Suppressing {symbol} SELL ({confidence}%) - Long-Only Strategy. Not saved.")
             return None
             
        if action == "BUY" and confidence < 80.0:
             logger.info(f"🛡️ PROTECTION (Tracker): Suppressing {symbol} BUY ({confidence}% < 80.0%). Not saved.")
             return None
        # Calculate estimated target date based on strategy if not provided
        if estimated_days is None:
            if strategy == "trading":
                estimated_days = 10  # Short-term: ~10 days
            elif strategy == "mixed":
                estimated_days = 21  # Medium-term: ~3 weeks
            else:  # investing
                estimated_days = 547  # Long-term: ~1.5 years (547 days)
        
        try:
            from utils.market_utils import add_trading_days
            estimated_target_date = add_trading_days(datetime.now(), estimated_days).isoformat()
        except ImportError:
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
            "market_regime": market_regime,
            "atr_percent": atr_percent,
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
        
        with self.lock:
            self.predictions.append(prediction)
            self.save()
            
            # Sync to paper_state.json for dashboard visibility
            self._sync_to_paper_state(prediction)
            
        return prediction

    def _sync_to_paper_state(self, prediction: Dict):
        """Sync a single prediction to paper_state.json for the dashboard"""
        try:
            if not STATE_FILE.exists():
                return

            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            symbol = prediction['symbol']
            
            # Determine reasoning format
            raw_reasoning = prediction.get('reasoning', [])
            
            # Validation: Ensure reasoning is never empty
            if not raw_reasoning or (isinstance(raw_reasoning, list) and not raw_reasoning):
                reasoning_list = ["No reasoning provided - check development logs"]
            elif isinstance(raw_reasoning, str):
                reasoning_list = [line for line in raw_reasoning.split('\n') if line.strip()]
                if not reasoning_list:
                    reasoning_list = ["No reasoning provided - check development logs"]
            elif isinstance(raw_reasoning, list):
                reasoning_list = [str(r) for r in raw_reasoning if str(r).strip()]
                if not reasoning_list:
                    reasoning_list = ["No reasoning provided - check development logs"]
            else:
                reasoning_list = [str(raw_reasoning)]

            # Convert internal prediction format to dashboard recommendation format
            # This allows manual scans from the GUI to show up in "Live Recommendations"
            recommendation = {
                "action": prediction['action'],
                "confidence": prediction['confidence'],
                "entry_price": prediction['entry_price'],
                "target_price": prediction['target_price'],
                "stop_loss": prediction['stop_loss'],
                "estimated_days": prediction['estimated_days'],
                "rr_ratio": 0.0, # Placeholder
                "reasoning": reasoning_list,
                "description": "\n".join(reasoning_list), # Alias for UI compatibility
                "context": "Manual Scan",
                "estimated_target_date": prediction['estimated_target_date']
            }
            
            if 'predictions' not in data:
                data['predictions'] = {}
                
            data['predictions'][symbol] = {
                "strategy": prediction.get('strategy', 'trading'),
                "recommendation": recommendation,
                "timestamp": prediction['timestamp']
            }
            
            with open(STATE_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
                
            logger.info(f"Synced {symbol} prediction to paper_state.json")
            
        except Exception as e:
            logger.error(f"Error syncing to paper_state: {e}")
    

    def verify_prediction(self, prediction_id: int, current_price: float, high: float = None, low: float = None, data_fetcher=None) -> Optional[Dict]:
        """Verify if a prediction came true
        
        Args:
            prediction_id: ID of the prediction to verify
            current_price: Current market price
            high: High price since prediction (optional, for retroactive check)
            low: Low price since prediction (optional, for retroactive check)
        """
        with self.lock:
            prediction = self.get_prediction_by_id(prediction_id)
            if not prediction or prediction['status'] != 'active':
                return None
                
            # GRACE PERIOD CHECK:
            # Don't verify predictions made in the last 15 minutes.
            try:
                pred_timestamp = datetime.fromisoformat(prediction['timestamp'])
                minutes_since_creation = (datetime.now() - pred_timestamp).total_seconds() / 60
                if minutes_since_creation < 15:
                    # Too new, skip verification
                    return None
            except Exception as e:
                logger.debug(f"Error checking grace period for prediction {prediction_id}: {e}")
            
            action = prediction['action']
            entry_price = prediction['entry_price']
            target_price = prediction['target_price']
            stop_loss = prediction['stop_loss']
            
            was_correct = None
            
            # Check for time-based expiry
            estimated_date_str = prediction.get('estimated_target_date')
            is_expired = False
            if estimated_date_str:
                try:
                    is_expired = datetime.now() >= datetime.fromisoformat(estimated_date_str)
                except:
                    pass

            if action == "BUY":
                # PATIENT SNIPER VERIFICATION:
                # 1. Retroactive High/Low Check (prioritized)
                #    If High >= Target -> WIN
                
                # Check High (Win)
                if high is not None and high >= target_price:
                    was_correct = True
                    current_price = high  # Record ACTUAL high, not target (for auditing)
                    logger.info(f"🔍 Retroactive HIT for {prediction.get('symbol')} #{prediction_id}: "
                                f"High={high:.2f} >= Target={target_price:.2f}, "
                                f"EntryDate={prediction['timestamp'][:10]}")
                # Check Current Price (Snapshot fallback)
                elif current_price >= target_price:
                    was_correct = True
                
                # NOTE: Stop Loss hits are IGNORED until Expiry.
                # We allow the trade to breathe even if it dips below stop.
                
                # 2. Expiry Check (if no levels hit)
                elif is_expired:
                    from config import MIN_PROFIT_TARGET_PCT
                    price_change = ((current_price - entry_price) / entry_price) * 100
                    
                    if price_change >= MIN_PROFIT_TARGET_PCT:
                        was_correct = True
                        logger.info(f"✅ Prediction {prediction_id} SUCCESS (Terminal Profit): {price_change:.2f}% >= {MIN_PROFIT_TARGET_PCT}%")
                    else:
                        was_correct = False 
                        logger.info(f"❌ Prediction {prediction_id} FAILURE (Expired without hitting target): {price_change:.2f}% < {MIN_PROFIT_TARGET_PCT}%")
                else:
                    # Still within lifespan, wait for level hit or expiry
                    return None
                        
            elif action == "SELL":
                # Retroactive High/Low Check
                # Check Low (Win for Short)
                if low is not None and low <= target_price:
                    was_correct = True
                    current_price = low  # Record ACTUAL low, not target (for auditing)
                    logger.info(f"🔍 Retroactive HIT for {prediction.get('symbol')} #{prediction_id}: "
                                f"Low={low:.2f} <= Target={target_price:.2f}, "
                                f"EntryDate={prediction['timestamp'][:10]}")
                # Snapshot fallback
                elif current_price <= target_price:
                    was_correct = True
                
                # NOTE: Stop Loss hits are IGNORED until Expiry.
                
                elif is_expired:
                    from config import MIN_PROFIT_TARGET_PCT
                    # For SELL, price_change needs to be negative (drop)
                    # Example: Entry 100, Current 90 -> -10%. We want <= -3.0%
                    price_change = ((current_price - entry_price) / entry_price) * 100
                    
                    if price_change <= -MIN_PROFIT_TARGET_PCT:
                         was_correct = True
                    else:
                         was_correct = False
            elif action in ["HOLD", "AVOID"]:
                # HOLD/AVOID are vetoes — they should never have been stored.
                # If one exists (legacy data), skip verification entirely.
                logger.warning(f"⚠️ Found legacy HOLD/AVOID prediction #{prediction_id} — skipping verification.")
                return None
            
            # KEY CHANGE: If was_correct is still None (neither Target nor Stop hit), 
            # do NOT verify. Keep it active until it hits a level.
            if was_correct is None:
                return None
                
            # Calculate price change
            price_change = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
            
            # --- QUANT MODE ALPHA VERIFICATION ---
            alpha = 0.0
            spy_change = 0.0
            
            try:
                from config import IS_QUANT_MODE
            except ImportError:
                IS_QUANT_MODE = False
                
            if IS_QUANT_MODE and data_fetcher and entry_price > 0 and was_correct is not None:
                try:
                    import pandas as pd
                    # Fetch benchmarking ETF (Sector-Relative Alpha)
                    benchmark_ticker = self.sector_mapper.get_sector_etf(prediction['symbol'])
                    benchmark_data = data_fetcher.fetch_stock_history(benchmark_ticker, period="3mo")
                    
                    if benchmark_data and benchmark_data.get('data'):
                        bench_df = pd.DataFrame(benchmark_data['data'])
                        if 'date' in bench_df.columns:
                            bench_df['date'] = pd.to_datetime(bench_df['date'])
                            bench_df.set_index('date', inplace=True)
                            
                            pred_timestamp_naive = datetime.fromisoformat(prediction['timestamp']).replace(tzinfo=None)
                            
                            # Find start and end prices
                            bench_start_idx = bench_df.index.get_indexer([pred_timestamp_naive], method='nearest')[0]
                            # Using last row for current benchmark level
                            bench_end_price = bench_df.iloc[-1]['close'] if 'close' in bench_df.columns else bench_df.iloc[-1]['Close']
                            bench_start_price = bench_df.iloc[bench_start_idx]['close'] if 'close' in bench_df.columns else bench_df.iloc[bench_start_idx]['Close']
                            
                            if bench_start_price > 0:
                                bench_change = ((bench_end_price - bench_start_price) / bench_start_price) * 100
                                alpha = price_change - bench_change
                                prediction['spy_change'] = bench_change # Keep key name for UI compatibility, but it's now bench
                                prediction['alpha'] = alpha
                                prediction['benchmark_etf'] = benchmark_ticker
                                
                                # --- NEW LOGIC: ALPHA > 0 HURDLE ---
                                # Simply require positive alpha for a win in Quant Mode.
                                if action == "BUY":
                                    if alpha <= 0.0:
                                        if was_correct:
                                            logger.info(f"📉 ALPHA VETO: {prediction['symbol']} had absolute target hit, but Alpha ({alpha:.2f}%) <= 0. Marking FALSE.")
                                        was_correct = False
                                    else:
                                        # was_correct is already True from target hit or 3% expiry
                                        pass
                                elif action == "SELL":
                                    # For shorts, alpha < 0 means outperformance (dropped more than sector)
                                    if alpha >= 0.0:
                                        if was_correct:
                                            logger.info(f"📈 ALPHA VETO: {prediction['symbol']} had absolute target hit, but Alpha ({alpha:.2f}%) >= 0. Marking FALSE.")
                                        was_correct = False
                                    else:
                                        was_correct = True
                except Exception as e:
                    logger.warning(f"Failed to calculate Alpha for {prediction['symbol']}: {e}")

            
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
        
        # Create a copy of active predictions to iterate safely without holding lock during network requests
        with self.lock:
            active_predictions = [p.copy() for p in self.predictions if p['status'] == 'active']
        
        for prediction in active_predictions:
                # ALWAYS check price conditions for active predictions.
                # We want to know if Target or Stop was hit immediately.
                
                try:
                    # Fetch current price AND history for retroactive check
                    # We need history to catch hits that happened between scans or while app was off
                    stock_data = data_fetcher.fetch_stock_data(prediction['symbol'], check_quality=False)
                    current_price = stock_data.get('price', 0)
                    
                    # EXTRACT HISTORY FOR RETROACTIVE CHECK
                    high_since_entry = None
                    low_since_entry = None
                    
                    if 'history' in stock_data and stock_data['history']:
                        try:
                            # 1. Parse prediction timestamp
                            pred_ts = datetime.fromisoformat(prediction['timestamp'])
                            
                            # 2. Filter history for candles AFTER prediction time
                            # We look for any candle that opened/closed/high/low AFTER the prediction
                            relevant_candles = []
                            for candle in stock_data['history']:
                                candle_date_str = candle.get('date') # YYYY-MM-DD
                                if not candle_date_str: continue
                                
                                # Basic date comparison (assuming candles are daily or have timestamps)
                                # If daily candles (YYYY-MM-DD), we include the day of prediction and after
                                candle_date = datetime.strptime(candle_date_str, '%Y-%m-%d')
                                
                                # If candle is from same day or later, include it
                                # (Note: This might include price action before the specific *time* of prediction
                                # on the same day, but for daily candles that's the best we can do without 1m data.
                                # Given "Patient Sniper" logic, confirming a hit on the same day is acceptable
                                # as long as it hit the target price.)
                                if candle_date.date() > pred_ts.date():
                                    relevant_candles.append(candle)
                            
                            if relevant_candles:
                                # 3. Find Max High and Min Low in the relevant period
                                highs = [c.get('high', -1) for c in relevant_candles if c.get('high')]
                                lows = [c.get('low', 999999999) for c in relevant_candles if c.get('low')]
                                
                                if highs: high_since_entry = max(highs)
                                if lows: low_since_entry = min(lows)
                                
                                # Log detection for debugging
                                # logger.debug(f"Retroactive check for {prediction['symbol']}: High={high_since_entry}, Low={low_since_entry}")
                                
                        except Exception as e:
                            logger.debug(f"Error processing history for {prediction['symbol']}: {e}")

                    
                    # FALLBACK FOR EXPIRED PREDICTIONS WITH NO PRICE DATA:
                    # If price fetch returned 0 but the prediction is expired,
                    # use entry_price as fallback (guarantees 0% change = INCORRECT)
                    if current_price <= 0:
                        estimated_date_str = prediction.get('estimated_target_date')
                        if estimated_date_str:
                            try:
                                is_past_expiry = datetime.now() >= datetime.fromisoformat(estimated_date_str)
                            except:
                                is_past_expiry = False
                            
                            if is_past_expiry:
                                # Use entry_price as fallback — 0% change will mark as INCORRECT
                                current_price = prediction.get('entry_price', 0)
                                logger.warning(
                                    f"⚠️ Price fetch failed for {prediction['symbol']} (pred #{prediction['id']}), "
                                    f"but prediction expired on {estimated_date_str[:10]}. "
                                    f"Using entry_price ${current_price} as fallback."
                                )
                    
                    if current_price > 0:
                        # self.verify_prediction handles its own locking
                        # Pass the retroactive High/Low values
                        result = self.verify_prediction(
                            prediction['id'], 
                            current_price, 
                            high=high_since_entry, 
                            low=low_since_entry,
                            data_fetcher=data_fetcher
                        )
                        
                        if result:
                            # If verify_prediction returns a result, it means it was verified (Target/Stop hit)
                            verified_count += 1
                            if result['was_correct']:
                                correct_count += 1
                            # Add to newly verified list
                            newly_verified.append(result)
                except Exception as e:
                    # SAFETY NET: If exception occurs but prediction is way past expiry,
                    # force-expire it so it doesn't stay active forever
                    try:
                        estimated_date_str = prediction.get('estimated_target_date')
                        if estimated_date_str:
                            days_past = (now - datetime.fromisoformat(estimated_date_str)).days
                            if days_past > 7:
                                logger.warning(
                                    f"⚠️ Force-expiring prediction #{prediction['id']} for {prediction['symbol']} "
                                    f"({days_past} days past expiry, data fetch failed)"
                                )
                                with self.lock:
                                    real_pred = self.get_prediction_by_id(prediction['id'])
                                    if real_pred and real_pred['status'] == 'active':
                                        real_pred['verified'] = True
                                        real_pred['status'] = 'verified'
                                        real_pred['was_correct'] = False
                                        real_pred['verification_date'] = now.isoformat()
                                        real_pred['actual_price_at_target'] = prediction.get('entry_price', 0)
                                        real_pred['actual_price_change'] = 0.0
                                        real_pred['days_to_target'] = (now - datetime.fromisoformat(prediction['timestamp'])).days
                                        self.save()
                                        verified_count += 1
                                        newly_verified.append(real_pred)
                    except Exception as inner_e:
                        logger.debug(f"Error in safety net expiry: {inner_e}")
                    
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
        
        with self.lock:
            # Re-fetch prediction inside lock to ensure we have latest state (though get_prediction_by_id didn't lock?)
            # Since get_prediction_by_id iterates self.predictions, we should lock it or rely on GIL/Reference.
            # Best to lock entire block.
            prediction = self.get_prediction_by_id(prediction_id)
            if not prediction:
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
            try:
                from utils.market_utils import add_trading_days
                prediction['estimated_target_date'] = add_trading_days(datetime.now(), estimated_days).isoformat()
            except ImportError:
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
        """Delete a prediction by ID"""
        with self.lock:
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

