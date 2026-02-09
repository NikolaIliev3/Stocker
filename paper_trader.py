import time
import json
import logging
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
from data_fetcher import StockDataFetcher
from hybrid_predictor import HybridStockPredictor
from mock_broker import MockBroker
from config import APP_DATA_DIR
from market_scanner import MarketScanner
from predictions_tracker import PredictionsTracker
from learning_tracker import LearningTracker
from utils.market_utils import is_market_open, get_market_status_message

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("paper_trader")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class PaperTrader:
    def __init__(self, symbols, initial_balance=50000.0, portfolio_file="paper_portfolio.json"):
        self.symbols = symbols
        self.broker = MockBroker(initial_balance=initial_balance, portfolio_file=portfolio_file)
        self.fetcher = StockDataFetcher()
        self.predictor = HybridStockPredictor(data_dir=APP_DATA_DIR, strategy='trading', data_fetcher=self.fetcher) # The Brain
        self.scanner = MarketScanner(self.fetcher, data_dir=APP_DATA_DIR) # The Eye
        self.tracker = PredictionsTracker(APP_DATA_DIR) # The Memory
        self.learning_tracker = LearningTracker(APP_DATA_DIR) # The Student
        self.min_confidence = 65.0 # Minimum confidence to trade
        self.position_size_pct = 0.10 # Max 10% of portfolio per stock
        
        # Initial sync of status for dashboard
        self.sync_and_save_state()

    def run_cycle(self):
        """Run one pass of analysis and trading"""
        # 00. Market Hours Guard
        # if not is_market_open():
        #     msg = get_market_status_message()
        #     logger.info(f"💤 {msg}. Skipping cycle...")
            
        #     # Update state with closed status for dashboard
        #     summary = self.broker.get_account_summary()
        #     stats = self.tracker.get_statistics()
        #     state = {
        #         "summary": summary,
        #         "predictions": {}, # Clear active predictions while closed
        #         "stats": stats,
        #         "holdings": self.broker.get_positions(),
        #         "market_status": "CLOSED",
        #         "market_message": msg,
        #         "timestamp": datetime.now().isoformat()
        #     }
        #     self._save_state(state)
        #     # return  <-- BYPASS FOR TESTING

        logger.info(f"🔄 Starting Trading Cycle for {len(self.symbols)} symbols...")
        
        # 0. Verify Past Predictions (Learning Loop)
        logger.info("📡 Verifying past work...")
        v_results = self.tracker.verify_all_active_predictions(self.fetcher)
        if v_results['verified'] > 0:
            logger.info(f"✅ Verified {v_results['verified']} predictions. Recent Accuracy: {v_results['accuracy']:.1f}%")
            # Feed outcomes to learning tracker
            for result in v_results.get('newly_verified', []):
                self.learning_tracker.record_verified_prediction(
                    was_correct=result['was_correct'],
                    strategy='trading',
                    prediction_data=result,
                    actual_outcome={'price': result['actual_price_at_target']}
                )

        # 1. Update Positions (Check for Exits)
        predictions = {} # Store for Dashboard
        self._manage_existing_positions(predictions)
        
        # 2. Scan for New Entries
        self._scan_for_entries(predictions)
        
        # 3. Summary & State Save
        summary = self.broker.get_account_summary()
        stats = self.tracker.get_statistics()
        logger.info(f"💰 Account Summary: Cash=${summary['cash']:,.2f} | Positions={summary['holdings_count']}")
        
        # --- FIX DASHBOARD SYNC ---
        # 1. Get ALL active predictions from persistent tracker
        active_preds_list = self.tracker.get_active_predictions()
        
        # 2. Convert to Dashboard Format
        merged_predictions = {}
        for p in active_preds_list:
            symbol = p['symbol']
            # Basic format from tracker
            merged_predictions[symbol] = {
                'recommendation': {
                    'action': p['action'],
                    'confidence': p['confidence'],
                    'entry_price': p['entry_price'],
                    'target_price': p['target_price'],
                    'stop_loss': p['stop_loss'],
                    'reasoning': p.get('reasoning', '')
                },
                'market_regime': p.get('market_regime', 'unknown'),
                'estimated_target_date': p.get('estimated_target_date'),
                'estimated_days': p.get('estimated_days'),
                'reasoning': p.get('reasoning', ''), # FIX: Top-level mapping for Dashboard
                'timestamp': p.get('timestamp')
            }
            
        # 3. Overlay Current Cycle Details (Rich Data)
        # If we just scanned it, we have more info (indicators, etc.)
        for symbol, pred in predictions.items():
            merged_predictions[symbol] = pred
            
        state = {
            "summary": summary,
            "predictions": merged_predictions, # Uses the merged full list
            "stats": stats,
            "holdings": self.broker.get_positions(),
            "timestamp": datetime.now().isoformat()
        }
        self._save_state(state)

    def sync_and_save_state(self):
        """Synchronize stats with tracker and save state"""
        summary = self.broker.get_account_summary()
        stats = self.tracker.get_statistics()
        
        market_open = is_market_open()
        msg = get_market_status_message()
        
        # --- FIX DASHBOARD SYNC (Startup) ---
        active_preds_list = self.tracker.get_active_predictions()
        merged_predictions = {}
        for p in active_preds_list:
            symbol = p['symbol']
            merged_predictions[symbol] = {
                'recommendation': {
                    'action': p['action'],
                    'confidence': p['confidence'],
                    'entry_price': p['entry_price'],
                    'target_price': p['target_price'],
                    'stop_loss': p['stop_loss'],
                    'reasoning': p.get('reasoning', '')
                },
                'market_regime': p.get('market_regime', 'unknown'),
                'estimated_target_date': p.get('estimated_target_date'),
                'estimated_days': p.get('estimated_days'),
                'reasoning': p.get('reasoning', ''), # FIX: Top-level mapping for Dashboard
                'timestamp': p.get('timestamp')
            }

        state = {
            "summary": summary,
            "predictions": merged_predictions, # Load persistent predictions on init
            "stats": stats,
            "holdings": self.broker.get_positions(),
            "market_status": "OPEN" if market_open else "CLOSED",
            "market_message": msg,
            "timestamp": datetime.now().isoformat()
        }
        self._save_state(state)

    def _save_state(self, state):
        state_file = Path.home() / ".stocker" / "paper_state.json"
        try:
            with open(state_file, "w") as f:
                json.dump(state, f, indent=4, cls=NpEncoder)
            logger.info(f"💾 Saved Paper State to {state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")

    def _manage_existing_positions(self, predictions):
        """Check if we should sell anything we own"""
        positions = self.broker.get_positions()
        
        for symbol, data in positions.items():
            qty = data['qty']
            logger.info(f"🔍 Checking Exit for {symbol} (Qty: {qty})...")
            
            try:
                # Get current price
                current_data = self.fetcher.get_current_data(symbol) 
                if not current_data:
                    logger.warning(f"Could not fetch current data for {symbol}, skipping.")
                    continue
                current_price = current_data['price']
                
                # --- CHECK AUTOMATIC EXITS (TP/SL) ---
                target_price = data.get('target_price')
                stop_loss = data.get('stop_loss')
                
                # ORPHAN LOCK: Check Intraday High
                # If high hit target, we assume limit order filled
                day_high = current_data.get('day_high', current_price)
                
                executed_exit = False
                
                if target_price and day_high >= target_price:
                    logger.info(f"✨ ORPHAN LOCK: {symbol} hit target {target_price:.2f} (High: {day_high:.2f}). Simulating Limit Fill.")
                    # Execute at TARGET price, not current price
                    success = self.broker.execute_order(symbol, 'SELL', qty, target_price) 
                    if success:
                        logger.info(f"💰 PROFIT SECURED on {symbol} (Limit Fill)")
                        executed_exit = True
                        
                elif stop_loss and current_price <= stop_loss:
                    if success:
                        logger.info(f"💰 PROFIT SECURED on {symbol}")
                        executed_exit = True
                        
                elif stop_loss and current_price <= stop_loss:
                    logger.info(f"🛑 STOP LOSS HIT for {symbol}! Price {current_price:.2f} <= Stop {stop_loss:.2f}")
                    success = self.broker.execute_order(symbol, 'SELL', qty, current_price)
                    if success:
                        logger.info(f"🛡️ STOPPED OUT on {symbol}")
                        executed_exit = True
                
                if executed_exit:
                    continue # Skip Brain check if already exited
                
                # --- BRAIN CHECK (Dynamic Exit) ---
                # Get HISTORY for Brain
                hist_data = self.fetcher.fetch_stock_history(symbol, period='2y')
                if not hist_data or not isinstance(hist_data, dict) or 'data' not in hist_data:
                     logger.warning(f"Could not fetch history for {symbol} (Invalid Format), skipping prediction.")
                     continue
                
                # --- PHASE 4: ADVANCED EXITS ---
                from config import GLOBAL_STOP_LOSS_PCT, TIME_STOP_DAYS
                
                # 1. HARD STOP LOSS (-4%)
                # We calculate this relative to AVG ENTRY PRICE from the broker
                entry_price = data.get('avg_price', current_price)
                if current_price <= entry_price * (1 - GLOBAL_STOP_LOSS_PCT/100):
                     logger.info(f"🛑 HARD STOP HIT for {symbol}: {current_price:.2f} <= {entry_price * (1 - GLOBAL_STOP_LOSS_PCT/100):.2f} (-{GLOBAL_STOP_LOSS_PCT}%)")
                     self.broker.execute_order(symbol, 'SELL', qty, current_price)
                     continue
                
                # 2. TIME STOP (15 Days)
                # We link the holding to the active prediction to get the start date
                # This assumes only one active prediction per symbol (reasonable for this system)
                active_preds = self.tracker.get_active_predictions()
                pred = next((p for p in active_preds if p['symbol'] == symbol), None)
                
                if pred:
                     pred_date = datetime.fromisoformat(pred['timestamp'])
                     days_in_trade = (datetime.now() - pred_date).days
                     
                     if days_in_trade >= TIME_STOP_DAYS:
                         logger.info(f"⌛ TIME STOP HIT for {symbol}: {days_in_trade} days >= {TIME_STOP_DAYS} days. Force Closing.")
                         self.broker.execute_order(symbol, 'SELL', qty, current_price)
                         
                         # Also expire the prediction in tracker (optional, verify_all handles it?)
                         # verify_all_active_predictions checks for expiry but defines success/fail.
                         # Here we just want to close the position.
                         continue

                
                # 3. TRAILING STOP (Breakeven at +2%)
                if current_price >= entry_price * 1.02:
                     # Check if we have a stop already. If not, or if it's below breakeven, raise it.
                     current_sl = data.get('stop_loss', 0)
                     if current_sl < entry_price:
                         logger.info(f"🛡️ TRAILING STOP: {symbol} is +2%. Raising Stop to Breakeven ({entry_price:.2f}).")
                         # We need to update the broker holding. MockBroker doesn't have an update method,
                         # so we hack it by accessing the dict directly (since we are in the same process)
                         self.broker.portfolio['holdings'][symbol]['stop_loss'] = entry_price
                         self.broker._save_portfolio()

                # 4. PARTIAL TAKING (+2.5%)
                # Complex state tracking required. Skipping for this iteration to ensure stability 
                # of the primary Hard Stop/Tiered Sizing first.

                # Ask the Brain (Pass the dicts)
                prediction = self.predictor.predict(current_data, history_data=hist_data)
                
                # Store for Dashboard
                predictions[symbol] = prediction
                
                # Add to Persistent Tracker (for verification later)
                active_preds = self.tracker.get_active_predictions()
                if not any(p['symbol'] == symbol for p in active_preds):
                    rec = prediction.get('recommendation', {})
                    self.tracker.add_prediction(
                        symbol=symbol,
                        strategy='trading',
                        action=rec.get('action', 'HOLD'),
                        entry_price=rec.get('entry_price', current_price),
                        target_price=rec.get('target_price', current_price * 1.05),
                        stop_loss=rec.get('stop_loss', current_price * 0.95),
                        confidence=rec.get('confidence', 0),
                        reasoning=prediction.get('reasoning', 'Live analysis'),
                        estimated_days=prediction.get('estimated_days'),
                        market_regime=prediction.get('market_regime', 'unknown')
                    )

                action = prediction['recommendation']['action']
                confidence = prediction['recommendation']['confidence']
                
                logger.info(f"🧠 Brain says for {symbol}: {action} ({confidence:.1f}%)")
                
                if action == 'SELL':
                    # Execute SELL
                    success = self.broker.execute_order(symbol, 'SELL', qty, current_price)
                    if success:
                        logger.info(f"📉 Exited position in {symbol} (Brain Signal)")
                
            except Exception as e:
                logger.error(f"Error managing position {symbol}: {e}")

    def _scan_for_entries(self, predictions):
        """Look for new buy opportunities using the scanner"""
        from config import MAX_CONCURRENT_POSITIONS, MAX_PORTFOLIO_ALLOCATION, BASE_POSITION_SIZE_PCT
        
        # 1. CHECK CONCURRENCY LIMITS
        current_positions = self.broker.get_positions()
        if len(current_positions) >= MAX_CONCURRENT_POSITIONS:
            logger.info(f"⚠️ Max Positions Reached ({len(current_positions)}/{MAX_CONCURRENT_POSITIONS}). Scanning paused.")
            return

        summary = self.broker.get_account_summary()
        cash = summary['cash']
        total_value = summary['total_value']
        
        # 2. CHECK ALLOCATION LIMIT
        invested_value = total_value - cash
        if invested_value >= total_value * MAX_PORTFOLIO_ALLOCATION:
             logger.info(f"⚠️ Max Efficiency Reached ({invested_value/total_value:.1%} used). Scanning paused.")
             return

        if cash < 10: # Min cash needed is now 10 EUR
            logger.info("⚠️ Not enough cash to buy new stocks (<$10).")
            # Still scan some to show on dashboard
            
        # Scan the symbols
        for symbol in self.symbols:
            # Skip if we already own it
            if symbol in self.broker.get_positions():
                continue
                
            logger.info(f"🔍 Analyzing {symbol} for potential entry...")
            try:
                # Get Current Data
                current_data = self.fetcher.get_current_data(symbol)
                if not current_data: continue
                current_price = current_data['price']
                
                # Get HISTORY for Brain
                hist_data = self.fetcher.fetch_stock_history(symbol, period='2y')
                if not hist_data or not isinstance(hist_data, dict) or 'data' not in hist_data: 
                     logger.warning(f"Could not fetch history for {symbol} (Invalid Format), skipping.")
                     continue
                
                # Ask the Brain
                prediction = self.predictor.predict(current_data, history_data=hist_data)
                
                # Validation
                if not prediction or not isinstance(prediction, dict) or 'recommendation' not in prediction:
                    logger.warning(f"Invalid prediction for {symbol}: {prediction}")
                    continue
                
                # Store for Dashboard
                predictions[symbol] = prediction

                # Add to Persistent Tracker (for verification later)
                active_preds = self.tracker.get_active_predictions()
                if not any(p['symbol'] == symbol for p in active_preds):
                    rec = prediction.get('recommendation', {})
                    self.tracker.add_prediction(
                        symbol=symbol,
                        strategy='trading',
                        action=rec.get('action', 'HOLD'),
                        entry_price=rec.get('entry_price', current_price),
                        target_price=rec.get('target_price', current_price * 1.05),
                        stop_loss=rec.get('stop_loss', current_price * 0.95),
                        confidence=rec.get('confidence', 0),
                        reasoning=prediction.get('reasoning', 'Live scan'),
                        estimated_days=prediction.get('estimated_days'),
                        market_regime=prediction.get('market_regime', 'unknown')
                    )

                action = prediction['recommendation']['action']
                confidence = prediction['recommendation']['confidence']
                
                logger.info(f"🧠 Brain says for {symbol}: {action} ({confidence:.1f}%)")
                
                if action == 'BUY' and confidence >= self.min_confidence and cash >= 10:
                    # --- TIERED RISK MANAGEMENT (Config 2 Standard) ---
                    from config import (ELITE_CONF_RANGE, ELITE_MAX_POS, ELITE_MULTIPLIER, 
                                      STANDARD_CONF_RANGE, STANDARD_MAX_POS, STANDARD_MULTIPLIER, 
                                      PLATEAU_THRESHOLD)
                    
                    # 1. CATEGORIZE EXISTING POSITIONS
                    active_holdings = self.broker.get_positions()
                    active_preds = self.tracker.get_active_predictions()
                    
                    elite_held = 0
                    standard_held = 0
                    
                    for sym in active_holdings:
                        # Find the confidence for this held stock in tracker
                        matching_pred = next((p for p in active_preds if p['symbol'] == sym), None)
                        if matching_pred:
                            conf = matching_pred.get('confidence', 0)
                            if ELITE_CONF_RANGE[0] <= conf < ELITE_CONF_RANGE[1]:
                                elite_held += 1
                            elif STANDARD_CONF_RANGE[0] <= conf < STANDARD_CONF_RANGE[1]:
                                standard_held += 1
                    
                    # 2. TIERED SIZING & CONCURRENCY CHECKS
                    size_multiplier = 0
                    tier_name = ""
                    
                    if confidence >= PLATEAU_THRESHOLD:
                        logger.info(f"🚫 PLATEAU SKIP: {symbol} at {confidence:.1f}% is overextended. Not worth the risk.")
                        continue
                    
                    elif ELITE_CONF_RANGE[0] <= confidence < ELITE_CONF_RANGE[1]:
                        if elite_held >= ELITE_MAX_POS:
                            logger.info(f"⚠️ [ELITE] Lapped: Already at Max Elite ({ELITE_MAX_POS}). Skipping {symbol}.")
                            continue
                        size_multiplier = ELITE_MULTIPLIER
                        tier_name = "ELITE"
                    
                    elif STANDARD_CONF_RANGE[0] <= confidence < STANDARD_CONF_RANGE[1]:
                        # User Rule: Standard only when no Elite held
                        if elite_held > 0:
                            logger.info(f"🛡️ ELITE PRIORITY: Skipping Standard signal {symbol} because {elite_held} Elite position(s) are active.")
                            continue
                        if standard_held >= STANDARD_MAX_POS:
                            logger.info(f"⚠️ [STANDARD] Lapped: Already at Max Standard ({STANDARD_MAX_POS}). Skipping {symbol}.")
                            continue
                        size_multiplier = STANDARD_MULTIPLIER
                        tier_name = "STANDARD"
                    
                    else:
                        logger.info(f"⚠️ SKIP: Confidence {confidence:.1f}% is below current production tiers.")
                        continue
                                         
                    # 3. POSITION SIZING
                    base_amount = total_value * BASE_POSITION_SIZE_PCT
                    ideal_amount = base_amount * size_multiplier
                    
                    # Apply Min/Max constraints
                    min_trade = 10.0
                    invest_amount = max(min_trade, ideal_amount)
                    
                    if invest_amount > cash:
                        logger.info(f"⚠️ Limited by Cash. Wanted ${ideal_amount:.2f}, having ${cash:.2f}")
                        invest_amount = cash
                    
                    # Calculate Fractional Quantity
                    qty = invest_amount / current_price
                    
                    if qty > 0.0001: # Minimum fractional share
                        # Get Targets from Prediction for Automatic Exits
                        rec = prediction.get('recommendation', {})
                        tp = rec.get('target_price')
                        sl = rec.get('stop_loss')
                        
                        success = self.broker.execute_order(
                            symbol, 'BUY', qty, current_price, 
                            target_price=tp, stop_loss=sl
                        )
                        if success:
                            logger.info(f"🚀 Entered {tier_name} position in {symbol} (${invest_amount:.2f}, {size_multiplier}x multiplier)")
                            cash -= invest_amount # deductive approx for loop
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Paper Trader for Stocker")
    parser.add_argument("--passive", action="store_true", help="Passive mode: only sync state, skip autonomous cycles")
    args = parser.parse_args()
    
    # Expand to a broader list to "look for buy opportunities"
    # Merging fixed tech list with scanner's popular list
    watchlist = list(set(["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "AMD", "META"] + MarketScanner.POPULAR_STOCKS))
    
    trader = PaperTrader(watchlist, initial_balance=275.0) 
    
    # Run loop
    logger.info("==========================================")
    if args.passive:
        logger.info("🚀 PAPER TRADER STARTED IN PASSIVE MODE 🚀")
        logger.info("   (Autonomous scanning disabled)      ")
    else:
        logger.info("🚀 PAPER TRADER V2 (SMART SIZING) STARTED 🚀")
    logger.info("==========================================")
    logger.info(f"🚀 Watchlist: {len(watchlist)} stocks")
    
    while True:
        try:
            if args.passive:
                # In passive mode, just sync and sleep
                trader.sync_and_save_state()
                logger.info("⌛ Passive sync complete. Sleeping for 300s...")
                time.sleep(300)
            else:
                trader.run_cycle()
                logger.info("⌛ Cycle complete. Sleeping for 60s...")
                time.sleep(60)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            time.sleep(10)
