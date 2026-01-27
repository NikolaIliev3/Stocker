import time
import json
import logging
import numpy as np
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
        if not is_market_open():
            msg = get_market_status_message()
            logger.info(f"💤 {msg}. Skipping cycle...")
            
            # Update state with closed status for dashboard
            summary = self.broker.get_account_summary()
            stats = self.tracker.get_statistics()
            state = {
                "summary": summary,
                "predictions": {}, # Clear active predictions while closed
                "stats": stats,
                "holdings": self.broker.get_positions(),
                "market_status": "CLOSED",
                "market_message": msg,
                "timestamp": datetime.now().isoformat()
            }
            self._save_state(state)
            return

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
        
        state = {
            "summary": summary,
            "predictions": predictions,
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
        
        state = {
            "summary": summary,
            "predictions": {}, # Active session predictions (resets on restart)
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
                
                # Get HISTORY for Brain
                hist_data = self.fetcher.fetch_stock_history(symbol, period='2y')
                if not hist_data or not isinstance(hist_data, dict) or 'data' not in hist_data:
                     logger.warning(f"Could not fetch history for {symbol} (Invalid Format), skipping prediction.")
                     continue
                
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
                        reasoning=rec.get('reasoning', 'Live analysis')
                    )

                action = prediction['recommendation']['action']
                confidence = prediction['recommendation']['confidence']
                
                logger.info(f"🧠 Brain says for {symbol}: {action} ({confidence:.1f}%)")
                
                if action == 'SELL':
                    # Execute SELL
                    success = self.broker.execute_order(symbol, 'SELL', qty, current_price)
                    if success:
                        logger.info(f"📉 Exited position in {symbol}")
                
            except Exception as e:
                logger.error(f"Error managing position {symbol}: {e}")

    def _scan_for_entries(self, predictions):
        """Look for new buy opportunities using the scanner"""
        cash = self.broker.get_account_summary()['cash']
        if cash < 100: # Min cash needed
            logger.info("⚠️ Not enough cash to buy new stocks.")
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
                        reasoning=rec.get('reasoning', 'Live scan')
                    )

                action = prediction['recommendation']['action']
                confidence = prediction['recommendation']['confidence']
                
                logger.info(f"🧠 Brain says for {symbol}: {action} ({confidence:.1f}%)")
                
                if action == 'BUY' and confidence >= self.min_confidence and cash >= 100:
                    # Calculate Position Size (e.g. 10% of total portfolio value? or just 10% of cash?)
                    # Simple: 10% of current cash
                    invest_amount = cash * self.position_size_pct
                    qty = int(invest_amount / current_price)
                    
                    if qty > 0:
                        success = self.broker.execute_order(symbol, 'BUY', qty, current_price)
                        if success:
                            logger.info(f"🚀 Entered position in {symbol}")
                            cash -= (qty * current_price) # deductions approx
            except Exception as e:
                logger.error(f"Error scanning {symbol}: {e}")
                
if __name__ == "__main__":
    # Expand to a broader list to "look for buy opportunities"
    # Merging fixed tech list with scanner's popular list
    watchlist = list(set(["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "AMD", "META"] + MarketScanner.POPULAR_STOCKS))
    
    trader = PaperTrader(watchlist, initial_balance=25000)
    
    # Run loop
    logger.info(f"🚀 Paper Trader starting with dynamic watchlist: {len(watchlist)} stocks")
    while True:
        try:
            trader.run_cycle()
            logger.info("⌛ Cycle complete. Sleeping for 60s...")
            time.sleep(60)
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            time.sleep(10)
