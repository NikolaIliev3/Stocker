import os
import json
import logging
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from pathlib import Path
from datetime import datetime, timedelta
try:
    from safe_storage import LockingJSONStorage
except ImportError:
    # Use dummy safe storage if not available (should be there)
    class LockingJSONStorage:
        def __init__(self, path): self.path = path
        def load(self, default=None):
            if not self.path.exists(): return default
            import json
            try:
                with open(self.path, 'r') as f: return json.load(f)
            except: return default

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("dashboard_api")

app = Flask(__name__)
CORS(app) # Enable CORS for development

try:
    from safe_storage import LockingJSONStorage
except ImportError:
    # Use dummy safe storage if not available (should be there)
    class LockingJSONStorage:
        def __init__(self, path): self.path = path
        def load(self, default=None):
            if not self.path.exists(): return default
            import json
            try:
                with open(self.path, 'r') as f: return json.load(f)
            except: return default

from config import PREDICTIONS_FILE, STATE_FILE, PORTFOLIO_FILE

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("dashboard_api")

app = Flask(__name__)
CORS(app) # Enable CORS for development

@app.route('/')
def index():
    try:
        html_path = Path(__file__).parent / "dashboard.html"
        logger.info(f"Serving dashboard from: {html_path}")
        if not html_path.exists():
            return f"Error: {html_path} not found on server", 404
        from flask import send_file
        return send_file(str(html_path))
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        return str(e), 500

@app.route('/api/state')
def get_state():
    state = {
        "summary": {"cash": 0, "holdings_count": 0},
        "predictions": {},
        "holdings": {},
        "global_history": []
    }
    
    # 1. Load Paper Trader State (Real-time loop)
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r") as f:
                state.update(json.load(f))
        except Exception as e:
            logger.error(f"Error reading state file: {e}")

    # 2. Load Global Prediction History (GUI + Bot)
    predictions_file = PREDICTIONS_FILE
    
    # Use Safe Storage to load predictions
    try:
        storage = LockingJSONStorage(predictions_file)
        history_data = storage.load(default={})
        history = history_data.get('predictions', [])
        
        # Dynamic Stats Calculation (Override stale state if needed)
        verified_preds = [p for p in history if p.get('verified') or p.get('status') == 'verified']
        correct = sum(1 for p in verified_preds if p.get('was_correct') == True)
        active = [p for p in history if p.get('status') == 'active']
                
        accuracy = (correct / len(verified_preds) * 100) if verified_preds else 0
        
        # If paper_state is stale or missing stats, use dynamic ones
        if "stats" not in state or state["stats"].get("verified", 0) == 0:
            state["stats"] = {
                "accuracy": accuracy,
                "verified": len(verified_preds),
                "active": len(active),
                "recent_accuracy": accuracy # Simplified for dashboard
            }

        # Sort by timestamp (newest first) and take last 20
        history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        state['global_history'] = history[:20] 

        # Check if paper_state is stale (older than 5 mins)
        is_stale = False
        if "timestamp" in state:
            try:
                state_time = datetime.fromisoformat(state.get("timestamp"))
                if datetime.now() - state_time > timedelta(minutes=5):
                    is_stale = True
            except:
                is_stale = True

        # [FIX] FORCE dynamic stats if state is stale OR just to be safe (Audit Trail > Cache)
        # We always trust the predictions.json for the cognitive health score
        state["stats"] = {
            "accuracy": accuracy,
            "verified": len(verified_preds),
            "active": len(active),
            "recent_accuracy": accuracy
        }

        # [FIX] If state is stale, the 'predictions' dict (Real-time signals) is also old.
        # Clear it so we can inject the ACTUAL active predictions from history.
        if is_stale:
            state['predictions'] = {}
            
            # [FIX] Also load accurate holdings from portfolio file if state is stale
            if PORTFOLIO_FILE.exists():
                try:
                    with open(PORTFOLIO_FILE, "r") as f:
                        portfolio = json.load(f)
                        state['holdings'] = portfolio.get('holdings', {})
                        state['summary'] = {
                            "cash": portfolio.get('cash', 0),
                            "holdings_count": len(portfolio.get('holdings', {}))
                        }
                except Exception as e:
                    logger.error(f"Error reading portfolio file: {e}")

        # Inject Active Predictions if list is empty (or emptied above)
        if not state['predictions'] or len(state['predictions']) == 0:
            active_from_history = [p for p in history if p.get('status') == 'active']
            # Convert list to dict format expected by dashboard
            injected_preds = {}
            for p in active_from_history:
                # Map history format to dashboard format
                injected_preds[p['symbol']] = {
                    "price": 0, # Current price might be stale/unknown w/o fetcher
                    "recommendation": {
                        "action": p['action'],
                        "confidence": p['confidence'],
                        "entry_price": p.get('entry_price'),
                        "target_price": p.get('target_price'),
                        "stop_loss": p.get('stop_loss'),
                    },
                    "market_regime": p.get('market_regime', 'unknown'),
                    "estimated_target_date": p.get('estimated_target_date'),
                    "estimated_days": p.get('estimated_days'),
                    "timestamp": p.get('timestamp'),
                    "safety_lock": {"lock_active": False},
                    "reasoning": p.get('reasoning', 'No detailed reasoning available.')
                }
            state['predictions'] = injected_preds

        # [FIX] Ensure estimated_target_date is present for all predictions
        if 'predictions' in state and state['predictions']:
            for symbol, pred in state['predictions'].items():
                # Check nested recommendation for better estimated_days if needed
                rec = pred.get('recommendation', {})
                days = pred.get('estimated_days') or rec.get('estimated_days') or 7
                
                if 'estimated_target_date' not in pred or not pred['estimated_target_date'] or pred.get('estimated_days') == 7:
                    # Re-calculate if missing OR potentially using the old '7 day' bug
                    timestamp_str = pred.get('timestamp')
                    if timestamp_str:
                        try:
                            ts = datetime.fromisoformat(timestamp_str)
                            target_date = ts + timedelta(days=int(days))
                            pred['estimated_target_date'] = target_date.isoformat()
                            pred['estimated_days'] = int(days) # Sync top-level
                        except Exception as e:
                            logger.warning(f"Could not calculate target date for {symbol}: {e}")
        
        # [FIX] Ensure market_regime is present and not "unknown"
        if 'predictions' in state and state['predictions']:
            for symbol, pred in state['predictions'].items():
                if pred.get('market_regime') == 'unknown':
                    # Try to infer from reasoning
                    reasoning = pred.get('reasoning', '').lower()
                    if 'bull' in reasoning or 'uptrend' in reasoning:
                        pred['market_regime'] = 'bull'
                    elif 'bear' in reasoning or 'downtrend' in reasoning:
                        pred['market_regime'] = 'bear'
                    elif 'sideways' in reasoning or 'neutral' in reasoning:
                        pred['market_regime'] = 'sideways'
                    else:
                        # Default to bull as a safe assumption for current high-conviction BUYs
                        pred['market_regime'] = 'bull'
    
    except Exception as e:
        logger.error(f"Error reading predictions history: {e}")
        
    return jsonify(state)

if __name__ == '__main__':
    # Force port 5000 or similar
    logger.info("🚀 Dashboard API starting on http://127.0.0.1:5001")
    app.run(host='127.0.0.1', port=5001, debug=True)
