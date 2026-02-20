import json
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("mock_broker")

class MockBroker:
    def __init__(self, initial_balance=10000.0, portfolio_file="paper_portfolio.json", slippage_pct=0.001):
        """
        initial_balance: Starting cash (USD)
        portfolio_file: Where to save the state
        slippage_pct: Simulated slippage deviation (default 0.1%)
        """
        self.portfolio_file = Path.home() / ".stocker" / portfolio_file
        self.slippage = slippage_pct
        
        # Load or Initialize
        if self.portfolio_file.exists():
            self._load_portfolio()
        else:
            self.portfolio = {
                "cash": float(initial_balance),
                "initial_balance": float(initial_balance),
                "holdings": {},  # { "AAPL": { "qty": 10, "avg_price": 150.0 } }
                "history": [],   # List of transaction dicts
                "created_at": datetime.now().isoformat()
            }
            self._save_portfolio()
            logger.info(f"🆕 Created new paper portfolio with ${initial_balance:,.2f}")

    def _save_portfolio(self):
        # Ensure dir exists
        self.portfolio_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.portfolio_file, "w") as f:
            json.dump(self.portfolio, f, indent=4)

    def _load_portfolio(self):
        try:
            with open(self.portfolio_file, "r") as f:
                self.portfolio = json.load(f)
            logger.info(f"📂 Loaded portfolio. Cash: ${self.portfolio['cash']:,.2f}")
        except Exception as e:
            logger.error(f"Failed to load portfolio: {e}")
            # Fallback?
    
    def get_account_summary(self):
        return {
            "cash": self.portfolio["cash"],
            "holdings_count": len(self.portfolio["holdings"]),
            "total_value": self._calculate_total_value() # This requires live prices, strictly implies simulated
        }
        
    def _calculate_total_value(self):
        # In a real broker, we'd fetch live prices. 
        # For simple summary, we return cash + cost_basis (approx).
        # To get Real/Market Value, we need to pass current prices in.
        total = self.portfolio["cash"]
        for symbol, data in self.portfolio["holdings"].items():
            # Using avg_price as a placeholder if active price unknown
            total += data["qty"] * data["avg_price"] 
        return total

    def execute_order(self, symbol, action, quantity, current_price, timestamp=None, target_price=None, stop_loss=None):
        """
        Execute a buy/sell order.
        action: 'BUY' or 'SELL'
        """
        if quantity <= 0:
            logger.warning("Order quantity must be positive.")
            return False

        if not timestamp:
            timestamp = datetime.now().isoformat()
            
        # Apply Slippage (Simulated market reality)
        # BUY: Pay slightly more. SELL: Get slightly less.
        executed_price = current_price
        if action == "BUY":
            executed_price = current_price * (1 + self.slippage)
        elif action == "SELL":
            executed_price = current_price * (1 - self.slippage)
            
        total_cost = executed_price * quantity
        
        if action == "BUY":
            if self.portfolio["cash"] >= total_cost:
                self.portfolio["cash"] -= total_cost
                
                # Update Holdings
                holding = self.portfolio["holdings"].get(symbol, {"qty": 0.0, "avg_price": 0.0})
                old_qty = holding["qty"]
                new_qty = old_qty + quantity
                # Recalculate average price
                old_cost = old_qty * holding["avg_price"]
                new_avg = (old_cost + total_cost) / new_qty
                
                self.portfolio["holdings"][symbol] = {
                    "qty": new_qty,
                    "avg_price": new_avg,
                    "target_price": target_price,
                    "stop_loss": stop_loss
                }
                
                self._record_transaction("BUY", symbol, quantity, executed_price, timestamp)
                self._save_portfolio()
                logger.info(f"✅ BOUGHT {quantity:.4f} {symbol} @ ${executed_price:.2f} (Total: ${total_cost:.2f})")
                return True
            else:
                logger.warning(f"❌ Insufficient Funds for {symbol}. Needed: ${total_cost:.2f}, Have: ${self.portfolio['cash']:.2f}")
                return False
                
        elif action == "SELL":
            holding = self.portfolio["holdings"].get(symbol)
            if holding and holding["qty"] >= quantity:
                self.portfolio["cash"] += total_cost
                
                # Update Holdings
                holding["qty"] -= quantity
                if holding["qty"] == 0:
                    del self.portfolio["holdings"][symbol]
                else:
                    # Avg Price doesn't change on SELL (FIFO/Avg logic)
                    pass
                
                self._record_transaction("SELL", symbol, quantity, executed_price, timestamp)
                self._save_portfolio()
                logger.info(f"✅ SOLD {quantity:.4f} {symbol} @ ${executed_price:.2f} (Total: ${total_cost:.2f})")
                return True
            else:
                logger.warning(f"❌ Insufficient share count to SELL {symbol}. Have: {holding['qty'] if holding else 0}")
                return False

    def _record_transaction(self, side, symbol, qty, price, timestamp):
        self.portfolio["history"].append({
            "timestamp": timestamp,
            "action": side,
            "symbol": symbol,
            "qty": qty,
            "price": price,
            "total": qty * price
        })
    
    def get_positions(self):
        return self.portfolio["holdings"]

    def reset_account(self, new_balance):
        self.portfolio = {
                "cash": float(new_balance),
                "initial_balance": float(new_balance),
                "holdings": {},
                "history": [],
                "created_at": datetime.now().isoformat()
        }
        self._save_portfolio()
        logger.info(f"🔄 Account Reset to ${new_balance:,.2f}")

if __name__ == "__main__":
    # Test
    broker = MockBroker(initial_balance=50000)
    broker.execute_order("AAPL", "BUY", 10, 150.00)
    broker.execute_order("AAPL", "SELL", 5, 155.00)
