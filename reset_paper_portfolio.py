
from mock_broker import MockBroker
from config import INITIAL_BALANCE
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("reset_portfolio")

def reset():
    print(f"🔄 Resetting Portfolio to {INITIAL_BALANCE}...")
    broker = MockBroker(initial_balance=INITIAL_BALANCE)
    broker.reset_account(INITIAL_BALANCE)
    print("✅ Portfolio Reset Complete.")

if __name__ == "__main__":
    reset()
