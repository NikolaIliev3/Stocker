
from pathlib import Path
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)

# Import the new selector
from smart_model_selector import select_best_strategy

# Define data dir
APP_DATA_DIR = Path("C:/Users/Никола/.stocker")

print(f"Testing Model Selection in: {APP_DATA_DIR}")
print("-" * 50)

# Run selection
best_strategy = select_best_strategy(APP_DATA_DIR)

print("-" * 50)
print(f"RESULT: System selected '{best_strategy}'")

if best_strategy == 'trading':
    print("SUCCESS: The robust 'trading' model was selected over the overfit 'mixed' model.")
else:
    print("WARNING: The system still preferred the 'mixed' model. Check the scores above.")
