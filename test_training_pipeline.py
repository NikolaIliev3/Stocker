import logging
import sys
from pathlib import Path
import threading
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from data_fetcher import StockDataFetcher
from training_pipeline import MLTrainingPipeline

def test_pipeline():
    print("Testing MLTrainingPipeline initialization...")
    try:
        data_dir = Path("data")
        fetcher = StockDataFetcher()
        pipeline = MLTrainingPipeline(fetcher, data_dir)
        print("Pipeline initialized successfully.")
        
        # Test train_on_symbols with a small subset
        symbols = ["AAPL", "MSFT"]
        start_date = "2023-01-01"
        end_date = "2023-06-01"
        strategy = "trading"
        
        print(f"Testing train_on_symbols with {symbols}...")
        
        def progress_callback(msg):
            print(f"PROGRESS: {msg}")
            
        result = pipeline.train_on_symbols(
            symbols, start_date, end_date, strategy, 
            progress_callback=progress_callback
        )
        
        print("\nTraining Result:")
        print(result)
        
        if 'error' in result:
            print(f"❌ Training failed: {result['error']}")
        else:
            print("✅ Training simulated successfully!")
            
    except Exception as e:
        print(f"❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pipeline()
