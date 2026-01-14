
import os
import logging
from pathlib import Path
from data_fetcher import StockDataFetcher
from market_scanner import MarketScanner
from hybrid_predictor import HybridStockPredictor
from config import APP_DATA_DIR

# Set up logging to console only for this script
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    # Setup paths
    # Note: Using project-local data if .stocker is not accessible
    data_dir = APP_DATA_DIR
    if not data_dir.exists():
        data_dir = Path("data")
        data_dir.mkdir(exist_ok=True)
    
    logger.info(f"Using data directory: {data_dir}")
    
    # Initialize components
    fetcher = StockDataFetcher()
    scanner = MarketScanner(fetcher, data_dir=data_dir)
    
    # Analyze a few favorite stocks
    favorites = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'AMZN']
    strategy = 'trading'
    
    logger.info(f"\n{'='*60}")
    logger.info(f"MARKET ANALYSIS - {strategy.upper()} STRATEGY")
    logger.info(f"{'='*60}\n")
    
    for symbol in favorites:
        try:
            logger.info(f"Analyzing {symbol}...")
            
            # Fetch data
            stock_data = fetcher.fetch_stock_data(symbol)
            history_data = fetcher.fetch_stock_history(symbol)
            
            # Predict
            predictor = scanner.hybrid_predictors.get(strategy)
            if predictor:
                analysis = predictor.predict(stock_data, history_data)
                
                recommendation = analysis.get('recommendation', {})
                action = recommendation.get('action', 'HOLD')
                confidence = recommendation.get('confidence', 0)
                reasoning = analysis.get('reasoning', '')
                
                # Print summary
                emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "🟡"
                logger.info(f"{emoji} {symbol}: {action} ({confidence:.1f}% confidence)")
                
                # Split reasoning to show the Hybrid part
                if "🤖 HYBRID ANALYSIS" in reasoning:
                    hybrid_part = reasoning.split("🤖 HYBRID ANALYSIS")[1]
                    logger.info(f"   🤖 Hybrid Info:{hybrid_part.strip()}")
                
                logger.info(f"   Price: ${stock_data.get('price', 0):.2f} ({stock_data.get('change_percent', 0):+.2f}%)")
                logger.info("-" * 40)
            else:
                logger.warning(f"No predictor found for {strategy}")
                
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")

if __name__ == "__main__":
    main()
