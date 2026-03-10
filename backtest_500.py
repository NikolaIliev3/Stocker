"""
500-Sample Walk-Forward Backtest — Validates retrained ML model
Uses the walk-forward backtester on a subset of training stocks
"""
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
from walk_forward_backtester import WalkForwardBacktester
from hybrid_predictor import HybridStockPredictor
from data_fetcher import StockDataFetcher

# Use a representative subset for backtest (diverse sectors)
BACKTEST_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META',    # Tech
    'JPM', 'GS', 'BAC',                           # Financials
    'JNJ', 'LLY', 'UNH',                          # Healthcare
    'KO', 'PG', 'WMT', 'COST',                    # Consumer
    'CAT', 'DE', 'BA',                             # Industrials
    'XOM', 'CVX',                                   # Energy
    'TSLA',                                         # Auto/Growth
]

print(f"\n{'='*60}")
print(f"500-SAMPLE WALK-FORWARD BACKTEST")
print(f"{'='*60}")
print(f"Stocks: {len(BACKTEST_STOCKS)}")
print(f"{'='*60}\n")

data_dir = Path.home() / '.stocker'
data_fetcher = StockDataFetcher()

# Initialize predictor
predictor = HybridStockPredictor(data_dir=data_dir, data_fetcher=data_fetcher, strategy="trading")

# Initialize backtester
backtester = WalkForwardBacktester(data_dir=data_dir)

# Run multi-stock walk-forward
results = backtester.run_multi_stock_walk_forward_test(
    symbols=BACKTEST_STOCKS,
    predictor=predictor,
    data_fetcher=data_fetcher,
    strategy="trading",
    train_window_days=252,   # 1 year training window
    test_window_days=63,     # 3 months test window
    step_size_days=21,       # Step forward 1 month
    lookforward_days=15,     # Match training lookforward
    max_stocks=len(BACKTEST_STOCKS)
)

print(f"\n{'='*60}")
print(f"BACKTEST RESULTS")
print(f"{'='*60}")

if results and 'aggregate_results' in results:
    agg = results['aggregate_results']
    print(f"Overall accuracy: {agg.get('aggregate_accuracy', 0)*100:.1f}%")
    print(f"Std accuracy:     {agg.get('aggregate_std', 0)*100:.1f}%")
    print(f"Min accuracy:     {agg.get('aggregate_min', 0)*100:.1f}%")
    print(f"Max accuracy:     {agg.get('aggregate_max', 0)*100:.1f}%")
    print(f"Sharpe ratio:     {agg.get('aggregate_sharpe', 0):.2f}")
    
    total_trades = 0
    for stock_res in results.get('per_stock_results', []):
        res = stock_res.get('result', {})
        if 'aggregated' in res:
            total_trades += res['aggregated'].get('total_predictions', 0)
    print(f"Total trades:     {total_trades}")
    print(f"Stocks tested:    {results.get('stocks_tested', 0)}")
    
    # Per-stock breakdown
    if 'per_stock_results' in results:
        print(f"\nPer-stock breakdown:")
        for stock_result in results['per_stock_results']:
            symbol = stock_result.get('symbol', '?')
            res = stock_result.get('result', {})
            if 'aggregated' in res:
                acc = res['aggregated'].get('overall_accuracy', 0) * 100
                trades = res['aggregated'].get('total_predictions', 0)
                print(f"  {symbol}: {acc:.1f}% ({trades} trades)")
else:
    print("No results returned")

print(f"{'='*60}")
