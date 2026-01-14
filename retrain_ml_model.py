"""
Script to retrain ML model with temporal split and increased regularization
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta
from config import APP_DATA_DIR
from data_fetcher import StockDataFetcher
from training_pipeline import MLTrainingPipeline

def retrain_ml_model(strategy='trading', symbols=None, start_date=None, end_date=None):
    """Retrain ML model with fixed temporal split and regularization"""
    
    # Default symbols if not provided
    if symbols is None:
        symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'JPM', 'BAC', 'WFC', 'GS', 'MS',
            'KO', 'PEP', 'WMT', 'HD', 'MCD',
            'JNJ', 'PFE', 'UNH', 'ABT',
            'BA', 'CAT', 'GE', 'MMM',
            'XOM', 'CVX', 'COP'
        ]
    
    # Default date range: last 5 years
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
    
    print("=" * 60)
    print("ML Model Retraining")
    print("=" * 60)
    print(f"Strategy: {strategy}")
    print(f"Symbols: {', '.join(symbols[:10])}{'...' if len(symbols) > 10 else ''} ({len(symbols)} total)")
    print(f"Date range: {start_date} to {end_date}")
    print()
    print("Fixes applied:")
    print("  [OK] Temporal split (prevents data leakage)")
    print("  [OK] Increased regularization (reduces overfitting)")
    print()
    print("=" * 60)
    print()
    
    # Initialize components
    data_fetcher = StockDataFetcher()
    pipeline = MLTrainingPipeline(data_fetcher, APP_DATA_DIR)
    
    # Progress callback
    def progress(msg):
        print(f"  {msg}")
    
    # Train model
    print("Starting training...")
    print()
    
    try:
        result = pipeline.train_on_symbols(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            strategy=strategy,
            progress_callback=progress
        )
        
        if 'error' in result:
            print(f"\n[ERROR] Training failed: {result['error']}")
            return False
        
        # Print results
        print()
        print("=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Test Accuracy: {result.get('test_accuracy', 0)*100:.1f}%")
        print(f"Train Accuracy: {result.get('train_accuracy', 0)*100:.1f}%")
        print(f"Cross-Validation: {result.get('cv_mean', 0)*100:.1f}% ± {result.get('cv_std', 0)*100:.1f}%")
        print(f"Training Samples: {result.get('total_samples', 0)}")
        print()
        
        # Check for overfitting
        train_acc = result.get('train_accuracy', 0)
        test_acc = result.get('test_accuracy', 0)
        gap = (train_acc - test_acc) * 100
        
        if gap > 25:
            print(f"[WARNING] High overfitting detected (gap: {gap:.1f}%)")
            print("   Consider further increasing regularization")
        elif gap > 15:
            print(f"[WARNING] Moderate overfitting (gap: {gap:.1f}%)")
        else:
            print(f"[OK] Good generalization (gap: {gap:.1f}%)")
        
        print()
        print("Next step: Run backtest to verify real-world accuracy")
        print("Expected: Backtest accuracy should be closer to test accuracy now")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Retrain ML model')
    parser.add_argument('--strategy', choices=['trading', 'mixed', 'investing'], 
                       default='trading', help='Strategy to train')
    parser.add_argument('--symbols', nargs='+', 
                       help='Stock symbols to train on (default: 20 popular stocks)')
    parser.add_argument('--start-date', 
                       help='Start date (YYYY-MM-DD, default: 5 years ago)')
    parser.add_argument('--end-date', 
                       help='End date (YYYY-MM-DD, default: today)')
    
    args = parser.parse_args()
    
    success = retrain_ml_model(
        strategy=args.strategy,
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    sys.exit(0 if success else 1)
