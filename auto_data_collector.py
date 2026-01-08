"""
Automatic Data Collection System
Continuously collects training data from stock market to improve ML models over time
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import time

from data_fetcher import StockDataFetcher
from training_pipeline import MLTrainingPipeline
from config import APP_DATA_DIR

logger = logging.getLogger(__name__)


class AutoDataCollector:
    """Automatically collects training data on a schedule"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or APP_DATA_DIR
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.collection_log_file = self.data_dir / "data_collection_log.json"
        self.data_fetcher = StockDataFetcher()
        self.is_running = False
        self.collection_thread = None
        
        # Load collection log
        self.collection_log = self._load_log()
        
        # Default stocks to track (can be expanded)
        self.tracked_symbols = [
            # Tech
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX', 'AMD', 'INTC',
            # Finance
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'V', 'MA',
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO',
            # Consumer
            'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT',
            # Industrial
            'BA', 'CAT', 'GE', 'HON', 'UPS',
            # Energy
            'XOM', 'CVX', 'SLB', 'COP',
            # Other
            'DIS', 'VZ', 'T', 'CMCSA', 'NEE'
        ]
    
    def _load_log(self) -> Dict:
        """Load collection log"""
        if self.collection_log_file.exists():
            try:
                with open(self.collection_log_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading collection log: {e}")
        return {
            'last_collection': None,
            'collections': [],
            'total_samples_collected': 0,
            'symbols_tracked': {}
        }
    
    def _save_log(self):
        """Save collection log"""
        try:
            with open(self.collection_log_file, 'w') as f:
                json.dump(self.collection_log, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving collection log: {e}")
    
    def collect_training_data(self, strategy: str = 'trading', lookforward_days: int = 10,
                              days_back: int = 30) -> Dict:
        """
        Collect training data for the specified time period
        
        Args:
            strategy: 'trading', 'mixed', or 'investing'
            lookforward_days: How many days forward to look for labels
            days_back: How many days of historical data to collect
        """
        logger.info(f"🔄 Starting automatic data collection (strategy: {strategy}, days_back: {days_back})")
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        tp = MLTrainingPipeline(self.data_fetcher, self.data_dir, app=None)
        total_samples = 0
        successful_symbols = 0
        failed_symbols = 0
        
        for symbol in self.tracked_symbols:
            try:
                samples = tp.data_generator.generate_training_samples(
                    symbol, start_date, end_date, strategy, lookforward_days
                )
                
                if samples:
                    total_samples += len(samples)
                    successful_symbols += 1
                    
                    # Update symbol tracking
                    if symbol not in self.collection_log['symbols_tracked']:
                        self.collection_log['symbols_tracked'][symbol] = {
                            'total_samples': 0,
                            'last_collection': None,
                            'collections_count': 0
                        }
                    
                    self.collection_log['symbols_tracked'][symbol]['total_samples'] += len(samples)
                    self.collection_log['symbols_tracked'][symbol]['last_collection'] = datetime.now().isoformat()
                    self.collection_log['symbols_tracked'][symbol]['collections_count'] += 1
                    
                    logger.info(f"  ✅ {symbol}: {len(samples)} samples")
                else:
                    failed_symbols += 1
                    logger.warning(f"  ⚠️  {symbol}: No samples generated")
                    
            except Exception as e:
                failed_symbols += 1
                logger.error(f"  ❌ {symbol}: Error - {e}")
        
        # Update collection log
        collection_entry = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'days_back': days_back,
            'lookforward_days': lookforward_days,
            'total_samples': total_samples,
            'successful_symbols': successful_symbols,
            'failed_symbols': failed_symbols
        }
        
        self.collection_log['collections'].append(collection_entry)
        self.collection_log['last_collection'] = datetime.now().isoformat()
        self.collection_log['total_samples_collected'] += total_samples
        
        # Keep only last 100 collection entries
        if len(self.collection_log['collections']) > 100:
            self.collection_log['collections'] = self.collection_log['collections'][-100:]
        
        self._save_log()
        
        logger.info(f"✅ Collection complete: {total_samples} total samples from {successful_symbols} symbols")
        
        return {
            'success': True,
            'total_samples': total_samples,
            'successful_symbols': successful_symbols,
            'failed_symbols': failed_symbols,
            'collection_entry': collection_entry
        }
    
    def start_automatic_collection(self, interval_hours: int = 24, strategy: str = 'trading',
                                   lookforward_days: int = 10, days_back: int = 30):
        """
        Start automatic data collection on a schedule
        
        Args:
            interval_hours: How often to collect data (default: 24 hours = daily)
            strategy: Strategy to use for data collection
            lookforward_days: Lookforward days for labels
            days_back: Days of historical data to collect
        """
        if self.is_running:
            logger.warning("Data collection is already running")
            return
        
        self.is_running = True
        logger.info(f"🚀 Starting automatic data collection (every {interval_hours} hours)")
        
        def collection_loop():
            while self.is_running:
                try:
                    self.collect_training_data(strategy, lookforward_days, days_back)
                except Exception as e:
                    logger.error(f"Error in automatic collection: {e}")
                
                # Wait for next collection
                if self.is_running:
                    time.sleep(interval_hours * 3600)  # Convert hours to seconds
        
        self.collection_thread = threading.Thread(target=collection_loop, daemon=True)
        self.collection_thread.start()
    
    def stop_automatic_collection(self):
        """Stop automatic data collection"""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info("🛑 Stopping automatic data collection")
        
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
    
    def get_statistics(self) -> Dict:
        """Get collection statistics"""
        return {
            'is_running': self.is_running,
            'last_collection': self.collection_log.get('last_collection'),
            'total_samples_collected': self.collection_log.get('total_samples_collected', 0),
            'total_collections': len(self.collection_log.get('collections', [])),
            'symbols_tracked': len(self.collection_log.get('symbols_tracked', {})),
            'recent_collections': self.collection_log.get('collections', [])[-10:]  # Last 10
        }


if __name__ == '__main__':
    # Test the collector
    logging.basicConfig(level=logging.INFO)
    collector = AutoDataCollector()
    
    # Run a test collection
    result = collector.collect_training_data(strategy='trading', days_back=30)
    print(f"\nCollection result: {result}")
    
    # Show statistics
    stats = collector.get_statistics()
    print(f"\nStatistics: {json.dumps(stats, indent=2)}")
