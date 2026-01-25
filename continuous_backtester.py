"""
Continuous Backtesting System
Tests the algorithm on random historical points to validate and improve performance
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import logging
from pathlib import Path
import random
import threading
import time

from data_fetcher import StockDataFetcher
from hybrid_predictor import HybridStockPredictor

# Import benchmarking if available
try:
    from model_benchmarking import ModelBenchmarker
    HAS_BENCHMARKING = True
except ImportError:
    HAS_BENCHMARKING = False
    logger = logging.getLogger(__name__)
    logger.debug("Model benchmarking not available")

# Import walk-forward backtester if available
try:
    from walk_forward_backtester import WalkForwardBacktester
    HAS_WALK_FORWARD = True
except ImportError:
    HAS_WALK_FORWARD = False
    logger = logging.getLogger(__name__)
    logger.debug("Walk-forward backtester not available")

logger = logging.getLogger(__name__)


class ContinuousBacktester:
    """Continuously backtests the algorithm on random historical points"""
    
    def __init__(self, app, data_dir: Path):
        self.app = app
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = data_dir / "backtest_results.json"
        self.is_running = False
        self.is_testing = False
        
        # Configuration
        self.test_interval_hours = 24  # Run backtests daily
        self.tests_per_run = 100  # Test 100 random points per run
        self.min_history_days = 60  # Reduced from 100 to 60 to find more valid test points
        self.lookforward_days = {
            'trading': 10,
            'mixed': 21,
            'investing': 547
        }
        
        # Results tracking
        self.results = self._load_results()
        
        # Track last retraining time to prevent too-frequent retraining
        self.last_retrain_time = {}
        
        # Track which strategies were tested in the last run (for report filtering)
        self.last_tested_strategies = []
        
        # Continuous backtesting mode (non-stop until stopped)
        self.continuous_mode = False
        self.continuous_loop_active = False
        self.continuous_delay_seconds = 60  # Wait 60 seconds between backtest runs in continuous mode
        
        # Track dates tested in current run (to allow same-run training while blocking previous runs)
        self.current_run_tested_dates = set()  # {(symbol, date_str)}
        self.current_run_start_time = None
        
        # ANTI-OVERFITTING SAFEGUARDS
        # Track which dates have been tested (symbol + date combinations)
        # Format: {(symbol, date_str): timestamp_when_tested}
        self.tested_dates = self._load_tested_dates()
        
        # Minimum cooldown between retrains (hours)
        self.retrain_cooldown_hours = 24  # Don't retrain more than once per day
        
        # Maximum overlap threshold: don't retrain if >X% of samples are from recently tested dates
        self.max_overlap_threshold = 0.3  # 30% max overlap with tested dates
        
        # Days to consider "recently tested" (dates tested within this window are excluded from training)
        self.recent_test_window_days = 90  # 3 months
        
        # Initialize walk-forward backtester if available
        if HAS_WALK_FORWARD:
            try:
                self.walk_forward_backtester = WalkForwardBacktester(data_dir)
                logger.info("Walk-forward backtester initialized")
            except Exception as e:
                logger.warning(f"Could not initialize walk-forward backtester: {e}")
                self.walk_forward_backtester = None
        else:
            self.walk_forward_backtester = None
        
        # Initialize benchmarker
        if HAS_BENCHMARKING:
            try:
                self.benchmarker = ModelBenchmarker(data_dir)
            except Exception as e:
                logger.warning(f"Could not initialize benchmarker: {e}")
                self.benchmarker = None
        else:
            self.benchmarker = None
        
        # Popular stocks to test
        self.test_symbols = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX',
            'JPM', 'BAC', 'WFC', 'GS', 'MS',
            'KO', 'PEP', 'WMT', 'TGT', 'HD', 'MCD',
            'JNJ', 'PFE', 'UNH', 'ABT',
            'BA', 'CAT', 'GE', 'MMM',
            'XOM', 'CVX', 'COP',
            'DIS', 'V', 'MA', 'PG', 'NKE'
        ]
        
        logger.info("Continuous backtester initialized")
    
    def _load_results(self) -> Dict:
        """Load previous backtest results"""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    data = json.load(f)
                    # Ensure strategy_results exists and only contains tested strategies
                    if 'strategy_results' not in data:
                        data['strategy_results'] = {}
                    return data
            except Exception as e:
                logger.error(f"Error loading backtest results: {e}")
        
        # Default: empty strategy_results (will be populated as strategies are tested)
        return {
            'total_tests': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'accuracy_history': [],
            'test_details': [],
            'last_test_date': None,
            'strategy_results': {}  # Empty - will be populated only for tested strategies
        }
    
    def _load_tested_dates(self) -> Dict:
        """Load tracked tested dates to prevent data leakage"""
        tested_dates_file = self.data_dir / "tested_dates.json"
        if tested_dates_file.exists():
            try:
                with open(tested_dates_file, 'r') as f:
                    data = json.load(f)
                    # Convert string keys back to tuples
                    tested_dates = {}
                    for key, value in data.items():
                        # Key format: "SYMBOL|YYYY-MM-DD"
                        if '|' in key:
                            symbol, date_str = key.split('|', 1)
                            tested_dates[(symbol.upper(), date_str)] = value
                    return tested_dates
            except Exception as e:
                logger.error(f"Error loading tested dates: {e}")
        
        return {}
    
    def _save_tested_dates(self):
        """Save tracked tested dates"""
        tested_dates_file = self.data_dir / "tested_dates.json"
        try:
            # Convert tuple keys to strings for JSON
            data = {}
            for (symbol, date_str), timestamp in self.tested_dates.items():
                key = f"{symbol}|{date_str}"
                data[key] = timestamp
            
            # Keep only last 10000 entries to prevent file from growing too large
            if len(data) > 10000:
                # Sort by timestamp and keep most recent
                sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)
                data = dict(sorted_items[:10000])
            
            with open(tested_dates_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving tested dates: {e}")
    
    def _mark_date_as_tested(self, symbol: str, test_date: datetime):
        """Mark a date as tested to prevent using it for training"""
        date_str = test_date.strftime('%Y-%m-%d')
        key = (symbol.upper(), date_str)
        self.tested_dates[key] = datetime.now().isoformat()
    
    def _is_date_tested(self, symbol: str, test_date: datetime) -> bool:
        """Check if a date has been tested before"""
        date_str = test_date.strftime('%Y-%m-%d')
        key = (symbol.upper(), date_str)
        return key in self.tested_dates
    
    def _is_date_recently_tested(self, symbol: str, test_date: datetime, allow_current_run: bool = False) -> bool:
        """Check if a date was tested recently (within recent_test_window_days)
        
        Args:
            symbol: Stock symbol
            test_date: Historical date that was tested
            allow_current_run: If True, allow dates from current run (for same-run training)
        """
        date_str = test_date.strftime('%Y-%m-%d')
        key = (symbol.upper(), date_str)
        
        # If allowing current run, check if this date is from current run first
        if allow_current_run and key in self.current_run_tested_dates:
            return False  # Not "recently tested" if from current run (allow it)
        
        if key not in self.tested_dates:
            return False
        
        try:
            tested_timestamp = datetime.fromisoformat(self.tested_dates[key])
            days_ago = (datetime.now() - tested_timestamp).days
            return days_ago < self.recent_test_window_days
        except Exception as e:
            logger.debug(f"Error checking if date recently tested: {e}")
            return False
    
    def _calculate_overlap_with_tested_dates(self, samples: List[Dict]) -> float:
        """Calculate what percentage of samples are from recently tested dates
        (excluding current run dates, as those are allowed for training)"""
        if not samples:
            return 0.0
        
        recently_tested_count = 0
        for sample in samples:
            symbol = sample.get('symbol', '').upper()
            date_str = sample.get('date', '')
            
            if symbol and date_str:
                try:
                    sample_date = datetime.fromisoformat(date_str) if 'T' in date_str else datetime.strptime(date_str, '%Y-%m-%d')
                    # Allow current run dates (allow_current_run=True)
                    if self._is_date_recently_tested(symbol, sample_date, allow_current_run=True):
                        recently_tested_count += 1
                except Exception as e:
                    logger.debug(f"Error parsing date in sample: {e}")
        
        return recently_tested_count / len(samples) if samples else 0.0
    
    def save_results(self, tested_strategies: List[str] = None):
        """Save backtest results
        
        Args:
            tested_strategies: List of strategies that were actually tested.
                              If provided, only these strategies will be saved in strategy_results.
        """
        try:
            # Keep only last 1000 test details
            if len(self.results['test_details']) > 1000:
                self.results['test_details'] = self.results['test_details'][-1000:]
            
            # Keep only last 100 accuracy history entries
            if len(self.results['accuracy_history']) > 100:
                self.results['accuracy_history'] = self.results['accuracy_history'][-100:]
            
            # Filter strategy_results to only include tested strategies
            if tested_strategies is not None:
                # Only keep results for strategies that were actually tested
                filtered_strategy_results = {}
                for strategy in tested_strategies:
                    if strategy in self.results['strategy_results']:
                        filtered_strategy_results[strategy] = self.results['strategy_results'][strategy]
                self.results['strategy_results'] = filtered_strategy_results
            
            with open(self.results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")
    
    def start(self):
        """Start continuous backtesting"""
        if self.is_running:
            return
        
        self.is_running = True
        logger.info("🚀 Continuous backtester started")
        logger.info(f"   • Will run {self.tests_per_run} tests per strategy every {self.test_interval_hours} hours")
        logger.info(f"   • Testing on {len(self.test_symbols)} popular stocks")
        logger.info(f"   • Running first backtest now...")
        
        # Run first test immediately (in background thread)
        # Use saved preferences for strategies, but exclude 'investing' from automatic backtests
        saved_strategies = None
        if hasattr(self.app, 'preferences'):
            saved_strategies = self.app.preferences.get_backtest_strategies()
            # Filter out 'investing' strategy - it should only be tested manually
            if saved_strategies:
                saved_strategies = [s for s in saved_strategies if s != 'investing']
                if not saved_strategies:
                    # If only investing was selected, default to trading and mixed
                    saved_strategies = ['trading', 'mixed']
                logger.info(f"   • Automatic backtest strategies: {', '.join(saved_strategies)} (investing excluded from automatic tests)")
        thread = threading.Thread(target=self._run_backtest, args=(saved_strategies,))
        thread.daemon = True
        thread.start()
        
        # Schedule next test after interval
        interval_ms = self.test_interval_hours * 3600000
        self.app.root.after(interval_ms, self._schedule_next_test)
    
    def stop(self):
        """Stop continuous backtesting"""
        self.is_running = False
        self.continuous_mode = False
        self.continuous_loop_active = False
        logger.info("Continuous backtester stopped")
    
    def start_continuous_mode(self, selected_strategies: List[str] = None):
        """Start non-stop continuous backtesting until manually stopped"""
        if self.continuous_mode:
            logger.info("Continuous mode already running")
            return
        
        self.continuous_mode = True
        self.is_running = True
        self.continuous_loop_active = True
        
        logger.info("🔄 Continuous backtesting mode started")
        logger.info(f"   • Will run backtests continuously every {self.continuous_delay_seconds} seconds")
        logger.info(f"   • Anti-overfitting safeguards are active")
        logger.info(f"   • Click 'Stop Continuous Backtesting' to terminate")
        
        # Start the continuous loop in a background thread
        thread = threading.Thread(target=self._continuous_backtest_loop, args=(selected_strategies,))
        thread.daemon = True
        thread.start()
    
    def stop_continuous_mode(self):
        """Stop non-stop continuous backtesting"""
        if not self.continuous_mode:
            return
        
        self.continuous_mode = False
        self.continuous_loop_active = False
        logger.info("⏹️ Continuous backtesting mode stopped")
    
    def _continuous_backtest_loop(self, selected_strategies: List[str] = None):
        """Main loop for continuous backtesting"""
        run_count = 0
        
        while self.continuous_loop_active and self.is_running:
            run_count += 1
            logger.info(f"🔄 Continuous backtest run #{run_count}...")
            
            # Run backtest
            self._run_backtest(selected_strategies)
            
            # Wait before next run (unless stopped)
            if self.continuous_loop_active:
                # Check every second if we should continue
                for _ in range(self.continuous_delay_seconds):
                    if not self.continuous_loop_active:
                        break
                    time.sleep(1)
        
        logger.info(f"Continuous backtesting loop ended (completed {run_count} runs)")
    
    def _schedule_next_test(self):
        """Schedule next backtest"""
        if not self.is_running:
            return
        
        logger.info(f"🔄 Running scheduled backtest (every {self.test_interval_hours} hours)...")
        
        # Run test in background with saved preferences, but exclude 'investing' from automatic backtests
        saved_strategies = None
        if hasattr(self.app, 'preferences'):
            saved_strategies = self.app.preferences.get_backtest_strategies()
            # Filter out 'investing' strategy - it should only be tested manually
            if saved_strategies:
                saved_strategies = [s for s in saved_strategies if s != 'investing']
                if not saved_strategies:
                    # If only investing was selected, default to trading and mixed
                    saved_strategies = ['trading', 'mixed']
        thread = threading.Thread(target=self._run_backtest, args=(saved_strategies,))
        thread.daemon = True
        thread.start()
        
        # Schedule next test
        interval_ms = self.test_interval_hours * 3600000
        self.app.root.after(interval_ms, self._schedule_next_test)
    
    def run_test_now(self, selected_strategies: List[str] = None, num_tests: int = None):
        """Manually trigger a backtest
        
        Args:
            selected_strategies: List of strategies to test. If None, tests all strategies.
            num_tests: Number of test points per strategy. If None, uses default (100).
        """
        if self.is_testing:
            logger.info("Backtest already in progress")
            return False
        
        # Default to all strategies if none specified
        if selected_strategies is None:
            selected_strategies = ['trading', 'mixed', 'investing']
        
        thread = threading.Thread(target=self._run_backtest, args=(selected_strategies, num_tests))
        thread.daemon = True
        thread.start()
        return True
    
    def _run_backtest(self, selected_strategies: List[str] = None, num_tests: int = None):
        """Run backtest on random historical points
        
        Args:
            selected_strategies: List of strategies to test. If None, tests all strategies.
            num_tests: Number of test points per strategy. If None, uses default.
        """
        if self.is_testing:
            return
        
        self.is_testing = True
        # Set is_running to True for manual test runs (so the loop doesn't exit immediately)
        was_running = self.is_running
        self.is_running = True
        
        # Default to all strategies if none specified
        if selected_strategies is None:
            selected_strategies = ['trading', 'mixed', 'investing']
        
        # Use provided num_tests or fall back to default
        tests_to_run = num_tests if num_tests is not None else self.tests_per_run
        
        # Track dates tested in this run (reset for each new run)
        self.current_run_tested_dates = set()
        self.current_run_start_time = datetime.now()
        
        try:
            logger.info("🧪 Starting continuous backtest...")
            logger.info(f"   • Will test {tests_to_run} points per strategy")
            logger.info(f"   • Selected strategies: {', '.join(selected_strategies)}")
            
            # Track total tests across all strategies
            total_tests_before = sum(r['total'] for r in self.results['strategy_results'].values())
            
            # Track which strategies were actually tested (had at least one valid test)
            tested_strategies = []
            
            # Test each selected strategy
            for strategy in selected_strategies:
                if not self.is_running:
                    break
                
                logger.info(f"Testing {strategy} strategy...")
                
                # Get hybrid predictor for this strategy
                hybrid_predictor = self.app.hybrid_predictors.get(strategy)
                if not hybrid_predictor:
                    logger.warning(f"No hybrid predictor for {strategy}, skipping")
                    continue
                
                # Test on random symbols and dates
                tests_run = 0
                correct = 0
                incorrect = 0
                skipped = 0
                consecutive_skips = 0
                
                lookforward = self.lookforward_days[strategy]
                logger.info(f"Testing {strategy} strategy (lookforward: {lookforward} days, need dates at least {lookforward + 30} days ago)")
                
                # Try 5x more attempts to find valid points (increased from 3x for better success rate)
                max_attempts = tests_to_run * 5
                for attempt in range(max_attempts):
                    if not self.is_running:
                        break
                    
                    if tests_run >= tests_to_run:
                        break  # Got enough valid tests
                    
                    # Pick random symbol
                    symbol = random.choice(self.test_symbols)
                    
                    # Pick random date based on strategy requirements
                    # End date: at least (lookforward + buffer) days ago to ensure we have future data
                    # Increased buffer to 50 days to account for weekends/holidays and data availability
                    end_date = datetime.now() - timedelta(days=lookforward + 50)  # Buffer for weekends/holidays
                    # Start date: 3 years ago (reduced from 5 to focus on more recent, available data)
                    start_date = datetime.now() - timedelta(days=1095)  # 3 years ago
                    
                    # Ensure start_date < end_date
                    if start_date >= end_date:
                        logger.warning(f"Invalid date range for {strategy}: start={start_date.date()}, end={end_date.date()}. Lookforward={lookforward}")
                        continue
                    
                    # Random date in this range
                    days_range = (end_date - start_date).days
                    if days_range < 200:  # Need at least 200 days of range
                        logger.debug(f"Date range too small for {strategy}: {days_range} days (need at least 200)")
                        continue
                    random_days = random.randint(0, days_range)
                    test_date = start_date + timedelta(days=random_days)
                    
                    # Test this point
                    result = self._test_historical_point(
                        symbol, test_date, strategy, hybrid_predictor, tests_run
                    )
                    
                    if result:
                        tests_run += 1
                        if result['was_correct']:
                            correct += 1
                        else:
                            incorrect += 1
                        if tests_run <= 3 or tests_run % 5 == 0:  # Log first 3 and every 5th
                            logger.info(f"✅ Test {tests_run}/{tests_to_run}: {symbol} at {test_date.strftime('%Y-%m-%d')} - {'✓' if result['was_correct'] else '✗'}")
                        
                        # Reset consecutive skip counter if successful
                        consecutive_skips = 0
                            
                    else:
                        skipped += 1
                        # Track consecutive skips to increase backoff
                        consecutive_skips = locals().get('consecutive_skips', 0) + 1
                        
                        # Add delay when skipping to prevent tight looping on errors
                        # Increase delay if we are hitting many errors (likely rate limits)
                        delay = 0.5
                        if consecutive_skips > 10:
                            delay = 2.0
                        elif consecutive_skips > 5:
                            delay = 1.0
                            
                        time.sleep(delay)
                        
                        if skipped == 1 or skipped % 20 == 0:  # Log first skip and every 20th (less spam)
                            logger.info(f"Skipped {skipped} tests so far for {strategy} (attempting to find valid points...) - consecutive skips: {consecutive_skips}")
                        # After many skips, log a sample of why it's failing
                        if skipped == 50:
                            logger.warning(f"Many skips for {strategy} - checking data availability. This might indicate data fetching issues.")
                
                # Update results (only if tests were run)
                if tests_run > 0:
                    logger.info(f"Found {tests_run} valid test points for {strategy} (skipped {skipped} invalid points)")
                    # Initialize strategy_results entry if it doesn't exist
                    if strategy not in self.results['strategy_results']:
                        self.results['strategy_results'][strategy] = {
                            'total': 0, 'correct': 0, 'accuracy': 0
                        }
                    strategy_results = self.results['strategy_results'][strategy]
                    strategy_results['total'] += tests_run
                    strategy_results['correct'] += correct
                    strategy_results['accuracy'] = (
                        strategy_results['correct'] / strategy_results['total'] * 100
                        if strategy_results['total'] > 0 else 0
                    )
                    
                    # Mark this strategy as tested
                    tested_strategies.append(strategy)
                    
                    logger.info(
                        f"✅ {strategy} backtest: {correct}/{tests_run} correct "
                        f"({correct/tests_run*100:.1f}%)"
                    )
                else:
                    # Log detailed failure info
                    logger.warning(
                        f"⚠️ {strategy} backtest: No valid test points found after {max_attempts} attempts. "
                        f"This might be due to insufficient historical data or date range issues. "
                        f"Date range: {start_date.date()} to {end_date.date()}, Lookforward: {lookforward} days"
                    )
            
            # Update overall results (sum across all strategies)
            total_tests_after = sum(r['total'] for r in self.results['strategy_results'].values())
            total_correct_after = sum(r['correct'] for r in self.results['strategy_results'].values())
            total_incorrect_after = total_tests_after - total_correct_after
            
            # Update overall stats
            self.results['total_tests'] = total_tests_after
            self.results['correct_predictions'] = total_correct_after
            self.results['incorrect_predictions'] = total_incorrect_after
            
            total = self.results['correct_predictions'] + self.results['incorrect_predictions']
            if total > 0:
                overall_accuracy = self.results['correct_predictions'] / total * 100
                self.results['accuracy_history'].append({
                    'date': datetime.now().isoformat(),
                    'accuracy': overall_accuracy,
                    'total_tests': total
                })
            
            self.results['last_test_date'] = datetime.now().isoformat()
            # Only save results for strategies that were actually tested
            self.save_results(tested_strategies=tested_strategies)
            
            # Calculate summary
            total_tests_after = sum(r['total'] for r in self.results['strategy_results'].values())
            total_correct_after = sum(r['correct'] for r in self.results['strategy_results'].values())
            new_tests = total_tests_after - total_tests_before
            
            if new_tests > 0:
                overall_accuracy = (total_correct_after / total_tests_after * 100) if total_tests_after > 0 else 0
                logger.info(f"✅ Continuous backtest complete: {new_tests} new tests run")
                logger.info(f"   • Overall accuracy: {overall_accuracy:.1f}% ({total_correct_after}/{total_tests_after} correct)")
                # Only show strategies that were actually tested in this run
                strategy_details = []
                for strategy in tested_strategies:
                    s = self.results['strategy_results'][strategy]
                    if s['total'] > 0:
                        logger.info(f"   • {strategy.title()}: {s['accuracy']:.1f}% ({s['correct']}/{s['total']})")
                        strategy_details.append(f"{strategy.title()}: {s['accuracy']:.1f}% ({s['correct']}/{s['total']})")
                
                # Show notification popup
                self.app.root.after(0, self._show_backtest_complete_notification, 
                                   new_tests, overall_accuracy, strategy_details)
            else:
                logger.warning("⚠️ Continuous backtest complete: No valid test points found. Check date ranges and data availability.")
                # Show notification for no results
                self.app.root.after(0, self._show_backtest_complete_notification, 0, 0, [])
            
            # Store tested strategies for this run (for report filtering)
            self.last_tested_strategies = tested_strategies
            
            # Learn from backtest results if we have enough data
            if new_tests > 0:
                self._learn_from_backtest_results()
                
                # Calculate benchmarks
                if self.benchmarker:
                    try:
                        # Get recent backtest results for benchmarking
                        recent_results = self.results['test_details'][-100:] if len(self.results['test_details']) >= 20 else self.results['test_details']
                        
                        if len(recent_results) >= 20:
                            predictions = [{
                                'action': r.get('predicted_action', 'HOLD'),
                                'strategy': r.get('strategy', 'trading'),
                                'confidence': r.get('confidence', 50)
                            } for r in recent_results]
                            
                            outcomes = [{
                                'was_correct': r.get('was_correct', False),
                                'actual_price_change': r.get('actual_change_pct', 0)
                            } for r in recent_results]
                            
                            # Calculate baselines (pass strategy to check ML model configuration)
                            baselines = self.benchmarker.calculate_baselines(predictions, outcomes, strategy='trading')
                            
                            # Compare each strategy
                            for strategy in tested_strategies:
                                if strategy in self.results['strategy_results']:
                                    strategy_acc = self.results['strategy_results'][strategy].get('accuracy', 0)
                                    comparisons = self.benchmarker.compare_to_baselines(strategy_acc, strategy)
                                    
                                    if comparisons:
                                        vs_random = comparisons.get('vs_random', {})
                                        diff = vs_random.get('difference', 0)
                                        improvement = vs_random.get('improvement_pct', 0)
                                        # Get actual random baseline from stored baselines
                                        baselines = self.benchmarker.benchmark_data.get('baseline_performance', {})
                                        random_baseline = baselines.get('random', {}).get('accuracy', 50)
                                        logger.info(f"📊 Benchmarking ({strategy}): Model {strategy_acc:.1f}% vs Random {random_baseline:.1f}% "
                                                   f"(diff: {diff:+.1f}pp, improvement: {improvement:+.1f}%)")
                    except Exception as e:
                        logger.debug(f"Error calculating benchmarks: {e}")
            
            # Update UI
            self.app.root.after(0, self._update_ui)
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}", exc_info=True)
        finally:
            self.is_testing = False
            # Restore is_running to its original state (only if it wasn't set by start())
            # For manual runs, we set it to True temporarily, so restore it
            if not was_running:
                self.is_running = False
    
    def _test_historical_point(self, symbol: str, test_date: datetime, 
                               strategy: str, hybrid_predictor: HybridStockPredictor, 
                               test_number: int = 0) -> Optional[Dict]:
        """Test algorithm on a specific historical point
        
        Args:
            symbol: Stock symbol
            test_date: Date to test on
            strategy: Trading strategy
            hybrid_predictor: Hybrid predictor instance
            test_number: Test number (for logging purposes)
        """
        try:
            lookforward = self.lookforward_days[strategy]
            
            # Get historical data up to test_date (algorithm doesn't know future)
            end_date = test_date.strftime('%Y-%m-%d')
            # Reduced buffer to allow more test points (was 100+50, now 60+20)
            start_date = (test_date - timedelta(days=self.min_history_days + 20)).strftime('%Y-%m-%d')  # Extra buffer
            
            # Fetch historical data using data_fetcher for consistency
            hist_data = self.app.data_fetcher.fetch_stock_history(symbol, start_date=start_date, end_date=end_date)
            
            if not hist_data or 'error' in hist_data:
                error_msg = hist_data.get('error', 'Unknown error') if hist_data else 'No data returned'
                logger.debug(f"Skipping {symbol} at {test_date.date()}: Historical data error - {error_msg}")
                return None
            
            if not hist_data.get('data'):
                logger.debug(f"Skipping {symbol} at {test_date.date()}: No historical data in response")
                return None
            
            hist_list = hist_data['data']
            # Allow even less history (50 days minimum) to find more valid test points
            min_required = max(50, self.min_history_days - 10)  # At least 50 days, or 10 less than min_history_days
            if len(hist_list) < min_required:
                logger.debug(f"Skipping {symbol} at {test_date.date()}: Insufficient history ({len(hist_list)} days, need {min_required})")
                return None
            
            # Convert to DataFrame for easier manipulation
            df_hist = pd.DataFrame(hist_list)
            df_hist['date'] = pd.to_datetime(df_hist['date'])
            df_hist = df_hist.sort_values('date')
            
            # Get price at test date (last available price on or before test_date)
            df_hist_filtered = df_hist[df_hist['date'] <= pd.Timestamp(test_date)]
            if df_hist_filtered.empty:
                logger.debug(f"Skipping {symbol} at {test_date.date()}: No data up to test date (got {len(df_hist)} rows, but none <= {test_date.date()})")
                return None
            
            test_price = float(df_hist_filtered.iloc[-1]['close'])
            
            # Get future price (what actually happened - algorithm doesn't know this)
            future_date = test_date + timedelta(days=lookforward)
            future_end = min(future_date, datetime.now() - timedelta(days=1))
            
            if future_end <= test_date:
                logger.debug(f"Skipping {symbol} at {test_date.date()}: Not enough time passed for {lookforward} day lookforward (have {(future_end - test_date).days} days, need {lookforward})")
                return None  # Not enough time has passed
            
            # Get future data (start from day after test_date)
            future_start = (test_date + timedelta(days=1)).strftime('%Y-%m-%d')
            future_end_str = future_end.strftime('%Y-%m-%d')
            
            future_hist_data = self.app.data_fetcher.fetch_stock_history(symbol, start_date=future_start, end_date=future_end_str)
            
            if not future_hist_data or 'error' in future_hist_data:
                error_msg = future_hist_data.get('error', 'Unknown error') if future_hist_data else 'No data returned'
                logger.debug(f"Skipping {symbol} at {test_date.date()}: Future data error - {error_msg} (tried {future_start} to {future_end_str})")
                return None
            
            if not future_hist_data.get('data'):
                logger.debug(f"Skipping {symbol} at {test_date.date()}: No future data in response (tried {future_start} to {future_end_str})")
                return None
            
            future_list = future_hist_data['data']
            if len(future_list) < 1:
                logger.debug(f"Skipping {symbol} at {test_date.date()}: Future data list is empty (tried {future_start} to {future_end_str})")
                return None
            
            df_future = pd.DataFrame(future_list)
            df_future['date'] = pd.to_datetime(df_future['date'])
            df_future = df_future.sort_values('date')
            
            # Check if we have enough future data (at least some data points)
            if len(df_future) < 1:
                logger.debug(f"Skipping {symbol} at {test_date.date()}: Future DataFrame empty after processing")
                return None
            
            future_price = float(df_future.iloc[-1]['close'])
            actual_change_pct = ((future_price - test_price) / test_price) * 100
            
            # Convert history to format expected by analyzer (already in correct format from data_fetcher)
            history_data = {'data': hist_list}
            
            # Create stock_data dict (simulate what we'd have at test_date)
            stock_data = {
                'symbol': symbol,
                'price': test_price,
                'info': {}
            }
            
            # Get prediction using current algorithm state
            if strategy == 'trading':
                analysis = hybrid_predictor.predict(stock_data, history_data, None, is_backtest=True)
            elif strategy == 'mixed':
                # For backtesting, we might not have financials - use None
                financials_data = None
                try:
                    financials_data = self.app.data_fetcher.fetch_financials(symbol)
                except:
                    pass
                analysis = hybrid_predictor.predict(stock_data, history_data, financials_data, is_backtest=True)
            else:  # investing
                financials_data = None
                try:
                    financials_data = self.app.data_fetcher.fetch_financials(symbol)
                except:
                    pass
                analysis = hybrid_predictor.predict(stock_data, history_data, financials_data, is_backtest=True)
            
            if 'error' in analysis:
                return None
            
            recommendation = analysis.get('recommendation', {})
            predicted_action = recommendation.get('action', 'HOLD')
            predicted_target = recommendation.get('target_price', test_price)
            
            # Diagnostic logging for backtesting
            method = analysis.get('method', 'unknown')
            ml_pred = analysis.get('ml_prediction', {})
            rule_pred = analysis.get('rule_prediction', {})
            ensemble_weights = analysis.get('ensemble_weights', {})
            
            # Track ML vs final action for first 20 tests to diagnose accuracy gap
            ml_action_raw = ml_pred.get('action', 'N/A')
            rule_action = rule_pred.get('action', 'N/A')
            ml_conf = ml_pred.get('confidence', 0)
            rule_conf = rule_pred.get('confidence', 0)
            
            # Log detailed info for first 20 tests
            if test_number < 20:
                logger.info(f"Backtest #{test_number} ({symbol} @ {test_date.date()}): "
                          f"FINAL={predicted_action}, ML={ml_action_raw} ({ml_conf:.1f}%), "
                          f"RULE={rule_action} ({rule_conf:.1f}%), "
                          f"weights=ML:{ensemble_weights.get('ml', 0):.2f}/Rule:{ensemble_weights.get('rule', 0):.2f}, "
                          f"price_change={actual_change_pct:.2f}%")
            else:
                logger.debug(f"Backtest prediction ({symbol} @ {test_date.date()}): "
                            f"action={predicted_action}, method={method}, "
                            f"ml={ml_action_raw} ({ml_conf:.1f}%), "
                            f"rule={rule_action} ({rule_conf:.1f}%), "
                            f"weights=ML:{ensemble_weights.get('ml', 0):.2f}/Rule:{ensemble_weights.get('rule', 0):.2f}, "
                            f"price_change={actual_change_pct:.2f}%")
            
            # Determine if prediction was correct
            # STANDARD LOGIC (Fixed):
            # - BUY = Bullish (expecting price to go UP)
            # - SELL = Bearish (expecting price to go DOWN)
            was_correct = None
            if predicted_action == 'BUY':
                # BUY is correct if price went UP
                was_correct = actual_change_pct > 0
                # Detailed logging for BUY predictions (first 20)
                if test_number < 20:
                    logger.debug(f"BUY prediction evaluation: price_change={actual_change_pct:.2f}%, "
                               f"correct={was_correct}, ml_action={ml_pred.get('action', 'N/A')}, "
                               f"rule_action={rule_pred.get('action', 'N/A')}")
            elif predicted_action == 'SELL':
                # SELL is correct if price went DOWN (avoided loss)
                was_correct = actual_change_pct < 0
                # Detailed logging for SELL predictions
                if test_number < 20:
                    logger.info(f"SELL prediction evaluation: price_change={actual_change_pct:.2f}%, "
                              f"correct={was_correct}, ml_action={ml_pred.get('action', 'N/A')}, "
                              f"rule_action={rule_pred.get('action', 'N/A')}, "
                              f"ml_conf={ml_pred.get('confidence', 0):.1f}%, "
                              f"rule_conf={rule_pred.get('confidence', 0):.1f}%")
            elif predicted_action == 'HOLD':
                # HOLD is correct if price didn't move much (strategy-specific thresholds)
                if strategy == "trading":
                    was_correct = abs(actual_change_pct) < 3
                elif strategy == "mixed":
                    was_correct = abs(actual_change_pct) < 5
                else:  # investing
                    # For investing (1.5 year lookforward), allow up to 20% movement for HOLD
                    # Over 1.5 years, stocks typically move 15-25%, so 20% is reasonable
                    was_correct = abs(actual_change_pct) < 20
            elif predicted_action == 'AVOID':
                # AVOID (legacy, now mapped to BUY in inverted logic) is correct if price went DOWN
                was_correct = actual_change_pct < 0
            else:
                return None  # Unknown action
            
            # Store test result
            test_result = {
                'symbol': symbol,
                'test_date': test_date.isoformat(),
                'strategy': strategy,
                'test_price': test_price,
                'future_price': future_price,
                'actual_change_pct': actual_change_pct,
                'predicted_action': predicted_action,
                'predicted_target': predicted_target,
                'was_correct': was_correct,
                'confidence': recommendation.get('confidence', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            self.results['test_details'].append(test_result)
            
            # ANTI-OVERFITTING: Mark this date as tested
            # Track in current run (for same-run training) and permanent (for future runs)
            date_str = test_date.strftime('%Y-%m-%d')
            self.current_run_tested_dates.add((symbol.upper(), date_str))
            self._mark_date_as_tested(symbol, test_date)
            
            return test_result
            
        except Exception as e:
            logger.debug(f"Error testing {symbol} at {test_date}: {e}")
            return None
    
    def _update_ui(self):
        """Update UI with backtest results"""
        try:
            if hasattr(self.app, '_update_ml_status'):
                self.app.root.after(0, self.app._update_ml_status)
        except:
            pass
    
    def _show_backtest_complete_notification(self, new_tests: int, overall_accuracy: float, strategy_details: List[str]):
        """Show notification popup when backtesting completes"""
        try:
            from tkinter import messagebox
            
            # Check notification preferences
            if self.app and hasattr(self.app, 'preferences'):
                show_notifications = self.app.preferences.get('show_backtest_complete_notifications', True)
                if not show_notifications:
                    logger.debug("Backtest complete notifications disabled by user preference")
                    return
            
            if new_tests > 0:
                message = f"✅ Backtesting Complete!\n\n"
                message += f"Tests Run: {new_tests}\n"
                message += f"Overall Accuracy: {overall_accuracy:.1f}%\n\n"
                
                if strategy_details:
                    message += "Strategy Results:\n"
                    for detail in strategy_details:
                        message += f"  • {detail}\n"
                
                messagebox.showinfo("Backtesting Complete", message)
            else:
                messagebox.showwarning(
                    "Backtesting Complete",
                    "⚠️ No valid test points found.\n\n"
                    "This might be due to:\n"
                    "• Insufficient historical data\n"
                    "• Date range issues\n"
                    "• Data availability problems\n\n"
                    "Check the logs for more details."
                )
        except Exception as e:
            logger.debug(f"Error showing backtest notification: {e}")
    
    def get_statistics(self) -> Dict:
        """Get backtest statistics"""
        total = self.results['correct_predictions'] + self.results['incorrect_predictions']
        accuracy = (self.results['correct_predictions'] / total * 100) if total > 0 else 0
        
        return {
            'total_tests': total,
            'correct': self.results['correct_predictions'],
            'incorrect': self.results['incorrect_predictions'],
            'overall_accuracy': accuracy,
            'strategy_results': self.results['strategy_results'],
            'last_test_date': self.results.get('last_test_date'),
            'recent_accuracy': (
                self.results['accuracy_history'][-1]['accuracy']
                if self.results['accuracy_history'] else 0
            )
        }
    
    def get_recent_results(self, limit: int = 20) -> List[Dict]:
        """Get recent backtest results"""
        return self.results['test_details'][-limit:]
    
    def _learn_from_backtest_results(self):
        """Analyze backtest results and use them to improve the algorithm"""
        try:
            # Get recent backtest results (last 100 for better learning)
            recent_results = self.results['test_details'][-100:]
            
            if len(recent_results) < 20:
                logger.debug("Not enough backtest results to learn from yet")
                return
            
            # Group by strategy
            strategy_results = {}
            for result in recent_results:
                strategy = result.get('strategy')
                if strategy not in strategy_results:
                    strategy_results[strategy] = []
                strategy_results[strategy].append(result)
            
            # Process each strategy
            for strategy, results in strategy_results.items():
                if len(results) < 10:
                    continue
                
                # Calculate accuracy
                correct = sum(1 for r in results if r.get('was_correct'))
                accuracy = correct / len(results) * 100
                
                # Get previous accuracy for comparison
                previous_accuracy = 0
                if 'insights' in self.results and strategy in self.results['insights']:
                    previous_accuracy = self.results['insights'][strategy].get('recent_accuracy', 0)
                
                # 1. ADJUST WEIGHTS BASED ON BACKTEST FAILURES
                incorrect_results = [r for r in results if not r.get('was_correct')]
                if len(incorrect_results) >= 5:
                    self._adjust_weights_from_backtest_failures(strategy, incorrect_results)
                
                # 2. CONVERT BACKTEST RESULTS TO TRAINING SAMPLES AND RETRAIN
                if len(results) >= 20:
                    self._use_backtest_results_as_training_data(strategy, results)
                
                # 3. AUTO-RETRAIN IF ACCURACY DROPS
                accuracy_drop = previous_accuracy - accuracy if previous_accuracy > 0 else 0
                if accuracy_drop > 15:  # Accuracy dropped by more than 15%
                    logger.warning(
                        f"⚠️ Accuracy Drop Detected ({strategy}): "
                        f"{previous_accuracy:.1f}% → {accuracy:.1f}% (drop: {accuracy_drop:.1f}%)"
                    )
                    self._trigger_auto_retrain(strategy, results)
                
                # Log accuracy status
                if accuracy < 40:
                    logger.warning(
                        f"⚠️ Backtest Alert ({strategy}): Low accuracy ({accuracy:.1f}%). "
                        f"Auto-retraining triggered."
                    )
                    self._trigger_auto_retrain(strategy, results)
                elif accuracy > 60:
                    logger.info(
                        f"✅ Backtest Success ({strategy}): Good accuracy ({accuracy:.1f}%)!"
                    )
                
                # Store insights
                if 'insights' not in self.results:
                    self.results['insights'] = {}
                
                self.results['insights'][strategy] = {
                    'recent_accuracy': accuracy,
                    'total_tests': len(results),
                    'correct': correct,
                    'previous_accuracy': previous_accuracy,
                    'accuracy_drop': accuracy_drop,
                    'last_updated': datetime.now().isoformat()
                }
            
            # Preserve only strategies that exist in strategy_results (were actually tested)
            tested_strategies = list(self.results['strategy_results'].keys())
            self.save_results(tested_strategies=tested_strategies)
            
            # ANTI-OVERFITTING: Save tested dates
            self._save_tested_dates()
            
            # Clear current run tracking (dates are now in permanent storage)
            self.current_run_tested_dates = set()
            self.current_run_start_time = None
            
        except Exception as e:
            logger.error(f"Error learning from backtest results: {e}", exc_info=True)
    
    def _adjust_weights_from_backtest_failures(self, strategy: str, failures: List[Dict]):
        """Adjust rule-based weights based on backtest failures"""
        try:
            from adaptive_weight_adjuster import AdaptiveWeightAdjuster
            
            weight_adjuster = AdaptiveWeightAdjuster(self.data_dir, strategy)
            
            # Analyze failure patterns
            action_failures = {}
            confidence_failures = []
            
            for failure in failures:
                action = failure.get('predicted_action', 'UNKNOWN')
                action_failures[action] = action_failures.get(action, 0) + 1
                confidence = failure.get('confidence', 50)
                confidence_failures.append(confidence)
            
            # If we have enough failures, make general adjustments
            if len(failures) >= 10:
                # Check if high-confidence predictions are failing (more concerning)
                avg_failed_confidence = np.mean(confidence_failures) if confidence_failures else 50
                
                if avg_failed_confidence > 70:
                    # Log at debug level - hybrid_predictor now handles weight adjustment automatically
                    logger.debug(
                        f"High-confidence failures detected ({strategy}): "
                        f"Avg confidence {avg_failed_confidence:.1f}% - "
                        f"Hybrid predictor will auto-adjust weights"
                    )
                
                # Log action failure patterns at debug level (informational only)
                if action_failures:
                    most_failed = max(action_failures.items(), key=lambda x: x[1])
                    failure_rate = most_failed[1] / len(failures) * 100
                    if failure_rate > 60:
                        # Changed to debug level - this is normal during ML learning phase
                        logger.debug(
                            f"Action pattern ({strategy}): "
                            f"{most_failed[0]} predictions failing {failure_rate:.0f}% of time "
                            f"- hybrid predictor auto-adjusting"
                        )
            
            # Note: Full weight adjustment requires indicator data which we don't store
            # This is a simplified analysis. For full adjustment, we'd need to:
            # 1. Store indicators in backtest results, OR
            # 2. Re-fetch and recalculate indicators when adjusting weights
            
        except Exception as e:
            logger.debug(f"Error adjusting weights from backtest: {e}")
    
    def _use_backtest_results_as_training_data(self, strategy: str, results: List[Dict]):
        """Convert backtest results to training samples and retrain model
        
        ANTI-OVERFITTING SAFEGUARDS:
        - Only uses dates that haven't been tested recently (prevents data leakage)
        - Checks cooldown period between retrains
        - Limits overlap with tested dates to prevent memorization
        """
        try:
            if not hasattr(self.app, 'training_manager'):
                return
            
            # SAFEGUARD 1: Check cooldown period
            last_retrain = self.last_retrain_time.get(strategy)
            if last_retrain:
                try:
                    last_retrain_dt = datetime.fromisoformat(last_retrain)
                    hours_since_retrain = (datetime.now() - last_retrain_dt).total_seconds() / 3600
                    if hours_since_retrain < self.retrain_cooldown_hours:
                        logger.info(
                            f"⏸️ Retrain cooldown active ({strategy}): "
                            f"{hours_since_retrain:.1f}h < {self.retrain_cooldown_hours}h. "
                            f"Skipping retrain to prevent overfitting."
                        )
                        return
                except Exception as e:
                    logger.debug(f"Error checking retrain cooldown: {e}")
            
            # SAFEGUARD 2: Filter out recently tested dates (prevent data leakage)
            # Only use results from dates that haven't been tested recently
            fresh_results = []
            skipped_recent = 0
            
            for result in results:
                symbol = result.get('symbol', '').upper()
                test_date_str = result.get('test_date')
                
                if symbol and test_date_str:
                    try:
                        test_date = datetime.fromisoformat(test_date_str)
                        # Skip if this date was tested recently (within recent_test_window_days)
                        # BUT allow dates from current run (allow_current_run=True)
                        if self._is_date_recently_tested(symbol, test_date, allow_current_run=True):
                            skipped_recent += 1
                            continue
                    except Exception as e:
                        logger.debug(f"Error checking if date recently tested: {e}")
                
                fresh_results.append(result)
            
            if skipped_recent > 0:
                logger.info(
                    f"🛡️ Anti-overfitting: Skipped {skipped_recent}/{len(results)} results "
                    f"from recently tested dates ({strategy})"
                )
            
            # Need at least 20 fresh results to retrain
            if len(fresh_results) < 20:
                logger.info(
                    f"⏸️ Insufficient fresh results for retraining ({strategy}): "
                    f"{len(fresh_results)} fresh (need 20). "
                    f"Skipped {skipped_recent} recently tested dates."
                )
                return
            
            # Convert fresh backtest results to training samples
            training_samples = []
            
            for result in fresh_results:
                try:
                    symbol = result.get('symbol', '').upper()
                    test_date_str = result.get('test_date')
                    test_price = result.get('test_price', 0)
                    future_price = result.get('future_price', 0)
                    was_correct = result.get('was_correct', False)
                    
                    if not symbol or not test_date_str or test_price <= 0 or future_price <= 0:
                        continue
                    
                    test_date = datetime.fromisoformat(test_date_str)
                    
                    # Calculate price change
                    price_change_pct = ((future_price - test_price) / test_price) * 100
                    
                    # Corrected Logic: Trend Following
                    # Future Price Higher → BUY (Signal to Enter/Long)
                    # Future Price Lower → SELL (Signal to Exit/Short)
                    if strategy == "trading":
                        if price_change_pct >= 3:
                            label = "BUY"
                        elif price_change_pct <= -3:
                            label = "SELL"
                        else:
                            label = "HOLD"
                    elif strategy == "investing":
                        if price_change_pct >= 10:
                            label = "BUY"
                        elif price_change_pct <= -10:
                            label = "SELL"
                        else:
                            label = "HOLD"
                    else:  # mixed
                        if price_change_pct >= 5:
                            label = "BUY"
                        elif price_change_pct <= -5:
                            label = "SELL"
                        else:
                            label = "HOLD"
                    
                    # Fetch historical data at test date using data_fetcher
                    start_date = (test_date - timedelta(days=100)).strftime('%Y-%m-%d')
                    end_date = test_date.strftime('%Y-%m-%d')
                    
                    hist_data = self.app.data_fetcher.fetch_stock_history(symbol, start_date=start_date, end_date=end_date)
                    
                    if not hist_data or 'error' in hist_data or not hist_data.get('data') or len(hist_data['data']) < 20:
                        continue
                    
                    # Convert to format expected by feature extractor (already in correct format)
                    history_data = {'data': hist_data['data']}
                    
                    stock_data = {
                        'symbol': symbol,
                        'price': test_price,
                        'info': {}
                    }
                    
                    # Extract features
                    from ml_training import FeatureExtractor
                    feature_extractor = FeatureExtractor()
                    
                    # Calculate indicators
                    indicators = {}
                    try:
                        from trading_analyzer import TradingAnalyzer
                        temp_analyzer = TradingAnalyzer()
                        temp_analysis = temp_analyzer.analyze(stock_data, history_data)
                        if 'error' not in temp_analysis:
                            indicators = temp_analysis.get('indicators', {})
                    except:
                        pass
                    
                    # Extract features
                    features = feature_extractor.extract_features(
                        stock_data, history_data, None, indicators if indicators else None
                    )
                    
                    if features is None or len(features) == 0:
                        continue
                    
                    # Convert to numpy array and handle invalid values
                    if not isinstance(features, np.ndarray):
                        features = np.array(features)
                    
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    training_samples.append({
                        'features': features.tolist(),
                        'label': label,
                        'price_change_pct': price_change_pct,
                        'date': test_date.strftime('%Y-%m-%d'),
                        'symbol': symbol,
                        'was_correct': was_correct
                    })
                    
                except Exception as e:
                    logger.debug(f"Error converting backtest result to training sample: {e}")
                    continue
            
            # SAFEGUARD 3: Check overlap with tested dates
            overlap_ratio = self._calculate_overlap_with_tested_dates(training_samples)
            if overlap_ratio > self.max_overlap_threshold:
                logger.warning(
                    f"🛡️ Anti-overfitting: Overlap too high ({strategy}): "
                    f"{overlap_ratio*100:.1f}% > {self.max_overlap_threshold*100:.1f}%. "
                    f"Skipping retrain to prevent memorization."
                )
                return
            
            # Log distribution of labels for diagnostics
            if len(training_samples) > 0:
                label_counts = {}
                for sample in training_samples:
                    label = sample.get('label', 'UNKNOWN')
                    label_counts[label] = label_counts.get(label, 0) + 1
                logger.info(
                    f"Backtest training samples ({strategy}): {len(training_samples)} samples, "
                    f"overlap: {overlap_ratio*100:.1f}%, distribution: {label_counts}"
                )
            
            # If we have samples, try to combine with verified predictions to meet minimum requirement
            # Reduced minimum to 30 to allow more frequent retraining from backtest results
            if len(training_samples) >= 30:
                logger.info(
                    f"📚 Converting {len(training_samples)} backtest results to training data for {strategy}"
                )
                
                # Try to combine with verified predictions if we don't have enough samples
                combined_samples = training_samples.copy()
                if len(training_samples) < 100:
                    logger.debug(f"Only {len(training_samples)} backtest samples, attempting to combine with verified predictions...")
                    
                    # Get verified predictions for this strategy
                    if hasattr(self.app, 'predictions_tracker'):
                        try:
                            from training_pipeline import MLTrainingPipeline
                            pipeline = MLTrainingPipeline(self.app.data_fetcher, self.data_dir)
                            verified_samples = pipeline._convert_verified_to_training_samples(
                                self.app.predictions_tracker.get_verified_predictions(),
                                strategy
                            )
                            
                            if verified_samples:
                                # Combine samples (prefer backtest results, but add verified if needed)
                                needed = 100 - len(training_samples)
                                verified_to_add = verified_samples[-needed:] if len(verified_samples) > needed else verified_samples
                                combined_samples.extend(verified_to_add)
                                logger.info(
                                    f"   • Combined with {len(verified_to_add)} verified predictions "
                                    f"({len(combined_samples)} total samples)"
                                )
                        except Exception as e:
                            logger.debug(f"Could not combine with verified predictions: {e}")
                
                # Only attempt training if we have enough samples
                if len(combined_samples) >= 100:
                    # SAFEGUARD 4: Final overlap check on combined samples
                    combined_overlap = self._calculate_overlap_with_tested_dates(combined_samples)
                    if combined_overlap > self.max_overlap_threshold:
                        logger.warning(
                            f"🛡️ Anti-overfitting: Combined overlap too high ({strategy}): "
                            f"{combined_overlap*100:.1f}% > {self.max_overlap_threshold*100:.1f}%. "
                            f"Skipping retrain."
                        )
                        return
                    
                    # SAFEGUARD 5: Update retrain timestamp before training
                    # (This prevents multiple simultaneous retrains)
                    self.last_retrain_time[strategy] = datetime.now().isoformat()
                    
                    # Use training pipeline to retrain with combined data
                    from ml_training import StockPredictionML
                    ml_model = StockPredictionML(self.data_dir, strategy)
                    
                    if ml_model.is_trained():
                        # Retrain with combined data using binary classification (same as original training)
                        # Check if original model was trained with binary classification
                        use_binary = False
                        if ml_model.metadata and ml_model.metadata.get('use_binary_classification', False):
                            use_binary = True
                        
                        result = ml_model.train(
                            combined_samples,
                            use_binary_classification=use_binary,  # Use same mode as original
                            model_family='voting'  # Use voting ensemble for consistency
                        )
                        
                        if 'error' not in result:
                            test_acc = result.get('test_accuracy', 0) * 100
                            train_acc = result.get('train_accuracy', 0) * 100
                            logger.info(
                                f"✅ Model retrained on combined data ({strategy}): "
                                f"Train: {train_acc:.1f}%, Test: {test_acc:.1f}% "
                                f"({len(training_samples)} backtest + {len(combined_samples) - len(training_samples)} verified)"
                            )
                        else:
                            logger.warning(f"⚠️ Retraining failed: {result.get('error')}")
                    else:
                        logger.debug(f"Model not trained yet for {strategy}, skipping retrain")
                else:
                    logger.debug(
                        f"⚠️ Insufficient samples for retraining ({strategy}): "
                        f"{len(combined_samples)} samples (need 100). "
                        f"Will retry when more backtest results or verified predictions are available."
                    )
            
        except Exception as e:
            logger.error(f"Error using backtest results as training data: {e}", exc_info=True)
    
    def _trigger_auto_retrain(self, strategy: str, results: List[Dict]):
        """Automatically retrain model when accuracy drops (with anti-overfitting safeguards)"""
        try:
            # SAFEGUARD: Check cooldown period (same as in _use_backtest_results_as_training_data)
            last_retrain = self.last_retrain_time.get(strategy)
            if last_retrain:
                try:
                    last_retrain_dt = datetime.fromisoformat(last_retrain) if isinstance(last_retrain, str) else last_retrain
                    hours_since = (datetime.now() - last_retrain_dt).total_seconds() / 3600
                    if hours_since < self.retrain_cooldown_hours:
                        logger.info(
                            f"⏸️ Auto-retrain cooldown active ({strategy}): "
                            f"{hours_since:.1f}h < {self.retrain_cooldown_hours}h. "
                            f"Skipping to prevent overfitting."
                        )
                        return
                except Exception as e:
                    logger.debug(f"Error checking retrain cooldown: {e}")
            
            if not hasattr(self.app, 'training_manager'):
                return
            
            logger.info(f"🔄 Auto-retraining {strategy} model due to accuracy drop...")
            
            # Use backtest results as training data
            self._use_backtest_results_as_training_data(strategy, results)
            
            # Also trigger standard retraining from verified predictions
            if hasattr(self.app, 'auto_training'):
                # Check if we have verified predictions to use
                verified = self.app.predictions_tracker.get_verified_predictions()
                strategy_verified = [p for p in verified if p.get('strategy') == strategy]
                
                if len(strategy_verified) >= 50:
                    # Use training pipeline to retrain
                    from training_pipeline import MLTrainingPipeline
                    pipeline = MLTrainingPipeline(self.app.data_fetcher, self.data_dir)
                    result = pipeline.train_on_verified_predictions(
                        self.app.predictions_tracker,
                        strategy,
                        retrain_ml=True
                    )
                    
                    if result.get('ml_retrained'):
                        logger.info(
                            f"✅ Auto-retraining complete ({strategy}): "
                            f"Combined backtest + verified predictions"
                        )
            
            # Update retrain timestamp (use ISO format for consistency)
            self.last_retrain_time[strategy] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"Error in auto-retrain: {e}", exc_info=True)

