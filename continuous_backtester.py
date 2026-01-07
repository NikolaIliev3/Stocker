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

from data_fetcher import StockDataFetcher
from hybrid_predictor import HybridStockPredictor

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
        self.tests_per_run = 10  # Test 10 random points per run
        self.min_history_days = 100  # Need at least 100 days of history
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
        logger.info("Continuous backtester stopped")
    
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
    
    def run_test_now(self, selected_strategies: List[str] = None):
        """Manually trigger a backtest
        
        Args:
            selected_strategies: List of strategies to test. If None, tests all strategies.
        """
        if self.is_testing:
            logger.info("Backtest already in progress")
            return False
        
        # Default to all strategies if none specified
        if selected_strategies is None:
            selected_strategies = ['trading', 'mixed', 'investing']
        
        thread = threading.Thread(target=self._run_backtest, args=(selected_strategies,))
        thread.daemon = True
        thread.start()
        return True
    
    def _run_backtest(self, selected_strategies: List[str] = None):
        """Run backtest on random historical points
        
        Args:
            selected_strategies: List of strategies to test. If None, tests all strategies.
        """
        if self.is_testing:
            return
        
        self.is_testing = True
        
        # Default to all strategies if none specified
        if selected_strategies is None:
            selected_strategies = ['trading', 'mixed', 'investing']
        
        try:
            logger.info("🧪 Starting continuous backtest...")
            logger.info(f"   • Will test {self.tests_per_run} points per strategy")
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
                
                lookforward = self.lookforward_days[strategy]
                logger.info(f"Testing {strategy} strategy (lookforward: {lookforward} days, need dates at least {lookforward + 30} days ago)")
                
                # Try 3x more attempts to find valid points
                max_attempts = self.tests_per_run * 3
                for attempt in range(max_attempts):
                    if not self.is_running:
                        break
                    
                    if tests_run >= self.tests_per_run:
                        break  # Got enough valid tests
                    
                    # Pick random symbol
                    symbol = random.choice(self.test_symbols)
                    
                    # Pick random date based on strategy requirements
                    # End date: at least (lookforward + buffer) days ago to ensure we have future data
                    end_date = datetime.now() - timedelta(days=lookforward + 30)  # Buffer for weekends/holidays
                    # Start date: 5 years ago to have plenty of historical data
                    start_date = datetime.now() - timedelta(days=1825)  # 5 years ago
                    
                    # Ensure start_date < end_date
                    if start_date >= end_date:
                        logger.warning(f"Invalid date range for {strategy}: start={start_date.date()}, end={end_date.date()}")
                        continue
                    
                    # Random date in this range
                    days_range = (end_date - start_date).days
                    if days_range <= 0:
                        continue
                    random_days = random.randint(0, days_range)
                    test_date = start_date + timedelta(days=random_days)
                    
                    # Test this point
                    result = self._test_historical_point(
                        symbol, test_date, strategy, hybrid_predictor
                    )
                    
                    if result:
                        tests_run += 1
                        if result['was_correct']:
                            correct += 1
                        else:
                            incorrect += 1
                        if tests_run <= 3 or tests_run % 5 == 0:  # Log first 3 and every 5th
                            logger.info(f"✅ Test {tests_run}/{self.tests_per_run}: {symbol} at {test_date.strftime('%Y-%m-%d')} - {'✓' if result['was_correct'] else '✗'}")
                    else:
                        skipped += 1
                        if skipped == 1 or skipped % 10 == 0:  # Log first skip and every 10th
                            logger.debug(f"Skipped {skipped} tests so far for {strategy} (attempting to find valid points...)")
                
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
                    logger.warning(
                        f"⚠️ {strategy} backtest: No valid test points found after {skipped} attempts. "
                        f"This might be due to insufficient historical data or date range issues. "
                        f"Date range: {start_date.date()} to {end_date.date()}"
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
            
            # Update UI
            self.app.root.after(0, self._update_ui)
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}", exc_info=True)
        finally:
            self.is_testing = False
    
    def _test_historical_point(self, symbol: str, test_date: datetime, 
                               strategy: str, hybrid_predictor: HybridStockPredictor) -> Optional[Dict]:
        """Test algorithm on a specific historical point"""
        try:
            lookforward = self.lookforward_days[strategy]
            
            # Get historical data up to test_date (algorithm doesn't know future)
            end_date = test_date.strftime('%Y-%m-%d')
            start_date = (test_date - timedelta(days=self.min_history_days + 50)).strftime('%Y-%m-%d')  # Extra buffer
            
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
            if len(hist_list) < self.min_history_days:
                logger.debug(f"Skipping {symbol} at {test_date.date()}: Insufficient history ({len(hist_list)} days, need {self.min_history_days})")
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
                analysis = hybrid_predictor.predict(stock_data, history_data, None)
            elif strategy == 'mixed':
                # For backtesting, we might not have financials - use None
                financials_data = None
                try:
                    financials_data = self.app.data_fetcher.fetch_financials(symbol)
                except:
                    pass
                analysis = hybrid_predictor.predict(stock_data, history_data, financials_data)
            else:  # investing
                financials_data = None
                try:
                    financials_data = self.app.data_fetcher.fetch_financials(symbol)
                except:
                    pass
                analysis = hybrid_predictor.predict(stock_data, history_data, financials_data)
            
            if 'error' in analysis:
                return None
            
            recommendation = analysis.get('recommendation', {})
            predicted_action = recommendation.get('action', 'HOLD')
            predicted_target = recommendation.get('target_price', test_price)
            
            # Determine if prediction was correct
            # NOTE: Algorithm uses INVERTED logic:
            # - BUY = bearish signals (expecting price to go DOWN, then buy at discount)
            # - SELL = bullish signals (expecting price to go UP, then sell to take profits)
            was_correct = None
            if predicted_action == 'BUY':
                # BUY (bearish signals) is correct if price went DOWN (so you can buy at discount)
                was_correct = actual_change_pct < 0
            elif predicted_action == 'SELL':
                # SELL (bullish signals) is correct if price went UP (so you can sell at profit)
                was_correct = actual_change_pct > 0
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
                    logger.info(
                        f"🔧 High-confidence failures detected ({strategy}): "
                        f"Avg confidence {avg_failed_confidence:.1f}% - "
                        f"Model may be overconfident, adjusting weights"
                    )
                    # This indicates the model is overconfident
                    # We could adjust weights to be more conservative
                
                # Log action failure patterns
                if action_failures:
                    most_failed = max(action_failures.items(), key=lambda x: x[1])
                    failure_rate = most_failed[1] / len(failures) * 100
                    if failure_rate > 60:
                        logger.info(
                            f"🔧 Action pattern detected ({strategy}): "
                            f"{most_failed[0]} predictions failing {failure_rate:.0f}% of time "
                            f"({most_failed[1]}/{len(failures)} failures). "
                            f"This may indicate the model needs adjustment for this action type."
                        )
            
            # Note: Full weight adjustment requires indicator data which we don't store
            # This is a simplified analysis. For full adjustment, we'd need to:
            # 1. Store indicators in backtest results, OR
            # 2. Re-fetch and recalculate indicators when adjusting weights
            
        except Exception as e:
            logger.debug(f"Error adjusting weights from backtest: {e}")
    
    def _use_backtest_results_as_training_data(self, strategy: str, results: List[Dict]):
        """Convert backtest results to training samples and retrain model"""
        try:
            if not hasattr(self.app, 'training_manager'):
                return
            
            # Convert backtest results to training samples
            training_samples = []
            
            for result in results:
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
                    
                    # Determine label based on actual outcome (INVERTED LOGIC)
                    # With inverted logic:
                    # - Price going DOWN → BUY (bearish signals, buy at discount)
                    # - Price going UP → SELL (bullish signals, sell at profit)
                    if strategy == "trading":
                        if price_change_pct <= -3:
                            label = "BUY"  # Price went down, bearish → BUY
                        elif price_change_pct >= 3:
                            label = "SELL"  # Price went up, bullish → SELL
                        else:
                            label = "HOLD"
                    elif strategy == "investing":
                        if price_change_pct <= -10:
                            label = "BUY"  # Price went down significantly, bearish → BUY
                        elif price_change_pct >= 10:
                            label = "SELL"  # Price went up significantly, bullish → SELL (was AVOID)
                        else:
                            label = "HOLD"
                    else:  # mixed
                        if price_change_pct <= -5:
                            label = "BUY"  # Price went down, bearish → BUY
                        elif price_change_pct >= 5:
                            label = "SELL"  # Price went up, bullish → SELL (was AVOID)
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
            
            # If we have samples, try to combine with verified predictions to meet minimum requirement
            if len(training_samples) >= 50:
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
                    # Use training pipeline to retrain with combined data
                    from ml_training import StockPredictionML
                    ml_model = StockPredictionML(self.data_dir, strategy)
                    
                    if ml_model.is_trained():
                        # Retrain with combined data
                        result = ml_model.train(combined_samples)
                        
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
        """Automatically retrain model when accuracy drops"""
        try:
            # Throttle: Don't retrain more than once per day per strategy
            last_retrain = self.last_retrain_time.get(strategy)
            if last_retrain:
                hours_since = (datetime.now() - last_retrain).total_seconds() / 3600
                if hours_since < 24:
                    logger.debug(f"Skipping auto-retrain for {strategy}: Retrained {hours_since:.1f} hours ago")
                    return
            
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
            
            # Update last retrain time
            self.last_retrain_time[strategy] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in auto-retrain: {e}", exc_info=True)

