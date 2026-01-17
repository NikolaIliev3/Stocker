"""
Main GUI Application for Stocker - Stock Trading/Investing Desktop App
Enhanced with themes, animations, and predictions tracking
"""
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import time
import subprocess
import sys
import socket
import os
import matplotlib  # pyright: ignore[reportMissingImports]
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # pyright: ignore[reportMissingImports]
from matplotlib.figure import Figure  # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

from config import APP_DATA_DIR, INITIAL_BALANCE
from data_fetcher import StockDataFetcher
from trading_analyzer import TradingAnalyzer
from investing_analyzer import InvestingAnalyzer
from mixed_analyzer import MixedAnalyzer
from chart_generator import ChartGenerator
from portfolio import Portfolio
from predictions_tracker import PredictionsTracker
from model_benchmarking import ModelBenchmarker
from ui_themes import ThemeManager
from modern_ui import ModernCard, ModernButton, GradientLabel, ModernScrollbar
from localization import Localization
from user_preferences import UserPreferences
from loading_screen import LoadingScreen
from modern_settings import ModernDropdownButton
from market_scanner import MarketScanner
from hybrid_predictor import HybridStockPredictor
from training_pipeline import MLTrainingPipeline, AutoTrainingManager
from model_tester import ModelTester
from auto_learner import AutoLearner
from learning_tracker import LearningTracker
from continuous_backtester import ContinuousBacktester
from stock_monitor import StockMonitor
from momentum_monitor import MomentumMonitor
from trend_change_predictor import TrendChangePredictor
from ai_researcher import SeekerAI
from config import SEEKER_AI_ENABLED
from scrollable_notebook import TwoRowNotebook
from llm_ai_predictor import LLMAIPredictor
from price_move_predictor import PriceMovePredictor
from holdings_tracker import HoldingsTracker
from potentials_tracker import PotentialsTracker

# Import new modules
try:
    from alert_system import AlertSystem
    from error_handler import ErrorHandler, safe_call
    from walk_forward_backtester import WalkForwardBacktester
    HAS_NEW_MODULES = True
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"New modules not available: {e}")
    HAS_NEW_MODULES = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AnimatedButton(tk.Button):
    """Button with hover animation effects"""
    def __init__(self, parent, *args, **kwargs):
        self.original_bg = kwargs.get('bg', '#2196F3')
        self.hover_bg = kwargs.get('hover_bg', '#1976D2')
        if 'hover_bg' in kwargs:
            del kwargs['hover_bg']
        super().__init__(parent, *args, **kwargs, 
                       relief=tk.RAISED, bd=2, cursor='hand2',
                       activebackground=self.hover_bg)
        self.bind('<Enter>', self._on_enter)
        self.bind('<Leave>', self._on_leave)
    
    def _on_enter(self, event):
        self.config(bg=self.hover_bg)
    
    def _on_leave(self, event):
        self.config(bg=self.original_bg)


class StockerApp:
    """Main application class with enhanced UI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Stocker - Stock Trading & Investing App")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 700)
        
        # Set window icon for taskbar
        icon_path = Path(__file__).parent / "stocker_icon.ico"
        if icon_path.exists():
            try:
                self.root.iconbitmap(str(icon_path))
            except Exception as e:
                logger.debug(f"Could not set window icon: {e}")
        else:
            logger.debug(f"Icon file not found: {icon_path}")
        
        # Enable window controls (minimize, maximize, close)
        self.root.resizable(True, True)
        # Store process status for loading screen
        self.current_processes = []
        
        # Initialize components
        self.data_fetcher = StockDataFetcher()
        self.learning_tracker = LearningTracker(APP_DATA_DIR)  # Track all learning sources
        self.price_move_predictor = PriceMovePredictor(APP_DATA_DIR)  # Predict magnitude of moves
        
        self.trading_analyzer = TradingAnalyzer(
            data_fetcher=self.data_fetcher,
            calibration_manager=self.learning_tracker.calibration_manager
        )
        self.investing_analyzer = InvestingAnalyzer()
        self.mixed_analyzer = MixedAnalyzer()
        self.chart_generator = ChartGenerator()
        self.portfolio = Portfolio(APP_DATA_DIR)
        self.model_benchmarker = ModelBenchmarker(APP_DATA_DIR, data_fetcher=self.data_fetcher)  # S&P 500 comparison
        
        # Pass price_move_predictor to predictions_tracker so it can learn from outcomes
        self.predictions_tracker = PredictionsTracker(APP_DATA_DIR, price_move_predictor=self.price_move_predictor)

        self.theme_manager = ThemeManager()
        self.preferences = UserPreferences(APP_DATA_DIR)
        self.market_scanner = MarketScanner(
            self.data_fetcher, 
            data_dir=APP_DATA_DIR,
            trading_analyzer=self.trading_analyzer,
            investing_analyzer=self.investing_analyzer,
            mixed_analyzer=self.mixed_analyzer
        )
        
        # Initialize hybrid AI system
        self.data_dir = APP_DATA_DIR
        # Get SeekerAI instance if available
        seeker_ai_instance = getattr(self, 'seeker_ai', None) if hasattr(self, 'seeker_ai') else None
        self.hybrid_predictors = {
            'trading': HybridStockPredictor(APP_DATA_DIR, 'trading', 
                                            trading_analyzer=self.trading_analyzer,
                                            seeker_ai=seeker_ai_instance),
            'mixed': HybridStockPredictor(APP_DATA_DIR, 'mixed',
                                         mixed_analyzer=self.mixed_analyzer,
                                         seeker_ai=seeker_ai_instance),
            'investing': HybridStockPredictor(APP_DATA_DIR, 'investing',
                                            investing_analyzer=self.investing_analyzer,
                                            seeker_ai=seeker_ai_instance)
        }
        self.training_manager = MLTrainingPipeline(self.data_fetcher, APP_DATA_DIR, app=self)
        self.model_tester = ModelTester(APP_DATA_DIR, self.data_fetcher)
        self.auto_training = AutoTrainingManager(self)
        self.auto_learner = AutoLearner(self)  # Automatic self-learning system
        
        # Load auto-learner settings from preferences
        saved_interval = self.preferences.get('auto_learner_scan_interval_hours')
        saved_predictions = self.preferences.get('auto_learner_predictions_per_scan')
        saved_confidence = self.preferences.get('auto_learner_min_confidence')
        
        if saved_interval is not None or saved_predictions is not None or saved_confidence is not None:
            self.auto_learner.update_config(
                scan_interval_hours=saved_interval,
                predictions_per_scan=saved_predictions,
                min_confidence=saved_confidence
            )
            logger.info(f"Loaded auto-learner settings: interval={saved_interval}h, predictions={saved_predictions}, confidence={saved_confidence}%")
        
        # self.learning_tracker moved up
        self.backtester = ContinuousBacktester(self, APP_DATA_DIR)  # Continuous backtesting system
        
        # ML configuration manager (for save/load configs)
        self._saved_training_config = None  # Stores loaded config for next training run
        
        # Initialize ML Scheduler
        try:
            from ml_scheduler import MLScheduler
            self.ml_scheduler = MLScheduler(self, APP_DATA_DIR)
            self.ml_scheduler.start()
            logger.info("ML Scheduler initialized and started")
        except Exception as e:
            logger.warning(f"Could not initialize ML Scheduler: {e}")
            self.ml_scheduler = None
        
        # Initialize new modules if available
        if HAS_NEW_MODULES:
            try:
                self.alert_system = AlertSystem(APP_DATA_DIR)
                self.walk_forward_backtester = WalkForwardBacktester(APP_DATA_DIR)
                logger.info("Alert system and walk-forward backtester initialized")
            except Exception as e:
                logger.warning(f"Could not initialize new modules: {e}")
                self.alert_system = None
                self.walk_forward_backtester = None
        else:
            self.alert_system = None
            self.walk_forward_backtester = None
        self.stock_monitor = StockMonitor(APP_DATA_DIR)  # Monitor stocks for BUY signals
        self.momentum_monitor = MomentumMonitor(APP_DATA_DIR)  # Monitor momentum and trend changes
        self.trend_change_predictor = TrendChangePredictor(data_dir=APP_DATA_DIR)  # Predict trend changes
        from trend_change_tracker import TrendChangeTracker
        self.trend_change_tracker = TrendChangeTracker(APP_DATA_DIR)  # Track trend change predictions for learning
        self.llm_ai_predictor = LLMAIPredictor(APP_DATA_DIR)  # AI Predictor for buy opportunities and reasoning
        self.holdings_tracker = HoldingsTracker(APP_DATA_DIR, app=self)  # Track user's stock holdings
        self.potentials_tracker = PotentialsTracker(APP_DATA_DIR)  # Track potential investment opportunities
        
        # Initialize momentum changes history from persistent storage
        self.momentum_changes_history = self.momentum_monitor.history
        
        # Initialize Seeker AI (before hybrid predictors so they can use it)
        if SEEKER_AI_ENABLED:
            try:
                self.seeker_ai = SeekerAI(APP_DATA_DIR)
                logger.info("Seeker AI initialized")
            except Exception as e:
                logger.error(f"Error initializing Seeker AI: {e}")
                self.seeker_ai = None
        else:
            self.seeker_ai = None
        
        # Backend server process
        self.backend_process = None
        
        # Start backend server automatically
        self._start_backend_server()
        
        # Load preferences and store as instance variables
        self.saved_language = self.preferences.get_language()
        self.saved_currency = self.preferences.get_currency()
        self.saved_theme = self.preferences.get_theme()
        self.saved_strategy = self.preferences.get_strategy()
        
        self.localization = Localization(language=self.saved_language, currency=self.saved_currency)
        
        # State variables
        self.current_strategy = self.saved_strategy
        self.current_symbol = None
        self.current_data = None
        self.current_analysis = None
        self.current_theme = self.saved_theme
        self.loading_screen = None
        
        # Initialize monitored stocks (Used by UI for favorites)
        self.monitored_stocks = self.preferences.get_monitored_stocks()
        if self.monitored_stocks is None:
            self.monitored_stocks = []
        
        # Create UI
        self._create_ui()
        
        # Load portfolio balance
        self._update_portfolio_display()
        
        # Verify predictions on startup
        self._verify_predictions_on_startup()
        
        # Start periodic automatic verification (every hour)
        self._start_periodic_verification()
        
        # Start periodic trend change verification
        self._start_trend_change_verification()
        
        # Start periodic peak/bottom detection verification
        self._start_peak_bottom_verification()
        
        # Start periodic buy opportunity verification
        self._start_buy_opportunity_verification()
        
        # Verify trend change predictions at startup (after a short delay)
        self.root.after(5000, self._verify_trend_changes_at_startup)
        
        # Verify peak/bottom detections at startup (after a short delay)
        self.root.after(5000, self._verify_peak_bottom_detections_at_startup)
        
        # Verify buy opportunities at startup
        self.root.after(7000, self._verify_buy_opportunities_at_startup)
        
        # Verify potentials at startup (and feed to Megamind)
        self.root.after(8000, self._verify_potentials_at_startup)
        
        # Schedule automatic training checks
        self._schedule_auto_training()
        
        # Check for auto-training on startup (after delay)
        self.root.after(5000, self.auto_training.check_and_train)
        
        # Start automatic self-learning (scans stocks and makes predictions automatically)
        self.root.after(10000, self.auto_learner.start)  # Start after 10 seconds
        
        # Start continuous backtesting (tests algorithm on random historical points)
        # DISABLED: Backtesting now runs manually via "Run Backtest" button
        # self.root.after(20000, self.backtester.start)  # Start after 20 seconds
        
        # Start stock monitoring for BUY signals (check every 30 minutes)
        self.root.after(30000, self._start_stock_monitoring)  # Start after 30 seconds
        
        # Start momentum monitoring for predictions (check every 15 minutes)
        self.root.after(60000, self._start_momentum_monitoring)  # Start after 1 minute
        
        # Initialize monitored stocks tracking (for peak/bottom/trend change notifications)
        saved_monitored = self.preferences.get('monitored_stocks', [])
        self.monitored_stocks = set(saved_monitored if isinstance(saved_monitored, list) else [])
        
        # Track last detected peak/bottom/trend changes to avoid duplicate notifications
        self._last_peak_bottom = {}  # {symbol: {'type': 'peak'|'bottom', 'price': float, 'time': str}}
        self._last_trend_change = {}  # {symbol: {'trend': str, 'time': str}}
        self.monitoring_status = {}  # {symbol: {price, status, time, change_pct, action}}
        self.monitoring_events = []  # Batch events for single notification
        self.is_monitoring_batch_active = False # Flag to suppress individual popups
        
        # Start periodic checking of monitored stocks for peaks/bottoms/trend changes
        self.root.after(120000, self._start_monitored_stocks_checking)  # Start after 2 minutes
        
        # Start periodic portfolio snapshot updates (daily)
        self._schedule_portfolio_snapshots()
        
        # Handle cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
        
        # Start periodic backend health check
        self._start_backend_health_check()
    
    def _is_backend_running(self) -> bool:
        """Check if backend server is already running on port 5000"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2)  # Increased timeout
            result = sock.connect_ex(('127.0.0.1', 5000))  # Use 127.0.0.1 instead of localhost
            sock.close()
            return result == 0
        except Exception as e:
            logger.debug(f"Error checking backend status: {e}")
            return False
    
    def _start_backend_health_check(self):
        """Periodically check if backend is running and restart if needed"""
        def check_backend():
            try:
                if not self._is_backend_running():
                    # Backend is not running, try to start it
                    if self.backend_process is None or (self.backend_process.poll() is not None):
                        logger.info("Backend server not running, attempting to start...")
                        self._start_backend_server()
                else:
                    # Backend is running, check if our process is still alive
                    if self.backend_process and self.backend_process.poll() is not None:
                        # Process died but port is still open (maybe another instance?)
                        logger.info("Backend process died but port is still open")
                        self.backend_process = None
            except Exception as e:
                logger.error(f"Error in backend health check: {e}")
            
            # Schedule next check in 30 seconds
            self.root.after(30000, check_backend)
        
        # Start first check after 10 seconds
        self.root.after(10000, check_backend)
    
    def _start_backend_server(self):
        """Start the backend server automatically if not already running"""
        def start_in_background():
            try:
                # Check if backend is already running
                if self._is_backend_running():
                    logger.info("Backend server is already running on port 5000")
                    return
                
                logger.info("Starting backend server automatically...")
                
                # Check if Flask is installed
                try:
                    import flask
                except ImportError:
                    logger.warning("Flask is not installed. Backend server cannot start. Install with: pip install flask flask-cors")
                    return
                
                # Get the path to backend_proxy.py
                backend_script = Path(__file__).parent / "backend_proxy.py"
                
                if not backend_script.exists():
                    logger.error(f"Backend script not found at {backend_script}")
                    return
                
                # Start backend in a separate process
                # Use CREATE_NO_WINDOW on Windows to hide the console window
                # Don't pipe stdout/stderr so we can see errors, but redirect to devnull to suppress normal output
                if sys.platform == 'win32':
                    self.backend_process = subprocess.Popen(
                        [sys.executable, str(backend_script)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                else:
                    self.backend_process = subprocess.Popen(
                        [sys.executable, str(backend_script)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.PIPE
                    )
                
                # Wait a bit for server to start
                time.sleep(5)  # Increased wait time
                
                # Check if process is still running
                if self.backend_process.poll() is not None:
                    # Process has terminated
                    stderr_output = self.backend_process.stderr.read().decode('utf-8', errors='ignore')
                    logger.error(f"Backend server process terminated immediately. Error: {stderr_output}")
                    self.backend_process = None
                    return
                
                # Verify it started by checking the port
                if self._is_backend_running():
                    logger.info("✅ Backend server started successfully on port 5000")
                else:
                    # Check for errors in stderr
                    try:
                        stderr_output = self.backend_process.stderr.read().decode('utf-8', errors='ignore')
                        if stderr_output:
                            logger.warning(f"Backend server may not have started properly. Error output: {stderr_output[:500]}")
                    except (IOError, OSError, AttributeError) as e:
                        logger.debug(f"Could not read backend stderr: {e}")
                    logger.warning("Backend server may not have started properly - port 5000 not responding")
                    
            except Exception as e:
                logger.error(f"Error starting backend server: {e}", exc_info=True)
                logger.info("App will continue using direct yfinance fallback")
        
        # Start in background thread to avoid blocking UI
        thread = threading.Thread(target=start_in_background)
        thread.daemon = True
        thread.start()
    
    def _on_closing(self):
        """Handle application closing - cleanup backend server"""
        try:
            if self.backend_process:
                logger.info("Shutting down backend server...")
                self.backend_process.terminate()
                # Wait up to 5 seconds for graceful shutdown
                try:
                    self.backend_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if it doesn't shut down gracefully
                    self.backend_process.kill()
                self.backend_process = None
        except Exception as e:
            logger.error(f"Error shutting down backend server: {e}")
        
        # Close the main window
        self.root.destroy()
    
    def _verify_predictions_on_startup(self):
        """Verify all active predictions when app starts - only checks predictions that have reached their target date"""
        def verify_in_background():
            try:
                logger.info("Verifying predictions that have reached their target date...")
                # Only verify predictions that have reached their estimated target date
                result = self.predictions_tracker.verify_all_active_predictions(self.data_fetcher, check_all=False)
                if result['verified'] > 0:
                    newly_verified = result.get('newly_verified', [])
                    # Record learning events for verified predictions
                    for pred in newly_verified:
                        if pred.get('was_correct') is not None:
                            self.learning_tracker.record_verified_prediction(
                                pred.get('was_correct', False),
                                pred.get('strategy', 'trading')
                            )
                    self.root.after(0, self._show_verification_results, result, newly_verified)
                    # Trigger auto-training on verified predictions
                    self.root.after(0, self._trigger_auto_training)
            except Exception as e:
                logger.error(f"Error verifying predictions: {e}")
        
        thread = threading.Thread(target=verify_in_background)
        thread.daemon = True
        thread.start()
    
    def _start_periodic_verification(self):
        """Start periodic automatic verification of predictions"""
        # Check every hour (3600000 milliseconds)
        check_interval_ms = 3600000  # 1 hour
        
        def periodic_check():
            def verify_in_background():
                try:
                    logger.info("Periodic verification of predictions...")
                    result = self.predictions_tracker.verify_all_active_predictions(self.data_fetcher)
                    if result['verified'] > 0:
                        newly_verified = result.get('newly_verified', [])
                        # Record learning events for verified predictions
                        for pred in newly_verified:
                            if pred.get('was_correct') is not None:
                                # Create outcome dict
                                outcome = {
                                    'was_correct': pred.get('was_correct', False),
                                    'actual_price_change': pred.get('actual_price_change', 0.0),
                                    'target_hit': pred.get('was_correct', False) and pred.get('action') in ['BUY', 'SELL'],
                                }
                                self.learning_tracker.record_verified_prediction(
                                    pred.get('was_correct', False),
                                    pred.get('strategy', 'trading'),
                                    prediction_data=pred,
                                    actual_outcome=outcome
                                )
                        self.root.after(0, self._show_verification_notifications, newly_verified)
                        self.root.after(0, self._update_predictions_display)
                        # Trigger auto-training
                        self.root.after(0, self._trigger_auto_training)
                        # Update ML status
                        self.root.after(0, self._update_ml_status)
                except Exception as e:
                    logger.error(f"Error in periodic verification: {e}")
            
            thread = threading.Thread(target=verify_in_background)
            thread.daemon = True
            thread.start()
            
            # Schedule next check
            self.root.after(check_interval_ms, periodic_check)
        
        # Start first check after 1 hour
        self.root.after(check_interval_ms, periodic_check)
    
    def _start_trend_change_verification(self):
        """Start periodic automatic verification of trend change predictions"""
        # Check every 6 hours (21600000 milliseconds)
        check_interval_ms = 21600000  # 6 hours
        
        def periodic_check():
            def verify_in_background():
                try:
                    logger.info("Periodic verification of trend change predictions...")
                    active_predictions = self.trend_change_tracker.get_active_predictions()
                    
                    if not active_predictions:
                        return
                    
                    verified_count = 0
                    for pred in active_predictions:
                        try:
                            # Check if estimated date has passed
                            estimated_date = datetime.fromisoformat(pred['estimated_date'])
                            if datetime.now() >= estimated_date:
                                # Verify this prediction
                                symbol = pred['symbol']
                                
                                # Fetch current data
                                stock_data = self.data_fetcher.fetch_stock_data(symbol, force_refresh=False)
                                history_data = self.data_fetcher.fetch_stock_history(symbol, period='3mo')
                                
                                if 'error' in stock_data or not history_data.get('data'):
                                    continue
                                
                                # Perform trading analysis to get current trend
                                trading_analysis = self.trading_analyzer.analyze(stock_data, history_data)
                                if 'error' in trading_analysis:
                                    continue
                                
                                current_trend = trading_analysis.get('price_action', {}).get('trend', 'sideways')
                                predicted_change = pred['predicted_change']
                                original_trend = pred['current_trend']
                                
                                # Get actual price data for price accuracy tracking
                                current_price = stock_data.get('price', 0)
                                price_history = history_data.get('data', [])
                                
                                # Find the actual low/high price during the prediction period
                                prediction_timestamp = datetime.fromisoformat(pred['timestamp'])
                                actual_low_price = None
                                actual_high_price = None
                                
                                for price_point in price_history:
                                    try:
                                        price_date = datetime.fromisoformat(price_point.get('date', ''))
                                        if price_date >= prediction_timestamp:
                                            price_low = price_point.get('low', current_price)
                                            price_high = price_point.get('high', current_price)
                                            
                                            if actual_low_price is None or price_low < actual_low_price:
                                                actual_low_price = price_low
                                            if actual_high_price is None or price_high > actual_high_price:
                                                actual_high_price = price_high
                                    except:
                                        continue
                                
                                # Determine if prediction was correct
                                was_correct = False
                                actual_change = None
                                
                                if predicted_change == 'bullish_reversal':
                                    # Predicted bullish reversal (downtrend -> uptrend)
                                    if original_trend == 'downtrend' and current_trend == 'uptrend':
                                        was_correct = True
                                        actual_change = 'bullish_reversal'
                                    elif current_trend == original_trend:
                                        actual_change = 'no_change'
                                    else:
                                        actual_change = f'{original_trend}_to_{current_trend}'
                                
                                elif predicted_change == 'bearish_reversal':
                                    # Predicted bearish reversal (uptrend -> downtrend)
                                    if original_trend == 'uptrend' and current_trend == 'downtrend':
                                        was_correct = True
                                        actual_change = 'bearish_reversal'
                                    elif current_trend == original_trend:
                                        actual_change = 'no_change'
                                    else:
                                        actual_change = f'{original_trend}_to_{current_trend}'
                                
                                # Verify the prediction with price data
                                verified_pred = self.trend_change_tracker.verify_prediction(
                                    pred['id'],
                                    actual_change or current_trend,
                                    was_correct,
                                    actual_low_price=actual_low_price,
                                    actual_high_price=actual_high_price,
                                    current_price=current_price
                                )
                                
                                if verified_pred:
                                    verified_count += 1
                                    # Record in learning tracker
                                    self.learning_tracker.record_verified_trend_change(
                                        was_correct=was_correct,
                                        predicted_change=predicted_change,
                                        actual_change=actual_change or current_trend,
                                        confidence=pred.get('confidence', 0)
                                    )
                                    
                                    logger.info(f"✅ Verified trend change prediction #{pred['id']} for {symbol}: {'correct' if was_correct else 'incorrect'}")
                        except Exception as e:
                            logger.error(f"Error verifying trend change prediction #{pred.get('id')}: {e}")
                    
                    if verified_count > 0:
                        logger.info(f"Verified {verified_count} trend change prediction(s)")
                        
                        # Get verified predictions from tracker (they're now marked as verified)
                        verified_predictions = []
                        for pred in active_predictions:
                            # Reload from tracker to get updated verification status
                            verified_pred = self.trend_change_tracker.get_prediction_by_id(pred['id'])
                            if verified_pred and verified_pred.get('verified') and verified_pred.get('was_correct') is not None:
                                verified_predictions.append(verified_pred)
                        
                        if verified_predictions:
                            logger.info(f"🧠 Learning from {len(verified_predictions)} verified trend change predictions...")
                            self.trend_change_predictor.learn_from_verified_predictions(verified_predictions)
                        
                        # Trigger auto-training to learn from verified predictions
                        self.root.after(0, self._trigger_auto_training)
                        # Update ML status
                        self.root.after(0, self._update_ml_status)
                        # Update trend change display to show verified predictions are gone
                        if hasattr(self, 'trend_change_tree'):
                            self.root.after(0, self._update_trend_change_display)
                except Exception as e:
                    logger.error(f"Error in trend change verification: {e}")
            
            thread = threading.Thread(target=verify_in_background)
            thread.daemon = True
            thread.start()
            
            # Schedule next check
            self.root.after(check_interval_ms, periodic_check)
        
        # Start first check after 1 hour
        self.root.after(3600000, periodic_check)
    
    def _verify_trend_changes_at_startup(self):
        """Verify trend change predictions at startup"""
        def verify_in_background():
            try:
                logger.info("Verifying trend change predictions at startup...")
                active_predictions = self.trend_change_tracker.get_active_predictions()
                
                if not active_predictions:
                    return
                
                verified_count = 0
                for pred in active_predictions:
                    try:
                        # Check if estimated date has passed
                        estimated_date = datetime.fromisoformat(pred['estimated_date'])
                        if datetime.now() >= estimated_date:
                            # Verify this prediction
                            symbol = pred['symbol']
                            
                            # Fetch current data
                            stock_data = self.data_fetcher.fetch_stock_data(symbol, force_refresh=False)
                            history_data = self.data_fetcher.fetch_stock_history(symbol, period='3mo')
                            
                            if 'error' in stock_data or not history_data.get('data'):
                                continue
                            
                            # Perform trading analysis to get current trend
                            trading_analysis = self.trading_analyzer.analyze(stock_data, history_data)
                            if 'error' in trading_analysis:
                                continue
                            
                            current_trend = trading_analysis.get('price_action', {}).get('trend', 'sideways')
                            predicted_change = pred['predicted_change']
                            original_trend = pred['current_trend']
                            
                            # Get actual price data for price accuracy tracking
                            current_price = stock_data.get('price', 0)
                            price_history = history_data.get('data', [])
                            
                            # Find the actual low/high price during the prediction period
                            prediction_timestamp = datetime.fromisoformat(pred['timestamp'])
                            actual_low_price = None
                            actual_high_price = None
                            
                            for price_point in price_history:
                                try:
                                    price_date = datetime.fromisoformat(price_point.get('date', ''))
                                    if price_date >= prediction_timestamp:
                                        price_low = price_point.get('low', current_price)
                                        price_high = price_point.get('high', current_price)
                                        
                                        if actual_low_price is None or price_low < actual_low_price:
                                            actual_low_price = price_low
                                        if actual_high_price is None or price_high > actual_high_price:
                                            actual_high_price = price_high
                                except:
                                    continue
                            
                            # Determine if prediction was correct
                            was_correct = False
                            actual_change = None
                            
                            if predicted_change == 'bullish_reversal':
                                if original_trend == 'downtrend' and current_trend == 'uptrend':
                                    was_correct = True
                                    actual_change = 'bullish_reversal'
                                elif current_trend == original_trend:
                                    actual_change = 'no_change'
                                else:
                                    actual_change = f'{original_trend}_to_{current_trend}'
                            
                            elif predicted_change == 'bearish_reversal':
                                if original_trend == 'uptrend' and current_trend == 'downtrend':
                                    was_correct = True
                                    actual_change = 'bearish_reversal'
                                elif current_trend == original_trend:
                                    actual_change = 'no_change'
                                else:
                                    actual_change = f'{original_trend}_to_{current_trend}'
                            
                            # Verify the prediction with price data
                            verified_pred = self.trend_change_tracker.verify_prediction(
                                pred['id'],
                                actual_change or current_trend,
                                was_correct,
                                actual_low_price=actual_low_price,
                                actual_high_price=actual_high_price,
                                current_price=current_price
                            )
                            
                            if verified_pred:
                                verified_count += 1
                                # Record in learning tracker
                                self.learning_tracker.record_verified_trend_change(
                                    was_correct=was_correct,
                                    predicted_change=predicted_change,
                                    actual_change=actual_change or current_trend,
                                    confidence=pred.get('confidence', 0)
                                )
                                
                                logger.info(f"✅ Verified trend change prediction #{pred['id']} for {symbol} at startup: {'correct' if was_correct else 'incorrect'}")
                    except Exception as e:
                        logger.error(f"Error verifying trend change prediction #{pred.get('id')} at startup: {e}")
                
                if verified_count > 0:
                    logger.info(f"Verified {verified_count} trend change prediction(s) at startup")
                    
                    # Get verified predictions from tracker (they're now marked as verified)
                    verified_predictions = []
                    for pred in active_predictions:
                        # Reload from tracker to get updated verification status
                        verified_pred = self.trend_change_tracker.get_prediction_by_id(pred['id'])
                        if verified_pred and verified_pred.get('verified') and verified_pred.get('was_correct') is not None:
                            verified_predictions.append(verified_pred)
                    
                    if verified_predictions:
                        logger.info(f"🧠 Learning from {len(verified_predictions)} verified trend change predictions...")
                        self.trend_change_predictor.learn_from_verified_predictions(verified_predictions)
                    
                    # Update trend change display to show verified predictions are gone
                    if hasattr(self, 'trend_change_tree'):
                        self.root.after(0, self._update_trend_change_display)
                    
                    # Trigger auto-training
                    self.root.after(0, self._trigger_auto_training)
                    self.root.after(0, self._update_ml_status)
            except Exception as e:
                logger.error(f"Error in startup trend change verification: {e}")
        
        thread = threading.Thread(target=verify_in_background)
        thread.daemon = True
        thread.start()
    
    def _verify_peak_bottom_detections_at_startup(self):
        """Verify peak/bottom detections at startup"""
        def verify_in_background():
            try:
                logger.info("Verifying peak/bottom detections at startup...")
                verified_count = 0
                
                # Verify peak detections
                pending_peaks = self.learning_tracker.get_pending_peak_verifications()
                for detection in pending_peaks:
                    try:
                        symbol = detection.get('symbol')
                        if not symbol:
                            continue
                        
                        # Fetch current price
                        stock_data = self.data_fetcher.fetch_stock_data(symbol, force_refresh=False)
                        if 'error' in stock_data:
                            continue
                        
                        current_price = stock_data.get('price', 0)
                        if current_price == 0:
                            continue
                        
                        # Verify the detection
                        verification_result = self.learning_tracker.verify_peak_detection(
                            detection, current_price
                        )
                        
                        # Record verified detection
                        self.learning_tracker.record_verified_peak_detection(
                            detection, verification_result
                        )
                        
                        verified_count += 1
                        logger.info(f"Verified peak detection for {symbol} at startup: {'correct' if verification_result.get('was_correct') else 'false positive'}")
                    except Exception as e:
                        logger.error(f"Error verifying peak detection for {detection.get('symbol', 'UNKNOWN')} at startup: {e}")
                
                # Verify bottom detections
                pending_bottoms = self.learning_tracker.get_pending_bottom_verifications()
                for detection in pending_bottoms:
                    try:
                        symbol = detection.get('symbol')
                        if not symbol:
                            continue
                        
                        # Fetch current price
                        stock_data = self.data_fetcher.fetch_stock_data(symbol, force_refresh=False)
                        if 'error' in stock_data:
                            continue
                        
                        current_price = stock_data.get('price', 0)
                        if current_price == 0:
                            continue
                        
                        # Verify the detection
                        verification_result = self.learning_tracker.verify_bottom_detection(
                            detection, current_price
                        )
                        
                        # Record verified detection
                        self.learning_tracker.record_verified_bottom_detection(
                            detection, verification_result
                        )
                        
                        verified_count += 1
                        logger.info(f"Verified bottom detection for {symbol} at startup: {'correct' if verification_result.get('was_correct') else 'false positive'}")
                    except Exception as e:
                        logger.error(f"Error verifying bottom detection for {detection.get('symbol', 'UNKNOWN')} at startup: {e}")
                
                if verified_count > 0:
                    logger.info(f"Verified {verified_count} peak/bottom detection(s) at startup")
                    # Update ML status if needed
                    self.root.after(0, self._update_ml_status)
            except Exception as e:
                logger.error(f"Error in startup peak/bottom verification: {e}")
        
        thread = threading.Thread(target=verify_in_background)
        thread.daemon = True
        thread.start()
    
    def _start_peak_bottom_verification(self):
        """Start periodic automatic verification of peak/bottom detections"""
        # Check every 12 hours (43200000 milliseconds)
        check_interval_ms = 43200000  # 12 hours
        
        def periodic_check():
            def verify_in_background():
                try:
                    logger.info("Periodic verification of peak/bottom detections...")
                    verified_count = 0
                    
                    # Verify peak detections
                    pending_peaks = self.learning_tracker.get_pending_peak_verifications()
                    for detection in pending_peaks:
                        try:
                            symbol = detection.get('symbol')
                            if not symbol:
                                continue
                            
                            # Fetch current price
                            stock_data = self.data_fetcher.fetch_stock_data(symbol, force_refresh=False)
                            if 'error' in stock_data:
                                continue
                            
                            current_price = stock_data.get('price', 0)
                            if current_price == 0:
                                continue
                            
                            # Verify the detection
                            verification_result = self.learning_tracker.verify_peak_detection(
                                detection, current_price
                            )
                            
                            # Record verified detection
                            self.learning_tracker.record_verified_peak_detection(
                                detection, verification_result
                            )
                            
                            verified_count += 1
                            logger.info(f"Verified peak detection for {symbol}: {'correct' if verification_result.get('was_correct') else 'false positive'}")
                        except Exception as e:
                            logger.error(f"Error verifying peak detection for {detection.get('symbol', 'UNKNOWN')}: {e}")
                    
                    # Verify bottom detections
                    pending_bottoms = self.learning_tracker.get_pending_bottom_verifications()
                    for detection in pending_bottoms:
                        try:
                            symbol = detection.get('symbol')
                            if not symbol:
                                continue
                            
                            # Fetch current price
                            stock_data = self.data_fetcher.fetch_stock_data(symbol, force_refresh=False)
                            if 'error' in stock_data:
                                continue
                            
                            current_price = stock_data.get('price', 0)
                            if current_price == 0:
                                continue
                            
                            # Verify the detection
                            verification_result = self.learning_tracker.verify_bottom_detection(
                                detection, current_price
                            )
                            
                            # Record verified detection
                            self.learning_tracker.record_verified_bottom_detection(
                                detection, verification_result
                            )
                            
                            verified_count += 1
                            logger.info(f"Verified bottom detection for {symbol}: {'correct' if verification_result.get('was_correct') else 'false positive'}")
                        except Exception as e:
                            logger.error(f"Error verifying bottom detection for {detection.get('symbol', 'UNKNOWN')}: {e}")
                    
                    if verified_count > 0:
                        logger.info(f"Verified {verified_count} peak/bottom detection(s) in periodic check")
                        # Update ML status if needed
                        self.root.after(0, self._update_ml_status)
                except Exception as e:
                    logger.error(f"Error in periodic peak/bottom verification: {e}")
            
            thread = threading.Thread(target=verify_in_background)
            thread.daemon = True
            thread.start()
            
            # Schedule next check
            self.root.after(check_interval_ms, periodic_check)
        
        # Start first check after initial delay (12 hours)
        self.root.after(check_interval_ms, periodic_check)

    def _start_buy_opportunity_verification(self):
        """Start periodic automatic verification of buy opportunity predictions (Megamind)"""
        # Check every 12 hours (43200000 milliseconds)
        check_interval_ms = 43200000
        
        def periodic_check():
            def verify_in_background():
                try:
                    logger.info("Periodic verification of buy opportunity predictions...")
                    pending = self.learning_tracker.data.get('pending_buy_opportunities', [])
                    if not pending:
                        return
                        
                    for record in pending:
                        if record.get('verified'):
                            continue
                            
                        symbol = record['symbol']
                        stock_data = self.data_fetcher.fetch_stock_data(symbol, force_refresh=False)
                        if 'error' in stock_data:
                            continue
                            
                        current_price = stock_data.get('price', 0)
                        result = self.learning_tracker.verify_buy_opportunity(record, current_price)
                        
                        if result:
                            self.learning_tracker.record_verified_buy_opportunity(record, result)
                    
                    self.root.after(0, self._update_ml_status)
                except Exception as e:
                    logger.error(f"Error in buy opportunity verification: {e}")
            
            thread = threading.Thread(target=verify_in_background)
            thread.daemon = True
            thread.start()
            
            # Schedule next check
            self.root.after(check_interval_ms, periodic_check)
        
        # Start first check after 1 hour instead of 12 to verify existing ones
        self.root.after(3600000, periodic_check)

    def _verify_buy_opportunities_at_startup(self):
        """Verify buy opportunity predictions at startup"""
        def verify_in_background():
            try:
                logger.info("Verifying buy opportunity predictions at startup...")
                pending = self.learning_tracker.data.get('pending_buy_opportunities', [])
                if not pending:
                    return
                    
                for record in pending:
                    if record.get('verified'):
                        continue
                        
                    symbol = record['symbol']
                    stock_data = self.data_fetcher.fetch_stock_data(symbol, force_refresh=False)
                    if 'error' in stock_data:
                        continue
                        
                    current_price = stock_data.get('price', 0)
                    result = self.learning_tracker.verify_buy_opportunity(record, current_price)
                    
                    if result:
                        self.learning_tracker.record_verified_buy_opportunity(record, result)
                
                self.root.after(0, self._update_ml_status)
            except Exception as e:
                logger.error(f"Error in startup buy opportunity verification: {e}")
                
        thread = threading.Thread(target=verify_in_background)
        thread.daemon = True
        thread.start()
    
    def _start_stock_monitoring(self):
        """Start periodic monitoring of stocks for BUY signals"""
        # Check every 30 minutes (1800000 milliseconds)
        check_interval_ms = 1800000  # 30 minutes
        
        def periodic_monitor():
            def monitor_in_background():
                try:
                    monitored_symbols = self.stock_monitor.get_monitored_stocks()
                    if not monitored_symbols:
                        logger.debug("No stocks to monitor")
                        return
                    
                    logger.info(f"Checking {len(monitored_symbols)} monitored stocks for BUY signals...")
                    buy_signals = []
                    
                    for symbol in monitored_symbols:
                        try:
                            stock_info = self.stock_monitor.get_stock_info(symbol)
                            if not stock_info:
                                continue
                            
                            strategy = stock_info.get('strategy', 'mixed')
                            
                            # Re-analyze the stock
                            stock_data = self.data_fetcher.fetch_stock_data(symbol)
                            if 'error' in stock_data:
                                logger.debug(f"Could not fetch data for {symbol}: {stock_data.get('error')}")
                                continue
                            
                            history_data = self.data_fetcher.fetch_stock_history(symbol)
                            if 'error' in history_data:
                                logger.debug(f"Could not fetch history for {symbol}: {history_data.get('error')}")
                                continue
                            
                            # Get financials for mixed/investing strategies
                            financials_data = None
                            if strategy in ['mixed', 'investing']:
                                try:
                                    financials_data = self.data_fetcher.fetch_financials(symbol)
                                except Exception as e:
                                    logger.debug(f"Could not fetch financials for {symbol}: {e}")
                            
                            # Get hybrid predictor for the strategy
                            hybrid_predictor = self.hybrid_predictors.get(strategy)
                            if not hybrid_predictor:
                                continue
                            
                            # Make prediction
                            analysis = hybrid_predictor.predict(stock_data, history_data, financials_data)
                            if 'error' in analysis:
                                logger.debug(f"Analysis error for {symbol}: {analysis.get('error')}")
                                continue
                            
                            recommendation = analysis.get('recommendation', {})
                            new_action = recommendation.get('action', 'HOLD')
                            new_confidence = recommendation.get('confidence', 0)
                            
                            # Check for bullish signal (SELL in inverted logic)
                            if self.stock_monitor.check_for_bullish_signal(symbol, new_action, new_confidence):
                                buy_signals.append({
                                    'symbol': symbol,
                                    'confidence': new_confidence,
                                    'entry_price': recommendation.get('entry_price', stock_data.get('price', 0)),
                                    'target_price': recommendation.get('target_price', 0),
                                    'strategy': strategy
                                })
                                logger.info(f"🟢 BULLISH SIGNAL: {symbol} changed to SELL ({new_confidence:.1f}% confidence)")
                        
                        except Exception as e:
                            logger.error(f"Error monitoring {symbol}: {e}")
                            continue
                    
                    # Show notifications for BUY signals
                    if buy_signals:
                        self.root.after(0, self._show_buy_signal_notifications, buy_signals)
                
                except Exception as e:
                    logger.error(f"Error in stock monitoring: {e}")
            
            thread = threading.Thread(target=monitor_in_background)
            thread.daemon = True
            thread.start()
            
            # Schedule next check
            self.root.after(check_interval_ms, periodic_monitor)
        
        # Start first check after 30 minutes
        self.root.after(check_interval_ms, periodic_monitor)
    
    def _show_buy_signal_notifications(self, buy_signals: List[Dict]):
        """Show batched notifications for BUY signals"""
        if not buy_signals:
            return
        
        # Check notification preferences
        show_buy_signals = self.preferences.get('show_buy_signal_notifications', True)
        if not show_buy_signals:
            logger.debug(f"Buy signal notifications disabled. {len(buy_signals)} signals detected but not shown.")
            return
        
        # Batch notifications if multiple signals
        if len(buy_signals) == 1:
            # Single signal - show detailed notification
            signal = buy_signals[0]
            symbol = signal['symbol']
            confidence = signal['confidence']
            entry_price = signal.get('entry_price', 0)
            target_price = signal.get('target_price', 0)
            strategy = signal.get('strategy', 'mixed')
            
            message = f"🟢 BULLISH SIGNAL: {symbol}\n\n"
            message += f"Stock changed to BULLISH (SELL) recommendation!\n"
            message += f"Confidence: {confidence:.1f}%\n"
            if entry_price > 0:
                entry_usd = f"${entry_price:,.2f}"
                entry_eur = f"€{self.localization.convert_from_usd(entry_price, 'EUR'):,.2f}"
                message += f"Entry Price: {entry_usd} USD / {entry_eur} EUR\n"
            if target_price > 0:
                target_usd = f"${target_price:,.2f}"
                target_eur = f"€{self.localization.convert_from_usd(target_price, 'EUR'):,.2f}"
                message += f"Target Price: {target_usd} USD / {target_eur} EUR\n"
            message += f"Strategy: {strategy.title()}\n\n"
            message += "Click OK to analyze this stock now."
            
            messagebox.showinfo("🟢 BULLISH Signal Detected!", message)
            logger.info(f"Showed BULLISH signal notification for {symbol}")
        else:
            # Multiple signals - show batched notification
            message = f"🟢 {len(buy_signals)} BULLISH Signals Detected!\n\n"
            message += "Stocks with BULLISH (SELL) recommendations:\n\n"
            for i, signal in enumerate(buy_signals[:10], 1):  # Show max 10
                symbol = signal['symbol']
                confidence = signal['confidence']
                message += f"{i}. {symbol} ({confidence:.1f}% confidence)\n"
            
            if len(buy_signals) > 10:
                message += f"\n... and {len(buy_signals) - 10} more\n"
            
            message += "\nView all signals in the Market Scanner tab."
            messagebox.showinfo("🟢 Multiple BUY Signals Detected!", message)
            logger.info(f"Showed batched BUY signal notification for {len(buy_signals)} signals")
    
    def _start_momentum_monitoring(self):
        """Start periodic monitoring of stocks in predictions for momentum/trend changes"""
        # Check every 15 minutes (900000 milliseconds)
        check_interval_ms = 900000  # 15 minutes
        
        def periodic_momentum_check():
            def check_in_background():
                try:
                    # Get all active predictions
                    active_predictions = [
                        p for p in self.predictions_tracker.predictions 
                        if p.get('status') == 'active'
                    ]
                    
                    if not active_predictions:
                        logger.debug("No active predictions to monitor for momentum changes")
                        return
                    
                    logger.info(f"Checking {len(active_predictions)} predictions for momentum/trend changes...")
                    momentum_changes = []
                    
                    for pred in active_predictions:
                        symbol = pred.get('symbol', '').upper()
                        if not symbol:
                            continue
                        
                        try:
                            strategy = pred.get('strategy', 'mixed')
                            
                            # Fetch current data
                            stock_data = self.data_fetcher.fetch_stock_data(symbol, force_refresh=True)
                            if 'error' in stock_data:
                                logger.debug(f"Could not fetch data for {symbol}: {stock_data.get('error')}")
                                continue
                            
                            history_data = self.data_fetcher.fetch_stock_history(symbol)
                            if 'error' in history_data:
                                logger.debug(f"Could not fetch history for {symbol}: {history_data.get('error')}")
                                continue
                            
                            # Check momentum for trading and mixed strategies (both use technical indicators)
                            if strategy in ["trading", "mixed"]:
                                # Perform trading analysis to get indicators
                                trading_analysis = self.trading_analyzer.analyze(stock_data, history_data)
                                if 'error' in trading_analysis:
                                    continue
                                
                                # Check for momentum changes
                                if 'indicators' in trading_analysis and 'price_action' in trading_analysis and 'momentum_analysis' in trading_analysis:
                                    changes = self.momentum_monitor.update_momentum_state(
                                        symbol,
                                        trading_analysis['indicators'],
                                        trading_analysis['price_action'],
                                        trading_analysis['momentum_analysis']
                                    )
                                    
                                    # If significant changes detected, add to notification list
                                    if changes.get('trend_reversed') or (changes.get('momentum_changed') and len(changes.get('changes', [])) > 0):
                                        momentum_changes.append({
                                            'symbol': symbol,
                                            'prediction_id': pred.get('id'),
                                            'action': pred.get('action'),
                                            'strategy': strategy,
                                            'changes': changes
                                        })
                                        logger.info(f"🔄 Momentum/Trend change detected for {symbol} (prediction #{pred.get('id')}): {len(changes.get('changes', []))} changes")
                            
                        except Exception as e:
                            logger.error(f"Error checking momentum for {symbol}: {e}")
                            continue
                    
                    # Show notifications for momentum changes and update dedicated tab
                    if momentum_changes:
                        # Add to history for the dedicated tab
                        for change in momentum_changes:
                            self.momentum_monitor.add_to_history(
                                symbol=change['symbol'],
                                action=change['action'], # Old action
                                changes=change['changes'],
                                strategy=change['strategy']
                            )
                        
                        # Update display
                        self.root.after(0, self._update_momentum_changes_display)
                        self.root.after(0, self._show_momentum_change_notifications, momentum_changes)
                
                except Exception as e:
                    logger.error(f"Error in momentum monitoring: {e}")
            
            thread = threading.Thread(target=check_in_background)
            thread.daemon = True
            thread.start()
            
            # Schedule next check
            self.root.after(check_interval_ms, periodic_momentum_check)
        
        # Start first check
        periodic_momentum_check()
    
    def _show_momentum_change_notification(self, symbol: str, changes: Dict):
        """Show notification for momentum/trend change in currently analyzed stock"""
        if not changes.get('changes'):
            return
        
        change_list = changes['changes']
        is_trend_reversal = changes.get('trend_reversed', False)
        recommendation = changes.get('recommendation', {})
        trend_change = changes.get('trend_change')
        
        # Determine title and action
        action = recommendation.get('action', 'HOLD')
        confidence = recommendation.get('confidence', 50)
        
        if is_trend_reversal:
            title = f"⚠️ Trend Reversal: {action} (Confidence: {confidence}%)"
        else:
            title = f"🔄 Momentum Change: {action} (Confidence: {confidence}%)"
        
        message = f"{symbol} - Technical Analysis Update\n\n"
        
        # Show trend change if available
        if trend_change:
            message += f"📊 TREND CHANGE: {trend_change}\n\n"
        
        # Show recommendation
        action_emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "🟡"
        message += f"{action_emoji} RECOMMENDATION: {action} (Confidence: {confidence}%)\n\n"
        
        # Show recommendation reasons
        reasons = recommendation.get('reasons', [])
        if reasons:
            message += "Key Factors:\n"
            for reason in reasons:
                message += f"• {reason}\n"
            message += "\n"
        
        # Show detected changes
        message += "Detected Changes:\n"
        for change in change_list[:5]:  # Show first 5 changes
            message += f"• {change}\n"
        
        if len(change_list) > 5:
            message += f"\n... and {len(change_list) - 5} more changes"
        
        messagebox.showinfo(title, message)
        logger.info(f"Showed momentum change notification for {symbol}: {action} ({confidence}%)")
    
    def _show_peak_notification(self, symbol: str, peak_price: float, current_price: float, confidence: float, reasoning: str, timestamp: str):
        """Show notification when peak is detected"""
        # Batching check
        if hasattr(self, 'is_monitoring_batch_active') and self.is_monitoring_batch_active:
            self.monitoring_events.append({
                'type': 'PEAK',
                'symbol': symbol,
                'price': peak_price,
                'message': f"Peak at ${peak_price:.2f}"
            })
            logger.info(f"Batched peak notification for {symbol}")
            return

        title = "🔴 PEAK DETECTED - REVERSAL INCOMING"
        message = f"📈 {symbol} - Highest Price Reached\n\n"
        message += f"Time: {timestamp}\n"
        message += f"Peak Price (Highest in 20 days): ${peak_price:,.2f}\n"
        message += f"Current Price: ${current_price:,.2f}\n"
        
        # Calculate price difference
        price_diff = peak_price - current_price
        price_diff_pct = ((peak_price - current_price) / peak_price) * 100
        
        # Determine if reversal is confirmed or just detected
        # Reversal is confirmed if price moved significantly away from peak (3%+)
        reversal_confirmed = price_diff_pct >= 3.0
        
        if reversal_confirmed:
            message += f"Price Change: -${price_diff:.2f} ({price_diff_pct:.1f}% down from peak)\n"
            message += f"\n🔄 REVERSAL CONFIRMED: Price has declined {price_diff_pct:.1f}% from peak!\n\n"
        elif current_price < peak_price:
            message += f"Price Change: -${price_diff:.2f} ({price_diff_pct:.1f}% below peak high)\n"
            message += f"\n⚠️ PEAK DETECTED: Price reached ${peak_price:.2f} (highest in 20 days). "
            message += f"Current price is {price_diff_pct:.1f}% below peak.\n"
            message += f"⚠️ WARNING: If price continues rising above ${peak_price:.2f}, this was a false peak detection.\n\n"
        else:
            message += f"\n⚠️ REVERSAL IMMINENT: Price at/near peak - expect downward movement!\n\n"
        
        message += f"Confidence: {confidence:.1f}%\n\n"
        message += f"Reasoning:\n{reasoning}\n\n"
        
        if reversal_confirmed:
            message += "⚠️ Price reached its highest point and has started reversing downward.\n\n"
            message += "💰 SELL SIGNAL: Consider taking profits NOW!\n"
            message += "   • Price has declined from peak\n"
            message += "   • Best time to exit positions\n"
            message += "   • Lock in gains before further decline\n\n"
        else:
            message += "⚠️ Peak detected based on indicators, but reversal not yet confirmed.\n"
            message += "   • Monitor if price breaks above peak (false signal)\n"
            message += "   • If price declines further, reversal will be confirmed\n\n"
        
        message += "🧠 Data fed to Megamind for learning!"
        
        messagebox.showinfo(title, message)
        logger.info(f"Peak notification shown for {symbol}: Peak ${peak_price:.2f}, Current ${current_price:.2f}, Reversal Confirmed: {reversal_confirmed}")
    
    def _show_bottom_notification(self, symbol: str, bottom_price: float, current_price: float, confidence: float, reasoning: str, timestamp: str):
        """Show notification when bottom is detected"""
        # Batching check
        if hasattr(self, 'is_monitoring_batch_active') and self.is_monitoring_batch_active:
            self.monitoring_events.append({
                'type': 'BOTTOM',
                'symbol': symbol,
                'price': bottom_price,
                'message': f"Bottom at ${bottom_price:.2f}"
            })
            logger.info(f"Batched bottom notification for {symbol}")
            return

        title = "🟢 BOTTOM DETECTED - REVERSAL INCOMING"
        message = f"📉 {symbol} - Lowest Price Reached\n\n"
        message += f"Time: {timestamp}\n"
        message += f"Bottom Price (Lowest): ${bottom_price:,.2f}\n"
        message += f"Current Price: ${current_price:,.2f}\n"
        
        if current_price > bottom_price:
            price_diff = current_price - bottom_price
            price_diff_pct = ((current_price - bottom_price) / bottom_price) * 100
            message += f"Price Change: +${price_diff:.2f} ({price_diff_pct:.1f}% up from bottom)\n"
            message += f"\n🔄 REVERSAL CONFIRMED: Price has already started rising!\n\n"
        else:
            message += f"\n⚠️ REVERSAL IMMINENT: Price at bottom - expect upward movement!\n\n"
        
        message += f"Confidence: {confidence:.1f}%\n\n"
        message += f"Reasoning:\n{reasoning}\n\n"
        message += "⚠️ Price reached its lowest point and is likely to reverse upward.\n"
        message += "Consider preparing for BUY signal.\n\n"
        message += "🧠 Data fed to Megamind for learning!"
        
        messagebox.showinfo(title, message)
        logger.info(f"Bottom notification shown for {symbol}: Bottom ${bottom_price:.2f}, Current ${current_price:.2f}")
    
    def _show_momentum_change_notifications(self, momentum_changes: List[Dict]):
        """Show batched notifications for momentum/trend changes in predictions"""
        if not momentum_changes:
            return
        
        # Check notification preferences
        show_momentum = self.preferences.get('show_momentum_change_notifications', True)
        if not show_momentum:
            logger.debug(f"Momentum change notifications disabled. {len(momentum_changes)} changes detected but not shown.")
            return
        
        # Throttle: Only show if significant changes (trend reversals or action changes)
        significant_changes = []
        for change_info in momentum_changes:
            changes = change_info['changes']
            is_trend_reversal = changes.get('trend_reversed', False)
            recommendation = changes.get('recommendation', {})
            pred_action = change_info['action']
            new_action = recommendation.get('action', 'HOLD')
            action_changed = (new_action != pred_action)
            
            # Only show if trend reversed OR action changed significantly
            if is_trend_reversal or action_changed:
                significant_changes.append(change_info)
        
        if not significant_changes:
            return  # No significant changes, don't notify
        
        # Batch notifications
        if len(significant_changes) == 1:
            # Single significant change - show detailed notification
            change_info = significant_changes[0]
            symbol = change_info['symbol']
            pred_id = change_info['prediction_id']
            pred_action = change_info['action']
            changes = change_info['changes']
            
            is_trend_reversal = changes.get('trend_reversed', False)
            recommendation = changes.get('recommendation', {})
            trend_change = changes.get('trend_change')
            new_action = recommendation.get('action', 'HOLD')
            confidence = recommendation.get('confidence', 50)
            
            if is_trend_reversal:
                title = f"⚠️ Trend Reversal in Prediction #{pred_id}!"
            else:
                title = f"🔄 Momentum Change in Prediction #{pred_id}"
            
            message = f"Prediction #{pred_id}: {symbol}\n"
            message += f"Previous Recommendation: {pred_action}\n\n"
            
            if trend_change:
                message += f"📊 TREND CHANGE: {trend_change}\n\n"
            
            action_emoji = "🟢" if new_action == "BUY" else "🔴" if new_action == "SELL" else "🟡"
            message += f"{action_emoji} NEW RECOMMENDATION: {new_action} (Confidence: {confidence}%)\n"
            
            if pred_action == "BUY" and new_action == "SELL":
                message += "⚠️ WARNING: Recommendation changed from BUY to SELL!\n"
            elif pred_action == "SELL" and new_action == "BUY":
                message += "⚠️ WARNING: Recommendation changed from SELL to BUY!\n"
            
            messagebox.showinfo(title, message)
            logger.info(f"Showed momentum change notification for prediction #{pred_id} ({symbol}): {pred_action} → {new_action} ({confidence}%)")
        else:
            # Multiple changes - show batched notification
            title = f"🔄 {len(significant_changes)} Momentum Changes Detected"
            message = f"{len(significant_changes)} prediction(s) have significant momentum/trend changes:\n\n"
            
            for change_info in significant_changes[:5]:  # Show max 5
                symbol = change_info['symbol']
                pred_id = change_info['prediction_id']
                pred_action = change_info['action']
                changes = change_info['changes']
                recommendation = changes.get('recommendation', {})
                new_action = recommendation.get('action', 'HOLD')
                
                if pred_action != new_action:
                    message += f"  #{pred_id} - {symbol}: {pred_action} → {new_action}\n"
                else:
                    message += f"  #{pred_id} - {symbol}: Trend reversal detected\n"
            
            if len(significant_changes) > 5:
                message += f"\n... and {len(significant_changes) - 5} more\n"
            
            message += "\nView details in the 🔄 Momentum Changes tab."
            messagebox.showinfo(title, message)
            logger.info(f"Showed batched momentum change notification for {len(significant_changes)} predictions")
    
    def _show_verification_results(self, result: dict, newly_verified: list = None):
        """Show prediction verification results"""
        if result['verified'] > 0:
            accuracy = result['accuracy']
            message = f"Verified {result['verified']} prediction(s). "
            message += f"Accuracy: {accuracy:.1f}% ({result['correct']}/{result['verified']} correct)"
            messagebox.showinfo("Predictions Verified", message)
            
            # Update performance tracking for hybrid system
            if newly_verified:
                self._update_performance_tracking(newly_verified)
            
            # Update ML status to show improvements
            self._update_ml_status()
            
            self._update_predictions_display()
    
    def _update_performance_tracking(self, verified_predictions: list):
        """Update performance tracking for hybrid system"""
        try:
            for pred in verified_predictions:
                # Get the original prediction to find which predictor was used
                pred_id = pred.get('id')
                original_pred = self.predictions_tracker.get_prediction_by_id(pred_id)
                
                if original_pred:
                    strategy = original_pred.get('strategy', self.current_strategy)
                    hybrid_predictor = self.hybrid_predictors.get(strategy)
                    
                    if hybrid_predictor:
                        # Create prediction dict for performance tracking
                        prediction_dict = {
                            'method': 'hybrid_ensemble',
                            'rule_prediction': {
                                'action': original_pred.get('action', 'HOLD'),
                                'confidence': original_pred.get('confidence', 50)
                            },
                            'ml_prediction': {
                                'action': original_pred.get('action', 'HOLD'),
                                'confidence': original_pred.get('confidence', 50)
                            },
                            'recommendation': {
                                'confidence': original_pred.get('confidence', 50)
                            }
                        }
                        
                        hybrid_predictor.update_performance(
                            prediction_dict,
                            pred.get('was_correct', False)
                        )
        except Exception as e:
            logger.error(f"Error updating performance tracking: {e}")
    
    def _trigger_auto_training(self):
        """Trigger automatic training on verified predictions"""
        def train_in_background():
            try:
                self.auto_training.check_and_train()
            except Exception as e:
                logger.error(f"Error triggering auto-training: {e}")
        
        thread = threading.Thread(target=train_in_background)
        thread.daemon = True
        thread.start()
    
    def _schedule_auto_training(self):
        """Schedule automatic training checks"""
        # Check every 6 hours
        check_interval_ms = 6 * 3600 * 1000  # 6 hours
        
        def periodic_check():
            self.auto_training.check_and_train()
            # Schedule next check
            self.root.after(check_interval_ms, periodic_check)
        
        # Start first check after 1 hour
        self.root.after(3600 * 1000, periodic_check)
    
    def _show_verification_notifications(self, newly_verified: list):
        """Show batched notifications for newly verified predictions"""
        if not newly_verified:
            return
        
        # Check notification preferences
        show_individual = self.preferences.get('show_individual_verification_notifications', False)
        show_batched = self.preferences.get('show_batched_verification_notifications', True)
        
        if show_individual:
            # Show individual notifications (old behavior)
            for pred in newly_verified:
                symbol = pred['symbol']
                action = pred['action']
                was_correct = pred.get('was_correct', False)
                actual_price = pred.get('actual_price_at_target', 0)
                
                # Create notification message
                status_icon = "✓" if was_correct else "✗"
                status_text = "CORRECT" if was_correct else "INCORRECT"
                
                message = f"{status_icon} Prediction #{pred['id']} - {symbol} ({action})\n\n"
                message += f"Status: {status_text}\n"
                message += f"Target: {self.localization.format_currency(pred['target_price'])}\n"
                message += f"Actual Price: {self.localization.format_currency(actual_price)}\n"
                
                if was_correct:
                    message += f"\n🎉 Your prediction was correct!"
                else:
                    message += f"\n❌ Your prediction did not come true."
                
                # Show notification
                title = f"Prediction Verified - {symbol}"
                messagebox.showinfo(title, message)
        elif show_batched and len(newly_verified) > 0:
            # Show batched notification (new default behavior)
            correct_count = sum(1 for p in newly_verified if p.get('was_correct', False))
            incorrect_count = len(newly_verified) - correct_count
            
            title = "Predictions Verified"
            message = f"✅ {len(newly_verified)} prediction(s) verified:\n\n"
            message += f"✓ Correct: {correct_count}\n"
            message += f"✗ Incorrect: {incorrect_count}\n\n"
            
            if len(newly_verified) <= 5:
                # Show details for small batches
                message += "Details:\n"
                for pred in newly_verified[:5]:
                    status_icon = "✓" if pred.get('was_correct', False) else "✗"
                    message += f"  {status_icon} #{pred['id']} - {pred['symbol']} ({pred['action']})\n"
            else:
                # Show summary for large batches
                message += f"View details in the Predictions tab."
            
            messagebox.showinfo(title, message)
        
        # Also update the predictions display
        self._update_predictions_display()
    
    def _apply_theme(self, theme_name: str):
        """Apply a theme to the entire application with full UI refresh"""
        self.current_theme = theme_name
        self.theme_manager.set_theme(theme_name)
        self.preferences.set_theme(theme_name)
        theme = self.theme_manager.get_theme(theme_name)
        
        # Update loading screen theme
        if self.loading_screen:
            self.loading_screen.theme = theme
        
        # Update dropdowns if they exist
        if hasattr(self, 'theme_dropdown'):
            self.theme_dropdown.set_value(theme_name)
        
        # Apply theme to root
        self.root.config(bg=theme['bg'])
        
        # Recreate UI with new theme (simplified approach - full recreation would be better)
        try:
            style = ttk.Style()
            style.theme_use('clam')
            
            # Configure ttk styles with modern colors
            style.configure('TFrame', background=theme['frame_bg'])
            style.configure('TLabelFrame', background=theme['frame_bg'], foreground=theme['fg'],
                          borderwidth=0, relief=tk.FLAT)
            style.configure('TLabel', background=theme['frame_bg'], foreground=theme['fg'])
            style.configure('TButton', background=theme['button_bg'], foreground=theme['button_fg'],
                          borderwidth=0, relief=tk.FLAT, padding=10)
            style.map('TButton', 
                     background=[('active', theme['button_hover']), ('pressed', theme['accent_hover'])])
            style.configure('TEntry', fieldbackground=theme['entry_bg'], foreground=theme['entry_fg'],
                          borderwidth=0, relief=tk.FLAT)
            style.configure('TNotebook', background=theme['frame_bg'], borderwidth=0)
            style.configure('TNotebook.Tab', background=theme['secondary_bg'], foreground=theme['fg'],
                          padding=[20, 10], borderwidth=0)
            style.map('TNotebook.Tab', 
                     background=[('selected', theme['frame_bg'])],
                     expand=[('selected', [1, 1, 1, 0])])
            style.configure('TRadiobutton', background=theme['frame_bg'], foreground=theme['fg'],
                          selectcolor=theme['accent'])
            style.configure('TScrollbar', background=theme['secondary_bg'], troughcolor=theme['bg'],
                          borderwidth=0, arrowcolor=theme['fg'], darkcolor=theme['secondary_bg'],
                          lightcolor=theme['secondary_bg'])
            style.map('TScrollbar', background=[('active', theme['accent'])])
            
            # Update matplotlib style
            if theme_name == "dark" or theme_name == "black":
                plt.style.use('dark_background')
            else:
                plt.style.use('seaborn-v0_8-whitegrid')
            
            # Update all existing widgets if they exist
            self._update_widgets_theme(theme)
            
            logger.info(f"Theme changed to: {theme_name}")
        except Exception as e:
            logger.error(f"Error applying theme: {e}")
    
    def _update_widgets_theme(self, theme: dict):
        """Update existing widgets with new theme colors"""
        # This would update all widgets, but for simplicity we'll just update key ones
        try:
            if hasattr(self, 'analysis_text'):
                self.analysis_text.config(bg=theme['frame_bg'], fg=theme['fg'],
                                        insertbackground=theme['fg'])
            if hasattr(self, 'potential_text'):
                self.potential_text.config(bg=theme['frame_bg'], fg=theme['fg'],
                                         insertbackground=theme['fg'])
            if hasattr(self, 'symbol_entry'):
                self.symbol_entry.config(bg=theme['entry_bg'], fg=theme['entry_fg'],
                                       insertbackground=theme['fg'])
            if hasattr(self, 'budget_entry'):
                self.budget_entry.config(bg=theme['entry_bg'], fg=theme['entry_fg'],
                                       insertbackground=theme['fg'])
        except Exception as e:
            logger.warning(f"Could not update all widgets: {e}")
    
    def _create_ui(self):
        """Create the modern, visually stunning user interface"""
        # Apply saved theme
        self._apply_theme(self.current_theme)
        theme = self.theme_manager.get_theme()
        
        # Initialize loading screen
        self.loading_screen = LoadingScreen(self.root, theme)
        
        # Main container - full window background
        self.root.config(bg=theme['bg'])
        main_container = tk.Frame(self.root, bg=theme['bg'])
        main_container.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        # Modern top bar with gradient effect
        top_bar = tk.Frame(main_container, bg=theme['secondary_bg'], height=70)
        top_bar.pack(fill=tk.X, padx=0, pady=0)
        top_bar.pack_propagate(False)
        
        # App title with modern styling
        title_frame = tk.Frame(top_bar, bg=theme['secondary_bg'])
        title_frame.pack(side=tk.LEFT, padx=25, pady=15)
        
        self.title_label = GradientLabel(title_frame, self.localization.t('app_title'), theme=theme, 
                     font=('Segoe UI', 24, 'bold'))
        self.title_label.pack(side=tk.LEFT)
        self.subtitle_label = tk.Label(title_frame, text=self.localization.t('app_subtitle'), 
                bg=theme['secondary_bg'], fg=theme['text_secondary'],
                font=('Segoe UI', 10))
        self.subtitle_label.pack(side=tk.LEFT, padx=(15, 0))
        
        # Settings frame - Theme, Language, Currency (Modern Dropdowns)
        settings_frame = tk.Frame(top_bar, bg=theme['secondary_bg'])
        settings_frame.pack(side=tk.RIGHT, padx=25, pady=15)
        
        # About button
        about_btn = tk.Button(settings_frame,
                            text="ℹ️ About",
                            command=self._show_about,
                            bg=theme['accent'],
                            fg=theme['button_fg'],
                            font=('Segoe UI', 9, 'bold'),
                            relief=tk.FLAT,
                            padx=15,
                            pady=8,
                            cursor='hand2',
                            activebackground=theme['accent_hover'],
                            activeforeground=theme['button_fg'])
        about_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Modern dropdown buttons
        self.theme_dropdown = ModernDropdownButton(
            settings_frame, 
            self.localization.t('theme'),
            ["light", "dark", "black"],
            self.current_theme,
            self._apply_theme,
            theme
        )
        self.theme_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        
        self.language_dropdown = ModernDropdownButton(
            settings_frame,
            self.localization.t('language'),
            ["en", "bg"],
            self.saved_language,
            self._change_language,
            theme
        )
        self.language_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        
        self.currency_dropdown = ModernDropdownButton(
            settings_frame,
            self.localization.t('currency'),
            ["USD", "EUR", "BGN"],
            self.saved_currency,
            self._change_currency,
            theme
        )
        self.currency_dropdown.pack(side=tk.LEFT, padx=(0, 10))
        
        # Close button
        self.close_btn = ModernButton(
            settings_frame,
            "✕ " + self.localization.t('close'),
            command=self._close_app,
            theme=theme
        )
        self.close_btn.pack(side=tk.LEFT)
        
        # Main content area
        content_frame = tk.Frame(main_container, bg=theme['bg'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Left sidebar - Modern cards with scrollbar
        sidebar_container = tk.Frame(content_frame, bg=theme['bg'], width=340)
        sidebar_container.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 20))
        sidebar_container.pack_propagate(False)
        
        # Create canvas and modern scrollbar for scrollable sidebar
        sidebar_canvas = tk.Canvas(sidebar_container, bg=theme['bg'], highlightthickness=0, width=320)
        sidebar = tk.Frame(sidebar_canvas, bg=theme['bg'], width=320)
        
        sidebar_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Modern scrollbar - make it visible
        sidebar_scrollbar = ModernScrollbar(sidebar_container, command=sidebar_canvas.yview, theme=theme)
        sidebar_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        # Force initial update
        sidebar_canvas.update_idletasks()
        sidebar_scrollbar._update_slider()
        
        # Configure scrollbar
        sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)
        sidebar_canvas.create_window((0, 0), window=sidebar, anchor=tk.NW)
        
        def configure_scroll_region(event=None):
            sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))
        
        sidebar.bind('<Configure>', configure_scroll_region)
        sidebar_canvas.bind('<Configure>', lambda e: sidebar_canvas.itemconfig(sidebar_canvas.find_all()[0], width=e.width) if sidebar_canvas.find_all() else None)
        
        # Bind mousewheel to entire sidebar container for smooth scrolling
        def on_sidebar_mousewheel(event):
            # Windows uses delta, Linux uses different values
            delta = event.delta if hasattr(event, 'delta') else (event.num == 4 and -1) or 1
            if isinstance(delta, (int, float)):
                # Smooth scrolling - use smaller increments
                scroll_amount = int(-1 * (delta / 120)) * 3  # 3 units for smoother scroll
                sidebar_canvas.yview_scroll(scroll_amount, "units")
            return "break"  # Prevent event propagation
        
        # Bind to all levels for maximum compatibility - with return "break" to prevent conflicts
        sidebar_container.bind("<MouseWheel>", on_sidebar_mousewheel)
        sidebar_canvas.bind("<MouseWheel>", on_sidebar_mousewheel)
        sidebar.bind("<MouseWheel>", on_sidebar_mousewheel)
        
        # Also bind to all child widgets in sidebar
        def bind_sidebar_children(parent):
            for child in parent.winfo_children():
                if isinstance(child, (tk.Frame, tk.Canvas, tk.Label, tk.Button)):
                    child.bind("<MouseWheel>", on_sidebar_mousewheel)
                    if isinstance(child, tk.Frame):
                        bind_sidebar_children(child)
        
        bind_sidebar_children(sidebar)
        
        # Linux/Unix support
        sidebar_container.bind("<Button-4>", lambda e: sidebar_canvas.yview_scroll(-3, "units") or "break")
        sidebar_container.bind("<Button-5>", lambda e: sidebar_canvas.yview_scroll(3, "units") or "break")
        sidebar_canvas.bind("<Button-4>", lambda e: sidebar_canvas.yview_scroll(-3, "units") or "break")
        sidebar_canvas.bind("<Button-5>", lambda e: sidebar_canvas.yview_scroll(3, "units") or "break")
        sidebar.bind("<Button-4>", lambda e: sidebar_canvas.yview_scroll(-3, "units") or "break")
        sidebar.bind("<Button-5>", lambda e: sidebar_canvas.yview_scroll(3, "units") or "break")
        
        # Store reference for later cleanup
        self.sidebar_container = sidebar_container
        self.sidebar_canvas = sidebar_canvas
        
        # Strategy Selection Card
        self.strategy_card = ModernCard(sidebar, self.localization.t('strategy'), theme=theme, padding=20)
        self.strategy_card.pack(fill=tk.X, pady=(0, 15))
        
        self.strategy_var = tk.StringVar(value=self.saved_strategy)
        strategy_inner = tk.Frame(self.strategy_card.content_frame, bg=theme['card_bg'])
        strategy_inner.pack(fill=tk.X)
        
        # Store strategy buttons for visual feedback
        self.strategy_buttons = {}
        
        # Modern radio buttons with visual selection feedback
        self.trading_btn_frame = tk.Frame(strategy_inner, bg=theme['card_bg'], relief=tk.FLAT, bd=0)
        self.trading_btn_frame.pack(fill=tk.X, pady=4)
        self.trading_btn = tk.Radiobutton(self.trading_btn_frame, text=f"⚡ {self.localization.t('trading')}", 
                                    variable=self.strategy_var, value="trading",
                                    command=self._on_strategy_change,
                                    bg=theme['card_bg'], fg=theme['fg'],
                                    font=('Segoe UI', 11, 'bold'), selectcolor=theme['accent'],
                                    activebackground=theme['card_bg'],
                                    activeforeground=theme['accent'])
        self.trading_btn.pack(anchor=tk.W, padx=10, pady=8)
        self.strategy_buttons['trading'] = self.trading_btn_frame
        
        self.mixed_btn_frame = tk.Frame(strategy_inner, bg=theme['card_bg'], relief=tk.FLAT, bd=0)
        self.mixed_btn_frame.pack(fill=tk.X, pady=4)
        self.mixed_btn = tk.Radiobutton(self.mixed_btn_frame, text=f"🔄 {self.localization.t('mixed')}", 
                                       variable=self.strategy_var, value="mixed",
                                       command=self._on_strategy_change,
                                       bg=theme['card_bg'], fg=theme['fg'],
                                       font=('Segoe UI', 11, 'bold'), selectcolor=theme['accent'],
                                       activebackground=theme['card_bg'],
                                       activeforeground=theme['accent'])
        self.mixed_btn.pack(anchor=tk.W, padx=10, pady=8)
        self.strategy_buttons['mixed'] = self.mixed_btn_frame
        
        # Add description label for mixed
        mixed_desc = tk.Label(self.mixed_btn_frame, 
                             text=f"  ({self.localization.t('mixed_description')})",
                             bg=theme['card_bg'], fg=theme['text_secondary'],
                             font=('Segoe UI', 8))
        mixed_desc.pack(anchor=tk.W, padx=(35, 0), pady=(0, 8))
        
        self.investing_btn_frame = tk.Frame(strategy_inner, bg=theme['card_bg'], relief=tk.FLAT, bd=0)
        self.investing_btn_frame.pack(fill=tk.X, pady=4)
        self.investing_btn = tk.Radiobutton(self.investing_btn_frame, text=f"📈 {self.localization.t('investing')}", 
                                       variable=self.strategy_var, value="investing",
                                       command=self._on_strategy_change,
                                       bg=theme['card_bg'], fg=theme['fg'],
                                       font=('Segoe UI', 11, 'bold'), selectcolor=theme['accent'],
                                       activebackground=theme['card_bg'],
                                       activeforeground=theme['accent'])
        self.investing_btn.pack(anchor=tk.W, padx=10, pady=8)
        self.strategy_buttons['investing'] = self.investing_btn_frame
        
        # Update visual feedback for saved strategy
        # Update visual feedback for saved strategy
        self._update_strategy_visual_feedback()
        
        # Scan Market Button (New Feature)
        self.sidebar_scan_frame = tk.Frame(self.strategy_card.content_frame, bg=theme['card_bg'])
        self.sidebar_scan_frame.pack(fill=tk.X, pady=(15, 0))
        
        self.sidebar_scan_btn = ModernButton(
            self.sidebar_scan_frame, 
            "🔍 Analyze Predictions", 
            command=self._scan_market,
            theme=theme
        )
        self.sidebar_scan_btn.pack(fill=tk.X)
        
        # Market Dip Scanner Button (New Feature)
        self.sidebar_dip_frame = tk.Frame(self.strategy_card.content_frame, bg=theme['card_bg'])
        self.sidebar_dip_frame.pack(fill=tk.X, pady=(8, 0))
        
        self.sidebar_dip_btn = ModernButton(
            self.sidebar_dip_frame, 
            "📉 Find Market Dips", 
            command=self._scan_for_dips,
            theme=theme
        )
        self.sidebar_dip_btn.pack(fill=tk.X)

        
        # Budget Card
        self.budget_card = ModernCard(sidebar, self.localization.t('investment_budget'), theme=theme, padding=20)
        self.budget_card.pack(fill=tk.X, pady=(0, 15))
        
        currency_symbol = self.localization.format_currency_symbol()
        self.amount_label = tk.Label(self.budget_card.content_frame, 
                text=f"{self.localization.t('amount')} ({currency_symbol})", bg=theme['card_bg'],
                fg=theme['text_secondary'], font=('Segoe UI', 9))
        self.amount_label.pack(anchor=tk.W, pady=(0, 8))
        
        budget_entry_frame = tk.Frame(self.budget_card.content_frame, bg=theme['card_bg'])
        budget_entry_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.budget_entry = tk.Entry(budget_entry_frame, font=('Segoe UI', 12),
                                     bg=theme['entry_bg'], fg=theme['entry_fg'],
                                     relief=tk.FLAT, bd=0, insertbackground=theme['fg'])
        self.budget_entry.pack(fill=tk.X, ipady=10)
        # Load saved budget
        saved_budget = self.preferences.get_budget()
        self.budget_entry.insert(0, str(saved_budget))
        # Save budget when changed
        self.budget_entry.bind('<FocusOut>', lambda e: self._save_budget())
        self.budget_entry.bind('<Return>', lambda e: self._save_budget())
        
        self.budget_underline = tk.Frame(budget_entry_frame, bg=theme['border'], height=2)
        self.budget_underline.pack(fill=tk.X, pady=(5, 0))
        
        def on_budget_focus_in(e):
            self.budget_underline.config(bg=theme['accent'], height=2)
        def on_budget_focus_out(e):
            self.budget_underline.config(bg=theme['border'], height=2)
        
        self.budget_entry.bind('<FocusIn>', on_budget_focus_in)
        self.budget_entry.bind('<FocusOut>', on_budget_focus_out)
        
        self.calc_btn = ModernButton(self.budget_card.content_frame, self.localization.t('calculate_potential'), 
                               command=self._calculate_potential, theme=theme)
        self.calc_btn.pack(fill=tk.X)
        
        # Portfolio Stats Card
        self.portfolio_card = ModernCard(sidebar, self.localization.t('portfolio'), theme=theme, padding=20)
        self.portfolio_card.pack(fill=tk.X, pady=(0, 15))
        
        self.portfolio_label = tk.Label(self.portfolio_card.content_frame, 
                                       text=self.localization.format_currency(0), bg=theme['card_bg'],
                                       fg=theme['accent'], font=('Segoe UI', 20, 'bold'))
        self.portfolio_label.pack(anchor=tk.W, pady=(0, 10))
        
        self.stats_label = tk.Label(self.portfolio_card.content_frame, 
                                   text=f"{self.localization.t('wins')}: 0 | {self.localization.t('losses')}: 0", bg=theme['card_bg'],
                                   fg=theme['text_secondary'], font=('Segoe UI', 10))
        self.stats_label.pack(anchor=tk.W, pady=(0, 10))
        
        # S&P 500 Comparison Label
        self.sp500_label = tk.Label(self.portfolio_card.content_frame, 
                                   text="S&P 500: Loading...", bg=theme['card_bg'],
                                   fg=theme['text_secondary'], font=('Segoe UI', 9))
        self.sp500_label.pack(anchor=tk.W, pady=(0, 15))
        
        self.set_balance_btn = ModernButton(self.portfolio_card.content_frame, self.localization.t('set_balance'), 
                    command=self._set_balance, theme=theme)
        self.set_balance_btn.pack(fill=tk.X)
        
        # Predictions Stats Card
        self.pred_card = ModernCard(sidebar, self.localization.t('predictions'), theme=theme, padding=20)
        self.pred_card.pack(fill=tk.X)
        
        self.pred_stats_label = tk.Label(self.pred_card.content_frame, 
                                        text=f"{self.localization.t('accuracy')}: N/A", bg=theme['card_bg'],
                                        fg=theme['accent'], font=('Segoe UI', 14, 'bold'))
        self.pred_stats_label.pack(anchor=tk.W, pady=(0, 15))
        
        self.view_all_btn = ModernButton(self.pred_card.content_frame, self.localization.t('view_all'), 
                    command=self._show_predictions_tab, theme=theme)
        self.view_all_btn.pack(fill=tk.X)
        
        # AI Training Status Card
        self.ml_status_card = ModernCard(sidebar, "🤖 AI Training Status", theme=theme, padding=20)
        self.ml_status_card.pack(fill=tk.X, pady=(0, 15))
        
        # Status label with detailed stats
        self.ml_status_label = tk.Label(
            self.ml_status_card.content_frame,
            text="Checking AI status...",
            bg=theme['card_bg'],
            fg=theme['text_secondary'],
            font=('Segoe UI', 9),
            justify=tk.LEFT,
            wraplength=280
        )
        self.ml_status_label.pack(anchor=tk.W, pady=(0, 8))
        
        # Performance stats labels
        self.ml_stats_frame = tk.Frame(self.ml_status_card.content_frame, bg=theme['card_bg'])
        self.ml_stats_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.ml_accuracy_label = tk.Label(
            self.ml_stats_frame,
            text="",
            bg=theme['card_bg'],
            fg=theme['accent'],
            font=('Segoe UI', 10, 'bold')
        )
        self.ml_accuracy_label.pack(anchor=tk.W)
        
        self.ml_rule_label = tk.Label(
            self.ml_stats_frame,
            text="",
            bg=theme['card_bg'],
            fg=theme['text_secondary'],
            font=('Segoe UI', 8)
        )
        self.ml_rule_label.pack(anchor=tk.W)
        
        self.ml_hybrid_label = tk.Label(
            self.ml_stats_frame,
            text="",
            bg=theme['card_bg'],
            fg=theme['text_secondary'],
            font=('Segoe UI', 8)
        )
        self.ml_hybrid_label.pack(anchor=tk.W)
        
        # Learning progress bar
        self.learning_progress_frame = tk.Frame(self.ml_status_card.content_frame, bg=theme['card_bg'])
        self.learning_progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.learning_progress_label = tk.Label(
            self.learning_progress_frame,
            text="Learning Progress:",
            bg=theme['card_bg'],
            fg=theme['text_secondary'],
            font=('Segoe UI', 8)
        )
        self.learning_progress_label.pack(anchor=tk.W)
        
        self.learning_progress_bar = ttk.Progressbar(
            self.learning_progress_frame,
            mode='determinate',
            length=280,
            maximum=100
        )
        self.learning_progress_bar.pack(fill=tk.X, pady=(5, 0))
        
        # Megamind Status (Learning Tracker)
        megamind_frame = tk.Frame(self.ml_status_card.content_frame, bg=theme['card_bg'])
        megamind_frame.pack(fill=tk.X, pady=(10, 0))
        
        megamind_header = tk.Label(
            megamind_frame,
            text="🧠 MEGAMIND STATUS",
            bg=theme['card_bg'],
            fg=theme['accent'],
            font=('Segoe UI', 9, 'bold')
        )
        megamind_header.pack(anchor=tk.W)
        
        self.megamind_status_label = tk.Label(
            megamind_frame,
            text="Initializing...",
            bg=theme['card_bg'],
            fg=theme['text_secondary'],
            font=('Segoe UI', 7),
            wraplength=280,
            justify=tk.LEFT
        )
        self.megamind_status_label.pack(anchor=tk.W, pady=(3, 0))
        
        # Auto-learner status
        auto_learner_frame = tk.Frame(self.ml_status_card.content_frame, bg=theme['card_bg'])
        auto_learner_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.auto_learner_status_label = tk.Label(
            auto_learner_frame,
            text="🤖 Auto-Learner: Starting...",
            bg=theme['card_bg'],
            fg=theme['text_secondary'],
            font=('Segoe UI', 8),
            wraplength=280,
            justify=tk.LEFT
        )
        self.auto_learner_status_label.pack(anchor=tk.W)
        
        # Button frame
        btn_frame = tk.Frame(self.ml_status_card.content_frame, bg=theme['card_bg'])
        btn_frame.pack(fill=tk.X)
        
        self.train_btn = ModernButton(
            btn_frame,
            "Train AI Model",
            command=self._show_training_dialog,
            theme=theme
        )
        self.train_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.view_stats_btn = ModernButton(
            btn_frame,
            "View Detailed Stats",
            command=self._show_detailed_stats,
            theme=theme
        )
        self.view_stats_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.test_model_btn = ModernButton(
            btn_frame,
            "Test Model",
            command=self._show_test_dialog,
            theme=theme
        )
        self.test_model_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.backtest_btn = ModernButton(
            btn_frame,
            "🧪 Run Backtest",
            command=self._run_backtest,
            theme=theme
        )
        self.backtest_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Continuous backtesting buttons
        self.continuous_backtest_btn = ModernButton(
            btn_frame,
            "🔄 Start Continuous Backtesting",
            command=self._start_continuous_backtest,
            theme=theme
        )
        self.continuous_backtest_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.stop_continuous_backtest_btn = ModernButton(
            btn_frame,
            "⏹️ Stop Continuous Backtesting",
            command=self._stop_continuous_backtest,
            theme=theme
        )
        self.stop_continuous_backtest_btn.pack(fill=tk.X, pady=(0, 5))
        self.stop_continuous_backtest_btn.config(state=tk.DISABLED)  # Disabled until started
        
        # Continuous backtesting status label
        self.continuous_backtest_status = tk.Label(
            btn_frame,
            text="",
            bg=theme['bg'],
            fg=theme['fg'],
            font=('Segoe UI', 9),
            wraplength=200
        )
        self.continuous_backtest_status.pack(fill=tk.X, pady=(5, 0))
        
        self.view_backtest_results_btn = ModernButton(
            btn_frame,
            "📊 View Backtest Results",
            command=self._show_backtest_results,
            theme=theme
        )
        self.view_backtest_results_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Scan Now button for immediate learning
        self.scan_now_btn = ModernButton(
            btn_frame,
            "🔍 Scan Market Now",
            command=self._trigger_manual_scan,
            theme=theme
        )
        self.scan_now_btn.pack(fill=tk.X, pady=(0, 5))
        
        # Auto-Learner Settings button
        self.auto_learner_settings_btn = ModernButton(
            btn_frame,
            "⚙️ Auto-Learner Settings",
            command=self._show_auto_learner_settings,
            theme=theme
        )
        self.auto_learner_settings_btn.pack(fill=tk.X)
        
        # Store previous accuracy for improvement detection
        self.previous_accuracy = {}
        
        # Update ML status on startup
        self.root.after(100, self._update_ml_status)
        
        # Update test button state based on model availability
        self.root.after(200, self._update_test_button_state)
        
        # Right panel - Results area
        results_panel = tk.Frame(content_frame, bg=theme['bg'])
        results_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Results header
        header_card = ModernCard(results_panel, "", theme=theme, padding=20)
        header_card.pack(fill=tk.X, pady=(0, 15))
        
        self.results_title = GradientLabel(header_card.content_frame, 
                                          self.localization.t('stock_analysis_results'), theme=theme,
                                          font=('Segoe UI', 18, 'bold'))
        self.results_title.pack(side=tk.LEFT)
        
        # Modern notebook with custom styling
        notebook_container = tk.Frame(results_panel, bg=theme['bg'])
        notebook_container.pack(fill=tk.BOTH, expand=True)
        
        self.notebook = TwoRowNotebook(notebook_container, bg=theme['bg'])
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Bind mousewheel to notebook for smooth scrolling in tabs
        def on_notebook_mousewheel(event):
            # Get the currently selected tab
            try:
                selected_tab = self.notebook.nametowidget(self.notebook.select())
                
                # Check if the tab has scrollable content
                delta = event.delta if hasattr(event, 'delta') else (event.num == 4 and -1) or 1
                if isinstance(delta, (int, float)):
                    scroll_amount = int(-1 * (delta / 120)) * 3
                    
                    # Try to scroll the active tab's content
                    for widget in selected_tab.winfo_children():
                        if isinstance(widget, tk.Canvas):
                            widget.yview_scroll(scroll_amount, "units")
                            return "break"
                        elif isinstance(widget, scrolledtext.ScrolledText):
                            widget.yview_scroll(scroll_amount, "units")
                            return "break"
            except (AttributeError, tk.TclError) as e:
                logger.debug(f"Error handling notebook mousewheel: {e}")
        
        notebook_container.bind("<MouseWheel>", on_notebook_mousewheel)
        self.notebook.bind("<MouseWheel>", on_notebook_mousewheel)
        
        # Universal handler that checks mouse position and routes to correct area
        # Defined here so it can reference both sidebar and notebook handlers
        def universal_wheel_handler(event):
            try:
                x, y = event.x_root, event.y_root
                
                # Check sidebar first (with some padding for better detection)
                try:
                    sidebar_x = sidebar_container.winfo_rootx()
                    sidebar_y = sidebar_container.winfo_rooty()
                    sidebar_w = sidebar_container.winfo_width()
                    sidebar_h = sidebar_container.winfo_height()
                    
                    if (sidebar_x <= x <= sidebar_x + sidebar_w and 
                        sidebar_y <= y <= sidebar_y + sidebar_h):
                        on_sidebar_mousewheel(event)
                        return "break"
                except (AttributeError, tk.TclError) as e:
                    logger.debug(f"Error checking sidebar bounds: {e}")
                
                # Check notebook area
                try:
                    notebook_x = notebook_container.winfo_rootx()
                    notebook_y = notebook_container.winfo_rooty()
                    notebook_w = notebook_container.winfo_width()
                    notebook_h = notebook_container.winfo_height()
                    
                    if (notebook_x <= x <= notebook_x + notebook_w and 
                        notebook_y <= y <= notebook_y + notebook_h):
                        on_notebook_mousewheel(event)
                        return "break"
                except (AttributeError, tk.TclError) as e:
                    logger.debug(f"Error checking notebook bounds: {e}")
            except (AttributeError, tk.TclError, ValueError) as e:
                logger.debug(f"Error in universal wheel handler: {e}")
        
        # Bind to parent containers and root window to catch events when hovering
        content_frame.bind("<MouseWheel>", universal_wheel_handler)
        results_panel.bind("<MouseWheel>", universal_wheel_handler)
        # Also bind to root window to catch all mousewheel events
        self.root.bind("<MouseWheel>", universal_wheel_handler)
        
        # Store notebook container reference
        self.notebook_container = notebook_container
        
        # Analysis tab
        self.analysis_frame = tk.Frame(self.notebook, bg=theme['frame_bg'])
        self.notebook.add(self.analysis_frame, text=self.localization.t('analysis'))
        
        self.analysis_text = scrolledtext.ScrolledText(self.analysis_frame, 
                                                      wrap=tk.WORD, height=20,
                                                      font=('Segoe UI', 10),
                                                      bg=theme['frame_bg'], fg=theme['fg'],
                                                      relief=tk.FLAT, bd=0,
                                                      insertbackground=theme['fg'])
        self.analysis_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Chart tab
        self.chart_frame = tk.Frame(self.notebook, bg=theme['frame_bg'])
        self.notebook.add(self.chart_frame, text=self.localization.t('charts'))
        
        self.chart_canvas = None
        
        # Indicators tab
        self.indicators_frame = tk.Frame(self.notebook, bg=theme['frame_bg'])
        self.notebook.add(self.indicators_frame, text=self.localization.t('indicators'))
        
        self.indicators_canvas = None
        
        # Potential tab
        self.potential_frame = tk.Frame(self.notebook, bg=theme['frame_bg'])
        self.notebook.add(self.potential_frame, text=self.localization.t('potential_trade'))
        
        # Gun tab will be created later (after predictions tab setup)
        
        self.potential_text = scrolledtext.ScrolledText(self.potential_frame, 
                                                       wrap=tk.WORD, height=20,
                                                       font=('Segoe UI', 10),
                                                       bg=theme['frame_bg'], fg=theme['fg'],
                                                       relief=tk.FLAT, bd=0,
                                                       insertbackground=theme['fg'])
        self.potential_text.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Seeker AI tab
        self.seeker_ai_frame = tk.Frame(self.notebook, bg=theme['frame_bg'])
        self.notebook.add(self.seeker_ai_frame, text="🔍 Seeker AI")
        
        # Create scrollable text area for Seeker AI results
        seeker_ai_container = tk.Frame(self.seeker_ai_frame, bg=theme['frame_bg'])
        seeker_ai_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        self.seeker_ai_text = scrolledtext.ScrolledText(seeker_ai_container,
                                                        wrap=tk.WORD, height=20,
                                                        font=('Segoe UI', 10),
                                                        bg=theme['frame_bg'], fg=theme['fg'],
                                                        relief=tk.FLAT, bd=0,
                                                        insertbackground=theme['fg'])
        self.seeker_ai_text.pack(fill=tk.BOTH, expand=True)
        
        # Add placeholder text
        placeholder = "🔍 Seeker AI Analysis\n\n"
        placeholder += "Seeker AI analysis will appear here after analyzing a stock.\n\n"
        placeholder += "Seeker AI provides:\n"
        placeholder += "• News sentiment analysis\n"
        placeholder += "• Company reputation insights\n"
        placeholder += "• Price movement explanations\n"
        placeholder += "• AI-generated summaries\n"
        placeholder += "• Risk factors and opportunities\n"
        placeholder += "• Recent news articles"
        self.seeker_ai_text.insert('1.0', placeholder)
        
        # Risk Management tab
        self.risk_management_frame = tk.Frame(self.notebook, bg=theme['frame_bg'])
        self.notebook.add(self.risk_management_frame, text="🛡️ Risk Management")
        self._create_risk_management_tab()
        
        # Advanced Analytics tab
        self.advanced_analytics_frame = tk.Frame(self.notebook, bg=theme['frame_bg'])
        self.notebook.add(self.advanced_analytics_frame, text="📊 Advanced Analytics")
        self._create_advanced_analytics_tab()
        
        # Walk-Forward Backtesting tab
        self.walk_forward_frame = tk.Frame(self.notebook, bg=theme['frame_bg'])
        self.notebook.add(self.walk_forward_frame, text="🔄 Walk-Forward")
        self._create_walk_forward_tab()
        
        # Alert System tab
        self.alerts_frame = tk.Frame(self.notebook, bg=theme['frame_bg'])
        self.notebook.add(self.alerts_frame, text="🔔 Alerts")
        self._create_alerts_tab()
        
        # My Holdings tab
        self.holdings_frame = tk.Frame(self.notebook, bg=theme['frame_bg'])
        self.notebook.add(self.holdings_frame, text="💼 My Holdings")
        self._create_holdings_tab()
        
        # Bind mousewheel to potential text for smooth scrolling
        def on_potential_scroll(event):
            delta = event.delta if hasattr(event, 'delta') else (event.num == 4 and -1) or 1
            if isinstance(delta, (int, float)):
                scroll_amount = int(-1 * (delta / 120)) * 3
                self.potential_text.yview_scroll(scroll_amount, "units")
        self.potential_frame.bind("<MouseWheel>", on_potential_scroll)
        self.potential_text.bind("<MouseWheel>", on_potential_scroll)
        
        # Bind mousewheel to all tab frames (after they're all created)
        def bind_tab_scrolling(frame):
            """Recursively bind mousewheel to all widgets in a frame"""
            frame.bind("<MouseWheel>", on_notebook_mousewheel)
            for child in frame.winfo_children():
                if isinstance(child, (tk.Frame, tk.Canvas, scrolledtext.ScrolledText)):
                    child.bind("<MouseWheel>", on_notebook_mousewheel)
                    if isinstance(child, tk.Frame):
                        bind_tab_scrolling(child)
        
        # Bind scrolling to all tab frames
        bind_tab_scrolling(self.analysis_frame)
        bind_tab_scrolling(self.chart_frame)
        bind_tab_scrolling(self.indicators_frame)
        bind_tab_scrolling(self.potential_frame)
        
        # Predictions tab - use Notebook for multiple tabs
        predictions_container = tk.Frame(self.notebook, bg=theme['bg'])
        self.notebook.add(predictions_container, text=self.localization.t('predictions'))
        
        # Monitoring Tab
        self.monitoring_frame = tk.Frame(self.notebook, bg=theme['bg'])
        self.notebook.add(self.monitoring_frame, text="Monitoring")
        self._create_monitoring_tab()
        bind_tab_scrolling(self.monitoring_frame)
        
        # Create notebook for predictions sub-tabs
        self.predictions_notebook = ttk.Notebook(predictions_container)
        self.predictions_notebook.pack(fill=tk.BOTH, expand=True)
        
        # Regular Predictions tab
        regular_predictions_container = tk.Frame(self.predictions_notebook, bg=theme['bg'])
        self.predictions_notebook.add(regular_predictions_container, text="Predictions")
        
        # Create canvas and scrollbar for scrollable predictions tab
        predictions_canvas = tk.Canvas(regular_predictions_container, bg=theme['bg'], highlightthickness=0)
        self.predictions_frame = tk.Frame(predictions_canvas, bg=theme['bg'])
        
        predictions_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Modern scrollbar for predictions tab
        predictions_scrollbar = ModernScrollbar(regular_predictions_container, command=predictions_canvas.yview, theme=theme)
        predictions_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        # Force initial update
        predictions_canvas.update_idletasks()
        predictions_scrollbar._update_slider()
        
        # Configure scrollbar
        predictions_canvas.configure(yscrollcommand=predictions_scrollbar.set)
        predictions_canvas.create_window((0, 0), window=self.predictions_frame, anchor=tk.NW)
        
        def configure_predictions_scroll(event=None):
            predictions_canvas.configure(scrollregion=predictions_canvas.bbox("all"))
        
        self.predictions_frame.bind('<Configure>', configure_predictions_scroll)
        predictions_canvas.bind('<Configure>', lambda e: predictions_canvas.itemconfig(predictions_canvas.find_all()[0], width=e.width) if predictions_canvas.find_all() else None)
        
        # Bind mousewheel for predictions tab - smooth scrolling
        def on_predictions_mousewheel(event):
            delta = event.delta if hasattr(event, 'delta') else (event.num == 4 and -1) or 1
            if isinstance(delta, (int, float)):
                scroll_amount = int(-1 * (delta / 120)) * 3  # Smooth scroll
                predictions_canvas.yview_scroll(scroll_amount, "units")
            return "break"  # Prevent event propagation
        
        regular_predictions_container.bind("<MouseWheel>", on_predictions_mousewheel)
        predictions_canvas.bind("<MouseWheel>", on_predictions_mousewheel)
        self.predictions_frame.bind("<MouseWheel>", on_predictions_mousewheel)
        regular_predictions_container.bind("<Button-4>", lambda e: predictions_canvas.yview_scroll(-3, "units") or "break")
        regular_predictions_container.bind("<Button-5>", lambda e: predictions_canvas.yview_scroll(3, "units") or "break")
        predictions_canvas.bind("<Button-4>", lambda e: predictions_canvas.yview_scroll(-3, "units") or "break")
        predictions_canvas.bind("<Button-5>", lambda e: predictions_canvas.yview_scroll(3, "units") or "break")
        self.predictions_frame.bind("<Button-4>", lambda e: predictions_canvas.yview_scroll(-3, "units") or "break")
        self.predictions_frame.bind("<Button-5>", lambda e: predictions_canvas.yview_scroll(3, "units") or "break")
        
        # Store reference
        self.predictions_container = predictions_container
        self.predictions_canvas = predictions_canvas
        
        # Trend Change Predictions tab
        trend_change_container = tk.Frame(self.predictions_notebook, bg=theme['bg'])
        self.predictions_notebook.add(trend_change_container, text="Trend Changes")
        
        # Create canvas and scrollbar for trend change predictions
        trend_change_canvas = tk.Canvas(trend_change_container, bg=theme['bg'], highlightthickness=0)
        self.trend_change_frame = tk.Frame(trend_change_canvas, bg=theme['bg'])
        
        trend_change_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        trend_change_scrollbar = ModernScrollbar(trend_change_container, command=trend_change_canvas.yview, theme=theme)
        trend_change_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        trend_change_canvas.update_idletasks()
        trend_change_scrollbar._update_slider()
        
        trend_change_canvas.configure(yscrollcommand=trend_change_scrollbar.set)
        trend_change_canvas.create_window((0, 0), window=self.trend_change_frame, anchor=tk.NW)
        
        def configure_trend_change_scroll(event=None):
            trend_change_canvas.configure(scrollregion=trend_change_canvas.bbox("all"))
        
        self.trend_change_frame.bind('<Configure>', configure_trend_change_scroll)
        trend_change_canvas.bind('<Configure>', lambda e: trend_change_canvas.itemconfig(trend_change_canvas.find_all()[0], width=e.width) if trend_change_canvas.find_all() else None)
        
        def on_trend_change_mousewheel(event):
            delta = event.delta if hasattr(event, 'delta') else (event.num == 4 and -1) or 1
            if isinstance(delta, (int, float)):
                scroll_amount = int(-1 * (delta / 120)) * 3
                trend_change_canvas.yview_scroll(scroll_amount, "units")
        
        trend_change_container.bind("<MouseWheel>", on_trend_change_mousewheel)
        trend_change_canvas.bind("<MouseWheel>", on_trend_change_mousewheel)
        self.trend_change_frame.bind("<MouseWheel>", on_trend_change_mousewheel)
        
        self.trend_change_canvas = trend_change_canvas
        
        self._create_predictions_tab()
        self._create_trend_change_tab()
        self._load_trend_change_predictions()  # Load stored predictions
        
        # Momentum Changes tab
        self.momentum_changes_frame = tk.Frame(self.notebook, bg=theme['frame_bg'])
        self.notebook.add(self.momentum_changes_frame, text="🔄 Momentum Changes")
        self._create_momentum_changes_tab()
        
        # Potentials tab
        self.potentials_frame = tk.Frame(self.notebook, bg=theme['frame_bg'])
        self.notebook.add(self.potentials_frame, text="🎯 Potentials")
        self._create_potentials_tab()
        
        self._create_gun_tab()
        
        # Update predictions display
        self._update_predictions_display()
        
        # Update strategy visual feedback after UI is created
        self.root.after(100, self._update_strategy_visual_feedback)
    
    def _change_language(self, language: str):
        """Change application language"""
        self.localization.set_language(language)
        self.preferences.set_language(language)
        if hasattr(self, 'language_dropdown'):
            self.language_dropdown.set_value(language)
        self._refresh_ui_texts()
        logger.info(f"Language changed to: {language}")
    
    def _change_currency(self, currency: str):
        """Change currency"""
        old_currency = self.saved_currency
        self.saved_currency = currency
        self.localization.set_currency(currency)
        self.preferences.set_currency(currency)
        if hasattr(self, 'currency_dropdown'):
            self.currency_dropdown.set_value(currency)
        
        # Update budget display with new currency conversion
        if hasattr(self, 'budget_entry'):
            try:
                current_budget = float(self.budget_entry.get())
                # Convert from old currency to USD, then to new currency
                from localization import Localization
                rates = Localization.CURRENCY_RATES
                usd_budget = current_budget / rates.get(old_currency, 1.0)
                new_budget = usd_budget * rates.get(currency, 1.0)
                self.budget_entry.delete(0, tk.END)
                self.budget_entry.insert(0, str(round(new_budget, 2)))
                self._save_budget()  # Save with new currency
            except ValueError:
                pass  # Invalid budget, don't update
        
        self._update_portfolio_display()
        self._update_predictions_display()
        # Update amount label
        currency_symbol = self.localization.format_currency_symbol()
        self.amount_label.config(text=f"{self.localization.t('amount')} ({currency_symbol})")
        logger.info(f"Currency changed to: {currency}")
    
    def _save_budget(self):
        """Save budget to preferences with current currency"""
        try:
            budget = float(self.budget_entry.get())
            if budget > 0:
                self.preferences.set_budget(budget, self.saved_currency)
                logger.info(f"Budget saved: {budget} {self.saved_currency}")
        except ValueError:
            pass  # Invalid input, don't save
    
    def _update_strategy_visual_feedback(self):
        """Update visual feedback for strategy selection"""
        theme = self.theme_manager.get_theme()
        selected = self.strategy_var.get()
        
        for strategy, frame in self.strategy_buttons.items():
            if strategy == selected:
                # Highlight selected strategy
                frame.config(bg=theme['accent'], highlightbackground=theme['accent'], highlightthickness=2)
                # Update button text to show selection
                for widget in frame.winfo_children():
                    if isinstance(widget, tk.Radiobutton):
                        widget.config(fg=theme['button_fg'] if theme.get('button_fg') else '#FFFFFF')
            else:
                # Reset unselected strategies
                frame.config(bg=theme['card_bg'], highlightbackground=theme['border'], highlightthickness=0)
                for widget in frame.winfo_children():
                    if isinstance(widget, tk.Radiobutton):
                        widget.config(fg=theme['fg'])
    
    def _refresh_ui_texts(self):
        """Refresh all UI texts with current language"""
        theme = self.theme_manager.get_theme()
        
        # Update title and subtitle
        self.title_label.config(text=self.localization.t('app_title'))
        self.subtitle_label.config(text=self.localization.t('app_subtitle'))
        
        # Update card titles
        # Note: Card titles are harder to update, so we'll skip them for now
        # or recreate the UI (simpler approach)
        
        # Update buttons and labels
        self.trading_btn.config(text=f"⚡ {self.localization.t('trading')}")
        self.mixed_btn.config(text=f"🔄 {self.localization.t('mixed')}")
        self.investing_btn.config(text=f"📈 {self.localization.t('investing')}")
        self.symbol_label.config(text=self.localization.t('symbol'))
        self.analyze_btn.config(text=self.localization.t('analyze_stock'))
        currency_symbol = self.localization.format_currency_symbol()
        self.amount_label.config(text=f"{self.localization.t('amount')} ({currency_symbol})")
        self.calc_btn.config(text=self.localization.t('calculate_potential'))
        self.set_balance_btn.config(text=self.localization.t('set_balance'))
        self.view_all_btn.config(text=self.localization.t('view_all'))
        self.close_btn.config(text="✕ " + self.localization.t('close'))
        
        # Update notebook tabs
        self.notebook.tab(0, text=self.localization.t('analysis'))
        self.notebook.tab(1, text=self.localization.t('charts'))
        self.notebook.tab(2, text=self.localization.t('indicators'))
        self.notebook.tab(3, text=self.localization.t('potential_trade'))
        self.notebook.tab(4, text=self.localization.t('predictions'))
        
        # Update other displays
        self._update_portfolio_display()
        self._update_predictions_display()
        self.results_title.config(text=self.localization.t('stock_analysis_results'))
    
    def _create_predictions_tab(self):
        """Create the modern predictions tracking tab with stock lookup"""
        theme = self.theme_manager.get_theme()
        
        # Stock Lookup Card - moved from sidebar
        stock_lookup_card = ModernCard(self.predictions_frame, self.localization.t('stock_lookup'), theme=theme, padding=20)
        stock_lookup_card.pack(fill=tk.X, pady=(0, 20))
        
        self.symbol_label = tk.Label(stock_lookup_card.content_frame, text=self.localization.t('symbol'), bg=theme['card_bg'],
                fg=theme['text_secondary'], font=('Segoe UI', 9))
        self.symbol_label.pack(anchor=tk.W, pady=(0, 8))
        
        entry_frame = tk.Frame(stock_lookup_card.content_frame, bg=theme['card_bg'])
        entry_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.symbol_entry = tk.Entry(entry_frame, font=('Segoe UI', 12),
                                    bg=theme['entry_bg'], fg=theme['entry_fg'],
                                    relief=tk.FLAT, bd=0, insertbackground=theme['fg'])
        self.symbol_entry.pack(fill=tk.X, ipady=10)
        self.symbol_entry.bind('<Return>', lambda e: self._analyze_stock())
        
        # Entry underline with focus animation
        self.entry_underline = tk.Frame(entry_frame, bg=theme['border'], height=2)
        self.entry_underline.pack(fill=tk.X, pady=(5, 0))
        
        def on_entry_focus_in(e):
            self.entry_underline.config(bg=theme['accent'], height=2)
        def on_entry_focus_out(e):
            self.entry_underline.config(bg=theme['border'], height=2)
        
        self.symbol_entry.bind('<FocusIn>', on_entry_focus_in)
        self.symbol_entry.bind('<FocusOut>', on_entry_focus_out)
        
        # Button frame for analyze and market scan
        btn_frame = tk.Frame(stock_lookup_card.content_frame, bg=theme['card_bg'])
        btn_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.analyze_btn = ModernButton(btn_frame, self.localization.t('analyze_stock'), 
                                   command=self._analyze_stock, theme=theme)
        self.analyze_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        self.scan_market_btn = ModernButton(btn_frame, "🔍 Analyze Predictions", 
                                        command=self._scan_market, theme=theme)
        self.scan_market_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))
        
        # Monitor trend changes button (separate row for emphasis)
        monitor_frame = tk.Frame(stock_lookup_card.content_frame, bg=theme['card_bg'])
        monitor_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.monitor_trend_btn = ModernButton(monitor_frame, "🔔 Monitor Trend Changes", 
                                             command=self._toggle_trend_monitoring, theme=theme)
        self.monitor_trend_btn.pack(fill=tk.X)
        
        # Search history dropdown
        history_frame = tk.Frame(stock_lookup_card.content_frame, bg=theme['card_bg'])
        history_frame.pack(fill=tk.X)
        
        history_label = tk.Label(history_frame, text=self.localization.t('recent_searches'), 
                                bg=theme['card_bg'], fg=theme['text_secondary'], 
                                font=('Segoe UI', 9))
        history_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.history_var = tk.StringVar()
        self.history_dropdown = ttk.Combobox(history_frame, textvariable=self.history_var,
                                           state='readonly', font=('Segoe UI', 10),
                                           width=20)
        self.history_dropdown.pack(fill=tk.X)
        self.history_dropdown.bind('<<ComboboxSelected>>', self._on_history_select)
        self._update_search_history()
        
        # Header card with stats
        header_card = ModernCard(self.predictions_frame, self.localization.t('prediction_statistics'), theme=theme, padding=20)
        header_card.pack(fill=tk.X, pady=(0, 20))
        
        stats = self.predictions_tracker.get_statistics()
        
        stats_inner = tk.Frame(header_card.content_frame, bg=theme['card_bg'])
        stats_inner.pack(fill=tk.X)
        
        # Modern stats display
        stats_grid = tk.Frame(stats_inner, bg=theme['card_bg'])
        stats_grid.pack(fill=tk.X)
        
        # Enhanced stats display with more metrics
        stat_items = [
            (self.localization.t('total'), stats['total_predictions'], theme['fg']),
            (self.localization.t('active'), stats['active'], theme['accent']),
            (self.localization.t('verified'), stats['verified'], theme['success']),
            (self.localization.t('accuracy'), f"{stats['accuracy']:.1f}%" if stats['verified'] > 0 else "N/A", 
             theme['accent'] if stats['verified'] > 0 else theme['text_secondary']),
        ]
        
        # Add recent accuracy if available
        if stats.get('recent_verified_count', 0) > 0:
            stat_items.append(
                ('30d Accuracy', f"{stats['recent_accuracy']:.1f}%", 
                 theme['success'] if stats['recent_accuracy'] >= stats.get('accuracy', 0) else theme['warning'])
            )
        
        for i, (label, value, color) in enumerate(stat_items):
            stat_frame = tk.Frame(stats_grid, bg=theme['card_bg'])
            stat_frame.grid(row=0, column=i, padx=15, pady=10, sticky=tk.W+tk.E)
            
            tk.Label(stat_frame, text=label, bg=theme['card_bg'], 
                    fg=theme['text_secondary'], font=('Segoe UI', 9)).pack()
            tk.Label(stat_frame, text=str(value), bg=theme['card_bg'], 
                    fg=color, font=('Segoe UI', 16, 'bold')).pack()
        
        # Add strategy breakdown if available
        if stats.get('strategy_stats'):
            strategy_frame = tk.Frame(header_card.content_frame, bg=theme['card_bg'])
            strategy_frame.pack(fill=tk.X, pady=(10, 0))
            
            tk.Label(strategy_frame, text="Strategy Breakdown:", bg=theme['card_bg'],
                    fg=theme['text_secondary'], font=('Segoe UI', 9, 'bold')).pack(anchor=tk.W)
            
            strategy_inner = tk.Frame(strategy_frame, bg=theme['card_bg'])
            strategy_inner.pack(fill=tk.X, pady=(5, 0))
            
            for strategy, s_stats in stats['strategy_stats'].items():
                strategy_label = tk.Label(strategy_inner, 
                    text=f"{strategy.title()}: {s_stats['accuracy']:.1f}% ({s_stats['correct']}/{s_stats['total']})",
                    bg=theme['card_bg'], fg=theme['text_secondary'], font=('Segoe UI', 8))
                strategy_label.pack(side=tk.LEFT, padx=10)
        
        ModernButton(header_card.content_frame, self.localization.t('refresh_stats'), 
                    command=self._refresh_predictions, theme=theme).pack(pady=(15, 0))
        
        # Predictions list card
        list_card = ModernCard(self.predictions_frame, self.localization.t('all_predictions'), theme=theme, padding=20)
        list_card.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        # Treeview with modern styling
        list_inner = tk.Frame(list_card.content_frame, bg=theme['card_bg'])
        list_inner.pack(fill=tk.BOTH, expand=True)
        
        columns = ('ID', 'Symbol', 'Sentiment', 'Action', 'Wait %', 'Entry', 'Target', 'Target Date', 'Status', 'Result', 'Delete')
        self.predictions_tree = ttk.Treeview(list_inner, columns=columns, show='headings', height=12)
        
        for col in columns:
            self.predictions_tree.heading(col, text=col, anchor=tk.CENTER)
            if col == 'ID':
                self.predictions_tree.column(col, width=0, stretch=False) # Hide ID
            elif col == 'Sentiment':
                self.predictions_tree.column(col, width=100, anchor=tk.CENTER)
            elif col == 'Wait %':
                self.predictions_tree.column(col, width=80, anchor=tk.CENTER)
            elif col == 'Target Date':
                self.predictions_tree.column(col, width=110, anchor=tk.CENTER)
            elif col == 'Delete':
                self.predictions_tree.column(col, width=50, anchor=tk.CENTER)
            else:
                self.predictions_tree.column(col, width=90, anchor=tk.CENTER)
        
        # Bind click event on Delete column
        self.predictions_tree.bind('<Button-1>', self._on_predictions_tree_click)
        # Bind motion event for hover effect on Delete column
        self.predictions_tree.bind('<Motion>', self._on_predictions_tree_motion)
        self.predictions_tree.bind('<Leave>', self._on_predictions_tree_leave)
        
        # Track currently hovered delete cell
        self.hovered_delete_item = None
        
        # Modern scrollbar for predictions tree
        tree_scrollbar = ModernScrollbar(list_inner, command=self.predictions_tree.yview, theme=theme)
        self.predictions_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        # Bind mousewheel to treeview for scrolling
        def on_tree_mousewheel(event):
            delta = event.delta if hasattr(event, 'delta') else (event.num == 4 and -1) or 1
            if isinstance(delta, (int, float)):
                scroll_amount = int(-1 * (delta / 120)) * 3
                self.predictions_tree.yview_scroll(scroll_amount, "units")
            return "break"
        
        self.predictions_tree.bind("<MouseWheel>", on_tree_mousewheel)
        self.predictions_tree.bind("<Button-4>", lambda e: self.predictions_tree.yview_scroll(-3, "units") or "break")
        self.predictions_tree.bind("<Button-5>", lambda e: self.predictions_tree.yview_scroll(3, "units") or "break")
        
        self.predictions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        # Force initial update
        list_inner.update_idletasks()
        tree_scrollbar._update_slider()
        
        # Action buttons card
        actions_card = ModernCard(self.predictions_frame, "", theme=theme, padding=15)
        actions_card.pack(fill=tk.X)
        
        btn_inner = tk.Frame(actions_card.content_frame, bg=theme['card_bg'])
        btn_inner.pack(fill=tk.X)
        
        ModernButton(btn_inner, self.localization.t('save_current'), command=self._save_current_prediction, 
                    theme=theme).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ModernButton(btn_inner, self.localization.t('verify_all'), command=self._verify_all_predictions, 
                    theme=theme).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ModernButton(btn_inner, self.localization.t('clear_old'), command=self._clear_old_predictions, 
                    theme=theme).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Clear All button (separate row for emphasis)
        clear_all_frame = tk.Frame(actions_card.content_frame, bg=theme['card_bg'])
        clear_all_frame.pack(fill=tk.X, pady=(10, 0))
        ModernButton(clear_all_frame, "🗑️ Clear All Predictions", command=self._clear_all_predictions, 
                    theme=theme).pack(fill=tk.X)
    
    def _show_predictions_tab(self):
        """Switch to predictions tab"""
        # Find predictions tab index (it's the last tab)
        try:
            # Predictions tab is typically the last one
            self.notebook.select(len(self.notebook.tabs()) - 1)
        except (tk.TclError, IndexError) as e:
            logger.debug(f"Could not select predictions tab by index: {e}")
            # Fallback: find by text
            for i in range(self.notebook.index("end")):
                if self.localization.t('predictions') in self.notebook.tab(i, "text"):
                    self.notebook.select(i)
                    break
    
    def _save_current_prediction(self):
        """Save current analysis as a prediction - using CURRENT price"""
        if not self.current_analysis or 'error' in self.current_analysis:
            messagebox.showwarning(self.localization.t('no_analysis'), self.localization.t('please_analyze_stock'))
            return
        
        if not self.current_symbol:
            messagebox.showwarning(self.localization.t('no_symbol'), self.localization.t('no_symbol'))
            return
        
        # Fetch CURRENT price (not the price from when analysis was done)
        symbol = self.current_symbol
        try:
            current_stock_data = self.data_fetcher.fetch_stock_data(symbol)
            current_price_usd = current_stock_data.get('price', 0)
            if current_price_usd <= 0:
                raise ValueError("Could not fetch current price")
        except Exception as e:
            logger.error(f"Error fetching current price for prediction: {e}")
            messagebox.showerror(self.localization.t('error'), 
                               f"Could not fetch current price for {symbol}. Cannot save prediction.")
            return
        
        recommendation = self.current_analysis.get('recommendation', {})
        action = recommendation.get('action', 'HOLD')
        old_entry_price = recommendation.get('entry_price', current_price_usd)
        old_target_price = recommendation.get('target_price', current_price_usd)
        old_stop_loss = recommendation.get('stop_loss', current_price_usd)
        confidence = recommendation.get('confidence', 0)
        reasoning = self.current_analysis.get('reasoning', '')
        
        # Use CURRENT price as entry
        entry_price = current_price_usd
        
        # Recalculate targets based on current price (proportional adjustment)
        if old_entry_price > 0:
            price_ratio = current_price_usd / old_entry_price
            target_price = old_target_price * price_ratio
            stop_loss = old_stop_loss * price_ratio
        else:
            target_price = old_target_price
            stop_loss = old_stop_loss
        
        # Validate that targets make sense and haven't already been reached
        if action == "BUY":
            if current_price_usd >= target_price:
                # Target already reached - recalculate a new target
                # Use strategy-appropriate target based on current price
                strategy = self.strategy_var.get()
                if strategy == "trading":
                    target_price = current_price_usd * 1.03  # 3% target for trading
                elif strategy == "mixed":
                    target_price = current_price_usd * 1.06  # 6% target for mixed
                else:  # investing
                    target_price = current_price_usd * 1.12  # 12% target for investing
                
                messagebox.showwarning(
                    self.localization.t('warning'),
                    f"Target price was already reached. New target set: {self.localization.format_currency(target_price)}"
                )
            
            if stop_loss >= entry_price:
                # Stop loss is above entry - invalid for BUY
                if strategy == "trading":
                    stop_loss = current_price_usd * 0.97  # 3% stop loss
                elif strategy == "mixed":
                    stop_loss = current_price_usd * 0.95  # 5% stop loss
                else:  # investing
                    stop_loss = current_price_usd * 0.80  # 20% stop loss
        
        elif action == "SELL":
            if current_price_usd >= target_price:
                # Target already reached - recalculate
                # NOTE: SELL = bullish (expecting price to go UP), so target should be ABOVE entry
                # If current price >= target, the target was reached (price went up)
                strategy = self.strategy_var.get()
                if strategy == "trading":
                    target_price = current_price_usd * 1.03  # 3% above for trading
                elif strategy == "mixed":
                    target_price = current_price_usd * 1.06  # 6% above for mixed
                else:  # investing
                    target_price = current_price_usd * 1.12  # 12% above for investing
                
                messagebox.showwarning(
                    self.localization.t('warning'),
                    f"Target price was already reached. New target set: {self.localization.format_currency(target_price)}"
                )
            
            if stop_loss <= 0 or stop_loss >= entry_price:
                # Stop loss is invalid for SELL (bullish) - should be BELOW entry
                # NOTE: SELL = bullish (expecting price to go UP), stop loss limits loss if price goes DOWN
                if strategy == "trading":
                    stop_loss = current_price_usd * 0.97  # 3% stop loss below entry
                elif strategy == "mixed":
                    stop_loss = current_price_usd * 0.95  # 5% stop loss below entry
                else:  # investing
                    stop_loss = current_price_usd * 0.80  # 20% stop loss below entry
        
        if entry_price <= 0 or target_price <= 0 or stop_loss <= 0:
            messagebox.showwarning(self.localization.t('invalid_prediction'), self.localization.t('invalid_prediction'))
            return
        
        strategy = self.strategy_var.get()
        
        # Calculate estimated days based on strategy
        estimated_days = recommendation.get('estimated_days')
        
        if not estimated_days:
            # Fallback to defaults if not provided by analyzer
            if strategy == "trading":
                estimated_days = 10  # Short-term: ~10 days
            elif strategy == "mixed":
                estimated_days = 21  # Medium-term: ~3 weeks
            else:  # investing
                estimated_days = 547  # Long-term: ~1.5 years
        
        prediction = self.predictions_tracker.add_prediction(
            self.current_symbol, strategy, action, entry_price, 
            target_price, stop_loss, confidence, reasoning, estimated_days
        )
        
        # Format target date for display
        from datetime import datetime, timedelta
        target_date = datetime.now() + timedelta(days=estimated_days)
        target_date_str = target_date.strftime("%Y-%m-%d")
        
        messagebox.showinfo(self.localization.t('prediction_saved'), 
                          f"{self.localization.t('prediction_saved')} #{prediction['id']} - {self.current_symbol}\n"
                          f"Entry: {self.localization.format_currency(entry_price)}\n"
                          f"Target: {self.localization.format_currency(target_price)}\n"
                          f"Estimated Target Date: {target_date_str}")
        self._update_predictions_display()
    
    def _verify_all_predictions(self):
        """Manually verify all active predictions (checks all, not just those past target date)"""
        def verify_in_background():
            try:
                self.root.after(0, self._update_status, self.localization.t('verifying_predictions'))
                
                # 1. Verify Trend Changes
                trend_verified = self.trend_change_tracker.check_for_verification(self.data_fetcher)
                if trend_verified:
                    logger.info(f"Verified {len(trend_verified)} trend predictions")
                    for pred in trend_verified:
                        if hasattr(self, 'learning_tracker'):
                            self.learning_tracker.record_verified_trend_change(
                                pred['symbol'],
                                pred['predicted_change'],
                                pred.get('actual_change', 'unknown'),
                                pred.get('was_correct', False)
                            )
                    self.root.after(0, self._update_trend_change_display)
                
                # 2. Verify Momentum History
                momentum_verified = self.momentum_monitor.verify_history(self.data_fetcher)
                if momentum_verified:
                    logger.info(f"Verified {len(momentum_verified)} momentum history items")
                    self.root.after(0, self._update_momentum_changes_display)
                
                # 3. Verify Regular Predictions
                # check_all=True to verify all active predictions regardless of target date
                result = self.predictions_tracker.verify_all_active_predictions(self.data_fetcher, check_all=True)
                if result['verified'] > 0:
                    newly_verified = result.get('newly_verified', [])
                    # Record learning events for verified predictions
                    for pred in newly_verified:
                        if pred.get('was_correct') is not None:
                            # Create outcome dict
                            outcome = {
                                'was_correct': pred.get('was_correct', False),
                                'actual_price_change': pred.get('actual_price_change', 0.0),
                                'target_hit': pred.get('was_correct', False) and pred.get('action') in ['BUY', 'SELL'], # simplified check
                            }
                            self.learning_tracker.record_verified_prediction(
                                pred.get('was_correct', False),
                                pred.get('strategy', 'trading'),
                                prediction_data=pred,
                                actual_outcome=outcome
                            )
                    self.root.after(0, self._show_verification_results, result, newly_verified)
                    self.root.after(0, self._update_predictions_display)
                else:
                    messagebox.showinfo(self.localization.t('info'), 
                                      self.localization.t('no_predictions_verified'))
            except Exception as e:
                logger.error(f"Error verifying predictions: {e}")
                self.root.after(0, lambda: messagebox.showerror(self.localization.t('error'), 
                    f"{self.localization.t('error')}: {str(e)}"))
        
        thread = threading.Thread(target=verify_in_background)
        thread.daemon = True
        thread.start()
    
    def _clear_all_predictions(self):
        """Clear ALL predictions (regular and trend change) - fresh start"""
        theme = self.theme_manager.get_theme()
        
        # Count predictions - include ALL predictions (active, verified, expired)
        regular_count = len(self.predictions_tracker.predictions)  # All regular predictions
        trend_count = len(self.trend_change_tracker.predictions)  # All trend change predictions
        total_count = regular_count + trend_count
        
        if total_count == 0:
            messagebox.showinfo("No Predictions", "There are no predictions to clear.")
            return
        
        # Confirm deletion
        response = messagebox.askyesno(
            "Clear All Predictions",
            f"This will delete ALL predictions:\n\n"
            f"• Regular predictions: {regular_count}\n"
            f"• Trend change predictions: {trend_count}\n\n"
            f"Total: {total_count} prediction(s)\n\n"
            f"Are you sure you want to continue?",
            icon='warning'
        )
        
        if not response:
            return
        
        # Clear regular predictions
        cleared_regular = 0
        if regular_count > 0:
            self.predictions_tracker.predictions = []
            self.predictions_tracker.save()
            cleared_regular = regular_count
        
        # Clear trend change predictions
        cleared_trend = 0
        if trend_count > 0:
            self.trend_change_tracker.predictions = []
            self.trend_change_tracker.save()
            cleared_trend = trend_count
        
        # Update displays
        self._update_predictions_display()
        self._update_trend_change_display()
        
        messagebox.showinfo(
            "Predictions Cleared",
            f"Successfully cleared all predictions:\n\n"
            f"• Regular predictions: {cleared_regular}\n"
            f"• Trend change predictions: {cleared_trend}\n\n"
            f"Total: {cleared_regular + cleared_trend} prediction(s) deleted.\n\n"
            f"You can now make fresh predictions!"
        )
        
        logger.info(f"Cleared all predictions: {cleared_regular} regular, {cleared_trend} trend change")
    
    def _clear_all_trend_predictions(self):
        """Clear ALL trend change predictions"""
        trend_count = len(self.trend_change_tracker.get_active_predictions())
        
        if trend_count == 0:
            messagebox.showinfo("No Predictions", "There are no trend change predictions to clear.")
            return
        
        # Confirm deletion
        response = messagebox.askyesno(
            "Clear All Trend Predictions",
            f"This will delete ALL trend change predictions:\n\n"
            f"• Trend change predictions: {trend_count}\n\n"
            f"Are you sure you want to continue?",
            icon='warning'
        )
        
        if not response:
            return
        
        # Clear trend change predictions
        self.trend_change_tracker.predictions = []
        self.trend_change_tracker.save()
        
        # Update display
        self._update_trend_change_display()
        
        messagebox.showinfo(
            "Trend Predictions Cleared",
            f"Successfully cleared {trend_count} trend change prediction(s)."
        )
        
        logger.info(f"Cleared all trend change predictions: {trend_count}")
    
    def _clear_old_predictions(self):
        """Clear old verified predictions"""
        if messagebox.askyesno(self.localization.t('confirm'), self.localization.t('clear_all_verified')):
            verified = self.predictions_tracker.get_verified_predictions()
            self.predictions_tracker.predictions = [
                p for p in self.predictions_tracker.predictions if p['status'] != 'verified'
            ]
            self.predictions_tracker.save()
            messagebox.showinfo(self.localization.t('cleared'), 
                             self.localization.t('removed_predictions').format(count=len(verified)))
            self._update_predictions_display()
    
    def _toggle_trend_monitoring(self):
        """Toggle trend monitoring for currently analyzed stock"""
        if not self.current_symbol:
            messagebox.showinfo("Info", "Please analyze a stock first to enable trend monitoring.")
            return
        
        symbol = self.current_symbol
        
        if symbol in self.monitored_stocks:
            # Disable monitoring
            self.monitored_stocks.remove(symbol)
            if hasattr(self, 'monitor_trend_btn'):
                self.monitor_trend_btn.config(text="🔔 Monitor Trend Changes")
            messagebox.showinfo("Monitoring Disabled", 
                              f"Trend monitoring disabled for {symbol}.\n"
                              f"You will no longer receive notifications for trend changes.")
        else:
            # Enable monitoring
            self.monitored_stocks.add(symbol)
            if hasattr(self, 'monitor_trend_btn'):
                self.monitor_trend_btn.config(text="🔕 Stop Monitoring")
            messagebox.showinfo("Monitoring Enabled", 
                              f"Trend monitoring enabled for {symbol}.\n"
                              f"You will receive notifications when:\n"
                              f"• Price reaches peak or bottom\n"
                              f"• Trend changes are detected\n\n"
                              f"Checking every 5 minutes...\n\n"
                              f"🧠 All events will feed Megamind for learning!")
        
        # Save to preferences
        self.preferences.set('monitored_stocks', list(self.monitored_stocks))
        logger.info(f"Trend monitoring {'enabled' if symbol in self.monitored_stocks else 'disabled'} for {symbol}")
    
    def _check_monitored_stock_for_changes(self, symbol: str):
        """Check monitored stock for peaks, bottoms, and trend changes"""
        try:
            # Fetch fresh data
            stock_data = self.data_fetcher.fetch_stock_data(symbol, force_refresh=True)
            history_data = self.data_fetcher.fetch_stock_history(symbol, period='3mo')
            
            if 'error' in stock_data or not history_data.get('data'):
                return
            
            # Perform trading analysis
            trading_analysis = self.trading_analyzer.analyze(stock_data, history_data)
            if 'error' in trading_analysis:
                return
            
            indicators = trading_analysis.get('indicators', {})
            price_action = trading_analysis.get('price_action', {})
            momentum_analysis = trading_analysis.get('momentum_analysis', {})
            
            current_price = stock_data.get('price', 0)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 1. Check for PEAKS and BOTTOMS
            peak_bottom_detection = self.momentum_monitor.detect_peaks_bottoms(
                symbol,
                history_data.get('data', []),
                indicators
            )
            
            # Track last detected peak/bottom to avoid duplicate notifications
            last_detection = self._last_peak_bottom.get(symbol, {})
            last_detection_time_str = last_detection.get('time', '2000-01-01T00:00:00')
            try:
                last_detection_time = datetime.fromisoformat(last_detection_time_str)
                time_since_last = (datetime.now() - last_detection_time).total_seconds() / 3600  # hours
            except (ValueError, AttributeError) as e:
                logger.debug(f"Could not parse last detection time: {e}")
                time_since_last = 999  # Force notification if parsing fails
            
            # Show peak notification (if not shown recently - within last 24 hours)
            if peak_bottom_detection.get('peak_detected') and \
               (last_detection.get('type') != 'peak' or time_since_last > 24):
                peak_price = peak_bottom_detection.get('peak_price', current_price)
                confidence = peak_bottom_detection.get('confidence', 0)
                reasoning = peak_bottom_detection.get('reasoning', '')
                
                # Feed to Megamind (with current price for tracking)
                self.learning_tracker.record_peak_detection(
                    symbol, peak_price, confidence, reasoning, current_price
                )
                
                # Show notification (pass both peak price and current price)
                self.root.after(0, self._show_peak_notification, symbol, peak_price, current_price, confidence, reasoning, timestamp)
                
                # Update last detection
                self._last_peak_bottom[symbol] = {
                    'type': 'peak',
                    'price': peak_price,
                    'time': datetime.now().isoformat()
                }
            
            # Show bottom notification
            if peak_bottom_detection.get('bottom_detected') and \
               (last_detection.get('type') != 'bottom' or time_since_last > 24):
                bottom_price = peak_bottom_detection.get('bottom_price', current_price)
                confidence = peak_bottom_detection.get('confidence', 0)
                reasoning = peak_bottom_detection.get('reasoning', '')
                
                # Feed to Megamind (with current price for tracking)
                self.learning_tracker.record_bottom_detection(
                    symbol, bottom_price, confidence, reasoning, current_price
                )
                
                # Show notification (pass both bottom price and current price)
                self.root.after(0, self._show_bottom_notification, symbol, bottom_price, current_price, confidence, reasoning, timestamp)
                
                # Update last detection
                self._last_peak_bottom[symbol] = {
                    'type': 'bottom',
                    'price': bottom_price,
                    'time': datetime.now().isoformat()
                }
            
            # 2. Check for TREND CHANGES (using existing momentum monitoring)
            momentum_changes = self.momentum_monitor.update_momentum_state(
                symbol,
                indicators,
                price_action,
                momentum_analysis
            )
            
            # Track last trend change to avoid duplicates
            last_trend = self._last_trend_change.get(symbol, {})
            last_trend_time_str = last_trend.get('time', '2000-01-01T00:00:00')
            try:
                last_trend_time = datetime.fromisoformat(last_trend_time_str)
                time_since_trend_change = (datetime.now() - last_trend_time).total_seconds() / 3600
            except (ValueError, AttributeError) as e:
                logger.debug(f"Could not parse last trend change time: {e}")
                time_since_trend_change = 999  # Force notification if parsing fails
            
            # Show trend change notification (if significant change)
            if momentum_changes.get('trend_reversed') or \
               (momentum_changes.get('momentum_changed') and len(momentum_changes.get('changes', [])) > 0):
                
                # Add to history
                self.momentum_monitor.add_to_history(
                    symbol=symbol,
                    action='MONITOR',  # Indicated it came from background monitoring
                    changes=momentum_changes,
                    strategy='monitored',
                    current_price=current_price
                )
                self.root.after(0, self._update_momentum_changes_display)
                
                # Only notify if trend actually changed (not just momentum shift)
                trend_change_desc = momentum_changes.get('trend_change')
                if trend_change_desc and (last_trend.get('trend') != trend_change_desc or time_since_trend_change > 24):
                    old_trend = last_trend.get('trend', 'unknown')
                    # Extract new trend from description (e.g., "Uptrend → Downtrend" -> "downtrend")
                    if '→' in trend_change_desc:
                        new_trend = trend_change_desc.split('→')[-1].strip().lower()
                    else:
                        new_trend = trend_change_desc.lower()
                    
                    recommendation = momentum_changes.get('recommendation', {})
                    confidence = recommendation.get('confidence', 50)
                    
                    # Feed to Megamind
                    self.learning_tracker.record_trend_change_detection(
                        symbol, old_trend, new_trend, current_price, confidence
                    )
                    
                    # Show notification
                    self.root.after(0, self._show_trend_change_notification, symbol, momentum_changes, timestamp)
                    
                    # Update last trend
                    self._last_trend_change[symbol] = {
                    }
            
            # UPDATE MONITORING STATUS (New Tab)
            status_text = "Monitoring"
            if peak_bottom_detection.get('peak_detected'):
                status_text = "Peak Detected 🔴"
            elif peak_bottom_detection.get('bottom_detected'):
                status_text = "Bottom Detected 🟢"
            elif momentum_changes.get('trend_reversed'):
                status_text = "Trend Reversal ⚠️"
            elif momentum_changes.get('momentum_changed'):
                status_text = "Momentum Shift 🔄"
                
            # Get latest recommendation action
            rec = momentum_changes.get('recommendation', {})
            rec_action = rec.get('action', 'HOLD')
            
            # Calculate daily change if available
            change_pct = stock_data.get('change_percent', 0.0)
            
            self.monitoring_status[symbol] = {
                'symbol': symbol,
                'price': current_price,
                'status': status_text,
                'last_update': timestamp,
                'action': rec_action,
                'change_pct': change_pct
            }
            
            # Trigger display update
            if hasattr(self, '_update_monitoring_display'):
                self.root.after(0, self._update_monitoring_display)
        except Exception as e:
            logger.error(f"Error checking monitored stock {symbol} for changes: {e}")
    
    def _show_trend_change_notification(self, symbol: str, changes: Dict, timestamp: str):
        """Show notification when trend changes (enhanced version)"""
        # Batching check
        if hasattr(self, 'is_monitoring_batch_active') and self.is_monitoring_batch_active:
            change_type = "Trend Reversal" if changes.get('trend_reversed') else "Momentum Shift"
            self.monitoring_events.append({
                'type': 'TREND',
                'symbol': symbol,
                'price': None, # Price might not be readily available in 'changes', mostly message
                'message': f"{change_type}: {changes.get('trend_change', 'Unknown')}"
            })
            logger.info(f"Batched trend notification for {symbol}")
            return
            
        change_list = changes.get('changes', [])
        is_trend_reversal = changes.get('trend_reversed', False)
        recommendation = changes.get('recommendation', {})
        trend_change = changes.get('trend_change')
        
        action = recommendation.get('action', 'HOLD')
        confidence = recommendation.get('confidence', 50)
        
        if is_trend_reversal:
            title = f"⚠️ TREND REVERSAL: {symbol}"
        else:
            title = f"🔄 TREND CHANGE: {symbol}"
        
        message = f"📊 TECHNICAL UPDATE - {symbol}\n\n"
        message += f"Time: {timestamp}\n\n"
        
        if trend_change:
            message += f"📈 Trend Change: {trend_change}\n\n"
        
        # Show momentum-based recommendation but clarify it's technical-only
        action_emoji = "🟢" if action == "BUY" else "🔴" if action == "SELL" else "🟡"
        message += f"{action_emoji} Momentum Signal: {action} (Confidence: {confidence}%)\n"
        message += f"⚠️ NOTE: This is a momentum-based signal. Check the Analysis tab for the comprehensive recommendation (ML + Technical + Fundamental).\n\n"
        
        message += "Key Factors:\n"
        reasons = recommendation.get('reasons', [])
        if reasons:
            for reason in reasons[:3]:
                message += f"• {reason}\n"
        else:
            for change in change_list[:3]:
                message += f"• {change}\n"
        
        if len(change_list) > 3:
            message += f"\n... and {len(change_list) - 3} more changes\n"
        
        message += "\n💡 TIP: View the Analysis tab for the full recommendation that combines ML predictions, technical analysis, and fundamental factors."
        message += "\n\n🧠 Data fed to Megamind for learning!"
        
        messagebox.showinfo(title, message)
        logger.info(f"Trend change notification for {symbol} at {timestamp}: Momentum signal {action} ({confidence}%) - Note: May differ from main analysis")
    
    def _create_monitoring_tab(self):
        """Create the comprehensive monitoring tab"""
        theme = self.theme_manager.get_theme()
        
        # Header
        header_frame = tk.Frame(self.monitoring_frame, bg=theme['bg'])
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = tk.Label(header_frame, text="🛡️ Live Stock Monitoring", 
                              font=('Segoe UI', 14, 'bold'), bg=theme['bg'], fg=theme['fg'])
        title_label.pack(side=tk.LEFT)
        
        # Controls
        controls_frame = tk.Frame(header_frame, bg=theme['bg'])
        controls_frame.pack(side=tk.RIGHT)
        
        refresh_btn = ModernButton(controls_frame, text="Refresh", 
                                  command=self._update_monitoring_display, theme=theme)
        refresh_btn.pack(side=tk.LEFT, padx=(5, 0))
        
        # Treeview
        list_card = ModernCard(self.monitoring_frame, "", theme=theme, padding=15)
        list_card.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        list_inner = tk.Frame(list_card, bg=theme['card_bg'])
        list_inner.pack(fill=tk.BOTH, expand=True)
        
        columns = ('Symbol', 'Price', 'Change', 'Status', 'Action', 'Last Update')
        self.monitoring_tree = ttk.Treeview(list_inner, columns=columns, show='headings', height=15)
        
        # Column config
        self.monitoring_tree.heading('Symbol', text='Symbol')
        self.monitoring_tree.column('Symbol', width=80, anchor=tk.CENTER)
        
        self.monitoring_tree.heading('Price', text='Price')
        self.monitoring_tree.column('Price', width=100, anchor=tk.CENTER)
        
        self.monitoring_tree.heading('Change', text='Change')
        self.monitoring_tree.column('Change', width=80, anchor=tk.CENTER)
        
        self.monitoring_tree.heading('Status', text='Status')
        self.monitoring_tree.column('Status', width=150, anchor=tk.CENTER)
        
        self.monitoring_tree.heading('Action', text='Action')
        self.monitoring_tree.column('Action', width=100, anchor=tk.CENTER)
        
        self.monitoring_tree.heading('Last Update', text='Last Update')
        self.monitoring_tree.column('Last Update', width=150, anchor=tk.CENTER)
        
        # Scrollbar
        scrollbar = ModernScrollbar(list_inner, command=self.monitoring_tree.yview, theme=theme)
        self.monitoring_tree.configure(yscrollcommand=scrollbar.set)
        
        self.monitoring_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Tags
        self.monitoring_tree.tag_configure('peak', foreground='#FF4444')
        self.monitoring_tree.tag_configure('bottom', foreground='#00C851')
        self.monitoring_tree.tag_configure('reversal', foreground='#FFBB33')
        
        # Initial update
        self._update_monitoring_display()

    def _update_monitoring_display(self):
        """Update the monitoring treeview"""
        if not hasattr(self, 'monitoring_tree'):
            return
            
        # Clear
        for item in self.monitoring_tree.get_children():
            self.monitoring_tree.delete(item)
            
        # Populate
        if hasattr(self, 'monitoring_status'):
            for symbol, data in self.monitoring_status.items():
                price = f"${data.get('price', 0):.2f}"
                change = f"{data.get('change_pct', 0):+.2f}%"
                status = data.get('status', 'Monitoring')
                action = data.get('action', 'HOLD')
                last_update = data.get('last_update', '').split('T')[-1][:8] # HH:MM:SS
                
                # Tag
                tags = []
                if "Peak" in status: tags.append('peak')
                elif "Bottom" in status: tags.append('bottom')
                elif "Reversal" in status: tags.append('reversal')
                
                self.monitoring_tree.insert('', 'end', values=(
                    symbol, price, change, status, action, last_update
                ), tags=tuple(tags))

    def _start_monitored_stocks_checking(self):
        """Start periodic checking of monitored stocks (every 5 minutes)"""
        check_interval_ms = 5 * 60 * 1000  # 5 minutes
        
        def check_monitored():
            # Init Batch
            self.monitoring_events = []
            self.is_monitoring_batch_active = True
            
            # Get all symbols to monitor:
            # 1. Stocks with active predictions
            # 2. Stocks manually added to monitored list
            symbols_to_monitor = set()
            
            # Add active predictions
            try:
                active_preds = self.predictions_tracker.get_active_predictions()
                for pred in active_preds:
                    if pred.get('symbol'):
                        symbols_to_monitor.add(pred['symbol'])
            except Exception as e:
                logger.error(f"Error getting active predictions for monitoring: {e}")
            
            # Add manual list
            if hasattr(self, 'monitored_stocks') and self.monitored_stocks:
                symbols_to_monitor.update(self.monitored_stocks)
            
            if not symbols_to_monitor:
                # Reschedule even if no stocks monitored
                self.root.after(check_interval_ms, check_monitored)
                return
            
            # Count active threads for this batch
            self._monitoring_active_threads_count = len(symbols_to_monitor)
            
            # Wrapper to decrement counter logic
            def check_wrapper(sym):
                try:
                    self._check_monitored_stock_for_changes(sym)
                finally:
                    # Decrement safely (in GUI thread to avoid lock complexity, or just basic atomic dec)
                    # Since this runs in thread, we can't update GUI vars safely without after/queue.
                    # But integers are atomic-ish in Python GIL. Let's use a safe method.
                    # Actually, simple way: call a method on main thread to decrement.
                    self.root.after(0, self._decrement_monitoring_thread)

            # Check each monitored stock
            for symbol in list(symbols_to_monitor):
                thread = threading.Thread(target=check_wrapper, args=(symbol,))
                thread.daemon = True
                thread.start()
            
            # Start poller to check for completion
            self.root.after(1000, self._check_monitoring_batch_completion)
                
            # Schedule next cycle
            self.root.after(check_interval_ms, check_monitored)
            
    def _decrement_monitoring_thread(self):
        """Decrement thread counter"""
        if hasattr(self, '_monitoring_active_threads_count'):
            self._monitoring_active_threads_count -= 1

    def _check_monitoring_batch_completion(self):
        """Check if all monitoring threads are done"""
        start_time = getattr(self, '_monitoring_batch_start_time', 0)
        
        # Check if done
        if getattr(self, '_monitoring_active_threads_count', 0) <= 0:
            self._process_monitoring_batch()
            self.is_monitoring_batch_active = False
            return
            
        # Timeout safety (e.g. 60 seconds) - stop waiting
        # Just reschedule check
        self.root.after(1000, self._check_monitoring_batch_completion)

    def _process_monitoring_batch(self):
        """Process collected monitoring events into one summary"""
        if not self.monitoring_events:
            return
            
        count = len(self.monitoring_events)
        
        # De-duplicate events by symbol and type
        unique_events = {}
        for event in self.monitoring_events:
            key = f"{event['symbol']}_{event['type']}"
            unique_events[key] = event
            
        final_events = list(unique_events.values())
        count = len(final_events)
        
        if count == 0:
            return
            
        # Create summary message
        title = f"🛡️ Monitoring Update: {count} Event{'s' if count > 1 else ''}"
        
        message = f"Detected {count} important event{'s' if count > 1 else ''} in your monitored stocks.\n\n"
        
        # Show top 3 examples
        examples = final_events[:3]
        for ex in examples:
            icon = "🔴" if ex['type'] == 'PEAK' else "🟢" if ex['type'] == 'BOTTOM' else "⚠️"
            message += f"{icon} {ex['symbol']}: {ex['message']}\n"
            
        if count > 3:
            message += f"\n...and {count - 3} more.\n"
            
        message += "\n👉 Check the 'Monitoring' tab for full details."
        
        messagebox.showinfo(title, message)
        logger.info(f"Showed batched monitoring notification for {count} events")
        
        # Start first check
        check_monitored()
        logger.info("Started periodic monitoring of stocks for peaks/bottoms/trend changes (every 5 minutes)")
    
    def _update_predictions_display(self):
        """Update the predictions display with modern styling"""
        theme = self.theme_manager.get_theme()
        stats = self.predictions_tracker.get_statistics()
        
        # Update stats label
        if stats['verified'] > 0:
            self.pred_stats_label.config(
                text=f"{stats['accuracy']:.1f}% {self.localization.t('accuracy')}",
                fg=theme['success'] if stats['accuracy'] >= 70 else theme['warning'] if stats['accuracy'] >= 50 else theme['error']
            )
        else:
            self.pred_stats_label.config(text=self.localization.t('no_verified_predictions'), fg=theme['text_secondary'])
        
        # Clear tree
        for item in self.predictions_tree.get_children():
            self.predictions_tree.delete(item)
        
        # Add all predictions with modern formatting
        for pred in self.predictions_tracker.predictions:
            status = pred['status']
            result = self.localization.t('pending')
            if pred['verified']:
                result = f"✓ {self.localization.t('correct')}" if pred['was_correct'] else f"✗ {self.localization.t('incorrect')}"
            
            # Determine sentiment based on action (inverted)
            action = pred.get('action', 'HOLD')
            sentiment = "BULLISH 🟢" if action == 'SELL' else "BEARISH 🔴" if action == 'BUY' else "NEUTRAL ⚪"
            
            # Set action label with hint
            action_label = f"{action} (Inv)"
            
            # Calculate Wait % if active and we have current price
            wait_pct_str = "N/A"
            if status == 'active':
                entry_price = pred.get('entry_price', 0)
                # Try to find current price from current_data or fetcher
                curr_price = 0
                if hasattr(self, 'current_data') and self.current_data and 'price' in self.current_data:
                    curr_price = self.current_data['price']
                
                if curr_price > 0 and entry_price > 0:
                    diff = ((curr_price - entry_price) / curr_price) * 100
                    if diff > 0 and action == 'SELL': # Price is above entry but we want to buy low
                        wait_pct_str = f"+{diff:.1f}%"
                    elif diff < 0 and action == 'SELL': # Price is below entry
                        wait_pct_str = f"{diff:.1f}%"
                    else:
                        wait_pct_str = f"{diff:+.1f}%"
            
            # Format target date
            target_date_str = "N/A"
            estimated_date_str = pred.get('estimated_target_date')
            if estimated_date_str:
                try:
                    from datetime import datetime
                    target_date = datetime.fromisoformat(estimated_date_str)
                    target_date_str = target_date.strftime("%Y-%m-%d")
                except:
                    target_date_str = "N/A"
            
            # Format prices in USD
            entry_usd = f"${pred['entry_price']:,.2f}"
            target_usd = f"${pred['target_price']:,.2f}"
            
            values = (
                pred['id'],
                pred['symbol'],
                sentiment,
                action_label,
                wait_pct_str,
                entry_usd,
                target_usd,
                target_date_str,
                status.title(),
                result,
                '✕'
            )
            
            item = self.predictions_tree.insert('', tk.END, values=values)
            # Store prediction ID in item for easy lookup
            self.predictions_tree.set(item, 'ID', pred['id'])
            
            # Tag items for color coding
            tags = []
            if pred['verified']:
                if pred['was_correct']:
                    self.predictions_tree.set(item, 'Result', f"✓ {self.localization.t('correct')}")
                    tags.append('success')
                else:
                    self.predictions_tree.set(item, 'Result', f"✗ {self.localization.t('incorrect')}")
                    tags.append('error')
            else:
                tags.append('pending')
            
            # Add delete tag for hover effect
            tags.append('delete_cell')
            self.predictions_tree.item(item, tags=tuple(tags))
        
        # Configure tags for colors
        self.predictions_tree.tag_configure('success', foreground=theme['success'])
        self.predictions_tree.tag_configure('error', foreground=theme['error'])
        self.predictions_tree.tag_configure('pending', foreground=theme['text_secondary'])
        # Configure delete cell tag - default color (this won't affect other columns due to tag priority)
        self.predictions_tree.tag_configure('delete_cell', foreground=theme['fg'])
        # Configure delete hover tag - red color (will override other tags when applied)
        self.predictions_tree.tag_configure('delete_hover', foreground='#FF0000')  # Red on hover
        
        # Ensure click handler is bound (in case tree was recreated)
        self.predictions_tree.bind('<Button-1>', self._on_predictions_tree_click)
        self.predictions_tree.bind('<Motion>', self._on_predictions_tree_motion)
        self.predictions_tree.bind('<Leave>', self._on_predictions_tree_leave)
    
    def _create_gun_tab(self):
        """Create the Gun tab with Magazine interface for bulk predictions"""
        theme = self.theme_manager.get_theme()
        
        # Create Gun tab frame
        gun_frame = tk.Frame(self.notebook, bg=theme['frame_bg'])
        self.notebook.add(gun_frame, text="🔫 Gun")
        
        # Main container with padding
        gun_container = tk.Frame(gun_frame, bg=theme['frame_bg'])
        gun_container.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)
        
        # Title
        title_label = GradientLabel(gun_container, "🔫 Gun - Bulk Prediction Generator", 
                                   theme=theme, font=('Segoe UI', 20, 'bold'))
        title_label.pack(pady=(0, 10))
        
        # Description
        desc_label = tk.Label(gun_container, 
                             text="Load multiple stocks (bullets) and fire predictions to feed Megamind!",
                             bg=theme['frame_bg'], fg=theme['text_secondary'],
                             font=('Segoe UI', 11))
        desc_label.pack(pady=(0, 20))
        
        # === CONTROLS SECTION (always visible at top) ===
        controls_frame = tk.Frame(gun_container, bg=theme['frame_bg'])
        controls_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Strategy selection
        strategy_label = tk.Label(controls_frame, text="Strategy:", 
                                 bg=theme['frame_bg'], fg=theme['fg'],
                                 font=('Segoe UI', 10, 'bold'))
        strategy_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.gun_strategy_var = tk.StringVar(value=self.strategy_var.get())
        
        for strategy in ['trading', 'mixed', 'investing']:
            radio = tk.Radiobutton(controls_frame, 
                                  text=strategy.title(),
                                  variable=self.gun_strategy_var,
                                  value=strategy,
                                  bg=theme['frame_bg'], fg=theme['fg'],
                                  selectcolor=theme['accent'],
                                  activebackground=theme['frame_bg'],
                                  activeforeground=theme['fg'],
                                  font=('Segoe UI', 10))
            radio.pack(side=tk.LEFT, padx=5)
        
        # FIRE button (prominent, always visible)
        self.fire_btn = ModernButton(controls_frame, "🔥 FIRE", 
                                     command=self._fire_bullets,
                                     theme=theme,
                                     font=('Segoe UI', 14, 'bold'))
        self.fire_btn.config(padx=30, pady=10)
        self.fire_btn.pack(side=tk.RIGHT, padx=(20, 0))
        
        # Status label
        self.gun_status_label = tk.Label(controls_frame,
                                        text="Ready to fire",
                                        bg=theme['frame_bg'], fg=theme['text_secondary'],
                                        font=('Segoe UI', 11))
        self.gun_status_label.pack(side=tk.RIGHT, padx=(15, 10))
        
        # === MAGAZINE SECTION (expandable) ===
        chamber_card = ModernCard(gun_container, "Magazine", theme=theme, padding=20)
        chamber_card.pack(fill=tk.BOTH, expand=True)
        
        # Instructions
        instructions = tk.Label(chamber_card.content_frame,
                              text="Enter stock symbols (bullets) - one per line or comma-separated:",
                              bg=theme['card_bg'], fg=theme['fg'],
                              font=('Segoe UI', 10), anchor=tk.W, justify=tk.LEFT)
        instructions.pack(fill=tk.X, pady=(0, 10))
        
        # Bullets input area (text widget for multiple lines)
        bullets_frame = tk.Frame(chamber_card.content_frame, bg=theme['card_bg'])
        bullets_frame.pack(fill=tk.BOTH, expand=True)
        
        bullets_label = tk.Label(bullets_frame, text="Bullets:", 
                                bg=theme['card_bg'], fg=theme['fg'],
                                font=('Segoe UI', 10, 'bold'))
        bullets_label.pack(anchor=tk.W, pady=(0, 5))
        
        self.bullets_text = scrolledtext.ScrolledText(bullets_frame, 
                                                      wrap=tk.WORD, height=15,
                                                      font=('Consolas', 10),
                                                      bg=theme['entry_bg'], fg=theme['entry_fg'],
                                                      relief=tk.FLAT, bd=2,
                                                      insertbackground=theme['fg'],
                                                      highlightthickness=1,
                                                      highlightbackground=theme['accent'],
                                                      highlightcolor=theme['accent'])
        self.bullets_text.pack(fill=tk.BOTH, expand=True)
        
        # Example text
        example_text = "AAPL\nMSFT\nGOOGL\nTSLA\nAMZN\n\nOr: AAPL, MSFT, GOOGL, TSLA, AMZN"
        self.bullets_text.insert('1.0', example_text)
        
        # Store reference
        self.gun_frame = gun_frame
    
    def _fire_bullets(self):
        """Process all bullets (stocks) and generate predictions"""
        # Get bullets from text widget
        bullets_text = self.bullets_text.get('1.0', tk.END).strip()
        if not bullets_text or bullets_text == "AAPL\nMSFT\nGOOGL\nTSLA\nAMZN\n\nOr: AAPL, MSFT, GOOGL, TSLA, AMZN":
            messagebox.showwarning("No Bullets", "Please enter at least one stock symbol (bullet) to fire!")
            return
        
        # Parse symbols (support both newline and comma separation)
        symbols = []
        for line in bullets_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Check if comma-separated
            if ',' in line:
                symbols.extend([s.strip().upper() for s in line.split(',') if s.strip()])
            else:
                symbols.append(line.upper())
        
        # Remove duplicates while preserving order
        seen = set()
        symbols = [s for s in symbols if s and s not in seen and not seen.add(s)]
        
        if not symbols:
            messagebox.showwarning("No Valid Bullets", "No valid stock symbols found!")
            return
        
        # Get strategy
        strategy = self.gun_strategy_var.get()
        
        # Disable fire button
        self.fire_btn.config(state='disabled')
        self.gun_status_label.config(text=f"Firing {len(symbols)} bullets...")
        
        # Show firing window
        self._show_firing_window(symbols, strategy)
        
        # Process in background thread
        thread = threading.Thread(target=self._process_bullets_background, 
                                 args=(symbols, strategy))
        thread.daemon = True
        thread.start()
    
    def _show_firing_window(self, symbols: List[str], strategy: str):
        """Show the firing menu window"""
        theme = self.theme_manager.get_theme()
        
        # Create firing window
        firing_window = tk.Toplevel(self.root)
        firing_window.title("🔥 FIRING")
        firing_window.config(bg=theme['bg'])
        firing_window.geometry("600x500")
        firing_window.transient(self.root)
        
        # Center window
        firing_window.update_idletasks()
        x = (firing_window.winfo_screenwidth() // 2) - (600 // 2)
        y = (firing_window.winfo_screenheight() // 2) - (500 // 2)
        firing_window.geometry(f"600x500+{x}+{y}")
        
        # Header
        header = GradientLabel(firing_window, "🔥 FIRING", 
                              theme=theme, font=('Segoe UI', 24, 'bold'))
        header.pack(pady=20)
        
        # Info
        info_label = tk.Label(firing_window,
                            text=f"Processing {len(symbols)} bullets with {strategy} strategy...",
                            bg=theme['bg'], fg=theme['fg'],
                            font=('Segoe UI', 12))
        info_label.pack(pady=10)
        
        # Progress frame
        progress_frame = tk.Frame(firing_window, bg=theme['bg'])
        progress_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
        
        # Progress label
        self.firing_progress_label = tk.Label(progress_frame,
                                            text="Starting...",
                                            bg=theme['bg'], fg=theme['text_secondary'],
                                            font=('Segoe UI', 10),
                                            anchor=tk.W, justify=tk.LEFT)
        self.firing_progress_label.pack(fill=tk.X, pady=(0, 10))
        
        # Results text area
        results_frame = tk.Frame(progress_frame, bg=theme['bg'])
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.firing_results_text = scrolledtext.ScrolledText(results_frame,
                                                            wrap=tk.WORD,
                                                            font=('Consolas', 9),
                                                            bg=theme['entry_bg'], fg=theme['entry_fg'],
                                                            relief=tk.FLAT, bd=2,
                                                            highlightthickness=1,
                                                            highlightbackground=theme['accent'])
        self.firing_results_text.pack(fill=tk.BOTH, expand=True)
        
        # Store window reference
        self.firing_window = firing_window
        self.firing_results_text.insert('1.0', f"🔫 Firing {len(symbols)} bullets...\n\n")
    
    def _process_bullets_background(self, symbols: List[str], strategy: str):
        """Process bullets in background thread"""
        try:
            total = len(symbols)
            predictions_created = 0
            trend_predictions_created = 0
            errors = []
            
            for i, symbol in enumerate(symbols, 1):
                try:
                    # Update progress
                    self.root.after(0, self._update_firing_progress, 
                                  f"Processing {symbol} ({i}/{total})...", i, total)
                    
                    # Fetch data
                    stock_data = self.data_fetcher.fetch_stock_data(symbol, force_refresh=True)
                    if 'error' in stock_data:
                        error_msg = f"{symbol}: {stock_data.get('error', 'Unknown error')}"
                        errors.append(error_msg)
                        self.root.after(0, self._append_firing_result, f"❌ {error_msg}\n")
                        continue
                    
                    history_data = self.data_fetcher.fetch_stock_history(symbol)
                    if 'error' in history_data:
                        error_msg = f"{symbol}: No history data"
                        errors.append(error_msg)
                        self.root.after(0, self._append_firing_result, f"❌ {error_msg}\n")
                        continue
                    
                    # Get hybrid predictor
                    hybrid_predictor = self.hybrid_predictors[strategy]
                    
                    # Perform analysis (for predictions only, not for display)
                    if strategy == "trading":
                        analysis = hybrid_predictor.predict(stock_data, history_data, None)
                    elif strategy == "mixed":
                        financials_data = self.data_fetcher.fetch_financials(symbol)
                        analysis = hybrid_predictor.predict(stock_data, history_data, financials_data)
                    else:  # investing
                        financials_data = self.data_fetcher.fetch_financials(symbol)
                        analysis = hybrid_predictor.predict(stock_data, history_data, financials_data)
                    
                    if 'error' in analysis:
                        error_msg = f"{symbol}: Analysis error"
                        errors.append(error_msg)
                        self.root.after(0, self._append_firing_result, f"❌ {error_msg}\n")
                        continue
                    
                    # Get current price
                    current_price = stock_data.get('price', 0)
                    if current_price <= 0:
                        error_msg = f"{symbol}: Invalid price"
                        errors.append(error_msg)
                        self.root.after(0, self._append_firing_result, f"❌ {error_msg}\n")
                        continue
                    
                    # Create prediction
                    recommendation = analysis.get('recommendation', {})
                    action = recommendation.get('action', 'HOLD')
                    confidence = recommendation.get('confidence', 0)
                    reasoning = analysis.get('reasoning', '')
                    
                    # Calculate targets based on strategy
                    if action == "BUY":
                        if strategy == "trading":
                            target_price = current_price * 1.03
                            stop_loss = current_price * 0.97
                        elif strategy == "mixed":
                            target_price = current_price * 1.06
                            stop_loss = current_price * 0.95
                        else:  # investing
                            target_price = current_price * 1.12
                            stop_loss = current_price * 0.80
                    elif action == "SELL":
                        if strategy == "trading":
                            target_price = current_price * 0.97
                            stop_loss = current_price * 1.03
                        elif strategy == "mixed":
                            target_price = current_price * 0.94
                            stop_loss = current_price * 1.05
                        else:  # investing
                            target_price = current_price * 0.88
                            stop_loss = current_price * 1.20
                    else:  # HOLD
                        target_price = current_price
                        stop_loss = current_price
                    
                    # Calculate estimated days
                    # Use dynamic date from recommendation if available
                    rec = analysis.get('recommendation', {})
                    estimated_days = rec.get('estimated_days')
                    
                    if not estimated_days:
                        if strategy == "trading":
                            estimated_days = 21 # Default to medium if unknown
                        elif strategy == "mixed":
                            estimated_days = 30
                        else:  # investing
                            estimated_days = 547
                    
                    # Save prediction
                    prediction = self.predictions_tracker.add_prediction(
                        symbol, strategy, action, current_price,
                        target_price, stop_loss, confidence, reasoning, estimated_days
                    )
                    predictions_created += 1
                    self.root.after(0, self._append_firing_result, 
                                  f"✅ {symbol}: Prediction #{prediction['id']} ({action}, {confidence:.1f}%)\n")
                    
                    # Generate trend change predictions (if trading or mixed strategy)
                    if strategy in ['trading', 'mixed']:
                        try:
                            # Always get trading analysis directly for indicators and price_action
                            # The hybrid predictor's analysis might not include all fields we need
                            trading_analysis = self.trading_analyzer.analyze(stock_data, history_data)
                            
                            # Check if we have the required data
                            has_indicators = 'indicators' in trading_analysis and trading_analysis.get('indicators')
                            has_price_action = 'price_action' in trading_analysis and trading_analysis.get('price_action')
                            
                            if has_indicators and has_price_action:
                                trend_predictions = self.trend_change_predictor.predict_trend_changes(
                                    symbol,
                                    history_data,
                                    trading_analysis['indicators'],
                                    trading_analysis['price_action']
                                )
                                
                                if trend_predictions and len(trend_predictions) > 0:
                                    # Store trend change predictions
                                    for pred in trend_predictions:
                                        stored_pred = self.trend_change_tracker.add_prediction(
                                            symbol=pred['symbol'],
                                            current_trend=pred['current_trend'],
                                            predicted_change=pred['predicted_change'],
                                            estimated_days=pred['estimated_days'],
                                            estimated_date=pred['estimated_date'],
                                            confidence=pred['confidence'],
                                            reasoning=pred['reasoning'],
                                            key_indicators=pred.get('key_indicators', {})
                                        )
                                        
                                        # Record in learning tracker
                                        self.learning_tracker.record_trend_change_prediction(
                                            symbol=pred['symbol'],
                                            predicted_change=pred['predicted_change'],
                                            estimated_days=pred['estimated_days'],
                                            confidence=pred['confidence']
                                        )
                                        
                                        trend_predictions_created += 1
                                        self.root.after(0, self._append_firing_result,
                                                      f"  📊 Trend change prediction #{stored_pred['id']}: {pred['predicted_change']} ({pred['confidence']:.1f}%)\n")
                                else:
                                    # No trend predictions generated (might be normal - not all stocks have trend changes)
                                    logger.debug(f"No trend change predictions generated for {symbol} (may be normal)")
                            else:
                                # Missing required data
                                missing = []
                                if not has_indicators:
                                    missing.append("indicators")
                                if not has_price_action:
                                    missing.append("price_action")
                                logger.warning(f"Missing data for trend predictions ({symbol}): {', '.join(missing)}")
                                self.root.after(0, self._append_firing_result,
                                              f"  ⚠️ Trend prediction skipped for {symbol}: Missing {', '.join(missing)}\n")
                        except Exception as e:
                            logger.warning(f"Could not generate trend change predictions for {symbol}: {e}")
                            self.root.after(0, self._append_firing_result,
                                          f"  ⚠️ Trend prediction skipped for {symbol}: {str(e)[:50]}\n")
                    
                except Exception as e:
                    error_msg = f"{symbol}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(f"Error processing bullet {symbol}: {e}")
                    self.root.after(0, self._append_firing_result, f"❌ {error_msg}\n")
            
            # Update displays
            self.root.after(0, self._update_predictions_display)
            self.root.after(0, self._update_trend_change_display)
            
            # Show completion message
            summary = f"\n{'='*50}\n"
            summary += f"✅ Firing Complete!\n\n"
            summary += f"Predictions created: {predictions_created}\n"
            summary += f"Trend change predictions: {trend_predictions_created}\n"
            summary += f"Errors: {len(errors)}\n"
            if errors:
                summary += f"\nErrors:\n"
                for err in errors[:10]:  # Show first 10 errors
                    summary += f"  • {err}\n"
                if len(errors) > 10:
                    summary += f"  ... and {len(errors) - 10} more\n"
            
            self.root.after(0, self._append_firing_result, summary)
            self.root.after(0, self._update_firing_progress, 
                          f"Complete! {predictions_created} predictions, {trend_predictions_created} trend changes", 
                          total, total)
            
            # Re-enable fire button
            self.root.after(0, lambda: self.fire_btn.config(state='normal'))
            self.root.after(0, lambda: self.gun_status_label.config(text="Ready to fire"))
            
        except Exception as e:
            logger.error(f"Error in _process_bullets_background: {e}")
            self.root.after(0, self._append_firing_result, f"\n❌ Fatal error: {str(e)}\n")
            self.root.after(0, lambda: self.fire_btn.config(state='normal'))
            self.root.after(0, lambda: self.gun_status_label.config(text="Error occurred"))
    
    def _update_firing_progress(self, message: str, current: int, total: int):
        """Update firing progress"""
        if hasattr(self, 'firing_progress_label'):
            self.firing_progress_label.config(text=message)
    
    def _append_firing_result(self, text: str):
        """Append text to firing results"""
        if hasattr(self, 'firing_results_text'):
            self.firing_results_text.insert(tk.END, text)
            self.firing_results_text.see(tk.END)
    
    def _create_risk_management_tab(self):
        """Create the Risk Management tab"""
        theme = self.theme_manager.get_theme()
        
        # Header
        header_frame = tk.Frame(self.risk_management_frame, bg=theme['bg'])
        header_frame.pack(fill=tk.X, pady=(0, 15), padx=15)
        
        title_label = tk.Label(header_frame, text="🛡️ Risk Management", 
                              font=('Segoe UI', 14, 'bold'), bg=theme['bg'], fg=theme['fg'])
        title_label.pack(side=tk.LEFT)
        
        # Container with scroll
        container = tk.Frame(self.risk_management_frame, bg=theme['bg'])
        container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        canvas = tk.Canvas(container, bg=theme['bg'], highlightthickness=0)
        scrollbar = ModernScrollbar(container, command=canvas.yview, theme=theme)
        scrollable_frame = tk.Frame(canvas, bg=theme['bg'])
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Position sizing card
        position_card = ModernCard(scrollable_frame, "Position Sizing Calculator", theme=theme, padding=15)
        position_card.pack(fill=tk.X, pady=(0, 15))
        
        pos_frame = tk.Frame(position_card, bg=theme['card_bg'])
        pos_frame.pack(fill=tk.X, padx=15, pady=15)
        
        # Input fields
        tk.Label(pos_frame, text="Entry Price:", bg=theme['card_bg'], fg=theme['fg']).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.rm_entry_price = tk.Entry(pos_frame, width=15)
        self.rm_entry_price.grid(row=0, column=1, pady=5, padx=5)
        
        tk.Label(pos_frame, text="Stop Loss:", bg=theme['card_bg'], fg=theme['fg']).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.rm_stop_loss = tk.Entry(pos_frame, width=15)
        self.rm_stop_loss.grid(row=1, column=1, pady=5, padx=5)
        
        tk.Label(pos_frame, text="Confidence (%):", bg=theme['card_bg'], fg=theme['fg']).grid(row=2, column=0, sticky=tk.W, pady=5)
        self.rm_confidence = tk.Entry(pos_frame, width=15)
        self.rm_confidence.insert(0, "80")
        self.rm_confidence.grid(row=2, column=1, pady=5, padx=5)
        
        tk.Label(pos_frame, text="Method:", bg=theme['card_bg'], fg=theme['fg']).grid(row=3, column=0, sticky=tk.W, pady=5)
        self.rm_method = ttk.Combobox(pos_frame, values=['fixed_fraction', 'kelly', 'optimal_f'], width=12, state='readonly')
        self.rm_method.set('fixed_fraction')
        self.rm_method.grid(row=3, column=1, pady=5, padx=5)
        
        calc_btn = ModernButton(pos_frame, text="Calculate Position Size", 
                               command=self._calculate_position_size_ui, theme=theme)
        calc_btn.grid(row=4, column=0, columnspan=2, pady=10)
        
        # Results
        self.rm_position_results = scrolledtext.ScrolledText(pos_frame, height=8, wrap=tk.WORD,
                                                             bg=theme['card_bg'], fg=theme['fg'],
                                                             font=('Consolas', 9))
        self.rm_position_results.grid(row=5, column=0, columnspan=2, pady=5, sticky=tk.EW)
        
        # Stop Loss Recommendations card
        stoploss_card = ModernCard(scrollable_frame, "Stop Loss Recommendations", theme=theme, padding=15)
        stoploss_card.pack(fill=tk.X, pady=(0, 15))
        
        sl_frame = tk.Frame(stoploss_card, bg=theme['card_bg'])
        sl_frame.pack(fill=tk.X, padx=15, pady=15)
        
        tk.Label(sl_frame, text="Current Price:", bg=theme['card_bg'], fg=theme['fg']).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.rm_current_price = tk.Entry(sl_frame, width=15)
        self.rm_current_price.grid(row=0, column=1, pady=5, padx=5)
        
        tk.Label(sl_frame, text="Volatility (ATR):", bg=theme['card_bg'], fg=theme['fg']).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.rm_volatility = tk.Entry(sl_frame, width=15)
        self.rm_volatility.grid(row=1, column=1, pady=5, padx=5)
        
        tk.Label(sl_frame, text="Method:", bg=theme['card_bg'], fg=theme['fg']).grid(row=2, column=0, sticky=tk.W, pady=5)
        self.rm_sl_method = ttk.Combobox(sl_frame, values=['atr', 'percentage', 'support', 'trailing'], width=12, state='readonly')
        self.rm_sl_method.set('atr')
        self.rm_sl_method.grid(row=2, column=1, pady=5, padx=5)
        
        calc_sl_btn = ModernButton(sl_frame, text="Get Stop Loss", 
                                   command=self._calculate_stop_loss_ui, theme=theme)
        calc_sl_btn.grid(row=3, column=0, columnspan=2, pady=10)
        
        self.rm_stoploss_results = scrolledtext.ScrolledText(sl_frame, height=6, wrap=tk.WORD,
                                                             bg=theme['card_bg'], fg=theme['fg'],
                                                             font=('Consolas', 9))
        self.rm_stoploss_results.grid(row=4, column=0, columnspan=2, pady=5, sticky=tk.EW)
        
        # Portfolio Risk Metrics card
        risk_card = ModernCard(scrollable_frame, "Portfolio Risk Metrics", theme=theme, padding=15)
        risk_card.pack(fill=tk.X, pady=(0, 15))
        
        risk_frame = tk.Frame(risk_card, bg=theme['card_bg'])
        risk_frame.pack(fill=tk.X, padx=15, pady=15)
        
        calc_risk_btn = ModernButton(risk_frame, text="Calculate Portfolio Risk", 
                                     command=self._calculate_portfolio_risk_ui, theme=theme)
        calc_risk_btn.pack(pady=10)
        
        self.rm_portfolio_results = scrolledtext.ScrolledText(risk_frame, height=10, wrap=tk.WORD,
                                                              bg=theme['card_bg'], fg=theme['fg'],
                                                              font=('Consolas', 9))
        self.rm_portfolio_results.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Update scroll region on mousewheel
        def on_rm_scroll(event):
            delta = event.delta if hasattr(event, 'delta') else (event.num == 4 and -1) or 1
            if isinstance(delta, (int, float)):
                scroll_amount = int(-1 * (delta / 120)) * 3
                canvas.yview_scroll(scroll_amount, "units")
        canvas.bind("<MouseWheel>", on_rm_scroll)
        scrollable_frame.bind("<MouseWheel>", on_rm_scroll)
    
    def _create_advanced_analytics_tab(self):
        """Create the Advanced Analytics tab"""
        theme = self.theme_manager.get_theme()
        
        # Header
        header_frame = tk.Frame(self.advanced_analytics_frame, bg=theme['bg'])
        header_frame.pack(fill=tk.X, pady=(0, 15), padx=15)
        
        title_label = tk.Label(header_frame, text="📊 Advanced Analytics", 
                              font=('Segoe UI', 14, 'bold'), bg=theme['bg'], fg=theme['fg'])
        title_label.pack(side=tk.LEFT)
        
        # Container with scroll
        container = tk.Frame(self.advanced_analytics_frame, bg=theme['bg'])
        container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        canvas = tk.Canvas(container, bg=theme['bg'], highlightthickness=0)
        scrollbar = ModernScrollbar(container, command=canvas.yview, theme=theme)
        scrollable_frame = tk.Frame(canvas, bg=theme['bg'])
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Correlation Analysis card
        corr_card = ModernCard(scrollable_frame, "Correlation Analysis", theme=theme, padding=15)
        corr_card.pack(fill=tk.X, pady=(0, 15))
        
        corr_frame = tk.Frame(corr_card, bg=theme['card_bg'])
        corr_frame.pack(fill=tk.X, padx=15, pady=15)
        
        tk.Label(corr_frame, text="Stock Symbols (comma-separated):", bg=theme['card_bg'], fg=theme['fg']).pack(anchor=tk.W, pady=5)
        self.aa_symbols_entry = tk.Entry(corr_frame, width=50)
        self.aa_symbols_entry.insert(0, "AAPL,MSFT,GOOGL,AMZN")
        self.aa_symbols_entry.pack(fill=tk.X, pady=5)
        
        calc_corr_btn = ModernButton(corr_frame, text="Calculate Correlation", 
                                     command=self._calculate_correlation_ui, theme=theme)
        calc_corr_btn.pack(pady=10)
        
        self.aa_correlation_results = scrolledtext.ScrolledText(corr_frame, height=8, wrap=tk.WORD,
                                                                bg=theme['card_bg'], fg=theme['fg'],
                                                                font=('Consolas', 9))
        self.aa_correlation_results.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Portfolio Optimization card
        opt_card = ModernCard(scrollable_frame, "Portfolio Optimization", theme=theme, padding=15)
        opt_card.pack(fill=tk.X, pady=(0, 15))
        
        opt_frame = tk.Frame(opt_card, bg=theme['card_bg'])
        opt_frame.pack(fill=tk.X, padx=15, pady=15)
        
        tk.Label(opt_frame, text="Stock Symbols (comma-separated):", bg=theme['card_bg'], fg=theme['fg']).pack(anchor=tk.W, pady=5)
        self.aa_opt_symbols_entry = tk.Entry(opt_frame, width=50)
        self.aa_opt_symbols_entry.insert(0, "AAPL,MSFT,GOOGL")
        self.aa_opt_symbols_entry.pack(fill=tk.X, pady=5)
        
        tk.Label(opt_frame, text="Optimization Method:", bg=theme['card_bg'], fg=theme['fg']).pack(anchor=tk.W, pady=5)
        self.aa_opt_method = ttk.Combobox(opt_frame, values=['sharpe', 'min_variance', 'max_return'], width=15, state='readonly')
        self.aa_opt_method.set('sharpe')
        self.aa_opt_method.pack(anchor=tk.W, pady=5)
        
        calc_opt_btn = ModernButton(opt_frame, text="Optimize Portfolio", 
                                    command=self._optimize_portfolio_ui, theme=theme)
        calc_opt_btn.pack(pady=10)
        
        self.aa_optimization_results = scrolledtext.ScrolledText(opt_frame, height=8, wrap=tk.WORD,
                                                                 bg=theme['card_bg'], fg=theme['fg'],
                                                                 font=('Consolas', 9))
        self.aa_optimization_results.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Monte Carlo Simulation card
        mc_card = ModernCard(scrollable_frame, "Monte Carlo Simulation", theme=theme, padding=15)
        mc_card.pack(fill=tk.X, pady=(0, 15))
        
        mc_frame = tk.Frame(mc_card, bg=theme['card_bg'])
        mc_frame.pack(fill=tk.X, padx=15, pady=15)
        
        mc_input_frame = tk.Frame(mc_frame, bg=theme['card_bg'])
        mc_input_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(mc_input_frame, text="Initial Price:", bg=theme['card_bg'], fg=theme['fg']).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.aa_mc_price = tk.Entry(mc_input_frame, width=15)
        self.aa_mc_price.grid(row=0, column=1, pady=5, padx=5)
        
        tk.Label(mc_input_frame, text="Expected Return (daily):", bg=theme['card_bg'], fg=theme['fg']).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.aa_mc_return = tk.Entry(mc_input_frame, width=15)
        self.aa_mc_return.insert(0, "0.001")
        self.aa_mc_return.grid(row=1, column=1, pady=5, padx=5)
        
        tk.Label(mc_input_frame, text="Volatility (daily):", bg=theme['card_bg'], fg=theme['fg']).grid(row=2, column=0, sticky=tk.W, pady=5)
        self.aa_mc_vol = tk.Entry(mc_input_frame, width=15)
        self.aa_mc_vol.insert(0, "0.02")
        self.aa_mc_vol.grid(row=2, column=1, pady=5, padx=5)
        
        tk.Label(mc_input_frame, text="Days:", bg=theme['card_bg'], fg=theme['fg']).grid(row=3, column=0, sticky=tk.W, pady=5)
        self.aa_mc_days = tk.Entry(mc_input_frame, width=15)
        self.aa_mc_days.insert(0, "252")
        self.aa_mc_days.grid(row=3, column=1, pady=5, padx=5)
        
        calc_mc_btn = ModernButton(mc_frame, text="Run Simulation", 
                                   command=self._run_monte_carlo_ui, theme=theme)
        calc_mc_btn.pack(pady=10)
        
        self.aa_mc_results = scrolledtext.ScrolledText(mc_frame, height=8, wrap=tk.WORD,
                                                       bg=theme['card_bg'], fg=theme['fg'],
                                                       font=('Consolas', 9))
        self.aa_mc_results.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Update scroll region on mousewheel
        def on_aa_scroll(event):
            delta = event.delta if hasattr(event, 'delta') else (event.num == 4 and -1) or 1
            if isinstance(delta, (int, float)):
                scroll_amount = int(-1 * (delta / 120)) * 3
                canvas.yview_scroll(scroll_amount, "units")
        canvas.bind("<MouseWheel>", on_aa_scroll)
        scrollable_frame.bind("<MouseWheel>", on_aa_scroll)
    
    def _create_walk_forward_tab(self):
        """Create the Walk-Forward Backtesting tab"""
        theme = self.theme_manager.get_theme()
        
        # Header
        header_frame = tk.Frame(self.walk_forward_frame, bg=theme['bg'])
        header_frame.pack(fill=tk.X, pady=(0, 15), padx=15)
        
        title_label = tk.Label(header_frame, text="🔄 Walk-Forward Backtesting", 
                              font=('Segoe UI', 14, 'bold'), bg=theme['bg'], fg=theme['fg'])
        title_label.pack(side=tk.LEFT)
        
        # Info
        info_label = tk.Label(self.walk_forward_frame, 
                             text="Walk-forward backtesting prevents overfitting by training on historical data\n"
                                  "and testing on forward data, then moving the window forward.",
                             font=('Segoe UI', 9), bg=theme['bg'], fg=theme['text_secondary'],
                             justify=tk.LEFT)
        info_label.pack(fill=tk.X, pady=(0, 15), padx=15)
        
        # Container
        container = tk.Frame(self.walk_forward_frame, bg=theme['bg'])
        container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Parameters card
        params_card = ModernCard(container, "Parameters", theme=theme, padding=15)
        params_card.pack(fill=tk.X, pady=(0, 15))
        
        params_frame = tk.Frame(params_card, bg=theme['card_bg'])
        params_frame.pack(fill=tk.X, padx=15, pady=15)
        
        tk.Label(params_frame, text="Train Window (days):", bg=theme['card_bg'], fg=theme['fg']).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.wf_train_window = tk.Entry(params_frame, width=15)
        self.wf_train_window.insert(0, "252")
        self.wf_train_window.grid(row=0, column=1, pady=5, padx=5)
        
        tk.Label(params_frame, text="Test Window (days):", bg=theme['card_bg'], fg=theme['fg']).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.wf_test_window = tk.Entry(params_frame, width=15)
        self.wf_test_window.insert(0, "63")
        self.wf_test_window.grid(row=1, column=1, pady=5, padx=5)
        
        tk.Label(params_frame, text="Step Size (days):", bg=theme['card_bg'], fg=theme['fg']).grid(row=2, column=0, sticky=tk.W, pady=5)
        self.wf_step_size = tk.Entry(params_frame, width=15)
        self.wf_step_size.insert(0, "21")
        self.wf_step_size.grid(row=2, column=1, pady=5, padx=5)
        
        # Multi-stock option
        self.wf_multi_stock = tk.BooleanVar(value=False)
        multi_stock_check = tk.Checkbutton(
            params_frame,
            text="Test Multiple Stocks (More Robust)",
            variable=self.wf_multi_stock,
            bg=theme['card_bg'],
            fg=theme['fg'],
            selectcolor=theme['card_bg'],
            activebackground=theme['card_bg'],
            activeforeground=theme['fg']
        )
        multi_stock_check.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        multi_stock_frame = tk.Frame(params_frame, bg=theme['card_bg'])
        multi_stock_frame.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=5)
        tk.Label(multi_stock_frame, text="Max stocks:", bg=theme['card_bg'], fg=theme['fg']).pack(side=tk.LEFT, padx=(20, 5))
        self.wf_max_stocks = tk.Entry(multi_stock_frame, width=10)
        self.wf_max_stocks.insert(0, "10")
        self.wf_max_stocks.pack(side=tk.LEFT)
        
        run_wf_btn = ModernButton(params_frame, text="Run Walk-Forward Test", 
                                  command=self._run_walk_forward_ui, theme=theme)
        run_wf_btn.grid(row=5, column=0, columnspan=2, pady=10)
        
        # Retraining recommendation display (initially hidden)
        self.wf_retrain_frame = tk.Frame(params_card, bg=theme['card_bg'])
        self.wf_retrain_label = tk.Label(self.wf_retrain_frame, 
                                        text="", 
                                        font=('Segoe UI', 10, 'bold'),
                                        bg=theme['card_bg'],
                                        fg=theme['fg'],
                                        wraplength=500,
                                        justify=tk.LEFT)
        self.wf_retrain_label.pack(fill=tk.X, padx=15, pady=10)
        
        # Results
        self.wf_results = scrolledtext.ScrolledText(container, height=20, wrap=tk.WORD,
                                                    bg=theme['frame_bg'], fg=theme['fg'],
                                                    font=('Consolas', 9))
        self.wf_results.pack(fill=tk.BOTH, expand=True, pady=5)
        self.wf_results.insert('1.0', "Walk-forward backtesting results will appear here.\n\n"
                                      "This method helps validate that your strategy works on unseen data\n"
                                      "and prevents overfitting to historical patterns.")
        self.wf_results.config(state='disabled')
    
    def _create_alerts_tab(self):
        """Create the Alert System tab"""
        theme = self.theme_manager.get_theme()
        
        # Header
        header_frame = tk.Frame(self.alerts_frame, bg=theme['bg'])
        header_frame.pack(fill=tk.X, pady=(0, 15), padx=15)
        
        title_frame = tk.Frame(header_frame, bg=theme['bg'])
        title_frame.pack(side=tk.LEFT)
        
        title_label = tk.Label(title_frame, text="🔔 Alert System", 
                              font=('Segoe UI', 14, 'bold'), bg=theme['bg'], fg=theme['fg'])
        title_label.pack(side=tk.LEFT)
        
        unread_count = 0
        if self.alert_system:
            unread_count = len(self.alert_system.get_unread_alerts())
            if unread_count > 0:
                count_label = tk.Label(title_frame, text=f" ({unread_count} unread)", 
                                      font=('Segoe UI', 10), bg=theme['bg'], fg=theme['accent'])
                count_label.pack(side=tk.LEFT, padx=(5, 0))
        
        refresh_btn = ModernButton(header_frame, text="Refresh", 
                                   command=self._refresh_alerts, theme=theme)
        refresh_btn.pack(side=tk.RIGHT)
        
        mark_all_read_btn = ModernButton(header_frame, text="Mark All Read", 
                                         command=self._mark_all_alerts_read, theme=theme)
        mark_all_read_btn.pack(side=tk.RIGHT, padx=(0, 5))
        
        # Container with scroll
        container = tk.Frame(self.alerts_frame, bg=theme['bg'])
        container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        canvas = tk.Canvas(container, bg=theme['bg'], highlightthickness=0)
        scrollbar = ModernScrollbar(container, command=canvas.yview, theme=theme)
        scrollable_frame = tk.Frame(canvas, bg=theme['bg'])
        
        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Create alerts list frame
        self.alerts_list_frame = scrollable_frame
        
        # Create alert button - store reference so it doesn't get destroyed
        self.create_alert_card = ModernCard(scrollable_frame, "Create Alert", theme=theme, padding=15)
        self.create_alert_card.pack(fill=tk.X, pady=(0, 15))
        
        # Use content_frame instead of creating a new frame
        create_frame = self.create_alert_card.content_frame
        
        tk.Label(create_frame, text="Alert Type:", bg=theme['card_bg'], fg=theme['fg']).grid(row=0, column=0, sticky=tk.W, pady=5)
        self.alert_type = ttk.Combobox(create_frame, values=['price_alert', 'prediction_alert', 'portfolio_alert'], width=20, state='readonly')
        self.alert_type.set('price_alert')
        self.alert_type.grid(row=0, column=1, pady=5, padx=5)
        
        tk.Label(create_frame, text="Symbol:", bg=theme['card_bg'], fg=theme['fg']).grid(row=1, column=0, sticky=tk.W, pady=5)
        self.alert_symbol = tk.Entry(create_frame, width=20)
        self.alert_symbol.grid(row=1, column=1, pady=5, padx=5)
        
        tk.Label(create_frame, text="Message:", bg=theme['card_bg'], fg=theme['fg']).grid(row=2, column=0, sticky=tk.W, pady=5)
        self.alert_message = tk.Entry(create_frame, width=40)
        self.alert_message.grid(row=2, column=1, pady=5, padx=5)
        
        tk.Label(create_frame, text="Priority:", bg=theme['card_bg'], fg=theme['fg']).grid(row=3, column=0, sticky=tk.W, pady=5)
        self.alert_priority = ttk.Combobox(create_frame, values=['low', 'medium', 'high', 'critical'], width=20, state='readonly')
        self.alert_priority.set('medium')
        self.alert_priority.grid(row=3, column=1, pady=5, padx=5)
        
        create_alert_btn = ModernButton(create_frame, text="Create Alert", 
                                       command=self._create_alert_ui, theme=theme)
        create_alert_btn.grid(row=4, column=0, columnspan=2, pady=10)
        
        # Update scroll region on mousewheel
        def on_alerts_scroll(event):
            delta = event.delta if hasattr(event, 'delta') else (event.num == 4 and -1) or 1
            if isinstance(delta, (int, float)):
                scroll_amount = int(-1 * (delta / 120)) * 3
                canvas.yview_scroll(scroll_amount, "units")
        canvas.bind("<MouseWheel>", on_alerts_scroll)
        scrollable_frame.bind("<MouseWheel>", on_alerts_scroll)
        
        # Initial display
        self._refresh_alerts()

    def _create_holdings_tab(self):
        """Create the My Holdings tab"""
        theme = self.theme_manager.get_theme()
        
        # Header
        header_frame = tk.Frame(self.holdings_frame, bg=theme['bg'])
        header_frame.pack(fill=tk.X, pady=(0, 15), padx=15)
        
        title_label = tk.Label(header_frame, text="💼 My Portfolio Holdings", 
                              font=('Segoe UI', 14, 'bold'), bg=theme['bg'], fg=theme['fg'])
        title_label.pack(side=tk.LEFT)
        
        # Action Buttons
        refresh_btn = ModernButton(header_frame, text="🔄 Refresh Prices", 
                                   command=self._refresh_holdings, theme=theme)
        refresh_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        add_btn = ModernButton(header_frame, text="➕ Add Holding", 
                               command=self._show_add_holding_dialog, theme=theme)
        add_btn.pack(side=tk.RIGHT)
        
        # Container with scroll
        container = tk.Frame(self.holdings_frame, bg=theme['bg'])
        container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Configure columns
        columns = ('symbol', 'buy_price', 'current_price', 'quantity', 'cost_basis', 'value', 'pl', 'recommendation', 'actions')
        self.holdings_tree = ttk.Treeview(container, columns=columns, show='headings', selectmode='browse')
        
        # Setup headers
        self.holdings_tree.heading('symbol', text='Symbol')
        self.holdings_tree.heading('buy_price', text='Buy Price')
        self.holdings_tree.heading('current_price', text='Current Price')
        self.holdings_tree.heading('quantity', text='Quantity')
        self.holdings_tree.heading('cost_basis', text='Cost Basis')
        self.holdings_tree.heading('value', text='Market Value')
        self.holdings_tree.heading('pl', text='P/L')
        self.holdings_tree.heading('recommendation', text='Status')
        self.holdings_tree.heading('actions', text='Actions')
        
        # Setup columns
        self.holdings_tree.column('symbol', width=80, anchor='center')
        self.holdings_tree.column('buy_price', width=100, anchor='center')
        self.holdings_tree.column('current_price', width=100, anchor='center')
        self.holdings_tree.column('quantity', width=80, anchor='center')
        self.holdings_tree.column('cost_basis', width=100, anchor='center')
        self.holdings_tree.column('value', width=100, anchor='center')
        self.holdings_tree.column('pl', width=120, anchor='center')
        self.holdings_tree.column('recommendation', width=150, anchor='center')
        self.holdings_tree.column('actions', width=100, anchor='center')
        
        # Add scrollbar
        scrollbar = ModernScrollbar(container, command=self.holdings_tree.yview, theme=theme)
        self.holdings_tree.configure(yscrollcommand=scrollbar.set)
        
        self.holdings_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Style treeview
        style = ttk.Style()
        style.configure("Treeview", 
                       background=theme['entry_bg'],
                       foreground=theme['fg'],
                       fieldbackground=theme['entry_bg'],
                       font=('Segoe UI', 10),
                       rowheight=35)
        style.configure("Treeview.Heading",
                       background=theme['secondary_bg'],
                       foreground=theme['fg'],
                       font=('Segoe UI', 10, 'bold'))
        style.map("Treeview", background=[('selected', theme['accent'])])
        
        # Bind double click to show details
        self.holdings_tree.bind("<Double-1>", self._on_holding_double_click)
        self.holdings_tree.bind("<Delete>", self._delete_selected_holding)
        
        # Summary footer
        footer_frame = tk.Frame(self.holdings_frame, bg=theme['bg'])
        footer_frame.pack(fill=tk.X, padx=15, pady=(0, 15))
        
        self.holdings_summary = tk.Label(
            footer_frame,
            text="Total Value: $0.00 | Total P/L: $0.00 (0.00%)",
            font=('Segoe UI', 11, 'bold'),
            bg=theme['bg'],
            fg=theme['fg']
        )
        self.holdings_summary.pack(side=tk.LEFT)
        
        # Initialize
        self._refresh_holdings()
        
        # Schedule auto-refresh every 5 minutes (if not already scheduled)
        if not hasattr(self, '_holdings_refresh_scheduled'):
             self.root.after(300000, self._auto_refresh_holdings)
             self._holdings_refresh_scheduled = True

    def _show_add_holding_dialog(self):
        """Show dialog to add a new holding"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add New Holding")
        dialog.geometry("400x450")
        dialog.transient(self.root)
        dialog.grab_set()
        
        theme = self.theme_manager.get_theme()
        dialog.config(bg=theme['bg'])
        
        header = tk.Label(dialog, text="Add Portfolio Holding", font=('Segoe UI', 14, 'bold'),
                         bg=theme['bg'], fg=theme['accent'])
        header.pack(pady=20)
        
        form_frame = tk.Frame(dialog, bg=theme['bg'])
        form_frame.pack(fill=tk.BOTH, expand=True, padx=40)
        
        # Symbol
        tk.Label(form_frame, text="Stock Symbol:", bg=theme['bg'], fg=theme['fg']).pack(anchor=tk.W, pady=(10, 5))
        symbol_entry = tk.Entry(form_frame, width=30)
        symbol_entry.pack(fill=tk.X)
        symbol_entry.focus_set()
        
        # Price
        tk.Label(form_frame, text="Purchase Price ($):", bg=theme['bg'], fg=theme['fg']).pack(anchor=tk.W, pady=(15, 5))
        price_entry = tk.Entry(form_frame, width=30)
        price_entry.pack(fill=tk.X)
        
        # Quantity
        tk.Label(form_frame, text="Quantity:", bg=theme['bg'], fg=theme['fg']).pack(anchor=tk.W, pady=(15, 5))
        qty_entry = tk.Entry(form_frame, width=30)
        qty_entry.pack(fill=tk.X)
        
        # Date
        tk.Label(form_frame, text="Purchase Date (YYYY-MM-DD):", bg=theme['bg'], fg=theme['fg']).pack(anchor=tk.W, pady=(15, 5))
        date_entry = tk.Entry(form_frame, width=30)
        date_entry.insert(0, datetime.now().strftime('%Y-%m-%d'))
        date_entry.pack(fill=tk.X)
        
        def save():
            symbol = symbol_entry.get().strip().upper()
            price_str = price_entry.get().strip()
            qty_str = qty_entry.get().strip()
            date_str = date_entry.get().strip()
            
            if not symbol or not price_str or not qty_str:
                messagebox.showerror("Error", "Please fill in all fields")
                return
            
            try:
                price = float(price_str)
                qty = float(qty_str)
            except ValueError:
                messagebox.showerror("Error", "Price and Quantity must be numbers")
                return
            
            self.holdings_tracker.add_holding(symbol, price, qty, date_str)
            self._refresh_holdings()
            dialog.destroy()
            messagebox.showinfo("Success", f"Added {symbol} to portfolio")
        
        btn_frame = tk.Frame(dialog, bg=theme['bg'])
        btn_frame.pack(pady=30)
        
        ModernButton(btn_frame, "Cancel", command=dialog.destroy, theme=theme).pack(side=tk.LEFT, padx=10)
        ModernButton(btn_frame, "Save Holding", command=save, theme=theme).pack(side=tk.LEFT, padx=10)

    def _refresh_holdings(self):
        """Refresh holdings data and UI"""
        if not hasattr(self, 'holdings_tracker'):
            return
            
        # Clear tree
        for item in self.holdings_tree.get_children():
            self.holdings_tree.delete(item)
        
        # Get analyzed holdings
        # Run in background to avoid freezing UI
        def analyze_background():
            results = self.holdings_tracker.analyze_all_holdings()
            self.root.after(0, lambda: self._update_holdings_ui(results))
        
        threading.Thread(target=analyze_background, daemon=True).start()

    def _update_holdings_ui(self, results):
        """Update holdings UI with analysis results"""
        total_value = 0
        total_cost = 0
        
        for r in results:
            h = r['holding']
            total_value += r['total_value']
            total_cost += r['total_cost']
            
            # Format values
            pl_color = '#4caf50' if r['profit_loss'] >= 0 else '#f44336'
            pl_text = f"${r['profit_loss']:,.2f} ({r['profit_loss_pct']:+.1f}%)"
            
            status = r['recommendation']
            status_icon = "🟢" if status == 'HOLD' else "🔴" if 'SELL' in status else "🟡"
            
            # Tags for row coloring
            tag = 'sell' if 'SELL' in status else 'hold'
            
            self.holdings_tree.insert('', 'end', values=(
                h.symbol,
                f"${h.buy_price:,.2f}",
                f"${r['current_price']:,.2f}",
                f"{h.quantity}",
                f"${r['total_cost']:,.2f}",
                f"${r['total_value']:,.2f}",
                pl_text,
                f"{status_icon} {status}",
                "🗑️ Delete" if 'SELL' in status else "Analyze"
            ), tags=(tag, str(h.id)))
        
        # Update summary
        total_pl = total_value - total_cost
        total_pl_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0
        pl_color = '#4caf50' if total_pl >= 0 else '#f44336'
        
        self.holdings_summary.config(
            text=f"Total Value: ${total_value:,.2f} | Total P/L: ${total_pl:,.2f} ({total_pl_pct:+.2f}%)",
            fg=pl_color
        )
        
        # Configure tag colors
        theme = self.theme_manager.get_theme()
        sell_bg = '#ffebee' if theme['name'] == 'light' else '#3e2723'
        self.holdings_tree.tag_configure('sell', background=sell_bg)
        
    def _on_holding_double_click(self, event):
        """Handle double click on holding"""
        try:
            item = self.holdings_tree.selection()[0]
            tags = self.holdings_tree.item(item, "tags")
            holding_id = int(tags[1]) if len(tags) > 1 else 0
            if holding_id:
                self._show_holding_details(holding_id)
        except IndexError:
            pass

    def _show_holding_details(self, holding_id):
        """Show detailed analysis for a holding"""
        holding = self.holdings_tracker.get_holding(holding_id)
        if not holding:
            return
            
        # Show loading indicator first
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Analysis: {holding.symbol}")
        dialog.geometry("500x600")
        theme = self.theme_manager.get_theme()
        dialog.config(bg=theme['bg'])
        
        def load_analysis():
            analysis = self.holdings_tracker.analyze_holding(holding)
            self.root.after(0, lambda: update_dialog(analysis))
            
        def update_dialog(analysis):
            # Clear dialog
            for widget in dialog.winfo_children():
                widget.destroy()
                
            # Header
            header = tk.Frame(dialog, bg=theme['bg'])
            header.pack(fill=tk.X, pady=20, padx=20)
            
            tk.Label(header, text=holding.symbol, font=('Segoe UI', 24, 'bold'),
                    bg=theme['bg'], fg=theme['accent']).pack(side=tk.LEFT)
            
            status_color = '#f44336' if 'SELL' in analysis['recommendation'] else '#4caf50'
            tk.Label(header, text=analysis['recommendation'], font=('Segoe UI', 14, 'bold'),
                    bg=status_color, fg='white', padx=10, pady=5).pack(side=tk.RIGHT)
            
            # Content
            content = scrolledtext.ScrolledText(dialog, height=20, font=('Segoe UI', 10))
            content.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
            
            text = f"""POSITION DETAILS
----------------
Buy Price:    ${holding.buy_price:.2f}
Quantity:     {holding.quantity}
Cost Basis:   ${analysis['total_cost']:.2f}
Current Price: ${analysis['current_price']:.2f}
Market Value: ${analysis['total_value']:.2f}
Profit/Loss:  ${analysis['profit_loss']:+.2f} ({analysis['profit_loss_pct']:+.1f}%)

ANALYSIS
----------------
Confidence: {analysis.get('confidence', 0)}%

SELL SIGNALS:
"""
            if analysis['sell_signals']:
                for signal in analysis['sell_signals']:
                    text += f"• {signal}\n"
            else:
                text += "• None detected (Safe to Hold)\n"
                
            if analysis['reasoning']:
                text += "\nNOTES:\n"
                for note in analysis['reasoning']:
                    text += f"• {note}\n"
                    
            content.insert('1.0', text)
            content.config(state='disabled')
            
            # Buttons
            btn_frame = tk.Frame(dialog, bg=theme['bg'])
            btn_frame.pack(pady=20)
            
            def sell_holding():
                if messagebox.askyesno("Confirm Sell", f"Are you sure you want to remove {holding.symbol} from portfolio?"):
                    self.holdings_tracker.remove_holding(holding.id)
                    self._refresh_holdings()
                    dialog.destroy()
            
            ModernButton(btn_frame, "Close", command=dialog.destroy, theme=theme).pack(side=tk.LEFT, padx=10)
            ModernButton(btn_frame, "Mark as Sold", command=sell_holding, theme=theme).pack(side=tk.LEFT, padx=10)

        # Show loading text
        tk.Label(dialog, text="Analyzing...", font=('Segoe UI', 12), bg=theme['bg'], fg=theme['fg']).pack(pady=50)
        
        threading.Thread(target=load_analysis, daemon=True).start()

    def _delete_selected_holding(self, event):
        """Delete selected holding via keyboard"""
        try:
            item = self.holdings_tree.selection()[0]
            tags = self.holdings_tree.item(item, "tags")
            holding_id = int(tags[1]) if len(tags) > 1 else 0
            
            if holding_id and messagebox.askyesno("Confirm Delete", "Remove this holding?"):
                self.holdings_tracker.remove_holding(holding_id)
                self._refresh_holdings()
        except IndexError:
            pass

    def _auto_refresh_holdings(self):
        """Periodically refresh holdings"""
        if hasattr(self, 'holdings_frame') and self.holdings_frame.winfo_viewable():
            self._refresh_holdings()
        # Schedule next check
        self.root.after(300000, self._auto_refresh_holdings)
    
    def _create_trend_change_tab(self):
        """Create the trend change predictions tab"""
        theme = self.theme_manager.get_theme()
        
        # Header
        header_frame = tk.Frame(self.trend_change_frame, bg=theme['bg'])
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = tk.Label(header_frame, text="Trend Change Predictions", 
                              font=('Segoe UI', 14, 'bold'), bg=theme['bg'], fg=theme['fg'])
        title_label.pack(side=tk.LEFT)
        
        refresh_btn = ModernButton(header_frame, text="Refresh", 
                                  command=self._refresh_trend_changes, theme=theme)
        refresh_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        clear_trend_btn = ModernButton(header_frame, text="🗑️ Clear All", 
                                      command=self._clear_all_trend_predictions, theme=theme)
        clear_trend_btn.pack(side=tk.RIGHT)
        
        # Info label
        info_label = tk.Label(self.trend_change_frame, 
                             text="Predictions are generated automatically when analyzing stocks.\n"
                                  "This tab shows when trend changes might occur based on technical indicators.",
                             font=('Segoe UI', 9), bg=theme['bg'], fg=theme['text_secondary'],
                             justify=tk.LEFT)
        info_label.pack(fill=tk.X, pady=(0, 15))
        
        # Treeview for trend change predictions
        list_card = ModernCard(self.trend_change_frame, "", theme=theme, padding=15)
        list_card.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        list_inner = tk.Frame(list_card, bg=theme['card_bg'])
        list_inner.pack(fill=tk.BOTH, expand=True)
        
        columns = ('Fav', 'ID', 'Symbol', 'Current Trend', 'Predicted Change', 'Estimated Days', 'Estimated Date', 'Confidence', 'Status', 'Delete')
        self.trend_change_tree = ttk.Treeview(list_inner, columns=columns, show='headings', height=12)
        
        for col in columns:
            self.trend_change_tree.heading(col, text=col, anchor=tk.CENTER)
            if col == 'Fav':
                self.trend_change_tree.column(col, width=30, anchor=tk.CENTER)
            elif col == 'ID':
                # Hide ID column
                self.trend_change_tree.column(col, width=0, stretch=False)
            elif col == 'Status':
                self.trend_change_tree.column(col, width=100, anchor=tk.CENTER)
            elif col == 'Estimated Date':
                self.trend_change_tree.column(col, width=120, anchor=tk.CENTER)
            elif col == 'Confidence':
                self.trend_change_tree.column(col, width=100, anchor=tk.CENTER)
            elif col == 'Delete':
                self.trend_change_tree.column(col, width=50, anchor=tk.CENTER)
            else:
                self.trend_change_tree.column(col, width=120, anchor=tk.CENTER)
        
        # Bind click event on Delete column
        self.trend_change_tree.bind('<Button-1>', self._on_trend_change_tree_click)
        # Bind double-click to verify prediction
        self.trend_change_tree.bind('<Double-1>', self._on_trend_change_tree_double_click)
        # Bind motion event for hover effect on Delete column
        self.trend_change_tree.bind('<Motion>', self._on_trend_change_tree_motion)
        self.trend_change_tree.bind('<Leave>', self._on_trend_change_tree_leave)
        
        # Track currently hovered delete cell
        self.hovered_trend_delete_item = None
        
        tree_scrollbar = ModernScrollbar(list_inner, command=self.trend_change_tree.yview, theme=theme)
        self.trend_change_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        # Bind mousewheel to trend change treeview for scrolling
        def on_trend_tree_mousewheel(event):
            delta = event.delta if hasattr(event, 'delta') else (event.num == 4 and -1) or 1
            if isinstance(delta, (int, float)):
                scroll_amount = int(-1 * (delta / 120)) * 3
                self.trend_change_tree.yview_scroll(scroll_amount, "units")
            return "break"
        
        self.trend_change_tree.bind("<MouseWheel>", on_trend_tree_mousewheel)
        self.trend_change_tree.bind("<Button-4>", lambda e: self.trend_change_tree.yview_scroll(-3, "units") or "break")
        self.trend_change_tree.bind("<Button-5>", lambda e: self.trend_change_tree.yview_scroll(3, "units") or "break")
        # Right-click context menu
        self.trend_change_tree.bind('<Button-3>', self._on_trend_tree_right_click)
        
        self.trend_change_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # Store trend change predictions
        self.trend_change_predictions = []
    
    def _update_trend_change_display(self):
        """Update the trend change predictions display with enhanced visuals"""
        theme = self.theme_manager.get_theme()
        
        # Load all predictions from persistent storage to show history
        try:
            # Get all predictions instead of just active ones
            all_predictions = self.trend_change_tracker.predictions
            # Sort: Active first, then by date newest
            active_preds = [p for p in all_predictions if p.get('status') == 'active']
            other_preds = [p for p in all_predictions if p.get('status') != 'active']
            
            # Sort others by verification date or timestamp
            other_preds.sort(key=lambda x: x.get('verification_date') or x.get('timestamp', ''), reverse=True)
            
            # Combine
            self.trend_change_predictions = active_preds + other_preds
        except Exception as e:
            logger.error(f"Error loading trend change predictions: {e}")
            self.trend_change_predictions = []
        
        # Clear tree
        for item in self.trend_change_tree.get_children():
            self.trend_change_tree.delete(item)
        
        # Emoji mapping for trends
        trend_emojis = {
            'uptrend': '📈 Uptrend',
            'downtrend': '📉 Downtrend',
            'sideways': '➡️ Sideways',
            'bullish_reversal': '🔼 Bullish Rev',
            'bearish_reversal': '🔽 Bearish Rev',
            'continuation': '➡️ Continuation'
        }
        
        # Add all trend change predictions
        for pred in self.trend_change_predictions:
            # Format date
            try:
                from datetime import datetime
                estimated_date = datetime.fromisoformat(pred['estimated_date'])
                date_str = estimated_date.strftime("%m-%d") # Use shorter date
            except (ValueError, KeyError, AttributeError):
                date_str = "N/A"
            
            # Format trends
            curr_trend = pred.get('current_trend', 'unknown').lower()
            curr_trend_display = trend_emojis.get(curr_trend, curr_trend.title())
            
            change_type = pred.get('predicted_change', 'unknown').lower()
            change_display = trend_emojis.get(change_type, change_type.replace('_', ' ').title())
            
            # Visual confidence bar
            conf = pred.get('confidence', 0)
            bar_len = 10
            filled = int((conf / 100) * bar_len)
            conf_bar = "█" * filled + "░" * (bar_len - filled)
            conf_display = f"{conf_bar} {conf}%"
            
            # Determine Status Display
            status = pred.get('status', 'active').title()
            was_correct = pred.get('was_correct')
            
            if status == 'Verified':
                if was_correct:
                    status = "Correct ✅"
                elif was_correct is False:
                    status = "Incorrect ❌"
            
            symbol = pred['symbol']
            # Favorite status
            is_fav = symbol in self.monitored_stocks
            fav_char = "★" if is_fav else "☆"

            pred_id = pred.get('id', '')
            values = (
                fav_char,
                str(pred_id),  # ID column (hidden)
                symbol,
                curr_trend_display,
                change_display,
                f"{pred.get('estimated_days', 0)}d",
                date_str,
                conf_display,
                status, # Status column
                '✕'
            )
            
            item = self.trend_change_tree.insert('', tk.END, values=values)
            
            # Color coding
            item_tags = []
            if 'Bullish' in change_display or 'Uptrend' in change_display:
                item_tags.append('bullish_tag')
            elif 'Bearish' in change_display or 'Downtrend' in change_display:
                item_tags.append('bearish_tag')
            
            # Add Fav tag
            if is_fav:
                item_tags.append('fav_active')
            else:
                item_tags.append('fav_inactive')
                
            self.trend_change_tree.item(item, tags=tuple(item_tags))
            
    def _on_momentum_changes_tree_click(self, event):
        """Handle clicks on momentum changes tree (specifically Favorite column)"""
        region = self.momentum_changes_tree.identify("region", event.x, event.y)
        if region != "cell":
            return
            
        column = self.momentum_changes_tree.identify_column(event.x)
        item_id = self.momentum_changes_tree.identify_row(event.y)
        
        if not item_id:
            return
            
        # Check if Fav column (index 0 which is #1)
        if column == '#1':
            values = self.momentum_changes_tree.item(item_id, 'values')
            if values and len(values) > 2:
                symbol = values[2] # Symbol is now at index 2
                self._toggle_trend_monitoring(symbol, quiet=True)
                
                # Refresh UI
                self._update_momentum_changes_display()
                if hasattr(self, '_update_trend_change_display'):
                    self._update_trend_change_display()

    def _on_trend_change_tree_click(self, event):
        """Handle clicks on trend change tree (specifically Fav and Delete columns)"""
        region = self.trend_change_tree.identify("region", event.x, event.y)
        if region != "cell":
            return
            
        column = self.trend_change_tree.identify_column(event.x)
        item_id = self.trend_change_tree.identify_row(event.y)
        
        if not item_id:
            return
            
        # Fav column (index 0 -> #1)
        if column == '#1':
            values = self.trend_change_tree.item(item_id, 'values')
            if values and len(values) > 2:
                symbol = values[2] # Symbol is at index 2 (Fav, ID, Symbol)
                self._toggle_trend_monitoring(symbol, quiet=True)
                
                # Refresh UI
                self._update_trend_change_display()
                self._update_momentum_changes_display()
            return

        # Delete column (index 9 -> #10)
        if column == '#10': 
            values = self.trend_change_tree.item(item_id, 'values')
            if values and len(values) > 1:
                # ID is at index 1 (Fav, ID, ...)
                try:
                    prediction_id = int(values[1])
                    if self.trend_change_tracker.delete_prediction(prediction_id):
                        self.trend_change_tree.delete(item_id)
                        # Remove from local list if exists
                        self.trend_change_predictions = [p for p in self.trend_change_predictions if p.get('id') != prediction_id]
                except ValueError:
                    pass
            
            # Tag based on confidence
            if pred['confidence'] >= 75:
                item_tags.append('high_confidence')
            elif pred['confidence'] >= 60:
                item_tags.append('medium_confidence')
            else:
                item_tags.append('low_confidence')
            
            # Add delete tag for hover effect
            item_tags.append('delete_cell')
            
            self.trend_change_tree.item(item, tags=tuple(item_tags))
        
        # Configure tags for colors
        self.trend_change_tree.tag_configure('bullish_tag', foreground=theme['success'])
        self.trend_change_tree.tag_configure('bearish_tag', foreground=theme['error'])
        self.trend_change_tree.tag_configure('high_confidence', font=('Segoe UI', 9, 'bold'))
        self.trend_change_tree.tag_configure('medium_confidence', foreground=theme['warning'])
        self.trend_change_tree.tag_configure('low_confidence', foreground=theme['text_secondary'])
        self.trend_change_tree.tag_configure('delete_cell', foreground=theme['fg'])
        self.trend_change_tree.tag_configure('delete_hover', foreground='#FF0000')  # Red on hover
        
        # Ensure click handler is bound (in case tree was recreated)
        self.trend_change_tree.bind('<Button-1>', self._on_trend_change_tree_click)
        self.trend_change_tree.bind('<Motion>', self._on_trend_change_tree_motion)
        self.trend_change_tree.bind('<Leave>', self._on_trend_change_tree_leave)
    
    def _create_momentum_changes_tab(self):
        """Create the dedicated momentum changes tab"""
        theme = self.theme_manager.get_theme()
        
        # Header
        header_frame = tk.Frame(self.momentum_changes_frame, bg=theme['bg'])
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = tk.Label(header_frame, text="🔄 Momentum & Action Changes", 
                              font=('Segoe UI', 14, 'bold'), bg=theme['bg'], fg=theme['fg'])
        title_label.pack(side=tk.LEFT)
        
        refresh_btn = ModernButton(header_frame, text="Refresh", 
                                  command=self._refresh_momentum_changes, theme=theme)
        refresh_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        clear_btn = ModernButton(header_frame, text="🗑️ Clear History", 
                                command=self._clear_momentum_changes, theme=theme)
        clear_btn.pack(side=tk.RIGHT)
        
        # Info label
        info_label = tk.Label(self.momentum_changes_frame, 
                             text="This tab tracks recommendation changes (BUY/SELL/HOLD) and momentum shifts.\n"
                                  "Double-click a row to analyze the stock immediately.",
                             font=('Segoe UI', 9), bg=theme['bg'], fg=theme['text_secondary'],
                             justify=tk.LEFT)
        info_label.pack(fill=tk.X, pady=(0, 15))
        
        # Treeview for momentum changes
        list_card = ModernCard(self.momentum_changes_frame, "", theme=theme, padding=15)
        list_card.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        list_inner = tk.Frame(list_card, bg=theme['card_bg'])
        list_inner.pack(fill=tk.BOTH, expand=True)
        
        columns = ('Fav', 'Time', 'Symbol', 'Type', 'From', 'To', 'Changes', 'Confidence', 'Strategy', 'Status')
        self.momentum_changes_tree = ttk.Treeview(list_inner, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.momentum_changes_tree.heading(col, text=col, anchor=tk.CENTER)
            if col == 'Fav':
                self.momentum_changes_tree.column(col, width=30, anchor=tk.CENTER)
            elif col == 'Time':
                self.momentum_changes_tree.column(col, width=150, anchor=tk.CENTER)
            elif col == 'Symbol':
                self.momentum_changes_tree.column(col, width=80, anchor=tk.CENTER)
            elif col == 'Changes':
                self.momentum_changes_tree.column(col, width=300, anchor=tk.W)
            elif col == 'Status':
                self.momentum_changes_tree.column(col, width=100, anchor=tk.CENTER)
            else:
                self.momentum_changes_tree.column(col, width=90, anchor=tk.CENTER)
        
        # Bind double-click to analyze stock
        self.momentum_changes_tree.bind('<Double-1>', self._on_momentum_changes_tree_double_click)
        # Bind click for favorite toggle
        self.momentum_changes_tree.bind('<Button-1>', self._on_momentum_changes_tree_click)
        
        tree_scrollbar = ModernScrollbar(list_inner, command=self.momentum_changes_tree.yview, theme=theme)
        self.momentum_changes_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        # Bind mousewheel to treeview for scrolling
        def on_mom_tree_mousewheel(event):
            delta = event.delta if hasattr(event, 'delta') else (event.num == 4 and -1) or 1
            if isinstance(delta, (int, float)):
                scroll_amount = int(-1 * (delta / 120)) * 3
                self.momentum_changes_tree.yview_scroll(scroll_amount, "units")
            return "break"
        
        self.momentum_changes_tree.bind("<MouseWheel>", on_mom_tree_mousewheel)
        # Right-click context menu
        self.momentum_changes_tree.bind('<Button-3>', self._on_momentum_tree_right_click)
        self.momentum_changes_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # Initial display update
        self._update_momentum_changes_display()

    def _update_momentum_changes_display(self):
        """Update the momentum changes tree with history"""
        if not hasattr(self, 'momentum_changes_tree'):
            return
            
        # Clear tree
        for item in self.momentum_changes_tree.get_children():
            self.momentum_changes_tree.delete(item)
        
        theme = self.theme_manager.get_theme()
        
        # Ensure we're using current history
        self.momentum_changes_history = self.momentum_monitor.history
        
        # Add changes from history (newest first)
        for change_info in reversed(self.momentum_changes_history):
            symbol = change_info['symbol']
            changes = change_info['changes']
            recommendation = changes.get('recommendation', {})
            
            new_action = recommendation.get('action', 'HOLD')
            prev_action = change_info.get('action', 'N/A')
            
            is_trend_reversal = changes.get('trend_reversed', False)
            change_type = "Trend Reversal" if is_trend_reversal else "Momentum shift"
            
            timestamp = change_info.get('time', datetime.now().strftime("%H:%M:%S"))
            
            # Format changes list
            detected_changes = changes.get('changes', [])
            changes_str = ", ".join(detected_changes) if detected_changes else "No specific changes"
            
            # Status
            status = change_info.get('status', 'pending').title()
            verified = change_info.get('verified', False)
            
            if verified:
                if status.lower() == 'correct':
                    status = "Correct ✅"
                elif status.lower() == 'incorrect':
                    status = "Incorrect ❌"
                elif status.lower() == 'neutral':
                    status = "Neutral ➖"
            
            # Favorite status
            is_fav = symbol in self.monitored_stocks
            fav_char = "★" if is_fav else "☆"

            values = (
                fav_char,
                timestamp,
                symbol,
                change_type,
                prev_action,
                new_action,
                changes_str,
                f"{recommendation.get('confidence', 50)}%",
                change_info.get('strategy', 'mixed').title(),
                status
            )
            
            item = self.momentum_changes_tree.insert('', tk.END, values=values)
            
            # Apply tags for coloring
            if new_action == "BUY":
                self.momentum_changes_tree.item(item, tags=('buy_action',))
            elif new_action == "SELL":
                self.momentum_changes_tree.item(item, tags=('sell_action',))
            else:
                self.momentum_changes_tree.item(item, tags=('hold_action',))
                
            # Apply Fav tag for color
            if is_fav:
                self.momentum_changes_tree.item(item, tags=self.momentum_changes_tree.item(item, 'tags') + ('fav_active',))
            else:
                self.momentum_changes_tree.item(item, tags=self.momentum_changes_tree.item(item, 'tags') + ('fav_inactive',))
        
        # Configure tags
        self.momentum_changes_tree.tag_configure('buy_action', foreground=theme['success'])
        self.momentum_changes_tree.tag_configure('sell_action', foreground=theme['warning'])
        self.momentum_changes_tree.tag_configure('hold_action', foreground=theme['text_secondary'])
        self.momentum_changes_tree.tag_configure('fav_active', foreground='#FFD700')  # Gold
        self.momentum_changes_tree.tag_configure('fav_inactive', foreground=theme['text_secondary'])

    def _on_momentum_changes_tree_click(self, event):
        """Handle clicks on momentum changes tree (specifically Favorite column)"""
        try:
            region = self.momentum_changes_tree.identify("region", event.x, event.y)
            logger.info(f"DEBUG: Mom Tree Click Region: {region} at {event.x},{event.y}")
            if region != "cell":
                return
                
            column = self.momentum_changes_tree.identify_column(event.x)
            item_id = self.momentum_changes_tree.identify_row(event.y)
            logger.info(f"DEBUG: Mom Tree Click Column: {column}, Item: {item_id}")
            
            if not item_id:
                return
                
            # Check if Fav column (index 0 which is #1)
            # Treeview columns are #1, #2, #3... so index 0 is indeed #1
            if column == '#1':
                values = self.momentum_changes_tree.item(item_id, 'values')
                logger.info(f"DEBUG: Mom Tree Values: {values}")
                if values and len(values) > 2:
                    symbol = values[2] # Symbol is now at index 2
                    logger.info(f"DEBUG: Toggling {symbol} from Mom Tree")
                    self._toggle_trend_monitoring(symbol, quiet=True)
                    
                    # Refresh UI
                    self._update_momentum_changes_display()
                    if hasattr(self, '_update_trend_change_display'):
                        self._update_trend_change_display()
        except Exception as e:
            logger.error(f"Error in momentum click handler: {e}")

    def _on_momentum_tree_right_click(self, event):
        """Show context menu on right-click in momentum changes tree"""
        try:
            item_id = self.momentum_changes_tree.identify_row(event.y)
            if not item_id:
                return
            
            # Select the row
            self.momentum_changes_tree.selection_set(item_id)
            values = self.momentum_changes_tree.item(item_id, 'values')
            
            if not values or len(values) < 3:
                return
            
            symbol = values[2]  # Symbol is at index 2
            is_fav = symbol.upper() in [s.upper() for s in self.monitored_stocks]
            
            # Create context menu
            menu = tk.Menu(self.root, tearoff=0)
            
            # Favorite toggle
            if is_fav:
                menu.add_command(label="☆ Remove from Favorites", 
                               command=lambda: self._toggle_favorite_from_menu(symbol))
            else:
                menu.add_command(label="★ Add to Favorites", 
                               command=lambda: self._toggle_favorite_from_menu(symbol))
            
            menu.add_separator()
            menu.add_command(label="📊 Analyze Stock", 
                           command=lambda: self._analyze_stock_from_menu(symbol))
            menu.add_command(label="📋 Copy Symbol", 
                           command=lambda: self._copy_symbol_to_clipboard(symbol))
            
            # Show menu
            menu.tk_popup(event.x_root, event.y_root)
        except Exception as e:
            logger.error(f"Error showing momentum context menu: {e}")
        finally:
            try:
                menu.grab_release()
            except:
                pass

    def _on_trend_tree_right_click(self, event):
        """Show context menu on right-click in trend changes tree"""
        try:
            item_id = self.trend_change_tree.identify_row(event.y)
            if not item_id:
                return
            
            # Select the row
            self.trend_change_tree.selection_set(item_id)
            values = self.trend_change_tree.item(item_id, 'values')
            
            if not values or len(values) < 3:
                return
            
            symbol = values[2]  # Symbol is at index 2 (Fav, ID, Symbol)
            is_fav = symbol.upper() in [s.upper() for s in self.monitored_stocks]
            
            # Create context menu
            menu = tk.Menu(self.root, tearoff=0)
            
            # Favorite toggle
            if is_fav:
                menu.add_command(label="☆ Remove from Favorites", 
                               command=lambda: self._toggle_favorite_from_menu(symbol))
            else:
                menu.add_command(label="★ Add to Favorites", 
                               command=lambda: self._toggle_favorite_from_menu(symbol))
            
            menu.add_separator()
            menu.add_command(label="📊 Analyze Stock", 
                           command=lambda: self._analyze_stock_from_menu(symbol))
            menu.add_command(label="📋 Copy Symbol", 
                           command=lambda: self._copy_symbol_to_clipboard(symbol))
            
            # Show menu
            menu.tk_popup(event.x_root, event.y_root)
        except Exception as e:
            logger.error(f"Error showing trend context menu: {e}")
        finally:
            try:
                menu.grab_release()
            except:
                pass

    def _toggle_favorite_from_menu(self, symbol: str):
        """Toggle favorite status from context menu and refresh all displays"""
        try:
            # Use preferences to toggle and persist
            is_now_favorite = self.preferences.toggle_favorite(symbol)
            
            # Update local cache
            self.monitored_stocks = self.preferences.get_monitored_stocks()
            
            # Log the change
            status = "added to" if is_now_favorite else "removed from"
            logger.info(f"★ {symbol} {status} favorites")
            
            # Refresh both displays
            self._update_momentum_changes_display()
            self._update_trend_change_display()
            
        except Exception as e:
            logger.error(f"Error toggling favorite: {e}")

    def _analyze_stock_from_menu(self, symbol: str):
        """Analyze stock from context menu"""
        self.symbol_entry.delete(0, tk.END)
        self.symbol_entry.insert(0, symbol)
        self.notebook.select(0)  # Analysis tab
        self._analyze_stock()

    def _copy_symbol_to_clipboard(self, symbol: str):
        """Copy symbol to clipboard"""
        self.root.clipboard_clear()
        self.root.clipboard_append(symbol)
        logger.info(f"Copied {symbol} to clipboard")

    def _on_trend_change_tree_click(self, event):
        """Handle clicks on trend change tree (specifically Fav and Delete columns)"""
        try:
            region = self.trend_change_tree.identify("region", event.x, event.y)
            logger.info(f"DEBUG: Trend Tree Click Region: {region} at {event.x},{event.y}")
            if region != "cell":
                return
                
            column = self.trend_change_tree.identify_column(event.x)
            item_id = self.trend_change_tree.identify_row(event.y)
            logger.info(f"DEBUG: Trend Tree Click Column: {column}, Item: {item_id}")
            
            if not item_id:
                return
            
            values = self.trend_change_tree.item(item_id, 'values')
            if not values:
                return
                
            # Fav column (index 0 -> #1)
            if column == '#1':
                logger.info(f"DEBUG: Trend Tree Values: {values}")
                if len(values) > 2:
                    symbol = values[2]  # Symbol is at index 2 (Fav, ID, Symbol)
                    logger.info(f"DEBUG: Toggling favorite for {symbol} from Trend Tree")
                    self._toggle_favorite_from_menu(symbol)
                return

            # Delete column (index 9 -> #10)
            if column == '#10': 
                logger.info(f"DEBUG: Delete clicked, values: {values}")
                if len(values) > 1:
                    # ID is at index 1 (Fav, ID, ...)
                    try:
                        prediction_id = int(values[1])
                        logger.info(f"DEBUG: Deleting prediction ID: {prediction_id}")
                        if self.trend_change_tracker.delete_prediction(prediction_id):
                            self.trend_change_tree.delete(item_id)
                            # Remove from local list if exists
                            self.trend_change_predictions = [p for p in self.trend_change_predictions if p.get('id') != prediction_id]
                            logger.info(f"DEBUG: Successfully deleted prediction {prediction_id}")
                        else:
                            logger.warning(f"DEBUG: delete_prediction returned False for ID {prediction_id}")
                    except ValueError as ve:
                        logger.error(f"DEBUG: ValueError parsing prediction ID: {ve}")
                return
                
        except Exception as e:
            logger.error(f"Error in trend click handler: {e}")


    def _on_momentum_changes_tree_double_click(self, event):
        """Handle double-click to analyze stock from momentum changes"""
        item = self.momentum_changes_tree.identify_row(event.y)
        if item:
            values = self.momentum_changes_tree.item(item, 'values')
            # Columns: Fav(0), Time(1), Symbol(2), Type(3), ...
            if values and len(values) > 2:
                symbol = values[2]  # Symbol is at index 2 (after Fav and Time)
                # Switch to analysis tab and analyze
                self.symbol_entry.delete(0, tk.END)
                self.symbol_entry.insert(0, symbol)
                self.notebook.select(0)  # Analysis tab
                self._analyze_stock()

    def _refresh_momentum_changes(self):
        """Refresh momentum changes display"""
        self._update_momentum_changes_display()

    def _clear_momentum_changes(self):
        """Clear momentum changes history for this session and persistent storage"""
        if messagebox.askyesno("Clear History", "Clear all momentum change history?"):
            self.momentum_monitor.clear_history()
            self.momentum_changes_history = self.momentum_monitor.history
            self._update_momentum_changes_display()
    
    def _create_potentials_tab(self):
        """Create the Potentials tab for tracking upcoming investment opportunities"""
        theme = self.theme_manager.get_theme()
        
        # Header
        header_frame = tk.Frame(self.potentials_frame, bg=theme['bg'])
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = tk.Label(header_frame, text="🎯 Potential Investments", 
                              font=('Segoe UI', 14, 'bold'), bg=theme['bg'], fg=theme['fg'])
        title_label.pack(side=tk.LEFT)
        
        # Scan scope dropdown
        self.potentials_scope_var = tk.StringVar(value="predictions")
        scope_frame = tk.Frame(header_frame, bg=theme['bg'])
        scope_frame.pack(side=tk.RIGHT, padx=(0, 10))
        
        scope_label = tk.Label(scope_frame, text="Scope:", font=('Segoe UI', 9), 
                              bg=theme['bg'], fg=theme['text_secondary'])
        scope_label.pack(side=tk.LEFT, padx=(0, 5))
        
        scope_combo = ttk.Combobox(scope_frame, textvariable=self.potentials_scope_var,
                                   values=["predictions", "market"], state="readonly", width=12)
        scope_combo.pack(side=tk.LEFT)
        
        scan_btn = ModernButton(header_frame, text="🔍 Scan for Potentials", 
                               command=self._scan_for_potentials, theme=theme)
        scan_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        refresh_btn = ModernButton(header_frame, text="🔄 Refresh", 
                                  command=self._update_potentials_display, theme=theme)
        refresh_btn.pack(side=tk.RIGHT, padx=(5, 0))
        
        # Info label
        info_label = tk.Label(self.potentials_frame, 
                             text="Stocks predicted to hit a target price soon. Verified automatically when target reached or date passed.",
                             font=('Segoe UI', 9), bg=theme['bg'], fg=theme['text_secondary'],
                             justify=tk.LEFT)
        info_label.pack(fill=tk.X, pady=(0, 15))
        
        # Treeview
        list_card = ModernCard(self.potentials_frame, "", theme=theme, padding=15)
        list_card.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        list_inner = tk.Frame(list_card, bg=theme['card_bg'])
        list_inner.pack(fill=tk.BOTH, expand=True)
        
        columns = ('Fav', 'Symbol', 'Current', 'Target', 'Target Date', 'Confidence', 'Reasoning', 'Status')
        self.potentials_tree = ttk.Treeview(list_inner, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.potentials_tree.heading(col, text=col, anchor=tk.CENTER)
            if col == 'Fav':
                self.potentials_tree.column(col, width=30, anchor=tk.CENTER)
            elif col == 'Symbol':
                self.potentials_tree.column(col, width=80, anchor=tk.CENTER)
            elif col == 'Current' or col == 'Target':
                self.potentials_tree.column(col, width=80, anchor=tk.CENTER)
            elif col == 'Target Date':
                self.potentials_tree.column(col, width=100, anchor=tk.CENTER)
            elif col == 'Confidence':
                self.potentials_tree.column(col, width=80, anchor=tk.CENTER)
            elif col == 'Reasoning':
                self.potentials_tree.column(col, width=250, anchor=tk.W)
            else:
                self.potentials_tree.column(col, width=100, anchor=tk.CENTER)
        
        # Bindings
        self.potentials_tree.bind('<Double-1>', self._on_potentials_tree_double_click)
        self.potentials_tree.bind('<Button-1>', self._on_potentials_tree_click)
        
        tree_scrollbar = ModernScrollbar(list_inner, command=self.potentials_tree.yview, theme=theme)
        self.potentials_tree.configure(yscrollcommand=tree_scrollbar.set)
        
        self.potentials_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # Stats card
        stats_card = ModernCard(self.potentials_frame, "📊 Statistics", theme=theme, padding=10)
        stats_card.pack(fill=tk.X, pady=(0, 15))
        
        self.potentials_stats_label = tk.Label(stats_card, text="No data yet", 
                                               font=('Segoe UI', 9), bg=theme['card_bg'], fg=theme['fg'])
        self.potentials_stats_label.pack()
        
        # Initial display
        self._update_potentials_display()
    
    def _update_potentials_display(self):
        """Update the potentials tree with current data"""
        if not hasattr(self, 'potentials_tree'):
            return
        
        theme = self.theme_manager.get_theme()
        
        # Clear tree
        for item in self.potentials_tree.get_children():
            self.potentials_tree.delete(item)
        
        # Get potentials (active first)
        all_potentials = self.potentials_tracker.potentials
        active = [p for p in all_potentials if p.get('status') == 'active']
        verified = [p for p in all_potentials if p.get('status') == 'verified']
        
        for p in active + verified:
            symbol = p.get('symbol', '')
            is_fav = symbol in self.monitored_stocks
            fav_char = "★" if is_fav else "☆"
            
            status = p.get('status', 'active').title()
            if p.get('verified'):
                status = "Correct ✅" if p.get('was_correct') else "Incorrect ❌"
            
            # Format date
            target_date = p.get('target_date', '')
            try:
                dt = datetime.fromisoformat(target_date)
                target_date_str = dt.strftime("%Y-%m-%d")
            except:
                target_date_str = target_date
            
            values = (
                fav_char,
                symbol,
                f"${p.get('current_price', 0):.2f}",
                f"${p.get('target_price', 0):.2f}",
                target_date_str,
                f"{p.get('confidence', 0):.0f}%",
                p.get('reasoning', '')[:50] + "..." if len(p.get('reasoning', '')) > 50 else p.get('reasoning', ''),
                status
            )
            
            item = self.potentials_tree.insert('', tk.END, values=values)
            
            # Tags
            tags = []
            if p.get('was_correct') is True:
                tags.append('correct')
            elif p.get('was_correct') is False:
                tags.append('incorrect')
            if is_fav:
                tags.append('fav_active')
            self.potentials_tree.item(item, tags=tuple(tags))
        
        # Configure tags
        self.potentials_tree.tag_configure('correct', foreground=theme['success'])
        self.potentials_tree.tag_configure('incorrect', foreground=theme['error'])
        self.potentials_tree.tag_configure('fav_active', foreground='#FFD700')
        
        # Update stats
        stats = self.potentials_tracker.get_statistics()
        stats_text = f"Total: {stats['total']} | Active: {stats['active']} | Verified: {stats['verified']} | Accuracy: {stats['accuracy']:.1f}%"
        if hasattr(self, 'potentials_stats_label'):
            self.potentials_stats_label.config(text=stats_text)
    
    def _scan_for_potentials(self):
        """Scan stocks for potential investment opportunities"""
        scope = self.potentials_scope_var.get() if hasattr(self, 'potentials_scope_var') else 'predictions'
        
        # Determine stock list
        if scope == 'predictions':
            # Get symbols from existing predictions
            symbols = list(set([p.get('symbol') for p in self.predictions_tracker.predictions if p.get('symbol')]))
            if not symbols:
                messagebox.showinfo("No Data", "No predictions found. Add some predictions first or scan the market.")
                return
        else:
            # S&P 500 sample (top 50 for speed)
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'V', 'WMT',
                      'JNJ', 'UNH', 'PG', 'HD', 'MA', 'BAC', 'XOM', 'DIS', 'PFE', 'CSCO',
                      'INTC', 'VZ', 'T', 'CMCSA', 'KO', 'PEP', 'ABT', 'MRK', 'ORCL', 'CRM',
                      'NFLX', 'AMD', 'COST', 'TXN', 'QCOM', 'LOW', 'UNP', 'MS', 'GS', 'BLK',
                      'SCHW', 'AXP', 'C', 'IBM', 'GE', 'RTX', 'BA', 'CAT', 'MMM', 'HON']
        
        total = len(symbols)
        potentials_found = []
        
        # Update status
        self.root.after(0, lambda: self._update_status(f"Scanning {total} stocks for potentials..."))
        
        def scan_in_background():
            nonlocal potentials_found
            for i, symbol in enumerate(symbols):
                try:
                    # Update status bar
                    self.root.after(0, lambda s=symbol, idx=i: self._update_status(
                        f"Scanning {s}... ({idx+1}/{total})"))
                    
                    # Fetch full analysis data (not just basic stock data)
                    data = self.data_fetcher.fetch_stock_data(symbol, force_refresh=True)
                    if not data or data.get('price', 0) <= 0:
                        continue
                    
                    current_price = data.get('price', 0)
                    
                    # Get history data - fetch separately if not in stock data
                    history_list = data.get('history', [])
                    if not history_list:
                        # Fetch history separately
                        history_result = self.data_fetcher.fetch_stock_history(symbol, period='1y')
                        history_list = history_result.get('data', []) if history_result else []
                    
                    if isinstance(history_list, list):
                        history_data = {'data': history_list}
                    else:
                        history_data = history_list if isinstance(history_list, dict) else {'data': []}
                    
                    # USE HYBRID ML PREDICTOR for enhanced predictions
                    # This combines rule-based analysis with trained ML model
                    try:
                        hybrid_predictor = self.hybrid_predictors.get('trading')
                        if hybrid_predictor and len(history_data.get('data', [])) >= 20:
                            # Get full hybrid prediction (Rules + ML)
                            analysis = hybrid_predictor.predict(data, history_data, None)
                            
                            if 'error' not in analysis:
                                recommendation = analysis.get('recommendation', {})
                                action = recommendation.get('action', 'HOLD')
                                ml_confidence = recommendation.get('confidence', 50)
                                
                                # Get indicators from analysis
                                indicators = analysis.get('indicators', {})
                                price_action = analysis.get('price_action', {'trend': 'sideways'})
                                
                                rsi = indicators.get('rsi', 50)
                                mfi = indicators.get('mfi', 50)
                                trend = price_action.get('trend', 'sideways')
                                
                                # Check if ML says BUY or if indicators show opportunity
                                is_ml_buy = action == 'BUY' and ml_confidence >= 55
                                is_oversold = rsi < 40 or mfi < 35
                                is_at_support = price_action.get('at_support', False)
                                has_golden_cross = indicators.get('golden_cross', False)
                                
                                # Build buy signals list
                                buy_signals = []
                                confidence = 50
                                
                                # ML-based signals (highest priority)
                                if is_ml_buy:
                                    buy_signals.append(f"🤖 ML BUY ({ml_confidence:.0f}%)")
                                    confidence += min(30, (ml_confidence - 50))
                                
                                # Technical signals
                                if rsi < 30:
                                    buy_signals.append(f"RSI oversold ({rsi:.1f})")
                                    confidence += 15
                                elif rsi < 40:
                                    buy_signals.append(f"RSI low ({rsi:.1f})")
                                    confidence += 8
                                
                                if mfi < 30:
                                    buy_signals.append(f"MFI low ({mfi:.1f})")
                                    confidence += 10
                                
                                if has_golden_cross:
                                    buy_signals.append("Golden cross")
                                    confidence += 15
                                
                                if is_at_support:
                                    buy_signals.append("At support")
                                    confidence += 10
                                
                                if trend == 'uptrend':
                                    buy_signals.append("Uptrend")
                                    confidence += 10
                                elif trend == 'downtrend' and rsi < 35:
                                    buy_signals.append("Reversal potential")
                                    confidence += 10
                                
                                # Use ML confidence as floor if ML says BUY
                                if is_ml_buy:
                                    confidence = max(confidence, ml_confidence)
                                
                                # BEARISH HOLD: Include stocks that are bearish but waiting for entry
                                is_bearish_hold = (action == 'HOLD' and trend == 'downtrend')
                                if is_bearish_hold and not buy_signals:
                                    buy_signals.append(f"🔜 Watching (Bearish HOLD)")
                                    confidence = max(45, ml_confidence * 0.8)  # Lower confidence for watching
                                
                                logger.info(f"DEBUG {symbol}: ML={action}({ml_confidence:.0f}%), RSI={rsi:.1f}, MFI={mfi:.1f}, trend={trend}")
                                logger.info(f"DEBUG {symbol}: signals={buy_signals}, confidence={confidence}")
                                
                                # Add potential if signals exist and confidence is high enough
                                # Include BEARISH HOLD with lower threshold (45) vs normal (55)
                                min_confidence = 45 if is_bearish_hold else 55
                                if buy_signals and confidence >= min_confidence:
                                    from datetime import timedelta
                                    
                                    # DYNAMIC TARGET DATE based on volatility/ATR
                                    estimated_days = recommendation.get('estimated_days')
                                    if not estimated_days or estimated_days <= 0:
                                        # Fallback: use 10 days for trading-like potentials
                                        estimated_days = 10
                                    target_date = (datetime.now() + timedelta(days=estimated_days)).isoformat()
                                    
                                    # Use ML's target price if available
                                    target_price = recommendation.get('target_price', current_price * 1.05)
                                    
                                    # Build reasoning with ML info
                                    ml_method = analysis.get('method', 'rule_based')
                                    reasoning_prefix = "🤖 ML+" if 'hybrid' in ml_method else "📊 Rules:"
                                    
                                    self.potentials_tracker.add_potential(
                                        symbol=symbol,
                                        current_price=current_price,
                                        target_price=target_price,
                                        target_date=target_date,
                                        confidence=min(confidence, 95),
                                        reasoning=f"{reasoning_prefix} {', '.join(buy_signals[:3])}",
                                        indicators={k: v for k, v in indicators.items() if not isinstance(v, bool) or v}
                                    )
                                    potentials_found.append(symbol)
                                    logger.info(f"DEBUG {symbol}: ADDED as potential (ML-enhanced, {estimated_days} days)!")
                                    
                    except Exception as e:
                        logger.warning(f"Hybrid prediction failed for {symbol}: {e}, skipping")
                                
                except Exception as e:
                    logger.error(f"Error scanning {symbol}: {e}")
            
            # Done
            self.root.after(0, lambda: self._update_status("Scan complete"))
            self.root.after(0, self._update_potentials_display)
            self.root.after(0, lambda: messagebox.showinfo("Scan Complete", 
                f"Found {len(potentials_found)} potential opportunities!"))
        
        thread = threading.Thread(target=scan_in_background)
        thread.daemon = True
        thread.start()
    
    def _verify_potentials_at_startup(self):
        """Verify potentials at startup and feed results to Megamind for recalibration"""
        try:
            verified = self.potentials_tracker.check_for_verification(
                self.data_fetcher, 
                self.calibration_manager
            )
            if verified:
                logger.info(f"🎯 Verified {len(verified)} potentials at startup, fed to Megamind for learning")
                # Update the display if potentials tab is visible
                if hasattr(self, 'potentials_tree'):
                    self.root.after(0, self._update_potentials_display)
        except Exception as e:
            logger.error(f"Error verifying potentials at startup: {e}")
    
    def _on_potentials_tree_click(self, event):
        """Handle clicks on potentials tree (Fav column)"""
        try:
            region = self.potentials_tree.identify("region", event.x, event.y)
            if region != "cell":
                return
            
            column = self.potentials_tree.identify_column(event.x)
            item_id = self.potentials_tree.identify_row(event.y)
            
            if not item_id:
                return
            
            # Fav column (#1)
            if column == '#1':
                values = self.potentials_tree.item(item_id, 'values')
                if values and len(values) > 1:
                    symbol = values[1]
                    self._toggle_trend_monitoring(symbol, quiet=True)
                    self._update_potentials_display()
        except Exception as e:
            logger.error(f"Error in potentials click handler: {e}")
    
    def _on_potentials_tree_double_click(self, event):
        """Handle double-click to analyze stock from potentials"""
        item = self.potentials_tree.identify_row(event.y)
        if item:
            values = self.potentials_tree.item(item, 'values')
            if values and len(values) > 1:
                symbol = values[1]
                self.symbol_entry.delete(0, tk.END)
                self.symbol_entry.insert(0, symbol)
                self.notebook.select(0)
                self._analyze_stock()

    
    def _on_trend_change_tree_click(self, event):
        """Handle click on trend change tree, specifically for delete column"""
        region = self.trend_change_tree.identify_region(event.x, event.y)
        
        if region == 'cell':
            column = self.trend_change_tree.identify_column(event.x)
            item = self.trend_change_tree.identify_row(event.y)
            
            if item and column:
                col_index = int(column.replace('#', '')) - 1
                columns = ('ID', 'Symbol', 'Current Trend', 'Predicted Change', 'Estimated Days', 'Estimated Date', 'Confidence', 'Delete')
                
                if col_index < len(columns) and columns[col_index] == 'Delete':
                    # Get prediction ID from ID column
                    pred_id = self.trend_change_tree.set(item, 'ID')
                    if pred_id:
                        try:
                            pred_id = int(pred_id)
                            # Confirm deletion
                            if messagebox.askyesno(
                                "Delete Prediction",
                                f"Are you sure you want to delete this trend change prediction?"
                            ):
                                # Delete from tracker
                                if self.trend_change_tracker.delete_prediction(pred_id):
                                    # Remove from display
                                    self.trend_change_predictions = [
                                        p for p in self.trend_change_predictions if p.get('id') != pred_id
                                    ]
                                    self._update_trend_change_display()
                                    messagebox.showinfo("Deleted", "Trend change prediction deleted successfully.")
                                else:
                                    messagebox.showwarning("Error", "Could not delete prediction.")
                        except (ValueError, TypeError) as e:
                            logger.error(f"Error parsing prediction ID: {e}")
    
    def _on_trend_change_tree_double_click(self, event):
        """Handle double-click on trend change tree to verify prediction"""
        region = self.trend_change_tree.identify_region(event.x, event.y)
        
        if region == 'cell':
            column = self.trend_change_tree.identify_column(event.x)
            item = self.trend_change_tree.identify_row(event.y)
            
            if item and column:
                col_index = int(column.replace('#', '')) - 1
                columns = ('ID', 'Symbol', 'Current Trend', 'Predicted Change', 'Estimated Days', 'Estimated Date', 'Confidence', 'Delete')
                
                # Don't open dialog if clicking on Delete column
                if col_index < len(columns) and columns[col_index] != 'Delete':
                    # Get prediction ID
                    pred_id = self.trend_change_tree.set(item, 'ID')
                    if pred_id:
                        try:
                            pred_id = int(pred_id)
                            self._show_verify_trend_change_dialog(pred_id)
                        except (ValueError, TypeError) as e:
                            logger.error(f"Error parsing prediction ID: {e}")
    
    def _show_verify_trend_change_dialog(self, prediction_id: int):
        """Show dialog to verify a trend change prediction"""
        # Get prediction
        prediction = self.trend_change_tracker.get_prediction_by_id(prediction_id)
        if not prediction:
            messagebox.showerror("Error", "Prediction not found.")
            return
        
        if prediction.get('status') != 'active':
            messagebox.showinfo("Already Verified", "This prediction has already been verified.")
            return
        
        # Create dialog
        dialog = tk.Toplevel(self.root)
        dialog.title("Verify Trend Change Prediction")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        theme = self.theme_manager.get_theme()
        dialog.config(bg=theme['bg'])
        
        # Header
        header = tk.Label(
            dialog,
            text="Verify Trend Change Prediction",
            bg=theme['bg'],
            fg=theme['accent'],
            font=('Segoe UI', 14, 'bold')
        )
        header.pack(pady=20)
        
        # Prediction info
        info_frame = tk.Frame(dialog, bg=theme['bg'])
        info_frame.pack(pady=10, padx=20, fill=tk.X)
        
        info_text = f"""
Symbol: {prediction['symbol']}
Current Trend: {prediction['current_trend'].title()}
Predicted Change: {prediction['predicted_change'].replace('_', ' ').title()}
Estimated Date: {prediction['estimated_date'][:10]}
Confidence: {prediction['confidence']:.0f}%
        """
        
        tk.Label(
            info_frame,
            text=info_text.strip(),
            bg=theme['bg'],
            fg=theme['fg'],
            font=('Segoe UI', 10),
            justify=tk.LEFT
        ).pack(anchor=tk.W)
        
        # Actual change selection
        change_frame = tk.Frame(dialog, bg=theme['bg'])
        change_frame.pack(pady=20, padx=20, fill=tk.X)
        
        tk.Label(
            change_frame,
            text="What actually happened?",
            bg=theme['bg'],
            fg=theme['fg'],
            font=('Segoe UI', 11, 'bold')
        ).pack(anchor=tk.W, pady=(0, 10))
        
        actual_change_var = tk.StringVar(value="bullish_reversal")
        
        change_options = [
            ("🔼 Bullish Reversal", "bullish_reversal"),
            ("🔽 Bearish Reversal", "bearish_reversal"),
            ("➡️ Continuation", "continuation"),
            ("↔️ No Change", "no_change")
        ]
        
        for text, value in change_options:
            rb = tk.Radiobutton(
                change_frame,
                text=text,
                variable=actual_change_var,
                value=value,
                bg=theme['bg'],
                fg=theme['fg'],
                font=('Segoe UI', 10),
                selectcolor=theme['frame_bg'],
                activebackground=theme['bg'],
                activeforeground=theme['fg']
            )
            rb.pack(anchor=tk.W, pady=5)
        
        # Buttons
        btn_frame = tk.Frame(dialog, bg=theme['bg'])
        btn_frame.pack(pady=20)
        
        def verify_correct():
            actual_change = actual_change_var.get()
            was_correct = (actual_change == prediction['predicted_change'])
            
            # Verify prediction
            verified = self.trend_change_tracker.verify_prediction(
                prediction_id,
                actual_change,
                was_correct
            )
            
            if verified:
                # Record for learning
                if hasattr(self, 'learning_tracker'):
                    self.learning_tracker.record_verified_trend_change(
                        prediction['symbol'],
                        prediction['predicted_change'],
                        actual_change,
                        was_correct
                    )
                
                messagebox.showinfo(
                    "Verified",
                    f"Prediction marked as {'✅ CORRECT' if was_correct else '❌ INCORRECT'}!"
                )
                dialog.destroy()
                self._update_trend_change_display()
            else:
                messagebox.showerror("Error", "Failed to verify prediction.")
        
        correct_btn = ModernButton(
            btn_frame,
            "✅ Mark as Correct",
            command=verify_correct,
            theme=theme
        )
        correct_btn.pack(side=tk.LEFT, padx=5)
        
        def verify_incorrect():
            actual_change = actual_change_var.get()
            was_correct = False  # Explicitly incorrect
            
            # Verify prediction
            verified = self.trend_change_tracker.verify_prediction(
                prediction_id,
                actual_change,
                was_correct
            )
            
            if verified:
                # Record for learning
                if hasattr(self, 'learning_tracker'):
                    self.learning_tracker.record_verified_trend_change(
                        prediction['symbol'],
                        prediction['predicted_change'],
                        actual_change,
                        was_correct
                    )
                
                messagebox.showinfo("Verified", "Prediction marked as ❌ INCORRECT!")
                dialog.destroy()
                self._update_trend_change_display()
            else:
                messagebox.showerror("Error", "Failed to verify prediction.")
        
        incorrect_btn = ModernButton(
            btn_frame,
            "❌ Mark as Incorrect",
            command=verify_incorrect,
            theme=theme
        )
        incorrect_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = ModernButton(
            btn_frame,
            "Cancel",
            command=dialog.destroy,
            theme=theme
        )
        cancel_btn.pack(side=tk.LEFT, padx=5)
    
    def _on_trend_change_tree_motion(self, event):
        """Handle mouse motion over trend change tree for delete hover effect"""
        region = self.trend_change_tree.identify_region(event.x, event.y)
        if region == 'cell':
            column = self.trend_change_tree.identify_column(event.x)
            item = self.trend_change_tree.identify_row(event.y)
            
            if item and column:
                col_index = int(column.replace('#', '')) - 1
                columns = ('ID', 'Symbol', 'Current Trend', 'Predicted Change', 'Estimated Days', 'Estimated Date', 'Confidence', 'Delete')
                
                if col_index < len(columns) and columns[col_index] == 'Delete':
                    # Remove hover from previous item if any
                    if hasattr(self, 'hovered_trend_delete_item') and self.hovered_trend_delete_item and self.hovered_trend_delete_item != item:
                        self._remove_trend_delete_hover(self.hovered_trend_delete_item)
                    
                    # Add hover effect to current item
                    if not hasattr(self, 'hovered_trend_delete_item') or self.hovered_trend_delete_item != item:
                        self._add_trend_delete_hover(item)
                        self.hovered_trend_delete_item = item
                else:
                    # Not hovering over Delete column, remove hover if any
                    if hasattr(self, 'hovered_trend_delete_item') and self.hovered_trend_delete_item:
                        self._remove_trend_delete_hover(self.hovered_trend_delete_item)
                        self.hovered_trend_delete_item = None
            else:
                # No item, remove hover if any
                if hasattr(self, 'hovered_trend_delete_item') and self.hovered_trend_delete_item:
                    self._remove_trend_delete_hover(self.hovered_trend_delete_item)
                    self.hovered_trend_delete_item = None
        else:
            # Not in a cell, remove hover if any
            if hasattr(self, 'hovered_trend_delete_item') and self.hovered_trend_delete_item:
                self._remove_trend_delete_hover(self.hovered_trend_delete_item)
                self.hovered_trend_delete_item = None
    
    def _on_trend_change_tree_leave(self, event):
        """Handle mouse leaving trend change tree"""
        if hasattr(self, 'hovered_trend_delete_item') and self.hovered_trend_delete_item:
            self._remove_trend_delete_hover(self.hovered_trend_delete_item)
            self.hovered_trend_delete_item = None
    
    def _add_trend_delete_hover(self, item):
        """Add hover effect to delete cell in trend change tree"""
        current_tags = list(self.trend_change_tree.item(item, 'tags'))
        if 'delete_hover' not in current_tags:
            current_tags.append('delete_hover')
            self.trend_change_tree.item(item, tags=tuple(current_tags))
    
    def _remove_trend_delete_hover(self, item):
        """Remove hover effect from delete cell in trend change tree"""
        current_tags = list(self.trend_change_tree.item(item, 'tags'))
        if 'delete_hover' in current_tags:
            current_tags.remove('delete_hover')
            self.trend_change_tree.item(item, tags=tuple(current_tags))
    
    def _load_trend_change_predictions(self):
        """Load trend change predictions from persistent storage"""
        try:
            active_predictions = self.trend_change_tracker.get_active_predictions()
            self.trend_change_predictions = active_predictions
            # Update display if tab is already created
            if hasattr(self, 'trend_change_tree'):
                self._update_trend_change_display()
            logger.info(f"Loaded {len(active_predictions)} trend change predictions from storage")
        except Exception as e:
            logger.error(f"Error loading trend change predictions: {e}")
            self.trend_change_predictions = []
    
    def _refresh_trend_changes(self):
        """Refresh trend change predictions display from storage"""
        # Reload predictions from persistent storage and update display
        try:
            self._load_trend_change_predictions()
            self._update_trend_change_display()
            logger.info("Trend change predictions refreshed from storage")
        except Exception as e:
            logger.error(f"Error refreshing trend change predictions: {e}")
            messagebox.showerror("Error", f"Failed to refresh trend change predictions: {e}")
    
    def _refresh_predictions(self):
        """Refresh predictions display"""
        self._update_predictions_display()
    
    def _on_predictions_tree_click(self, event):
        """Handle click on predictions tree, specifically for delete column"""
        # Prevent default selection behavior when clicking Delete column
        region = self.predictions_tree.identify_region(event.x, event.y)
        
        # Debug: log what region was clicked
        logger.info(f"Click detected - region: {region}, x={event.x}, y={event.y}")
        
        if region == 'cell':
            column = self.predictions_tree.identify_column(event.x)  # Only takes x coordinate
            item = self.predictions_tree.identify_row(event.y)
            
            logger.debug(f"Clicked: column={column}, item={item}")
            
            if item and column:
                # Get column index (column is '#1', '#2', etc. - 1-indexed)
                # When show='headings', columns start at '#1' (no '#0' tree column)
                try:
                    col_num = int(column.replace('#', ''))
                    col_index = col_num - 1  # Convert to 0-indexed
                    # MATCH COLUMNS WITH _create_predictions_tab
                    columns = ('ID', 'Symbol', 'Sentiment', 'Action', 'Wait %', 'Entry', 'Target', 'Target Date', 'Status', 'Result', 'Delete')
                    
                    logger.debug(f"Column number: {col_num}, index: {col_index}, total columns: {len(columns)}")
                    
                    # Check if clicked on Delete column (should be column #9, index 8)
                    if col_index >= 0 and col_index < len(columns):
                        column_name = columns[col_index]
                        logger.debug(f"Column name: {column_name}")
                        
                        if column_name == 'Delete':
                            # Prevent row selection when clicking delete
                            self.predictions_tree.selection_remove(self.predictions_tree.selection())
                            
                            # Get prediction ID
                            values = self.predictions_tree.item(item, 'values')
                            logger.info(f"Delete column clicked! Item values: {values}")
                            
                            if len(values) > 0:
                                pred_id = values[0]
                                try:
                                    pred_id = int(pred_id)
                                    # Confirm deletion
                                    symbol = values[1] if len(values) > 1 else "Unknown"
                                    
                                    logger.info(f"Delete clicked for prediction #{pred_id} ({symbol})")
                                    
                                    # Show confirmation dialog
                                    result = messagebox.askyesno(
                                        self.localization.t('confirm_delete'),
                                        f"{self.localization.t('delete_prediction')} #{pred_id} ({symbol})?"
                                    )
                                    
                                    if result:
                                        logger.info(f"User confirmed deletion of prediction #{pred_id}")
                                        if self.predictions_tracker.delete_prediction(pred_id):
                                            self._update_predictions_display()
                                            messagebox.showinfo(
                                                self.localization.t('success'),
                                                f"{self.localization.t('prediction_deleted')} #{pred_id}"
                                            )
                                            logger.info(f"Successfully deleted prediction #{pred_id}")
                                        else:
                                            logger.error(f"Failed to delete prediction #{pred_id}")
                                            messagebox.showerror(
                                                self.localization.t('error'),
                                                f"Could not delete prediction #{pred_id}"
                                            )
                                    else:
                                        logger.info(f"User cancelled deletion of prediction #{pred_id}")
                                    
                                    # Return break to prevent further event processing
                                    return "break"
                                except (ValueError, IndexError) as e:
                                    logger.error(f"Error deleting prediction: {e}", exc_info=True)
                                    messagebox.showerror(
                                        self.localization.t('error'),
                                        f"Error deleting prediction: {str(e)}"
                                    )
                            else:
                                logger.warning(f"No values found for item: {item}")
                                messagebox.showwarning(
                                    self.localization.t('warning'),
                                    "Could not find prediction data. Please try again."
                                )
                except (ValueError, IndexError) as e:
                    logger.error(f"Error parsing column click: {e}", exc_info=True)
        else:
            logger.debug(f"Click was not on a cell, region: {region}")
    
    def _on_predictions_tree_motion(self, event):
        """Handle mouse motion over predictions tree for delete hover effect"""
        region = self.predictions_tree.identify_region(event.x, event.y)
        if region == 'cell':
            column = self.predictions_tree.identify_column(event.x)  # Only takes x coordinate
            item = self.predictions_tree.identify_row(event.y)
            
            if item and column:
                # Get column index (column is '#1', '#2', etc.)
                col_index = int(column.replace('#', '')) - 1
                columns = ('ID', 'Symbol', 'Action', 'Entry', 'Target', 'Target Date', 'Status', 'Result', 'Delete')
                
                # Check if hovering over Delete column
                if col_index < len(columns) and columns[col_index] == 'Delete':
                    # Remove hover from previous item if any
                    if self.hovered_delete_item and self.hovered_delete_item != item:
                        self._remove_delete_hover(self.hovered_delete_item)
                    
                    # Add hover effect to current item
                    if self.hovered_delete_item != item:
                        self._add_delete_hover(item)
                        self.hovered_delete_item = item
                else:
                    # Not hovering over Delete column, remove hover if any
                    if self.hovered_delete_item:
                        self._remove_delete_hover(self.hovered_delete_item)
                        self.hovered_delete_item = None
            else:
                # No item, remove hover if any
                if self.hovered_delete_item:
                    self._remove_delete_hover(self.hovered_delete_item)
                    self.hovered_delete_item = None
        else:
            # Not in a cell, remove hover if any
            if self.hovered_delete_item:
                self._remove_delete_hover(self.hovered_delete_item)
                self.hovered_delete_item = None
    
    def _on_predictions_tree_leave(self, event):
        """Handle mouse leaving predictions tree"""
        if self.hovered_delete_item:
            self._remove_delete_hover(self.hovered_delete_item)
            self.hovered_delete_item = None
    
    def _add_delete_hover(self, item):
        """Add hover effect to delete cell"""
        current_tags = list(self.predictions_tree.item(item, 'tags'))
        if 'delete_hover' not in current_tags:
            current_tags.append('delete_hover')
            self.predictions_tree.item(item, tags=tuple(current_tags))
    
    def _remove_delete_hover(self, item):
        """Remove hover effect from delete cell"""
        current_tags = list(self.predictions_tree.item(item, 'tags'))
        if 'delete_hover' in current_tags:
            current_tags.remove('delete_hover')
            self.predictions_tree.item(item, tags=tuple(current_tags))
    
    def _on_strategy_change(self):
        """Handle strategy change with animation"""
        self.current_strategy = self.strategy_var.get()
        self.preferences.set_strategy(self.current_strategy)
        self._update_strategy_visual_feedback()
        self._update_ml_status()  # Update ML status for new strategy
        logger.info(f"Strategy changed to: {self.current_strategy}")
        
        # Update hybrid predictor for new strategy if needed
        if self.current_strategy not in self.hybrid_predictors:
            self.hybrid_predictors[self.current_strategy] = HybridStockPredictor(
                self.data_dir, self.current_strategy,
                trading_analyzer=self.trading_analyzer if self.current_strategy == 'trading' else None,
                investing_analyzer=self.investing_analyzer if self.current_strategy == 'investing' else None,
                seeker_ai=getattr(self, 'seeker_ai', None),
                mixed_analyzer=self.mixed_analyzer if self.current_strategy == 'mixed' else None
            )
    
    def _analyze_stock(self):
        """Analyze a stock based on selected strategy"""
        symbol = self.symbol_entry.get().strip().upper()
        if not symbol:
            messagebox.showwarning(self.localization.t('input_error'), self.localization.t('input_error'))
            return
        
        # Save to search history
        self.preferences.add_search(symbol)
        self._update_search_history()
        
        # Show loading screen with processes
        theme = self.theme_manager.get_theme()
        if self.loading_screen:
            self.loading_screen.theme = theme
        else:
            self.loading_screen = LoadingScreen(self.root, theme)
        
        # Reset cancel flag
        self.analysis_cancelled = False
        
        # Estimate time based on strategy (rough estimates)
        strategy = self.strategy_var.get()
        if strategy == "trading":
            estimated_time = 15  # seconds
        elif strategy == "mixed":
            estimated_time = 25  # seconds
        else:  # investing
            estimated_time = 20  # seconds
        
        processes = [f"Fetching data for {symbol}"]
        self.loading_screen.show(
            f"Analyzing {symbol}...", 
            processes=processes,
            cancel_callback=self._cancel_analysis,
            estimated_time=estimated_time
        )
        
        # Disable button during analysis
        self.symbol_entry.config(state='disabled')
        self.analyze_btn.config(state='disabled')
        
        # Run analysis in thread to prevent UI freezing
        thread = threading.Thread(target=self._perform_analysis, args=(symbol,))
        thread.daemon = True
        thread.start()
    
    def _scan_market(self):
        """Scan market for recommended stocks"""
        # Reset cancel flag
        self.market_scan_cancelled = False
        
        # Show loading screen with processes
        theme = self.theme_manager.get_theme()
        if self.loading_screen:
            self.loading_screen.theme = theme
        else:
            self.loading_screen = LoadingScreen(self.root, theme)
        
        processes = ["Preparing to scan stocks..."]
        # Estimate time based on number of predictions (approx 3-5s per stock)
        tracked_count = len(self.predictions_tracker.get_active_predictions()) # Estimate
        est_time = max(60, tracked_count * 5)
        
        self.loading_screen.show(
            "Scanning Your Predictions", 
            processes=processes,
            cancel_callback=self._cancel_market_scan,
            estimated_time=est_time
        )
        
        # Disable buttons during scan
        if hasattr(self, 'scan_market_btn'):
            self.scan_market_btn.config(state='disabled')
        if hasattr(self, 'sidebar_scan_btn'):
            self.sidebar_scan_btn.config(state='disabled')
        
        # Run scan in thread
        thread = threading.Thread(target=self._perform_market_scan)
        thread.daemon = True
        thread.start()
    
    def _scan_for_dips(self):
        """Scan market for oversold dip buying opportunities"""
        # Show loading screen
        theme = self.theme_manager.get_theme()
        if self.loading_screen:
            self.loading_screen.theme = theme
        else:
            self.loading_screen = LoadingScreen(self.root, theme)
        
        # ~120 stocks, ~2-3s each = ~4-6 minutes
        est_time = 300
        
        self.loading_screen.show(
            "📉 Scanning Market for Dips", 
            processes=["Initializing dip scanner..."],
            cancel_callback=self._cancel_dip_scan,
            estimated_time=est_time
        )
        
        # Disable button during scan
        if hasattr(self, 'sidebar_dip_btn'):
            self.sidebar_dip_btn.config(state='disabled')
        
        # Reset cancel flag
        self.dip_scan_cancelled = False
        
        # Run scan in thread
        thread = threading.Thread(target=self._perform_dip_scan)
        thread.daemon = True
        thread.start()
    
    def _cancel_dip_scan(self):
        """Cancel the dip scan"""
        self.dip_scan_cancelled = True
        logger.info("Dip scan cancelled by user")
    
    def _perform_dip_scan(self):
        """Perform dip scan in background thread"""
        try:
            def progress_callback(current, total, symbol):
                if self.dip_scan_cancelled:
                    return
                pct = int((current / total) * 100)
                self.root.after(0, lambda: self.loading_screen.update_processes([
                    f"Scanning {symbol}... ({current}/{total})",
                    f"Progress: {pct}%"
                ]) if self.loading_screen else None)
            
            dips = self.market_scanner.scan_for_dips(
                max_results=30,
                rsi_threshold=35.0,
                mfi_threshold=25.0,
                progress_callback=progress_callback
            )
            
            if self.dip_scan_cancelled:
                return
            
            # Show results on main thread
            self.root.after(0, lambda: self._display_dip_results(dips))
            
        except Exception as e:
            logger.error(f"Dip scan error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Scan Error", f"Error scanning for dips: {e}"))
        finally:
            # Hide loading and re-enable button
            self.root.after(0, self._finish_dip_scan)
    
    def _finish_dip_scan(self):
        """Clean up after dip scan"""
        if self.loading_screen:
            self.loading_screen.hide()
        if hasattr(self, 'sidebar_dip_btn'):
            self.sidebar_dip_btn.config(state='normal')
    
    def _display_dip_results(self, dips: list):
        """Display dip scan results in a popup window"""
        theme = self.theme_manager.get_theme()
        
        # Create results window
        dip_window = tk.Toplevel(self.root)
        dip_window.title("📉 Market Dip Opportunities")
        dip_window.config(bg=theme['bg'])
        dip_window.geometry("900x700")
        
        # Header
        header = tk.Label(dip_window, text="📉 Market Dip Opportunities",
                         bg=theme['bg'], fg=theme['accent'], font=('Segoe UI', 18, 'bold'))
        header.pack(pady=(20, 5))
        
        # Summary
        summary_text = f"Found {len(dips)} oversold stocks (RSI < 35 or MFI < 25)"
        summary_label = tk.Label(dip_window, text=summary_text,
                               bg=theme['bg'], fg=theme['text_secondary'], font=('Segoe UI', 11))
        summary_label.pack(pady=(0, 15))
        
        if not dips:
            no_dips_label = tk.Label(dip_window, 
                                    text="🎉 No oversold stocks found right now.\nThe market is not in a dip condition.",
                                    bg=theme['bg'], fg=theme['text_secondary'], 
                                    font=('Segoe UI', 12), justify=tk.CENTER)
            no_dips_label.pack(pady=50)
            return
        
        # Save to Potential Trade tab
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.potential_text.insert(tk.END, f"\n\n{'='*60}\n")
        self.potential_text.insert(tk.END, f"📉 Market Dip Scan: {timestamp}\n")
        self.potential_text.insert(tk.END, f"{'='*60}\n")
        self.potential_text.insert(tk.END, f"{summary_text}\n\n")
        
        # Scrollable frame
        canvas = tk.Canvas(dip_window, bg=theme['bg'], highlightthickness=0)
        scrollbar = ModernScrollbar(dip_window, command=canvas.yview, theme=theme)
        scroll_frame = tk.Frame(canvas, bg=theme['bg'])
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.create_window((0, 0), window=scroll_frame, anchor=tk.NW)
        
        # Mousewheel scrolling
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind("<MouseWheel>", on_mousewheel)
        
        # Display each dip opportunity
        for dip in dips:
            card = ModernCard(scroll_frame, f"💎 {dip['symbol']} - {dip['name']}", theme=theme, padding=15)
            card.pack(fill=tk.X, padx=20, pady=10)
            
            rsi = dip.get('rsi', 50)
            mfi = dip.get('mfi', 50)
            dip_score = dip.get('dip_score', 0)
            potential = dip.get('potential_gain', 0)
            
            # RSI/MFI bars
            rsi_bar_len = 15
            rsi_filled = int((rsi / 100) * rsi_bar_len)
            rsi_bar = "█" * rsi_filled + "░" * (rsi_bar_len - rsi_filled)
            
            mfi_filled = int((mfi / 100) * rsi_bar_len)
            mfi_bar = "█" * mfi_filled + "░" * (rsi_bar_len - mfi_filled)
            
            info_text = f"📊 Dip Score: {dip_score:.1f}/100 (Higher = Deeper Dip)\n\n"
            info_text += f"🔴 RSI: [{rsi_bar}] {rsi:.1f} {'⚠️ OVERSOLD' if rsi < 30 else ''}\n"
            info_text += f"🔴 MFI: [{mfi_bar}] {mfi:.1f} {'⚠️ OVERSOLD' if mfi < 20 else ''}\n"
            info_text += f"📈 Trend: {dip.get('trend', 'Unknown')}\n\n"
            info_text += f"💰 Current Price: ${dip['price']:,.2f}\n"
            info_text += f"🎯 Target (EMA20): ${dip.get('target_price', 0):,.2f}\n"
            info_text += f"🛑 Stop Loss: ${dip.get('stop_loss', 0):,.2f}\n"
            info_text += f"📈 Potential Gain: +{potential:.1f}%\n\n"
            info_text += f"📝 {dip.get('reasoning', '')}"
            
            info_label = tk.Label(card.content_frame, text=info_text, bg=theme['card_bg'],
                                fg=theme['fg'], font=('Segoe UI', 10), justify=tk.LEFT)
            info_label.pack(anchor=tk.W)
            
            # Analyze button
            analyze_btn = ModernButton(card.content_frame, f"Analyze {dip['symbol']}",
                                      command=lambda s=dip['symbol']: self._analyze_symbol_from_rec(s, dip_window),
                                      theme=theme)
            analyze_btn.pack(fill=tk.X, pady=(10, 0))
            
            # Save to Potential Trade tab
            self.potential_text.insert(tk.END, f"─" * 50 + "\n")
            self.potential_text.insert(tk.END, f"💎 {dip['symbol']} - {dip['name']}\n")
            self.potential_text.insert(tk.END, f"   RSI: {rsi:.1f} | MFI: {mfi:.1f} | Dip Score: {dip_score:.1f}\n")
            self.potential_text.insert(tk.END, f"   Price: ${dip['price']:,.2f} | Target: ${dip.get('target_price', 0):,.2f} (+{potential:.1f}%)\n")
            self.potential_text.insert(tk.END, f"   {dip.get('reasoning', '')}\n\n")
        
        self.potential_text.see(tk.END)
        
        def configure_scroll(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))
        scroll_frame.bind('<Configure>', configure_scroll)
        
        dip_window.update_idletasks()
        scrollbar._update_slider()
    
    def _cancel_market_scan(self):
        """Cancel the market scan"""
        self.market_scan_cancelled = True
        logger.info("Market scan cancelled by user")
    
    def _perform_market_scan(self):
        """Perform market scan (runs in background thread)"""
        try:
            strategy = self.strategy_var.get()
            processes = ["Scanning market stocks..."]
            self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
            
            if self.market_scan_cancelled:
                return
                
            # FEATURE UPDATE: Verify existing predictions first
            verify_processes = ["Checking past predictions..."]
            self.root.after(0, lambda: self.loading_screen.update_processes(verify_processes) if self.loading_screen else None)
            
            # Verify ALL active predictions
            verification_results = self.predictions_tracker.verify_all_active_predictions(
                self.data_fetcher, check_all=True
            )
            
            # If verification happened, feed Megamind and update stats
            verified_count = verification_results.get('verified', 0)
            if verified_count > 0:
                logger.info(f"Verified {verified_count} predictions during scan")
                verify_processes.append(f"Verified {verified_count} predictions")
                self.root.after(0, lambda: self.loading_screen.update_processes(verify_processes) if self.loading_screen else None)
                
                # Feed Data to Megamind (Auto-Train)
                if hasattr(self, 'learning_tracker'):
                    self._update_performance_tracking(verification_results.get('newly_verified', []))
                    
                    # Trigger training in background
                    train_thread = threading.Thread(target=self._trigger_auto_training)
                    train_thread.daemon = True
                    train_thread.start()
            
            # Refresh predictions display in main UI
            self.root.after(0, self._update_predictions_display)
            
            # FEATURE UPDATE: Scan ALL tracked predictions instead of just POPULAR_STOCKS
            tracked_symbols = self.predictions_tracker.get_all_tracked_symbols()
            
            # If no tracked symbols, fallback to popular stocks (for new users)
            custom_tickers = tracked_symbols if tracked_symbols else None
            
            scan_msg = f"Scanning {len(tracked_symbols)} tracked stocks..." if tracked_symbols else "Scanning market stocks..."
            processes = [scan_msg]
            self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
            
            def scan_progress_callback(current, total, symbol, found_count):
                if self.market_scan_cancelled:
                    return
                pct = int((current / total) * 100)
                # Update the last process line
                self.root.after(0, lambda: self.loading_screen.update_processes([
                    f"Scanning {symbol}... ({current}/{total}) - {pct}%",
                    f"Found {found_count} potential plays"
                ]) if self.loading_screen else None)
                
            recommendations = self.market_scanner.scan_market(
                strategy, 
                max_results=100 if custom_tickers else 10,  # Allow many results for tracked stocks
                predictions_tracker=self.predictions_tracker,
                custom_tickers=custom_tickers,
                progress_callback=scan_progress_callback
            )
            
            if self.market_scan_cancelled:
                return
            
            processes.append(f"Found {len(recommendations)} recommendations")
            self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
            
            # Update UI in main thread IMMEDIATELY found results
            self.root.after(0, lambda: self._display_market_recommendations(
                recommendations, 
                title="Prediction Analysis Results" if custom_tickers else None
            ))
            
            # Generate trend change predictions for scanned stocks
            if strategy in ['trading', 'mixed']:
                processes.append("Generating trend change predictions...")
                self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
                
                trend_predictions_count = 0
                for rec in recommendations:
                    if self.market_scan_cancelled:
                        break
                    
                    symbol = rec['symbol']
                    try:
                        # Fetch history data for trend change prediction
                        history_data = self.data_fetcher.fetch_stock_history(symbol, period='3mo')
                        if not history_data or 'error' in history_data:
                            continue
                        
                        # Perform trading analysis to get indicators
                        stock_data = self.data_fetcher.fetch_stock_data(symbol, force_refresh=False)
                        if 'error' in stock_data:
                            continue
                        
                        trading_analysis = self.trading_analyzer.analyze(stock_data, history_data)
                        if 'error' in trading_analysis:
                            continue
                        
                        if 'indicators' in trading_analysis and 'price_action' in trading_analysis:
                            # Generate trend change predictions
                            trend_predictions = self.trend_change_predictor.predict_trend_changes(
                                symbol,
                                history_data,
                                trading_analysis['indicators'],
                                trading_analysis['price_action']
                            )
                            
                            if trend_predictions:
                                # Remove old active predictions for this symbol from tracker (to prevent duplicates)
                                active_predictions = self.trend_change_tracker.get_active_predictions()
                                for old_pred in active_predictions:
                                    if old_pred.get('symbol', '').upper() == symbol.upper():
                                        # Mark old prediction as expired (replaced by new analysis)
                                        old_pred['status'] = 'expired'
                                        logger.debug(f"Expired old trend change prediction #{old_pred.get('id')} for {symbol}")
                                
                                # Save the expired predictions
                                if any(p.get('symbol', '').upper() == symbol.upper() for p in active_predictions):
                                    self.trend_change_tracker.save()
                                
                                # Remove old predictions for this symbol (from display)
                                self.trend_change_predictions = [
                                    p for p in self.trend_change_predictions if p.get('symbol', '').upper() != symbol.upper()
                                ]
                                
                                # Store predictions
                                for pred in trend_predictions:
                                    try:
                                        stored_pred = self.trend_change_tracker.add_prediction(
                                            symbol=pred['symbol'],
                                            current_trend=pred['current_trend'],
                                            predicted_change=pred['predicted_change'],
                                            estimated_days=pred['estimated_days'],
                                            estimated_date=pred['estimated_date'],
                                            confidence=pred['confidence'],
                                            reasoning=pred['reasoning'],
                                            key_indicators=pred.get('key_indicators', {})
                                        )
                                        
                                        # Update the prediction in memory with the stored ID
                                        pred['id'] = stored_pred['id']
                                        
                                        # Add to in-memory list
                                        self.trend_change_predictions.append(pred)
                                        
                                        # Record in learning tracker
                                        self.learning_tracker.record_trend_change_prediction(
                                            symbol=pred['symbol'],
                                            predicted_change=pred['predicted_change'],
                                            estimated_days=pred['estimated_days'],
                                            confidence=pred['confidence']
                                        )
                                        
                                        trend_predictions_count += 1
                                    except Exception as e:
                                        logger.error(f"Error recording trend change prediction for {symbol}: {e}")
                    except Exception as e:
                        logger.error(f"Error generating trend change predictions for {symbol}: {e}")
                
                if trend_predictions_count > 0:
                    processes.append(f"Generated {trend_predictions_count} trend change predictions")
                    self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
                    # Update trend change display
                    self.root.after(0, self._update_trend_change_display)
            
        except Exception as e:
            if not self.market_scan_cancelled:
                logger.error(f"Error in market scan: {e}")
                self.root.after(0, self._show_error, f"Error scanning market: {str(e)}")
        finally:
            # Hide loading screen and re-enable button
            self.root.after(0, lambda: self.loading_screen.hide() if self.loading_screen else None)
            
            def enable_buttons():
                if hasattr(self, 'scan_market_btn'):
                    self.scan_market_btn.config(state='normal')
                if hasattr(self, 'sidebar_scan_btn'):
                    self.sidebar_scan_btn.config(state='normal')
            
            self.root.after(0, enable_buttons)
    
    def _display_market_recommendations(self, recommendations: list, title: str = None):
        """Display market recommendations"""
        display_title = title if title else self.localization.t('market_recommendations')
        
        if not recommendations:
            messagebox.showinfo(display_title, 
                              self.localization.t('no_recommendations'))
            return
        
        # Create a new window to display recommendations
        theme = self.theme_manager.get_theme()
        rec_window = tk.Toplevel(self.root)
        rec_window.title(display_title)
        rec_window.config(bg=theme['bg'])
        rec_window.geometry("800x600")
        
        # Header
        header = tk.Label(rec_window, text=display_title,
                         bg=theme['bg'], fg=theme['accent'], font=('Segoe UI', 18, 'bold'))
        header.pack(pady=(20, 5))
        
        # Calculate summary
        buy_count = sum(1 for r in recommendations if r.get('action') == 'BUY')
        total_count = len(recommendations)
        
        summary_text = f"Found {buy_count} BUY opportunities in {total_count} predictions reviewed."
        summary_label = tk.Label(rec_window, text=summary_text,
                               bg=theme['bg'], fg=theme['text_secondary'], font=('Segoe UI', 11))
        summary_label.pack(pady=(0, 15))
        
        # Auto-save BUY opportunities to Potential Trade tab
        if buy_count > 0:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            self.potential_text.insert(tk.END, f"\n\n{'='*60}\n")
            self.potential_text.insert(tk.END, f"📊 Scan Results: {timestamp}\n")
            self.potential_text.insert(tk.END, f"{'='*60}\n")
            self.potential_text.insert(tk.END, f"{summary_text}\n\n")
            for rec in recommendations:
                if rec.get('action') == 'BUY':
                    symbol = rec['symbol']
                    name = rec.get('name', symbol)
                    price = rec['price']
                    conf = rec.get('confidence', 0)
                    entry = rec.get('entry_price', price)
                    target = rec.get('target_price', price * 1.05)
                    stop = rec.get('stop_loss', price * 0.95)
                    
                    # Calculate potential gain/risk
                    potential_gain = ((target - entry) / entry) * 100 if entry > 0 else 0
                    potential_loss = ((entry - stop) / entry) * 100 if entry > 0 else 0
                    risk_reward = potential_gain / potential_loss if potential_loss > 0 else 0
                    
                    self.potential_text.insert(tk.END, f"─" * 50 + "\n")
                    self.potential_text.insert(tk.END, f"🎯 {symbol} - {name}\n")
                    self.potential_text.insert(tk.END, f"   Action: BUY | Confidence: {conf:.1f}%\n")
                    self.potential_text.insert(tk.END, f"   Current: ${price:,.2f}\n")
                    self.potential_text.insert(tk.END, f"   Entry:   ${entry:,.2f} | Target: ${target:,.2f} | Stop: ${stop:,.2f}\n")
                    self.potential_text.insert(tk.END, f"   Potential: +{potential_gain:.1f}% / -{potential_loss:.1f}% (R:R {risk_reward:.1f}:1)\n\n")
                    
                    # Extract key reasoning points (truncate if too long)
                    reasoning = rec.get('reasoning', 'No detailed reasoning available')
                    # Get first 3 meaningful lines of reasoning
                    reasoning_lines = [l.strip() for l in reasoning.split('\n') if l.strip() and not l.strip().startswith('Trading Analysis')]
                    key_reasons = reasoning_lines[:5]  # First 5 key points
                    if key_reasons:
                        self.potential_text.insert(tk.END, f"   📝 Key Reasons:\n")
                        for reason in key_reasons:
                            # Clean up and truncate long lines
                            reason_clean = reason[:80] + "..." if len(reason) > 80 else reason
                            self.potential_text.insert(tk.END, f"      {reason_clean}\n")
                    self.potential_text.insert(tk.END, "\n")
            self.potential_text.see(tk.END)
            # Switch to Potential Tab to notify user they are saved there? 
            # No, keep focus on popup, but maybe mentioned it.
            
            tk.Label(rec_window, text="(Buy opportunities saved to 'Potential Trade' tab)",
                    bg=theme['bg'], fg=theme['success'], font=('Segoe UI', 9)).pack(pady=(0, 10))
        
        # Sorting Controls
        sort_frame = tk.Frame(rec_window, bg=theme['bg'])
        sort_frame.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        tk.Label(sort_frame, text="Sort by:", bg=theme['bg'], fg=theme['fg'], 
                font=('Segoe UI', 10)).pack(side=tk.LEFT, padx=(0, 10))
        
        sort_var = tk.StringVar(value="Default (Confidence)")
        sort_combo = ttk.Combobox(sort_frame, textvariable=sort_var, 
                                 values=["Default (Confidence)", "Signal (Buy/Sell)", "Trend (Bullish/Bearish)", "Potential Gain"],
                                 state="readonly", width=25)
        sort_combo.pack(side=tk.LEFT)
        
        # Scrollable frame setup
        canvas = tk.Canvas(rec_window, bg=theme['bg'], highlightthickness=0)
        scrollbar = ModernScrollbar(rec_window, command=canvas.yview, theme=theme)
        scroll_frame = tk.Frame(canvas, bg=theme['bg'])
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas_window = canvas.create_window((0, 0), window=scroll_frame, anchor=tk.NW)
        
        # Mousewheel scrolling handler
        def on_mousewheel(event):
            delta = event.delta if hasattr(event, 'delta') else (event.num == 4 and -1) or 1
            if isinstance(delta, (int, float)):
                scroll_amount = int(-1 * (delta / 120)) * 3
                canvas.yview_scroll(scroll_amount, "units")
            return "break"
        
        # Function to bind scroll to all children
        def bind_scroll_to_children(parent):
            parent.bind("<MouseWheel>", on_mousewheel)
            parent.bind("<Button-4>", lambda e: canvas.yview_scroll(-3, "units") or "break")
            parent.bind("<Button-5>", lambda e: canvas.yview_scroll(3, "units") or "break")
            for child in parent.winfo_children():
                if isinstance(child, (tk.Frame, tk.Label, tk.Button)):
                    child.bind("<MouseWheel>", on_mousewheel)
                    child.bind("<Button-4>", lambda e: canvas.yview_scroll(-3, "units") or "break")
                    child.bind("<Button-5>", lambda e: canvas.yview_scroll(3, "units") or "break")
                    if isinstance(child, tk.Frame):
                        bind_scroll_to_children(child)

        # Main render function
        def render_recommendations(recs_to_show):
            # Clear existing
            for widget in scroll_frame.winfo_children():
                widget.destroy()
                
            # Display each recommendation
            for i, rec in enumerate(recs_to_show):
                card = ModernCard(scroll_frame, f"{rec['symbol']} - {rec['name']}", theme=theme, padding=15)
                card.pack(fill=tk.X, padx=20, pady=10)
                
                # Determine actual market trend from reasoning
                action = rec.get('action', 'BUY')
                reasoning = rec.get('reasoning', '').lower()
                
                if 'uptrend' in reasoning or 'bullish' in reasoning or 'momentum' in reasoning:
                    actual_trend = "BULLISH 📈"
                    trend_val = 2
                elif 'downtrend' in reasoning or 'bearish' in reasoning or 'falling' in reasoning:
                    actual_trend = "BEARISH 📉"
                    trend_val = 0
                else:
                    actual_trend = "NEUTRAL ↔️"
                    trend_val = 1
                
                # Store trend value for sorting if needed later (though we sort list before calling this)
                rec['_trend_val'] = trend_val
                
                # Create action context
                if action == 'BUY':
                    action_context = "BUY (Accumulate)" if 'downtrend' in reasoning else "BUY (Momentum)"
                elif action == 'SELL':
                    action_context = "SELL (Take Profits)" if 'uptrend' in reasoning or 'overbought' in reasoning else "SELL (Exit)"
                else:
                    action_context = "HOLD (Wait)"
                
                # Visual confidence bar
                conf = rec.get('confidence', 0)
                bar_len = 15
                filled = int((conf / 100) * bar_len)
                conf_bar = "█" * filled + "░" * (bar_len - filled)
                
                price_usd = f"${rec['price']:,.2f}"
                price_eur = f"€{self.localization.convert_from_usd(rec['price'], 'EUR'):,.2f}"
                
                info_text = f"🔭 Market Trend: {actual_trend}\n"
                info_text += f"📊 Strategy Signal: {action_context}\n"
                info_text += f"🎯 Confidence: [{conf_bar}] {conf:.1f}%\n"
                info_text += f"💰 Current Price: {price_usd} / {price_eur}\n\n"
                
                entry_usd = f"${rec['entry_price']:,.2f}"
                entry_eur = f"€{self.localization.convert_from_usd(rec['entry_price'], 'EUR'):,.2f}"
                info_text += f"🟢 Entry Target: {entry_usd} / {entry_eur}\n"
                
                target_usd = f"${rec['target_price']:,.2f}"
                target_eur = f"€{self.localization.convert_from_usd(rec['target_price'], 'EUR'):,.2f}"
                info_text += f"🏁 Target Exit:  {target_usd} / {target_eur}\n"
                
                stop_usd = f"${rec['stop_loss']:,.2f}"
                stop_eur = f"€{self.localization.convert_from_usd(rec['stop_loss'], 'EUR'):,.2f}"
                info_text += f"🛑 Stop Loss:    {stop_usd} / {stop_eur}\n\n"
                info_text += f"📝 Reasoning: {rec['reasoning']}"
                
                info_label = tk.Label(card.content_frame, text=info_text, bg=theme['card_bg'],
                                    fg=theme['fg'], font=('Segoe UI', 10), justify=tk.LEFT)
                info_label.pack(anchor=tk.W)
                
                # Button to analyze this stock
                analyze_btn = ModernButton(card.content_frame, f"Analyze {rec['symbol']}",
                                          command=lambda s=rec['symbol']: self._analyze_symbol_from_rec(s, rec_window),
                                          theme=theme)
                analyze_btn.pack(fill=tk.X, pady=(10, 0))
                
                # Bind mousewheel to card components
                card.bind("<MouseWheel>", on_mousewheel)
                card.content_frame.bind("<MouseWheel>", on_mousewheel)
                info_label.bind("<MouseWheel>", on_mousewheel)
                analyze_btn.bind("<MouseWheel>", on_mousewheel)
            
            # Re-bind main scroll container
            bind_scroll_to_children(scroll_frame)
            
            # Update scroll region
            scroll_frame.update_idletasks()
            canvas.configure(scrollregion=canvas.bbox("all"))
        
        # Sort Handler
        def on_sort_change(event=None):
            sort_mode = sort_var.get()
            sorted_recs = list(recommendations)
            
            if "Signal" in sort_mode:
                # BUY -> SELL -> HOLD
                action_priority = {'BUY': 2, 'SELL': 1, 'HOLD': 0}
                sorted_recs.sort(key=lambda x: (action_priority.get(x.get('action', 'HOLD'), 0), x.get('confidence', 0)), reverse=True)
            elif "Trend" in sort_mode:
                # Sort by trend (need to pre-calc or calc on fly)
                # To be efficient, we'll let the render loop parse trends, but for sorting we need to do it here
                def get_trend_score(r):
                    txt = r.get('reasoning', '').lower()
                    if 'uptrend' in txt or 'bullish' in txt: return 2
                    if 'downtrend' in txt or 'bearish' in txt: return 0
                    return 1
                sorted_recs.sort(key=lambda x: get_trend_score(x), reverse=True)
            elif "Potential" in sort_mode:
                # Potential Gain (Target - Entry) / Entry
                def get_potential(r):
                    entry = r.get('entry_price', 1)
                    target = r.get('target_price', entry)
                    return ((target - entry) / entry) if entry else 0
                sorted_recs.sort(key=lambda x: get_potential(x), reverse=True)
            else:
                # Default: Confidence
                sorted_recs.sort(key=lambda x: x.get('confidence', 0), reverse=True)
                
            render_recommendations(sorted_recs)
        
        sort_combo.bind("<<ComboboxSelected>>", on_sort_change)

        # Configure resize handling for canvas window
        def configure_scroll(event=None):
            canvas.itemconfig(canvas_window, width=canvas.winfo_width())
            canvas.configure(scrollregion=canvas.bbox("all"))
            
        canvas.bind('<Configure>', configure_scroll)
        scroll_frame.bind('<Configure>', configure_scroll)
        
        # Bind mousewheel to canvas
        canvas.bind("<MouseWheel>", on_mousewheel)
        canvas.bind("<Button-4>", lambda e: canvas.yview_scroll(-3, "units") or "break")
        canvas.bind("<Button-5>", lambda e: canvas.yview_scroll(3, "units") or "break")
        
        # Initial Render
        render_recommendations(recommendations)
        
        # Force initial scrollbar update
        rec_window.update_idletasks()
        scrollbar._update_slider()
    
    def _analyze_symbol_from_rec(self, symbol: str, window):
        """Analyze a symbol from recommendations window"""
        window.destroy()
        self.symbol_entry.delete(0, tk.END)
        self.symbol_entry.insert(0, symbol)
        self._analyze_stock()
    
    def _update_search_history(self):
        """Update search history dropdown"""
        history = self.preferences.get_search_history()
        if hasattr(self, 'history_dropdown'):
            self.history_dropdown['values'] = history
            if history:
                self.history_dropdown.current(0)
    
    def _on_history_select(self, event=None):
        """Handle history selection"""
        selected = self.history_var.get()
        if selected:
            self.symbol_entry.delete(0, tk.END)
            self.symbol_entry.insert(0, selected)
    
    def _animate_loading(self):
        """Animate loading indicator (deprecated - using loading screen now)"""
        # This method is kept for compatibility but loading screen is used instead
        pass
    
    def _cancel_analysis(self):
        """Cancel the current analysis"""
        self.analysis_cancelled = True
        logger.info("Analysis cancelled by user")
    
    def _perform_analysis(self, symbol: str):
        """Perform stock analysis (runs in background thread)"""
        try:
            # Reset cancellation flag at start of new analysis
            self.analysis_cancelled = False
            
            # Check if cancelled
            if self.analysis_cancelled:
                return
            
            # Update processes
            processes = [f"Fetching stock data for {symbol}"]
            self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
            self.root.after(0, self._update_status, self.localization.t('fetching_data', symbol=symbol))
            
            # Fetch stock data with cache bypass to ensure fresh price
            stock_data = self.data_fetcher.fetch_stock_data(symbol, force_refresh=True)
            
            if self.analysis_cancelled:
                return
            
            processes.append(f"Fetching price history for {symbol}")
            self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
            
            history_data = self.data_fetcher.fetch_stock_history(symbol)
            
            if self.analysis_cancelled:
                return
            
            self.current_symbol = symbol
            self.current_data = stock_data
            # Log the price we received from the backend
            logger.info(f"🔍 PRICE DEBUG: Received price from backend for {symbol}: {stock_data.get('price')}")
            
            # Update monitoring button state based on whether stock is monitored
            if hasattr(self, 'monitor_trend_btn'):
                if symbol in self.monitored_stocks:
                    self.root.after(0, lambda: self.monitor_trend_btn.config(text="🔕 Stop Monitoring"))
                else:
                    self.root.after(0, lambda: self.monitor_trend_btn.config(text="🔔 Monitor Trend Changes"))
            
            strategy = self.strategy_var.get()
            
            # Perform analysis using hybrid AI system
            processes.append("Analyzing with hybrid AI system...")
            self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
            
            # Get hybrid predictor for current strategy
            hybrid_predictor = self.hybrid_predictors[strategy]
            
            if strategy == "trading":
                processes.append("Performing technical analysis...")
                self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
                self.root.after(0, self._update_status, self.localization.t('performing_trading_analysis'))
                analysis = hybrid_predictor.predict(stock_data, history_data, None)
            elif strategy == "mixed":
                if self.analysis_cancelled:
                    return
                processes.append("Fetching financial data...")
                self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
                financials_data = self.data_fetcher.fetch_financials(symbol)
                if self.analysis_cancelled:
                    return
                processes.append("Performing mixed analysis...")
                self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
                self.root.after(0, self._update_status, self.localization.t('performing_mixed_analysis'))
                analysis = hybrid_predictor.predict(stock_data, history_data, financials_data)
            else:  # investing
                if self.analysis_cancelled:
                    return
                processes.append("Fetching financial data...")
                self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
                financials_data = self.data_fetcher.fetch_financials(symbol)
                if self.analysis_cancelled:
                    return
                processes.append("Performing fundamental analysis...")
                self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
                self.root.after(0, self._update_status, self.localization.t('performing_fundamental_analysis'))
                analysis = hybrid_predictor.predict(stock_data, history_data, financials_data)
            
            if self.analysis_cancelled:
                return
            
            processes.append("Generating recommendations...")
            self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
            
            # Check if analysis has errors
            if 'error' in analysis:
                error_msg = analysis.get('error', 'Unknown error during analysis')
                logger.error(f"Error in analysis for {symbol}: {error_msg}")
                self.root.after(0, self._show_error, f"Error analyzing {symbol}: {error_msg}")
                return
            
            self.current_analysis = analysis
            
            # Check for momentum/trend changes (for currently analyzed stock)
            # For trading strategy, we have full indicator data
            trading_analysis_for_trend = None
            if strategy == "trading" and 'indicators' in analysis and 'price_action' in analysis and 'momentum_analysis' in analysis:
                trading_analysis_for_trend = analysis
                momentum_changes = self.momentum_monitor.update_momentum_state(
                    symbol,
                    analysis['indicators'],
                    analysis['price_action'],
                    analysis['momentum_analysis']
                )
                
                # Show notification if significant changes detected
                if momentum_changes.get('trend_reversed') or (momentum_changes.get('momentum_changed') and len(momentum_changes.get('changes', [])) > 0):
                    # Add to history
                    self.momentum_monitor.add_to_history(
                        symbol=symbol,
                        action='ANALYSIS',  # From manual analysis
                        changes=momentum_changes,
                        strategy=strategy
                    )
                    self.root.after(0, self._update_momentum_changes_display)
                    self.root.after(0, self._show_momentum_change_notification, symbol, momentum_changes)
            
            # For mixed strategy, we need to get technical indicators separately
            elif strategy == "mixed":
                try:
                    # Perform trading analysis to get indicators for momentum monitoring
                    trading_analysis = self.trading_analyzer.analyze(stock_data, history_data)
                    trading_analysis_for_trend = trading_analysis
                    if 'indicators' in trading_analysis and 'price_action' in trading_analysis and 'momentum_analysis' in trading_analysis:
                        momentum_changes = self.momentum_monitor.update_momentum_state(
                            symbol,
                            trading_analysis['indicators'],
                            trading_analysis['price_action'],
                            trading_analysis['momentum_analysis']
                        )
                        
                        # Show notification if significant changes detected
                        if momentum_changes.get('trend_reversed') or (momentum_changes.get('momentum_changed') and len(momentum_changes.get('changes', [])) > 0):
                            # Add to history
                            self.momentum_monitor.add_to_history(
                                symbol=symbol,
                                action='ANALYSIS',
                                changes=momentum_changes,
                                strategy=strategy
                            )
                            self.root.after(0, self._update_momentum_changes_display)
                            self.root.after(0, self._show_momentum_change_notification, symbol, momentum_changes)
                except Exception as e:
                    logger.debug(f"Could not check momentum for mixed strategy: {e}")
            
            # Generate trend change predictions
            trend_predictions = []
            if trading_analysis_for_trend and 'indicators' in trading_analysis_for_trend and 'price_action' in trading_analysis_for_trend:
                try:
                    trend_predictions = self.trend_change_predictor.predict_trend_changes(
                        symbol,
                        history_data,
                        trading_analysis_for_trend['indicators'],
                        trading_analysis_for_trend['price_action']
                    )
                    # Store predictions and update display
                    if trend_predictions:
                        # Remove old active predictions for this symbol from tracker (to prevent duplicates)
                        active_predictions = self.trend_change_tracker.get_active_predictions()
                        for old_pred in active_predictions:
                            if old_pred.get('symbol', '').upper() == symbol.upper():
                                # Mark old prediction as expired (replaced by new analysis)
                                old_pred['status'] = 'expired'
                                logger.debug(f"Expired old trend change prediction #{old_pred.get('id')} for {symbol}")
                        
                        # Save the expired predictions
                        if any(p.get('symbol', '').upper() == symbol.upper() for p in active_predictions):
                            self.trend_change_tracker.save()
                        
                        # Remove old predictions for this symbol (from display)
                        self.trend_change_predictions = [
                            p for p in self.trend_change_predictions if p.get('symbol', '').upper() != symbol.upper()
                        ]
                        
                        # Add new predictions to display
                        self.trend_change_predictions.extend(trend_predictions)
                        
                        # Record predictions in tracker for learning system
                        for pred in trend_predictions:
                            try:
                                stored_pred = self.trend_change_tracker.add_prediction(
                                    symbol=pred['symbol'],
                                    current_trend=pred['current_trend'],
                                    predicted_change=pred['predicted_change'],
                                    estimated_days=pred['estimated_days'],
                                    estimated_date=pred['estimated_date'],
                                    confidence=pred['confidence'],
                                    reasoning=pred['reasoning'],
                                    key_indicators=pred.get('key_indicators', {})
                                )
                                
                                # Update the prediction in memory with the stored ID
                                pred['id'] = stored_pred['id']
                                
                                # Record in learning tracker
                                self.learning_tracker.record_trend_change_prediction(
                                    symbol=pred['symbol'],
                                    predicted_change=pred['predicted_change'],
                                    estimated_days=pred['estimated_days'],
                                    confidence=pred['confidence']
                                )
                                
                                logger.info(f"📊 Recorded trend change prediction #{stored_pred['id']} for {symbol}")
                            except Exception as e:
                                logger.error(f"Error recording trend change prediction: {e}")
                        
                        # Update display
                        self.root.after(0, self._update_trend_change_display)
                except Exception as e:
                    logger.error(f"Error generating trend change predictions: {e}")
            
            # Store trend predictions in analysis for display
            analysis['trend_predictions'] = trend_predictions
            
            # Get Seeker AI research
            if SEEKER_AI_ENABLED and self.seeker_ai:
                try:
                    processes.append("Researching with Seeker AI...")
                    self.root.after(0, lambda: self.loading_screen.update_processes(processes) if self.loading_screen else None)
                    
                    research_data = self.seeker_ai.research_stock(symbol, stock_data, history_data)
                    
                    # Add research to analysis
                    if 'error' not in research_data:
                        analysis['seeker_ai_research'] = research_data
                        logger.info(f"Seeker AI research completed for {symbol}")
                    else:
                        logger.warning(f"Seeker AI research failed for {symbol}: {research_data.get('error')}")
                except Exception as e:
                    logger.error(f"Error getting Seeker AI research: {e}")
                    # Continue without Seeker AI if it fails
            
            # Analyze for better buying opportunities (Buy Opportunity Data for Megamind)
            try:
                # Get technical indicators and price action (use trading analysis if current strategy is mixed/investing)
                indicators_for_opp = analysis.get('indicators')
                price_action_for_opp = analysis.get('price_action')
                
                if not indicators_for_opp or not price_action_for_opp:
                    # Fallback to trading analyzer if indicators missing (common for mixed/investing)
                    trading_fallback = self.trading_analyzer.analyze(stock_data, history_data)
                    indicators_for_opp = trading_fallback.get('indicators')
                    price_action_for_opp = trading_fallback.get('price_action')
                
                if indicators_for_opp and price_action_for_opp:
                    buy_opp = self.llm_ai_predictor.analyze_buy_opportunity(
                        symbol, stock_data.get('price', 0), 
                        indicators_for_opp, price_action_for_opp, strategy
                    )
                    if buy_opp:
                        analysis['buy_opportunity'] = buy_opp
                        # Feed to Megamind (Learning Tracker)
                        self.learning_tracker.record_buy_opportunity_prediction(
                            symbol=symbol,
                            current_price=stock_data.get('price', 0),
                            predicted_price=buy_opp['predicted_price'],
                            estimated_date=buy_opp['predicted_date'],
                            confidence=buy_opp['confidence'],
                            reasoning=buy_opp['reasoning']
                        )
                        logger.info(f"🧠 Buy opportunity detected for {symbol} and fed to Megamind for learning")
            except Exception as e:
                logger.error(f"Error in buy opportunity analysis: {e}")


            # Apply Consensus Mechanism (Unify conflicting signals)
            self._apply_consensus_check(analysis, symbol)
            
            # Update UI in main thread
            self.root.after(0, self._display_analysis, analysis, history_data, symbol)


        except Exception as e:
            if not self.analysis_cancelled:
                logger.error(f"Error in analysis: {e}")
                self.root.after(0, self._show_error, f"Error analyzing stock: {str(e)}")
        finally:
            # Hide loading screen and re-enable controls
            self.root.after(0, lambda: self.loading_screen.hide() if self.loading_screen else None)
            self.root.after(0, lambda: self.symbol_entry.config(state='normal'))
            self.root.after(0, lambda: self.analyze_btn.config(state='normal'))
    
    def _apply_consensus_check(self, analysis: Dict, symbol: str):
        """
        Unify conflicting signals from different analyzers into a single consensus.
        Prioritizes Trend Reversals as they are leading indicators.
        Modifies analysis dict in-place.
        """
        try:
            # 1. Get Core Components
            recommendation = analysis.get('recommendation', {})
            rec_action = recommendation.get('action', 'HOLD')
            rec_conf = recommendation.get('confidence', 50)
            rec_reasons = recommendation.get('reasons', [])
            
            # 2. Get Trend Signals
            trend_preds = analysis.get('trend_predictions', [])
            trend_reversal = None
            for p in trend_preds:
                if 'Reversal' in p.get('predicted_change', ''):
                    trend_reversal = p
                    break
            
            # 3. Get Consensus Conflict/Confirmation
            consensus_msg = ""
            boost_conf = 0
            
            
            if trend_reversal:
                trend_type = trend_reversal.get('predicted_change', '')
                trend_conf = trend_reversal.get('confidence', 0)
                
                # Check for BUY conflicts
                if rec_action == 'BUY':
                    if 'Bearish' in trend_type and trend_conf > 60:
                        # CONFLICT: Analyzer says BUY, but Strong Bearish Trend predicted
                        recommendation['action'] = 'HOLD'
                        consensus_msg = f"⚠️ CONFLICT: Technicals suggest BUY, but a predicted Bearish Reversal ({trend_conf}% conf) indicates price dip. Wait."
                        boost_conf = -20
                    elif 'Bullish' in trend_type:
                        # CONFIRMATION
                        consensus_msg = "✅ CONSENSUS: Strong Buy signal. Technicals align with predicted Bullish Reversal."
                        boost_conf = 15
                
                # Check for SELL conflicts
                elif rec_action == 'SELL':
                    if 'Bullish' in trend_type and trend_conf > 60:
                        # CONFLICT: Analyzer says SELL, but Strong Bullish Trend predicted
                        recommendation['action'] = 'HOLD' # or HOLD to wait for peak?
                        consensus_msg = f"⚠️ CONFLICT: Technicals suggest SELL, but a predicted Bullish Reversal ({trend_conf}% conf) suggests bounce. Hold."
                        boost_conf = -20
                    elif 'Bearish' in trend_type:
                        # CONFIRMATION
                        consensus_msg = "✅ CONSENSUS: Strong Sell signal. Technicals align with predicted Bearish Reversal."
                        boost_conf = 15
                        
                # Check for HOLD upgrades (Crucial for active trading)
                elif rec_action == 'HOLD':
                    if 'Bullish' in trend_type and trend_conf > 70:
                        # UPGRADE: Technicals neutral, but AI predicts strong reversal up
                        recommendation['action'] = 'BUY'
                        consensus_msg = f"🚀 OPPORTUNITY: Technicals neutral, but high-confidence ({trend_conf}%) Bullish Reversal predicted. Speculative BUY."
                        boost_conf = 10
                    elif 'Bearish' in trend_type and trend_conf > 70:
                        # UPGRADE: Technicals neutral, but AI predicts strong reversal down
                        recommendation['action'] = 'SELL'
                        consensus_msg = f"🔻 WARNING: Technicals neutral, but high-confidence ({trend_conf}%) Bearish Reversal predicted. Speculative SELL."
                        boost_conf = 10
            
            # 4. Integrate Buy Opportunity (LLM/Price Action)
            buy_opp = analysis.get('buy_opportunity')
            if buy_opp and rec_action != 'BUY':
                 # If LLM sees a buy opp but technicals don't, trust LLM if confidence is high
                 if buy_opp.get('confidence', 0) > 75:
                     recommendation['action'] = 'BUY' # Upgrade to BUY
                     consensus_msg = f"✅ AI OVERRIDE: High-confidence Buy Opportunity detected by AI pattern analysis, overriding neutral technicals."
                     boost_conf = 10
            
            # 5. Apply Changes
            if consensus_msg:
                # Add to TOP of reasons
                rec_reasons.insert(0, consensus_msg)
                
            # Apply confidence adjustments
            new_conf = min(100, max(0, rec_conf + boost_conf))
            recommendation['confidence'] = new_conf
            
            # Update original dict
            analysis['recommendation'] = recommendation
            
        except Exception as e:
            logger.error(f"Error in consensus check: {e}")

    def _update_status(self, message: str):
        """Update status message with animation"""
        theme = self.theme_manager.get_theme()
        self.results_title.config(text=message, fg=theme['accent'])
    
    def _display_analysis(self, analysis: dict, history_data: dict, symbol: str = None):
        """Display analysis results with animations
        
        Args:
            analysis: Analysis results dictionary
            history_data: Historical price data
            symbol: Stock symbol (if None, uses self.current_symbol)
        """
        # Use provided symbol or fall back to current_symbol
        if symbol is None:
            symbol = self.current_symbol
        
        if 'error' in analysis:
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(tk.END, f"Error: {analysis['error']}")
            return
        
        # Display reasoning with formatting
        reasoning = analysis.get('reasoning', 'No reasoning available')
        self.analysis_text.delete(1.0, tk.END)
        
        # Add some formatting
        recommendation = analysis.get('recommendation', {})
        action = recommendation.get('action', 'HOLD')
        confidence = recommendation.get('confidence', 0)
        
        # Add stock to monitoring system (track all analyzed stocks)
        if self.current_symbol:
            self.stock_monitor.add_stock(
                self.current_symbol, 
                action, 
                self.strategy_var.get(),
                confidence
            )
        
        # Legacy logic removed - Consensus Mechanism now handles HOLD/BUY decisions based on trends
        # See _apply_consensus_check method
        # if action == 'HOLD' and symbol:
        #     pass


        

        
        # Automatically save prediction if it's a BUY or SELL recommendation
        if action in ['BUY', 'SELL'] and symbol:
            entry_price = recommendation.get('entry_price', 0)
            target_price = recommendation.get('target_price', 0)
            stop_loss = recommendation.get('stop_loss', 0)
            
            # Only save if we have valid prices
            if entry_price > 0 and target_price > 0:
                # Check if this prediction already exists (same symbol with active prediction)
                # Replace if action contrasts (BUY vs SELL or vice versa)
                strategy = self.strategy_var.get()
                existing = None
                for pred in self.predictions_tracker.predictions:
                    if (pred['symbol'] == symbol.upper() and 
                        pred['status'] == 'active'):
                        # Found an active prediction for this symbol
                        existing = pred
                        break
                
                # Calculate estimated days based on strategy
                if strategy == "trading":
                    estimated_days = 10  # Short-term: ~10 days
                elif strategy == "mixed":
                    estimated_days = 21  # Medium-term: ~3 weeks
                else:  # investing
                    estimated_days = 547  # Long-term: ~1.5 years
                
                if existing:
                    old_action = existing.get('action', 'HOLD')
                    # Check if actions contrast (BUY vs SELL or SELL vs BUY)
                    actions_contrast = (
                        (old_action == 'BUY' and action == 'SELL') or
                        (old_action == 'SELL' and action == 'BUY') or
                        (old_action != action)  # Also update if action changed in any way
                    )
                    
                    # Update existing prediction (always update, but log if contrasting)
                    existing['action'] = action  # Update action
                    existing['entry_price'] = entry_price
                    existing['target_price'] = target_price
                    existing['stop_loss'] = stop_loss
                    existing['confidence'] = confidence
                    existing['reasoning'] = reasoning
                    existing['strategy'] = strategy  # Update strategy if changed
                    existing['timestamp'] = datetime.now().isoformat()
                    # Update estimated target date
                    from datetime import timedelta
                    existing['estimated_days'] = estimated_days
                    existing['estimated_target_date'] = (datetime.now() + timedelta(days=estimated_days)).isoformat()
                    self.predictions_tracker.save()
                    
                    if actions_contrast:
                        logger.info(f"Replaced contrasting prediction for {symbol}: {old_action} → {action}")
                    else:
                        logger.info(f"Updated existing prediction for {symbol}")
                else:
                    # Calculate magnitude prediction if possible
                    predicted_move_pct = None
                    if hasattr(self, 'price_move_predictor') and self.price_move_predictor:
                        try:
                            # Get required indicators
                            indicators = analysis.get('indicators', {})
                            
                            # Estimate volatility (ATR if available, else fallback)
                            atr = indicators.get('atr', 0)
                            if atr > 0 and entry_price > 0:
                                atr_percent = (atr / entry_price * 100)
                            else:
                                atr_percent = 2.0 # Default if unavailable
                                
                            volatility_metrics = {
                                'atr': atr,
                                'atr_percent': atr_percent
                            }
                            
                            move_pred = self.price_move_predictor.predict_move(
                                action, entry_price, indicators, volatility_metrics
                            )
                            
                            predicted_move_pct = move_pred.get('predicted_pct')
                            
                            # Add this info to reasoning if it's a significant prediction
                            if predicted_move_pct:
                                reasoning += f"\n\n🎯 Magnitude Prediction: Price expected to {move_pred.get('direction', 'move')} by ~{predicted_move_pct:.1f}% " \
                                             f"(Confidence: {move_pred.get('confidence')}%)"
                        except Exception as e:
                            logger.warn(f"Error calculating magnitude prediction: {e}")
                    
                    # Create new prediction
                    prediction = self.predictions_tracker.add_prediction(
                        symbol, strategy, action, entry_price,
                        target_price, stop_loss, confidence, reasoning, estimated_days,
                        predicted_move_pct=predicted_move_pct
                    )
                    logger.info(f"Auto-saved prediction #{prediction['id']} for {symbol} (target date: {prediction.get('estimated_target_date', 'N/A')})")
                    
                    # Add to monitoring (already added above, but ensure it's updated)
                    self.stock_monitor.add_stock(symbol, action, strategy, confidence)
                    
                    # Record learning event
                    self.learning_tracker.record_manual_analysis(symbol)
                
                # Update predictions display
                self._update_predictions_display()
                
                # Update new feature tabs
                if self.current_data:
                    self._update_risk_management_tab(self.current_data, analysis)
                    self._update_advanced_analytics_tab(symbol)
                
                # Trigger background training on this stock's historical data
                self._train_on_stock_background(symbol, strategy)
        
        theme = self.theme_manager.get_theme()
        
        # Get recommendation details
        recommendation = analysis.get('recommendation', {})
        entry_price = recommendation.get('entry_price', 0)
        target_price = recommendation.get('target_price', 0)
        stop_loss = recommendation.get('stop_loss', 0)
        
        # Calculate estimated days based on strategy
        strategy = self.strategy_var.get()
        if strategy == "trading":
            estimated_days = 10  # Short-term: ~10 days
        elif strategy == "mixed":
            estimated_days = 21  # Medium-term: ~3 weeks
        else:  # investing
            estimated_days = 547  # Long-term: ~1.5 years
        
        # Calculate target date
        from datetime import timedelta
        target_date = datetime.now() + timedelta(days=estimated_days)
        target_date_str = target_date.strftime("%Y-%m-%d")
        
        # Get current price for display
        # Use the price from stock_data that was just fetched during analysis (most recent)
        current_price_display = None
        try:
            # First, try to use the price from the stock_data that was just fetched
            # This is the most current since it was fetched right before analysis
            if hasattr(self, 'current_data') and self.current_data and 'price' in self.current_data:
                current_price_display = self.current_data.get('price', 0)
                logger.info(f"🔍 PRICE DEBUG: Using price from current_data: {current_price_display} (raw value)")
            
            # If not available, fetch fresh data with cache bypass
            if not current_price_display or current_price_display <= 0:
                symbol_to_fetch = symbol or self.current_symbol
                if symbol_to_fetch:
                    logger.debug(f"Fetching fresh price for {symbol_to_fetch} (bypassing cache)")
                    current_stock_data = self.data_fetcher.fetch_stock_data(symbol_to_fetch, force_refresh=True)
                    current_price_display = current_stock_data.get('price', 0)
                    # Update current_data with fresh price
                    if current_stock_data and 'price' in current_stock_data:
                        if not hasattr(self, 'current_data') or not self.current_data:
                            self.current_data = {}
                        self.current_data['price'] = current_price_display
                        logger.debug(f"Updated current_data with fresh price: {current_price_display}")
        except Exception as e:
            logger.error(f"Could not fetch current price for display: {e}")
            # Fallback to price from recommendation if available
            if not current_price_display:
                current_price_display = recommendation.get('entry_price', 0)
                logger.debug(f"Using fallback price from recommendation: {current_price_display}")
        
        # Determine actual market trend from price_action (NOT from text matching)
        price_action = analysis.get('price_action', {})
        trend_from_analysis = price_action.get('trend', 'sideways').lower()
        analysis_reasoning = analysis.get('reasoning', '').lower()  # Still needed for action context
        
        # Use the actual computed trend from technical analysis
        if trend_from_analysis == 'uptrend':
            actual_trend = "BULLISH 📈"
        elif trend_from_analysis == 'downtrend':
            actual_trend = "BEARISH 📉"
        else:
            actual_trend = "NEUTRAL ↔️"
        
        # Create action context for aggressive strategy
        if action == 'BUY':
            action_context = "BUY (Accumulate)" if trend_from_analysis == 'downtrend' else "BUY (Momentum)"
        elif action == 'SELL':
            action_context = "SELL (Take Profits)" if trend_from_analysis == 'uptrend' or 'overbought' in analysis_reasoning else "SELL (Exit)"
        else:
            action_context = "HOLD (Wait)"
            # Check if it's a strategic wait for entry
            if trend_from_analysis == 'downtrend' or 'trend reversal' in analysis_reasoning:
                action_context = "HOLD (Wait for Better Entry)"
        
        # Visual confidence bar
        bar_len = 20
        filled = int((confidence / 100) * bar_len)
        conf_bar = "█" * filled + "░" * (bar_len - filled)
        
        header = f"{'='*60}\n"
        header += f"🔭 MARKET TREND: {actual_trend}\n"
        header += f"📊 STRATEGY SIGNAL: {action_context}\n"
        header += f"🎯 CONFIDENCE: [{conf_bar}] {confidence}%\n"
        
        # Add context for wait scenarios
        if "Wait" in action_context:
            header += f"   ⚠️ STRATEGY: Waiting for optimal entry point\n"
        
        header += f"{'='*60}\n\n"
        
        # Display current market price in both USD and EUR
        if current_price_display and current_price_display > 0:
            # Format in USD (always show USD first as it's the base currency)
            price_usd = f"${current_price_display:,.2f}"
            # Convert to EUR
            price_eur = self.localization.convert_from_usd(current_price_display, 'EUR')
            price_eur_formatted = f"€{price_eur:,.2f}"
            logger.info(f"🔍 PRICE DEBUG: Displaying price - Raw: {current_price_display}, USD: {price_usd}, EUR: {price_eur_formatted}")
            header += f"📊 CURRENT MARKET PRICE:\n"
            header += f"   {price_usd} USD / {price_eur_formatted} EUR\n\n"
        
        # Add prediction details (show in both USD and EUR)
        if action in ['BUY', 'SELL'] and entry_price > 0 and target_price > 0:
            header += f"📊 PREDICTION DETAILS:\n"
            # Format entry price
            entry_usd = f"${entry_price:,.2f}"
            entry_eur = f"€{self.localization.convert_from_usd(entry_price, 'EUR'):,.2f}"
            header += f"   Entry Price: {entry_usd} USD / {entry_eur} EUR\n"
            
            # Check if entry price differs significantly from current price (indicates "wait for entry")
            if current_price_display and current_price_display > 0 and entry_price > 0:
                price_diff_pct = abs(current_price_display - entry_price) / current_price_display * 100
                if price_diff_pct > 5:  # Entry price is >5% different from current
                    if entry_price < current_price_display:
                        header += f"   ⚠️ WAIT FOR ENTRY: Entry price (${entry_price:.2f}) is {price_diff_pct:.1f}% below current price (${current_price_display:.2f})\n"
                        header += f"      💡 Recommendation: Wait for price to drop to entry level before buying\n"
                    else:
                        header += f"   ⚠️ ENTRY ABOVE CURRENT: Entry price (${entry_price:.2f}) is {price_diff_pct:.1f}% above current price (${current_price_display:.2f})\n"
                else:
                    header += f"   ✅ OPTIMAL ENTRY: Current price aligns with calculated entry target (${entry_price:.2f})\n"
                    header += f"      💡 Recommendation: This is a favorable position according to hybrid calculations.\n"
            # Format target price
            target_usd = f"${target_price:,.2f}"
            target_eur = f"€{self.localization.convert_from_usd(target_price, 'EUR'):,.2f}"
            header += f"   Target Price: {target_usd} USD / {target_eur} EUR\n"
            # Format stop loss
            stop_usd = f"${stop_loss:,.2f}"
            stop_eur = f"€{self.localization.convert_from_usd(stop_loss, 'EUR'):,.2f}"
            header += f"   Stop Loss: {stop_usd} USD / {stop_eur} EUR\n"
            header += f"   Estimated Days to Target: {estimated_days} days\n"
            header += f"   Estimated Target Date: {target_date_str}\n\n"
        
        # Add trend reversal predictions if available
        trend_predictions = analysis.get('trend_predictions', [])
        if trend_predictions:
            header += f"{'='*60}\n"
            header += f"📈 TREND REVERSAL PREDICTIONS:\n"
            header += f"{'='*60}\n\n"
            
            for pred in trend_predictions[:3]:  # Show top 3 predictions
                change_type = pred.get('predicted_change', '')
                estimated_days = pred.get('estimated_days', 0)
                confidence = pred.get('confidence', 0)
                reasoning = pred.get('reasoning', '')
                
                # Calculate predicted price movement
                current_price = current_price_display or entry_price
                if current_price > 0:
                    # Estimate price drop/rise based on trend change type
                    # Apply learned adjustment factor from Megamind
                    price_adjustment = self.trend_change_predictor.learned_adjustments.get('price_prediction_adjustment', 1.0)
                    
                    if change_type == 'bearish_reversal':
                        # Price will fall before reversal
                        # Estimate based on typical reversal patterns (5-15% drop)
                        base_drop_pct = min(15, max(5, confidence / 5))  # 5-15% based on confidence
                        estimated_drop_pct = base_drop_pct * price_adjustment  # Apply learned adjustment
                        predicted_price = current_price * (1 - estimated_drop_pct / 100)
                        price_change = current_price - predicted_price
                        header += f"🔽 Bearish Reversal Expected:\n"
                        header += f"   Predicted Price Drop: ${price_change:.2f} ({estimated_drop_pct:.1f}%)\n"
                        header += f"   Predicted Low Price: ${predicted_price:.2f}\n"
                    elif change_type == 'bullish_reversal':
                        # Price will fall before reversal (oversold bounce)
                        base_drop_pct = min(10, max(3, confidence / 7))  # 3-10% drop before bounce
                        estimated_drop_pct = base_drop_pct * price_adjustment  # Apply learned adjustment
                        predicted_price = current_price * (1 - estimated_drop_pct / 100)
                        price_change = current_price - predicted_price
                        header += f"🔼 Bullish Reversal Expected:\n"
                        header += f"   Predicted Price Drop Before Bounce: ${price_change:.2f} ({estimated_drop_pct:.1f}%)\n"
                        header += f"   Predicted Low Price: ${predicted_price:.2f}\n"
                    else:
                        # Continuation - show expected movement
                        base_move_pct = min(8, max(2, confidence / 10))  # 2-8% movement
                        estimated_move_pct = base_move_pct * price_adjustment  # Apply learned adjustment
                        predicted_price = current_price * (1 + estimated_move_pct / 100)
                        price_change = predicted_price - current_price
                        header += f"➡️ Trend Continuation Expected:\n"
                        header += f"   Predicted Price Movement: ${abs(price_change):.2f} ({estimated_move_pct:.1f}%)\n"
                        header += f"   Predicted Price: ${predicted_price:.2f}\n"
                    
                    header += f"   Estimated Days Until Change: {estimated_days} days\n"
                    header += f"   Confidence: {confidence:.0f}%\n"
                    header += f"   Reasoning: {reasoning}\n\n"
            
            header += f"{'='*60}\n"
        
        header += f"{'='*60}\n"
        header += f"ANALYSIS:\n"
        header += f"{'='*60}\n\n"
        
        self.analysis_text.insert(tk.END, header)
        
        # Add Buy Opportunity Data if available
        buy_opp = analysis.get('buy_opportunity')
        if buy_opp:
            opp_header = f"{'='*60}\n"
            opp_header += "💎 BUY OPPORTUNITY DATA (MEGAMIND)\n"
            opp_header += f"{'='*60}\n\n"
            
            opp_header += f"   Predicted Better Entry: ${buy_opp['predicted_price']:,.2f} USD / €{self.localization.convert_from_usd(buy_opp['predicted_price'], 'EUR'):,.2f} EUR\n"
            opp_header += f"   Estimated Date: {buy_opp['predicted_date']}\n"
            opp_header += f"   Confidence: {buy_opp['confidence']:.1f}%\n"
            opp_header += f"   Reasoning: {buy_opp['reasoning']}\n\n"
            
            opp_header += f"{'='*60}\n\n"
            self.analysis_text.insert(tk.END, opp_header)
            
        self.analysis_text.insert(tk.END, reasoning)
        
        # Display Seeker AI research if available
        if 'seeker_ai_research' in analysis:
            self.root.after(100, self._display_seeker_ai_analysis, analysis['seeker_ai_research'], symbol)
        
        # Highlight recommendation with modern styling
        self.analysis_text.tag_add("header", "1.0", "3.0")
        self.analysis_text.tag_config("header", font=('Segoe UI', 12, 'bold'), 
                                     foreground=theme['accent'])
        
        # Update title with animation
        display_symbol = symbol or self.current_symbol or "Stock"
        self._fade_title(f"{display_symbol} - Analysis Results")
        
        # Generate and display charts
        self._display_charts(history_data, analysis.get('indicators', {}))
        
        # Store for potential calculation
        self.current_analysis = analysis
    
    def _display_seeker_ai_analysis(self, research: dict, symbol: str):
        """Display Seeker AI research results"""
        if not research or 'error' in research:
            if 'error' in research:
                error_msg = f"Seeker AI Error: {research['error']}\n\n"
                error_msg += "This might be due to:\n"
                error_msg += "• No API keys configured\n"
                error_msg += "• Network connectivity issues\n"
                error_msg += "• Rate limiting\n\n"
                error_msg += "Check the logs for more details."
                self.seeker_ai_text.delete('1.0', tk.END)
                self.seeker_ai_text.insert('1.0', error_msg)
            return
        
        theme = self.theme_manager.get_theme()
        
        # Clear and format Seeker AI tab
        self.seeker_ai_text.delete('1.0', tk.END)
        
        # Header
        header = f"{'='*70}\n"
        header += f"🔍 SEEKER AI ANALYSIS - {symbol}\n"
        header += f"{'='*70}\n\n"
        self.seeker_ai_text.insert(tk.END, header)
        
        # Summary
        if research.get('summary'):
            self.seeker_ai_text.insert(tk.END, "📋 SUMMARY\n")
            self.seeker_ai_text.insert(tk.END, f"{research['summary']}\n\n")
        
        # Sentiment Analysis
        sentiment_score = research.get('sentiment_score', 0.5)
        sentiment_label = research.get('sentiment_label', 'neutral')
        sentiment_emoji = "🟢" if sentiment_label == 'positive' else "🔴" if sentiment_label == 'negative' else "🟡"
        
        self.seeker_ai_text.insert(tk.END, f"{sentiment_emoji} SENTIMENT ANALYSIS\n")
        self.seeker_ai_text.insert(tk.END, f"Score: {sentiment_score:.2f} ({sentiment_label})\n")
        if research.get('sentiment_score'):
            sentiment_percent = sentiment_score * 100
            if sentiment_percent > 60:
                self.seeker_ai_text.insert(tk.END, f"→ Positive sentiment detected in recent news\n")
            elif sentiment_percent < 40:
                self.seeker_ai_text.insert(tk.END, f"→ Negative sentiment detected in recent news\n")
            else:
                self.seeker_ai_text.insert(tk.END, f"→ Neutral sentiment in recent news\n")
        self.seeker_ai_text.insert(tk.END, "\n")
        
        # Reputation
        reputation_score = research.get('reputation_score', 0.5)
        reputation_emoji = "⭐" if reputation_score > 0.7 else "⚠️" if reputation_score < 0.3 else "➖"
        self.seeker_ai_text.insert(tk.END, f"{reputation_emoji} REPUTATION SCORE: {reputation_score:.2f}\n")
        if reputation_score > 0.7:
            self.seeker_ai_text.insert(tk.END, "→ Strong reputation indicators\n")
        elif reputation_score < 0.3:
            self.seeker_ai_text.insert(tk.END, "→ Reputation concerns detected\n")
        else:
            self.seeker_ai_text.insert(tk.END, "→ Neutral reputation\n")
        self.seeker_ai_text.insert(tk.END, "\n")
        
        # Price Movement Explanation
        if research.get('price_movement_explanation'):
            self.seeker_ai_text.insert(tk.END, "📈 PRICE MOVEMENT EXPLANATION\n")
            self.seeker_ai_text.insert(tk.END, f"{research['price_movement_explanation']}\n\n")
        
        # Key Insights
        if research.get('key_insights'):
            self.seeker_ai_text.insert(tk.END, "💡 KEY INSIGHTS\n")
            for insight in research['key_insights']:
                self.seeker_ai_text.insert(tk.END, f"• {insight}\n")
            self.seeker_ai_text.insert(tk.END, "\n")
        
        # Risk Factors
        if research.get('risk_factors'):
            self.seeker_ai_text.insert(tk.END, "⚠️ RISK FACTORS\n")
            for risk in research['risk_factors']:
                self.seeker_ai_text.insert(tk.END, f"• {risk}\n")
            self.seeker_ai_text.insert(tk.END, "\n")
        
        # Opportunities
        if research.get('opportunities'):
            self.seeker_ai_text.insert(tk.END, "🚀 OPPORTUNITIES\n")
            for opp in research['opportunities']:
                self.seeker_ai_text.insert(tk.END, f"• {opp}\n")
            self.seeker_ai_text.insert(tk.END, "\n")
        
        # News Articles
        articles = research.get('news_articles', [])
        if articles:
            self.seeker_ai_text.insert(tk.END, f"📰 RECENT NEWS ({len(articles)} articles)\n")
            self.seeker_ai_text.insert(tk.END, f"{'-'*70}\n")
            for i, article in enumerate(articles[:5], 1):  # Show top 5
                title = article.get('title', 'No title')
                source = article.get('source', 'Unknown source')
                self.seeker_ai_text.insert(tk.END, f"{i}. {title}\n")
                self.seeker_ai_text.insert(tk.END, f"   Source: {source}\n")
                if article.get('published_at'):
                    self.seeker_ai_text.insert(tk.END, f"   Date: {article['published_at'][:10]}\n")
                if article.get('description'):
                    desc = article['description'][:150]  # Limit description length
                    self.seeker_ai_text.insert(tk.END, f"   {desc}...\n")
                self.seeker_ai_text.insert(tk.END, "\n")
            if len(articles) > 5:
                self.seeker_ai_text.insert(tk.END, f"... and {len(articles) - 5} more articles\n")
        else:
            self.seeker_ai_text.insert(tk.END, "📰 RECENT NEWS\n")
            self.seeker_ai_text.insert(tk.END, "No recent news articles found.\n\n")
        
        # Reputation Factors (if available)
        if research.get('reputation_factors'):
            self.seeker_ai_text.insert(tk.END, f"{'='*70}\n")
            self.seeker_ai_text.insert(tk.END, "REPUTATION FACTORS\n")
            for factor in research['reputation_factors'][:5]:  # Show top 5
                self.seeker_ai_text.insert(tk.END, f"• {factor}\n")
        
        # Apply formatting
        self.seeker_ai_text.tag_add("header", "1.0", "3.0")
        self.seeker_ai_text.tag_config("header", font=('Segoe UI', 12, 'bold'), foreground=theme['accent'])
        
        # Scroll to top
        self.seeker_ai_text.see('1.0')
    
    def _train_on_stock_background(self, symbol: str, strategy: str):
        """Train ML model on a specific stock's historical data in the background"""
        if not symbol:
            return
        
        def train_in_background():
            try:
                logger.info(f"🔄 Starting background training on {symbol} for {strategy} strategy...")
                
                # Calculate date range (use last 5 years for training)
                from datetime import datetime, timedelta
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
                
                # Generate training samples for this stock
                try:
                    # Determine lookforward_days based on strategy
                    if strategy == "trading":
                        lookforward_days = 10
                    elif strategy == "investing":
                        lookforward_days = 547  # ~1.5 years
                    else:  # mixed
                        lookforward_days = 21  # ~3 weeks
                    
                    # Generate training samples
                    training_samples = self.training_manager.data_generator.generate_training_samples(
                        symbol, start_date, end_date, strategy, lookforward_days
                    )
                    
                    if len(training_samples) < 50:
                        logger.info(f"⚠️ Insufficient training samples for {symbol}: {len(training_samples)} < 50. "
                                  f"Data will be used in next full training cycle.")
                        return
                    
                    logger.info(f"✅ Generated {len(training_samples)} training samples for {symbol}")
                    
                    # Load existing ML model
                    from ml_training import StockPredictionML
                    ml_model = StockPredictionML(self.data_dir, strategy)
                    
                    # If model exists, we can do incremental training
                    # Otherwise, we need at least 100 samples to train a new model
                    if ml_model.is_trained():
                        # Model exists - we can do incremental training
                        # For now, we'll retrain with existing + new data
                        # In the future, we could implement true incremental learning
                        logger.info(f"🔄 Model exists for {strategy}, preparing for incremental update...")
                        
                        # Check if we have enough samples to retrain
                        # We need at least 100 samples total, so if this stock alone has 100+, we can retrain
                        if len(training_samples) >= 100:
                            logger.info(f"✅ Enough samples ({len(training_samples)}) to retrain model on {symbol}")
                            result = ml_model.train(training_samples)
                            
                            if 'error' not in result:
                                test_acc = result.get('test_accuracy', 0) * 100
                                train_acc = result.get('train_accuracy', 0) * 100
                                logger.info(f"✅ Model retrained on {symbol}! "
                                          f"Train: {train_acc:.1f}%, Test: {test_acc:.1f}%")
                                
                                # Record learning event
                                self.learning_tracker.record_background_training(
                                    symbol, len(training_samples), strategy, True
                                )
                                
                                # Update ML status in UI
                                self.root.after(0, self._update_ml_status)
                            else:
                                logger.warning(f"⚠️ Retraining failed: {result.get('error')}")
                        else:
                            logger.info(f"📊 {len(training_samples)} samples from {symbol} - "
                                      f"will be combined with other stocks in next full training")
                    else:
                        # No model exists yet - need at least 100 samples to train
                        if len(training_samples) >= 100:
                            logger.info(f"✅ Training new model on {symbol} ({len(training_samples)} samples)...")
                            result = ml_model.train(training_samples)
                            
                            if 'error' not in result:
                                test_acc = result.get('test_accuracy', 0) * 100
                                train_acc = result.get('train_accuracy', 0) * 100
                                logger.info(f"✅ New model trained on {symbol}! "
                                          f"Train: {train_acc:.1f}%, Test: {test_acc:.1f}%")
                                
                                # Record learning event
                                self.learning_tracker.record_background_training(
                                    symbol, len(training_samples), strategy, True
                                )
                                
                                # Update ML status in UI
                                self.root.after(0, self._update_ml_status)
                            else:
                                logger.warning(f"⚠️ Training failed: {result.get('error')}")
                        else:
                            logger.info(f"📊 {len(training_samples)} samples from {symbol} - "
                                      f"need {100 - len(training_samples)} more to train new model")
                    
                except Exception as e:
                    logger.warning(f"Error generating training samples for {symbol}: {e}")
                    
            except Exception as e:
                logger.error(f"Error in background training for {symbol}: {e}")
        
        # Run in background thread (non-blocking)
        thread = threading.Thread(target=train_in_background)
        thread.daemon = True
        thread.start()
    
    def _fade_title(self, new_text: str):
        """Animate title change"""
        theme = self.theme_manager.get_theme()
        self.results_title.config(text=new_text, fg=theme['fg'])
    
    def _display_charts(self, history_data: dict, indicators: dict):
        """Display stock charts"""
        try:
            # Main candlestick chart
            if self.chart_canvas:
                self.chart_canvas.get_tk_widget().destroy()
            
            fig = self.chart_generator.create_candlestick_chart(history_data, indicators)
            self.chart_canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            self.chart_canvas.draw()
            self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Indicators chart
            if self.indicators_canvas:
                self.indicators_canvas.get_tk_widget().destroy()
            
            fig2 = self.chart_generator.create_indicator_chart(history_data, indicators)
            self.indicators_canvas = FigureCanvasTkAgg(fig2, self.indicators_frame)
            self.indicators_canvas.draw()
            self.indicators_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            logger.error(f"Error displaying charts: {e}")
    
    def _update_risk_management_tab(self, stock_data: dict, analysis: dict):
        """Update risk management tab with current stock data"""
        if not HAS_NEW_MODULES or not self.portfolio.risk_manager:
            return
        
        try:
            current_price = stock_data.get('price', 0)
            if current_price > 0:
                # Pre-fill entry price
                self.rm_entry_price.delete(0, tk.END)
                self.rm_entry_price.insert(0, str(current_price))
                self.rm_current_price.delete(0, tk.END)
                self.rm_current_price.insert(0, str(current_price))
                
                # Get ATR from analysis if available
                if 'indicators' in analysis:
                    atr = analysis['indicators'].get('atr', 0)
                    if atr > 0:
                        self.rm_volatility.delete(0, tk.END)
                        self.rm_volatility.insert(0, str(atr))
        except Exception as e:
            logger.debug(f"Could not update risk management tab: {e}")
    
    def _update_advanced_analytics_tab(self, symbol: str):
        """Update advanced analytics tab with current symbol"""
        if symbol:
            # Pre-fill symbol in correlation and optimization
            if hasattr(self, 'aa_symbols_entry'):
                current_symbols = self.aa_symbols_entry.get()
                if symbol.upper() not in current_symbols.upper():
                    if current_symbols:
                        self.aa_symbols_entry.delete(0, tk.END)
                        self.aa_symbols_entry.insert(0, f"{current_symbols},{symbol.upper()}")
                    else:
                        self.aa_symbols_entry.insert(0, symbol.upper())
            
            if hasattr(self, 'aa_opt_symbols_entry'):
                current_opt = self.aa_opt_symbols_entry.get()
                if symbol.upper() not in current_opt.upper():
                    if current_opt:
                        self.aa_opt_symbols_entry.delete(0, tk.END)
                        self.aa_opt_symbols_entry.insert(0, f"{current_opt},{symbol.upper()}")
                    else:
                        self.aa_opt_symbols_entry.insert(0, symbol.upper())
    
    def _calculate_potential(self):
        """Calculate potential win/loss for current analysis"""
        if not self.current_analysis or 'error' in self.current_analysis:
            messagebox.showwarning(self.localization.t('no_analysis'), self.localization.t('please_analyze_stock'))
            return
        
        try:
            budget = float(self.budget_entry.get())
            if budget <= 0:
                raise ValueError(self.localization.t('balance_must_positive'))
        except ValueError as e:
            messagebox.showerror(self.localization.t('input_error'), f"{self.localization.t('input_error')}: {str(e)}")
            return
        
        # Fetch CURRENT price (not the price from when analysis was done)
        # Force refresh to get the latest price, not cached data
        symbol = self.current_symbol or "UNKNOWN"
        try:
            current_stock_data = self.data_fetcher.fetch_stock_data(symbol, force_refresh=True)
            current_price_usd = current_stock_data.get('price', 0)
            if current_price_usd <= 0:
                raise ValueError("Could not fetch current price")
        except Exception as e:
            logger.error(f"Error fetching current price: {e}")
            messagebox.showerror(self.localization.t('error'), 
                               f"Could not fetch current price for {symbol}. Using analysis price.")
            current_price_usd = None
        
        recommendation = self.current_analysis.get('recommendation', {})
        action = recommendation.get('action', 'HOLD')
        
        # Use current price as entry if available, otherwise use recommendation entry price
        if current_price_usd and current_price_usd > 0:
            entry_price = current_price_usd
            # Adjust target and stop loss proportionally based on price change
            old_entry = recommendation.get('entry_price', current_price_usd)
            if old_entry > 0:
                price_ratio = current_price_usd / old_entry
                target_price = recommendation.get('target_price', current_price_usd) * price_ratio
                stop_loss = recommendation.get('stop_loss', current_price_usd) * price_ratio
            else:
                target_price = recommendation.get('target_price', current_price_usd)
                stop_loss = recommendation.get('stop_loss', current_price_usd)
        else:
            entry_price = recommendation.get('entry_price', 0)
            target_price = recommendation.get('target_price', 0)
            stop_loss = recommendation.get('stop_loss', 0)
        
        if entry_price <= 0:
            messagebox.showwarning(self.localization.t('warning'), self.localization.t('warning'))
            return
        
        # Validate targets haven't already been reached
        strategy = self.strategy_var.get()
        if action == "BUY" and current_price_usd >= target_price:
            # Target already reached - recalculate new target
            if strategy == "trading":
                target_price = current_price_usd * 1.03
            elif strategy == "mixed":
                target_price = current_price_usd * 1.06
            else:  # investing
                target_price = current_price_usd * 1.12
            
            messagebox.showwarning(
                self.localization.t('warning'),
                f"⚠️ Target already reached!\n"
                f"Current: {self.localization.format_currency(current_price_usd)}\n"
                f"Old Target: {self.localization.format_currency(recommendation.get('target_price', 0))}\n"
                f"New Target: {self.localization.format_currency(target_price)}"
            )
        
        elif action == "SELL" and current_price_usd >= target_price:
            # Target already reached - recalculate new target
            # NOTE: SELL = bullish (expecting price to go UP), so target should be ABOVE entry
            # If current price >= target, the target was reached (price went up)
            if strategy == "trading":
                target_price = current_price_usd * 1.03  # 3% above current
            elif strategy == "mixed":
                target_price = current_price_usd * 1.06  # 6% above current
            else:  # investing
                target_price = current_price_usd * 1.12  # 12% above current
            
            messagebox.showwarning(
                self.localization.t('warning'),
                f"⚠️ Target already reached!\n"
                f"Current: {self.localization.format_currency(current_price_usd)}\n"
                f"Old Target: {self.localization.format_currency(recommendation.get('target_price', 0))}\n"
                f"New Target: {self.localization.format_currency(target_price)}"
            )
        
        # Validate stop loss is correct for the action type
        if action == "BUY":
            if stop_loss <= 0 or stop_loss >= entry_price:
                # Stop loss is invalid for BUY (should be below entry)
                if strategy == "trading":
                    stop_loss = entry_price * 0.97  # 3% stop loss
                elif strategy == "mixed":
                    stop_loss = entry_price * 0.95  # 5% stop loss
                else:  # investing
                    stop_loss = entry_price * 0.80  # 20% stop loss
        elif action == "SELL":
            if stop_loss <= 0 or stop_loss >= entry_price:
                # Stop loss is invalid for SELL (bullish) - should be BELOW entry
                # NOTE: SELL = bullish (expecting price to go UP), stop loss limits loss if price goes DOWN
                if strategy == "trading":
                    stop_loss = entry_price * 0.97  # 3% stop loss below entry
                elif strategy == "mixed":
                    stop_loss = entry_price * 0.95  # 5% stop loss below entry
                else:  # investing
                    stop_loss = entry_price * 0.80  # 20% stop loss below entry
        
        # Check if HOLD recommendation (needs special handling)
        is_hold_recommendation = (action == 'HOLD')
        
        # For HOLD recommendations, calculate as BUY scenario for hypothetical analysis
        # Determine if it's a bullish or bearish HOLD based on target vs entry
        if is_hold_recommendation:
            # If target is above entry, treat as bullish (BUY scenario)
            # If target is below entry, treat as bearish (SELL scenario)
            # If all prices are the same, use 5% up/down hypothetical
            if entry_price == target_price == stop_loss:
                # All prices same - use hypothetical 5% up/down
                current_price = entry_price
                target_price = current_price * 1.05  # 5% target
                stop_loss = current_price * 0.95     # 5% stop loss
                action = 'BUY'  # Use BUY for hypothetical calculation
            elif target_price > entry_price:
                # Target above entry - bullish HOLD, calculate as BUY
                action = 'BUY'
            else:
                # Target below entry - bearish HOLD, calculate as SELL
                action = 'SELL'
        
        potential = self.portfolio.calculate_potential_trade(
            symbol, budget, entry_price, target_price, stop_loss, action
        )
        
        if 'error' in potential:
            self.potential_text.delete(1.0, tk.END)
            self.potential_text.insert(tk.END, f"{self.localization.t('error')}: {potential['error']}")
            return
        
        # Format output with better styling and localization
        output = f"{'='*60}\n"
        output += f"{self.localization.t('potential_trade')} - {potential['symbol']}\n"
        output += f"{'='*60}\n\n"
        
        # Add trend reversal predictions if available
        trend_predictions = self.current_analysis.get('trend_predictions', [])
        if trend_predictions:
            output += f"{'='*60}\n"
            output += f"📈 TREND REVERSAL PREDICTIONS:\n"
            output += f"{'='*60}\n\n"
            
            for pred in trend_predictions[:3]:  # Show top 3 predictions
                change_type = pred.get('predicted_change', '')
                estimated_days = pred.get('estimated_days', 0)
                confidence = pred.get('confidence', 0)
                reasoning = pred.get('reasoning', '')
                
                # Calculate predicted price movement
                current_price = current_price_usd or entry_price
                if current_price > 0:
                    # Estimate price drop/rise based on trend change type
                    # Apply learned adjustment factor from Megamind
                    price_adjustment = self.trend_change_predictor.learned_adjustments.get('price_prediction_adjustment', 1.0)
                    
                    if change_type == 'bearish_reversal':
                        # Price will fall before reversal
                        base_drop_pct = min(15, max(5, confidence / 5))  # 5-15% based on confidence
                        estimated_drop_pct = base_drop_pct * price_adjustment  # Apply learned adjustment
                        predicted_price = current_price * (1 - estimated_drop_pct / 100)
                        price_change = current_price - predicted_price
                        output += f"🔽 Bearish Reversal Expected:\n"
                        output += f"   Predicted Price Drop: ${price_change:.2f} ({estimated_drop_pct:.1f}%)\n"
                        output += f"   Predicted Low Price: ${predicted_price:.2f}\n"
                    elif change_type == 'bullish_reversal':
                        # Price will fall before reversal (oversold bounce)
                        base_drop_pct = min(10, max(3, confidence / 7))  # 3-10% drop before bounce
                        estimated_drop_pct = base_drop_pct * price_adjustment  # Apply learned adjustment
                        predicted_price = current_price * (1 - estimated_drop_pct / 100)
                        price_change = current_price - predicted_price
                        output += f"🔼 Bullish Reversal Expected:\n"
                        output += f"   Predicted Price Drop Before Bounce: ${price_change:.2f} ({estimated_drop_pct:.1f}%)\n"
                        output += f"   Predicted Low Price: ${predicted_price:.2f}\n"
                    else:
                        # Continuation - show expected movement
                        base_move_pct = min(8, max(2, confidence / 10))  # 2-8% movement
                        estimated_move_pct = base_move_pct * price_adjustment  # Apply learned adjustment
                        predicted_price = current_price * (1 + estimated_move_pct / 100)
                        price_change = predicted_price - current_price
                        output += f"➡️ Trend Continuation Expected:\n"
                        output += f"   Predicted Price Movement: ${abs(price_change):.2f} ({estimated_move_pct:.1f}%)\n"
                        output += f"   Predicted Price: ${predicted_price:.2f}\n"
                    
                    output += f"   Estimated Days Until Change: {estimated_days} days\n"
                    output += f"   Confidence: {confidence:.0f}%\n"
                    output += f"   Reasoning: {reasoning}\n\n"
            
            output += f"{'='*60}\n\n"
        
        # Add HOLD warning if applicable
        if is_hold_recommendation:
            output += f"⚠️ {self.localization.t('hold_no_trade')}\n"
            output += f"{self.localization.t('hold_explanation')}\n\n"
            output += f"{self.localization.t('hypothetical_scenario')}\n"
            output += f"{'='*60}\n\n"
        
        # Add current price display in both USD and EUR
        output += f"📊 CURRENT MARKET PRICE:\n"
        if current_price_usd and current_price_usd > 0:
            # Format in USD (always show USD first as it's the base currency)
            price_usd_str = f"${current_price_usd:,.2f}"
            # Convert to EUR
            price_eur = self.localization.convert_from_usd(current_price_usd, 'EUR')
            price_eur_str = f"€{price_eur:,.2f}"
            output += f"   Current Price: {price_usd_str} USD / {price_eur_str} EUR\n"
            price_change = ((current_price_usd - entry_price) / entry_price * 100) if entry_price > 0 else 0
            if abs(price_change) > 0.01:
                change_symbol = "📈" if price_change > 0 else "📉"
                output += f"   {change_symbol} Change from Entry: {price_change:+.2f}%\n"
        else:
            # Fallback: show entry price in both currencies
            price_usd_str = f"${entry_price:,.2f}"
            price_eur = self.localization.convert_from_usd(entry_price, 'EUR')
            price_eur_str = f"€{price_eur:,.2f}"
            output += f"   Current Price: {price_usd_str} USD / {price_eur_str} EUR (from analysis)\n"
        output += "\n"
        
        # Check for existing prediction for this stock (exclude HOLD/AVOID as they're not actionable)
        symbol_upper = symbol.upper()
        
        # First, clean up any invalid HOLD/AVOID predictions for this symbol
        cleaned = False
        for pred in self.predictions_tracker.predictions:
            if (pred.get('symbol', '').upper() == symbol_upper 
                and pred.get('status') == 'active'
                and pred.get('action') in ['HOLD', 'AVOID']):
                # Mark invalid HOLD/AVOID predictions as expired
                pred['status'] = 'expired'
                cleaned = True
                logger.debug(f"Cleaned up invalid {pred.get('action')} prediction for {symbol}")
            elif (pred.get('symbol', '').upper() == symbol_upper 
                  and pred.get('status') == 'active'
                  and pred.get('action') in ['BUY', 'SELL']
                  and pred.get('entry_price', 0) > 0
                  and abs(pred.get('target_price', 0) - pred.get('entry_price', 0)) < 0.01):
                # Mark predictions with same entry/target as expired
                pred['status'] = 'expired'
                cleaned = True
                logger.debug(f"Cleaned up invalid prediction with same entry/target for {symbol}")
        
        if cleaned:
            self.predictions_tracker.save()
        
        # Now get only valid actionable predictions
        active_predictions = [
            p for p in self.predictions_tracker.predictions 
            if p.get('symbol', '').upper() == symbol_upper 
            and p.get('status') == 'active'
            and p.get('action') in ['BUY', 'SELL']  # Only show actionable predictions
            and p.get('entry_price', 0) > 0
            and abs(p.get('target_price', 0) - p.get('entry_price', 0)) >= 0.01  # Entry and target must be different
        ]
        
        if active_predictions:
            # Get the most recent active prediction
            latest_prediction = max(active_predictions, key=lambda p: p.get('timestamp', ''))
            pred_action = latest_prediction.get('action', 'HOLD')
            pred_target = latest_prediction.get('target_price', 0)
            pred_entry = latest_prediction.get('entry_price', 0)
            pred_estimated_date = latest_prediction.get('estimated_target_date')
            pred_estimated_days = latest_prediction.get('estimated_days', 0)
            pred_confidence = latest_prediction.get('confidence', 0)
            
            # Skip if entry and target are the same (invalid prediction)
            if pred_entry > 0 and abs(pred_target - pred_entry) < 0.01:
                logger.debug(f"Skipping invalid prediction for {symbol}: entry and target are the same")
            else:
                output += f"🤖 ACTIVE PREDICTION FOUND:\n"
                output += f"   Action: {pred_action}\n"
                # Format entry price in both currencies
                pred_entry_usd = f"${pred_entry:,.2f}"
                pred_entry_eur = f"€{self.localization.convert_from_usd(pred_entry, 'EUR'):,.2f}"
                output += f"   Entry Price: {pred_entry_usd} USD / {pred_entry_eur} EUR\n"
                # Format target price in both currencies
                pred_target_usd = f"${pred_target:,.2f}"
                pred_target_eur = f"€{self.localization.convert_from_usd(pred_target, 'EUR'):,.2f}"
                output += f"   Target Price: {pred_target_usd} USD / {pred_target_eur} EUR\n"
                output += f"   Confidence: {pred_confidence:.1f}%\n"
                
                if pred_estimated_date:
                    try:
                        from datetime import datetime
                        target_date = datetime.fromisoformat(pred_estimated_date)
                        now = datetime.now()
                        days_remaining = (target_date - now).days
                        
                        if days_remaining > 0:
                            output += f"   📅 Estimated Target Date: {target_date.strftime('%Y-%m-%d')}\n"
                            output += f"   ⏰ Days Remaining: {days_remaining} days\n"
                            
                            # Calculate expected price movement (show in both currencies)
                            if current_price_usd and current_price_usd > 0:
                                if pred_action == "BUY":
                                    price_diff = pred_target - current_price_usd
                                    price_change_pct = (price_diff / current_price_usd) * 100
                                    price_diff_usd = f"${price_diff:,.2f}"
                                    price_diff_eur = f"€{self.localization.convert_from_usd(price_diff, 'EUR'):,.2f}"
                                    output += f"   📈 Expected Rise: {price_diff_usd} USD / {price_diff_eur} EUR ({price_change_pct:+.2f}%)\n"
                                elif pred_action == "SELL":
                                    price_diff = current_price_usd - pred_target
                                    price_change_pct = (price_diff / current_price_usd) * 100
                                    price_diff_usd = f"${price_diff:,.2f}"
                                    price_diff_eur = f"€{self.localization.convert_from_usd(price_diff, 'EUR'):,.2f}"
                                    output += f"   📉 Expected Fall: {price_diff_usd} USD / {price_diff_eur} EUR ({price_change_pct:+.2f}%)\n"
                        else:
                            output += f"   ⚠️ Target date passed ({abs(days_remaining)} days ago)\n"
                    except Exception as e:
                        logger.debug(f"Error parsing estimated date: {e}")
                elif pred_estimated_days:
                    from datetime import datetime, timedelta
                    target_date = datetime.now() + timedelta(days=pred_estimated_days)
                    output += f"   📅 Estimated Target Date: {target_date.strftime('%Y-%m-%d')}\n"
                    output += f"   ⏰ Days Remaining: {pred_estimated_days} days\n"
                
                output += "\n"
        
        # Show the calculated action (BUY/SELL) for the scenario
        output += f"{'='*60}\n"
        output += f"TRADE CALCULATION:\n"
        output += f"{'='*60}\n\n"
        # For HOLD recommendations, clarify that this is hypothetical
        action_display = potential['action']
        if is_hold_recommendation:
            action_display = f"{action_display} (Hypothetical - Original Recommendation: HOLD)"
        output += f"{self.localization.t('action')}: {action_display}\n"
        output += f"{self.localization.t('shares')}: {potential['shares']}\n"
        
        # Format budget used in both currencies
        budget_usd = f"${potential['budget_used']:,.2f}"
        budget_eur = f"€{self.localization.convert_from_usd(potential['budget_used'], 'EUR'):,.2f}"
        output += f"{self.localization.t('budget_used')}: {budget_usd} USD / {budget_eur} EUR\n\n"
        
        # Format entry price in both currencies
        entry_usd = f"${potential['entry_price']:,.2f}"
        entry_eur = f"€{self.localization.convert_from_usd(potential['entry_price'], 'EUR'):,.2f}"
        output += f"{self.localization.t('entry_price')}: {entry_usd} USD / {entry_eur} EUR\n"
        
        # Format target price in both currencies
        target_usd = f"${potential['target_price']:,.2f}"
        target_eur = f"€{self.localization.convert_from_usd(potential['target_price'], 'EUR'):,.2f}"
        output += f"{self.localization.t('target_price')}: {target_usd} USD / {target_eur} EUR\n"
        
        # Format stop loss in both currencies
        stop_usd = f"${potential['stop_loss']:,.2f}"
        stop_eur = f"€{self.localization.convert_from_usd(potential['stop_loss'], 'EUR'):,.2f}"
        output += f"{self.localization.t('stop_loss')}: {stop_usd} USD / {stop_eur} EUR\n\n"
        
        # Format potential win in both currencies
        win_usd = f"${potential['potential_win']:,.2f}"
        win_eur = f"€{self.localization.convert_from_usd(potential['potential_win'], 'EUR'):,.2f}"
        output += f"{self.localization.t('potential_win')}: {win_usd} USD / {win_eur} EUR ({potential['potential_win_percent']:.2f}%)\n"
        
        # Format potential loss in both currencies
        loss_usd = f"${potential['potential_loss']:,.2f}"
        loss_eur = f"€{self.localization.convert_from_usd(potential['potential_loss'], 'EUR'):,.2f}"
        output += f"{self.localization.t('potential_loss')}: {loss_usd} USD / {loss_eur} EUR ({potential['potential_loss_percent']:.2f}%)\n\n"
        
        output += f"{self.localization.t('risk_reward_ratio')}: {potential['risk_reward_ratio']:.2f}\n"
        
        self.potential_text.delete(1.0, tk.END)
        self.potential_text.insert(tk.END, output)
    
    def _show_about(self):
        """Show About dialog with disclaimer"""
        theme = self.theme_manager.get_theme()
        
        dialog = tk.Toplevel(self.root)
        dialog.title("About Stocker")
        dialog.geometry("700x600")
        dialog.resizable(False, False)
        dialog.config(bg=theme['bg'])
        
        # Center dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (700 // 2)
        y = (dialog.winfo_screenheight() // 2) - (600 // 2)
        dialog.geometry(f"700x600+{x}+{y}")
        
        # Try to set icon
        icon_path = Path(__file__).parent / "stocker_icon.ico"
        if icon_path.exists():
            try:
                dialog.iconbitmap(str(icon_path))
            except (tk.TclError, OSError) as e:
                logger.debug(f"Could not set dialog icon: {e}")
        
        # Main container
        main_frame = tk.Frame(dialog, bg=theme['bg'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=30)
        
        # App icon/logo
        icon_frame = tk.Frame(main_frame, bg=theme['bg'])
        icon_frame.pack(pady=(0, 20))
        
        icon_path = Path(__file__).parent / "stocker_icon.png"
        if icon_path.exists():
            try:
                from PIL import Image, ImageTk
                img = Image.open(icon_path)
                img = img.resize((80, 80), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                icon_label = tk.Label(icon_frame, image=photo, bg=theme['bg'])
                icon_label.image = photo
                icon_label.pack()
            except (OSError, tk.TclError, AttributeError) as e:
                logger.debug(f"Could not load icon image: {e}")
                icon_label = tk.Label(icon_frame, text="📈", font=('Segoe UI', 48),
                                    bg=theme['bg'], fg=theme['accent'])
                icon_label.pack()
        else:
            icon_label = tk.Label(icon_frame, text="📈", font=('Segoe UI', 48), 
                                bg=theme['bg'], fg=theme['accent'])
            icon_label.pack()
        
        # App name and version
        title_label = tk.Label(main_frame, text="Stocker", 
                             font=('Segoe UI', 28, 'bold'),
                             bg=theme['bg'], fg=theme['fg'])
        title_label.pack(pady=(10, 5))
        
        version_label = tk.Label(main_frame, text="Stock Trading & Investing Desktop App", 
                               font=('Segoe UI', 11),
                               bg=theme['bg'], fg=theme['text_secondary'])
        version_label.pack(pady=(0, 30))
        
        # Disclaimer section
        disclaimer_frame = tk.Frame(main_frame, bg=theme['card_bg'], relief=tk.RAISED, bd=1)
        disclaimer_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 20))
        
        disclaimer_inner = tk.Frame(disclaimer_frame, bg=theme['card_bg'])
        disclaimer_inner.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        disclaimer_title = tk.Label(disclaimer_inner, 
                                   text="⚠️ IMPORTANT DISCLAIMER",
                                   font=('Segoe UI', 14, 'bold'),
                                   bg=theme['card_bg'], fg=theme['warning'])
        disclaimer_title.pack(pady=(0, 15), anchor=tk.W)
        
        disclaimer_text = """The projections, predictions, and recommendations provided by this application are for informational and educational purposes only.

This application is NOT financial advice, and the creator of this application:
• Takes NO responsibility for any financial decisions or losses
• Is NOT a trained financial professional
• Does NOT guarantee the accuracy of any predictions or recommendations

All trading and investment decisions are made at your own risk. Always consult with a qualified financial advisor before making any investment decisions.

By using this application, you acknowledge that you understand and accept these terms."""
        
        disclaimer_label = tk.Label(disclaimer_inner, 
                                   text=disclaimer_text,
                                   font=('Segoe UI', 10),
                                   bg=theme['card_bg'], fg=theme['fg'],
                                   justify=tk.LEFT,
                                   wraplength=620)
        disclaimer_label.pack(anchor=tk.W)
        
        # Close button
        button_frame = tk.Frame(main_frame, bg=theme['bg'])
        button_frame.pack(fill=tk.X)
        
        close_btn = ModernButton(button_frame,
                                "Close",
                                command=dialog.destroy,
                                theme=theme)
        close_btn.pack()
    
    def _set_balance(self):
        """Set initial portfolio balance"""
        dialog = tk.Toplevel(self.root)
        dialog.title(self.localization.t('set_initial_balance'))
        dialog.geometry("300x150")
        dialog.transient(self.root)
        dialog.grab_set()
        
        currency_symbol = self.localization.format_currency_symbol()
        ttk.Label(dialog, text=f"{self.localization.t('initial_balance')} ({currency_symbol}):").pack(pady=10)
        balance_entry = ttk.Entry(dialog, width=20)
        balance_entry.pack(pady=5)
        # Display current balance in user's selected currency
        current_balance_display = self.localization.convert_from_usd(self.portfolio.balance)
        balance_entry.insert(0, str(current_balance_display))
        balance_entry.focus()
        
        def save_balance():
            try:
                balance_input = float(balance_entry.get())
                if balance_input < 0:
                    raise ValueError(self.localization.t('balance_must_positive'))
                
                # Convert the input from current currency to USD before storing
                # Portfolio always stores balance in USD (base currency)
                balance_usd = self.localization.convert_to_usd(balance_input, self.localization.currency)
                
                self.portfolio.set_balance(balance_usd)
                self._update_portfolio_display()
                dialog.destroy()
                messagebox.showinfo(self.localization.t('success'), self.localization.t('balance_updated'))
            except ValueError as e:
                messagebox.showerror(self.localization.t('error'), f"{self.localization.t('invalid_balance')}: {str(e)}")
        
        ttk.Button(dialog, text=self.localization.t('save'), command=save_balance).pack(pady=10)
        balance_entry.bind('<Return>', lambda e: save_balance())
    
    def _update_portfolio_display(self):
        """Update portfolio display with modern formatting"""
        theme = self.theme_manager.get_theme()
        stats = self.portfolio.get_statistics()
        self.portfolio_label.config(text=self.localization.format_currency(stats['balance']), fg=theme['accent'])
        self.stats_label.config(text=f"{self.localization.t('wins')}: {stats['wins']} | {self.localization.t('losses')}: {stats['losses']} | "
                                     f"{self.localization.t('win_rate')}: {stats['win_rate']:.1f}%",
                               fg=theme['text_secondary'])
        
        # Update S&P 500 comparison (in background to avoid blocking UI)
        self._update_sp500_comparison()
    
    def _schedule_portfolio_snapshots(self):
        """Schedule daily portfolio snapshot updates (at midnight)"""
        from datetime import datetime, timedelta
        
        def schedule_next_snapshot():
            now = datetime.now()
            # Calculate milliseconds until next midnight
            tomorrow = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            ms_until_midnight = int((tomorrow - now).total_seconds() * 1000)
            
            # Schedule snapshot update at midnight
            self.root.after(ms_until_midnight, lambda: self._take_portfolio_snapshot())
        
        # Take initial snapshot if needed
        snapshots = self.portfolio.get_portfolio_history()
        if not snapshots:
            self.portfolio.update_snapshot()
        
        # Schedule first snapshot update
        schedule_next_snapshot()
        
        # After each snapshot, schedule the next one
        def _take_and_schedule():
            self.portfolio.update_snapshot()
            schedule_next_snapshot()
            # Update S&P 500 comparison after snapshot
            self._update_sp500_comparison()
        
        # Store method for scheduling
        self._take_portfolio_snapshot = _take_and_schedule
        
        # Start scheduling (first check after 1 hour, then daily)
        self.root.after(3600000, schedule_next_snapshot)  # Check after 1 hour
    
    def _update_sp500_comparison(self):
        """Update S&P 500 comparison display (runs in background)"""
        def update_in_background():
            try:
                # Get portfolio snapshots
                snapshots = self.portfolio.get_portfolio_history()
                
                if not snapshots or len(snapshots) < 2:
                    # Not enough data yet
                    self.root.after(0, lambda: self.sp500_label.config(
                        text="S&P 500: Need more data",
                        fg=self.theme_manager.get_theme()['text_secondary']
                    ))
                    return
                
                # Compare to S&P 500
                initial_balance = self.portfolio.initial_balance
                comparison = self.model_benchmarker.compare_portfolio_to_sp500(snapshots, initial_balance)
                
                # Update UI on main thread
                def update_ui():
                    theme = self.theme_manager.get_theme()
                    if 'error' in comparison:
                        self.sp500_label.config(
                            text=f"S&P 500: {comparison.get('error', 'Error')}",
                            fg=theme['text_secondary']
                        )
                    elif comparison.get('comparison_available', False):
                        portfolio_return = comparison['portfolio']['return_pct']
                        sp500_return = comparison['sp500']['return_pct']
                        outperformance = comparison['comparison']['outperformance_pct']
                        
                        # Format text with color coding
                        if outperformance > 0:
                            color = '#4CAF50'  # Green for outperforming
                            icon = "📈"
                        elif outperformance < 0:
                            color = '#F44336'  # Red for underperforming
                            icon = "📉"
                        else:
                            color = theme['text_secondary']
                            icon = "➡️"
                        
                        text = f"{icon} Portfolio: {portfolio_return:+.1f}% | S&P 500: {sp500_return:+.1f}%"
                        if abs(outperformance) > 0.1:  # Only show if significant difference
                            text += f" ({outperformance:+.1f}%)"
                        
                        self.sp500_label.config(text=text, fg=color)
                    else:
                        self.sp500_label.config(
                            text="S&P 500: Calculating...",
                            fg=theme['text_secondary']
                        )
                
                self.root.after(0, update_ui)
                
            except Exception as e:
                logger.error(f"Error updating S&P 500 comparison: {e}")
                self.root.after(0, lambda: self.sp500_label.config(
                    text="S&P 500: Error",
                    fg=self.theme_manager.get_theme()['text_secondary']
                ))
        
        # Run in background thread to avoid blocking UI
        thread = threading.Thread(target=update_in_background, daemon=True)
        thread.start()
    
    def _close_app(self):
        """Close the application"""
        self.root.quit()
        self.root.destroy()
    
    def _show_error(self, message: str):
        """Show error message"""
        theme = self.theme_manager.get_theme()
        messagebox.showerror(self.localization.t('error'), message)
    
    def _safe_update_text(self, dialog, text_widget, message, scroll_to_end=True):
        """Safely update text widget, catching errors if dialog/widget is destroyed"""
        try:
            if dialog.winfo_exists() and text_widget.winfo_exists():
                text_widget.insert(tk.END, message + "\n")
                if scroll_to_end:
                    text_widget.see(tk.END)
        except tk.TclError:
            pass  # Widget was destroyed, ignore
    
    def _safe_dialog_update(self, dialog, update_func):
        """Safely execute an update function in a dialog, catching errors if dialog is destroyed"""
        try:
            if dialog.winfo_exists():
                update_func()
        except tk.TclError:
            pass  # Dialog/widget was destroyed, ignore
    
    def _update_ml_status(self):
        """Update ML training status with detailed stats"""
        theme = self.theme_manager.get_theme()
        strategy = self.current_strategy
        
        # Check if ML model exists
        from secure_ml_storage import SecureMLStorage
        storage = SecureMLStorage(self.data_dir)
        ml_model_file = f"ml_model_{strategy}.pkl"
        ml_trained = storage.model_exists(ml_model_file)
        
        hybrid_predictor = self.hybrid_predictors.get(strategy)
        
        if not hybrid_predictor:
            self.ml_status_label.config(
                text="AI System: Initializing...",
                fg=theme['text_secondary']
            )
            return
        
        # Get comprehensive performance stats
        rule_perf = hybrid_predictor._get_recent_performance('weight_based')
        ml_perf = hybrid_predictor._get_recent_performance('ml_based')
        hybrid_perf = hybrid_predictor._get_recent_performance('hybrid')
        
        # Get total verified predictions
        verified = self.predictions_tracker.get_verified_predictions()
        strategy_verified = [p for p in verified if p.get('strategy') == strategy]
        total_verified = len(strategy_verified)
        
        # Update status label
        if ml_trained:
            self.ml_status_label.config(
                text=f"🤖 AI Model: ✓ Trained\nStrategy: {strategy.title()}",
                fg=theme['success']
            )
        else:
            self.ml_status_label.config(
                text=f"🤖 AI Model: Not Trained\nStrategy: {strategy.title()}",
                fg=theme['text_secondary']
            )
        
        # Update accuracy stats
        if hybrid_perf and hybrid_perf.get('total', 0) > 0:
            hybrid_acc = hybrid_perf['accuracy']
            hybrid_correct = hybrid_perf['correct']
            hybrid_total = hybrid_perf['total']
            
            # Check for improvement
            prev_acc = self.previous_accuracy.get(strategy, hybrid_acc)
            if hybrid_acc > prev_acc + 2:  # Improved by 2% or more
                self._show_improvement_notification(strategy, prev_acc, hybrid_acc)
            self.previous_accuracy[strategy] = hybrid_acc
            
            # Update accuracy label
            color = theme['success'] if hybrid_acc >= 70 else theme['warning'] if hybrid_acc >= 60 else theme['error']
            self.ml_accuracy_label.config(
                text=f"Hybrid Accuracy: {hybrid_acc:.1f}% ({hybrid_correct}/{hybrid_total})",
                fg=color
            )
            
            # Update rule-based stats
            if rule_perf and rule_perf.get('total', 0) > 0:
                rule_acc = rule_perf['accuracy']
                self.ml_rule_label.config(
                    text=f"Rule-based: {rule_acc:.1f}% | ML: {ml_perf['accuracy']:.1f}%" if ml_perf else f"Rule-based: {rule_acc:.1f}%",
                    fg=theme['text_secondary']
                )
            else:
                self.ml_rule_label.config(text="", fg=theme['text_secondary'])
            
            # Update hybrid label
            self.ml_hybrid_label.config(
                text=f"Best Method: {'Hybrid' if hybrid_acc >= max(rule_perf.get('accuracy', 0) if rule_perf else 0, ml_perf.get('accuracy', 0) if ml_perf else 0) else 'Individual'}",
                fg=theme['accent']
            )
        else:
            self.ml_accuracy_label.config(text="No verified predictions yet", fg=theme['text_secondary'])
            self.ml_rule_label.config(text="", fg=theme['text_secondary'])
            self.ml_hybrid_label.config(text="", fg=theme['text_secondary'])
        
        # Update learning progress bar
        # Progress based on number of verified predictions (max at 200)
        progress = min(100, (total_verified / 200) * 100)
        self.learning_progress_bar['value'] = progress
        
        # Update progress label
        if total_verified < 50:
            progress_text = f"Learning Progress: {total_verified}/50 predictions"
            self.learning_progress_label.config(text=progress_text, fg=theme['text_secondary'])
        elif total_verified < 200:
            progress_text = f"Learning Progress: {total_verified}/200 predictions"
            self.learning_progress_label.config(text=progress_text, fg=theme['accent'])
        else:
            progress_text = f"Learning Progress: {total_verified}+ predictions (Advanced)"
            self.learning_progress_label.config(text=progress_text, fg=theme['success'])
        
        # Update auto-learner status
        if hasattr(self, 'auto_learner'):
            status = self.auto_learner.get_status()
            if status['is_running']:
                if status['is_scanning']:
                    status_text = "🤖 Auto-Learner: Scanning market..."
                else:
                    last_scan = status.get('last_scan_time', {})
                    last_predictions = status.get('last_scan_predictions', 0)
                    
                    if last_scan:
                        # Find most recent scan
                        recent_scans = [v for v in last_scan.values() if v]
                        if recent_scans:
                            if last_predictions > 0:
                                status_text = f"🤖 Auto-Learner: Active\nLast scan: {last_predictions} predictions"
                            else:
                                status_text = f"🤖 Auto-Learner: Active\nLast scan: {len(recent_scans)} strategies"
                        else:
                            status_text = "🤖 Auto-Learner: Active (Waiting for first scan...)"
                    else:
                        status_text = "🤖 Auto-Learner: Active (Starting in 1 hour...)"
                
                self.auto_learner_status_label.config(
                    text=status_text,
                    fg=theme['success']
                )
            else:
                self.auto_learner_status_label.config(
                    text="🤖 Auto-Learner: Stopped",
                    fg=theme['text_secondary']
                )
        
        # Update test button state
        self._update_test_button_state()
        
        # Update Megamind status
        self._update_megamind_status()
    
    def _update_megamind_status(self):
        """Update the Megamind learning status display"""
        theme = self.theme_manager.get_theme()
        stats = self.learning_tracker.get_statistics()
        
        # Create a compact status display
        status_lines = []
        
        # Data sources summary
        total_events = (stats['data_sources']['manual_analyses'] + 
                       stats['data_sources']['auto_scans'] + 
                       stats['data_sources']['background_trainings'])
        
        if total_events > 0:
            status_lines.append(f"📊 Data Sources: {total_events} events")
            status_lines.append(f"   • Manual: {stats['data_sources']['manual_analyses']}")
            status_lines.append(f"   • Auto-scans: {stats['data_sources']['auto_scans']}")
            status_lines.append(f"   • Training: {stats['data_sources']['background_trainings']}")
        
        # Knowledge summary
        if stats['knowledge']['total_training_samples'] > 0:
            status_lines.append(f"💡 Knowledge: {stats['knowledge']['total_training_samples']:,} samples")
            status_lines.append(f"   • Stocks: {stats['data_sources']['unique_stocks']}")
            status_lines.append(f"   • Predictions: {stats['knowledge']['total_predictions_made']:,}")
            status_lines.append(f"   • Verified: {stats['knowledge']['total_predictions_verified']:,}")
        
        # Learning rate
        if stats['learning_rate'] > 0:
            status_lines.append(f"⚡ Learning Rate: {stats['learning_rate']:.1f}/day")
        
        if status_lines:
            status_text = "\n".join(status_lines)
        else:
            status_text = "🧠 Ready to learn...\nFeed me data!"
        
        self.megamind_status_label.config(text=status_text, fg=theme['text_secondary'])
    
    def _update_test_button_state(self):
        """Update test button enabled state"""
        theme = self.theme_manager.get_theme()
        strategy = self.current_strategy
        
        from secure_ml_storage import SecureMLStorage
        storage = SecureMLStorage(self.data_dir)
        ml_model_file = f"ml_model_{strategy}.pkl"
        ml_trained = storage.model_exists(ml_model_file)
        
        if ml_trained:
            self.test_model_btn.config(state='normal')
        else:
            self.test_model_btn.config(state='disabled')
    
    def _show_improvement_notification(self, strategy: str, old_acc: float, new_acc: float):
        """Show notification when accuracy improves"""
        improvement = new_acc - old_acc
        message = f"🎉 AI Improvement Detected!\n\n"
        message += f"Strategy: {strategy.title()}\n"
        message += f"Accuracy improved: {old_acc:.1f}% → {new_acc:.1f}%\n"
        message += f"Improvement: +{improvement:.1f}%\n\n"
        message += f"The AI is learning from your predictions!"
        
        messagebox.showinfo("AI Learning Progress", message)
    
    def _show_detailed_stats(self):
        """Show detailed performance statistics dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("AI Performance Statistics")
        dialog.geometry("700x600")
        dialog.transient(self.root)
        
        theme = self.theme_manager.get_theme()
        dialog.config(bg=theme['bg'])
        
        # Header
        header = tk.Label(
            dialog,
            text="📊 AI Performance Statistics",
            bg=theme['bg'],
            fg=theme['accent'],
            font=('Segoe UI', 16, 'bold')
        )
        header.pack(pady=20)
        
        # Strategy selection
        strategy_frame = tk.Frame(dialog, bg=theme['bg'])
        strategy_frame.pack(pady=10)
        
        tk.Label(
            strategy_frame,
            text="Strategy:",
            bg=theme['bg'],
            fg=theme['fg'],
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT, padx=10)
        
        strategy_var = tk.StringVar(value=self.current_strategy)
        strategy_combo = ttk.Combobox(
            strategy_frame,
            textvariable=strategy_var,
            values=["trading", "mixed", "investing"],
            state='readonly',
            width=15
        )
        strategy_combo.pack(side=tk.LEFT, padx=10)
        
        # Stats display area
        stats_frame = tk.Frame(dialog, bg=theme['bg'])
        stats_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(stats_frame)
        notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Stats tab
        stats_tab = tk.Frame(notebook, bg=theme['bg'])
        notebook.add(stats_tab, text="Statistics")
        
        stats_text = scrolledtext.ScrolledText(
            stats_tab,
            height=15,
            bg=theme['frame_bg'],
            fg=theme['fg'],
            font=('Consolas', 9),
            wrap=tk.WORD
        )
        stats_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Performance graph tab
        graph_tab = tk.Frame(notebook, bg=theme['bg'])
        notebook.add(graph_tab, text="Performance Trend")
        
        graph_canvas_frame = tk.Frame(graph_tab, bg=theme['bg'])
        graph_canvas_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        def update_stats():
            strategy = strategy_var.get()
            hybrid_predictor = self.hybrid_predictors.get(strategy)
            
            if not hybrid_predictor:
                stats_text.delete(1.0, tk.END)
                stats_text.insert(tk.END, "No data available for this strategy.")
                return
            
            # Get all performance data
            rule_perf = hybrid_predictor._get_recent_performance('weight_based')
            ml_perf = hybrid_predictor._get_recent_performance('ml_based')
            hybrid_perf = hybrid_predictor._get_recent_performance('hybrid')
            
            # Get verified predictions
            verified = self.predictions_tracker.get_verified_predictions()
            strategy_verified = [p for p in verified if p.get('strategy') == strategy]
            
            # Get weight info
            from adaptive_weight_adjuster import AdaptiveWeightAdjuster
            weight_adjuster = AdaptiveWeightAdjuster(self.data_dir, strategy)
            weights = weight_adjuster.get_weights()
            
            # Build stats text
            stats = f"{'='*60}\n"
            stats += f"AI PERFORMANCE STATISTICS - {strategy.upper()}\n"
            stats += f"{'='*60}\n\n"
            
            stats += f"VERIFIED PREDICTIONS:\n"
            stats += f"  Total Verified: {len(strategy_verified)}\n"
            stats += f"  Active Predictions: {len([p for p in self.predictions_tracker.get_active_predictions() if p.get('strategy') == strategy])}\n\n"
            
            stats += f"ACCURACY METRICS (Last 50 Predictions):\n"
            if hybrid_perf and hybrid_perf.get('total', 0) > 0:
                stats += f"  🤖 Hybrid System: {hybrid_perf['accuracy']:.1f}% ({hybrid_perf['correct']}/{hybrid_perf['total']})\n"
            else:
                stats += f"  🤖 Hybrid System: No data yet\n"
            
            if rule_perf and rule_perf.get('total', 0) > 0:
                stats += f"  📊 Rule-based: {rule_perf['accuracy']:.1f}% ({rule_perf['correct']}/{rule_perf['total']})\n"
            else:
                stats += f"  📊 Rule-based: No data yet\n"
            
            if ml_perf and ml_perf.get('total', 0) > 0:
                stats += f"  🧠 ML Model: {ml_perf['accuracy']:.1f}% ({ml_perf['correct']}/{ml_perf['total']})\n"
            else:
                stats += f"  🧠 ML Model: Not trained or no data\n"
            
            stats += f"\n{'='*60}\n"
            stats += f"LEARNED WEIGHTS (Rule-based):\n"
            stats += f"{'='*60}\n"
            
            # Show top weights
            sorted_weights = sorted(weights.items(), key=lambda x: abs(x[1]) if isinstance(x[1], (int, float)) else 0, reverse=True)
            for key, value in sorted_weights[:10]:  # Top 10
                if isinstance(value, (int, float)):
                    stats += f"  {key}: {value:.3f}\n"
            
            stats += f"\n{'='*60}\n"
            stats += f"ENSEMBLE WEIGHTS:\n"
            stats += f"{'='*60}\n"
            ensemble_weights = hybrid_predictor.ensemble_weights
            stats += f"  Rule-based: {ensemble_weights.get('rule', 0.5)*100:.0f}%\n"
            stats += f"  ML-based: {ensemble_weights.get('ml', 0.5)*100:.0f}%\n"
            
            # Check ML model status
            from secure_ml_storage import SecureMLStorage
            storage = SecureMLStorage(self.data_dir)
            ml_model_file = f"ml_model_{strategy}.pkl"
            if storage.model_exists(ml_model_file):
                stats += f"\n  ML Model: ✓ Trained\n"
                model_info = storage.get_model_info(ml_model_file)
                if model_info:
                    stats += f"  Model Size: {model_info.get('size', 0) / 1024:.1f} KB\n"
            else:
                stats += f"\n  ML Model: Not Trained\n"
            
            stats += f"\n{'='*60}\n"
            stats += f"LEARNING PROGRESS:\n"
            stats += f"{'='*60}\n"
            
            if len(strategy_verified) < 50:
                stats += f"  Status: Early Learning ({len(strategy_verified)}/50 predictions)\n"
                stats += f"  Progress: {len(strategy_verified)/50*100:.0f}%\n"
            elif len(strategy_verified) < 200:
                stats += f"  Status: Active Learning ({len(strategy_verified)}/200 predictions)\n"
                stats += f"  Progress: {len(strategy_verified)/200*100:.0f}%\n"
            else:
                stats += f"  Status: Advanced Learning ({len(strategy_verified)}+ predictions)\n"
                stats += f"  Progress: 100%+\n"
            
            stats_text.delete(1.0, tk.END)
            stats_text.insert(tk.END, stats)
            
            # Update performance graph
            _update_performance_graph(strategy)
        
        def _update_performance_graph(strategy):
            """Update performance trend graph"""
            # Clear previous graph
            for widget in graph_canvas_frame.winfo_children():
                widget.destroy()
            
            hybrid_predictor = self.hybrid_predictors.get(strategy)
            if not hybrid_predictor:
                return
            
            # Get performance history
            rule_history = hybrid_predictor.performance_history.get('weight_based', [])
            ml_history = hybrid_predictor.performance_history.get('ml_based', [])
            hybrid_history = hybrid_predictor.performance_history.get('hybrid', [])
            
            if not hybrid_history:
                no_data_label = tk.Label(
                    graph_canvas_frame,
                    text="No performance data yet.\nMake predictions and verify them to see trends!",
                    bg=theme['bg'],
                    fg=theme['text_secondary'],
                    font=('Segoe UI', 11)
                )
                no_data_label.pack(expand=True)
                return
            
            # Calculate rolling accuracy (last 20 predictions)
            window_size = 20
            hybrid_accuracies = []
            rule_accuracies = []
            ml_accuracies = []
            
            for i in range(len(hybrid_history)):
                start_idx = max(0, i - window_size + 1)
                window = hybrid_history[start_idx:i+1]
                if window:
                    correct = sum(1 for h in window if h.get('was_correct') is True)
                    accuracy = (correct / len(window)) * 100
                    hybrid_accuracies.append(accuracy)
            
            for i in range(len(rule_history)):
                start_idx = max(0, i - window_size + 1)
                window = rule_history[start_idx:i+1]
                if window:
                    correct = sum(1 for h in window if h.get('was_correct') is True)
                    accuracy = (correct / len(window)) * 100
                    rule_accuracies.append(accuracy)
            
            for i in range(len(ml_history)):
                start_idx = max(0, i - window_size + 1)
                window = ml_history[start_idx:i+1]
                if window:
                    correct = sum(1 for h in window if h.get('was_correct') is True)
                    accuracy = (correct / len(window)) * 100
                    ml_accuracies.append(accuracy)
            
            # Create graph
            fig = Figure(figsize=(8, 4), dpi=100)
            ax = fig.add_subplot(111)
            
            if hybrid_accuracies:
                ax.plot(hybrid_accuracies, label='Hybrid', linewidth=2, color='#2196F3')
            if rule_accuracies:
                ax.plot(rule_accuracies, label='Rule-based', linewidth=1.5, color='#4CAF50', alpha=0.7)
            if ml_accuracies:
                ax.plot(ml_accuracies, label='ML Model', linewidth=1.5, color='#FF9800', alpha=0.7)
            
            ax.set_xlabel('Prediction Number', fontsize=10)
            ax.set_ylabel('Accuracy (%)', fontsize=10)
            ax.set_title(f'Performance Trend - {strategy.title()} Strategy', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 100])
            
            # Apply theme
            if theme.get('bg') == '#000000' or theme.get('bg') == '#121212':
                fig.patch.set_facecolor('#121212')
                ax.set_facecolor('#1E1E1E')
                ax.tick_params(colors='white')
                ax.xaxis.label.set_color('white')
                ax.yaxis.label.set_color('white')
                ax.title.set_color('white')
            
            canvas = FigureCanvasTkAgg(fig, graph_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Update stats when strategy changes
        strategy_combo.bind('<<ComboboxSelected>>', lambda e: update_stats())
        
        # Initial update
        update_stats()
        
        # Close button
        close_btn = ModernButton(
            dialog,
            "Close",
            command=dialog.destroy,
            theme=theme
        )
        close_btn.pack(pady=10)
    
    def _show_training_dialog(self):
        """Show ML training dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Train AI Model")
        dialog.geometry("600x500")
        dialog.transient(self.root)
        # Don't use grab_set() - allows user to interact with app while training
        # dialog.grab_set()  # Removed to allow non-blocking UI
        
        theme = self.theme_manager.get_theme()
        dialog.config(bg=theme['bg'])
        
        # Header
        header = tk.Label(
            dialog,
            text="🤖 Train Machine Learning Model",
            bg=theme['bg'],
            fg=theme['accent'],
            font=('Segoe UI', 16, 'bold')
        )
        header.pack(pady=20)
        
        # Strategy selection
        strategy_frame = tk.Frame(dialog, bg=theme['bg'])
        strategy_frame.pack(pady=10)
        
        tk.Label(
            strategy_frame,
            text="Strategy:",
            bg=theme['bg'],
            fg=theme['fg'],
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT, padx=10)
        
        strategy_var = tk.StringVar(value=self.current_strategy)
        strategy_combo = ttk.Combobox(
            strategy_frame,
            textvariable=strategy_var,
            values=["trading", "mixed", "investing"],
            state='readonly',
            width=15
        )
        strategy_combo.pack(side=tk.LEFT, padx=10)
        
        # Stock symbols input
        symbols_frame = tk.Frame(dialog, bg=theme['bg'])
        symbols_frame.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(
            symbols_frame,
            text="Stock Symbols (comma-separated):",
            bg=theme['bg'],
            fg=theme['fg'],
            font=('Segoe UI', 10)
        ).pack(anchor=tk.W)
        
        symbols_entry = tk.Entry(
            symbols_frame,
            font=('Segoe UI', 10),
            bg=theme['entry_bg'],
            fg=theme['entry_fg'],
            width=50
        )
        symbols_entry.pack(fill=tk.X, pady=5)
        # Load last used symbols for this strategy
        last_symbols = self.preferences.get_training_symbols(strategy_var.get())
        symbols_entry.insert(0, last_symbols)
        
        # Preset symbols button
        preset_frame = tk.Frame(symbols_frame, bg=theme['bg'])
        preset_frame.pack(fill=tk.X, pady=(0, 5))
        
        def load_preset_symbols():
            """Load diverse preset symbols for better training"""
            # Diverse portfolio: Tech, Finance, Healthcare, Consumer, Industrial, Energy
            preset = "AAPL,MSFT,GOOGL,NVDA,AMD,JPM,BAC,GS,JNJ,PFE,UNH,AMZN,WMT,HD,BA,CAT,XOM,CVX,TSLA,META"
            symbols_entry.delete(0, tk.END)
            symbols_entry.insert(0, preset)
            status_text.insert(tk.END, "✅ Loaded 20 diverse symbols (Tech, Finance, Healthcare, Consumer, Industrial, Energy)\n")
            status_text.see(tk.END)
        
        preset_btn = tk.Button(
            preset_frame,
            text="📊 Load Diverse Portfolio (20 symbols)",
            command=load_preset_symbols,
            bg=theme['button_bg'],
            fg=theme['button_fg'],
            font=('Segoe UI', 9),
            relief=tk.FLAT,
            cursor='hand2'
        )
        preset_btn.pack(side=tk.LEFT)
        
        # Info label
        info_preset = tk.Label(
            preset_frame,
            text="💡 Using diverse symbols improves model generalization",
            bg=theme['bg'],
            fg=theme['text_secondary'],
            font=('Segoe UI', 8),
            justify=tk.LEFT
        )
        info_preset.pack(side=tk.LEFT, padx=(10, 0))
        
        # Date range
        date_frame = tk.Frame(dialog, bg=theme['bg'])
        date_frame.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(
            date_frame,
            text="Training Period:",
            bg=theme['bg'],
            fg=theme['fg'],
            font=('Segoe UI', 10)
        ).pack(anchor=tk.W)
        
        date_inner = tk.Frame(date_frame, bg=theme['bg'])
        date_inner.pack(fill=tk.X, pady=5)
        
        tk.Label(date_inner, text="Start:", bg=theme['bg'], fg=theme['fg']).pack(side=tk.LEFT, padx=5)
        start_entry = tk.Entry(date_inner, width=12, bg=theme['entry_bg'], fg=theme['entry_fg'])
        start_entry.pack(side=tk.LEFT, padx=5)
        start_entry.insert(0, "2010-01-01")  # 14 years for better generalization (includes 2008 recovery)
        
        tk.Label(date_inner, text="End:", bg=theme['bg'], fg=theme['fg']).pack(side=tk.LEFT, padx=5)
        end_entry = tk.Entry(date_inner, width=12, bg=theme['entry_bg'], fg=theme['entry_fg'])
        end_entry.pack(side=tk.LEFT, padx=5)
        end_entry.insert(0, "2023-12-31")  # End before current year for out-of-sample testing
        
        # Add helpful info label
        info_label = tk.Label(
            date_frame,
            text="💡 Recommended: 2010-2023 (14 years) includes bull, bear, and recovery markets.\n"
                 "Use 2024-2025 as out-of-sample testing period. Longer ranges = better patterns.",
            bg=theme['bg'],
            fg=theme['text_secondary'],
            font=('Segoe UI', 8),
            justify=tk.LEFT
        )
        info_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Retrain from verified predictions option
        retrain_frame = tk.Frame(dialog, bg=theme['bg'])
        retrain_frame.pack(pady=10, padx=20, fill=tk.X)
        
        retrain_var = tk.BooleanVar(value=False)
        retrain_check = tk.Checkbutton(
            retrain_frame,
            text="🔄 Also retrain ML model from verified predictions (if available)",
            variable=retrain_var,
            bg=theme['bg'],
            fg=theme['fg'],
            font=('Segoe UI', 9),
            selectcolor=theme['frame_bg'],
            activebackground=theme['bg'],
            activeforeground=theme['fg']
        )
        retrain_check.pack(anchor=tk.W)
        
        retrain_info = tk.Label(
            retrain_frame,
            text="💡 This will combine historical training with your verified predictions for better accuracy",
            bg=theme['bg'],
            fg=theme['text_secondary'],
            font=('Segoe UI', 8),
            justify=tk.LEFT
        )
        retrain_info.pack(anchor=tk.W, pady=(5, 0))
        
        # Status text
        status_text = scrolledtext.ScrolledText(
            dialog,
            height=10,
            bg=theme['frame_bg'],
            fg=theme['fg'],
            font=('Consolas', 9)
        )
        status_text.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Configuration save/load section
        config_frame = tk.Frame(dialog, bg=theme['bg'])
        config_frame.pack(pady=10, padx=20, fill=tk.X)
        
        config_label = tk.Label(
            config_frame,
            text="💾 Configuration Save Slots:",
            bg=theme['bg'],
            fg=theme['fg'],
            font=('Segoe UI', 10, 'bold')
        )
        config_label.pack(anchor=tk.W, pady=(0, 5))
        
        config_btn_frame = tk.Frame(config_frame, bg=theme['bg'])
        config_btn_frame.pack(fill=tk.X)
        
        from ml_config_manager import MLConfigManager
        config_manager = MLConfigManager(self.data_dir)
        
        def save_current_config():
            """Save current ML training configuration"""
            name_dialog = tk.Toplevel(dialog)
            name_dialog.title("Save Configuration")
            name_dialog.geometry("400x200")
            name_dialog.transient(dialog)
            name_dialog.config(bg=theme['bg'])
            
            tk.Label(
                name_dialog,
                text="Configuration Name:",
                bg=theme['bg'],
                fg=theme['fg'],
                font=('Segoe UI', 10)
            ).pack(pady=10)
            
            name_entry = tk.Entry(
                name_dialog,
                font=('Segoe UI', 10),
                bg=theme['entry_bg'],
                fg=theme['entry_fg'],
                width=30
            )
            name_entry.pack(pady=5, padx=20, fill=tk.X)
            name_entry.insert(0, f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            tk.Label(
                name_dialog,
                text="Description (optional):",
                bg=theme['bg'],
                fg=theme['fg'],
                font=('Segoe UI', 10)
            ).pack(pady=(10, 5))
            
            desc_entry = tk.Entry(
                name_dialog,
                font=('Segoe UI', 10),
                bg=theme['entry_bg'],
                fg=theme['entry_fg'],
                width=30
            )
            desc_entry.pack(pady=5, padx=20, fill=tk.X)
            desc_entry.insert(0, "Current good configuration")
            
            def do_save():
                name = name_entry.get().strip()
                if not name:
                    messagebox.showerror("Error", "Configuration name cannot be empty!")
                    return
                
                # Validate name (no special characters for filename safety)
                import re
                if not re.match(r'^[a-zA-Z0-9_-]+$', name):
                    messagebox.showerror("Error", "Configuration name can only contain letters, numbers, underscores, and hyphens!")
                    return
                
                description = desc_entry.get().strip()
                if config_manager.save_config(name, description):
                    messagebox.showinfo(
                        "Success",
                        f"Configuration '{name}' saved successfully!\n\n"
                        f"Description: {description or '(none)'}\n\n"
                        "You can load this configuration anytime from the 'Load Config' button."
                    )
                    name_dialog.destroy()
                else:
                    messagebox.showerror("Error", f"Failed to save configuration '{name}'")
            
            save_btn = ModernButton(
                name_dialog,
                "💾 Save",
                command=do_save,
                theme=theme
            )
            save_btn.pack(pady=10)
        
        def load_config():
            """Load a saved ML training configuration"""
            configs = config_manager.list_configs()
            
            if not configs:
                messagebox.showinfo("No Configurations", "No saved configurations found.\n\nSave your current configuration first!")
                return
            
            load_dialog = tk.Toplevel(dialog)
            load_dialog.title("Load Configuration")
            load_dialog.geometry("500x400")
            load_dialog.transient(dialog)
            load_dialog.config(bg=theme['bg'])
            
            tk.Label(
                load_dialog,
                text="Select Configuration to Load:",
                bg=theme['bg'],
                fg=theme['fg'],
                font=('Segoe UI', 12, 'bold')
            ).pack(pady=10)
            
            # Listbox with scrollbar
            list_frame = tk.Frame(load_dialog, bg=theme['bg'])
            list_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
            
            scrollbar = ModernScrollbar(list_frame)
            scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
            
            config_listbox = tk.Listbox(
                list_frame,
                font=('Segoe UI', 10),
                bg=theme['frame_bg'],
                fg=theme['fg'],
                selectbackground=theme['accent'],
                yscrollcommand=scrollbar.set,
                height=10
            )
            config_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            scrollbar.config(command=config_listbox.yview)
            
            # Populate list
            for config in configs:
                name = config.get('name', 'Unknown')
                desc = config.get('description', '')
                saved = config.get('saved_at', '')
                display = f"{name}"
                if desc:
                    display += f" - {desc}"
                if saved:
                    try:
                        saved_dt = datetime.fromisoformat(saved)
                        display += f" ({saved_dt.strftime('%Y-%m-%d %H:%M')})"
                    except:
                        pass
                config_listbox.insert(tk.END, display)
            
            if configs:
                config_listbox.selection_set(0)
            
            selected_config = {'name': None}
            
            def do_load():
                selection = config_listbox.curselection()
                if not selection:
                    messagebox.showwarning("No Selection", "Please select a configuration to load.")
                    return
                
                selected_name = configs[selection[0]]['name']
                config = config_manager.load_config(selected_name)
                
                if config:
                    # Apply config to training (for next training run)
                    # Store in a way that training can use it
                    self._saved_training_config = config_manager.apply_config_to_training(config)
                    messagebox.showinfo(
                        "Configuration Loaded",
                        f"Configuration '{selected_name}' loaded!\n\n"
                        f"Description: {config.get('description', 'N/A')}\n\n"
                        "This configuration will be used for the next training run."
                    )
                    load_dialog.destroy()
                else:
                    messagebox.showerror("Error", f"Failed to load configuration '{selected_name}'")
            
            def do_delete():
                selection = config_listbox.curselection()
                if not selection:
                    messagebox.showwarning("No Selection", "Please select a configuration to delete.")
                    return
                
                selected_name = configs[selection[0]]['name']
                response = messagebox.askyesno(
                    "Delete Configuration?",
                    f"Are you sure you want to delete '{selected_name}'?\n\nThis cannot be undone."
                )
                
                if response:
                    if config_manager.delete_config(selected_name):
                        messagebox.showinfo("Deleted", f"Configuration '{selected_name}' deleted.")
                        load_dialog.destroy()
                        load_config()  # Refresh list
                    else:
                        messagebox.showerror("Error", f"Failed to delete configuration '{selected_name}'")
            
            btn_frame_load = tk.Frame(load_dialog, bg=theme['bg'])
            btn_frame_load.pack(pady=10)
            
            load_btn = ModernButton(
                btn_frame_load,
                "📂 Load",
                command=do_load,
                theme=theme
            )
            load_btn.pack(side=tk.LEFT, padx=5)
            
            delete_btn = ModernButton(
                btn_frame_load,
                "🗑️ Delete",
                command=do_delete,
                theme=theme
            )
            delete_btn.pack(side=tk.LEFT, padx=5)
        
        save_config_btn = ModernButton(
            config_btn_frame,
            "💾 Save Current Config",
            command=save_current_config,
            theme=theme
        )
        save_config_btn.pack(side=tk.LEFT, padx=5)
        
        load_config_btn = ModernButton(
            config_btn_frame,
            "📂 Load Config",
            command=load_config,
            theme=theme
        )
        load_config_btn.pack(side=tk.LEFT, padx=5)
        
        # Factory Reset Button
        def do_factory_reset():
            if messagebox.askyesno(
                "⚠️ Factory Reset ML System?", 
                "This will completely wipe all ML models, learning data, and statistics.\n\n"
                "Your accuracy will reset to 0% and the system will relearn from scratch.\n\n"
                "Are you sure you want to do this?",
                icon='warning'
            ):
                try:
                    # 1. Clear in-memory data structures to prevent save-on-exit
                    logger.info("Clearing in-memory data structures...")
                    if hasattr(self, 'hybrid_predictors'):
                        for strategy, predictor in self.hybrid_predictors.items():
                            predictor.performance_history = {'weight_based': [], 'ml_based': [], 'hybrid': []}
                            predictor.ensemble_weights = {'rule': 1.0, 'ml': 0.0, 'ai': 0.0}
                    
                    if hasattr(self, 'learning_tracker'):
                        self.learning_tracker.data = {}
                        
                    if hasattr(self, 'predictions_tracker'):
                        self.predictions_tracker.verified_predictions = []
                        self.predictions_tracker.predictions = []
                        
                    if hasattr(self, 'model_benchmarker'):
                        self.model_benchmarker.benchmark_data = []

                    # 2. Run reset scripts
                    import force_clear_stats
                    force_clear_stats.clear_performance_stats()
                    
                    import complete_reset
                    # Redirect stdout to capture output
                    import io
                    import sys
                    from contextlib import redirect_stdout
                    
                    f = io.StringIO()
                    with redirect_stdout(f):
                        complete_reset.main()
                    
                    messagebox.showinfo(
                        "Reset Complete", 
                        "System reset successfully!\n\n"
                        "The application will now close immediately.\n"
                        "Please restart it to begin fresh training."
                    )
                    
                    # 3. Force exit immediately to skip any cleanup/save handlers
                    import os
                    os._exit(0)
                    
                except Exception as e:
                    messagebox.showerror("Reset Failed", f"Error during reset: {e}")

        reset_btn = ModernButton(
            config_btn_frame,
            "⚠️ Factory Reset",
            command=do_factory_reset,
            theme=theme
        )
        reset_btn.pack(side=tk.RIGHT, padx=20)
        
        # Buttons
        btn_frame = tk.Frame(dialog, bg=theme['bg'])
        btn_frame.pack(pady=10)
        
        def start_training():
            symbols = [s.strip().upper() for s in symbols_entry.get().split(',')]
            start_date = start_entry.get()
            end_date = end_entry.get()
            strategy = strategy_var.get()
            retrain_from_verified = retrain_var.get()
            
            # Save symbols for next time
            symbols_str = ','.join(symbols)
            self.preferences.set_training_symbols(strategy, symbols_str)
            
            status_text.delete(1.0, tk.END)
            status_text.delete(1.0, tk.END)
            status_text.insert(tk.END, f"Starting training for {strategy} strategy...\n")
            status_text.insert(tk.END, f"Symbols: {', '.join(symbols)}\n")
            status_text.insert(tk.END, f"Period: {start_date} to {end_date}\n")
            if retrain_from_verified:
                status_text.insert(tk.END, f"🔄 Will also retrain from verified predictions\n")
            status_text.insert(tk.END, "\n")
            status_text.insert(tk.END, "💡 You can minimize this window and continue using the app!\n")
            status_text.insert(tk.END, "Training runs in the background and won't block the UI.\n\n")
            status_text.update()
            
            # Helper function to safely update status text (handles destroyed widgets)
            def safe_insert(text):
                """Safely insert text into status_text, ignoring errors if widget is destroyed"""
                try:
                    if dialog.winfo_exists():
                        status_text.insert(tk.END, text + "\n")
                        status_text.see(tk.END)
                except (tk.TclError, AttributeError):
                    pass  # Widget was destroyed, ignore
            
            # Run training in background
            def train():
                try:
                    # First, train on historical data
                    result = self.training_manager.train_on_symbols(
                        symbols, start_date, end_date, strategy,
                        progress_callback=lambda msg: dialog.after(0, lambda: safe_insert(msg))
                    )
                    
                    if 'error' in result:
                        dialog.after(0, lambda: safe_insert(f"\n✗ Error: {result['error']}"))
                    else:
                        dialog.after(0, lambda: safe_insert(
                            f"\n✓ Historical Training Complete!\n"
                            f"Test Accuracy: {result.get('test_accuracy', 0)*100:.1f}%\n"
                            f"Training Samples: {result.get('total_samples', 0)}\n"
                            f"Cross-Validation: {result.get('cv_mean', 0)*100:.1f}% ± {result.get('cv_std', 0)*100:.1f}%"
                        ))
                        
                        # If requested, also retrain from verified predictions
                        if retrain_from_verified:
                            dialog.after(0, lambda: safe_insert("\n🔄 Retraining from verified predictions..."))
                            
                            try:
                                from training_pipeline import MLTrainingPipeline
                                pipeline = MLTrainingPipeline(self.data_fetcher, self.data_dir, app=self)
                                verified_result = pipeline.train_on_verified_predictions(
                                    self.predictions_tracker,
                                    strategy,
                                    retrain_ml=True
                                )
                                
                                if 'error' in verified_result:
                                    dialog.after(0, lambda: safe_insert(f"⚠️ Verified predictions: {verified_result['error']}"))
                                else:
                                    verified_msg = f"\n✅ Verified Predictions Training Complete!\n"
                                    verified_msg += f"Verified Predictions Used: {verified_result.get('verified_predictions_used', 0)}\n"
                                    if verified_result.get('ml_retrained'):
                                        verified_msg += f"ML Retrained: ✅\n"
                                        verified_msg += f"ML Train Accuracy: {verified_result.get('ml_train_accuracy', 0):.1f}%\n"
                                        verified_msg += f"ML Test Accuracy: {verified_result.get('ml_test_accuracy', 0):.1f}%\n"
                                        verified_msg += f"Training Samples: {verified_result.get('training_samples_used', 0)}\n"
                                    else:
                                        verified_msg += f"ML Retrained: ❌ ({verified_result.get('ml_error', 'N/A')})\n"
                                    dialog.after(0, lambda: safe_insert(verified_msg))
                            except Exception as e:
                                error_msg = str(e)
                                dialog.after(0, lambda msg=error_msg: safe_insert(f"⚠️ Error retraining from verified: {msg}"))
                        
                        dialog.after(0, self._update_ml_status)
                except Exception as e:
                    error_msg = str(e)  # Capture error message in local variable
                    dialog.after(0, lambda msg=error_msg: safe_insert(f"\n✗ Error: {msg}"))
            
            thread = threading.Thread(target=train)
            thread.daemon = True
            thread.start()
        
        train_btn = ModernButton(
            btn_frame,
            "Start Training",
            command=start_training,
            theme=theme
        )
        train_btn.pack(side=tk.LEFT, padx=5)
        
        close_btn = ModernButton(
            btn_frame,
            "Close",
            command=dialog.destroy,
            theme=theme
        )
        close_btn.pack(side=tk.LEFT, padx=5)
    
    def _show_auto_learner_settings(self):
        """Show auto-learner settings dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Auto-Learner Settings")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        theme = self.theme_manager.get_theme()
        dialog.config(bg=theme['bg'])
        
        # Header
        header = tk.Label(
            dialog,
            text="⚙️ Auto-Learner Settings",
            bg=theme['bg'],
            fg=theme['accent'],
            font=('Segoe UI', 16, 'bold')
        )
        header.pack(pady=20)
        
        # Explanation
        explanation = tk.Label(
            dialog,
            text="Configure how often the app automatically scans stocks and learns.",
            bg=theme['bg'],
            fg=theme['text_secondary'],
            font=('Segoe UI', 9),
            wraplength=450,
            justify=tk.LEFT
        )
        explanation.pack(pady=(0, 20), padx=20)
        
        # Scan interval setting
        interval_frame = tk.Frame(dialog, bg=theme['bg'])
        interval_frame.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(
            interval_frame,
            text="Scan Interval:",
            bg=theme['bg'],
            fg=theme['fg'],
            font=('Segoe UI', 10, 'bold')
        ).pack(anchor=tk.W)
        
        info_text = (
            "How often to automatically scan stocks.\n\n"
            "• 1-2 hours: Very frequent (faster learning, more API calls)\n"
            "• 3-6 hours: Balanced (recommended, good learning speed)\n"
            "• 12-24 hours: Conservative (slower learning, fewer API calls)\n\n"
            "Note: You can always use 'Scan Now' for immediate scans."
        )
        
        tk.Label(
            interval_frame,
            text=info_text,
            bg=theme['bg'],
            fg=theme['text_secondary'],
            font=('Segoe UI', 8),
            justify=tk.LEFT,
            wraplength=450
        ).pack(anchor=tk.W, pady=(5, 10))
        
        interval_inner = tk.Frame(interval_frame, bg=theme['bg'])
        interval_inner.pack(fill=tk.X)
        
        tk.Label(interval_inner, text="Every:", bg=theme['bg'], fg=theme['fg']).pack(side=tk.LEFT, padx=5)
        
        interval_var = tk.IntVar()
        if hasattr(self, 'auto_learner'):
            interval_var.set(self.auto_learner.scan_interval_hours)
        else:
            interval_var.set(6)
        
        interval_spinbox = tk.Spinbox(
            interval_inner,
            from_=1,
            to=24,
            textvariable=interval_var,
            width=5,
            bg=theme['entry_bg'],
            fg=theme['entry_fg'],
            font=('Segoe UI', 10)
        )
        interval_spinbox.pack(side=tk.LEFT, padx=5)
        
        tk.Label(interval_inner, text="hours", bg=theme['bg'], fg=theme['fg']).pack(side=tk.LEFT, padx=5)
        
        # Predictions per scan
        predictions_frame = tk.Frame(dialog, bg=theme['bg'])
        predictions_frame.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(
            predictions_frame,
            text="Predictions Per Scan:",
            bg=theme['bg'],
            fg=theme['fg'],
            font=('Segoe UI', 10, 'bold')
        ).pack(anchor=tk.W)
        
        tk.Label(
            predictions_frame,
            text="How many predictions to make per strategy during each scan.",
            bg=theme['bg'],
            fg=theme['text_secondary'],
            font=('Segoe UI', 8),
            justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(5, 10))
        
        predictions_inner = tk.Frame(predictions_frame, bg=theme['bg'])
        predictions_inner.pack(fill=tk.X)
        
        predictions_var = tk.IntVar()
        if hasattr(self, 'auto_learner'):
            predictions_var.set(self.auto_learner.predictions_per_scan)
        else:
            predictions_var.set(5)
        
        predictions_spinbox = tk.Spinbox(
            predictions_inner,
            from_=1,
            to=20,
            textvariable=predictions_var,
            width=5,
            bg=theme['entry_bg'],
            fg=theme['entry_fg'],
            font=('Segoe UI', 10)
        )
        predictions_spinbox.pack(side=tk.LEFT, padx=5)
        
        tk.Label(predictions_inner, text="per strategy", bg=theme['bg'], fg=theme['fg']).pack(side=tk.LEFT, padx=5)
        
        # Save button
        def save_settings():
            if hasattr(self, 'auto_learner'):
                scan_interval = interval_var.get()
                predictions = predictions_var.get()
                
                # Update auto-learner config
                self.auto_learner.update_config(
                    scan_interval_hours=scan_interval,
                    predictions_per_scan=predictions
                )
                
                # Save to preferences
                self.preferences.set('auto_learner_scan_interval_hours', scan_interval)
                self.preferences.set('auto_learner_predictions_per_scan', predictions)
                self.preferences.set('auto_learner_min_confidence', self.auto_learner.min_confidence)
                logger.info(f"Saved auto-learner settings: interval={scan_interval}h, predictions={predictions}")
                
                messagebox.showinfo("Settings Saved", 
                    f"Auto-learner will now scan every {scan_interval} hours\n"
                    f"and make {predictions} predictions per strategy.\n\n"
                    f"Settings have been saved and will persist.")
                dialog.destroy()
                self._update_ml_status()
            else:
                messagebox.showwarning("Error", "Auto-learner not initialized")
        
        btn_frame = tk.Frame(dialog, bg=theme['bg'])
        btn_frame.pack(pady=20)
        
        save_btn = ModernButton(btn_frame, "Save Settings", command=save_settings, theme=theme)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        cancel_btn = ModernButton(btn_frame, "Cancel", command=dialog.destroy, theme=theme)
        cancel_btn.pack(side=tk.LEFT, padx=5)
    
    def _trigger_manual_scan(self):
        """Trigger immediate market scan for faster learning"""
        if not hasattr(self, 'auto_learner'):
            messagebox.showwarning("Auto-Learner", "Auto-learner not initialized")
            return
        
        # Check if already scanning
        status = self.auto_learner.get_status()
        if status['is_scanning']:
            messagebox.showinfo("Scanning", "Market scan already in progress. Please wait...")
            return
        
        # Trigger scan
        success = self.auto_learner.scan_now()
        if success:
            messagebox.showinfo(
                "Scan Started", 
                "🔍 Market scan started!\n\n"
                "The app will:\n"
                "• Scan popular stocks\n"
                "• Make predictions automatically\n"
                "• Save predictions for verification\n\n"
                "Check the status below for progress."
            )
            # Update status immediately
            self._update_ml_status()
        else:
            messagebox.showwarning("Scan Failed", "Could not start scan. Please try again.")
    
    def _show_test_dialog(self):
        """Show model testing dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Test ML Model")
        dialog.geometry("600x500")
        dialog.transient(self.root)
        dialog.grab_set()
        
        theme = self.theme_manager.get_theme()
        dialog.config(bg=theme['bg'])
        
        # Header
        header = tk.Label(
            dialog,
            text="🧪 Test ML Model Performance",
            bg=theme['bg'],
            fg=theme['accent'],
            font=('Segoe UI', 16, 'bold')
        )
        header.pack(pady=20)
        
        # Strategy selection
        strategy_frame = tk.Frame(dialog, bg=theme['bg'])
        strategy_frame.pack(pady=10)
        
        tk.Label(
            strategy_frame,
            text="Strategy:",
            bg=theme['bg'],
            fg=theme['fg'],
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT, padx=10)
        
        strategy_var = tk.StringVar(value=self.current_strategy)
        strategy_combo = ttk.Combobox(
            strategy_frame,
            textvariable=strategy_var,
            values=["trading", "mixed", "investing"],
            state='readonly',
            width=15
        )
        strategy_combo.pack(side=tk.LEFT, padx=10)
        
        # Update symbols when strategy changes (define before symbols_entry)
        def on_strategy_change_test(event=None):
            strategy = strategy_var.get()
            last_symbols = self.preferences.get_test_symbols(strategy)
            if 'symbols_entry' in locals():
                symbols_entry.delete(0, tk.END)
                symbols_entry.insert(0, last_symbols)
        
        strategy_combo.bind('<<ComboboxSelected>>', on_strategy_change_test)
        
        # Test period
        period_frame = tk.Frame(dialog, bg=theme['bg'])
        period_frame.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(
            period_frame,
            text="Test Period (Out-of-Sample):",
            bg=theme['bg'],
            fg=theme['fg'],
            font=('Segoe UI', 10)
        ).pack(anchor=tk.W)
        
        date_inner = tk.Frame(period_frame, bg=theme['bg'])
        date_inner.pack(fill=tk.X, pady=5)
        
        tk.Label(date_inner, text="Start:", bg=theme['bg'], fg=theme['fg']).pack(side=tk.LEFT, padx=5)
        start_entry = tk.Entry(date_inner, width=12, bg=theme['entry_bg'], fg=theme['entry_fg'])
        start_entry.pack(side=tk.LEFT, padx=5)
        start_entry.insert(0, "2024-01-01")
        
        tk.Label(date_inner, text="End:", bg=theme['bg'], fg=theme['fg']).pack(side=tk.LEFT, padx=5)
        end_entry = tk.Entry(date_inner, width=12, bg=theme['entry_bg'], fg=theme['entry_fg'])
        end_entry.pack(side=tk.LEFT, padx=5)
        end_entry.insert(0, datetime.now().strftime("%Y-%m-%d"))
        
        # Info label
        info_label = tk.Label(
            period_frame,
            text="💡 Test on data NOT used for training (e.g., 2024-2025)",
            bg=theme['bg'],
            fg=theme['text_secondary'],
            font=('Segoe UI', 8)
        )
        info_label.pack(anchor=tk.W, pady=(5, 0))
        
        # Stock symbols input
        symbols_frame = tk.Frame(dialog, bg=theme['bg'])
        symbols_frame.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(
            symbols_frame,
            text="Stock Symbols (comma-separated):",
            bg=theme['bg'],
            fg=theme['fg'],
            font=('Segoe UI', 10)
        ).pack(anchor=tk.W)
        
        symbols_entry = tk.Entry(
            symbols_frame,
            font=('Segoe UI', 10),
            bg=theme['entry_bg'],
            fg=theme['entry_fg'],
            width=50
        )
        symbols_entry.pack(fill=tk.X, pady=5)
        symbols_entry.insert(0, "AAPL,MSFT,GOOGL,AMZN,TSLA")
        
        # Status text
        status_text = scrolledtext.ScrolledText(
            dialog,
            height=10,
            bg=theme['frame_bg'],
            fg=theme['fg'],
            font=('Consolas', 9)
        )
        status_text.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        # Buttons
        btn_frame = tk.Frame(dialog, bg=theme['bg'])
        btn_frame.pack(pady=10)
        
        def start_test():
            symbols = [s.strip().upper() for s in symbols_entry.get().split(',')]
            start_date = start_entry.get()
            end_date = end_entry.get()
            strategy = strategy_var.get()
            
            # Save symbols for next time
            symbols_str = ','.join(symbols)
            self.preferences.set_test_symbols(strategy, symbols_str)
            
            status_text.delete(1.0, tk.END)
            status_text.insert(tk.END, f"Starting test for {strategy} strategy...\n")
            status_text.insert(tk.END, f"Test Period: {start_date} to {end_date}\n")
            status_text.insert(tk.END, f"Symbols: {', '.join(symbols)}\n\n")
            status_text.update()
            
            # Run test in background
            def test():
                try:
                    result = self.model_tester.test_model(
                        strategy, symbols, start_date, end_date,
                        progress_callback=lambda msg: dialog.after(0, lambda: status_text.insert(tk.END, msg + "\n") or status_text.see(tk.END))
                    )
                    
                    if 'error' in result:
                        dialog.after(0, lambda: status_text.insert(tk.END, f"\n✗ Error: {result['error']}\n"))
                    else:
                        # Show results in status
                        dialog.after(0, lambda: status_text.insert(
                            tk.END,
                            f"\n{'='*50}\n"
                            f"TEST RESULTS\n"
                            f"{'='*50}\n\n"
                            f"Test Accuracy: {result['test_accuracy']:.1f}%\n"
                            f"Training Accuracy: {result['training_accuracy']:.1f}%\n"
                            f"Difference: {result['accuracy_difference']:+.1f}%\n\n"
                            f"Status: {result['overfitting_message']}\n"
                            f"Test Samples: {result['total_samples']}\n"
                            f"Correct: {result['correct']}\n\n"
                            f"Per-Action Accuracy:\n"
                            f"  BUY: {result['buy_accuracy']:.1f}% ({result['buy_count']} predictions)\n"
                            f"  SELL: {result['sell_accuracy']:.1f}% ({result['sell_count']} predictions)\n"
                            f"  HOLD: {result['hold_accuracy']:.1f}% ({result['hold_count']} predictions)\n"
                        ))
                        
                        # Show detailed results dialog
                        dialog.after(0, lambda: self._show_test_results(result))
                except Exception as e:
                    error_msg = str(e)  # Capture error message in local variable
                    dialog.after(0, lambda msg=error_msg: status_text.insert(tk.END, f"\n✗ Error: {msg}\n"))
            
            thread = threading.Thread(target=test)
            thread.daemon = True
            thread.start()
        
        test_btn = ModernButton(
            btn_frame,
            "Start Test",
            command=start_test,
            theme=theme
        )
        test_btn.pack(side=tk.LEFT, padx=5)
        
        close_btn = ModernButton(
            btn_frame,
            "Close",
            command=dialog.destroy,
            theme=theme
        )
        close_btn.pack(side=tk.LEFT, padx=5)
    
    def _show_test_results(self, test_result: dict):
        """Show detailed test results dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Test Results")
        dialog.geometry("700x600")
        dialog.transient(self.root)
        # Allow closing via window manager (X button)
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        
        theme = self.theme_manager.get_theme()
        dialog.config(bg=theme['bg'])
        
        # Header
        header = tk.Label(
            dialog,
            text="📊 Model Test Results",
            bg=theme['bg'],
            fg=theme['accent'],
            font=('Segoe UI', 16, 'bold')
        )
        header.pack(pady=20)
        
        # Results summary
        summary_frame = tk.Frame(dialog, bg=theme['bg'])
        summary_frame.pack(pady=10, padx=20, fill=tk.X)
        
        # Training vs Test comparison
        train_acc = test_result.get('training_accuracy', 0)
        test_acc = test_result.get('test_accuracy', 0)
        diff = test_result.get('accuracy_difference', 0)
        status = test_result.get('overfitting_status', 'unknown')
        
        # Color based on status
        if status == "good":
            status_color = theme['success']
            status_icon = "✅"
        elif status == "poor":
            status_color = theme['error']
            status_icon = "⚠️"
        else:
            status_color = theme['warning']
            status_icon = "⚡"
        
        comparison_text = f"{status_icon} {test_result.get('overfitting_message', '')}\n\n"
        comparison_text += f"Training Accuracy: {train_acc:.1f}%\n"
        comparison_text += f"Test Accuracy:      {test_acc:.1f}%\n"
        comparison_text += f"Difference:        {diff:+.1f}%"
        
        comparison_label = tk.Label(
            summary_frame,
            text=comparison_text,
            bg=theme['card_bg'],
            fg=status_color,
            font=('Segoe UI', 12, 'bold'),
            justify=tk.LEFT,
            padx=20,
            pady=15,
            relief=tk.RAISED,
            bd=2
        )
        comparison_label.pack(fill=tk.X)
        
        # Detailed stats
        stats_frame = tk.Frame(dialog, bg=theme['bg'])
        stats_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        stats_text = scrolledtext.ScrolledText(
            stats_frame,
            height=15,
            bg=theme['frame_bg'],
            fg=theme['fg'],
            font=('Consolas', 9),
            wrap=tk.WORD
        )
        stats_text.pack(fill=tk.BOTH, expand=True)
        
        # Build stats
        stats = f"{'='*60}\n"
        stats += f"TEST RESULTS - {test_result['strategy'].upper()}\n"
        stats += f"{'='*60}\n\n"
        
        stats += f"TEST PERIOD:\n"
        stats += f"  {test_result.get('test_period', 'N/A')}\n"
        stats += f"  Symbols: {', '.join(test_result.get('symbols_tested', []))}\n\n"
        
        stats += f"ACCURACY COMPARISON:\n"
        stats += f"  Training: {train_acc:.1f}%\n"
        stats += f"  Test:     {test_acc:.1f}%\n"
        stats += f"  Difference: {diff:+.1f}%\n\n"
        
        stats += f"INTERPRETATION:\n"
        if abs(diff) < 5:
            stats += f"  ✅ Model generalizes well\n"
            stats += f"  ✅ Small gap indicates good fit\n"
            stats += f"  ✅ Can trust predictions\n"
        elif diff < -10:
            stats += f"  ⚠️ Possible overfitting\n"
            stats += f"  ⚠️ Large gap suggests memorization\n"
            stats += f"  ⚠️ Consider retraining with regularization\n"
        else:
            stats += f"  ⚡ Model performance acceptable\n"
            stats += f"  ⚡ Some gap is normal\n"
        
        stats += f"\n{'='*60}\n"
        stats += f"DETAILED METRICS:\n"
        stats += f"{'='*60}\n\n"
        
        stats += f"Total Test Samples: {test_result.get('total_samples', 0)}\n"
        stats += f"Correct Predictions: {test_result.get('correct', 0)}\n"
        stats += f"Test Accuracy: {test_acc:.1f}%\n\n"
        
        stats += f"PER-ACTION ACCURACY:\n"
        stats += f"  BUY:  {test_result.get('buy_accuracy', 0):.1f}% ({test_result.get('buy_count', 0)} predictions)\n"
        stats += f"  SELL: {test_result.get('sell_accuracy', 0):.1f}% ({test_result.get('sell_count', 0)} predictions)\n"
        stats += f"  HOLD: {test_result.get('hold_accuracy', 0):.1f}% ({test_result.get('hold_count', 0)} predictions)\n"
        
        stats_text.insert(tk.END, stats)
        stats_text.config(state='disabled')
        
        # Close button
        close_btn = ModernButton(
            dialog,
            "Close",
            command=dialog.destroy,
            theme=theme
        )
        close_btn.pack(pady=10)
    
    def _start_continuous_backtest(self):
        """Start continuous backtesting mode"""
        if self.backtester.continuous_mode:
            messagebox.showinfo("Continuous Backtesting", "Continuous backtesting is already running!")
            return
        
        if self.backtester.is_testing:
            messagebox.showinfo("Backtest", "A backtest is already in progress. Please wait...")
            return
        
        # Show strategy selection dialog
        selected_strategies = self._show_backtest_strategy_selection()
        if not selected_strategies:
            return  # User cancelled
        
        # Show warning about safeguards
        response = messagebox.askyesno(
            "Start Continuous Backtesting?",
            "🔄 Start non-stop continuous backtesting?\n\n"
            "This will run backtests continuously until you click 'Stop'.\n\n"
            "⚠️ IMPORTANT:\n"
            "• Anti-overfitting safeguards are active\n"
            "• Model will only retrain once per day (24h cooldown)\n"
            "• Only fresh dates (not tested recently) will be used for training\n"
            "• This may take hours/days to see significant improvement\n\n"
            "Continue?"
        )
        
        if not response:
            return
        
        # Start continuous mode
        strategies_str = ", ".join([s.capitalize() for s in selected_strategies])
        self.backtester.start_continuous_mode(selected_strategies=selected_strategies)
        
        # Update UI
        self.continuous_backtest_btn.config(state=tk.DISABLED)
        self.stop_continuous_backtest_btn.config(state=tk.NORMAL)
        self._update_continuous_backtest_status()
        
        messagebox.showinfo(
            "Continuous Backtesting Started",
            f"🔄 Continuous backtesting started!\n\n"
            f"Strategies: {strategies_str}\n"
            f"Running every 60 seconds...\n\n"
            "Click 'Stop Continuous Backtesting' to terminate.\n\n"
            "Monitor the logs to see progress and when safeguards are active."
        )
    
    def _stop_continuous_backtest(self):
        """Stop continuous backtesting mode"""
        if not self.backtester.continuous_mode:
            messagebox.showinfo("Continuous Backtesting", "Continuous backtesting is not running.")
            return
        
        response = messagebox.askyesno(
            "Stop Continuous Backtesting?",
            "⏹️ Stop continuous backtesting?\n\n"
            "The current backtest will complete, then backtesting will stop."
        )
        
        if not response:
            return
        
        self.backtester.stop_continuous_mode()
        
        # Update UI
        self.continuous_backtest_btn.config(state=tk.NORMAL)
        self.stop_continuous_backtest_btn.config(state=tk.DISABLED)
        self._update_continuous_backtest_status()
        
        messagebox.showinfo(
            "Continuous Backtesting Stopped",
            "⏹️ Continuous backtesting stopped.\n\n"
            "The current backtest will finish, then no new backtests will start."
        )
    
    def _update_continuous_backtest_status(self):
        """Update the continuous backtesting status label"""
        if self.backtester.continuous_mode:
            if self.backtester.is_testing:
                status = "🔄 Running backtest..."
            else:
                status = "⏸️ Waiting for next run..."
            self.continuous_backtest_status.config(text=status, fg=self.theme_manager.get_theme()['accent'])
        else:
            self.continuous_backtest_status.config(text="", fg=self.theme_manager.get_theme()['fg'])
        
        # Schedule next update if running
        if self.backtester.continuous_mode:
            self.root.after(5000, self._update_continuous_backtest_status)  # Update every 5 seconds
    
    def _run_backtest(self):
        """Run backtest manually with strategy selection"""
        if self.backtester.is_testing:
            messagebox.showinfo("Backtest", "Backtest already in progress. Please wait...")
            return

        # Show strategy selection dialog
        result = self._show_backtest_strategy_selection()
        if not result:
            return  # User cancelled

        selected_strategies = result['strategies']
        num_tests = result['num_tests']

        # Run in background
        def run():
            success = self.backtester.run_test_now(selected_strategies=selected_strategies, num_tests=num_tests)
            if success:
                strategies_str = ", ".join([s.capitalize() for s in selected_strategies])
                self.root.after(0, lambda: messagebox.showinfo(
                    "Backtest Started",
                    f"🧪 Backtest started!\n\n"
                    f"Testing strategies: {strategies_str}\n"
                    f"Tests per strategy: {num_tests}\n\n"
                    "The algorithm will test itself on random historical points.\n"
                    "Results will be available in 'View Backtest Results'."
                ))
            else:
                self.root.after(0, lambda: messagebox.showwarning(
                    "Backtest", "Backtest already in progress."
                ))
        
        thread = threading.Thread(target=run)
        thread.daemon = True
        thread.start()
    
    def _show_backtest_strategy_selection(self):
        """Show dialog to select strategies for backtesting"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Select Strategies for Backtesting")
        dialog.geometry("400x320")
        dialog.transient(self.root)
        dialog.grab_set()  # Make dialog modal
        dialog.protocol("WM_DELETE_WINDOW", lambda: dialog.destroy())
        
        theme = self.theme_manager.get_theme()
        dialog.config(bg=theme['bg'])
        
        # Header
        header = tk.Label(
            dialog,
            text="Select Strategies to Test",
            bg=theme['bg'],
            fg=theme['accent'],
            font=('Segoe UI', 14, 'bold')
        )
        header.pack(pady=20)
        
        # Strategy checkboxes
        strategy_frame = tk.Frame(dialog, bg=theme['bg'])
        strategy_frame.pack(pady=10, padx=20)
        
        # Load saved preferences
        saved_strategies = self.preferences.get_backtest_strategies()
        if not saved_strategies:
            saved_strategies = ['trading', 'mixed', 'investing']  # Default: all
        
        strategy_vars = {}
        strategies = [
            ('trading', 'Trading (Short-term)'),
            ('mixed', 'Mixed (Medium-term)'),
            ('investing', 'Investing (Long-term)')
        ]
        
        for strategy_key, strategy_label in strategies:
            var = tk.BooleanVar(value=strategy_key in saved_strategies)
            strategy_vars[strategy_key] = var
            
            checkbox = tk.Checkbutton(
                strategy_frame,
                text=strategy_label,
                variable=var,
                bg=theme['bg'],
                fg=theme['fg'],
                selectcolor=theme['accent'],
                activebackground=theme['bg'],
                activeforeground=theme['fg'],
                font=('Segoe UI', 11)
            )
            checkbox.pack(anchor=tk.W, pady=5)
        
        # Number of tests input
        tests_frame = tk.Frame(dialog, bg=theme['bg'])
        tests_frame.pack(pady=10, padx=20, fill=tk.X)
        
        tk.Label(
            tests_frame,
            text="Number of Tests (per strategy):",
            bg=theme['bg'],
            fg=theme['fg'],
            font=('Segoe UI', 10)
        ).pack(side=tk.LEFT, padx=(0, 10))
        
        num_tests_var = tk.StringVar(value="100")
        num_tests_entry = tk.Entry(
            tests_frame,
            textvariable=num_tests_var,
            width=8,
            font=('Segoe UI', 10),
            bg=theme.get('entry_bg', '#ffffff'),
            fg=theme.get('entry_fg', '#000000')
        )
        num_tests_entry.pack(side=tk.LEFT)
        
        tk.Label(
            tests_frame,
            text="(200+ recommended)",
            bg=theme['bg'],
            fg=theme['text_secondary'],
            font=('Segoe UI', 8)
        ).pack(side=tk.LEFT, padx=(10, 0))
        
        # Buttons
        btn_frame = tk.Frame(dialog, bg=theme['bg'])
        btn_frame.pack(pady=20)
        
        result = {'strategies': [], 'num_tests': 100}
        
        def on_ok():
            selected = [key for key, var in strategy_vars.items() if var.get()]
            if not selected:
                messagebox.showwarning("No Selection", "Please select at least one strategy.")
                return
            
            # Parse num_tests
            try:
                num_tests = int(num_tests_var.get())
                if num_tests < 10:
                    num_tests = 10
                elif num_tests > 1000:
                    num_tests = 1000
            except ValueError:
                num_tests = 100
            
            result['strategies'] = selected
            result['num_tests'] = num_tests
            
            # Save preferences
            self.preferences.set_backtest_strategies(selected)
            dialog.destroy()
        
        def on_cancel():
            dialog.destroy()
        
        ModernButton(btn_frame, "OK", command=on_ok, theme=theme).pack(side=tk.LEFT, padx=5)
        ModernButton(btn_frame, "Cancel", command=on_cancel, theme=theme).pack(side=tk.LEFT, padx=5)
        
        # Wait for dialog to close
        dialog.wait_window()
        
        return result if result['strategies'] else None
    
    def _show_backtest_results(self):
        """Show backtest results dialog"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Backtest Results")
        dialog.geometry("800x700")
        dialog.transient(self.root)
        dialog.protocol("WM_DELETE_WINDOW", dialog.destroy)
        
        theme = self.theme_manager.get_theme()
        dialog.config(bg=theme['bg'])
        
        # Header
        header = tk.Label(
            dialog,
            text="🧪 Continuous Backtest Results",
            bg=theme['bg'],
            fg=theme['accent'],
            font=('Segoe UI', 16, 'bold')
        )
        header.pack(pady=20)
        
        # Get statistics
        stats = self.backtester.get_statistics()
        
        # Summary frame
        summary_frame = tk.Frame(dialog, bg=theme['bg'])
        summary_frame.pack(pady=10, padx=20, fill=tk.X)
        
        summary_text = f"📊 OVERALL PERFORMANCE:\n"
        summary_text += f"{'='*60}\n"
        summary_text += f"Total Tests: {stats['total_tests']:,}\n"
        summary_text += f"Correct: {stats['correct']:,}\n"
        summary_text += f"Incorrect: {stats['incorrect']:,}\n"
        summary_text += f"Overall Accuracy: {stats['overall_accuracy']:.1f}%\n"
        
        if stats.get('last_test_date'):
            try:
                last_test = datetime.fromisoformat(stats['last_test_date'])
                summary_text += f"Last Test: {last_test.strftime('%Y-%m-%d %H:%M')}\n"
            except (ValueError, KeyError, AttributeError) as e:
                logger.debug(f"Could not parse last test date: {e}")
        
        summary_text += f"\n📈 STRATEGY BREAKDOWN:\n"
        summary_text += f"{'='*60}\n"
        # Only show strategies that have been tested (have tests > 0)
        for strategy, results in stats['strategy_results'].items():
            if results['total'] > 0:
                summary_text += f"\n{strategy.upper()}:\n"
                summary_text += f"  Tests: {results['total']}\n"
                summary_text += f"  Correct: {results['correct']}\n"
                summary_text += f"  Accuracy: {results['accuracy']:.1f}%\n"
        
        summary_label = tk.Label(
            summary_frame,
            text=summary_text,
            bg=theme['bg'],
            fg=theme['fg'],
            font=('Consolas', 10),
            justify=tk.LEFT,
            anchor='w'
        )
        summary_label.pack(fill=tk.X)
        
        # Recent results
        recent_frame = tk.Frame(dialog, bg=theme['bg'])
        recent_frame.pack(pady=10, padx=20, fill=tk.BOTH, expand=True)
        
        tk.Label(
            recent_frame,
            text="📋 Recent Test Results:",
            bg=theme['bg'],
            fg=theme['accent'],
            font=('Segoe UI', 12, 'bold')
        ).pack(anchor=tk.W, pady=(0, 5))
        
        recent_results = self.backtester.get_recent_results(limit=20)
        
        results_text = scrolledtext.ScrolledText(
            recent_frame,
            height=15,
            bg=theme['frame_bg'],
            fg=theme['fg'],
            font=('Consolas', 9),
            wrap=tk.WORD
        )
        results_text.pack(fill=tk.BOTH, expand=True)
        
        if recent_results:
            for result in reversed(recent_results[-20:]):  # Show last 20
                symbol = result.get('symbol', 'N/A')
                test_date = result.get('test_date', 'N/A')
                strategy = result.get('strategy', 'N/A')
                was_correct = result.get('was_correct', False)
                actual_change = result.get('actual_change_pct', 0)
                predicted_action = result.get('predicted_action', 'N/A')
                confidence = result.get('confidence', 0)
                
                status = "✅" if was_correct else "❌"
                results_text.insert(tk.END, 
                    f"{status} {symbol} ({strategy}) - {test_date[:10]}\n"
                    f"   Action: {predicted_action} | Change: {actual_change:+.2f}% | "
                    f"Confidence: {confidence:.1f}%\n\n"
                )
        else:
            results_text.insert(tk.END, "No backtest results yet. Run a backtest to see results here.")
        
        results_text.config(state='disabled')
        
        # Buttons
        btn_frame = tk.Frame(dialog, bg=theme['bg'])
        btn_frame.pack(pady=10)
        
        run_btn = ModernButton(
            btn_frame,
            "🧪 Run Backtest Now",
            command=lambda: (self._run_backtest(), dialog.destroy()),
            theme=theme
        )
        run_btn.pack(side=tk.LEFT, padx=5)
        
        close_btn = ModernButton(
            btn_frame,
            "Close",
            command=dialog.destroy,
            theme=theme
        )
        close_btn.pack(side=tk.LEFT, padx=5)
    
    def _calculate_position_size_ui(self):
        """Calculate position size using risk management"""
        if not HAS_NEW_MODULES or not self.portfolio.risk_manager:
            messagebox.showwarning("Not Available", "Risk management module not available")
            return
        
        try:
            entry_price = float(self.rm_entry_price.get())
            stop_loss = float(self.rm_stop_loss.get())
            confidence = float(self.rm_confidence.get())
            method = self.rm_method.get()
            
            if entry_price <= 0 or stop_loss <= 0:
                messagebox.showerror("Error", "Entry price and stop loss must be positive")
                return
            
            result = self.portfolio.risk_manager.calculate_position_size(
                entry_price=entry_price,
                stop_loss=stop_loss,
                confidence=confidence,
                method=method
            )
            
            if 'error' in result:
                messagebox.showerror("Error", result['error'])
                return
            
            output = f"Position Size Calculation ({method})\n"
            output += "=" * 50 + "\n\n"
            output += f"Shares: {result['shares']:.4f}\n"
            output += f"Position Value: ${result['position_value']:.2f}\n"
            output += f"Position %: {result['position_percentage']:.2f}%\n"
            output += f"Risk Amount: ${result['risk_amount']:.2f}\n"
            output += f"Risk %: {result['risk_percentage']:.2f}%\n"
            output += f"Risk/Reward Ratio: {result.get('risk_reward_ratio', 0):.2f}\n"
            
            self.rm_position_results.config(state='normal')
            self.rm_position_results.delete('1.0', tk.END)
            self.rm_position_results.insert('1.0', output)
            self.rm_position_results.config(state='disabled')
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")
        except Exception as e:
            messagebox.showerror("Error", f"Calculation failed: {e}")
    
    def _calculate_stop_loss_ui(self):
        """Calculate stop loss recommendation"""
        if not HAS_NEW_MODULES or not self.portfolio.risk_manager:
            messagebox.showwarning("Not Available", "Risk management module not available")
            return
        
        try:
            entry_price = float(self.rm_entry_price.get() or self.rm_current_price.get())
            current_price = float(self.rm_current_price.get() or entry_price)
            volatility = float(self.rm_volatility.get() or "2.0")
            method = self.rm_sl_method.get()
            
            if entry_price <= 0:
                messagebox.showerror("Error", "Price must be positive")
                return
            
            result = self.portfolio.risk_manager.recommend_stop_loss(
                entry_price=entry_price,
                current_price=current_price,
                volatility=volatility,
                method=method
            )
            
            if 'error' in result:
                messagebox.showerror("Error", result['error'])
                return
            
            output = f"Stop Loss Recommendation ({method})\n"
            output += "=" * 50 + "\n\n"
            output += f"Stop Loss: ${result['stop_loss']:.2f}\n"
            output += f"Risk Amount: ${result['risk_amount']:.2f}\n"
            output += f"Risk %: {result['risk_percentage']:.2f}%\n\n"
            output += f"Reasoning:\n{result.get('reasoning', 'N/A')}\n"
            
            self.rm_stoploss_results.config(state='normal')
            self.rm_stoploss_results.delete('1.0', tk.END)
            self.rm_stoploss_results.insert('1.0', output)
            self.rm_stoploss_results.config(state='disabled')
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")
        except Exception as e:
            messagebox.showerror("Error", f"Calculation failed: {e}")
    
    def _calculate_portfolio_risk_ui(self):
        """Calculate portfolio risk metrics"""
        if not HAS_NEW_MODULES or not self.portfolio.risk_manager:
            messagebox.showwarning("Not Available", "Risk management module not available")
            return
        
        try:
            stats = self.portfolio.get_statistics()
            
            output = "Portfolio Risk Metrics\n"
            output += "=" * 50 + "\n\n"
            output += f"Total Balance: ${stats['balance']:,.2f}\n"
            output += f"Total Return: {stats['total_return']:.2f}%\n"
            output += f"Win Rate: {stats['win_rate']:.2f}%\n"
            
            if 'var_95' in stats:
                output += f"\nValue at Risk (95%): {stats['var_95']:.2f}%\n"
            
            if 'drawdown' in stats:
                dd = stats['drawdown']
                output += f"\nMaximum Drawdown: {dd.get('max_drawdown_percentage', 0):.2f}%\n"
                output += f"Average Drawdown: {dd.get('average_drawdown_percentage', 0):.2f}%\n"
            
            # Calculate portfolio risk if we have trades
            if self.portfolio.trades:
                positions = []
                for trade in self.portfolio.trades[-10:]:  # Last 10 trades
                    positions.append({
                        'symbol': trade.get('symbol', 'UNKNOWN'),
                        'value': trade.get('budget_used', 0),
                        'volatility': 0.20  # Estimate
                    })
                
                if positions:
                    risk_result = self.portfolio.risk_manager.calculate_portfolio_risk(positions)
                    if 'error' not in risk_result:
                        output += f"\nPortfolio Risk Analysis:\n"
                        output += f"  Portfolio Volatility: {risk_result['portfolio_volatility']:.4f}\n"
                        output += f"  Concentration Index: {risk_result['concentration_index']:.4f}\n"
                        output += f"  Diversification Score: {risk_result['diversification_score']:.4f}\n"
                        output += f"  Risk Level: {risk_result['risk_level']}\n"
            
            self.rm_portfolio_results.config(state='normal')
            self.rm_portfolio_results.delete('1.0', tk.END)
            self.rm_portfolio_results.insert('1.0', output)
            self.rm_portfolio_results.config(state='disabled')
        except Exception as e:
            messagebox.showerror("Error", f"Calculation failed: {e}")
    
    def _calculate_correlation_ui(self):
        """Calculate correlation between stocks"""
        if not HAS_NEW_MODULES or not self.portfolio.analytics:
            messagebox.showwarning("Not Available", "Advanced analytics module not available")
            return
        
        try:
            symbols_str = self.aa_symbols_entry.get()
            symbols = [s.strip().upper() for s in symbols_str.split(',') if s.strip()]
            
            if len(symbols) < 2:
                messagebox.showerror("Error", "Please enter at least 2 stock symbols")
                return
            
            # Fetch returns data
            returns_data = {}
            for symbol in symbols:
                try:
                    history = self.data_fetcher.fetch_stock_history(symbol, period='1y')
                    if history and 'data' in history:
                        import pandas as pd
                        df = pd.DataFrame(history['data'])
                        if 'close' in df.columns:
                            returns = df['close'].pct_change().dropna()
                            returns_data[symbol] = returns
                except Exception as e:
                    logger.debug(f"Could not fetch data for {symbol}: {e}")
            
            if len(returns_data) < 2:
                messagebox.showerror("Error", "Could not fetch data for enough symbols")
                return
            
            result = self.portfolio.analytics.calculate_correlation_matrix(
                symbols=list(returns_data.keys()),
                returns_data=returns_data
            )
            
            if 'error' in result:
                messagebox.showerror("Error", result['error'])
                return
            
            output = f"Correlation Analysis\n"
            output += "=" * 50 + "\n\n"
            output += f"Highest Correlation: {result['max_correlation']['value']:.4f}\n"
            output += f"  Between: {', '.join(result['max_correlation']['symbols'])}\n\n"
            output += f"Lowest Correlation: {result['min_correlation']['value']:.4f}\n"
            output += f"  Between: {', '.join(result['min_correlation']['symbols'])}\n\n"
            output += f"Average Correlation: {result['average_correlation']:.4f}\n"
            
            self.aa_correlation_results.config(state='normal')
            self.aa_correlation_results.delete('1.0', tk.END)
            self.aa_correlation_results.insert('1.0', output)
            self.aa_correlation_results.config(state='disabled')
        except Exception as e:
            messagebox.showerror("Error", f"Calculation failed: {e}")
    
    def _optimize_portfolio_ui(self):
        """Optimize portfolio weights"""
        if not HAS_NEW_MODULES or not self.portfolio.analytics:
            messagebox.showwarning("Not Available", "Advanced analytics module not available")
            return
        
        try:
            symbols_str = self.aa_opt_symbols_entry.get()
            symbols = [s.strip().upper() for s in symbols_str.split(',') if s.strip()]
            method = self.aa_opt_method.get()
            
            if len(symbols) < 2:
                messagebox.showerror("Error", "Please enter at least 2 stock symbols")
                return
            
            # Fetch returns data
            returns_data = {}
            for symbol in symbols:
                try:
                    history = self.data_fetcher.fetch_stock_history(symbol, period='1y')
                    if history and 'data' in history:
                        import pandas as pd
                        df = pd.DataFrame(history['data'])
                        if 'close' in df.columns:
                            returns = df['close'].pct_change().dropna()
                            returns_data[symbol] = returns
                except Exception as e:
                    logger.debug(f"Could not fetch data for {symbol}: {e}")
            
            if len(returns_data) < 2:
                messagebox.showerror("Error", "Could not fetch data for enough symbols")
                return
            
            result = self.portfolio.analytics.optimize_portfolio(
                symbols=list(returns_data.keys()),
                returns_data=returns_data,
                method=method
            )
            
            if 'error' in result:
                messagebox.showerror("Error", result['error'])
                return
            
            output = f"Portfolio Optimization ({method})\n"
            output += "=" * 50 + "\n\n"
            output += "Optimal Weights:\n"
            for symbol, weight in result['optimal_weights'].items():
                output += f"  {symbol}: {weight*100:.2f}%\n"
            output += f"\nExpected Return: {result['expected_return']*100:.2f}%\n"
            output += f"Expected Volatility: {result['expected_volatility']*100:.2f}%\n"
            output += f"Sharpe Ratio: {result['sharpe_ratio']:.4f}\n"
            
            self.aa_optimization_results.config(state='normal')
            self.aa_optimization_results.delete('1.0', tk.END)
            self.aa_optimization_results.insert('1.0', output)
            self.aa_optimization_results.config(state='disabled')
        except Exception as e:
            messagebox.showerror("Error", f"Optimization failed: {e}")
    
    def _run_monte_carlo_ui(self):
        """Run Monte Carlo simulation"""
        if not HAS_NEW_MODULES or not self.portfolio.analytics:
            messagebox.showwarning("Not Available", "Advanced analytics module not available")
            return
        
        try:
            initial_price = float(self.aa_mc_price.get())
            expected_return = float(self.aa_mc_return.get())
            volatility = float(self.aa_mc_vol.get())
            days = int(self.aa_mc_days.get())
            
            if initial_price <= 0:
                messagebox.showerror("Error", "Initial price must be positive")
                return
            
            result = self.portfolio.analytics.monte_carlo_simulation(
                initial_price=initial_price,
                expected_return=expected_return,
                volatility=volatility,
                days=days
            )
            
            if 'error' in result:
                messagebox.showerror("Error", result['error'])
                return
            
            output = f"Monte Carlo Simulation Results\n"
            output += "=" * 50 + "\n\n"
            output += f"Initial Price: ${result['initial_price']:.2f}\n"
            output += f"Simulations: {result['simulations']:,}\n"
            output += f"Days: {result['days']}\n\n"
            output += f"Mean Final Price: ${result['mean_final_price']:.2f}\n"
            output += f"Median Final Price: ${result['median_final_price']:.2f}\n"
            output += f"Expected Final Price: ${result['expected_final_price']:.2f}\n\n"
            output += f"95% Confidence Interval:\n"
            output += f"  Lower: ${result['confidence_interval']['lower']:.2f}\n"
            output += f"  Upper: ${result['confidence_interval']['upper']:.2f}\n\n"
            output += f"Probability of Profit: {result['probability_profit']*100:.2f}%\n"
            
            self.aa_mc_results.config(state='normal')
            self.aa_mc_results.delete('1.0', tk.END)
            self.aa_mc_results.insert('1.0', output)
            self.aa_mc_results.config(state='disabled')
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")
        except Exception as e:
            messagebox.showerror("Error", f"Simulation failed: {e}")
    
    def _run_walk_forward_ui(self):
        """Run walk-forward backtest"""
        if not HAS_NEW_MODULES or not self.walk_forward_backtester:
            messagebox.showwarning("Not Available", "Walk-forward backtester not available")
            return
        
        try:
            train_window = int(self.wf_train_window.get())
            test_window = int(self.wf_test_window.get())
            step_size = int(self.wf_step_size.get())
            multi_stock = self.wf_multi_stock.get() if hasattr(self, 'wf_multi_stock') else False
            
            # Get current strategy
            strategy = self.strategy_var.get() if hasattr(self, 'strategy_var') else 'trading'
            hybrid_predictor = self.hybrid_predictors.get(strategy)
            
            if not hybrid_predictor:
                messagebox.showerror("Error", f"No predictor available for strategy: {strategy}")
                return
            
            # Multi-stock mode
            if multi_stock:
                max_stocks = int(self.wf_max_stocks.get()) if hasattr(self, 'wf_max_stocks') and self.wf_max_stocks.get() else 10
                symbols = MarketScanner.POPULAR_STOCKS[:max_stocks]
                
                # Update UI to show progress
                self.wf_results.config(state='normal')
                self.wf_results.delete('1.0', tk.END)
                self.wf_results.insert('1.0', "Running MULTI-STOCK walk-forward backtest...\n\n"
                                              f"Stocks to test: {len(symbols)}\n"
                                              f"Strategy: {strategy.title()}\n"
                                              f"Train window: {train_window} days\n"
                                              f"Test window: {test_window} days\n"
                                              f"Step size: {step_size} days\n\n"
                                              f"Testing: {', '.join(symbols[:5])}{'...' if len(symbols) > 5 else ''}\n\n"
                                              "This may take several minutes...")
                self.wf_results.config(state='disabled')
                self.root.update()
                
                # Run multi-stock walk-forward test
                result = self.walk_forward_backtester.run_multi_stock_walk_forward_test(
                    symbols=symbols,
                    predictor=hybrid_predictor,
                    strategy=strategy,
                    data_fetcher=self.data_fetcher,
                    train_window_days=train_window,
                    test_window_days=test_window,
                    step_size_days=step_size,
                    lookforward_days=10,
                    max_stocks=max_stocks
                )
                
                if 'error' in result:
                    messagebox.showerror("Error", result['error'])
                    self.wf_results.config(state='normal')
                    self.wf_results.delete('1.0', tk.END)
                    self.wf_results.insert('1.0', f"Error: {result['error']}")
                    self.wf_results.config(state='disabled')
                    return
                
                # Display multi-stock results
                aggregate_results = result.get('aggregate_results', {})
                per_stock_results = result.get('per_stock_results', [])
                retrain_rec = aggregate_results.get('retrain_recommendation', {})
                
                output = f"MULTI-STOCK Walk-Forward Backtest Results\n"
                output += "=" * 60 + "\n\n"
                output += f"Stocks Tested: {result.get('stocks_tested', 0)}\n"
                output += f"Strategy: {strategy.title()}\n"
                output += f"Symbols: {', '.join(result.get('symbols', []))}\n\n"
                
                # Display retraining recommendation prominently
                if retrain_rec:
                    rec_emoji = retrain_rec.get('recommendation_emoji', '⚪')
                    rec_text = retrain_rec.get('recommendation_text', 'UNKNOWN')
                    priority = retrain_rec.get('priority', 'none')
                    reasons = retrain_rec.get('reasons', [])
                    positive_indicators = retrain_rec.get('positive_indicators', [])
                    confidence = retrain_rec.get('confidence', 0)
                    
                    # Update visual indicator in UI
                    self.wf_retrain_frame.pack(fill=tk.X, pady=(10, 0))
                    theme = self.theme_manager.get_theme()
                    
                    if rec_text == "RETRAIN NOW":
                        bg_color = '#ffebee'
                        fg_color = '#c62828'
                        action_text = "Run 'python retrain_models.py' to retrain models"
                    elif rec_text == "CONSIDER RETRAINING":
                        bg_color = '#fff3e0'
                        fg_color = '#e65100'
                        action_text = "Consider retraining models for better performance"
                    else:
                        bg_color = '#e8f5e9'
                        fg_color = '#2e7d32'
                        action_text = "Models performing well - no retraining needed"
                    
                    self.wf_retrain_frame.config(bg=bg_color)
                    recommendation_display = f"{rec_emoji} {rec_text} ({priority.upper()} priority, {confidence}% confidence)\n"
                    if reasons:
                        recommendation_display += f"Reasons: {', '.join(reasons[:2])}"
                        if len(reasons) > 2:
                            recommendation_display += f" (+{len(reasons)-2} more)"
                    recommendation_display += f"\n💡 {action_text}"
                    
                    self.wf_retrain_label.config(text=recommendation_display, bg=bg_color, fg=fg_color)
                    
                    output += f"{'='*60}\n"
                    output += f"🧠 RETRAINING RECOMMENDATION (Multi-Stock Aggregate)\n"
                    output += f"{'='*60}\n"
                    output += f"\n{rec_emoji} {rec_text}\n"
                    output += f"Priority: {priority.upper()}\n"
                    output += f"Confidence: {confidence}%\n\n"
                    
                    if reasons:
                        output += "⚠️ Reasons to Retrain:\n"
                        for i, reason in enumerate(reasons, 1):
                            output += f"  {i}. {reason}\n"
                        output += "\n"
                    
                    if positive_indicators:
                        output += "✅ Positive Indicators:\n"
                        for i, indicator in enumerate(positive_indicators, 1):
                            output += f"  {i}. {indicator}\n"
                        output += "\n"
                    
                    if rec_text == "RETRAIN NOW":
                        output += "💡 ACTION REQUIRED:\n"
                        output += "  Run 'python retrain_models.py' or use ML Training in the app\n"
                        output += "  Models need immediate retraining to improve accuracy\n\n"
                    elif rec_text == "CONSIDER RETRAINING":
                        output += "💡 RECOMMENDATION:\n"
                        output += "  Consider retraining models to improve performance\n"
                        output += "  Current performance is acceptable but could be better\n\n"
                    else:
                        output += "✅ STATUS:\n"
                        output += "  Models are performing well - no retraining needed\n"
                        output += "  Continue monitoring with regular walk-forward tests\n\n"
                    
                    output += f"{'='*60}\n\n"
                else:
                    self.wf_retrain_frame.pack_forget()
                
                # Aggregate metrics
                if aggregate_results:
                    output += "Aggregate Results (Across All Stocks):\n"
                    output += "-" * 60 + "\n"
                    output += f"Average Accuracy: {aggregate_results.get('aggregate_accuracy', 0)*100:.2f}%\n"
                    output += f"Std Deviation: {aggregate_results.get('aggregate_std', 0)*100:.2f}%\n"
                    output += f"Min Accuracy: {aggregate_results.get('aggregate_min', 0)*100:.2f}%\n"
                    output += f"Max Accuracy: {aggregate_results.get('aggregate_max', 0)*100:.2f}%\n"
                    output += f"Average Return: {aggregate_results.get('aggregate_return', 0)*100:.2f}%\n"
                    output += f"Average Sharpe Ratio: {aggregate_results.get('aggregate_sharpe', 0):.2f}\n\n"
                    
                    output += f"Retraining Breakdown:\n"
                    output += f"  🔴 RETRAIN NOW: {aggregate_results.get('retrain_now_count', 0)} stocks\n"
                    output += f"  🟡 CONSIDER RETRAINING: {aggregate_results.get('consider_retrain_count', 0)} stocks\n"
                    output += f"  🟢 NO RETRAIN NEEDED: {aggregate_results.get('no_retrain_count', 0)} stocks\n\n"
                
                # Per-stock summary
                if per_stock_results:
                    output += "Per-Stock Summary:\n"
                    output += "-" * 60 + "\n"
                    for stock_result in per_stock_results[:20]:  # Show first 20
                        symbol = stock_result.get('symbol', 'UNKNOWN')
                        agg = stock_result.get('result', {}).get('aggregated', {})
                        acc = agg.get('overall_accuracy', 0) * 100
                        rec = agg.get('retrain_recommendation', {}).get('recommendation_text', 'UNKNOWN')
                        output += f"{symbol}: {acc:.1f}% accuracy - {rec}\n"
                    
                    if len(per_stock_results) > 20:
                        output += f"\n... and {len(per_stock_results) - 20} more stocks\n"
                
                self.wf_results.config(state='normal')
                self.wf_results.delete('1.0', tk.END)
                self.wf_results.insert('1.0', output)
                self.wf_results.config(state='disabled')
                return
            
            # Single-stock mode (existing code)
            if not self.current_symbol or not self.current_data:
                messagebox.showwarning("No Data", "Please analyze a stock first")
                return
            
            # Update UI to show progress
            self.wf_results.config(state='normal')
            self.wf_results.delete('1.0', tk.END)
            self.wf_results.insert('1.0', "Running walk-forward backtest...\n\n"
                                          f"Symbol: {self.current_symbol}\n"
                                          f"Strategy: {strategy.title()}\n"
                                          f"Train window: {train_window} days\n"
                                          f"Test window: {test_window} days\n"
                                          f"Step size: {step_size} days\n\n"
                                          "Fetching historical data...")
            self.wf_results.config(state='disabled')
            self.root.update()
            
            # Get history data (need enough for walk-forward)
            period_days = train_window + test_window + 100  # Extra buffer
            if period_days <= 365:
                period = '1y'
            elif period_days <= 730:
                period = '2y'
            else:
                period = '5y'
            
            history_data = self.data_fetcher.fetch_stock_history(self.current_symbol, period=period)
            if not history_data or 'data' not in history_data:
                messagebox.showerror("Error", "Could not fetch historical data. Need at least {} days of data.".format(train_window + test_window))
                return
            
            # Check if we have enough data
            if len(history_data['data']) < train_window + test_window:
                messagebox.showerror("Error", 
                    f"Insufficient data. Have {len(history_data['data'])} days, "
                    f"need at least {train_window + test_window} days.")
                return
            
            # Update UI
            self.wf_results.config(state='normal')
            self.wf_results.delete('1.0', tk.END)
            self.wf_results.insert('1.0', "Running walk-forward backtest...\n\n"
                                          f"Symbol: {self.current_symbol}\n"
                                          f"Strategy: {strategy.title()}\n"
                                          f"Data points: {len(history_data['data'])}\n\n"
                                          "Generating predictions for each window...")
            self.wf_results.config(state='disabled')
            self.root.update()
            
            # Run walk-forward test with predictor
            result = self.walk_forward_backtester.run_walk_forward_test(
                history_data=history_data,
                predictor=hybrid_predictor,
                symbol=self.current_symbol,
                strategy=strategy,
                train_window_days=train_window,
                test_window_days=test_window,
                step_size_days=step_size,
                lookforward_days=10
            )
            
            if 'error' in result:
                messagebox.showerror("Error", result['error'])
                self.wf_results.config(state='normal')
                self.wf_results.delete('1.0', tk.END)
                self.wf_results.insert('1.0', f"Error: {result['error']}")
                self.wf_results.config(state='disabled')
                return
            
            # Display results
            aggregated = result.get('aggregated', {})
            window_results = result.get('window_results', [])
            retrain_rec = aggregated.get('retrain_recommendation', {})
            
            output = f"Walk-Forward Backtest Results\n"
            output += "=" * 60 + "\n\n"
            output += f"Symbol: {self.current_symbol}\n"
            output += f"Strategy: {strategy.title()}\n"
            output += f"Windows Tested: {result.get('windows_tested', 0)}\n\n"
            
            # Display retraining recommendation prominently
            if retrain_rec:
                rec_emoji = retrain_rec.get('recommendation_emoji', '⚪')
                rec_text = retrain_rec.get('recommendation_text', 'UNKNOWN')
                priority = retrain_rec.get('priority', 'none')
                reasons = retrain_rec.get('reasons', [])
                positive_indicators = retrain_rec.get('positive_indicators', [])
                confidence = retrain_rec.get('confidence', 0)
                
                # Update visual indicator in UI
                self.wf_retrain_frame.pack(fill=tk.X, pady=(10, 0))
                theme = self.theme_manager.get_theme()
                
                if rec_text == "RETRAIN NOW":
                    bg_color = '#ffebee'  # Light red
                    fg_color = '#c62828'  # Dark red
                    action_text = "Run 'python retrain_models.py' to retrain models"
                elif rec_text == "CONSIDER RETRAINING":
                    bg_color = '#fff3e0'  # Light orange
                    fg_color = '#e65100'  # Dark orange
                    action_text = "Consider retraining models for better performance"
                else:
                    bg_color = '#e8f5e9'  # Light green
                    fg_color = '#2e7d32'  # Dark green
                    action_text = "Models performing well - no retraining needed"
                
                self.wf_retrain_frame.config(bg=bg_color)
                recommendation_display = f"{rec_emoji} {rec_text} ({priority.upper()} priority, {confidence}% confidence)\n"
                if reasons:
                    recommendation_display += f"Reasons: {', '.join(reasons[:2])}"
                    if len(reasons) > 2:
                        recommendation_display += f" (+{len(reasons)-2} more)"
                recommendation_display += f"\n💡 {action_text}"
                
                self.wf_retrain_label.config(text=recommendation_display, bg=bg_color, fg=fg_color)
                
                output += f"{'='*60}\n"
                output += f"🧠 RETRAINING RECOMMENDATION\n"
                output += f"{'='*60}\n"
                output += f"\n{rec_emoji} {rec_text}\n"
                output += f"Priority: {priority.upper()}\n"
                output += f"Confidence: {confidence}%\n\n"
                
                if reasons:
                    output += "⚠️ Reasons to Retrain:\n"
                    for i, reason in enumerate(reasons, 1):
                        output += f"  {i}. {reason}\n"
                    output += "\n"
                
                if positive_indicators:
                    output += "✅ Positive Indicators:\n"
                    for i, indicator in enumerate(positive_indicators, 1):
                        output += f"  {i}. {indicator}\n"
                    output += "\n"
                
                if rec_text == "RETRAIN NOW":
                    output += "💡 ACTION REQUIRED:\n"
                    output += "  Run 'python retrain_models.py' or use ML Training in the app\n"
                    output += "  Models need immediate retraining to improve accuracy\n\n"
                elif rec_text == "CONSIDER RETRAINING":
                    output += "💡 RECOMMENDATION:\n"
                    output += "  Consider retraining models to improve performance\n"
                    output += "  Current performance is acceptable but could be better\n\n"
                else:
                    output += "✅ STATUS:\n"
                    output += "  Models are performing well - no retraining needed\n"
                    output += "  Continue monitoring with regular walk-forward tests\n\n"
                
                output += f"{'='*60}\n\n"
            else:
                # Hide recommendation frame if no recommendation
                self.wf_retrain_frame.pack_forget()
            
            if aggregated:
                output += "Overall Results:\n"
                output += "-" * 60 + "\n"
                output += f"Overall Accuracy: {aggregated.get('overall_accuracy', 0)*100:.2f}%\n"
                output += f"Mean Accuracy: {aggregated.get('mean_accuracy', 0)*100:.2f}%\n"
                output += f"Std Accuracy: {aggregated.get('std_accuracy', 0)*100:.2f}%\n"
                output += f"Min Accuracy: {aggregated.get('min_accuracy', 0)*100:.2f}%\n"
                output += f"Max Accuracy: {aggregated.get('max_accuracy', 0)*100:.2f}%\n"
                output += f"Mean Return: {aggregated.get('mean_return', 0)*100:.2f}%\n"
                output += f"Sharpe Ratio: {aggregated.get('sharpe_ratio', 0):.2f}\n\n"
            
            if window_results:
                output += "Window Details:\n"
                output += "-" * 60 + "\n"
                for wr in window_results[:10]:  # Show first 10 windows
                    output += f"Window {wr.get('window_num', 0)}:\n"
                    output += f"  Train: {wr.get('train_period', ('', ''))[0]} to {wr.get('train_period', ('', ''))[1]}\n"
                    output += f"  Test: {wr.get('test_period', ('', ''))[0]} to {wr.get('test_period', ('', ''))[1]}\n"
                    output += f"  Predictions: {wr.get('predictions_tested', 0)}\n"
                    output += f"  Accuracy: {wr.get('accuracy', 0)*100:.2f}% ({wr.get('correct', 0)}/{wr.get('correct', 0) + wr.get('incorrect', 0)})\n"
                    output += f"  Avg Return: {wr.get('avg_return', 0)*100:.2f}%\n\n"
                
                if len(window_results) > 10:
                    output += f"... and {len(window_results) - 10} more windows\n"
            
            self.wf_results.config(state='normal')
            self.wf_results.delete('1.0', tk.END)
            self.wf_results.insert('1.0', output)
            self.wf_results.config(state='disabled')
            
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers")
        except Exception as e:
            logger.error(f"Walk-forward backtest error: {e}", exc_info=True)
            messagebox.showerror("Error", f"Backtest failed: {e}")
            self.wf_results.config(state='normal')
            self.wf_results.delete('1.0', tk.END)
            self.wf_results.insert('1.0', f"Error: {str(e)}")
            self.wf_results.config(state='disabled')
    
    def _create_alert_ui(self):
        """Create a new alert"""
        if not HAS_NEW_MODULES or not self.alert_system:
            messagebox.showwarning("Not Available", "Alert system not available. Please ensure alert_system module is installed.")
            logger.warning("Alert system not available when trying to create alert")
            return
        
        try:
            alert_type = self.alert_type.get()
            symbol = self.alert_symbol.get().strip().upper()
            message = self.alert_message.get().strip()
            priority = self.alert_priority.get()
            
            # Validate inputs
            if not message:
                messagebox.showerror("Error", "Please enter an alert message")
                return
            
            if alert_type == 'price_alert':
                if not symbol:
                    messagebox.showerror("Error", "Please enter a symbol for price alert")
                    return
                # For price alerts, try to fetch current price and create price alert
                try:
                    stock_data = self.data_fetcher.fetch_stock_data(symbol)
                    if 'error' in stock_data:
                        raise Exception(stock_data.get('error', 'Failed to fetch stock data'))
                    current_price = stock_data.get('price', 0)
                    if current_price <= 0:
                        raise Exception("Invalid stock price")
                    # Use message as target price if it's a number, otherwise use 5% above
                    try:
                        target_price = float(message)
                        direction = 'above' if target_price > current_price else 'below'
                    except ValueError:
                        # Message is not a number, use default 5% above
                        target_price = current_price * 1.05
                        direction = 'above'
                    self.alert_system.create_price_alert(symbol, current_price, target_price, direction)
                except Exception as e:
                    logger.error(f"Error creating price alert: {e}")
                    # Fallback: create regular alert with the message
                    self.alert_system.add_alert(
                        alert_type, 
                        f"{symbol}: {message}", 
                        priority, 
                        {'symbol': symbol}
                    )
            else:
                # For other alert types, use the message as provided
                alert_data = {}
                if symbol:
                    alert_data['symbol'] = symbol
                self.alert_system.add_alert(alert_type, message, priority, alert_data)
            
            # Clear inputs
            self.alert_symbol.delete(0, tk.END)
            self.alert_message.delete(0, tk.END)
            
            # Refresh display
            self._refresh_alerts()
            messagebox.showinfo("Success", "Alert created successfully!")
            logger.info(f"Alert created: {alert_type} - {message}")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Failed to create alert: {error_msg}", exc_info=True)
            messagebox.showerror("Error", f"Failed to create alert: {error_msg}")
    
    def _refresh_alerts(self):
        """Refresh alerts display"""
        if not HAS_NEW_MODULES or not self.alert_system:
            return
        
        # Clear existing alerts (but keep the Create Alert card)
        for widget in self.alerts_list_frame.winfo_children():
            if isinstance(widget, ModernCard):
                # Don't destroy the Create Alert card
                if hasattr(self, 'create_alert_card') and widget == self.create_alert_card:
                    continue
                widget.destroy()
        
        # Get alerts
        alerts = self.alert_system.alerts
        unread_alerts = [a for a in alerts if not a.read]
        
        if not alerts:
            no_alerts_label = tk.Label(self.alerts_list_frame, 
                                     text="No alerts yet. Create one above!",
                                     font=('Segoe UI', 10), bg=self.theme_manager.get_theme()['bg'],
                                     fg=self.theme_manager.get_theme()['text_secondary'])
            no_alerts_label.pack(pady=20)
            return
        
        # Display alerts (most recent first)
        theme = self.theme_manager.get_theme()
        for alert in reversed(alerts[-20:]):  # Show last 20
            alert_card = ModernCard(self.alerts_list_frame, "", theme=theme, padding=10)
            alert_card.pack(fill=tk.X, pady=(0, 10))
            
            card_frame = tk.Frame(alert_card, bg=theme['card_bg'])
            card_frame.pack(fill=tk.X, padx=10, pady=10)
            
            # Header
            header = tk.Frame(card_frame, bg=theme['card_bg'])
            header.pack(fill=tk.X)
            
            priority_colors = {'low': '#4CAF50', 'medium': '#FF9800', 'high': '#F44336', 'critical': '#D32F2F'}
            priority_color = priority_colors.get(alert.priority, theme['fg'])
            
            priority_label = tk.Label(header, text=alert.priority.upper(), 
                                     font=('Segoe UI', 9, 'bold'), bg=theme['card_bg'], fg=priority_color)
            priority_label.pack(side=tk.LEFT)
            
            if not alert.read:
                unread_label = tk.Label(header, text="● NEW", 
                                       font=('Segoe UI', 9, 'bold'), bg=theme['card_bg'], fg=theme['accent'])
                unread_label.pack(side=tk.LEFT, padx=(10, 0))
            
            time_label = tk.Label(header, text=alert.timestamp.strftime('%Y-%m-%d %H:%M'), 
                                 font=('Segoe UI', 8), bg=theme['card_bg'], fg=theme['text_secondary'])
            time_label.pack(side=tk.RIGHT)
            
            # Message
            msg_label = tk.Label(card_frame, text=alert.message, 
                               font=('Segoe UI', 10), bg=theme['card_bg'], fg=theme['fg'],
                               wraplength=600, justify=tk.LEFT)
            msg_label.pack(anchor=tk.W, pady=(5, 0))
            
            # Buttons
            btn_frame = tk.Frame(card_frame, bg=theme['card_bg'])
            btn_frame.pack(fill=tk.X, pady=(10, 0))
            
            if not alert.read:
                mark_read_btn = ModernButton(btn_frame, text="Mark Read", 
                                           command=lambda a=alert: self._mark_alert_read(a),
                                           theme=theme)
                mark_read_btn.pack(side=tk.LEFT, padx=(0, 5))
            
            delete_btn = ModernButton(btn_frame, text="Delete", 
                                     command=lambda a=alert: self._delete_alert(a),
                                     theme=theme)
            delete_btn.pack(side=tk.LEFT)
    
    def _mark_alert_read(self, alert):
        """Mark alert as read"""
        if self.alert_system:
            self.alert_system.mark_as_read(alert)
            self._refresh_alerts()
    
    def _mark_all_alerts_read(self):
        """Mark all alerts as read"""
        if self.alert_system:
            self.alert_system.mark_all_as_read()
            self._refresh_alerts()
    
    def _delete_alert(self, alert):
        """Delete an alert"""
        if self.alert_system:
            self.alert_system.delete_alert(alert)
            self._refresh_alerts()


def main():
    """Main entry point"""
    # Create temporary root for splash screen
    splash_root = tk.Tk()
    splash_root.withdraw()  # Hide initially
    
    # Show splash screen with disclaimer first
    from splash_screen import SplashScreen
    splash = SplashScreen(splash_root)
    splash.show()
    
    # Splash screen destroys itself, so we don't need to destroy it here
    # Create main application root
    root = tk.Tk()
    app = StockerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

