"""
Automatic ML Training and Backtesting Scheduler
Runs scheduled tasks and provides notifications with results
"""
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, List
from config import APP_DATA_DIR

logger = logging.getLogger(__name__)


class MLScheduler:
    """Manages automatic scheduling of ML training and backtesting"""
    
    def __init__(self, app, data_dir: Path = None):
        self.app = app
        self.data_dir = data_dir or APP_DATA_DIR
        self.schedule_file = self.data_dir / "ml_schedule.json"
        self.accuracy_history_file = self.data_dir / "accuracy_history.json"
        
        # Default schedule
        self.schedule = {
            "backtesting": {
                "enabled": True,
                "interval_hours": 24,  # Daily
                "last_run": None,
                "next_run": None
            },
            "ml_training": {
                "enabled": True,
                "interval_days": 30,  # Monthly
                "last_run": None,
                "next_run": None
            }
        }
        
        # Accuracy tracking
        self.accuracy_history = {
            "backtesting": [],
            "ml_training": []
        }
        
        self.load_schedule()
        self.load_accuracy_history()
        
        # Set initial next_run times if not set
        if not self.schedule["backtesting"].get("next_run"):
            if self.schedule["backtesting"]["enabled"]:
                interval = self.schedule["backtesting"]["interval_hours"]
                self.schedule["backtesting"]["next_run"] = (datetime.now() + timedelta(hours=interval)).isoformat()
                self.save_schedule()
        
        if not self.schedule["ml_training"].get("next_run"):
            if self.schedule["ml_training"]["enabled"]:
                interval = self.schedule["ml_training"]["interval_days"]
                self.schedule["ml_training"]["next_run"] = (datetime.now() + timedelta(days=interval)).isoformat()
                self.save_schedule()
        
        # Start scheduler thread
        self.is_running = False
        self.scheduler_thread = None
    
    def load_schedule(self):
        """Load schedule from file"""
        if self.schedule_file.exists():
            try:
                with open(self.schedule_file, 'r') as f:
                    saved = json.load(f)
                    # Merge with defaults
                    for key in self.schedule:
                        if key in saved:
                            self.schedule[key].update(saved[key])
            except Exception as e:
                logger.error(f"Error loading schedule: {e}")
    
    def save_schedule(self):
        """Save schedule to file"""
        try:
            with open(self.schedule_file, 'w') as f:
                json.dump(self.schedule, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving schedule: {e}")
    
    def load_accuracy_history(self):
        """Load accuracy history"""
        if self.accuracy_history_file.exists():
            try:
                with open(self.accuracy_history_file, 'r') as f:
                    self.accuracy_history = json.load(f)
            except Exception as e:
                logger.error(f"Error loading accuracy history: {e}")
    
    def save_accuracy_history(self):
        """Save accuracy history"""
        try:
            with open(self.accuracy_history_file, 'w') as f:
                json.dump(self.accuracy_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving accuracy history: {e}")
    
    def start(self):
        """Start the scheduler"""
        if self.is_running:
            return
        
        # Check for overdue tasks on startup
        self._check_overdue_tasks()
        
        self.is_running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
        self.scheduler_thread.start()
        logger.info("ML Scheduler started")
    
    def _check_overdue_tasks(self):
        """Check if any scheduled tasks are overdue and run them"""
        try:
            now = datetime.now()
            logger.info("Checking for overdue scheduled tasks...")
            
            # Check backtesting
            if self.schedule["backtesting"]["enabled"]:
                next_run = self.schedule["backtesting"].get("next_run")
                if next_run:
                    next_run_dt = datetime.fromisoformat(next_run) if isinstance(next_run, str) else next_run
                    if now >= next_run_dt:
                        logger.info(f"Backtest is overdue (scheduled: {next_run_dt}), running now...")
                        self._run_scheduled_backtest()
                    else:
                        logger.info(f"Next backtest scheduled for: {next_run_dt}")
                else:
                    logger.info("No backtest scheduled yet - will schedule first run")
            else:
                logger.info("Backtesting scheduler is disabled")
            
            # Check ML training
            if self.schedule["ml_training"]["enabled"]:
                next_run = self.schedule["ml_training"].get("next_run")
                if next_run:
                    next_run_dt = datetime.fromisoformat(next_run) if isinstance(next_run, str) else next_run
                    if now >= next_run_dt:
                        logger.info(f"ML training is overdue (scheduled: {next_run_dt}), running now...")
                        self._run_scheduled_ml_training()
                    else:
                        logger.info(f"Next ML training scheduled for: {next_run_dt}")
                else:
                    logger.info("No ML training scheduled yet - will schedule first run")
            else:
                logger.info("ML training scheduler is disabled")
                        
        except Exception as e:
            logger.error(f"Error checking overdue tasks: {e}")
    
    def stop(self):
        """Stop the scheduler"""
        self.is_running = False
        logger.info("ML Scheduler stopped")
    
    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self.is_running:
            try:
                now = datetime.now()
                
                # Check backtesting schedule
                if self.schedule["backtesting"]["enabled"]:
                    next_run = self.schedule["backtesting"].get("next_run")
                    if next_run:
                        next_run_dt = datetime.fromisoformat(next_run) if isinstance(next_run, str) else next_run
                        if now >= next_run_dt:
                            self._run_scheduled_backtest()
                
                # Check ML training schedule
                if self.schedule["ml_training"]["enabled"]:
                    next_run = self.schedule["ml_training"].get("next_run")
                    if next_run:
                        next_run_dt = datetime.fromisoformat(next_run) if isinstance(next_run, str) else next_run
                        if now >= next_run_dt:
                            self._run_scheduled_ml_training()
                
                # Sleep for 1 hour before checking again
                import time
                time.sleep(3600)  # 1 hour
                
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
                import time
                time.sleep(3600)
    
    def _run_scheduled_backtest(self):
        """Run scheduled backtesting"""
        try:
            logger.info("Running scheduled backtest...")
            
            # Run backtest
            if hasattr(self.app, 'backtester') and self.app.backtester:
                # Run in background thread
                def run_backtest():
                    try:
                        # Get current results before backtest
                        if hasattr(self.app.backtester, 'results'):
                            results_before = self.app.backtester.results.copy()
                        else:
                            results_before = None
                        
                        # Run backtest
                        self.app.backtester.run_test_now(['trading'])
                        
                        # Wait for backtest to complete (check every second, max 5 minutes)
                        import time
                        max_wait = 300  # 5 minutes max
                        waited = 0
                        while waited < max_wait and self.app.backtester.is_testing:
                            time.sleep(1)
                            waited += 1
                        
                        # Get results from backtester after completion
                        if hasattr(self.app.backtester, 'results'):
                            results = self.app.backtester.results
                            if results:
                                self._process_backtest_results(results)
                    except Exception as e:
                        logger.error(f"Error running scheduled backtest: {e}")
                        self._show_notification(
                            "Backtest Error",
                            f"Error running scheduled backtest: {str(e)}",
                            "error"
                        )
                
                thread = threading.Thread(target=run_backtest, daemon=True)
                thread.start()
            
            # Update schedule
            self.schedule["backtesting"]["last_run"] = datetime.now().isoformat()
            interval = self.schedule["backtesting"]["interval_hours"]
            self.schedule["backtesting"]["next_run"] = (datetime.now() + timedelta(hours=interval)).isoformat()
            self.save_schedule()
            
        except Exception as e:
            logger.error(f"Error in scheduled backtest: {e}")
    
    def _run_scheduled_ml_training(self):
        """Run scheduled ML training"""
        try:
            logger.info("Running scheduled ML training...")
            
            # Get default training symbols from preferences
            symbols = "AAPL,MSFT,GOOGL,AMZN,TSLA,NVDA,META,NFLX,AMD,INTC,BA,CAT,GE,IBM,CSCO,JPM,BAC,WFC,GS,MS"
            if hasattr(self.app, 'preferences'):
                symbols = self.app.preferences.get_training_symbols("trading") or symbols
            
            # Run training in background
            def run_training():
                try:
                    from training_pipeline import MLTrainingPipeline
                    pipeline = MLTrainingPipeline(self.app.data_fetcher, self.data_dir, app=self.app)
                    
                    symbol_list = [s.strip().upper() for s in symbols.split(',')]
                    start_date = "2010-01-01"
                    end_date = datetime.now().strftime("%Y-%m-%d")
                    
                    result = pipeline.train_on_symbols(
                        symbol_list, start_date, end_date, "trading"
                    )
                    
                    if 'error' not in result:
                        test_acc = result.get('test_accuracy', 0) * 100
                        self._process_training_results(result)
                        self._show_notification(
                            "ML Training Complete",
                            f"Training completed successfully!\n"
                            f"Test Accuracy: {test_acc:.1f}%\n"
                            f"Training Samples: {result.get('total_samples', 0)}",
                            "success"
                        )
                    else:
                        self._show_notification(
                            "ML Training Error",
                            f"Training failed: {result.get('error', 'Unknown error')}",
                            "error"
                        )
                except Exception as e:
                    logger.error(f"Error running scheduled ML training: {e}")
                    self._show_notification(
                        "ML Training Error",
                        f"Error: {str(e)}",
                        "error"
                    )
            
            thread = threading.Thread(target=run_training, daemon=True)
            thread.start()
            
            # Update schedule
            self.schedule["ml_training"]["last_run"] = datetime.now().isoformat()
            interval = self.schedule["ml_training"]["interval_days"]
            self.schedule["ml_training"]["next_run"] = (datetime.now() + timedelta(days=interval)).isoformat()
            self.save_schedule()
            
        except Exception as e:
            logger.error(f"Error in scheduled ML training: {e}")
    
    def _process_backtest_results(self, results: Dict):
        """Process backtest results and check for accuracy drops"""
        try:
            # Get current accuracy
            strategy_results = results.get('strategy_results', {})
            trading_results = strategy_results.get('trading', {})
            current_accuracy = trading_results.get('accuracy', 0)
            
            # Add to history
            self.accuracy_history["backtesting"].append({
                "date": datetime.now().isoformat(),
                "accuracy": current_accuracy,
                "correct": trading_results.get('correct', 0),
                "total": trading_results.get('total', 0)
            })
            
            # Keep only last 30 entries
            if len(self.accuracy_history["backtesting"]) > 30:
                self.accuracy_history["backtesting"] = self.accuracy_history["backtesting"][-30:]
            
            self.save_accuracy_history()
            
            # Check for accuracy drop
            if len(self.accuracy_history["backtesting"]) >= 2:
                previous_accuracy = self.accuracy_history["backtesting"][-2]["accuracy"]
                accuracy_drop = previous_accuracy - current_accuracy
                
                if accuracy_drop > 5:  # Drop of more than 5%
                    suggestions = self._generate_fix_suggestions(current_accuracy, accuracy_drop, "backtesting")
                    self._show_notification(
                        "Accuracy Drop Detected",
                        f"Backtest accuracy dropped from {previous_accuracy:.1f}% to {current_accuracy:.1f}%\n"
                        f"Drop: {accuracy_drop:.1f}%\n\n"
                        f"Suggestions:\n" + "\n".join(f"• {s}" for s in suggestions),
                        "warning"
                    )
                else:
                    # Show success notification
                    self._show_notification(
                        "Backtest Complete",
                        f"Backtest completed successfully!\n"
                        f"Accuracy: {current_accuracy:.1f}%\n"
                        f"Tests: {trading_results.get('total', 0)}\n"
                        f"Correct: {trading_results.get('correct', 0)}",
                        "info"
                    )
            else:
                # First run - just show results
                self._show_notification(
                    "Backtest Complete",
                    f"Backtest completed successfully!\n"
                    f"Accuracy: {current_accuracy:.1f}%\n"
                    f"Tests: {trading_results.get('total', 0)}\n"
                    f"Correct: {trading_results.get('correct', 0)}",
                    "info"
                )
                
        except Exception as e:
            logger.error(f"Error processing backtest results: {e}")
    
    def _process_training_results(self, results: Dict):
        """Process training results and check for accuracy changes"""
        try:
            test_acc = results.get('test_accuracy', 0) * 100
            
            # Add to history
            self.accuracy_history["ml_training"].append({
                "date": datetime.now().isoformat(),
                "test_accuracy": test_acc,
                "train_accuracy": results.get('train_accuracy', 0) * 100,
                "samples": results.get('total_samples', 0)
            })
            
            # Keep only last 10 entries
            if len(self.accuracy_history["ml_training"]) > 10:
                self.accuracy_history["ml_training"] = self.accuracy_history["ml_training"][-10:]
            
            self.save_accuracy_history()
            
            # Check for overfitting or accuracy drop
            if len(self.accuracy_history["ml_training"]) >= 2:
                train_acc = results.get('train_accuracy', 0) * 100
                overfitting = train_acc - test_acc
                
                if overfitting > 20:  # Significant overfitting
                    self._show_notification(
                        "Overfitting Detected",
                        f"Training accuracy ({train_acc:.1f}%) is much higher than test accuracy ({test_acc:.1f}%)\n"
                        f"Difference: {overfitting:.1f}%\n\n"
                        f"Suggestions:\n"
                        f"• Model may be overfitting to training data\n"
                        f"• Consider reducing model complexity\n"
                        f"• Add more regularization\n"
                        f"• Use more diverse training data",
                        "warning"
                    )
                elif test_acc < 50:
                    self._show_notification(
                        "Low Training Accuracy",
                        f"Training accuracy is below 50%: {test_acc:.1f}%\n\n"
                        f"Suggestions:\n"
                        f"• Add more training data\n"
                        f"• Use more diverse stock symbols\n"
                        f"• Check feature quality\n"
                        f"• Consider different model parameters",
                        "warning"
                    )
                
        except Exception as e:
            logger.error(f"Error processing training results: {e}")
    
    def _generate_fix_suggestions(self, current_accuracy: float, accuracy_drop: float, task_type: str) -> List[str]:
        """Generate fix suggestions based on accuracy drop"""
        suggestions = []
        
        if task_type == "backtesting":
            if current_accuracy < 45:
                suggestions.append("Accuracy is very low - consider retraining ML model from historical data")
                suggestions.append("Check if model is overfitting (train accuracy >> test accuracy)")
            
            if accuracy_drop > 10:
                suggestions.append("Large accuracy drop detected - model may need fresh training")
                suggestions.append("Clear old backtest results and start fresh")
            
            if current_accuracy < 50:
                suggestions.append("Run manual ML training with more diverse stock symbols")
                suggestions.append("Increase training data date range (e.g., 2010-2024)")
            
            suggestions.append("Let the model learn from more backtest results (automatic retraining)")
            suggestions.append("Check if market conditions have changed significantly")
        
        return suggestions
    
    def _show_notification(self, title: str, message: str, notification_type: str = "info"):
        """Show notification to user"""
        try:
            from tkinter import messagebox
            
            # Schedule notification on main thread
            if hasattr(self.app, 'root'):
                self.app.root.after(0, lambda: self._show_messagebox(title, message, notification_type))
        except Exception as e:
            logger.error(f"Error showing notification: {e}")
    
    def _show_messagebox(self, title: str, message: str, notification_type: str):
        """Show messagebox (called on main thread)"""
        try:
            from tkinter import messagebox
            
            if notification_type == "error":
                messagebox.showerror(title, message)
            elif notification_type == "warning":
                messagebox.showwarning(title, message)
            elif notification_type == "success":
                messagebox.showinfo(title, message)
            else:
                messagebox.showinfo(title, message)
        except Exception as e:
            logger.error(f"Error showing messagebox: {e}")
    
    def get_schedule_status(self) -> Dict:
        """Get current schedule status"""
        return {
            "backtesting": {
                "enabled": self.schedule["backtesting"]["enabled"],
                "interval_hours": self.schedule["backtesting"]["interval_hours"],
                "last_run": self.schedule["backtesting"].get("last_run"),
                "next_run": self.schedule["backtesting"].get("next_run")
            },
            "ml_training": {
                "enabled": self.schedule["ml_training"]["enabled"],
                "interval_days": self.schedule["ml_training"]["interval_days"],
                "last_run": self.schedule["ml_training"].get("last_run"),
                "next_run": self.schedule["ml_training"].get("next_run")
            }
        }
    
    def update_schedule(self, task: str, enabled: bool = None, interval: int = None):
        """Update schedule settings"""
        if task in self.schedule:
            if enabled is not None:
                self.schedule[task]["enabled"] = enabled
            if interval is not None:
                if task == "backtesting":
                    self.schedule[task]["interval_hours"] = interval
                else:
                    self.schedule[task]["interval_days"] = interval
            
            # Update next run time
            if self.schedule[task]["enabled"]:
                if task == "backtesting":
                    interval_hours = self.schedule[task]["interval_hours"]
                    self.schedule[task]["next_run"] = (datetime.now() + timedelta(hours=interval_hours)).isoformat()
                else:
                    interval_days = self.schedule[task]["interval_days"]
                    self.schedule[task]["next_run"] = (datetime.now() + timedelta(days=interval_days)).isoformat()
            
            self.save_schedule()
