"""
Automatic Self-Learning System
Automatically scans stocks, makes predictions, and learns from them
Works in background while app is idle
"""
import logging
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class AutoLearner:
    """
    Automatically scans stocks, makes predictions, and learns from results
    Runs in background while app is idle
    """
    
    def __init__(self, app):
        """
        Initialize auto-learner
        
        Args:
            app: Main application instance (needs market_scanner, predictions_tracker, hybrid_predictors)
        """
        self.app = app
        self.is_running = False
        self.is_scanning = False
        
        # Configuration (can be adjusted)
        # Why 6 hours? Balance between:
        # - Market data changes (stocks don't change dramatically every hour)
        # - API rate limits (60 requests/min, but each scan = ~40 stocks × 3 strategies = 120+ calls)
        # - Learning speed (4 scans/day = ~20 predictions/day is good for steady learning)
        # - Resource usage (don't want to overwhelm system)
        # - Markets only open ~6.5 hours/day anyway
        # You can change this to 1-24 hours based on your needs
        self.scan_interval_hours = 6  # Scan every 6 hours (configurable)
        self.predictions_per_scan = 5  # Make 5 predictions per scan
        self.min_confidence = 62  # Aligned with Quant Mode Gate 1 (ML Confidence Floor)
        
        # Track last scan time per strategy
        self.last_scan_time = {}
        
        # Track symbols already predicted (to avoid duplicates)
        self.recently_predicted = {}  # {symbol: timestamp}
        self.recent_prediction_window_days = 7  # Don't re-predict same symbol within 7 days
        
        # Track scan results for UI updates
        self.last_scan_results = {}  # {strategy: {'predictions_made': int, 'timestamp': datetime}}
        
        logger.info("Auto-learner initialized")
    
    def start(self):
        """Start automatic learning (background scanning and prediction)"""
        if self.is_running:
            logger.warning("Auto-learner already running")
            return
        
        self.is_running = True
        logger.info("🚀 Auto-learner started - will scan and learn automatically")
        
        # Start first scan after 1 hour (give app time to initialize)
        self.app.root.after(3600000, self._schedule_next_scan)  # 1 hour = 3600000 ms
    
    def stop(self):
        """Stop automatic learning"""
        self.is_running = False
        logger.info("Auto-learner stopped")
    
    def _schedule_next_scan(self):
        """Schedule next automatic scan"""
        if not self.is_running:
            return
        
        # Run scan in background thread
        thread = threading.Thread(target=self._auto_scan_and_predict)
        thread.daemon = True
        thread.start()
        
        # Schedule next scan
        interval_ms = self.scan_interval_hours * 3600000
        self.app.root.after(interval_ms, self._schedule_next_scan)
    
    def scan_now(self):
        """Manually trigger immediate scan (public method for UI)"""
        if self.is_scanning:
            logger.info("Scan already in progress, please wait...")
            return False
        
        logger.info("🔍 Manual scan triggered by user")
        thread = threading.Thread(target=self._auto_scan_and_predict, args=(True,))  # Force scan
        thread.daemon = True
        thread.start()
        return True
    
    def _auto_scan_and_predict(self, force: bool = False):
        """Automatically scan market and make predictions
        
        Args:
            force: If True, bypass time checks and scan immediately
        """
        if self.is_scanning:
            logger.debug("Scan already in progress, skipping")
            return
        
        self.is_scanning = True
        
        try:
            logger.info("🔍 Auto-learner: Starting market scan...")
            
            # Scan for each strategy
            strategies = ['trading', 'mixed', 'investing']
            
            for strategy in strategies:
                if not self.is_running and not force:
                    break
                
                # Check if enough time has passed since last scan for this strategy (skip if forced)
                if not force:
                    last_scan = self.last_scan_time.get(strategy)
                    if last_scan:
                        hours_since = (datetime.now() - last_scan).total_seconds() / 3600
                        if hours_since < self.scan_interval_hours:
                            continue  # Too soon, skip this strategy
                
                try:
                    logger.info(f"Auto-learner: Scanning {strategy} strategy...")
                    
                    # Use market scanner to find opportunities
                    # Pass predictions_tracker to skip stocks with active predictions
                    recommendations = self.app.market_scanner.scan_market(
                        strategy=strategy,
                        max_results=self.predictions_per_scan * 2,  # Get more, filter by confidence
                        predictions_tracker=self.app.predictions_tracker
                    )
                    
                    # Filter by confidence and avoid symbols with active predictions
                    valid_recommendations = []
                    for rec in recommendations:
                        symbol = rec.get('symbol', '').upper()
                        confidence = rec.get('confidence', 0)
                        
                        # Check confidence threshold
                        if confidence < self.min_confidence:
                            continue
                        
                        # Check if symbol has an active prediction (prevent duplicates)
                        if self._has_active_prediction(symbol):
                            logger.debug(f"Skipping {symbol} - already has active prediction")
                            continue
                        
                        # Also check if recently predicted (additional safeguard)
                        if self._was_recently_predicted(symbol):
                            continue
                        
                        valid_recommendations.append(rec)
                    
                    # Limit to predictions_per_scan
                    valid_recommendations = valid_recommendations[:self.predictions_per_scan]
                    
                    # Make predictions
                    predictions_made = 0
                    for rec in valid_recommendations:
                        if not self.is_running:
                            break
                        
                        try:
                            symbol = rec['symbol']
                            action = rec.get('action', 'BUY')
                            
                            # HOLD/AVOID are vetoes — skip them
                            if action in ('HOLD', 'AVOID'):
                                logger.debug(f"Skipping {action} for {symbol} — veto signal")
                                continue
                            
                            entry_price = rec.get('entry_price', rec.get('price', 0))
                            target_price = rec.get('target_price', 0)
                            stop_loss = rec.get('stop_loss', 0)
                            confidence = rec.get('confidence', 0)
                            reasoning = rec.get('reasoning', 'Auto-generated prediction')
                            
                            # Calculate estimated days based on strategy
                            # Use dynamic date from recommendation if available
                            estimated_days = rec.get('estimated_days')
                            
                            if not estimated_days:
                                if strategy == "trading":
                                    estimated_days = 10
                                elif strategy == "mixed":
                                    estimated_days = 21
                                else:  # investing
                                    estimated_days = 547
                            
                            # Add prediction
                            prediction = self.app.predictions_tracker.add_prediction(
                                symbol=symbol,
                                strategy=strategy,
                                action=action,
                                entry_price=entry_price,
                                target_price=target_price,
                                stop_loss=stop_loss,
                                confidence=confidence,
                                reasoning=f"[Auto-Learner] {reasoning}",
                                estimated_days=estimated_days
                            )
                            
                            # Mark as recently predicted
                            self.recently_predicted[symbol] = datetime.now()
                            
                            predictions_made += 1
                            logger.info(f"✅ Auto-learner: Made prediction #{prediction['id']} for {symbol} ({strategy}, {confidence:.1f}% confidence)")
                            
                        except Exception as e:
                            logger.error(f"Error making auto-prediction for {rec.get('symbol', 'unknown')}: {e}")
                            continue
                    
                    # Update last scan time
                    self.last_scan_time[strategy] = datetime.now()
                    
                    # Store results for UI
                    self.last_scan_results[strategy] = {
                        'predictions_made': predictions_made,
                        'timestamp': datetime.now()
                    }
                    
                    # Record learning event
                    if hasattr(self.app, 'learning_tracker') and predictions_made > 0:
                        self.app.learning_tracker.record_auto_scan(predictions_made, strategy)
                    
                    # Update UI if available
                    if hasattr(self.app, '_update_ml_status'):
                        try:
                            self.app.root.after(0, self.app._update_ml_status)
                        except:
                            pass
                    
                    if predictions_made > 0:
                        logger.info(f"🎯 Auto-learner: Made {predictions_made} predictions for {strategy} strategy")
                        # Show notification if significant
                        if predictions_made >= 3:
                            try:
                                total_made = sum(r.get('predictions_made', 0) for r in self.last_scan_results.values())
                                if total_made >= 5:
                                    self.app.root.after(0, lambda: self._show_scan_complete_notification(total_made))
                            except:
                                pass
                    
                except Exception as e:
                    logger.error(f"Error in auto-scan for {strategy}: {e}")
                    continue
            
            # Clean up old recently_predicted entries
            self._cleanup_recent_predictions()
            
            logger.info("✅ Auto-learner: Scan complete")
            
        except Exception as e:
            logger.error(f"Error in auto-scan: {e}")
        finally:
            self.is_scanning = False
    
    def _has_active_prediction(self, symbol: str) -> bool:
        """Check if symbol already has an active prediction"""
        if not hasattr(self.app, 'predictions_tracker'):
            return False
        
        symbol_upper = symbol.upper()
        for pred in self.app.predictions_tracker.predictions:
            if (pred.get('symbol', '').upper() == symbol_upper and 
                pred.get('status') == 'active'):
                return True
        return False
    
    def _was_recently_predicted(self, symbol: str) -> bool:
        """Check if symbol was recently predicted"""
        symbol = symbol.upper()
        if symbol not in self.recently_predicted:
            return False
        
        last_prediction = self.recently_predicted[symbol]
        days_since = (datetime.now() - last_prediction).days
        
        return days_since < self.recent_prediction_window_days
    
    def _cleanup_recent_predictions(self):
        """Remove old entries from recently_predicted"""
        cutoff = datetime.now() - timedelta(days=self.recent_prediction_window_days)
        self.recently_predicted = {
            symbol: timestamp 
            for symbol, timestamp in self.recently_predicted.items()
            if timestamp > cutoff
        }
    
    def _show_scan_complete_notification(self, total_predictions: int):
        """Show notification when scan completes with predictions"""
        try:
            from tkinter import messagebox
            messagebox.showinfo(
                "Scan Complete",
                f"✅ Market scan complete!\n\n"
                f"Made {total_predictions} new predictions.\n"
                f"These will be verified automatically.\n\n"
                f"More predictions = faster learning! 🚀"
            )
        except:
            pass
    
    def get_status(self) -> Dict:
        """Get current status of auto-learner"""
        # Calculate total predictions from last scan
        total_last_scan = sum(r.get('predictions_made', 0) for r in self.last_scan_results.values())
        
        return {
            'is_running': self.is_running,
            'is_scanning': self.is_scanning,
            'last_scan_time': {k: v.isoformat() if v else None for k, v in self.last_scan_time.items()},
            'last_scan_predictions': total_last_scan,
            'recently_predicted_count': len(self.recently_predicted),
            'scan_interval_hours': self.scan_interval_hours,
            'predictions_per_scan': self.predictions_per_scan,
            'min_confidence': self.min_confidence
        }
    
    def update_config(self, scan_interval_hours: Optional[int] = None,
                     predictions_per_scan: Optional[int] = None,
                     min_confidence: Optional[int] = None):
        """Update auto-learner configuration"""
        if scan_interval_hours is not None:
            self.scan_interval_hours = max(1, scan_interval_hours)  # At least 1 hour
        if predictions_per_scan is not None:
            self.predictions_per_scan = max(1, min(20, predictions_per_scan))  # 1-20
        if min_confidence is not None:
            self.min_confidence = max(0, min(100, min_confidence))  # 0-100
        
        logger.info(f"Auto-learner config updated: interval={self.scan_interval_hours}h, "
                   f"predictions={self.predictions_per_scan}, min_confidence={self.min_confidence}%")

