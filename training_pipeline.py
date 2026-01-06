"""
Training Pipeline Manager
Manages automatic and manual training of ML models and weight adjustment
"""
import json
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import threading
import ssl
import os

from data_fetcher import StockDataFetcher
from ml_training import StockPredictionML, FeatureExtractor
from adaptive_weight_adjuster import AdaptiveWeightAdjuster
from predictions_tracker import PredictionsTracker

logger = logging.getLogger(__name__)

# Fix SSL certificate issues for yfinance/curl_cffi
# This is a workaround for Windows SSL certificate path issues
try:
    # Disable SSL verification as workaround (less secure but works)
    ssl._create_default_https_context = ssl._create_unverified_context
    # Clear problematic environment variables
    for env_var in ['CURL_CA_BUNDLE', 'REQUESTS_CA_BUNDLE', 'SSL_CERT_FILE', 'CURL_CAINFO']:
        os.environ.pop(env_var, None)
except:
    pass


class TrainingDataGenerator:
    """Generates training data from historical stock movements"""
    
    def __init__(self, data_fetcher: StockDataFetcher, feature_extractor: FeatureExtractor):
        self.data_fetcher = data_fetcher
        self.feature_extractor = feature_extractor
        self._patch_yfinance_ssl()
    
    def _patch_yfinance_ssl(self):
        """Patch yfinance to work around SSL certificate issues"""
        try:
            # Try to patch curl_cffi to disable SSL verification
            import curl_cffi.requests as cr
            original_request = cr.Session.request
            
            def patched_request(self, method, url, **kwargs):
                kwargs['verify'] = False
                return original_request(self, method, url, **kwargs)
            
            cr.Session.request = patched_request
            logger.debug("Patched curl_cffi to disable SSL verification")
        except Exception as patch_error:
            logger.debug(f"Could not patch curl_cffi: {patch_error}")
            # Try alternative: patch requests if yfinance uses it
            try:
                import requests
                original_get = requests.get
                original_post = requests.post
                
                def patched_get(*args, **kwargs):
                    kwargs['verify'] = False
                    return original_get(*args, **kwargs)
                
                def patched_post(*args, **kwargs):
                    kwargs['verify'] = False
                    return original_post(*args, **kwargs)
                
                requests.get = patched_get
                requests.post = patched_post
                logger.debug("Patched requests to disable SSL verification")
            except:
                pass
    
    def generate_training_samples(self, symbol: str, start_date: str, end_date: str,
                                 strategy: str = "trading", lookforward_days: int = 10) -> List[Dict]:
        """Generate training samples from historical data"""
        logger.info(f"🔍 Starting sample generation for {symbol} (strategy: {strategy}, lookforward: {lookforward_days} days)")
        try:
            logger.info(f"📥 Fetching data for {symbol} from {start_date} to {end_date}")
            
            # Use data_fetcher which has better SSL handling
            hist = None
            try:
                # Calculate period needed (max 10 years)
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                days_diff = (end_dt - start_dt).days
                if days_diff > 3650:  # More than 10 years
                    period = "10y"
                elif days_diff > 1825:  # More than 5 years
                    period = "5y"
                elif days_diff > 730:  # More than 2 years
                    period = "2y"
                else:
                    period = "1y"
                
                # Try using data_fetcher first (has SSL workarounds)
                history_data = self.data_fetcher.fetch_stock_history(symbol, period=period)
                if history_data and history_data.get('data'):
                    # Convert to DataFrame format
                    df_data = history_data['data']
                    hist = pd.DataFrame(df_data)
                    if 'date' in hist.columns:
                        hist['date'] = pd.to_datetime(hist['date'])
                        hist.set_index('date', inplace=True)
                        # Filter to date range
                        start_dt = pd.to_datetime(start_date)
                        end_dt = pd.to_datetime(end_date)
                        hist = hist[(hist.index >= start_dt) & (hist.index <= end_dt)]
                        # Rename columns to match yfinance format
                        hist.columns = [col.capitalize() for col in hist.columns]
                        if 'Close' not in hist.columns and 'close' in hist.columns:
                            hist.rename(columns={'close': 'Close', 'open': 'Open', 
                                                'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)
                    logger.info(f"✅ Fetched {len(hist)} rows of data for {symbol} (method: data_fetcher)")
            except Exception as df_error:
                logger.debug(f"data_fetcher method failed for {symbol}: {df_error}")
            
            # Fallback to direct yfinance if data_fetcher fails
            if hist is None or hist.empty:
                try:
                    ticker = yf.Ticker(symbol.upper())
                    
                    # Try download() method first (more reliable with SSL issues)
                    try:
                        logger.info(f"   Trying yfinance.download() for {symbol}...")
                        hist = yf.download(symbol.upper(), start=start_date, end=end_date, 
                                         progress=False, show_errors=False, timeout=60)
                        if not hist.empty:
                            # Download returns multi-index DataFrame, convert to single index
                            if isinstance(hist.columns, pd.MultiIndex):
                                hist = hist.droplevel(0, axis=1)
                            logger.info(f"✅ Fetched {len(hist)} rows of data for {symbol} (method: download)")
                    except Exception as e1:
                        logger.debug(f"download() failed for {symbol}: {e1}")
                        
                        # Try history() method
                        try:
                            logger.info(f"   Trying ticker.history() for {symbol}...")
                            hist = ticker.history(start=start_date, end=end_date, timeout=60)
                            if not hist.empty:
                                logger.info(f"✅ Fetched {len(hist)} rows of data for {symbol} (method: history)")
                        except Exception as e2:
                            logger.error(f"❌ All yfinance methods failed for {symbol}")
                            logger.error(f"   download() error: {str(e1)[:150]}")
                            logger.error(f"   history() error: {str(e2)[:150]}")
                            return []
                except Exception as yf_error:
                    logger.error(f"❌ yfinance initialization failed for {symbol}: {yf_error}")
                    return []
            
            if hist is None or hist.empty:
                logger.error(f"❌ No historical data returned for {symbol} (date range: {start_date} to {end_date})")
                return []
            
            if hist.empty:
                logger.error(f"❌ No historical data returned for {symbol} (date range: {start_date} to {end_date})")
                # Try to get info about the ticker
                try:
                    info = ticker.info
                    logger.info(f"ℹ️ Ticker info available: {info.get('symbol', 'N/A')}")
                except:
                    pass
                return []
            
            if len(hist) < 50:
                logger.error(f"❌ Insufficient data for {symbol}: {len(hist)} days (need at least 50)")
                return []
            
            # Need enough data for lookback + lookforward
            min_required = 50 + lookforward_days
            if len(hist) < min_required:
                logger.error(f"❌ Insufficient data for {symbol}: {len(hist)} days (need at least {min_required} for {lookforward_days} day lookforward)")
                return []
            
            samples = []
            dates = hist.index.tolist()
            
            logger.info(f"📊 Processing {symbol}: {len(hist)} days of data, generating samples with {lookforward_days} day lookforward")
            
            # Use sliding window - need at least 50 days history + lookforward_days future
            # Sample every 5 days to avoid too many samples and speed up training
            step = 5
            error_count = 0
            success_count = 0
            loop_count = 0
            
            # Calculate how many iterations we'll do
            start_idx = 50
            end_idx = len(dates) - lookforward_days
            total_iterations = len(range(start_idx, end_idx, step))
            logger.info(f"🔄 Will process {total_iterations} potential samples (range: {start_idx} to {end_idx}, step: {step})")
            
            for i in range(start_idx, end_idx, step):
                loop_count += 1
                current_date = dates[i]
                current_price = float(hist.iloc[i]['Close'])
                
                # Get historical data up to current date
                hist_to_date = hist.iloc[:i+1]
                
                # Convert to format expected by analyzers
                # Make sure we have enough data points
                if len(hist_to_date) < 20:
                    error_count += 1
                    continue
                
                history_data = {
                    'data': [
                        {
                            'date': date.strftime('%Y-%m-%d'),
                            'open': float(row['Open']),
                            'high': float(row['High']),
                            'low': float(row['Low']),
                            'close': float(row['Close']),
                            'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0
                        }
                        for date, row in hist_to_date.iterrows()
                    ]
                }
                
                # Validate history_data
                if not history_data.get('data') or len(history_data['data']) < 20:
                    error_count += 1
                    if error_count <= 3:
                        logger.debug(f"Insufficient history_data for {symbol} at {current_date}: {len(history_data.get('data', []))} points")
                    continue
                
                stock_data = {
                    'symbol': symbol,
                    'price': current_price,
                    'info': {}
                }
                
                # Get future price
                future_idx = min(i + lookforward_days, len(dates) - 1)
                future_price = float(hist.iloc[future_idx]['Close'])
                price_change_pct = ((future_price - current_price) / current_price) * 100
                
                # Label based on strategy (INVERTED LOGIC)
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
                        label = "SELL"  # Price went up significantly, bullish → SELL (was AVOID, now SELL)
                    else:
                        label = "HOLD"
                else:  # mixed
                    if price_change_pct <= -5:
                        label = "BUY"  # Price went down, bearish → BUY
                    elif price_change_pct >= 5:
                        label = "SELL"  # Price went up, bullish → SELL (was AVOID, now SELL)
                    else:
                        label = "HOLD"
                
                # Extract features - try to calculate indicators, but fallback to defaults if it fails
                indicators = {}
                try:
                    # Calculate technical indicators using the analyzer
                    # Reuse analyzer instance to avoid recreating it for each sample (saves API calls)
                    if not hasattr(self, '_cached_analyzer'):
                        from trading_analyzer import TradingAnalyzer
                        self._cached_analyzer = TradingAnalyzer(data_fetcher=None)  # No data_fetcher to avoid SPY API calls during training
                    
                    temp_analyzer = self._cached_analyzer
                    
                    # Initialize enhanced features variables (skip market regime/relative strength for training to avoid API calls)
                    market_regime = None
                    timeframe_analysis = None
                    relative_strength = None
                    support_resistance = None
                    
                    # Convert history_data to DataFrame for direct indicator calculation
                    if history_data and history_data.get('data') and len(history_data['data']) >= 20:
                        # Create DataFrame from history_data
                        hist_df = pd.DataFrame(history_data['data'])
                        if 'date' in hist_df.columns:
                            hist_df['date'] = pd.to_datetime(hist_df['date'])
                            hist_df.set_index('date', inplace=True)
                        
                        # Ensure columns are lowercase (analyzer expects lowercase)
                        column_mapping = {}
                        for col in hist_df.columns:
                            col_lower = col.lower()
                            if col_lower in ['close', 'open', 'high', 'low', 'volume']:
                                column_mapping[col] = col_lower
                        if column_mapping:
                            hist_df.rename(columns=column_mapping, inplace=True)
                        
                        # Ensure we have required columns
                        required_cols = ['close', 'open', 'high', 'low', 'volume']
                        missing_cols = [col for col in required_cols if col not in hist_df.columns]
                        if missing_cols:
                            if error_count < 3:
                                logger.debug(f"Missing columns for {symbol} at {current_date}: {missing_cols}")
                        else:
                            # Calculate indicators directly without full analysis (avoids SPY API calls)
                            try:
                                indicators = temp_analyzer._calculate_indicators(hist_df)
                                price_action = temp_analyzer._analyze_price_action(hist_df)
                                timeframe_analysis = temp_analyzer.timeframe_analyzer.analyze_timeframes(hist_df)
                                support_resistance = temp_analyzer._identify_levels(hist_df)
                            except Exception as calc_error:
                                if error_count < 3:
                                    logger.debug(f"Indicator calculation failed for {symbol} at {current_date}: {calc_error}")
                    else:
                        if error_count < 3:
                            logger.debug(f"Insufficient history data for {symbol} at {current_date}: {len(history_data.get('data', []))} days, using defaults")
                except Exception as analyzer_error:
                    # Analyzer failed, but we can still extract features with defaults
                    if error_count < 3:
                        logger.debug(f"Analyzer exception for {symbol} at {current_date}, using default indicators: {analyzer_error}")
                    # Initialize variables in case of exception
                    market_regime = None
                    timeframe_analysis = None
                    relative_strength = None
                    support_resistance = None
                
                # Extract features with enhanced analysis data (will use defaults if not available)
                try:
                    features = self.feature_extractor.extract_features(
                        stock_data, history_data, None, indicators if indicators else None,
                        market_regime, timeframe_analysis, relative_strength, support_resistance
                    )
                except Exception as feat_error:
                    error_count += 1
                    if error_count <= 5:
                        logger.error(f"FeatureExtractor failed for {symbol} at {current_date}: {feat_error}", exc_info=True)
                    continue
                
                # Validate features (moved outside except block - was unreachable before!)
                if features is None:
                    error_count += 1
                    if error_count <= 3:
                        logger.warning(f"None features returned for {symbol} at {current_date}")
                    continue
                
                # Convert to numpy array if needed
                if not isinstance(features, np.ndarray):
                    features = np.array(features)
                
                if len(features) == 0:
                    error_count += 1
                    if error_count <= 3:
                        logger.warning(f"Empty features array for {symbol} at {current_date}")
                    continue
                
                # Check for invalid values and fix them
                if np.any(np.isnan(features)) or np.any(np.isinf(features)):
                    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                
                samples.append({
                    'features': features.tolist(),
                    'label': label,
                    'price_change_pct': price_change_pct,
                    'date': current_date.strftime('%Y-%m-%d'),
                    'symbol': symbol
                })
                success_count += 1
                
                # Log progress every 50 samples
                if success_count % 50 == 0:
                    logger.info(f"   ✓ {symbol}: Generated {success_count} samples so far...")
            
            logger.info(f"✅ Completed processing {symbol}: Generated {len(samples)} samples (success: {success_count}, errors: {error_count}, loops: {loop_count})")
            
            if len(samples) == 0:
                logger.error(f"❌❌❌ CRITICAL: No samples generated for {symbol}! ❌❌❌")
                logger.error(f"   📅 Total dates in history: {len(dates)}")
                logger.error(f"   📅 Date range: {dates[0] if dates else 'N/A'} to {dates[-1] if dates else 'N/A'}")
                logger.error(f"   🔢 Loop iterations: {loop_count}")
                logger.error(f"   🔢 Success count: {success_count}")
                logger.error(f"   ❌ Error count: {error_count}")
                logger.error(f"   📊 Valid range for sampling: indices {start_idx} to {end_idx} (step {step})")
                if loop_count == 0:
                    logger.error(f"   ⚠️ LOOP NEVER EXECUTED! Check range calculation.")
                elif error_count == 0 and success_count == 0:
                    logger.error(f"   ⚠️ Loop executed but no samples added - all iterations were skipped silently")
                else:
                    logger.error(f"   ⚠️ Check logs above for specific error details")
            
            return samples
        
        except Exception as e:
            logger.error(f"Error generating training samples for {symbol}: {e}", exc_info=True)
            return []


class MLTrainingPipeline:
    """Manages ML training pipeline"""
    
    def __init__(self, data_fetcher: StockDataFetcher, data_dir: Path, app=None):
        self.data_fetcher = data_fetcher
        self.data_dir = data_dir
        self.app = app  # Optional app reference for notifications
        self.feature_extractor = FeatureExtractor()
        self.data_generator = TrainingDataGenerator(data_fetcher, self.feature_extractor)
    
    def train_on_symbols(self, symbols: List[str], start_date: str, end_date: str,
                        strategy: str = "trading", progress_callback=None) -> Dict:
        """Train ML model on multiple symbols"""
        logger.info(f"Training ML model for {strategy} on {len(symbols)} symbols...")
        
        # Determine lookforward_days based on strategy
        if strategy == "trading":
            lookforward_days = 10
        elif strategy == "investing":
            lookforward_days = 547  # ~1.5 years
        else:  # mixed
            lookforward_days = 21  # ~3 weeks
        
        all_samples = []
        
        for i, symbol in enumerate(symbols):
            if progress_callback:
                progress_callback(f"Processing {symbol} ({i+1}/{len(symbols)})...")
            
            logger.info(f"Generating samples for {symbol} (lookforward: {lookforward_days} days)...")
            try:
                samples = self.data_generator.generate_training_samples(
                    symbol, start_date, end_date, strategy, lookforward_days
                )
                all_samples.extend(samples)
                logger.info(f"Generated {len(samples)} samples for {symbol}")
                if len(samples) == 0:
                    logger.warning(f"No samples generated for {symbol}. Check date range and data availability.")
            except Exception as e:
                logger.error(f"Error generating samples for {symbol}: {e}", exc_info=True)
                if progress_callback:
                    progress_callback(f"Error processing {symbol}: {str(e)}")
        
        if len(all_samples) < 100:
            return {
                'error': f'Insufficient total samples: {len(all_samples)} < 100',
                'samples': len(all_samples)
            }
        
        logger.info(f"Total samples: {len(all_samples)}")
        
        # Train model
        ml_model = StockPredictionML(self.data_dir, strategy)
        
        if progress_callback:
            progress_callback("Training ML model...")
        
        results = ml_model.train(all_samples)
        
        results['total_samples'] = len(all_samples)
        results['symbols_used'] = symbols
        
        # Show notification if training was successful
        if 'error' not in results and self.app and hasattr(self.app, 'root'):
            self.app.root.after(0, self._show_training_complete_notification, strategy, results)
        
        return results
    
    def _show_training_complete_notification(self, strategy: str, results: Dict):
        """Show notification popup when ML training completes"""
        try:
            from tkinter import messagebox
            
            # Check notification preferences
            if self.app and hasattr(self.app, 'preferences'):
                show_notifications = self.app.preferences.get('show_training_complete_notifications', True)
                if not show_notifications:
                    logger.debug("Training complete notifications disabled by user preference")
                    return
            
            train_acc = results.get('train_accuracy', 0) * 100
            test_acc = results.get('test_accuracy', 0) * 100
            samples = results.get('total_samples', 0)
            
            message = f"✅ ML Training Complete!\n\n"
            message += f"Strategy: {strategy.title()}\n"
            message += f"Training Samples: {samples:,}\n"
            message += f"Training Accuracy: {train_acc:.1f}%\n"
            message += f"Test Accuracy: {test_acc:.1f}%\n"
            
            if results.get('cv_mean'):
                cv_mean = results.get('cv_mean', 0) * 100
                cv_std = results.get('cv_std', 0) * 100
                message += f"Cross-Validation: {cv_mean:.1f}% (±{cv_std:.1f}%)\n"
            
            messagebox.showinfo("ML Training Complete", message)
        except Exception as e:
            logger.debug(f"Error showing training notification: {e}")
    
    def _convert_verified_to_training_samples(self, verified_predictions: List[Dict],
                                            strategy: str) -> List[Dict]:
        """Convert verified predictions to training samples for ML model"""
        training_samples = []
        
        logger.info(f"Converting {len(verified_predictions)} verified predictions to training samples...")
        
        for pred in verified_predictions:
            try:
                symbol = pred.get('symbol', '').upper()
                if not symbol:
                    continue
                
                # Get prediction timestamp
                pred_timestamp = pred.get('timestamp')
                if not pred_timestamp:
                    continue
                
                pred_date = datetime.fromisoformat(pred_timestamp)
                
                # Get entry price and actual outcome price
                entry_price = pred.get('entry_price', 0)
                actual_price = pred.get('actual_price_at_target', 0)
                
                if entry_price <= 0 or actual_price <= 0:
                    continue
                
                # Calculate actual price change
                price_change_pct = ((actual_price - entry_price) / entry_price) * 100
                
                # Determine correct label based on actual outcome and strategy
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
                        label = "AVOID"
                    else:
                        label = "HOLD"
                else:  # mixed
                    if price_change_pct >= 5:
                        label = "BUY"
                    elif price_change_pct <= -5:
                        label = "AVOID"
                    else:
                        label = "HOLD"
                
                # Fetch historical data up to prediction date
                # We need at least 50 days of history before prediction
                start_date = (pred_date - timedelta(days=100)).strftime('%Y-%m-%d')
                end_date = pred_date.strftime('%Y-%m-%d')
                
                # Try to fetch historical data
                hist = None
                try:
                    # Use data_fetcher first
                    period = "3mo"  # Get 3 months of data
                    history_data = self.data_fetcher.fetch_stock_history(symbol, period=period)
                    if history_data and history_data.get('data'):
                        df_data = history_data['data']
                        hist = pd.DataFrame(df_data)
                        if 'date' in hist.columns:
                            hist['date'] = pd.to_datetime(hist['date'])
                            hist.set_index('date', inplace=True)
                            # Filter to before prediction date
                            hist = hist[hist.index <= pred_date]
                            # Rename columns
                            hist.columns = [col.capitalize() for col in hist.columns]
                            if 'Close' not in hist.columns and 'close' in hist.columns:
                                hist.rename(columns={'close': 'Close', 'open': 'Open', 
                                                    'high': 'High', 'low': 'Low', 'volume': 'Volume'}, inplace=True)
                except Exception as e:
                    logger.debug(f"Error fetching historical data for {symbol} at {pred_date}: {e}")
                
                # Fallback to yfinance if needed
                if hist is None or hist.empty:
                    try:
                        ticker = yf.Ticker(symbol)
                        hist = ticker.history(start=start_date, end=end_date, timeout=30)
                        if not hist.empty and isinstance(hist.columns, pd.MultiIndex):
                            hist = hist.droplevel(0, axis=1)
                    except Exception as e:
                        logger.debug(f"yfinance fallback failed for {symbol}: {e}")
                        continue
                
                if hist is None or hist.empty or len(hist) < 20:
                    continue
                
                # Convert to format expected by analyzers
                history_data = {
                    'data': [
                        {
                            'date': date.strftime('%Y-%m-%d'),
                            'open': float(row['Open']),
                            'high': float(row['High']),
                            'low': float(row['Low']),
                            'close': float(row['Close']),
                            'volume': int(row['Volume']) if not pd.isna(row['Volume']) else 0
                        }
                        for date, row in hist.iterrows()
                    ]
                }
                
                # Get current price at prediction time (last close price)
                current_price = float(hist.iloc[-1]['Close'])
                
                stock_data = {
                    'symbol': symbol,
                    'price': current_price,
                    'info': {}
                }
                
                # Calculate indicators and enhanced features
                indicators = {}
                market_regime = None
                timeframe_analysis = None
                relative_strength = None
                support_resistance = None
                try:
                    from trading_analyzer import TradingAnalyzer
                    temp_analyzer = TradingAnalyzer(data_fetcher=self.data_fetcher)
                    temp_analysis = temp_analyzer.analyze(stock_data, history_data)
                    if 'error' not in temp_analysis:
                        indicators = temp_analysis.get('indicators', {})
                        # Get enhanced analysis data
                        market_regime = temp_analysis.get('market_regime', None)
                        timeframe_analysis = temp_analysis.get('timeframe_analysis', None)
                        relative_strength = temp_analysis.get('relative_strength', None)
                        support_resistance = temp_analysis.get('support_resistance', None)
                except Exception as e:
                    logger.debug(f"Error calculating indicators for {symbol}: {e}")
                
                # Extract features with enhanced analysis data
                try:
                    features = self.feature_extractor.extract_features(
                        stock_data, history_data, None, indicators if indicators else None,
                        market_regime, timeframe_analysis, relative_strength, support_resistance
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
                        'date': pred_date.strftime('%Y-%m-%d'),
                        'symbol': symbol,
                        'was_correct': pred.get('was_correct', False)
                    })
                    
                except Exception as e:
                    logger.debug(f"Error extracting features for {symbol}: {e}")
                    continue
                    
            except Exception as e:
                logger.warning(f"Error processing verified prediction: {e}")
                continue
        
        logger.info(f"✅ Converted {len(training_samples)} verified predictions to training samples")
        return training_samples
    
    def train_on_verified_predictions(self, predictions_tracker: PredictionsTracker,
                                     strategy: str, retrain_ml: bool = True) -> Dict:
        """Train on verified predictions - updates weights and optionally retrains ML model"""
        verified = predictions_tracker.get_verified_predictions()
        strategy_verified = [p for p in verified if p.get('strategy') == strategy]
        
        if len(strategy_verified) < 50:
            return {
                'error': f'Insufficient verified predictions: {len(strategy_verified)} < 50'
            }
        
        # Always update weights based on verified predictions
        weight_adjuster = AdaptiveWeightAdjuster(self.data_dir, strategy)
        weight_adjuster.update_weights_from_results(strategy_verified)
        
        result = {
            'verified_predictions_used': len(strategy_verified),
            'weights_updated': True,
            'ml_retrained': False
        }
        
        # Optionally retrain ML model
        if retrain_ml:
            try:
                # Convert verified predictions to training samples
                training_samples = self._convert_verified_to_training_samples(strategy_verified, strategy)
                
                if len(training_samples) >= 50:
                    # Load existing ML model
                    ml_model = StockPredictionML(self.data_dir, strategy)
                    
                    # If model exists, we'll combine with existing training data
                    # For now, we'll retrain on verified predictions only
                    # In the future, we could combine with historical training data
                    training_result = ml_model.train(training_samples, test_size=0.2)
                    
                    if 'error' not in training_result:
                        result['ml_retrained'] = True
                        result['ml_train_accuracy'] = training_result.get('train_accuracy', 0) * 100
                        result['ml_test_accuracy'] = training_result.get('test_accuracy', 0) * 100
                        result['training_samples_used'] = len(training_samples)
                        logger.info(f"✅ ML model retrained on {len(training_samples)} verified predictions")
                    else:
                        result['ml_error'] = training_result.get('error')
                        logger.warning(f"ML retraining failed: {training_result.get('error')}")
                else:
                    result['ml_error'] = f'Insufficient training samples: {len(training_samples)} < 50'
                    logger.warning(f"Not enough training samples from verified predictions: {len(training_samples)}")
                    
            except Exception as e:
                logger.error(f"Error retraining ML model from verified predictions: {e}", exc_info=True)
                result['ml_error'] = str(e)
        
        return result


class AutoTrainingManager:
    """Manages automatic background training"""
    
    def __init__(self, app):
        self.app = app
        self.data_dir = app.data_dir if hasattr(app, 'data_dir') else app.predictions_tracker.data_dir
        self.last_training_time = {}
        self.min_predictions_for_training = 50
        self.training_interval_days = 7  # Weekly
        self.is_training = False
        self._show_notifications = True  # Enable notifications by default
    
    def check_and_train(self):
        """Check if training should happen automatically"""
        if self.is_training:
            return  # Already training
        
        now = datetime.now()
        strategy = self.app.current_strategy
        
        # Check if enough time has passed
        last_train = self.last_training_time.get(strategy)
        if last_train:
            days_since = (now - last_train).days
            if days_since < self.training_interval_days:
                return  # Too soon
        
        # Check if we have enough verified predictions
        verified = self.app.predictions_tracker.get_verified_predictions()
        strategy_verified = [p for p in verified if p.get('strategy') == strategy]
        
        if len(strategy_verified) >= self.min_predictions_for_training:
            # AUTOMATIC TRAINING TRIGGERED
            self._auto_train(strategy, strategy_verified)
    
    def _auto_train(self, strategy: str, verified_predictions: List[Dict]):
        """Automatically train on verified predictions"""
        logger.info(f"Auto-training {strategy} model on {len(verified_predictions)} verified predictions")
        self.is_training = True
        
        def train_in_background():
            try:
                # Use MLTrainingPipeline to train on verified predictions
                # This will update weights AND retrain ML model if enough data
                pipeline = MLTrainingPipeline(self.app.data_fetcher, self.data_dir)
                
                # Retrain ML model from verified predictions (if model exists)
                ml_model = StockPredictionML(self.data_dir, strategy)
                retrain_ml = ml_model.is_trained()  # Only retrain if model already exists
                
                result = pipeline.train_on_verified_predictions(
                    self.app.predictions_tracker, 
                    strategy,
                    retrain_ml=retrain_ml
                )
                
                if result.get('ml_retrained'):
                    test_acc = result.get('ml_test_accuracy', 0) * 100
                    logger.info(f"✅ Auto-training: ML model retrained (accuracy: {test_acc:.1f}%)")
                    # Show notification
                    if self._show_notifications:
                        self.app.root.after(0, self._show_auto_training_notification, 
                                           strategy, True, test_acc, None)
                else:
                    ml_error = result.get('ml_error', 'skipped')
                    logger.info(f"✅ Auto-training: Weights updated (ML retraining: {ml_error})")
                    # Show notification
                    if self._show_notifications:
                        self.app.root.after(0, self._show_auto_training_notification, 
                                           strategy, False, 0, ml_error)
                
                # Update last training time
                self.last_training_time[strategy] = datetime.now()
                
                logger.info(f"Auto-training complete for {strategy}")
            except Exception as e:
                logger.error(f"Error in auto-training: {e}", exc_info=True)
            finally:
                self.is_training = False
    
    def _show_auto_training_notification(self, strategy: str, ml_retrained: bool, 
                                        test_accuracy: float, error: Optional[str]):
        """Show notification popup when auto-training completes"""
        try:
            from tkinter import messagebox
            
            # Check notification preferences
            if self.app and hasattr(self.app, 'preferences'):
                show_notifications = self.app.preferences.get('show_auto_training_notifications', True)
                if not show_notifications:
                    logger.debug("Auto-training complete notifications disabled by user preference")
                    return
            
            message = f"✅ Auto-Training Complete!\n\n"
            message += f"Strategy: {strategy.title()}\n\n"
            
            if ml_retrained:
                message += f"ML Model Retrained\n"
                message += f"Test Accuracy: {test_accuracy:.1f}%\n"
                message += f"\nThe model has been improved with new data! 🚀"
            else:
                message += f"Weights Updated\n"
                if error:
                    message += f"ML Retraining: {error}\n"
                message += f"\nRule-based weights have been adjusted."
            
            messagebox.showinfo("Auto-Training Complete", message)
        except Exception as e:
            logger.debug(f"Error showing auto-training notification: {e}")
        
        # Run in background thread
        thread = threading.Thread(target=train_in_background)
        thread.daemon = True
        thread.start()
    
    def manual_train_ml(self, symbols: List[str], start_date: str, end_date: str,
                       strategy: str, progress_callback=None) -> Dict:
        """Manual ML training on historical data"""
        if self.is_training:
            return {'error': 'Training already in progress'}
        
        self.is_training = True
        
        def train():
            try:
                pipeline = MLTrainingPipeline(self.app.data_fetcher, self.app.data_dir)
                result = pipeline.train_on_symbols(
                    symbols, start_date, end_date, strategy, progress_callback
                )
                self.last_training_time[strategy] = datetime.now()
                return result
            except Exception as e:
                logger.error(f"Error in manual training: {e}")
                return {'error': str(e)}
            finally:
                self.is_training = False
        
        # Run in background thread
        thread = threading.Thread(target=train)
        thread.daemon = True
        thread.start()
        
        # Return immediately, training happens in background
        return {'status': 'training_started'}

