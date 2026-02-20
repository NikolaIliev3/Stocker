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
from config import ML_LABEL_THRESHOLD
# Import Quant Settings
try:
    from quant_config import IS_QUANT_MODE
except ImportError:
    IS_QUANT_MODE = False
import logging
import threading
import ssl
import os

from data_fetcher import StockDataFetcher
from ml_training import StockPredictionML, FeatureExtractor
from adaptive_weight_adjuster import AdaptiveWeightAdjuster
from predictions_tracker import PredictionsTracker
from pattern_matcher import PatternMatcher

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
        
        # Initialize analyzers
        from algorithm_improvements import MarketRegimeDetector, RelativeStrengthAnalyzer
        from macro_regime import MacroRegimeDetector
        self.regime_detector = MarketRegimeDetector()
        self.relative_strength_analyzer = RelativeStrengthAnalyzer()
        self.macro_detector = MacroRegimeDetector(data_fetcher)
        
        # Cache SPY data for market regime/relative strength
        self.spy_history = None
        self._fetch_spy_data()
        
        # Cache Sector ETF data
        self.sector_data_cache = {}

        # Cache Macro history
        self.macro_history = None
        self._fetch_macro_data()

    def _fetch_spy_data(self):
        """Fetch SPY data once for market regime calculations"""
        try:
            logger.info("Fetching SPY data for market regime analysis...")
            # Fetch long history for SPY
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
            
            # Use same logic as generate_training_samples to safely fetch
            # Try 1: data_fetcher
            spy_data = self.data_fetcher.fetch_stock_history('SPY', period='10y')
            if spy_data and spy_data.get('data'):
                self.spy_history = pd.DataFrame(spy_data['data'])
            
            # Try 2: yfinance direct
            if self.spy_history is None or self.spy_history.empty:
                import yfinance as yf
                self.spy_history = yf.download('SPY', start=start_date, end=end_date, progress=False)
            
            # Process SPY data
            if self.spy_history is not None and not self.spy_history.empty:
                if 'date' in self.spy_history.columns:
                    self.spy_history['date'] = pd.to_datetime(self.spy_history['date'])
                    self.spy_history.set_index('date', inplace=True)
                
                # Normalize columns
                self.spy_history.columns = [col.capitalize() for col in self.spy_history.columns]
                if 'Close' not in self.spy_history.columns and 'close' in self.spy_history.columns:
                     self.spy_history.rename(columns={'close': 'Close', 'volume': 'Volume'}, inplace=True)
                
                logger.info(f"Cached {len(self.spy_history)} rows of SPY data")
            else:
                logger.warning("Failed to fetch SPY data - Market Regime features will be neutral")
        except Exception as e:
            logger.error(f"Error fetching SPY data: {e}")
            
    def _fetch_macro_data(self):
        """Fetch historical macro data (VIX, TNX, DXY) for training"""
        try:
            logger.info("Fetching Macro data (VIX, TNX, DXY) for training...")
            import yfinance as yf
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365*10)
            
            # Usingtickers from MacroRegimeDetector
            tickers = ['^VIX', '^TNX', 'DX-Y.NYB']
            data = yf.download(tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)
            
            if not data.empty:
                # Standardize columns
                if isinstance(data.columns, pd.MultiIndex):
                    try:
                        closes = data['Close']
                    except KeyError:
                        closes = data['Adj Close']
                else:
                    closes = data
                
                # Map to internal names
                mapping = {'^VIX': 'VIX', '^TNX': 'TNX', 'DX-Y.NYB': 'DXY'}
                self.macro_history = closes.rename(columns=mapping)
                self.macro_history.index.name = 'date'
                
                logger.info(f"Cached {len(self.macro_history)} rows of Macro data")
            else:
                logger.warning("Failed to fetch Macro data")
        except Exception as e:
            logger.error(f"Error fetching Macro data: {e}")
    
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
        """Generate training samples from historical data (Optimized/Vectorized)"""
        logger.info(f"Starting sample generation for {symbol} (strategy: {strategy}, lookforward: {lookforward_days} days)")
        try:
            # 1. FETCH DATA (Full History)
            # ------------------------------------------------------------------
            # Calculate period needed (max 15 years to be safe)
            fetch_start = (datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # Fetch Stock Data
            hist = None
            try:
                # Use fetch_stock_history which handles SSL/caching
                history_data = self.data_fetcher.fetch_stock_history(
                    symbol, 
                    start_date=fetch_start, 
                    end_date=end_date,
                    period=None
                )
                if history_data and history_data.get('data'):
                    hist = pd.DataFrame(history_data['data'])
                    if 'date' in hist.columns:
                        hist['date'] = pd.to_datetime(hist['date'])
                        hist.set_index('date', inplace=True)
                        hist.columns = [col.lower() for col in hist.columns]
            except Exception as e:
                logger.debug(f"Data fetch error: {e}")

            if hist is None or hist.empty:
                # Fallback to direct yfinance
                import yfinance as yf
                hist = yf.download(symbol.upper(), start=fetch_start, end=end_date, progress=False)
                if not hist.empty:
                     # Standardize
                    if isinstance(hist.columns, pd.MultiIndex):
                        hist = hist.droplevel(0, axis=1)
                    hist.columns = [col.lower() for col in hist.columns]
            
            if hist is None or hist.empty or len(hist) < 200:
                logger.error(f"Insufficient data for {symbol}")
                return []

            # Ensure we have required columns
            required_cols = ['close', 'open', 'high', 'low', 'volume']
            for col in required_cols:
                if col not in hist.columns:
                    logger.error(f"Missing column {col} in data for {symbol}")
                    return []

            # 2. PRE-CALCULATE FEATURES (Vectorized)
            # ------------------------------------------------------------------
            logger.info("   Pre-calculating features vectorially...")
            
            # Use TradingAnalyzer to calc indicators on FULL DF
            from trading_analyzer import TradingAnalyzer
            if not hasattr(self, '_cached_analyzer'):
                self._cached_analyzer = TradingAnalyzer(data_fetcher=None)
            
            # This adds indicator columns to the DataFrame or returns a dict of Series
            # We want them in the DataFrame for easy row extraction
            indicators_df = self._cached_analyzer._calculate_indicators_vectorized(hist)
            
            # Macro Regime (Vectorized join)
            if self.macro_history is not None:
                # Reindex macro to match hist index (ffill)
                macro_aligned = self.macro_history.reindex(hist.index, method='ffill')
                indicators_df['VIX'] = macro_aligned['VIX']
                indicators_df['TNX'] = macro_aligned['TNX']
                indicators_df['DXY'] = macro_aligned['DXY']
            
            # Market Regime (Vectorized join)
            if self.spy_history is not None:
                try:
                    regime_df = self.regime_detector.compute_regimes_vectorized(self.spy_history)
                    # Reindex to match stock data
                    regime_aligned = regime_df.reindex(hist.index, method='ffill')
                    indicators_df['market_regime'] = regime_aligned['Regime']
                except Exception as e:
                    logger.warning(f"Vectorized regime sync failed: {e}")
                    indicators_df['market_regime'] = 'sideways'
            
            # Sector Analysis (Vectorized)
            sector_etf = self.data_fetcher.get_sector_etf(symbol)
            if sector_etf and sector_etf != 'SPY':
                if sector_etf not in self.sector_data_cache:
                     s_data = self.data_fetcher.fetch_sector_data(sector_etf, days=365*10)
                     if s_data is not None:
                         if 'date' in s_data.columns:
                             s_data['date'] = pd.to_datetime(s_data['date'])
                             s_data.set_index('date', inplace=True)
                         self.sector_data_cache[sector_etf] = s_data
                
                if sector_etf in self.sector_data_cache:
                    s_hist = self.sector_data_cache[sector_etf]
                    # Calculate sector momentum
                    col = 'close' if 'close' in s_hist.columns else 'Close'
                    s_mom = s_hist[col].pct_change(20)
                    # Align
                    s_mom_aligned = s_mom.reindex(hist.index, method='ffill')
                    indicators_df['sector_momentum_20d'] = s_mom_aligned

            # 3. SAMPLE GENERATION LOOP (Simplified)
            # ------------------------------------------------------------------
            # Filter to requested start/end date
            mask = (hist.index >= pd.to_datetime(start_date)) & (hist.index <= pd.to_datetime(end_date))
            target_indices = np.where(mask)[0]
            
            # Step size
            step = 5
            valid_indices = target_indices[::step]
            
            samples = []
            dates = hist.index
            prices = hist['close'].values
            highs = hist['high'].values
            lows = hist['low'].values
            opens = hist['open'].values
            
            logger.info(f"   Extracting {len(valid_indices)} samples...")
            
            for idx in valid_indices:
                # Ensure we have lookforward space
                if idx + lookforward_days >= len(hist):
                    continue
                    
                current_date = dates[idx]
                current_price = prices[idx]
                
                if current_price <= 0: continue

                if current_price <= 0: continue
                
                # Get window for MAX PROFIT calculation
                window_end = min(idx + lookforward_days + 1, len(hist))
                future_highs = highs[idx+1 : window_end]
                
                max_price = 0
                max_profit_pct = 0.0
                if len(future_highs) > 0:
                    max_price = np.max(future_highs)
                    if current_price > 0:
                        max_profit_pct = ((max_price - current_price) / current_price) * 100

                # --- TERMINAL SUCCESS LABELING (3% HURDLE) ---
                # A trade is a success if it is >= 3% above entry at EXPRTY.
                # Mid-period drops and stop-losses are IGNORED (Patient Sniper Mode).
                
                from config import ML_LABEL_THRESHOLD
                success_threshold = ML_LABEL_THRESHOLD / 100.0
                
                # Check price at the EXACT end of the lookforward period (Expiry)
                expiry_idx = min(idx + lookforward_days, len(prices) - 1)
                expiry_price = prices[expiry_idx]
                
                price_change = (expiry_price - current_price) / current_price
                
                if price_change >= success_threshold:
                    label = 'BUY' # Terminal Winner
                else:
                    label = 'HOLD' # Failed to hit 3% at expiry
                
                if strategy == "investing":
                     # Simple return based - strict 10% gain required for BUY
                     ret = (prices[min(idx+lookforward_days, len(prices)-1)] - current_price) / current_price
                     if ret > 0.10: 
                        label = 'BUY'
                     else: 
                        label = 'HOLD'

                # --- FEATURE EXTRACTION (O(1) Lookup) ---
                # We simply grab the row from indicators_df and format it
                # The feature extractor expects a specific format, so we might need
                # to manually construct the feature vector to match exactly what the model expects.
                # However, to be safe and consistent, we can pass the PRE-CALCULATED dict to the extractor.
                
                # Construct indicators dict from the row
                row_data = indicators_df.iloc[idx]
                
                # We need to map the flat DF row back to the nested structure FeatureExtractor expects?
                # Actually, FeatureExtractor takes `indicators` dict. We can just pass the row as dict.
                row_dict = row_data.to_dict()
                
                # EXTRACT KEY PATTERN FEATURES FOR SAVING
                pattern_features = {
                    'rsi': float(row_dict.get('rsi', 50)),
                    'atr_percent': float(row_dict.get('atr_percent', 0)),
                    'sma50_dist': float((current_price / row_dict.get('ema_50', current_price) - 1) * 100) if row_dict.get('ema_50') else 0.0
                }
                
                # Market Regime features
                m_regime = row_dict.get('market_regime', 'sideways')
                regime_dict = {'regime': m_regime, 'strength': 50} # Simplified
                
                # Macro
                macro_dict = {'regime': 'neutral'} # Simplified
                
                 # Sector
                sector_dict = None
                if 'sector_momentum_20d' in row_dict and not pd.isna(row_dict['sector_momentum_20d']):
                     s_mom = row_dict['sector_momentum_20d']
                     # Stock mom
                     stock_mom = (current_price - prices[idx-20])/prices[idx-20] if idx >= 20 else 0
                     rel = stock_mom - s_mom
                     sector_dict = {
                         'available': True,
                         'stock_vs_sector': rel * 100,
                         'sector_trend': 'uptrend' if s_mom > 0 else 'downtrend'
                     }

                # Prepare history slice for TS features (Need ~50 days for MA/Vol features)
                # This ensures Training features match Prediction features (which use full history)
                # Mismatch (0 vs 50M) caused Scaler to fail.
                hist_slice_start = max(0, idx - 60)
                hist_slice = hist.iloc[hist_slice_start:idx+1] # Include current row
                
                # Convert slice to dict format expected by extractor
                # We reset index to make it JSON-serializable list of dicts
                hist_data_dict = {'data': hist_slice.reset_index().to_dict(orient='records')}

                # CALL EXTRACTOR
                try:
                    features = self.feature_extractor.extract_features(
                        stock_data={'price': current_price},
                        history_data=hist_data_dict, # Pass valid history slice
                        indicators=row_dict, # PASS PRE-CALCED INDICATORS
                        market_regime=regime_dict,
                        sector_analysis=sector_dict,
                        macro_analysis=macro_dict,
                        as_of_date=current_date
                    )
                    
                    # Sanitize
                    if np.any(np.isnan(features)):
                         features = np.nan_to_num(features)
                         
                    samples.append({
                        'features': features.tolist(),
                        'label': label,
                        'meta_label': 1 if label != 'HOLD' else 0, # Simplified meta
                        'rule_action': label, # Using label as proxy for rule action in fast mode
                        'price_change_pct': float(price_change * 100), # Terminal profit
                        'max_profit_pct': float(max_profit_pct), # PEAK profit (New for Phase 5)
                        'pattern_features': pattern_features, # RSI, ATR, etc. (New for Phase 5)
                        'date': current_date.strftime('%Y-%m-%d'),
                        'symbol': symbol,
                        'sample_weight': 1.0
                    })
                except Exception as e:
                    pass
            
            logger.info(f"Generated {len(samples)} samples for {symbol}")
            return samples

        except Exception as e:
            logger.error(f"Error generating samples for {symbol}: {e}", exc_info=True)
            return []


class MLTrainingPipeline:
    """Manages ML training pipeline"""
    
    def __init__(self, data_fetcher: StockDataFetcher, data_dir: Path, app=None):
        self.data_fetcher = data_fetcher
        self.data_dir = data_dir
        self.app = app  # Optional app reference for notifications
        # Initialize FeatureExtractor with data_fetcher to enable enhanced features
        self.feature_extractor = FeatureExtractor(data_fetcher=data_fetcher)
        self.data_generator = TrainingDataGenerator(data_fetcher, self.feature_extractor)
    
    def train_on_symbols(self, symbols: List[str], start_date: str = "2020-01-01", end_date: str = None,
                        strategy: str = "trading", progress_callback=None, **kwargs) -> Dict:
        """Train ML model on multiple symbols"""
        logger.info(f"Training ML model for {strategy} on {len(symbols)} symbols...")
        
        # Determine lookforward_days based on strategy
        if strategy == "trading":
            lookforward_days = 15  # Expanded from 10 to allow 3% targets to develop
        elif strategy == "investing":
            lookforward_days = 60  # 3 months
        else:  # mixed
            lookforward_days = 21  # ~3 weeks

        if end_date is None:
            from datetime import datetime
            end_date = datetime.now().strftime("%Y-%m-%d")
        
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
        
        # Auto-merge verified predictions (weighted 20x) if available
        # This fixes the "Retrain from verified overwrites main model" bug by merging instead
        if self.app and hasattr(self.app, 'predictions_tracker'):
            try:
                verified = self.app.predictions_tracker.get_verified_predictions()
                strategy_verified = [p for p in verified if p.get('strategy') == strategy]
                
                if strategy_verified:
                    verified_samples = self._convert_verified_to_training_samples(strategy_verified, strategy)
                    if verified_samples:
                        # Add high weight to verified samples (20x)
                        for s in verified_samples:
                            s['sample_weight'] = 20.0
                        
                        # PREPEND verified samples so they survive duplicate removal (which keeps first occurrence)
                        # This works because np.unique with return_index keeps the first index
                        all_samples = verified_samples + all_samples
                        logger.info(f"Merged {len(verified_samples)} verified predictions into training set with BOOSTER weight (20x)")
            except Exception as e:
                logger.error(f"Error merging verified predictions: {e}")
        
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
        
        # Check if a saved configuration should be used
        training_params = {}
        if self.app and hasattr(self.app, '_saved_training_config') and self.app._saved_training_config:
            training_params = self.app._saved_training_config.copy()
            logger.info(f"Using saved configuration for training")
            # Clear after use (one-time use)
            self.app._saved_training_config = None
        
        # Use binary classification mode for better accuracy (60-75% vs 45-48%)
        # Binary mode removes ambiguous HOLD class and focuses on clear UP/DOWN signals
        # Apply saved config parameters if available, otherwise use defaults
        train_kwargs = {
            'use_hyperparameter_tuning': training_params.get('use_hyperparameter_tuning', True),  # ENABLED: Dynamic tuning for better accuracy
            'use_binary_classification': training_params.get('use_binary_classification', True),
            'use_regime_models': training_params.get('use_regime_models', True),
            'feature_selection_method': training_params.get('feature_selection_method', 'importance'),
            'model_family': training_params.get('model_family', 'voting'),
        }
        
        # Add optional hyperparameters if provided in config
        for param in ['rf_n_estimators', 'rf_max_depth', 'rf_min_samples_split', 'rf_min_samples_leaf',
                      'gb_n_estimators', 'gb_max_depth', 'lgb_reg_alpha', 'lgb_reg_lambda',
                      'lgb_subsample', 'lgb_colsample_bytree', 'lgb_min_child_samples',
                      'xgb_reg_alpha', 'xgb_reg_lambda', 'xgb_subsample', 'xgb_colsample_bytree',
                      'xgb_min_child_weight', 'xgb_gamma']:
            if training_params.get(param) is not None:
                train_kwargs[param] = training_params[param]
        
        # Override with any explicit kwargs passed to the function
        train_kwargs.update(kwargs)
        
        results = ml_model.train(
            all_samples,
            **train_kwargs
        )
        
        # Phase 3 Meta-Labeling: Train second model on signal correctness
        # This helps the hybrid predictor filter out rule violations.
        if 'error' not in results:
            if progress_callback:
                progress_callback("Training Meta-Classifier...")
            
            # Prepare meta-samples: only those where rule_action was non-HOLD and we have a meta_label
            meta_samples = [s for s in all_samples if s.get('rule_action') != 'HOLD' and 'meta_label' in s]
            if len(meta_samples) >= 30:
                # Meta-train expects 'label' key to contain the target (1 or 0)
                # Ensure we don't modify original samples indefinitely by using a shallow copy for the dict
                pipeline_meta_samples = []
                for s in meta_samples:
                    meta_s = s.copy()
                    meta_s['label'] = s.get('meta_label', 0)
                    pipeline_meta_samples.append(meta_s)
                
                ml_model.train_meta(pipeline_meta_samples)
                logger.info(f"Trained Meta-Classifier on {len(pipeline_meta_samples)} rule-based signals.")
            else:
                logger.warning(f"Skipping Meta-Classifier: Insufficient valid signals ({len(meta_samples)}).")
        
        # ===== PHASE 5: HISTORICAL UPSIDE ANALYSIS =====
        if 'error' not in results:
            try:
                if progress_callback:
                    progress_callback("Indexing Historical Patterns...")
                
                # 1. Get Confidence Scores for ALL samples (Batch Prediction)
                feature_vectors = [s['features'] for s in all_samples]
                
                # Predict (Returns probabilities [N, 3])
                probs = ml_model.predict_batch(feature_vectors)
                
                # Find BUY index
                classes = ml_model.label_encoder.classes_
                buy_idx = -1
                if 'BUY' in classes:
                    buy_idx = list(classes).index('BUY')
                elif 'UP' in classes: # Binary mode
                    buy_idx = list(classes).index('UP')
                
                if buy_idx != -1:
                    # Collect patterns
                    patterns_to_save = []
                    for i, sample in enumerate(all_samples):
                        # Only save if we have pattern_features (added in Phase 5)
                        if 'pattern_features' not in sample:
                            continue
                            
                        # Get Confidence for BUY
                        conf = float(probs[i][buy_idx] * 100)
                        
                        # Prepare simplified record
                        pat = {
                            **sample['pattern_features'], # rsi, atr, etc.
                            'confidence': conf, # The confidence the model ASSIGNS to this pattern
                            'max_profit_pct': sample.get('max_profit_pct', 0.0), # The ACTUAL outcome
                            'date': sample.get('date'),
                            'symbol': sample.get('symbol')
                        }
                        patterns_to_save.append(pat)
                    
                    # Save using PatternMatcher
                    pm = PatternMatcher(self.data_dir)
                    pm.save_patterns(patterns_to_save)
                    logger.info(f"Indexed {len(patterns_to_save)} historical patterns for Upside Analysis")
                else:
                    logger.warning("Could not identify BUY class for pattern indexing")
                    
            except Exception as e:
                logger.error(f"Error indexing patterns: {e}")
        
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
            
            message = f"ML Training Complete!\n\n"
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
                
                # --- QUANT MODE LABELING --
                threshold_buy = 3.0
                threshold_sell = -3.0
                
                if IS_QUANT_MODE:
                     # Calculate ATR for dynamic labeling at prediction time
                     # Requirement: We need hist up to pred_date
                     # This will be handled inside the history fetch block below
                     pass # We'll set labels after fetching hist
                
                # Determine correct label based on actual outcome and strategy
                # STRICT BINARY LOGIC: Only BUY or HOLD. No SELL.
                if strategy == "trading":
                    if not IS_QUANT_MODE:
                        if price_change_pct >= 3:
                            label = "BUY"
                        else:
                            label = "HOLD"
                    else:
                        label = None # To be determined after hist/ATR
                elif strategy == "investing":
                    if price_change_pct >= 10:
                        label = "BUY"
                    else:
                        label = "HOLD"
                else:  # mixed
                    if price_change_pct >= 5:
                        label = "BUY"
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
                
                # Assign labels if in Quant Mode (after ATR calculation)
                if IS_QUANT_MODE and strategy == "trading" and label is None:
                    try:
                        # Calculate ATR(14) locally from 'hist'
                        df_tr = hist.copy()
                        df_tr['tr1'] = df_tr['High'] - df_tr['Low']
                        df_tr['tr2'] = abs(df_tr['High'] - df_tr['Close'].shift())
                        df_tr['tr3'] = abs(df_tr['Low'] - df_tr['Close'].shift())
                        df_tr['tr'] = df_tr[['tr1', 'tr2', 'tr3']].max(axis=1)
                        atr = df_tr['tr'].rolling(14).mean().iloc[-1]
                        
                        # Dynamic Threshold: 2.0 * ATR
                        atr_pct = (atr / current_price) * 100
                        t_buy = max(2.0 * atr_pct, 1.5)
                        t_sell = min(-2.0 * atr_pct, -1.5)
                        
                        if price_change_pct >= t_buy:
                            label = "BUY"
                        else:
                            label = "HOLD"
                    except Exception as e:
                        # Fallback to 3% if ATR fails
                        label = "BUY" if price_change_pct >= 3 else "HOLD"
                
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
                
                # Calculate volume analysis for liquidity features
                volume_analysis = None
                if history_data and history_data.get('data') and len(history_data['data']) >= 20:
                    try:
                        hist_df = pd.DataFrame(history_data['data'])
                        if 'close' in hist_df.columns and 'volume' in hist_df.columns:
                            avg_volume = hist_df['volume'].tail(20).mean()
                            avg_price = hist_df['close'].tail(20).mean()
                            avg_dollar_volume = avg_volume * avg_price
                            current_volume = hist_df['volume'].iloc[-1] if len(hist_df) > 0 else 0
                            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
                            
                            if avg_dollar_volume >= 50_000_000:
                                liquidity_grade = 'high'
                            elif avg_dollar_volume >= 10_000_000:
                                liquidity_grade = 'medium'
                            elif avg_dollar_volume >= 1_000_000:
                                liquidity_grade = 'low'
                            else:
                                liquidity_grade = 'very_low'
                            
                            volume_analysis = {
                                'volume_ratio': float(volume_ratio),
                                'avg_dollar_volume': float(avg_dollar_volume),
                                'liquidity_grade': liquidity_grade
                            }
                    except:
                        pass
                
                # Fetch macro analysis data
                macro_analysis = None
                if hasattr(self.data_generator, 'macro_detector'):
                    try:
                        # CRITICAL SYNC FIX: Use pred_date, NOT hist.index[-1]
                        # This prevents "Future Snoop" where 2018 samples see 2026 macro levels.
                        macro_slice = self.data_generator.macro_history[self.data_generator.macro_history.index <= pred_date]
                        if len(macro_slice) >= 50:
                            data_dict = {
                                'VIX': macro_slice['VIX'],
                                'TNX': macro_slice['TNX'],
                                'DXY': macro_slice['DXY']
                            }
                            macro_analysis = self.data_generator.macro_detector.analyze_regime_from_data(data_dict)
                    except Exception as e:
                        logger.debug(f"Error fetching macro analysis for {symbol} at {pred_date}: {e}")

                # Initialize other potential data sources to None if not fetched
                financials_data = None
                sector_analysis = None
                catalyst_info = None
                
                try:
                    # Extract features with enhanced analysis data (Date-Aware)
                    features = self.feature_extractor.extract_features(
                        stock_data, history_data, financials_data, indicators,
                        market_regime, timeframe_analysis, relative_strength, support_resistance,
                        None, sector_analysis, catalyst_info, volume_analysis, macro_analysis,
                        as_of_date=pred_date
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
                        'was_correct': pred.get('was_correct', False),
                        'sample_weight': 10.0  # Verified samples = 10x more important!
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
                    training_result = ml_model.train(
                        training_samples, 
                        test_size=0.2,
                        use_binary_classification=True,  # Use binary mode for better accuracy
                        model_family='voting'
                    )
                    
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

