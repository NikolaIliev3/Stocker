"""
Sector-Specific ML System
Trains specialized ML models per sector (Tech, Financial, Healthcare, etc.)
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Default sector stock lists - diversified within each sector
SECTOR_STOCKS = {
    'technology': [
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'META', 'ADBE', 'CRM', 'ORCL', 'INTC',
        'CSCO', 'IBM', 'NOW', 'QCOM', 'TXN', 'AVGO', 'MU', 'AMAT', 'LRCX', 'KLAC'
    ],
    'financial': [
        'JPM', 'BAC', 'GS', 'MS', 'WFC', 'C', 'BLK', 'SCHW', 'AXP', 'V',
        'MA', 'COF', 'USB', 'PNC', 'TFC', 'BK', 'STT', 'SPGI', 'ICE', 'CME'
    ],
    'healthcare': [
        'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY', 'BMY', 'AMGN',
        'GILD', 'CVS', 'CI', 'MDT', 'ISRG', 'SYK', 'BDX', 'ZTS', 'VRTX', 'REGN'
    ],
    'consumer': [
        'AMZN', 'WMT', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'COST', 'LOW', 'PG',
        'KO', 'PEP', 'PM', 'MO', 'CL', 'EL', 'KMB', 'GIS', 'K', 'HSY'
    ],
    'energy': [
        'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'HAL',
        'DVN', 'HES', 'PXD', 'FANG', 'BKR', 'KMI', 'WMB', 'OKE', 'TRGP', 'LNG'
    ],
    'industrials': [
        'CAT', 'DE', 'BA', 'UPS', 'FDX', 'GE', 'HON', 'LMT', 'RTX', 'MMM',
        'UNP', 'CSX', 'NSC', 'WM', 'RSG', 'EMR', 'ETN', 'ITW', 'PH', 'ROK'
    ],
    'materials': [
        'LIN', 'APD', 'SHW', 'ECL', 'NEM', 'FCX', 'NUE', 'VMC', 'MLM', 'DD',
        'DOW', 'PPG', 'ALB', 'CTVA', 'CF', 'MOS', 'IFF', 'CE', 'EMN', 'SEE'
    ],
    'utilities': [
        'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC', 'SRE', 'XEL', 'WEC', 'ED',
        'ES', 'DTE', 'EIX', 'FE', 'PPL', 'AEE', 'CMS', 'CNP', 'NI', 'EVRG'
    ],
    'real_estate': [
        'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'O', 'WELL', 'AVB', 'DLR',
        'ARE', 'EQR', 'VTR', 'SUI', 'EXR', 'MAA', 'UDR', 'ESS', 'HST', 'PEAK'
    ],
    'communication': [
        'NFLX', 'DIS', 'CMCSA', 'T', 'VZ', 'TMUS', 'CHTR', 'EA', 'TTWO', 'WBD',
        'PARA', 'FOX', 'OMC', 'IPG', 'LYV', 'MTCH', 'ZG', 'PINS', 'SNAP', 'RBLX'
    ]
}

SECTOR_DISPLAY_NAMES = {
    'technology': '💻 Technology',
    'financial': '🏦 Financial',
    'healthcare': '🏥 Healthcare',
    'consumer': '🛒 Consumer',
    'energy': '⚡ Energy',
    'industrials': '🏭 Industrials',
    'materials': '🧱 Materials',
    'utilities': '💡 Utilities',
    'real_estate': '🏠 Real Estate',
    'communication': '📡 Communication'
}


class SectorML:
    """
    ML model trained on a specific sector for better pattern recognition.
    
    Sectors share common characteristics (all tech stocks react to rate changes similarly),
    making sector-specific models more powerful than single-stock models.
    """
    
    DEFAULT_STOCK_COUNT = 10
    
    def __init__(self, data_dir: Path, sector: str, strategy: str = 'trading'):
        """
        Initialize Sector ML model.
        
        Args:
            data_dir: Directory for storing model files
            sector: Sector name (technology, financial, etc.)
            strategy: Trading strategy (trading/mixed/investing)
        """
        self.data_dir = Path(data_dir)
        self.sector = sector.lower()
        self.strategy = strategy
        
        # File naming: ml_model_sector_technology_trading.pkl
        self.model_file = self.data_dir / f"ml_model_sector_{self.sector}_{strategy}.pkl"
        self.metadata_file = self.data_dir / f"ml_metadata_sector_{self.sector}_{strategy}.json"
        self.scaler_file = self.data_dir / f"scaler_sector_{self.sector}_{strategy}.pkl"
        self.encoder_file = self.data_dir / f"label_encoder_sector_{self.sector}_{strategy}.pkl"
        
        # Internal state
        self._ml_model = None
        self._scaler = None
        self._label_encoder = None
        self._metadata = {}
        self._is_trained = False
        self._selected_features = None
        
        # Try to load existing model
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load existing sector model if available."""
        try:
            if self.model_file.exists() and self.metadata_file.exists():
                from secure_ml_storage import SecureMLStorage
                
                storage = SecureMLStorage(self.data_dir)
                
                self._ml_model = storage.load_model(self.model_file.name)
                
                if self.scaler_file.exists():
                    self._scaler = storage.load_model(self.scaler_file.name)
                
                if self.encoder_file.exists():
                    self._label_encoder = storage.load_model(self.encoder_file.name)
                
                with open(self.metadata_file, 'r') as f:
                    self._metadata = json.load(f)
                
                # Load and convert selected_features to numpy array
                selected = self._metadata.get('selected_features')
                if selected is not None:
                    import numpy as np
                    self._selected_features = np.array(selected)
                else:
                    self._selected_features = None
                    
                self._is_trained = True
                
                sf_count = len(self._selected_features) if self._selected_features is not None else 0
                logger.info(f"✅ Loaded {SECTOR_DISPLAY_NAMES.get(self.sector, self.sector)} model: {self._metadata.get('test_accuracy', 0)*100:.1f}% accuracy ({sf_count} selected features)")
                return True
                
        except Exception as e:
            logger.warning(f"Could not load sector model for {self.sector}: {e}")
        
        return False
    
    def is_trained(self) -> bool:
        """Check if sector model is trained."""
        return self._is_trained and self._ml_model is not None
    
    def get_accuracy(self) -> float:
        """Get the test accuracy of this sector model."""
        return self._metadata.get('test_accuracy', 0.0)
    
    def get_training_date(self) -> Optional[str]:
        """Get when this sector model was last trained."""
        return self._metadata.get('trained_date')
    
    def get_sample_count(self) -> int:
        """Get number of samples used for training."""
        return self._metadata.get('training_samples', 0)
    
    def get_stocks_used(self) -> List[str]:
        """Get list of stocks used for training."""
        return self._metadata.get('stocks_used', [])
    
    def train(self, data_fetcher, start_date: str, end_date: str,
              stock_count: int = None, custom_stocks: List[str] = None,
              progress_callback=None) -> Dict:
        """
        Train sector model on multiple stocks from the sector.
        
        Args:
            data_fetcher: StockDataFetcher instance
            start_date: Training start date (YYYY-MM-DD)
            end_date: Training end date (YYYY-MM-DD)
            stock_count: Number of stocks to use (from default list)
            custom_stocks: Custom stock list (overrides default)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dict with training results
        """
        from ml_training import StockPredictionML, FeatureExtractor
        from training_pipeline import TrainingDataGenerator
        
        # Get stocks for training
        if custom_stocks:
            stocks = custom_stocks
        else:
            count = stock_count or self.DEFAULT_STOCK_COUNT
            default_stocks = SECTOR_STOCKS.get(self.sector, [])
            stocks = default_stocks[:count]
        
        if not stocks:
            return {'error': f'No stocks defined for sector: {self.sector}'}
        
        sector_name = SECTOR_DISPLAY_NAMES.get(self.sector, self.sector)
        logger.info(f"🏭 Training {sector_name} model with {len(stocks)} stocks...")
        
        try:
            feature_extractor = FeatureExtractor(data_fetcher)
            generator = TrainingDataGenerator(data_fetcher, feature_extractor)
            
            # Get lookforward days based on strategy
            lookforward_days = {
                'trading': 10,
                'mixed': 21,
                'investing': 63
            }.get(self.strategy, 10)
            
            # Generate samples from all sector stocks
            all_samples = []
            for i, symbol in enumerate(stocks):
                if progress_callback:
                    progress_callback(f"Generating samples for {symbol} ({i+1}/{len(stocks)})...")
                
                try:
                    samples = generator.generate_training_samples(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        strategy=self.strategy,
                        lookforward_days=lookforward_days
                    )
                    if samples:
                        all_samples.extend(samples)
                        logger.info(f"  {symbol}: {len(samples)} samples")
                except Exception as e:
                    logger.warning(f"  {symbol}: Failed - {e}")
            
            if len(all_samples) < 500:
                return {
                    'error': f'Insufficient training data for {sector_name}: {len(all_samples)} samples (need 500+)'
                }
            
            if progress_callback:
                progress_callback(f"Training model on {len(all_samples)} samples...")
            
            # Sector models now use SAME settings as main model for consistency
            # Key settings (matching training_pipeline.py):
            # - Hyperparameter tuning ON (Optuna optimization)
            # - Voting ensemble (RF + LightGBM + LogReg)
            # - Regime models ON (bull/bear/sideways)
            # - 20% test size (same as main)
            temp_ml = StockPredictionML(self.data_dir, f"sector_{self.sector}_{self.strategy}")
            result = temp_ml.train(
                training_samples=all_samples,
                test_size=0.2,  # Same as main model
                use_hyperparameter_tuning=True,  # ENABLED for maximum accuracy
                use_regime_models=True,  # ENABLED - same as main model
                use_binary_classification=True,
                use_smote=True,
                feature_selection_method='importance',
                model_family='voting',  # VOTING ensemble - same as main model
            )
            
            if 'error' in result:
                return result
            
            new_accuracy = result.get('test_accuracy', 0)
            old_accuracy = self.get_accuracy()
            
            # Only save if new model is better or no previous model
            if new_accuracy >= old_accuracy or old_accuracy == 0:
                self._ml_model = temp_ml.model
                self._scaler = temp_ml.scaler
                self._label_encoder = temp_ml.label_encoder
                self._selected_features = getattr(temp_ml, 'selected_features', None)
                
                self._save_model(result, len(all_samples), stocks)
                self._is_trained = True
                
                if old_accuracy > 0 and new_accuracy > old_accuracy:
                    logger.info(f"✅ {sector_name} IMPROVED: {old_accuracy*100:.1f}% → {new_accuracy*100:.1f}%")
                else:
                    logger.info(f"✅ {sector_name} trained: {new_accuracy*100:.1f}% accuracy")
                
                return {
                    'success': True,
                    'sector': self.sector,
                    'sector_display': sector_name,
                    'strategy': self.strategy,
                    'stocks_used': stocks,
                    'training_samples': len(all_samples),
                    'training_accuracy': result.get('training_accuracy', 0),
                    'test_accuracy': new_accuracy,
                    'previous_accuracy': old_accuracy if old_accuracy > 0 else None,
                    'cross_validation_accuracy': result.get('cv_accuracy', 0),
                    'cv_std': result.get('cv_std', 0),
                    'trained_date': datetime.now().isoformat(),
                    'improved': new_accuracy > old_accuracy and old_accuracy > 0
                }
            else:
                logger.warning(f"⚠️ {sector_name}: New ({new_accuracy*100:.1f}%) worse than existing ({old_accuracy*100:.1f}%). Keeping old.")
                return {
                    'success': False,
                    'sector': self.sector,
                    'new_accuracy': new_accuracy,
                    'existing_accuracy': old_accuracy,
                    'kept_existing': True,
                    'message': f"New model ({new_accuracy*100:.1f}%) performs worse than existing ({old_accuracy*100:.1f}%). Old model preserved."
                }
            
        except Exception as e:
            logger.error(f"Error training sector model for {self.sector}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'error': str(e)}
    
    def _save_model(self, training_result: Dict, sample_count: int, stocks_used: List[str]):
        """Save trained sector model to disk."""
        from secure_ml_storage import SecureMLStorage
        import numpy as np
        
        storage = SecureMLStorage(self.data_dir)
        
        if self._ml_model:
            storage.save_model(self._ml_model, self.model_file.name)
        if self._scaler:
            storage.save_model(self._scaler, self.scaler_file.name)
        if self._label_encoder:
            storage.save_model(self._label_encoder, self.encoder_file.name)
        
        # Convert numpy arrays to lists for JSON serialization
        selected_features = self._selected_features
        if selected_features is not None:
            if isinstance(selected_features, np.ndarray):
                selected_features = selected_features.tolist()
            elif hasattr(selected_features, 'tolist'):
                selected_features = selected_features.tolist()
        
        self._metadata = {
            'sector': self.sector,
            'strategy': self.strategy,
            'stocks_used': stocks_used,
            'training_samples': sample_count,
            'training_accuracy': float(training_result.get('training_accuracy', 0)),
            'test_accuracy': float(training_result.get('test_accuracy', 0)),
            'cv_accuracy': float(training_result.get('cv_accuracy', 0)),
            'cv_std': float(training_result.get('cv_std', 0)),
            'trained_date': datetime.now().isoformat(),
            'selected_features': selected_features
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(self._metadata, f, indent=2)
        
        logger.info(f"💾 Saved {SECTOR_DISPLAY_NAMES.get(self.sector, self.sector)} model")
    
    def predict(self, features, general_model=None) -> Dict:
        """
        Make prediction using sector model.
        
        Args:
            features: Feature vector for prediction
            general_model: General model to compare confidence with
            
        Returns:
            Dict with prediction and metadata
        """
        import numpy as np
        
        result = {
            'sector': self.sector,
            'sector_display': SECTOR_DISPLAY_NAMES.get(self.sector, self.sector),
            'used_sector_model': False,
            'fallback_reason': None,
            'action': 'HOLD',
            'confidence': 0.0,
            'sector_accuracy': self.get_accuracy()
        }
        
        if not self.is_trained():
            result['fallback_reason'] = f"No trained model for {self.sector}"
            return result
        
        try:
            # Apply feature selection if it was used during training
            features_to_use = features.reshape(1, -1)
            
            if self._selected_features is not None and len(self._selected_features) > 0:
                # Get indices of selected features
                selected_indices = self._selected_features
                if isinstance(selected_indices, list):
                    selected_indices = np.array(selected_indices)
                
                # Ensure we don't exceed feature bounds
                max_feature_idx = features_to_use.shape[1] - 1
                valid_indices = selected_indices[selected_indices <= max_feature_idx]
                
                if len(valid_indices) > 0:
                    features_to_use = features_to_use[:, valid_indices]
                    logger.debug(f"Applied feature selection: {features.shape[0]} -> {features_to_use.shape[1]} features")
            
            # Scale features
            if self._scaler:
                try:
                    features_scaled = self._scaler.transform(features_to_use)
                except ValueError as e:
                    # Feature count mismatch - fallback to general model
                    result['fallback_reason'] = f"Feature mismatch: {str(e)}"
                    logger.warning(f"Sector {self.sector} feature mismatch: {e}")
                    return result
            else:
                features_scaled = features_to_use
            
            # Get prediction
            prediction = self._ml_model.predict(features_scaled)[0]
            
            if hasattr(self._ml_model, 'predict_proba'):
                probas = self._ml_model.predict_proba(features_scaled)[0]
                confidence = float(max(probas))
            else:
                confidence = 0.5
            
            # Decode prediction
            if self._label_encoder:
                action = self._label_encoder.inverse_transform([prediction])[0]
            else:
                action = prediction
            
            # Convert UP/DOWN to BUY/SELL
            if action == 'UP':
                action = 'BUY'
            elif action == 'DOWN':
                action = 'SELL'
            
            result['used_sector_model'] = True
            result['action'] = action
            result['confidence'] = confidence
            
            return result
            
        except Exception as e:
            logger.error(f"Sector prediction error for {self.sector}: {e}")
            result['fallback_reason'] = f"Prediction error: {str(e)}"
            return result
    
    def delete(self) -> bool:
        """Delete this sector model's files."""
        try:
            for file in [self.model_file, self.metadata_file, self.scaler_file, self.encoder_file]:
                if file.exists():
                    file.unlink()
            self._is_trained = False
            self._ml_model = None
            logger.info(f"🗑️ Deleted {SECTOR_DISPLAY_NAMES.get(self.sector, self.sector)} model")
            return True
        except Exception as e:
            logger.error(f"Error deleting sector model for {self.sector}: {e}")
            return False


class SectorMLManager:
    """
    Manages multiple Sector ML models.
    """
    
    def __init__(self, data_dir: Path, strategy: str = 'trading'):
        self.data_dir = Path(data_dir)
        self.strategy = strategy
        self._sectors: Dict[str, SectorML] = {}
        
        self._discover_sectors()
    
    def _discover_sectors(self):
        """Discover all trained sector models."""
        pattern = f"ml_model_sector_*_{self.strategy}.pkl"
        
        for model_file in self.data_dir.glob(pattern):
            name = model_file.stem
            parts = name.split('_')
            if len(parts) >= 4:
                sector = parts[3]
                sector_ml = SectorML(self.data_dir, sector, self.strategy)
                if sector_ml.is_trained():
                    self._sectors[sector] = sector_ml
                    logger.debug(f"Discovered sector model for {sector}")
    
    def has_model(self, sector: str) -> bool:
        """Check if a sector model exists."""
        sector = sector.lower()
        if sector in self._sectors:
            return self._sectors[sector].is_trained()
        
        sector_ml = SectorML(self.data_dir, sector, self.strategy)
        if sector_ml.is_trained():
            self._sectors[sector] = sector_ml
            return True
        return False
    
    def get_model(self, sector: str) -> Optional[SectorML]:
        """Get sector model."""
        sector = sector.lower()
        if self.has_model(sector):
            return self._sectors.get(sector)
        return None
    
    def get_accuracy(self, sector: str) -> float:
        """Get accuracy for a sector model."""
        model = self.get_model(sector)
        return model.get_accuracy() if model else 0.0
    
    def list_models(self) -> List[Dict]:
        """List all available sector models with stats."""
        models = []
        for sector, model in self._sectors.items():
            if model.is_trained():
                models.append({
                    'sector': sector,
                    'display_name': SECTOR_DISPLAY_NAMES.get(sector, sector),
                    'accuracy': model.get_accuracy(),
                    'samples': model.get_sample_count(),
                    'stocks_used': model.get_stocks_used(),
                    'trained_date': model.get_training_date()
                })
        return sorted(models, key=lambda x: x['accuracy'], reverse=True)
    
    def predict(self, sector: str, features, general_model=None) -> Dict:
        """Get prediction from sector model."""
        model = self.get_model(sector)
        if model and model.is_trained():
            return model.predict(features, general_model)
        return {
            'sector': sector,
            'used_sector_model': False,
            'fallback_reason': f"No model for sector: {sector}",
            'action': 'HOLD',
            'confidence': 0.0
        }
    
    def get_available_sectors(self) -> List[str]:
        """Get list of all available sectors (not just trained ones)."""
        return list(SECTOR_STOCKS.keys())
    
    def delete_model(self, sector: str) -> bool:
        """Delete a sector model."""
        sector = sector.lower()
        if sector in self._sectors:
            result = self._sectors[sector].delete()
            if result:
                del self._sectors[sector]
            return result
        return False
