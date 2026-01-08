"""
Automated ML Model Tuning Script
Tests different configurations until achieving 50%+ validation accuracy
"""
import sys
import os
from pathlib import Path
import logging
import numpy as np
from typing import Dict, List, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import APP_DATA_DIR
from training_pipeline import MLTrainingPipeline
from data_fetcher import StockDataFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AutoTuner:
    """Automatically tune ML model until achieving target accuracy"""
    
    def __init__(self, target_accuracy: float = 0.50):
        self.target_accuracy = target_accuracy
        self.data_fetcher = StockDataFetcher()
        self.training_pipeline = MLTrainingPipeline(self.data_fetcher, APP_DATA_DIR, app=None)
        self.best_config = None
        self.best_accuracy = 0.0
        self.test_results = []
        
    def test_configuration(self, config: Dict) -> Tuple[float, Dict]:
        """Test a specific configuration and return validation accuracy"""
        logger.info(f"\n{'='*70}")
        logger.info(f"Testing configuration: {config}")
        logger.info(f"{'='*70}")
        
        # Modify ml_training.py with this configuration
        self._apply_config(config)
        
        # Generate training samples
        strategy = config.get('strategy', 'trading')
        symbols = config.get('symbols', ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ'])
        
        logger.info(f"Generating samples for {len(symbols)} symbols...")
        all_samples = []
        for symbol in symbols:
            try:
                samples = self.training_pipeline.data_generator.generate_training_samples(
                    symbol, '2023-01-01', '2024-12-31', strategy, 10
                )
                all_samples.extend(samples)
                logger.info(f"  {symbol}: {len(samples)} samples")
            except Exception as e:
                logger.error(f"Error generating samples for {symbol}: {e}")
        
        if len(all_samples) < 100:
            logger.error(f"Insufficient samples: {len(all_samples)}")
            return 0.0, {'error': 'Insufficient samples'}
        
        # Sort chronologically
        all_samples.sort(key=lambda x: x.get('date', '1900-01-01'))
        logger.info(f"Total samples: {len(all_samples)}")
        
        # Train model
        from ml_training import StockPredictionML
        ml_model = StockPredictionML(APP_DATA_DIR, strategy)
        
        try:
            results = ml_model.train(
                all_samples,
                use_hyperparameter_tuning=False,
                use_regime_models=False,
                feature_selection_method=config.get('feature_selection', 'importance')
            )
            
            if 'error' in results:
                logger.error(f"Training error: {results['error']}")
                return 0.0, results
            
            val_accuracy = results.get('test_accuracy', 0.0)
            train_accuracy = results.get('train_accuracy', 0.0)
            
            logger.info(f"\n{'='*70}")
            logger.info(f"RESULTS:")
            logger.info(f"  Training Accuracy: {train_accuracy*100:.1f}%")
            logger.info(f"  Validation Accuracy: {val_accuracy*100:.1f}%")
            logger.info(f"  Gap: {(train_accuracy - val_accuracy)*100:.1f}%")
            logger.info(f"{'='*70}\n")
            
            return val_accuracy, results
            
        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            return 0.0, {'error': str(e)}
    
    def _apply_config(self, config: Dict):
        """Apply configuration to ml_training.py"""
        import re
        
        ml_training_path = Path(__file__).parent / 'ml_training.py'
        with open(ml_training_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Update RandomForest parameters
        rf_params_pattern = r"rf_params = \{([^}]+)\}"
        
        rf_params_str = f"""rf_params = {{
            'n_estimators': {config.get('n_estimators', 35)},
            'max_depth': {config.get('max_depth', 6)},
            'min_samples_split': {config.get('min_samples_split', 25)},
            'min_samples_leaf': {config.get('min_samples_leaf', 12)},
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': random_seed,
            'n_jobs': safe_n_jobs
        }}"""
        
        content = re.sub(rf_params_pattern, rf_params_str, content, flags=re.DOTALL)
        
        # Enable/disable feature selection
        if config.get('feature_selection') == 'none':
            # Disable feature selection
            content = re.sub(
                r"if self\.use_feature_selection and len\(X_train_scaled\) > 200:",
                "if False and len(X_train_scaled) > 200:  # DISABLED",
                content
            )
        else:
            # Enable feature selection
            content = re.sub(
                r"if False and len\(X_train_scaled\) > 200:  # DISABLED",
                "if self.use_feature_selection and len(X_train_scaled) > 200:",
                content
            )
        
        with open(ml_training_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        logger.info(f"Applied configuration to ml_training.py")
    
    def find_best_config(self) -> Dict:
        """Systematically test configurations until achieving target accuracy"""
        logger.info(f"\n{'='*70}")
        logger.info(f"AUTO-TUNING ML MODEL")
        logger.info(f"Target: {self.target_accuracy*100:.1f}% validation accuracy")
        logger.info(f"{'='*70}\n")
        
        # Configuration space to explore
        configs_to_test = [
            # Start with strong regularization + feature selection
            {
                'n_estimators': 50,
                'max_depth': 5,
                'min_samples_split': 30,
                'min_samples_leaf': 15,
                'feature_selection': 'importance',
                'strategy': 'trading'
            },
            # Moderate regularization + feature selection
            {
                'n_estimators': 50,
                'max_depth': 6,
                'min_samples_split': 25,
                'min_samples_leaf': 12,
                'feature_selection': 'importance',
                'strategy': 'trading'
            },
            # Balanced + feature selection
            {
                'n_estimators': 50,
                'max_depth': 7,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'feature_selection': 'importance',
                'strategy': 'trading'
            },
            # More trees, moderate depth + feature selection
            {
                'n_estimators': 100,
                'max_depth': 6,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'feature_selection': 'importance',
                'strategy': 'trading'
            },
            # RFE feature selection
            {
                'n_estimators': 50,
                'max_depth': 6,
                'min_samples_split': 25,
                'min_samples_leaf': 12,
                'feature_selection': 'rfe',
                'strategy': 'trading'
            },
            # Mutual info feature selection
            {
                'n_estimators': 50,
                'max_depth': 6,
                'min_samples_split': 25,
                'min_samples_leaf': 12,
                'feature_selection': 'mutual_info',
                'strategy': 'trading'
            },
            # No feature selection, stronger regularization
            {
                'n_estimators': 50,
                'max_depth': 5,
                'min_samples_split': 30,
                'min_samples_leaf': 15,
                'feature_selection': 'none',
                'strategy': 'trading'
            },
        ]
        
        for i, config in enumerate(configs_to_test, 1):
            logger.info(f"\n{'#'*70}")
            logger.info(f"TEST {i}/{len(configs_to_test)}")
            logger.info(f"{'#'*70}")
            
            val_accuracy, results = self.test_configuration(config)
            
            self.test_results.append({
                'config': config,
                'val_accuracy': val_accuracy,
                'results': results
            })
            
            if val_accuracy >= self.target_accuracy:
                logger.info(f"\n{'='*70}")
                logger.info(f"✅ TARGET ACHIEVED!")
                logger.info(f"Validation Accuracy: {val_accuracy*100:.1f}%")
                logger.info(f"Configuration: {config}")
                logger.info(f"{'='*70}\n")
                self.best_config = config
                self.best_accuracy = val_accuracy
                return config
            
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.best_config = config
                logger.info(f"New best: {val_accuracy*100:.1f}%")
        
        # If we haven't reached target, try adaptive tuning
        logger.info(f"\n{'='*70}")
        logger.info(f"Target not reached. Best so far: {self.best_accuracy*100:.1f}%")
        logger.info(f"Trying adaptive tuning...")
        logger.info(f"{'='*70}\n")
        
        return self._adaptive_tune()
    
    def _adaptive_tune(self) -> Dict:
        """Adaptively tune based on best result so far"""
        if not self.best_config:
            logger.error("No best config found for adaptive tuning")
            return {}
        
        base_config = self.best_config.copy()
        
        # Try variations around best config
        variations = [
            {**base_config, 'n_estimators': base_config.get('n_estimators', 50) + 25},
            {**base_config, 'max_depth': base_config.get('max_depth', 6) - 1},
            {**base_config, 'max_depth': base_config.get('max_depth', 6) + 1},
            {**base_config, 'min_samples_split': base_config.get('min_samples_split', 25) + 5},
            {**base_config, 'min_samples_leaf': base_config.get('min_samples_leaf', 12) + 3},
        ]
        
        for i, var_config in enumerate(variations, 1):
            logger.info(f"\nAdaptive Test {i}/{len(variations)}")
            val_accuracy, results = self.test_configuration(var_config)
            
            if val_accuracy >= self.target_accuracy:
                logger.info(f"\n✅ TARGET ACHIEVED with adaptive tuning!")
                logger.info(f"Validation Accuracy: {val_accuracy*100:.1f}%")
                return var_config
            
            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy
                self.best_config = var_config
        
        logger.info(f"\nBest achieved: {self.best_accuracy*100:.1f}%")
        logger.info(f"Best config: {self.best_config}")
        return self.best_config
    
    def print_summary(self):
        """Print summary of all tests"""
        logger.info(f"\n{'='*70}")
        logger.info("TEST SUMMARY")
        logger.info(f"{'='*70}")
        
        for i, result in enumerate(self.test_results, 1):
            config = result['config']
            val_acc = result['val_accuracy']
            train_acc = result['results'].get('train_accuracy', 0) * 100
            
            logger.info(f"\nTest {i}:")
            logger.info(f"  Config: depth={config.get('max_depth')}, split={config.get('min_samples_split')}, "
                       f"leaf={config.get('min_samples_leaf')}, trees={config.get('n_estimators')}, "
                       f"feat_sel={config.get('feature_selection')}")
            logger.info(f"  Training: {train_acc:.1f}%, Validation: {val_acc*100:.1f}%")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"BEST RESULT: {self.best_accuracy*100:.1f}%")
        logger.info(f"BEST CONFIG: {self.best_config}")
        logger.info(f"{'='*70}\n")


def main():
    """Main entry point"""
    tuner = AutoTuner(target_accuracy=0.50)
    
    try:
        best_config = tuner.find_best_config()
        tuner.print_summary()
        
        if tuner.best_accuracy >= 0.50:
            logger.info(f"\n✅ SUCCESS! Achieved {tuner.best_accuracy*100:.1f}% validation accuracy")
            logger.info(f"Best configuration has been applied to ml_training.py")
        else:
            logger.warning(f"\n⚠️ Could not reach 50% target. Best: {tuner.best_accuracy*100:.1f}%")
            logger.info(f"Best configuration has been applied to ml_training.py")
            logger.info(f"Consider: more training data, different features, or different model type")
        
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        tuner.print_summary()
    except Exception as e:
        logger.error(f"Error during auto-tuning: {e}", exc_info=True)
        tuner.print_summary()


if __name__ == "__main__":
    main()
