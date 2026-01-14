"""
ML Training Configuration Manager
Save and load ML training configurations like "save slots"
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class MLConfigManager:
    """Manages ML training configuration save slots"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir = data_dir / "ml_configs"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
    
    def get_current_config(self) -> Dict:
        """Extract current ML training configuration from code defaults"""
        from config import (
            ML_FEATURE_SELECTION_THRESHOLD,
            ML_RF_N_ESTIMATORS, ML_RF_MAX_DEPTH,
            ML_RF_MIN_SAMPLES_SPLIT, ML_RF_MIN_SAMPLES_LEAF,
            ML_GB_N_ESTIMATORS, ML_GB_MAX_DEPTH, ML_GB_LEARNING_RATE
        )
        
        # Current configuration (from code defaults and recent fixes)
        config = {
            # Training settings
            'test_size': 0.2,
            'use_binary_classification': True,  # Current default
            'use_regime_models': True,
            'use_hyperparameter_tuning': False,
            'use_smote': True,  # Auto-enabled for binary classification
            'use_interactions': True,
            'use_temporal_split': True,  # Recent fix
            
            # Feature selection
            'feature_selection_method': 'importance',
            'disable_feature_selection': False,
            'feature_selection_threshold': ML_FEATURE_SELECTION_THRESHOLD,
            
            # Model family
            'model_family': 'voting',
            'hold_confidence_threshold': 0.0,
            
            # Random Forest hyperparameters
            'rf_n_estimators': ML_RF_N_ESTIMATORS,
            'rf_max_depth': ML_RF_MAX_DEPTH,
            'rf_min_samples_split': ML_RF_MIN_SAMPLES_SPLIT,
            'rf_min_samples_leaf': ML_RF_MIN_SAMPLES_LEAF,
            
            # Gradient Boosting hyperparameters (LightGBM/XGBoost)
            'gb_n_estimators': ML_GB_N_ESTIMATORS,
            'gb_max_depth': ML_GB_MAX_DEPTH,
            'gb_learning_rate': ML_GB_LEARNING_RATE,
            
            # Regularization (anti-overfitting - recent fixes)
            'lgb_reg_alpha': 1.0,  # Increased from 0.5
            'lgb_reg_lambda': 3.0,  # Increased from 2.0
            'lgb_subsample': 0.6,  # Reduced from 0.7
            'lgb_colsample_bytree': 0.6,  # Reduced from 0.7
            'lgb_min_child_samples': 50,  # Increased from 30
            
            'xgb_reg_alpha': 1.0,  # Increased from 0.5
            'xgb_reg_lambda': 3.0,  # Increased from 2.0
            'xgb_subsample': 0.6,  # Reduced from 0.7
            'xgb_colsample_bytree': 0.6,  # Reduced from 0.7
            'xgb_min_child_weight': 5,  # Increased from 3
            'xgb_gamma': 0.2,  # Increased from 0.1
            
            # Metadata
            'created_at': datetime.now().isoformat(),
            'description': 'Current configuration with temporal split and increased regularization'
        }
        
        return config
    
    def save_config(self, name: str, description: str = "", config: Optional[Dict] = None) -> bool:
        """Save current configuration to a named slot
        
        Args:
            name: Configuration name (e.g., "current_good", "experimental_v1")
            description: Optional description of this configuration
            config: Optional config dict (if None, uses current config)
        
        Returns:
            True if saved successfully
        """
        try:
            if config is None:
                config = self.get_current_config()
            
            # Add metadata
            config['name'] = name
            config['description'] = description or config.get('description', '')
            config['saved_at'] = datetime.now().isoformat()
            
            # Save to file
            filename = f"{name}.json"
            filepath = self.configs_dir / filename
            
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"💾 Saved ML configuration: {name}")
            if description:
                logger.info(f"   Description: {description}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving ML configuration: {e}")
            return False
    
    def load_config(self, name: str) -> Optional[Dict]:
        """Load a saved configuration
        
        Args:
            name: Configuration name
        
        Returns:
            Configuration dict or None if not found
        """
        try:
            filename = f"{name}.json"
            filepath = self.configs_dir / filename
            
            if not filepath.exists():
                logger.warning(f"Configuration '{name}' not found")
                return None
            
            with open(filepath, 'r') as f:
                config = json.load(f)
            
            logger.info(f"📂 Loaded ML configuration: {name}")
            if config.get('description'):
                logger.info(f"   Description: {config['description']}")
            
            return config
            
        except Exception as e:
            logger.error(f"Error loading ML configuration: {e}")
            return None
    
    def list_configs(self) -> List[Dict]:
        """List all saved configurations
        
        Returns:
            List of config metadata (name, description, saved_at)
        """
        configs = []
        
        try:
            for filepath in self.configs_dir.glob("*.json"):
                try:
                    with open(filepath, 'r') as f:
                        config = json.load(f)
                    
                    configs.append({
                        'name': config.get('name', filepath.stem),
                        'description': config.get('description', ''),
                        'saved_at': config.get('saved_at', ''),
                        'created_at': config.get('created_at', '')
                    })
                except Exception as e:
                    logger.debug(f"Error reading config file {filepath}: {e}")
            
            # Sort by saved_at (most recent first)
            configs.sort(key=lambda x: x.get('saved_at', ''), reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing configurations: {e}")
        
        return configs
    
    def delete_config(self, name: str) -> bool:
        """Delete a saved configuration
        
        Args:
            name: Configuration name
        
        Returns:
            True if deleted successfully
        """
        try:
            filename = f"{name}.json"
            filepath = self.configs_dir / filename
            
            if not filepath.exists():
                logger.warning(f"Configuration '{name}' not found")
                return False
            
            filepath.unlink()
            logger.info(f"🗑️ Deleted ML configuration: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting ML configuration: {e}")
            return False
    
    def apply_config_to_training(self, config: Dict) -> Dict:
        """Convert saved config to training parameters
        
        Args:
            config: Saved configuration dict
        
        Returns:
            Dict of parameters ready for ml_model.train()
        """
        # Extract parameters that match train() method signature
        params = {
            'test_size': config.get('test_size', 0.2),
            'use_binary_classification': config.get('use_binary_classification', False),
            'use_regime_models': config.get('use_regime_models', True),
            'use_hyperparameter_tuning': config.get('use_hyperparameter_tuning', False),
            'feature_selection_method': config.get('feature_selection_method', 'importance'),
            'disable_feature_selection': config.get('disable_feature_selection', False),
            'use_smote': config.get('use_smote', None),  # None = auto
            'use_interactions': config.get('use_interactions', True),
            'model_family': config.get('model_family', 'voting'),
            'hold_confidence_threshold': config.get('hold_confidence_threshold', 0.0),
            'rf_n_estimators': config.get('rf_n_estimators'),
            'rf_max_depth': config.get('rf_max_depth'),
            'rf_min_samples_split': config.get('rf_min_samples_split'),
            'rf_min_samples_leaf': config.get('rf_min_samples_leaf'),
            'gb_n_estimators': config.get('gb_n_estimators'),
            'gb_max_depth': config.get('gb_max_depth'),
            # Regularization parameters (directly passable to train method)
            'lgb_reg_alpha': config.get('lgb_reg_alpha'),
            'lgb_reg_lambda': config.get('lgb_reg_lambda'),
            'lgb_subsample': config.get('lgb_subsample'),
            'lgb_colsample_bytree': config.get('lgb_colsample_bytree'),
            'lgb_min_child_samples': config.get('lgb_min_child_samples'),
            'xgb_reg_alpha': config.get('xgb_reg_alpha'),
            'xgb_reg_lambda': config.get('xgb_reg_lambda'),
            'xgb_subsample': config.get('xgb_subsample'),
            'xgb_colsample_bytree': config.get('xgb_colsample_bytree'),
            'xgb_min_child_weight': config.get('xgb_min_child_weight'),
            'xgb_gamma': config.get('xgb_gamma'),
        }
        
        return params
