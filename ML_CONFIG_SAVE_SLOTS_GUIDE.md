# ML Training Configuration Save Slots

## Overview

You can now **save and load ML training configurations** like "save slots" in a game! This lets you:
- ✅ Save configurations that work well
- ✅ Experiment with different settings
- ✅ Revert to a known good configuration anytime
- ✅ Share configurations between different training runs

## How to Use

### 💾 **Saving a Configuration**

#### Method 1: From the GUI
1. Click **"Train AI Model"** button
2. Click **"💾 Save Current Config"** button
3. Enter a name (e.g., "current_good", "experimental_v1")
4. Enter an optional description
5. Click **"💾 Save"**

#### Method 2: From Command Line
```bash
python save_ml_config.py [name] [description]

# Examples:
python save_ml_config.py current_good "Current working configuration"
python save_ml_config.py experimental_v1 "Testing higher regularization"
python save_ml_config.py baseline "Baseline before changes"
```

### 📂 **Loading a Configuration**

1. Click **"Train AI Model"** button
2. Click **"📂 Load Config"** button
3. Select a configuration from the list
4. Click **"📂 Load"**
5. The configuration will be applied to the **next training run**

### 🗑️ **Deleting a Configuration**

1. Click **"Train AI Model"** → **"📂 Load Config"**
2. Select the configuration to delete
3. Click **"🗑️ Delete"**
4. Confirm deletion

## What Gets Saved

The configuration includes:

### Training Settings
- `test_size`: Test set size (default: 0.2)
- `use_binary_classification`: Binary mode (UP/DOWN only)
- `use_regime_models`: Regime-specific models (bull/bear/sideways)
- `use_hyperparameter_tuning`: Enable hyperparameter tuning
- `use_smote`: Class balancing with SMOTE
- `use_interactions`: Feature interactions
- `use_temporal_split`: Temporal train/test split (prevents data leakage)

### Feature Selection
- `feature_selection_method`: 'importance', 'rfe', 'mutual_info'
- `disable_feature_selection`: Disable feature selection
- `feature_selection_threshold`: Importance threshold

### Model Family
- `model_family`: 'voting' or 'rf'
- `hold_confidence_threshold`: HOLD confidence threshold

### Random Forest Hyperparameters
- `rf_n_estimators`: Number of trees
- `rf_max_depth`: Max tree depth
- `rf_min_samples_split`: Min samples to split
- `rf_min_samples_leaf`: Min samples per leaf

### Gradient Boosting Hyperparameters
- `gb_n_estimators`: Number of boosting rounds
- `gb_max_depth`: Max tree depth
- `gb_learning_rate`: Learning rate

### Regularization (Anti-Overfitting)
- **LightGBM**:
  - `lgb_reg_alpha`: L1 regularization (default: 1.0)
  - `lgb_reg_lambda`: L2 regularization (default: 3.0)
  - `lgb_subsample`: Row sampling (default: 0.6)
  - `lgb_colsample_bytree`: Feature sampling (default: 0.6)
  - `lgb_min_child_samples`: Min samples per leaf (default: 50)

- **XGBoost**:
  - `xgb_reg_alpha`: L1 regularization (default: 1.0)
  - `xgb_reg_lambda`: L2 regularization (default: 3.0)
  - `xgb_subsample`: Row sampling (default: 0.6)
  - `xgb_colsample_bytree`: Feature sampling (default: 0.6)
  - `xgb_min_child_weight`: Min child weight (default: 5)
  - `xgb_gamma`: Minimum loss reduction (default: 0.2)

## Example Workflow

### Scenario: You like the current configuration

1. **Save it**:
   ```
   Name: current_good
   Description: Temporal split + increased regularization - works well!
   ```

2. **Experiment** with different settings:
   - Try higher regularization
   - Try different feature selection
   - Try disabling regime models

3. **If experiment fails**, load `current_good` to revert

4. **If experiment succeeds**, save it as `experimental_v1`

## Configuration Files

Saved configurations are stored in:
```
~/.stocker/ml_configs/[name].json
```

Each file contains:
- All hyperparameters
- Training settings
- Metadata (name, description, saved_at, created_at)

## Tips

### ✅ **Best Practices**:
1. **Save before experimenting** - Always save a working config before trying new settings
2. **Use descriptive names** - "current_good" is better than "config1"
3. **Add descriptions** - Helps you remember what each config does
4. **Save milestones** - Save configs after significant improvements

### 📝 **Naming Conventions**:
- `current_good` - Your current working configuration
- `baseline` - Original/default configuration
- `experimental_v1`, `experimental_v2` - Testing different settings
- `high_regularization` - Configuration with increased regularization
- `no_regime_models` - Configuration without regime-specific models
- `date_based` - Include date if needed: `20260112_current`

## Current Default Configuration

The current default configuration includes:
- ✅ **Temporal split** (prevents data leakage)
- ✅ **Increased regularization** (reduces overfitting)
- ✅ **Binary classification** (UP/DOWN only, better accuracy)
- ✅ **Regime-specific models** (bull/bear/sideways)
- ✅ **SMOTE balancing** (handles class imbalance)
- ✅ **Feature interactions** (improves pattern recognition)

**This is saved as**: `current_good` (if you ran the save command)

## Advanced Usage

### Loading Config Programmatically

```python
from ml_config_manager import MLConfigManager
from config import APP_DATA_DIR

config_manager = MLConfigManager(APP_DATA_DIR)

# Load a config
config = config_manager.load_config("current_good")

# Apply to training
params = config_manager.apply_config_to_training(config)

# Use params in ml_model.train(**params)
```

### Listing All Configs

```python
configs = config_manager.list_configs()
for config in configs:
    print(f"{config['name']}: {config['description']}")
```

## Troubleshooting

### "Configuration not found"
- Check the name spelling
- List all configs to see available names
- Configs are saved in `~/.stocker/ml_configs/`

### "Configuration didn't apply"
- Make sure you loaded it **before** starting training
- The config is applied to the **next** training run
- Check logs to see if config was used

### "Can't delete configuration"
- Make sure the app isn't using it
- Check file permissions
- Try closing and reopening the app

## Summary

**Save slots** let you:
- ✅ Experiment safely (always have a backup)
- ✅ Track what works (save successful configs)
- ✅ Revert easily (load previous config)
- ✅ Share settings (config files are portable)

**Your current good configuration is saved!** 🎉

Use it anytime by loading `current_good` from the "Load Config" button.
