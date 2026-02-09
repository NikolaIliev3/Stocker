"""
Quant Mode Configuration
Overrides for maximum accuracy. Prioritizes Precision over Recall.
"""
from config import * 

# 1. Confidence Thresholds (Drastically Higher)
ML_HIGH_ACCURACY_THRESHOLD = 0.75      # Only trade if model is 75% sure
ML_VERY_HIGH_ACCURACY_THRESHOLD = 0.85 # Strong conviction

# 2. Labeling Logic (Dynamic)
# We do NOT use fixed 3.0% here. The labeling logic in `hybrid_predictor` will override this 
# with ATR-based targets. But as a fallback floor:
ML_LABEL_THRESHOLD = 2.0  # Minimum move to even consider (2%)

# 3. Model Parameters (Prevent Overfitting)
ML_RF_N_ESTIMATORS = 500  # More trees -> Lower Variance
ML_RF_MIN_SAMPLES_LEAF = 100 # Highly generalized nodes only
ML_GB_LEARNING_RATE = 0.01 # Slower learning -> Better convergence
ML_GB_N_ESTIMATORS = 300

# 4. Filters
ENABLE_REGIME_FILTER = True # Block trades in low ADX / High Volatility clusters
MIN_ATR_PERCENT = 1.5       # Don't trade if stock moves < 1.5% per day (Zombie prevention)
