"""
Configuration file for Stocker App
Contains API endpoints, security settings, and app configuration
"""
import os
import shutil
import logging
from pathlib import Path

# App Configuration
APP_NAME = "Stocker"
APP_VERSION = "1.0.0"
APP_DATA_DIR = Path.home() / ".stocker"

# --- Auto-seed pretrained models on first run ---
# If no ML model exists in APP_DATA_DIR, copy bundled pretrained models
def _seed_pretrained_models():
    """Copy pretrained models from repo into user data dir on first run."""
    pretrained_dir = Path(__file__).parent / "pretrained_models"
    if not pretrained_dir.exists():
        return  # No bundled models (e.g., dev environment without them)
    
    # Check if any model already exists (don't overwrite user-retrained models)
    existing_models = list(APP_DATA_DIR.glob("ml_model_*.pkl"))
    if existing_models:
        return  # User already has models, skip seeding
    
    # Create data dir if needed
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    _logger = logging.getLogger(__name__)
    _logger.info("First run detected — seeding pretrained ML models...")
    
    seeded = 0
    for src_file in pretrained_dir.iterdir():
        if src_file.suffix in ('.pkl', '.json'):
            dst_file = APP_DATA_DIR / src_file.name
            if not dst_file.exists():
                shutil.copy2(src_file, dst_file)
                seeded += 1
    
    if seeded > 0:
        _logger.info(f"Seeded {seeded} pretrained model files into {APP_DATA_DIR}")

_seed_pretrained_models()

# Centralized File Paths (Decoupling)
PREDICTIONS_FILE = APP_DATA_DIR / "predictions.json"
STATE_FILE = APP_DATA_DIR / "paper_state.json"
PORTFOLIO_FILE = APP_DATA_DIR / "paper_portfolio.json"
BACKTEST_RESULTS_FILE = APP_DATA_DIR / "backtest_results.json"

# Backend API Configuration
# In production, this should point to your secure backend proxy
BACKEND_API_URL = os.getenv("STOCKER_API_URL", "http://localhost:5000/api")

# Security Configuration
ENFORCE_HTTPS = True  # Blocks external HTTP, allows localhost HTTP
ALLOW_LOCALHOST_HTTP = True  # Allow HTTP for localhost (development)
MIN_TLS_VERSION = "1.2"
CERTIFICATE_PINNING_ENABLED = True

# Data Validation
MIN_PRICE = 0.01
MAX_PRICE = 1000000
MIN_VOLUME = 0
MAX_VOLUME = 1e15

# Cache Configuration
CACHE_ENABLED = True
CACHE_DURATION_SECONDS = 300  # 5 minutes

# Chart Configuration
CHART_DAYS_HISTORY = 252  # 1 year of trading days
CHART_INTERVALS = ["1d", "1wk", "1mo"]

# Analysis Configuration
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# Portfolio Configuration
INITIAL_BALANCE = 350.0

# API Security Configuration
API_REQUEST_TIMEOUT = 30  # seconds
API_MAX_RETRIES = 3
API_RATE_LIMIT = 60  # requests per minute
API_MAX_CONNECTIONS = 10

# Input Validation
MAX_SYMBOL_LENGTH = 10
MAX_REQUEST_SIZE = 1024 * 1024  # 1MB
MAX_STRING_LENGTH = 10000

# Data Retention (days)
AUDIT_LOG_RETENTION_DAYS = 90
RESEARCH_CACHE_RETENTION_DAYS = 30
API_KEY_ROTATION_DAYS = 90

# Secure Logging
REDACT_SENSITIVE_IN_LOGS = True
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Anomaly Detection
ENABLE_ANOMALY_DETECTION = True
ANOMALY_ALERT_THRESHOLD = 10  # alerts before blocking

# Seeker AI Security Configuration
SEEKER_AI_ENABLED = False
SEEKER_AI_RATE_LIMIT = 60  # Calls per minute
SEEKER_AI_CACHE_HOURS = 1
SEEKER_AI_AUDIT_LOGGING = True

# API Key Storage
API_KEYS_ENCRYPTED = True
API_KEYS_FILE_PERMISSIONS = 0o600  # Read/write owner only

# CPU/Performance Configuration
# Set to -1 to use all available CPU cores, or specify a number (e.g., 4 for 4 cores)
# For intensive tasks (ML training, backtesting), using all cores significantly speeds up processing
ML_TRAINING_CPU_CORES = -1  # Use all cores for ML training (-1 = all, or specify number)
BACKTESTING_CPU_CORES = -1  # Use all cores for backtesting (-1 = all, or specify number)
HYPERPARAMETER_TUNING_PARALLEL_TRIALS = None  # None = auto (uses all cores), or specify number
# Note: Setting to -1 uses all available CPU cores, which maximizes speed but uses more resources
# If you want to leave some cores free for other tasks, set to a lower number (e.g., 4 or 6)

# ML Model Configuration - HIGH CONVICTION MODE (3.0% Target)
# Tuned for >60% accuracy on significant price moves
ML_FEATURE_SELECTION_THRESHOLD = 0.02   # Raised from 0.005 to aggressively drop low-importance features
ML_RF_N_ESTIMATORS = 300    
ML_RF_MAX_DEPTH = 4         # Reduced from 5 to prevent memorization
ML_RF_MIN_SAMPLES_SPLIT = 100 
ML_RF_MIN_SAMPLES_LEAF = 80   # Raised from 50 for smoother decision surface
ML_GB_N_ESTIMATORS = 100    # Reduced from 150 to limit boosting rounds
ML_GB_MAX_DEPTH = 2         # Reduced from 3 for shallower trees
ML_GB_LEARNING_RATE = 0.05  
ML_HIGH_ACCURACY_THRESHOLD = 0.38   # Lowered from 0.60 (Base rate is ~33% without SMOTE)
ML_VERY_HIGH_ACCURACY_THRESHOLD = 0.42  # Lowered from 0.65
ML_MIN_TRAINING_SAMPLES = 50 
ML_MIN_BINARY_SAMPLES = 30   
ML_PERFORMANCE_HISTORY_LIMIT = 1000  
ML_BINARY_HOLD_THRESHOLD = 0.38  # Lowered from 0.60
ML_RECENT_PERFORMANCE_WINDOW = 50  
ML_ENSEMBLE_WEIGHT_UPDATE_THRESHOLD = 5  
ML_PERFORMANCE_ACCURACY_DIFF_THRESHOLD = 10  
ML_ENSEMBLE_WEIGHT_MAX = 0.80  
ML_ENSEMBLE_WEIGHT_MIN = 0.20  
ML_CONFIDENCE_DIFF_THRESHOLD_HIGH = 25  
ML_CONFIDENCE_DIFF_THRESHOLD_NORMAL = 15  
ML_RULE_CONFIDENCE_ADVANTAGE_THRESHOLD = 15  
ML_LABEL_THRESHOLD = 3.0  # Strict 3% hurdle for Success (Legacy)
MIN_PROFIT_TARGET_PCT = 3.0  # Minimum required profit for verification (Legacy)
MIN_STOP_LOSS_PCT = 4.0      # Production stop loss aligned with Audit V4

# --- QUANT MODE SETTINGS ---
IS_QUANT_MODE = True

if IS_QUANT_MODE:
    # Quant Mode Overrides: Prioritizes Precision over Recall for maximum accuracy.
    
    # 1. Confidence Thresholds (Drastically Higher)
    # Only trade if model is 75% sure (Standard is 0.38-0.60)
    ML_HIGH_ACCURACY_THRESHOLD = 0.75      
    ML_VERY_HIGH_ACCURACY_THRESHOLD = 0.85 

    # 2. Labeling Logic (Dynamic)
    # Minimum price move to even consider (2%). Overridden by ATR targets in hybrid_predictor.
    ML_LABEL_THRESHOLD = 2.0  

    # 3. Model Parameters (Prevent Overfitting)
    ML_RF_N_ESTIMATORS = 500    
    ML_RF_MAX_DEPTH = 6         # Increased from 4 to allow 62% confidence gate clearance
    ML_RF_MIN_SAMPLES_LEAF = 60 
    ML_GB_LEARNING_RATE = 0.05  
    ML_GB_MAX_DEPTH = 3         # Increased from 2
    ML_GB_N_ESTIMATORS = 300
    
    # 4. Filters
    ENABLE_REGIME_FILTER = True # Block trades in low ADX / High Volatility clusters
    MIN_ATR_PERCENT = 1.5       # Don't trade if stock moves < 1.5% per day (Zombie prevention)

# Legacy / Non-Quant Defaults for non-ML modules
QUANT_ATR_MULTIPLIER = 0.0   # Success if Alpha > 0 (Simplified)
QUANT_MIN_ALPHA = 0.0        # Simplified beats sector ETF requirement
QUANT_ALPHA_FALLBACK_R2 = 0.4 # If stock-sector R2 lower than this, use SPY-alpha
ML_MIN_SIGNAL_CONFIDENCE = 0.62  # Filter out signals with confidence lower than 62%
ML_MODEL_FAMILY = 'stacking'     # Use Empirical Meta-Stacking (RF + GB + LR meta)

# Phase 4: Deployment Safeguards
GLOBAL_STOP_LOSS_PCT = 4.0   # Hard stop at -4%
MAX_CONCURRENT_POSITIONS = 8 # Max 8 active trades
MAX_PORTFOLIO_ALLOCATION = 0.15 # Max 15% of total account value invested
TIME_STOP_DAYS = 15          # Force exit after 15 days
BASE_POSITION_SIZE_PCT = 0.02 # Base position size (2% of equity)

# --- Tiered Risk Management (Config 2 Standard) ---
# Elite: 80-90% Conf -> 3.0% risk (1.5x), Max 3
ELITE_CONF_RANGE = (80.0, 90.0)
ELITE_MAX_POS = 3
ELITE_MULTIPLIER = 1.5

# Standard: 70-80% Conf -> 2.0% risk (1.0x), Max 2 (Only if 0 Elite)
STANDARD_CONF_RANGE = (70.0, 80.0)
STANDARD_MAX_POS = 2
STANDARD_MULTIPLIER = 1.0

# Plateau: 90%+ Conf -> Skip
PLATEAU_THRESHOLD = 90.0
      

# AI Predictor Configuration
AI_PREDICTOR_ENABLED = False  # Disable SeekerAI (not helpful per testing)
AI_PREDICTOR_DEFAULT_WEIGHT = 0.2  # Default weight for AI predictions (20%)
AI_PREDICTOR_MIN_CONFIDENCE = 40.0  # Minimum confidence to use AI prediction
AI_PREDICTOR_SENTIMENT_WEIGHT = 0.4  # Weight of sentiment in AI score calculation
AI_PREDICTOR_REPUTATION_WEIGHT = 0.3  # Weight of reputation in AI score calculation
AI_PREDICTOR_INSIGHT_WEIGHT = 0.3  # Weight of insights in AI score calculation

# LLM AI Predictor Configuration (more useful than SeekerAI)
LLM_AI_PREDICTOR_ENABLED = False  # Enable LLM-based AI predictor
LLM_AI_PREDICTOR_DEFAULT_WEIGHT = 0.25  # Default weight for LLM AI predictions (25%)
LLM_AI_PREDICTOR_MIN_CONFIDENCE = 45.0  # Minimum confidence to use LLM AI prediction
LLM_AI_USE_OPENAI = True  # Use OpenAI GPT if API key available
LLM_AI_MODEL = "gpt-4o-mini"  # OpenAI model to use
LLM_AI_FALLBACK_ENABLED = False  # Use advanced rule-based fallback if no LLM
