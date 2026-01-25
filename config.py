"""
Configuration file for Stocker App
Contains API endpoints, security settings, and app configuration
"""
import os
from pathlib import Path

# App Configuration
APP_NAME = "Stocker"
APP_VERSION = "1.0.0"
APP_DATA_DIR = Path.home() / ".stocker"

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
INITIAL_BALANCE = 10000.0

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

# ML Model Configuration - HIGH ACCURACY MODE (Robust)
# Tuned for >60% accuracy while minimizing overfitting gap (<5%)
ML_FEATURE_SELECTION_THRESHOLD = 0.005  # Keep strict threshold
ML_RF_N_ESTIMATORS = 100    # Increased from 50 to capture more patterns (reducing variance)
ML_RF_MAX_DEPTH = 6         # Increased from 4 to 6 (Nuance without memorization)
ML_RF_MIN_SAMPLES_SPLIT = 20  # Slightly lower to allow learning specific patterns
ML_RF_MIN_SAMPLES_LEAF = 8    # Strict leaf size (must have 8 samples to make a rule)
ML_GB_N_ESTIMATORS = 100    # Matches RF for balance
ML_GB_MAX_DEPTH = 4         # Keep GB shallow to prevent boosting noise
ML_GB_LEARNING_RATE = 0.03  # Low rate = learns slowly/safely
ML_HIGH_ACCURACY_THRESHOLD = 0.60  
ML_VERY_HIGH_ACCURACY_THRESHOLD = 0.65  
ML_MIN_TRAINING_SAMPLES = 100 # Ensure sufficient data before training
ML_MIN_BINARY_SAMPLES = 50   # Higher minimum for binary classes
ML_PERFORMANCE_HISTORY_LIMIT = 1000  
ML_BINARY_HOLD_THRESHOLD = 0.60  # Balanced hold threshold
ML_RECENT_PERFORMANCE_WINDOW = 50  
ML_ENSEMBLE_WEIGHT_UPDATE_THRESHOLD = 5  
ML_PERFORMANCE_ACCURACY_DIFF_THRESHOLD = 10  
ML_ENSEMBLE_WEIGHT_MAX = 0.70  # Cap ML dominance (Safety net)
ML_ENSEMBLE_WEIGHT_MIN = 0.20  # Ensure ML always has a voice
ML_CONFIDENCE_DIFF_THRESHOLD_HIGH = 25  
ML_CONFIDENCE_DIFF_THRESHOLD_NORMAL = 15  
ML_RULE_CONFIDENCE_ADVANTAGE_THRESHOLD = 20  # Rules need 20% lead to override ML

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
