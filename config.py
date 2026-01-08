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
SEEKER_AI_ENABLED = True
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

# Backtesting Configuration
# Number of historical test points to evaluate per strategy during each backtest run
# RECOMMENDATIONS:
#   - Quick test (5-10 min):  20-30 tests per strategy
#   - Standard test (10-20 min): 50-100 tests per strategy
#   - Thorough test (30-60 min): 150-200 tests per strategy
#   - Deep analysis (1-2 hours): 300-500 tests per strategy
# Higher numbers = more accurate accuracy measurement, but takes longer
BACKTEST_TESTS_PER_RUN = 100  # Default: 100 tests per strategy (recommended for good accuracy)
