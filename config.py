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
INITIAL_BALANCE = 10000.0  # Default starting balance

