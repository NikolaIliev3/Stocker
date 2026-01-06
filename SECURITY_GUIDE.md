# Security Implementation Guide

This document describes all security measures implemented in the Stocker application, particularly for the Seeker AI module.

## Security Modules

### 1. Secure Logging (`secure_logging.py`)
- **Purpose**: Prevents sensitive data leakage in logs
- **Features**:
  - Automatically redacts API keys, tokens, passwords, and secrets
  - Uses regex patterns to detect and mask sensitive information
  - Safe logging methods that redact before logging

**Usage**:
```python
from secure_logging import SecureLogger

# Instead of logger.info()
SecureLogger.info("API key is: sk-1234567890")  # Will redact the key
```

### 2. Request Signing (`request_signer.py`)
- **Purpose**: Signs requests to prevent tampering
- **Features**:
  - HMAC-SHA256 signatures
  - Timestamp validation (prevents replay attacks)
  - Constant-time comparison (prevents timing attacks)

**Usage**:
```python
from request_signer import RequestSigner

signer = RequestSigner(secret_key="your-secret-key")
signed_request = signer.create_signed_request({'data': 'value'})
is_valid = signer.verify_request(data, timestamp, signature)
```

### 3. API Key Management (`ai_researcher.py` - SecureCredentialManager)
- **Purpose**: Secure storage and retrieval of API keys
- **Features**:
  - Encryption using Fernet (symmetric encryption)
  - Environment variable support (most secure)
  - Encrypted file storage with restrictive permissions
  - Automatic key generation

**Usage**:
```python
from ai_researcher import SecureCredentialManager
from config import APP_DATA_DIR

manager = SecureCredentialManager(APP_DATA_DIR)

# Store API key
manager.store_api_key('newsapi', 'your-api-key')

# Retrieve API key (checks env vars first, then encrypted file)
key = manager.get_api_key('newsapi')
```

### 4. API Key Rotation (`api_key_rotator.py`)
- **Purpose**: Manages API key rotation for security
- **Features**:
  - Automatic rotation scheduling (default: 90 days)
  - Backup of old keys before rotation
  - Rotation status tracking

**Usage**:
```python
from api_key_rotator import APIKeyRotator

rotator = APIKeyRotator(credential_manager, rotation_days=90)

if rotator.should_rotate('newsapi'):
    rotator.rotate_key('newsapi', 'new-api-key')
```

### 5. Secure Deletion (`secure_deletion.py`)
- **Purpose**: Securely delete files by overwriting with random data
- **Features**:
  - Multiple overwrite passes (default: 3)
  - Secure deletion of directories
  - Prevents data recovery

**Usage**:
```python
from secure_deletion import secure_delete_file, secure_delete_directory
from pathlib import Path

# Securely delete a file
secure_delete_file(Path('sensitive_file.txt'), passes=3)

# Securely delete a directory
secure_delete_directory(Path('sensitive_dir/'), passes=3)
```

### 6. Input Validation (`input_validator.py`)
- **Purpose**: Comprehensive input validation and sanitization
- **Features**:
  - Symbol validation (format, length, characters)
  - JSON size validation
  - String sanitization (removes control characters)
  - URL validation (HTTPS only)
  - Email validation
  - Numeric range validation
  - Injection attack pattern detection

**Usage**:
```python
from input_validator import InputValidator

# Validate stock symbol
if InputValidator.validate_symbol('AAPL'):
    # Process symbol

# Sanitize string
safe_text = InputValidator.sanitize_string(user_input, max_length=1000)

# Validate JSON size
if InputValidator.validate_json_size(data):
    # Process data
```

### 7. Secure API Client (`secure_api_client.py`)
- **Purpose**: API client with security timeouts and connection limits
- **Features**:
  - Connection timeout (5 seconds)
  - Read timeout (configurable, default: 30 seconds)
  - Maximum connections limit (default: 10)
  - Maximum redirects (5)
  - Retry logic with exponential backoff
  - Uses secure session from security.py

**Usage**:
```python
from secure_api_client import SecureAPIClient
from security import get_secure_session

client = SecureAPIClient(get_secure_session())
response = client.get('https://api.example.com/data')
```

### 8. Response Validation (`response_validator.py`)
- **Purpose**: Validates API responses for security
- **Features**:
  - Content type validation
  - Response size limits (10MB max)
  - JSON structure validation
  - Suspicious pattern detection (XSS, injection)
  - Status code validation

**Usage**:
```python
from response_validator import ResponseValidator

if ResponseValidator.validate_response(response):
    data = response.json()
    # Process data
```

### 9. Anomaly Detection (`anomaly_detector.py`)
- **Purpose**: Detects suspicious API usage patterns
- **Features**:
  - Rapid request detection (100+ per minute)
  - Failed request tracking (10+ consecutive)
  - Unusual symbol count (50+ unique per hour)
  - Large response detection (5MB+)
  - Alert threshold system

**Usage**:
```python
from anomaly_detector import AnomalyDetector

detector = AnomalyDetector()

if detector.check_anomaly('news', 'AAPL', success=True, response_size=1024):
    # Anomaly detected, take action
    logger.warning("Suspicious activity detected")
```

### 10. Data Retention Manager (`data_retention.py`)
- **Purpose**: Manages data retention and secure deletion
- **Features**:
  - Configurable retention policies
  - Automatic cleanup of old data
  - Secure deletion of expired files
  - Audit log cleanup
  - Statistics tracking

**Usage**:
```python
from data_retention import DataRetentionManager
from config import APP_DATA_DIR

manager = DataRetentionManager(APP_DATA_DIR)

# Cleanup old data
deleted_counts = manager.cleanup_old_data()
# Returns: {'audit_log_entries': 10, 'research_cache_files': 5, ...}

# Get statistics
stats = manager.get_retention_statistics()
```

### 11. Seeker AI (`ai_researcher.py`)
- **Purpose**: AI-powered stock research with comprehensive security
- **Features**:
  - All security measures integrated
  - Rate limiting
  - Input validation
  - Response validation
  - Anomaly detection
  - Audit logging
  - Secure credential management
  - HTTPS enforcement
  - Error handling without information leakage

**Usage**:
```python
from ai_researcher import SeekerAI
from config import APP_DATA_DIR

seeker_ai = SeekerAI(APP_DATA_DIR)

# Store API keys (one-time setup)
seeker_ai.credential_manager.store_api_key('newsapi', 'your-news-api-key')
seeker_ai.credential_manager.store_api_key('openai', 'your-openai-key')

# Research a stock
research = seeker_ai.research_stock('AAPL', stock_data, history_data)

# Cleanup old data
seeker_ai.cleanup_old_data()
```

## Configuration

All security settings are in `config.py`:

```python
# API Security
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

# Anomaly Detection
ENABLE_ANOMALY_DETECTION = True
ANOMALY_ALERT_THRESHOLD = 10
```

## Security Best Practices

### 1. API Key Storage
- **Best**: Use environment variables
  ```bash
  export NEWSAPI_API_KEY=your-key
  export OPENAI_API_KEY=your-key
  ```
- **Good**: Use encrypted file storage (automatic)
- **Never**: Hardcode keys in source code

### 2. Regular Maintenance
- Run dependency security checks:
  ```bash
  pip install safety
  python check_dependencies.py
  ```
- Rotate API keys every 90 days (automatic reminder)
- Cleanup old data regularly (automatic)

### 3. Monitoring
- Check audit logs: `~/.stocker/api_audit.log`
- Monitor anomaly detection alerts
- Review security logs regularly

### 4. Updates
- Keep dependencies updated (pinned versions in requirements.txt)
- Review security patches regularly
- Update encryption keys if compromised

## Security Checklist

- [x] Encrypted API key storage
- [x] Environment variable support
- [x] Rate limiting
- [x] Input validation
- [x] HTTPS enforcement
- [x] Certificate verification
- [x] Audit logging
- [x] Secure error handling
- [x] Secure logging (redaction)
- [x] Request signing
- [x] API key rotation
- [x] Secure file deletion
- [x] Response validation
- [x] Anomaly detection
- [x] Data retention policies
- [x] Dependency scanning
- [x] Timeout limits
- [x] Connection limits
- [x] Security headers (backend)
- [x] .gitignore updates

## Integration with Main Application

To integrate Seeker AI into the main application:

```python
# In main.py __init__
from ai_researcher import SeekerAI

self.seeker_ai = SeekerAI(APP_DATA_DIR)

# In _perform_analysis method
def _perform_analysis(self, symbol: str):
    # ... existing code ...
    
    # Get Seeker AI research
    research_data = self.seeker_ai.research_stock(symbol, stock_data, history_data)
    
    # Add research to analysis
    if 'error' not in research_data:
        analysis['seeker_ai_research'] = research_data
```

## Troubleshooting

### API Keys Not Working
1. Check environment variables: `echo $NEWSAPI_API_KEY`
2. Check encrypted storage: Keys are in `~/.stocker/.api_keys.encrypted`
3. Verify key format and permissions

### Rate Limiting Issues
- Check rate limit settings in `config.py`
- Review `api_audit.log` for rate limit violations
- Adjust `SEEKER_AI_RATE_LIMIT` if needed

### SSL Errors
- Ensure certificates are up to date
- Check `certifi` package installation
- Verify HTTPS URLs (no HTTP for external APIs)

### Anomaly Detection Alerts
- Review `api_audit.log` for patterns
- Check if legitimate high-volume usage
- Adjust thresholds in `anomaly_detector.py` if needed

## Support

For security issues or questions, review:
- `security.py` - Core security module
- `config.py` - Security configuration
- This guide - Security implementation details
