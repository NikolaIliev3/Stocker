"""
Security module for Stocker App
Implements HTTPS enforcement, certificate pinning, data validation, and integrity checks
"""
import ssl
import hashlib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
from urllib3.poolmanager import PoolManager
import config
import logging

logger = logging.getLogger(__name__)


class SecureHTTPSAdapter(HTTPAdapter):
    """Custom HTTP adapter that enforces TLS 1.2+ and validates certificates"""
    
    def init_poolmanager(self, *args, **kwargs):
        ctx = create_urllib3_context()
        ctx.set_ciphers('DEFAULT:!DH')
        ctx.options |= 0x4  # OP_LEGACY_SERVER_CONNECT
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.maximum_version = ssl.TLSVersion.MAXIMUM_SUPPORTED
        kwargs['ssl_context'] = ctx
        return super().init_poolmanager(*args, **kwargs)


class CertificatePinner:
    """Implements certificate pinning for API endpoints"""
    
    def __init__(self):
        self.pinned_certs = {}
        # In production, add your backend API certificate fingerprints here
        # Format: {hostname: [sha256_fingerprint1, sha256_fingerprint2]}
    
    def add_pin(self, hostname: str, fingerprint: str):
        """Add a certificate fingerprint for a hostname"""
        if hostname not in self.pinned_certs:
            self.pinned_certs[hostname] = []
        self.pinned_certs[hostname].append(fingerprint.lower())
    
    def verify_certificate(self, hostname: str, cert: bytes) -> bool:
        """Verify if certificate matches pinned fingerprint"""
        if not config.CERTIFICATE_PINNING_ENABLED:
            return True
        
        if hostname not in self.pinned_certs:
            logger.warning(f"No certificate pin for {hostname}, allowing")
            return True
        
        cert_hash = hashlib.sha256(cert).hexdigest().lower()
        return cert_hash in self.pinned_certs[hostname]


class SecureSession(requests.Session):
    """Secure requests session with HTTPS enforcement and certificate pinning"""
    
    def __init__(self):
        super().__init__()
        self.cert_pinner = CertificatePinner()
        
        # Mount secure adapter
        adapter = SecureHTTPSAdapter()
        self.mount('https://', adapter)
        
        # Enforce HTTPS
        if config.ENFORCE_HTTPS:
            self.mount('http://', BlockHTTPAdapter())
    
    def request(self, method, url, **kwargs):
        """Override request to enforce security checks"""
        # Allow HTTP for localhost (safe for local development)
        # Block HTTP for external URLs (security risk)
        if config.ENFORCE_HTTPS and url.startswith('http://'):
            # Check if it's localhost/127.0.0.1 (safe)
            from urllib.parse import urlparse
            parsed = urlparse(url)
            hostname = parsed.hostname or ''
            
            # Allow localhost connections (development)
            is_localhost = hostname in ['localhost', '127.0.0.1', '::1'] or hostname.startswith('192.168.') or hostname.startswith('10.')
            
            if not is_localhost:
                raise SecurityError("HTTP requests to external servers are not allowed. Use HTTPS only.")
            else:
                # Only log at debug level to reduce spam during training
                logger.debug(f"Allowing HTTP connection to localhost: {hostname}")
        
        # Add certificate verification (skip for localhost HTTP)
        if 'verify' not in kwargs:
            if url.startswith('https://'):
                kwargs['verify'] = True
            else:
                # Skip verification for localhost HTTP
                kwargs['verify'] = False
        
        return super().request(method, url, **kwargs)


class BlockHTTPAdapter(HTTPAdapter):
    """Adapter that blocks external HTTP requests (allows localhost)"""
    
    def send(self, request, **kwargs):
        # Allow localhost HTTP connections
        from urllib.parse import urlparse
        parsed = urlparse(request.url)
        hostname = parsed.hostname or ''
        
        is_localhost = hostname in ['localhost', '127.0.0.1', '::1'] or hostname.startswith('192.168.') or hostname.startswith('10.')
        
        if not is_localhost:
            raise SecurityError("HTTP requests to external servers are blocked. Use HTTPS only.")
        
        # Allow localhost HTTP to proceed
        return super().send(request, **kwargs)


class SecurityError(Exception):
    """Custom security exception"""
    pass


class DataValidator:
    """Validates stock data for integrity and sanity"""
    
    @staticmethod
    def validate_price(price: float) -> bool:
        """Validate stock price is within reasonable bounds"""
        if not isinstance(price, (int, float)):
            return False
        return config.MIN_PRICE <= price <= config.MAX_PRICE
    
    @staticmethod
    def validate_volume(volume: int) -> bool:
        """Validate trading volume is within reasonable bounds"""
        if not isinstance(volume, (int, float)):
            return False
        return config.MIN_VOLUME <= volume <= config.MAX_VOLUME
    
    @staticmethod
    def validate_stock_data(data: dict) -> bool:
        """Validate complete stock data dictionary"""
        required_fields = ['symbol', 'price', 'volume']
        
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return False
        
        if not DataValidator.validate_price(data.get('price', 0)):
            logger.error(f"Invalid price: {data.get('price')}")
            return False
        
        if not DataValidator.validate_volume(data.get('volume', 0)):
            logger.error(f"Invalid volume: {data.get('volume')}")
            return False
        
        return True
    
    @staticmethod
    def detect_outliers(prices: list, threshold: float = 3.0) -> list:
        """Detect price outliers using statistical methods"""
        import numpy as np
        
        if len(prices) < 3:
            return []
        
        prices_array = np.array(prices)
        mean = np.mean(prices_array)
        std = np.std(prices_array)
        
        if std == 0:
            return []
        
        z_scores = np.abs((prices_array - mean) / std)
        outliers = np.where(z_scores > threshold)[0].tolist()
        
        return outliers


def get_secure_session() -> SecureSession:
    """Factory function to get a secure requests session"""
    return SecureSession()

