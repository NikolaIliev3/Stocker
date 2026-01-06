"""
Anomaly Detection Module
Detects suspicious API usage patterns
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List
from collections import defaultdict
from config import ENABLE_ANOMALY_DETECTION, ANOMALY_ALERT_THRESHOLD

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detect suspicious API usage patterns"""
    
    def __init__(self):
        self.enabled = ENABLE_ANOMALY_DETECTION
        self.request_history: List[Dict] = []
        self.alert_count = 0
        self.alert_thresholds = {
            'rapid_requests': 100,  # requests per minute
            'failed_requests': 10,  # consecutive failures
            'unusual_symbols': 50,  # unique symbols per hour
            'large_responses': 5,  # large responses per hour
        }
    
    def check_anomaly(self, request_type: str, symbol: str, success: bool, 
                     response_size: int = 0) -> bool:
        """
        Check for anomalous patterns
        
        Args:
            request_type: Type of request (e.g., 'news', 'llm')
            symbol: Stock symbol
            success: Whether request succeeded
            response_size: Size of response in bytes
        
        Returns:
            True if anomaly detected, False otherwise
        """
        if not self.enabled:
            return False
        
        now = datetime.now()
        self.request_history.append({
            'timestamp': now,
            'type': request_type,
            'symbol': symbol,
            'success': success,
            'response_size': response_size
        })
        
        # Keep only last hour
        cutoff = now - timedelta(hours=1)
        self.request_history = [
            r for r in self.request_history
            if r['timestamp'] > cutoff
        ]
        
        anomalies_detected = []
        
        # Check rapid requests
        recent_requests = [
            r for r in self.request_history
            if (now - r['timestamp']).seconds < 60
        ]
        if len(recent_requests) > self.alert_thresholds['rapid_requests']:
            logger.warning(f"Anomaly detected: Rapid requests ({len(recent_requests)} in last minute)")
            anomalies_detected.append('rapid_requests')
        
        # Check failed requests
        recent_failures = [
            r for r in self.request_history[-10:]
            if not r['success']
        ]
        if len(recent_failures) >= self.alert_thresholds['failed_requests']:
            logger.warning(f"Anomaly detected: Multiple failures ({len(recent_failures)} consecutive)")
            anomalies_detected.append('failed_requests')
        
        # Check unusual symbols (many unique symbols)
        unique_symbols = set(r['symbol'] for r in self.request_history)
        if len(unique_symbols) > self.alert_thresholds['unusual_symbols']:
            logger.warning(f"Anomaly detected: Unusual symbol count ({len(unique_symbols)} unique)")
            anomalies_detected.append('unusual_symbols')
        
        # Check large responses
        large_responses = [
            r for r in self.request_history
            if r['response_size'] > 5 * 1024 * 1024  # 5MB
        ]
        if len(large_responses) > self.alert_thresholds['large_responses']:
            logger.warning(f"Anomaly detected: Large responses ({len(large_responses)} large)")
            anomalies_detected.append('large_responses')
        
        if anomalies_detected:
            self.alert_count += 1
            if self.alert_count >= ANOMALY_ALERT_THRESHOLD:
                logger.error(f"CRITICAL: Multiple anomalies detected. Consider blocking requests.")
                return True
        
        return len(anomalies_detected) > 0
    
    def reset_alerts(self):
        """Reset alert counter"""
        self.alert_count = 0
    
    def get_statistics(self) -> Dict:
        """Get anomaly detection statistics"""
        now = datetime.now()
        last_hour = [
            r for r in self.request_history
            if (now - r['timestamp']).total_seconds() < 3600
        ]
        
        return {
            'total_requests_last_hour': len(last_hour),
            'successful_requests': len([r for r in last_hour if r['success']]),
            'failed_requests': len([r for r in last_hour if not r['success']]),
            'unique_symbols': len(set(r['symbol'] for r in last_hour)),
            'alert_count': self.alert_count,
            'enabled': self.enabled
        }
