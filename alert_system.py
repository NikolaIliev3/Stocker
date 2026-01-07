"""
Alert System Foundation
Basic alert system for price movements, predictions, and portfolio events
"""
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class Alert:
    """Represents a single alert"""
    
    def __init__(self, alert_type: str, message: str, priority: str = "medium",
                 data: Optional[Dict] = None):
        self.alert_type = alert_type
        self.message = message
        self.priority = priority  # 'low', 'medium', 'high', 'critical'
        self.timestamp = datetime.now()
        self.data = data or {}
        self.read = False
    
    def to_dict(self) -> Dict:
        """Convert alert to dictionary"""
        return {
            'alert_type': self.alert_type,
            'message': self.message,
            'priority': self.priority,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
            'read': self.read
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        """Create alert from dictionary"""
        alert = cls(
            alert_type=data['alert_type'],
            message=data['message'],
            priority=data.get('priority', 'medium'),
            data=data.get('data', {})
        )
        alert.timestamp = datetime.fromisoformat(data['timestamp'])
        alert.read = data.get('read', False)
        return alert


class AlertSystem:
    """Manages alerts and notifications"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.alerts_file = data_dir / "alerts.json"
        self.alerts: List[Alert] = []
        self.callbacks: List[Callable] = []
        self.load_alerts()
    
    def add_alert(self, alert_type: str, message: str, priority: str = "medium",
                  data: Optional[Dict] = None) -> Alert:
        """
        Add a new alert
        
        Args:
            alert_type: Type of alert (e.g., 'price_alert', 'prediction_alert')
            message: Alert message
            priority: Alert priority
            data: Additional data
            
        Returns:
            Created Alert object
        """
        alert = Alert(alert_type, message, priority, data)
        self.alerts.append(alert)
        self.save_alerts()
        
        # Trigger callbacks
        for callback in self.callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")
        
        logger.info(f"Alert created: {alert_type} - {message}")
        return alert
    
    def create_price_alert(self, symbol: str, current_price: float,
                          target_price: float, direction: str = "above") -> Alert:
        """Create price alert"""
        direction_text = "above" if direction == "above" else "below"
        message = f"{symbol} price is {direction_text} ${target_price:.2f} (current: ${current_price:.2f})"
        
        return self.add_alert(
            alert_type='price_alert',
            message=message,
            priority='medium',
            data={
                'symbol': symbol,
                'current_price': current_price,
                'target_price': target_price,
                'direction': direction
            }
        )
    
    def create_prediction_alert(self, symbol: str, action: str, 
                               confidence: float) -> Alert:
        """Create prediction alert"""
        message = f"New {action} recommendation for {symbol} (confidence: {confidence:.1f}%)"
        priority = 'high' if confidence > 80 else 'medium'
        
        return self.add_alert(
            alert_type='prediction_alert',
            message=message,
            priority=priority,
            data={
                'symbol': symbol,
                'action': action,
                'confidence': confidence
            }
        )
    
    def create_portfolio_alert(self, message: str, portfolio_data: Dict) -> Alert:
        """Create portfolio alert"""
        return self.add_alert(
            alert_type='portfolio_alert',
            message=message,
            priority='medium',
            data=portfolio_data
        )
    
    def get_unread_alerts(self) -> List[Alert]:
        """Get all unread alerts"""
        return [a for a in self.alerts if not a.read]
    
    def get_alerts_by_type(self, alert_type: str) -> List[Alert]:
        """Get alerts by type"""
        return [a for a in self.alerts if a.alert_type == alert_type]
    
    def mark_as_read(self, alert: Alert):
        """Mark alert as read"""
        alert.read = True
        self.save_alerts()
    
    def mark_all_as_read(self):
        """Mark all alerts as read"""
        for alert in self.alerts:
            alert.read = True
        self.save_alerts()
    
    def delete_alert(self, alert: Alert):
        """Delete an alert"""
        if alert in self.alerts:
            self.alerts.remove(alert)
            self.save_alerts()
    
    def clear_old_alerts(self, days: int = 30):
        """Clear alerts older than specified days"""
        cutoff = datetime.now().timestamp() - (days * 24 * 3600)
        self.alerts = [
            a for a in self.alerts
            if a.timestamp.timestamp() > cutoff
        ]
        self.save_alerts()
    
    def register_callback(self, callback: Callable):
        """Register callback for new alerts"""
        self.callbacks.append(callback)
    
    def load_alerts(self):
        """Load alerts from file"""
        if self.alerts_file.exists():
            try:
                with open(self.alerts_file, 'r') as f:
                    data = json.load(f)
                    self.alerts = [Alert.from_dict(a) for a in data.get('alerts', [])]
            except Exception as e:
                logger.error(f"Error loading alerts: {e}")
                self.alerts = []
        else:
            self.alerts = []
    
    def save_alerts(self):
        """Save alerts to file"""
        try:
            data = {
                'alerts': [a.to_dict() for a in self.alerts],
                'last_updated': datetime.now().isoformat()
            }
            with open(self.alerts_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving alerts: {e}")

