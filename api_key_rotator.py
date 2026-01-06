"""
API Key Rotation Module
Manages API key rotation for security
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class APIKeyRotator:
    """Manages API key rotation"""
    
    def __init__(self, credential_manager, rotation_days: int = 90):
        self.credential_manager = credential_manager
        self.rotation_days = rotation_days
        self.rotation_schedule = {}  # {service: days_until_rotation}
    
    def should_rotate(self, service: str) -> bool:
        """Check if key should be rotated"""
        rotation_file = self.credential_manager.data_dir / f".{service}_rotation.json"
        if not rotation_file.exists():
            return False
        
        try:
            with open(rotation_file, 'r') as f:
                data = json.load(f)
                last_rotation_str = data.get('last_rotation', '')
                if not last_rotation_str:
                    return False
                
                last_rotation = datetime.fromisoformat(last_rotation_str)
                days_since = (datetime.now() - last_rotation).days
                return days_since >= self.rotation_days
        except Exception as e:
            logger.error(f"Error checking rotation for {service}: {e}")
            return False
    
    def rotate_key(self, service: str, new_key: str) -> bool:
        """Rotate API key"""
        try:
            # Store old key as backup (encrypted)
            old_key = self.credential_manager.get_api_key(service)
            if old_key:
                backup_file = self.credential_manager.data_dir / f".{service}_backup.key"
                # Store encrypted backup
                try:
                    encrypted_backup = self.credential_manager._cipher.encrypt(old_key.encode())
                    with open(backup_file, 'wb') as f:
                        f.write(encrypted_backup)
                    # Set restrictive permissions
                    try:
                        import os
                        os.chmod(backup_file, 0o600)
                    except:
                        pass
                except Exception as e:
                    logger.warning(f"Could not backup old key: {e}")
            
            # Store new key
            self.credential_manager.store_api_key(service, new_key)
            
            # Update rotation date
            rotation_file = self.credential_manager.data_dir / f".{service}_rotation.json"
            with open(rotation_file, 'w') as f:
                json.dump({
                    'last_rotation': datetime.now().isoformat(),
                    'service': service,
                    'rotation_days': self.rotation_days
                }, f)
            
            logger.info(f"API key rotated for {service}")
            return True
            
        except Exception as e:
            logger.error(f"Error rotating key for {service}: {e}")
            return False
    
    def get_rotation_status(self, service: str) -> Optional[Dict]:
        """Get rotation status for a service"""
        rotation_file = self.credential_manager.data_dir / f".{service}_rotation.json"
        if not rotation_file.exists():
            return None
        
        try:
            with open(rotation_file, 'r') as f:
                data = json.load(f)
                last_rotation = datetime.fromisoformat(data.get('last_rotation', ''))
                days_since = (datetime.now() - last_rotation).days
                days_until = self.rotation_days - days_since
                
                return {
                    'last_rotation': last_rotation.isoformat(),
                    'days_since': days_since,
                    'days_until_rotation': max(0, days_until),
                    'should_rotate': days_since >= self.rotation_days
                }
        except Exception as e:
            logger.error(f"Error getting rotation status: {e}")
            return None
