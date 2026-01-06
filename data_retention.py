"""
Data Retention Manager
Manages data retention and secure deletion
"""
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict
from secure_deletion import secure_delete_file
from config import (
    AUDIT_LOG_RETENTION_DAYS,
    RESEARCH_CACHE_RETENTION_DAYS,
    API_KEY_ROTATION_DAYS
)

logger = logging.getLogger(__name__)


class DataRetentionManager:
    """Manage data retention and secure deletion"""
    
    RETENTION_POLICIES = {
        'api_audit_log': timedelta(days=AUDIT_LOG_RETENTION_DAYS),
        'research_cache': timedelta(days=RESEARCH_CACHE_RETENTION_DAYS),
        'api_keys_backup': timedelta(days=API_KEY_ROTATION_DAYS * 2),  # Keep backups longer
    }
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def cleanup_old_data(self) -> Dict[str, int]:
        """
        Remove data older than retention policy
        
        Returns:
            Dictionary with counts of deleted items by type
        """
        now = datetime.now()
        deleted_counts = {}
        
        # Cleanup audit log
        deleted_counts['audit_log_entries'] = self._cleanup_log_file(
            self.data_dir / "api_audit.log",
            self.RETENTION_POLICIES['api_audit_log']
        )
        
        # Cleanup research cache
        cache_dir = self.data_dir / "research_cache"
        if cache_dir.exists():
            deleted_count = 0
            for cache_file in cache_dir.glob("*_research.json"):
                if self._is_file_older_than(cache_file, self.RETENTION_POLICIES['research_cache']):
                    if secure_delete_file(cache_file):
                        deleted_count += 1
            deleted_counts['research_cache_files'] = deleted_count
        
        # Cleanup old API key backups
        deleted_count = 0
        for backup_file in self.data_dir.glob("*_backup.key"):
            if self._is_file_older_than(backup_file, self.RETENTION_POLICIES['api_keys_backup']):
                if secure_delete_file(backup_file):
                    deleted_count += 1
        deleted_counts['api_key_backups'] = deleted_count
        
        logger.info(f"Data retention cleanup completed: {deleted_counts}")
        return deleted_counts
    
    def _is_file_older_than(self, file_path: Path, max_age: timedelta) -> bool:
        """Check if file is older than max_age"""
        try:
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            return (datetime.now() - mtime) > max_age
        except Exception as e:
            logger.error(f"Error checking file age for {file_path}: {e}")
            return False
    
    def _cleanup_log_file(self, log_file: Path, max_age: timedelta) -> int:
        """Clean up old entries from log file"""
        if not log_file.exists():
            return 0
        
        try:
            cutoff_date = datetime.now() - max_age
            kept_lines = []
            deleted_count = 0
            
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        entry = json.loads(line.strip())
                        timestamp_str = entry.get('timestamp', '')
                        if timestamp_str:
                            entry_date = datetime.fromisoformat(timestamp_str)
                            if entry_date > cutoff_date:
                                kept_lines.append(line)
                            else:
                                deleted_count += 1
                        else:
                            # Keep entries without timestamps (shouldn't happen)
                            kept_lines.append(line)
                    except json.JSONDecodeError:
                        # Keep malformed lines
                        kept_lines.append(line)
            
            # Rewrite file with kept lines
            if deleted_count > 0:
                with open(log_file, 'w') as f:
                    f.writelines(kept_lines)
                logger.debug(f"Cleaned {deleted_count} old entries from {log_file}")
            
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning log file {log_file}: {e}")
            return 0
    
    def get_retention_statistics(self) -> Dict:
        """Get statistics about data retention"""
        stats = {
            'policies': {},
            'file_counts': {},
            'oldest_files': {}
        }
        
        for data_type, max_age in self.RETENTION_POLICIES.items():
            stats['policies'][data_type] = {
                'retention_days': max_age.days,
                'max_age': max_age.total_seconds()
            }
        
        # Count files
        cache_dir = self.data_dir / "research_cache"
        if cache_dir.exists():
            stats['file_counts']['research_cache'] = len(list(cache_dir.glob("*_research.json")))
        
        stats['file_counts']['api_key_backups'] = len(list(self.data_dir.glob("*_backup.key")))
        
        # Check audit log
        log_file = self.data_dir / "api_audit.log"
        if log_file.exists():
            try:
                with open(log_file, 'r') as f:
                    stats['file_counts']['audit_log_entries'] = sum(1 for _ in f)
            except:
                stats['file_counts']['audit_log_entries'] = 0
        
        return stats
