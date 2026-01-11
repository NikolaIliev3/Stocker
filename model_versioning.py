"""
Model Versioning and Rollback System
Tracks model versions, compares performance, and automatically rolls back if new model performs worse
"""
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ModelVersionManager:
    """Manages model versions, tracks performance, and handles rollbacks"""
    
    def __init__(self, data_dir: Path, strategy: str):
        self.data_dir = data_dir
        self.strategy = strategy
        self.versions_dir = data_dir / "model_versions" / strategy
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        self.version_history_file = self.versions_dir / "version_history.json"
        self.version_history = self._load_version_history()
        
        # Current model files
        self.current_model_file = f"ml_model_{strategy}.pkl"
        self.current_scaler_file = f"scaler_{strategy}.pkl"
        self.current_encoder_file = f"label_encoder_{strategy}.pkl"
        self.current_metadata_file = data_dir / f"ml_metadata_{strategy}.json"
    
    def _load_version_history(self) -> List[Dict]:
        """Load version history from file"""
        if self.version_history_file.exists():
            try:
                with open(self.version_history_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading version history: {e}")
                return []
        return []
    
    def _save_version_history(self):
        """Save version history to file"""
        try:
            with open(self.version_history_file, 'w') as f:
                json.dump(self.version_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving version history: {e}")
    
    def get_current_version(self) -> Optional[Dict]:
        """Get current active model version"""
        # Find the version marked as 'active'
        for version in reversed(self.version_history):  # Check newest first
            if version.get('status') == 'active':
                return version
        return None
    
    def get_latest_version(self) -> Optional[Dict]:
        """Get the latest version (regardless of status)"""
        if self.version_history:
            return self.version_history[-1]
        return None
    
    def create_version_backup(self, version_number: str, metadata: Dict) -> bool:
        """Create a backup of the current model files before training new model"""
        try:
            version_dir = self.versions_dir / version_number
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy model files
            files_to_backup = [
                (self.current_model_file, "model.pkl"),
                (self.current_scaler_file, "scaler.pkl"),
                (self.current_encoder_file, "encoder.pkl"),
                (self.current_metadata_file, "metadata.json")
            ]
            
            backed_up = []
            for source_file, dest_name in files_to_backup:
                source_path = self.data_dir / source_file if isinstance(source_file, str) else source_file
                if source_path.exists():
                    dest_path = version_dir / dest_name
                    shutil.copy2(source_path, dest_path)
                    backed_up.append(dest_name)
                    logger.debug(f"Backed up {source_file} to version {version_number}")
            
            # Save version metadata
            version_metadata = {
                'version': version_number,
                'created_at': datetime.now().isoformat(),
                'strategy': self.strategy,
                'files_backed_up': backed_up,
                'metadata': metadata,
                'status': 'backup'  # Not active yet
            }
            
            version_meta_file = version_dir / "version_info.json"
            with open(version_meta_file, 'w') as f:
                json.dump(version_metadata, f, indent=2)
            
            logger.info(f"Created version backup {version_number} with {len(backed_up)} files")
            return True
            
        except Exception as e:
            logger.error(f"Error creating version backup: {e}")
            return False
    
    def generate_version_number(self) -> str:
        """Generate a new version number"""
        # Format: YYYYMMDD_HHMMSS
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def register_new_version(self, metadata: Dict, test_accuracy: float, 
                            train_accuracy: float, cv_mean: float = None) -> str:
        """Register a new model version after training"""
        version_number = self.generate_version_number()
        
        # Get previous version for comparison
        previous_version = self.get_current_version()
        previous_accuracy = previous_version.get('test_accuracy', 0) if previous_version else None
        
        # Determine if this version should be active
        should_activate = True
        rollback_reason = None
        
        if previous_accuracy is not None:
            accuracy_drop = previous_accuracy - test_accuracy
            if accuracy_drop > 0.05:  # 5% drop threshold
                should_activate = False
                rollback_reason = f"Accuracy dropped by {accuracy_drop*100:.1f}% (from {previous_accuracy*100:.1f}% to {test_accuracy*100:.1f}%)"
                logger.warning(f"⚠️ New model performs worse: {rollback_reason}")
            elif accuracy_drop > 0.02:  # 2-5% drop - warning but allow
                logger.warning(f"⚠️ New model accuracy slightly lower: {accuracy_drop*100:.1f}% drop")
        
        # Create version entry
        version_entry = {
            'version': version_number,
            'created_at': datetime.now().isoformat(),
            'strategy': self.strategy,
            'test_accuracy': float(test_accuracy),
            'train_accuracy': float(train_accuracy),
            'cv_mean': float(cv_mean) if cv_mean is not None else None,
            'cv_std': metadata.get('cv_std'),
            'samples': metadata.get('samples', 0),
            'status': 'active' if should_activate else 'rejected',
            'rollback_reason': rollback_reason,
            'previous_version': previous_version.get('version') if previous_version else None,
            'previous_accuracy': previous_accuracy,
            'accuracy_change': float(test_accuracy - previous_accuracy) if previous_accuracy is not None else None,
            'metadata': metadata
        }
        
        self.version_history.append(version_entry)
        self._save_version_history()
        
        if should_activate:
            # Deactivate previous version
            if previous_version:
                previous_version['status'] = 'superseded'
                previous_version['superseded_at'] = datetime.now().isoformat()
                previous_version['superseded_by'] = version_number
            
            logger.info(f"✅ Registered new active version {version_number} (accuracy: {test_accuracy*100:.1f}%)")
        else:
            logger.warning(f"❌ Rejected new version {version_number} due to performance degradation")
        
        return version_number, should_activate
    
    def rollback_to_version(self, target_version: str) -> bool:
        """Rollback to a specific version"""
        try:
            # Find target version
            target_version_entry = None
            for version in self.version_history:
                if version.get('version') == target_version:
                    target_version_entry = version
                    break
            
            if not target_version_entry:
                logger.error(f"Version {target_version} not found in history")
                return False
            
            version_dir = self.versions_dir / target_version
            
            # Restore files
            files_to_restore = [
                ("model.pkl", self.current_model_file),
                ("scaler.pkl", self.current_scaler_file),
                ("encoder.pkl", self.current_encoder_file),
                ("metadata.json", self.current_metadata_file)
            ]
            
            restored = []
            for source_name, dest_file in files_to_restore:
                source_path = version_dir / source_name
                if source_path.exists():
                    dest_path = self.data_dir / dest_file if isinstance(dest_file, str) else dest_file
                    shutil.copy2(source_path, dest_path)
                    restored.append(dest_file)
            
            # Update version statuses
            current_version = self.get_current_version()
            if current_version:
                current_version['status'] = 'superseded'
                current_version['superseded_at'] = datetime.now().isoformat()
                current_version['superseded_by'] = target_version
            
            target_version_entry['status'] = 'active'
            target_version_entry['rolled_back_at'] = datetime.now().isoformat()
            
            self._save_version_history()
            
            logger.info(f"✅ Rolled back to version {target_version} (restored {len(restored)} files)")
            return True
            
        except Exception as e:
            logger.error(f"Error rolling back to version {target_version}: {e}")
            return False
    
    def auto_rollback_if_worse(self, new_test_accuracy: float, 
                               rollback_threshold: float = 0.05) -> Tuple[bool, Optional[str]]:
        """Automatically rollback if new model is worse than current"""
        current_version = self.get_current_version()
        
        if not current_version:
            # No current version, can't rollback
            return False, None
        
        current_accuracy = current_version.get('test_accuracy', 0)
        accuracy_drop = current_accuracy - new_test_accuracy
        
        if accuracy_drop > rollback_threshold:
            # New model is significantly worse, rollback
            current_version_number = current_version.get('version')
            logger.warning(f"⚠️ Auto-rollback triggered: New accuracy {new_test_accuracy*100:.1f}% is {accuracy_drop*100:.1f}% worse than current {current_accuracy*100:.1f}%")
            
            # The rollback will happen by not activating the new version
            # But we should also ensure current version stays active
            return True, current_version_number
        
        return False, None
    
    def get_version_performance_history(self) -> List[Dict]:
        """Get performance history of all versions"""
        return [
            {
                'version': v.get('version'),
                'test_accuracy': v.get('test_accuracy'),
                'train_accuracy': v.get('train_accuracy'),
                'created_at': v.get('created_at'),
                'status': v.get('status'),
                'samples': v.get('samples', 0)
            }
            for v in self.version_history
        ]
    
    def cleanup_old_versions(self, keep_last_n: int = 10):
        """Clean up old version backups, keeping only the last N"""
        if len(self.version_history) <= keep_last_n:
            return
        
        # Sort by creation date
        sorted_versions = sorted(self.version_history, key=lambda x: x.get('created_at', ''))
        
        # Keep last N and active version
        versions_to_keep = set()
        active_version = None
        for version in self.version_history:
            if version.get('status') == 'active':
                active_version = version.get('version')
                versions_to_keep.add(active_version)
        
        # Add last N versions
        for version in sorted_versions[-keep_last_n:]:
            versions_to_keep.add(version.get('version'))
        
        # Delete old versions
        deleted_count = 0
        for version in self.version_history:
            version_num = version.get('version')
            if version_num not in versions_to_keep:
                version_dir = self.versions_dir / version_num
                if version_dir.exists():
                    try:
                        shutil.rmtree(version_dir)
                        deleted_count += 1
                        logger.debug(f"Deleted old version backup: {version_num}")
                    except Exception as e:
                        logger.warning(f"Could not delete version {version_num}: {e}")
        
        # Remove from history
        self.version_history = [v for v in self.version_history if v.get('version') in versions_to_keep]
        self._save_version_history()
        
        if deleted_count > 0:
            logger.info(f"Cleaned up {deleted_count} old version backups (kept {len(versions_to_keep)} versions)")
