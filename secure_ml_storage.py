"""
Secure ML Storage System
Implements security measures for ML model storage and loading
"""
import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Any
import logging

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Security-related exception"""
    pass


class SecureMLStorage:
    """Secure storage for ML models and training data with integrity checks"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.checksums_file = self.data_dir / "model_checksums.json"
        self._secure_directory()
        self.checksums = self._load_checksums()
    
    def _secure_directory(self):
        """Set secure directory permissions"""
        try:
            if os.name != 'nt':  # Unix/Linux/Mac
                os.chmod(self.data_dir, 0o700)  # rwx------
        except Exception as e:
            logger.warning(f"Could not set directory permissions: {e}")
    
    def _secure_file(self, filepath: Path):
        """Set secure file permissions"""
        try:
            if os.name != 'nt':
                os.chmod(filepath, 0o600)  # rw-------
        except Exception as e:
            logger.warning(f"Could not set file permissions: {e}")
    
    def _load_checksums(self) -> dict:
        """Load checksums from file with error recovery"""
        if self.checksums_file.exists():
            try:
                with open(self.checksums_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                # Only log once per session - use a flag to prevent spam
                if not hasattr(self, '_checksum_error_logged'):
                    logger.warning(f"JSON syntax error in checksums file (line {e.lineno}, col {e.colno}): {e.msg}")
                    self._checksum_error_logged = True
                
                # Try to create a backup of the corrupted file
                try:
                    backup_file = self.checksums_file.with_suffix('.json.bak')
                    # Only create backup if it doesn't exist yet
                    if not backup_file.exists():
                        import shutil
                        shutil.copy2(self.checksums_file, backup_file)
                        logger.info(f"Created backup of corrupted checksums file: {backup_file}")
                except Exception as backup_error:
                    if not hasattr(self, '_checksum_error_logged'):
                        logger.warning(f"Could not create backup: {backup_error}")
                
                # Delete the corrupted file and create a new empty one
                try:
                    self.checksums_file.unlink()  # Delete corrupted file
                    # Create new empty file
                    with open(self.checksums_file, 'w', encoding='utf-8') as f:
                        json.dump({}, f)
                    if not hasattr(self, '_checksum_error_logged'):
                        logger.info("Deleted corrupted checksums file - will rebuild on next save")
                except Exception as delete_error:
                    if not hasattr(self, '_checksum_error_logged'):
                        logger.warning(f"Could not delete corrupted file: {delete_error}")
                
                # Return empty dict - checksums will be rebuilt
                return {}
            except Exception as e:
                if not hasattr(self, '_checksum_error_logged'):
                    logger.warning(f"Error loading checksums: {e}")
                    self._checksum_error_logged = True
                return {}
        return {}
    
    def _save_checksums(self):
        """Save checksums to file"""
        try:
            with open(self.checksums_file, 'w') as f:
                json.dump(self.checksums, f, indent=2)
            self._secure_file(self.checksums_file)
        except Exception as e:
            logger.error(f"Error saving checksums: {e}")
    
    def _calculate_checksum(self, filepath: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256 = hashlib.sha256()
        try:
            with open(filepath, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error calculating checksum: {e}")
            return ""
    
    def _restricted_find_class(self, module: str, name: str):
        """Restrict pickle to only load trusted classes"""
        # Only allow trusted modules
        # sklearn uses internal Cython modules (_loss, _tree, etc.) that need to be allowed
        trusted_modules = [
            'sklearn', 'numpy', 'pandas', 'scipy',
            '__builtin__', 'builtins', '_pickle',
            # scikit-learn internal modules (Cython extensions)
            '_loss', '_tree', '_splitter', '_criterion',
            '_utils', '_base', '_classes', '_ensemble',
            '_forest', '_gb_losses', '_gradient_boosting',
            '_weight_boosting', '_logistic', '_linear_model',
            '_preprocessing', '_validation', '_feature_selection'
        ]
        
        module_base = module.split('.')[0]
        # Also allow sklearn submodules (sklearn.ensemble, sklearn.tree, etc.)
        if module_base not in trusted_modules and not module.startswith('sklearn.'):
            raise SecurityError(f"Untrusted module attempted: {module}.{name}")
        
        try:
            return getattr(__import__(module, fromlist=[name]), name)
        except Exception as e:
            raise SecurityError(f"Could not import {module}.{name}: {e}")
    
    def save_model(self, model: Any, filename: str):
        """Save ML model securely with integrity check"""
        filepath = self.data_dir / filename
        
        try:
            # Save model
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
            
            # Set secure permissions
            self._secure_file(filepath)
            
            # Calculate and save checksum
            checksum = self._calculate_checksum(filepath)
            self.checksums[filename] = checksum
            self._save_checksums()
            
            logger.info(f"Saved model {filename} with integrity check")
        except Exception as e:
            logger.error(f"Error saving model {filename}: {e}")
            raise
    
    def load_model(self, filename: str) -> Optional[Any]:
        """Load ML model with integrity verification"""
        filepath = self.data_dir / filename
        
        if not filepath.exists():
            logger.warning(f"Model file not found: {filename}")
            return None
        
        try:
            # Verify checksum if available
            expected_checksum = self.checksums.get(filename)
            if expected_checksum:
                current_checksum = self._calculate_checksum(filepath)
                if current_checksum != expected_checksum:
                    raise SecurityError(
                        f"Model {filename} integrity check failed! "
                        f"File may have been tampered with."
                    )
            
            # Load with restricted unpickler
            # Create a custom Unpickler class instead of modifying find_class directly
            class RestrictedUnpickler(pickle.Unpickler):
                def __init__(self, file, storage_instance):
                    super().__init__(file)
                    self.storage = storage_instance
                
                def find_class(self, module, name):
                    return self.storage._restricted_find_class(module, name)
            
            with open(filepath, 'rb') as f:
                unpickler = RestrictedUnpickler(f, self)
                model = unpickler.load()
            
            logger.info(f"Loaded model {filename} successfully")
            return model
            
        except SecurityError:
            raise
        except Exception as e:
            logger.error(f"Error loading model {filename}: {e}")
            return None
    
    def model_exists(self, filename: str) -> bool:
        """Check if model file exists"""
        return (self.data_dir / filename).exists()
    
    def get_model_info(self, filename: str) -> dict:
        """Get information about a model"""
        filepath = self.data_dir / filename
        if not filepath.exists():
            return {}
        
        try:
            stat = filepath.stat()
            return {
                'exists': True,
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'has_checksum': filename in self.checksums
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {}

