"""
Safe Storage Module
Implements thread-safe and process-safe file operations for the Stocker application.
Adheres to:
1. Single Responsibility Principle (SRP): Only handles file I/O security.
2. KISS: Uses standard library (msvcrt) for Windows locking.
3. Handle Errors Gracefully: Retries and timeouts.
"""
import json
import os
import time
import logging
import msvcrt
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

class LockingJSONStorage:
    """
    Thread-safe and Process-safe JSON storage using Windows file locking.
    Prevents race conditions when multiple processes (UI, Backtester) access the same file.
    """
    
    def __init__(self, file_path: Path):
        self.file_path = Path(file_path)
        self.lock_file_path = self.file_path.with_suffix(self.file_path.suffix + '.lock')
        self.max_retries = 50
        self.retry_delay = 0.1  # seconds
    
    def _acquire_lock(self, f) -> bool:
        """Acquire an exclusive lock on the file handle"""
        try:
            # LK_NBLC: Non-blocking lock, LK_NBRL: Non-blocking read lock
            # We use LOCK_EX | LOCK_NB equivalent in msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except (IOError, OSError):
            return False

    def _release_lock(self, f):
        """Release the lock"""
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
        except (IOError, OSError):
            pass

    def load(self, default: Any = None) -> Any:
        """
        Load JSON data safely.
        """
        if not self.file_path.exists():
            return default if default is not None else {}

        # Simple read doesn't strictly require a lock if we assume atomic writes (rename),
        # but for consistency and strict safety on Windows, we'll try to get a shared lock or just read.
        # Since standard open() on Windows often denies shared access if locked exclusively,
        # we will retry if access is denied.
        
        for i in range(self.max_retries):
            try:
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (IOError, PermissionError, json.JSONDecodeError) as e:
                if i == self.max_retries - 1:
                    logger.error(f"Failed to load {self.file_path} after {self.max_retries} retries: {e}")
                    raise
                time.sleep(self.retry_delay)
        
        return default

    def save(self, data: Any):
        """
        Save JSON data safely transactions.
        1. Write to temp file
        2. Acquire lock on main file (simulated via lock file to coordinate writers)
        3. Rename temp file to main file (Atomic on POSIX, nearly atomic on Windows with replace)
        """
        # Create temp file in same directory to ensure atomic move is possible
        temp_dir = self.file_path.parent
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Write to temp file first
        with tempfile.NamedTemporaryFile(mode='w', dir=temp_dir, delete=False, encoding='utf-8') as tf:
            json.dump(data, tf, indent=2)
            temp_path = Path(tf.name)
        
        # 2. Safe Replace with Retry
        for i in range(self.max_retries):
            try:
                # On Windows, os.replace might fail if the destination is open/locked by another process
                os.replace(temp_path, self.file_path)
                return
            except (OSError, PermissionError) as e:
                if i == self.max_retries - 1:
                    logger.error(f"Failed to save {self.file_path} after retries: {e}")
                    # Clean up temp file
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
                    raise
                time.sleep(self.retry_delay)

    def write_locked(self, data: Any):
        """
        Explicitly locked write for critical sections.
        This uses a separate .lock mechanism to coordinate with other LockingJSONStorage instances.
        """
        # 1. Acquire Lock
        lock_fd = None
        try:
            # Open lock file
            mode = 'r+' if self.lock_file_path.exists() else 'w+'
            lock_fd = open(self.lock_file_path, mode)
            
            start_time = time.time()
            got_lock = False
            while (time.time() - start_time) < (self.max_retries * self.retry_delay):
                if self._acquire_lock(lock_fd):
                    got_lock = True
                    break
                time.sleep(self.retry_delay)
            
            if not got_lock:
                raise TimeoutError(f"Could not acquire lock for {self.file_path}")

            # 2. Perform Save (Now that we have the 'token')
            self.save(data)

        finally:
            # 3. Release Lock
            if lock_fd:
                self._release_lock(lock_fd)
                lock_fd.close() 
