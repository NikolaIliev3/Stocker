"""
Secure Deletion Module
Securely deletes files by overwriting with random data
"""
import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def secure_delete_file(file_path: Path, passes: int = 3) -> bool:
    """
    Securely delete file by overwriting with random data
    
    Args:
        file_path: Path to file to delete
        passes: Number of overwrite passes (default: 3)
    
    Returns:
        True if successful, False otherwise
    """
    if not file_path.exists():
        return True
    
    try:
        file_size = file_path.stat().st_size
        
        # Open file in binary append mode for overwriting
        with open(file_path, 'ba+', buffering=0) as f:
            for pass_num in range(passes):
                f.seek(0)
                # Write random data
                random_data = os.urandom(file_size)
                f.write(random_data)
                f.flush()
                os.fsync(f.fileno())  # Force write to disk
        
        # Finally delete
        file_path.unlink()
        logger.debug(f"Securely deleted {file_path} ({passes} passes)")
        return True
        
    except PermissionError:
        logger.error(f"Permission denied deleting {file_path}")
        return False
    except Exception as e:
        logger.error(f"Error securely deleting {file_path}: {e}")
        return False


def secure_delete_directory(dir_path: Path, passes: int = 3) -> bool:
    """
    Securely delete directory and all contents
    
    Args:
        dir_path: Path to directory to delete
        passes: Number of overwrite passes for files
    
    Returns:
        True if successful, False otherwise
    """
    if not dir_path.exists() or not dir_path.is_dir():
        return True
    
    try:
        # Delete all files in directory
        for item in dir_path.iterdir():
            if item.is_file():
                secure_delete_file(item, passes)
            elif item.is_dir():
                secure_delete_directory(item, passes)
        
        # Remove empty directory
        dir_path.rmdir()
        logger.debug(f"Securely deleted directory {dir_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error securely deleting directory {dir_path}: {e}")
        return False
