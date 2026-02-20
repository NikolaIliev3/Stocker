"""
Restore Backup Script
Restores the ML model from a specific backup version.
"""
import shutil
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import APP_DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('RestoreBackup')

def restore_backup(version_id):
    logger.info(f"🚀 RESTORING BACKUP VERSION: {version_id}")
    
    # Define paths
    backup_dir = APP_DATA_DIR / "model_versions" / "trading" / version_id
    target_dir = APP_DATA_DIR
    
    if not backup_dir.exists():
        logger.error(f"❌ Backup directory not found: {backup_dir}")
        return
    
    # Restore files
    files_restored = 0
    try:
        for file_path in backup_dir.glob("*"):
            if file_path.is_file():
                target_path = target_dir / file_path.name
                shutil.copy2(file_path, target_path)
                logger.info(f"   Restored: {file_path.name}")
                files_restored += 1
        
        logger.info(f"\n✅ Restore complete. {files_restored} files restored.")
        
    except Exception as e:
        logger.error(f"❌ Restore failed: {e}", exc_info=True)

if __name__ == "__main__":
    # Version ID verified from previous ls command
    restore_backup("20260128_131433")
