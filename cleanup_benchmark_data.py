"""
Cleanup script to remove old benchmark data with incorrect baselines
This will ensure fresh baseline calculations with the corrected logic
"""
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_benchmark_data():
    """Remove old benchmark data file to force recalculation"""
    data_dir = Path("data")
    benchmark_file = data_dir / "model_benchmarks.json"
    
    if benchmark_file.exists():
        # Backup first
        backup_file = data_dir / "model_benchmarks_backup.json"
        try:
            with open(benchmark_file, 'r') as f:
                old_data = json.load(f)
            
            with open(backup_file, 'w') as f:
                json.dump(old_data, f, indent=2)
            logger.info(f"[OK] Backed up old benchmark data to {backup_file}")
            
            # Remove the file
            benchmark_file.unlink()
            logger.info(f"[OK] Removed {benchmark_file}")
            logger.info("   The file will be regenerated with correct baselines on next backtest run")
            
            # Check if comparisons are incorrect
            if old_data.get('baseline_performance', {}).get('random', {}).get('accuracy') == 33.3:
                logger.warning("   [WARNING] Old data had incorrect random baseline (33.3% instead of 50%)")
                logger.info("   [OK] This will be corrected in the next calculation")
            
            return True
        except Exception as e:
            logger.error(f"❌ Error cleaning up benchmark data: {e}")
            return False
    else:
        logger.info("✅ No benchmark data file found - already clean!")
        return True

    if __name__ == "__main__":
    print("=" * 60)
    print("CLEANUP: Benchmark Data")
    print("=" * 60)
    print("\nThis will:")
    print("  1. Backup old benchmark data")
    print("  2. Remove benchmark file (will regenerate with correct baselines)")
    print("\nOther data will be preserved:")
    print("  - Model version history (kept for tracking)")
    print("  - Production monitoring (kept for historical performance)")
    print("  - Performance attribution (kept for historical analysis)")
    print("  - A/B test data (kept for experimental results)")
    print("\nThe benchmark file will be automatically regenerated")
    print("with correct baselines (50% for binary) on next backtest run.")
    print("\n" + "=" * 60)
    
    success = cleanup_benchmark_data()
    
    if success:
        print("\n[OK] Cleanup complete!")
        print("   Run a backtest to regenerate benchmark data with correct baselines.")
    else:
        print("\n[ERROR] Cleanup failed. Check logs for details.")
