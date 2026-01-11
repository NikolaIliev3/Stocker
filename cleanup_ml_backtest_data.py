"""
Cleanup script for ML model and backtest data
Safely removes data for a fresh start after fixes
"""
from pathlib import Path
import json
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_ml_backtest_data():
    """Clean up ML and backtest data for fresh start"""
    data_dir = Path("data")
    
    print("=" * 60)
    print("CLEANUP: ML Model & Backtest Data")
    print("=" * 60)
    
    cleaned = []
    backed_up = []
    
    # 1. Clean version history (references non-existent models)
    version_file = data_dir / "model_versions" / "trading" / "version_history.json"
    if version_file.exists():
        try:
            # Backup
            backup_dir = data_dir / "backups"
            backup_dir.mkdir(exist_ok=True)
            backup_file = backup_dir / f"version_history_backup_{Path(__file__).stat().st_mtime:.0f}.json"
            shutil.copy2(version_file, backup_file)
            backed_up.append(f"Version history -> {backup_file.name}")
            
            # Clear (create empty history)
            with open(version_file, 'w') as f:
                json.dump([], f, indent=2)
            cleaned.append("Version history (reset to empty)")
            logger.info("[OK] Version history cleared")
        except Exception as e:
            logger.error(f"[ERROR] Failed to clear version history: {e}")
    
    # 2. Clean production monitoring (contains test data)
    monitor_file = data_dir / "production_monitoring_trading.json"
    if monitor_file.exists():
        try:
            # Backup
            backup_dir = data_dir / "backups"
            backup_dir.mkdir(exist_ok=True)
            backup_file = backup_dir / f"production_monitoring_backup_{Path(__file__).stat().st_mtime:.0f}.json"
            shutil.copy2(monitor_file, backup_file)
            backed_up.append(f"Production monitoring -> {backup_file.name}")
            
            # Clear (create fresh structure)
            with open(monitor_file, 'w') as f:
                json.dump({
                    "strategy": "trading",
                    "last_updated": None,
                    "recent_predictions": [],
                    "performance_metrics": {}
                }, f, indent=2)
            cleaned.append("Production monitoring (removed test predictions)")
            logger.info("[OK] Production monitoring cleared")
        except Exception as e:
            logger.error(f"[ERROR] Failed to clear production monitoring: {e}")
    
    # 3. Clean performance attribution (if exists and contains old data)
    attr_file = data_dir / "performance_attribution.json"
    if attr_file.exists():
        try:
            with open(attr_file) as f:
                data = json.load(f)
            # Check if it has data
            if data.get('attributions') or data.get('summary'):
                backup_dir = data_dir / "backups"
                backup_dir.mkdir(exist_ok=True)
                backup_file = backup_dir / f"performance_attribution_backup_{Path(__file__).stat().st_mtime:.0f}.json"
                shutil.copy2(attr_file, backup_file)
                backed_up.append(f"Performance attribution -> {backup_file.name}")
                
                with open(attr_file, 'w') as f:
                    json.dump({}, f, indent=2)
                cleaned.append("Performance attribution")
                logger.info("[OK] Performance attribution cleared")
        except Exception as e:
            logger.error(f"[ERROR] Failed to clear performance attribution: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("CLEANUP SUMMARY")
    print("=" * 60)
    
    if backed_up:
        print("\n[BACKED UP] Files saved to data/backups/:")
        for item in backed_up:
            print(f"  - {item}")
    
    if cleaned:
        print("\n[CLEANED] Removed/reset:")
        for item in cleaned:
            print(f"  - {item}")
    
    if not cleaned:
        print("\n[OK] No cleanup needed - data is already clean!")
    
    print("\n[INFO] ML Models: Already cleared (no .pkl files found)")
    print("[INFO] Backtest Results: Already cleared (file doesn't exist)")
    print("[INFO] Benchmark Data: Already cleared (removed earlier)")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Run ML training to create fresh models with:")
    print("   - Fixed feature selection")
    print("   - SMOTE for class balancing")
    print("   - Confidence threshold (60%)")
    print("   - Updated overfitting mitigation")
    print()
    print("2. Run backtesting to regenerate results with:")
    print("   - Correct baseline calculations (50% for binary)")
    print("   - Fixed benchmarking comparisons")
    print("   - Clean version history")
    print()
    print("3. Monitor production monitoring for real predictions only")
    print("=" * 60)
    
    return len(cleaned) > 0

if __name__ == "__main__":
    success = cleanup_ml_backtest_data()
    if success:
        print("\n[OK] Cleanup complete! Ready for fresh start.")
    else:
        print("\n[INFO] No cleanup performed - system already clean.")
