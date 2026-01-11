"""
Cleanup Test Data Script
Safely removes test data files created during component testing
"""
import json
from pathlib import Path
import shutil

def cleanup_test_data(keep_version_history=True):
    """Clean up test data files
    
    Args:
        keep_version_history: If True, keeps version history (default: True)
    """
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("No data directory found. Nothing to clean.")
        return
    
    files_to_remove = []
    dirs_to_remove = []
    
    # Test files created during component testing
    test_files = [
        "ab_tests_trading.json",
        "model_benchmarks.json",
        "performance_attribution.json",
        "production_monitoring_trading.json"
    ]
    
    # Check what exists
    for test_file in test_files:
        file_path = data_dir / test_file
        if file_path.exists():
            files_to_remove.append(file_path)
            print(f"  Found: {test_file}")
    
    # Check model_versions directory (test data)
    version_dir = data_dir / "model_versions" / "trading"
    if version_dir.exists():
        if keep_version_history:
            # Only remove test version files, keep structure
            version_files = list(version_dir.glob("*.json"))
            for vf in version_files:
                if "version_history" in vf.name:
                    # Check if it's test data (small file or recent test entries)
                    try:
                        with open(vf, 'r') as f:
                            data = json.load(f)
                            if isinstance(data, list) and len(data) > 0:
                                # Check if entries are from our tests
                                first_entry = data[0] if data else {}
                                if "20260110" in first_entry.get('created_at', ''):  # Test date
                                    files_to_remove.append(vf)
                                    print(f"  Found test version history: {vf.name}")
                    except:
                        pass
            
            # Remove test version backup directories
            for version_backup_dir in version_dir.iterdir():
                if version_backup_dir.is_dir() and "20260110" in version_backup_dir.name:
                    dirs_to_remove.append(version_backup_dir)
                    print(f"  Found test version backup: {version_backup_dir.name}")
        else:
            # Remove entire version directory if requested
            dirs_to_remove.append(version_dir)
            print(f"  Found version directory: {version_dir}")
    
    if not files_to_remove and not dirs_to_remove:
        print("✅ No test data files found. Nothing to clean.")
        return
    
    print(f"\n📋 Files/Directories to remove:")
    print(f"  Files: {len(files_to_remove)}")
    print(f"  Directories: {len(dirs_to_remove)}")
    
    # Confirm
    response = input("\n⚠️  Remove these test files? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("❌ Cleanup cancelled.")
        return
    
    # Remove files
    removed_count = 0
    for file_path in files_to_remove:
        try:
            file_path.unlink()
            print(f"✓ Removed: {file_path.name}")
            removed_count += 1
        except Exception as e:
            print(f"✗ Error removing {file_path.name}: {e}")
    
    # Remove directories
    for dir_path in dirs_to_remove:
        try:
            shutil.rmtree(dir_path)
            print(f"✓ Removed directory: {dir_path.name}")
            removed_count += 1
        except Exception as e:
            print(f"✗ Error removing {dir_path.name}: {e}")
    
    # Clean up empty directories
    try:
        version_parent = data_dir / "model_versions"
        if version_parent.exists():
            trading_dir = version_parent / "trading"
            if trading_dir.exists() and not any(trading_dir.iterdir()):
                trading_dir.rmdir()
                print(f"✓ Removed empty directory: trading")
            
            if version_parent.exists() and not any(version_parent.iterdir()):
                version_parent.rmdir()
                print(f"✓ Removed empty directory: model_versions")
    except:
        pass
    
    print(f"\n✅ Cleanup complete! Removed {removed_count} items.")
    print("\n💡 Note: Actual model files and backtesting data were not found.")
    print("   If you want to remove actual model files later, they would be:")
    print("   - ml_model_*.pkl")
    print("   - scaler_*.pkl")
    print("   - label_encoder_*.pkl")
    print("   - ml_metadata_*.json")
    print("   - backtest_results.json")

if __name__ == "__main__":
    print("=" * 60)
    print("CLEANUP TEST DATA")
    print("=" * 60)
    print("\nThis script will remove test data files created during")
    print("component testing. It will NOT remove actual model files")
    print("or backtesting data (none found).\n")
    
    cleanup_test_data(keep_version_history=True)
