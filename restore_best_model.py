"""
Restore Best Model
Restores the 66.7% accuracy bull model from 2026-01-28T08:37
"""
import shutil
from pathlib import Path
import json

DATA_DIR = Path(r'C:\Users\Никола\.stocker')
VERSIONS_DIR = DATA_DIR / 'model_versions' / 'trading_bull'

# Target: The BEST model we had
BEST_VERSION = '20260128_083728'  # 66.7% test accuracy

def restore_model():
    print("🔍 Looking for best model version...")
    
    source_dir = VERSIONS_DIR / BEST_VERSION
    if not source_dir.exists():
        # Try with slightly different timestamp
        for d in VERSIONS_DIR.iterdir():
            if d.name.startswith('20260128_083'):
                source_dir = d
                break
    
    if not source_dir.exists():
        print(f"❌ Could not find version {BEST_VERSION}")
        print(f"Available versions:")
        for d in VERSIONS_DIR.iterdir():
            if d.is_dir():
                print(f"   - {d.name}")
        return
    
    print(f"✅ Found source: {source_dir.name}")
    
    # Copy model files
    files_to_copy = [
        ('ml_model_trading_bull.pkl', DATA_DIR / 'ml_model_trading_bull.pkl'),
        ('scaler_trading_bull.pkl', DATA_DIR / 'scaler_trading_bull.pkl'),
        ('label_encoder_trading_bull.pkl', DATA_DIR / 'label_encoder_trading_bull.pkl'),
    ]
    
    for filename, dest in files_to_copy:
        src = source_dir / filename
        if src.exists():
            shutil.copy(src, dest)
            print(f"   ✅ Restored {filename}")
        else:
            print(f"   ⚠️ Missing {filename}")
    
    # Also copy metadata
    meta_src = source_dir / 'ml_metadata_trading_bull.json'
    if meta_src.exists():
        shutil.copy(meta_src, DATA_DIR / 'ml_metadata_trading_bull.json')
        print(f"   ✅ Restored metadata")
    
    print(f"\n🎯 Model restored! Expected accuracy: ~66.7%")
    print(f"   Run backtest to verify: python run_backtest_manual.py --samples 200")

if __name__ == "__main__":
    restore_model()
