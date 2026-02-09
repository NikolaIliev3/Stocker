
import os
import joblib
import json
import logging
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('BearAudit')

def audit_bear_model():
    model_dir = Path("C:/Users/Никола/.stocker")
    scaler_path = model_dir / "scaler_trading_bear.pkl"
    metadata_path = model_dir / "ml_metadata_trading.json"
    
    logger.info("Starting Bear-Alpha Structural Audit...")
    
    if not scaler_path.exists():
        logger.error(f"Scaler not found: {scaler_path}")
        return
    
    # 1. Check Scaler Dimensions
    try:
        scaler = joblib.load(scaler_path)
        expected_features = scaler.n_features_in_
        logger.info(f"🔍 SCALER AUDIT: n_features_in = {expected_features}")
        
        if expected_features == 190:
            logger.info("✅ SUCCESS: Scaler maintains 190-feature global dimensionality.")
        else:
            logger.error(f"❌ FAILURE: Scaler dimensionality mismatch! Expected 190, got {expected_features}")
    except Exception as e:
        logger.error(f"Failed to load scaler: {e}")
    
    # 2. Check Metadata Consistency
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                bear_meta = metadata.get('regime_metadata', {}).get('bear', {})
                scaler_n = bear_meta.get('scaler_n_features')
                logger.info(f"🔍 METADATA AUDIT: bear.scaler_n_features = {scaler_n}")
                
                if scaler_n == 190:
                    logger.info("✅ SUCCESS: Metadata correctly reflects 190-feature requirement.")
                else:
                    logger.warning(f"⚠️ METADATA MISMATCH: Metadata says {scaler_n}, but we need 190.")
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
    else:
        logger.warning(f"Metadata file missing: {metadata_path}")

if __name__ == "__main__":
    audit_bear_model()
