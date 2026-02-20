"""
Diagnostic to analyze which features are being selected by RFE
and why predictions may be identical
"""
import sys
sys.path.insert(0, '.')

import numpy as np
from pathlib import Path
from ml_training import StockPredictionML, FeatureExtractor

def analyze_selected_features():
    """Check which features are selected and their indices"""
    print(f"\n{'='*60}")
    print("SELECTED FEATURES ANALYSIS")
    print(f"{'='*60}\n")
    
    # Load main model
    data_dir = Path.home() / '.stocker'
    ml = StockPredictionML(data_dir, 'trading')
    
    if not ml.is_trained():
        print("❌ Model not trained!")
        return
    
    # Get feature names
    extractor = FeatureExtractor()
    all_feature_names = extractor.get_feature_names()
    print(f"Total available features: {len(all_feature_names)}")
    
    # Check each regime model
    for regime, regime_ml in ml.regime_models.items():
        print(f"\n--- {regime.upper()} REGIME ---")
        selected = regime_ml.selected_features
        if selected is not None:
            print(f"  Selected {len(selected)} features:")
            for idx in selected:
                if idx < len(all_feature_names):
                    print(f"    [{idx:3d}] {all_feature_names[idx]}")
                else:
                    print(f"    [{idx:3d}] <OUT OF RANGE>")
        else:
            print("  No feature selection applied")
        
        # Check metadata for selected feature names
        if regime_ml.metadata and 'selected_feature_names' in regime_ml.metadata:
            print(f"\n  Stored feature names from metadata:")
            for name in regime_ml.metadata['selected_feature_names'][:5]:
                print(f"    - {name}")
            print(f"    ... and {len(regime_ml.metadata['selected_feature_names']) - 5} more")
    
    # Check if the same indices are used for all regimes
    print(f"\n{'='*60}")
    print("FEATURE OVERLAP BETWEEN REGIMES")
    print(f"{'='*60}\n")
    
    regime_features = {}
    for regime, regime_ml in ml.regime_models.items():
        if regime_ml.selected_features is not None:
            regime_features[regime] = set(regime_ml.selected_features)
    
    # Compare
    regimes = list(regime_features.keys())
    for i in range(len(regimes)):
        for j in range(i+1, len(regimes)):
            r1, r2 = regimes[i], regimes[j]
            common = regime_features[r1] & regime_features[r2]
            total = len(regime_features[r1])
            pct = len(common) / total * 100 if total > 0 else 0
            print(f"  {r1} ∩ {r2}: {len(common)}/{total} features ({pct:.0f}%) in common")

if __name__ == '__main__':
    analyze_selected_features()
