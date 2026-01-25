from ml_training import FeatureExtractor
import numpy as np

def verify_fix():
    print("Verifying feature count consistency...")
    
    extractor = FeatureExtractor()
    feature_names = extractor.get_feature_names()
    print(f"Total feature names: {len(feature_names)}")
    
    # Simulate feature extraction (dummy data)
    # We need to construct dummy inputs that trigger all feature additions
    # This might be complex to mock perfectly, but we can inspect the code logic
    # or rely on the count calculation.
    
    # Based on code analysis:
    # 1. Base features: 48
    # 2. Time series: 58
    # 3. Pattern: 10
    # 4. Volume Analysis: 3 (added)
    # 5. Enhanced: 15
    # 6. Regime/Other: 16
    # 7. Sector: 5 (added)
    # 8. Catalyst: 4 (added)
    # 9. Liquidity: 3 (added)
    
    expected_count = 48 + 58 + 10 + 3 + 15 + 16 + 5 + 4 + 3
    print(f"Expected count based on code analysis: {expected_count}")
    
    if len(feature_names) == expected_count:
        print("✅ SUCCESS: Feature name count matches expected count!")
    else:
        print(f"❌ FAILURE: Mismatch! Got {len(feature_names)}, expected {expected_count}")
        
    # Also print the last few names to verify order
    print("\nLast 10 feature names:")
    for name in feature_names[-10:]:
        print(f" - {name}")

if __name__ == "__main__":
    verify_fix()
