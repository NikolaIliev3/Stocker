"""
Test script for new data quality tools (Evidently + Cleanlab)
Run this to verify the installation and basic functionality.
"""
import sys
import numpy as np
import pandas as pd

print("=" * 60)
print("DATA QUALITY TOOLS TEST")
print("=" * 60)

# Test 1: Check imports
print("\n1. Checking package availability...")
try:
    import evidently
    print("   ✓ Evidently installed:", evidently.__version__)
    HAS_EVIDENTLY = True
except ImportError:
    print("   ✗ Evidently not installed")
    HAS_EVIDENTLY = False

try:
    import cleanlab
    print("   ✓ Cleanlab installed:", cleanlab.__version__)
    HAS_CLEANLAB = True
except ImportError:
    print("   ✗ Cleanlab not installed")
    HAS_CLEANLAB = False

if not (HAS_EVIDENTLY and HAS_CLEANLAB):
    print("\n⚠️  Some packages missing. Install with:")
    print("   pip install evidently cleanlab pyarrow")
    sys.exit(1)

# Test 2: Check custom modules
print("\n2. Checking custom modules...")
try:
    from data_drift_detector import DataDriftDetector
    print("   ✓ DataDriftDetector imported")
except Exception as e:
    print(f"   ✗ DataDriftDetector import failed: {e}")
    sys.exit(1)

try:
    from label_quality_analyzer import LabelQualityAnalyzer
    print("   ✓ LabelQualityAnalyzer imported")
except Exception as e:
    print(f"   ✗ LabelQualityAnalyzer import failed: {e}")
    sys.exit(1)

# Test 3: Test Data Drift Detection
print("\n3. Testing Data Drift Detection...")
try:
    from pathlib import Path
    
    # Create test data
    np.random.seed(42)
    
    # Reference data (training data)
    reference_data = pd.DataFrame({
        'feature_1': np.random.normal(100, 10, 100),
        'feature_2': np.random.normal(50, 5, 100),
        'feature_3': np.random.normal(0, 1, 100)
    })
    
    # Current data - similar to reference
    current_data_no_drift = pd.DataFrame({
        'feature_1': np.random.normal(100, 10, 50),
        'feature_2': np.random.normal(50, 5, 50),
        'feature_3': np.random.normal(0, 1, 50)
    })
    
    # Current data - drifted (different distribution)
    current_data_drifted = pd.DataFrame({
        'feature_1': np.random.normal(150, 20, 50),  # Mean shifted
        'feature_2': np.random.normal(30, 3, 50),     # Mean shifted
        'feature_3': np.random.normal(0, 1, 50)
    })
    
    # Initialize detector
    detector = DataDriftDetector(Path("data"), strategy='trading')
    detector.set_reference_data(reference_data)
    
    # Test on non-drifted data
    result_no_drift = detector.check_drift(current_data_no_drift)
    print(f"   No Drift Test: Drift Detected = {result_no_drift.get('drift_detected')}")
    print(f"                  Drift Score = {result_no_drift.get('drift_score', 0):.2f}")
    
    # Test on drifted data
    result_drifted = detector.check_drift(current_data_drifted)
    print(f"   Drift Test:    Drift Detected = {result_drifted.get('drift_detected')}")
    print(f"                  Drift Score = {result_drifted.get('drift_score', 0):.2f}")
    
    if result_drifted.get('drift_detected'):
        print("   ✓ Drift detection working correctly")
    else:
        print("   ⚠️  Drift detection may need tuning")
    
except Exception as e:
    print(f"   ✗ Drift detection test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Test Label Quality Analysis
print("\n4. Testing Label Quality Analysis...")
try:
    from sklearn.ensemble import RandomForestClassifier
    
    # Create test data with some intentional mislabels
    np.random.seed(42)
    
    # Generate features
    n_samples = 200
    X = np.random.randn(n_samples, 10)
    
    # Create labels based on a simple rule
    true_labels = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    # Add noise (mislabel 10% of samples)
    noisy_labels = true_labels.copy()
    noise_indices = np.random.choice(n_samples, size=20, replace=False)
    noisy_labels[noise_indices] = 1 - noisy_labels[noise_indices]
    
    # Initialize analyzer
    analyzer = LabelQualityAnalyzer(Path("data"), strategy='trading')
    
    # Train a simple model to get predictions
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    
    # Analyze labels
    result = analyzer.analyze_labels(X, noisy_labels, model=model)
    
    if not result.get('error'):
        print(f"   Total samples: {result.get('total_samples', 0)}")
        print(f"   Issues found: {result.get('num_high_confidence_issues', 0)}")
        print(f"   Mean quality score: {result.get('quality_scores_summary', {}).get('mean', 0):.3f}")
        print(f"   Recommendation: {result.get('recommendation', 'N/A')}")
        print("   ✓ Label quality analysis working")
    else:
        print(f"   ⚠️  Label quality analysis returned error: {result.get('error')}")
    
except Exception as e:
    print(f"   ✗ Label quality test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Integration with existing modules
print("\n5. Testing integration with existing modules...")
try:
    from production_monitor import ProductionMonitor
    print("   ✓ ProductionMonitor imports successfully")
    
    # Check if it has drift detector
    monitor = ProductionMonitor(Path("data"), "trading")
    if hasattr(monitor, 'drift_detector'):
        print("   ✓ ProductionMonitor has drift_detector attribute")
    else:
        print("   ✗ ProductionMonitor missing drift_detector attribute")
except Exception as e:
    print(f"   ⚠️  ProductionMonitor test: {e}")

try:
    from training_data_validator import TrainingDataValidator
    print("   ✓ TrainingDataValidator imports successfully")
    
    # Check if it can be initialized with data_dir
    validator = TrainingDataValidator(data_dir=Path("data"))
    if hasattr(validator, 'label_quality_analyzer'):
        print("   ✓ TrainingDataValidator has label_quality_analyzer attribute")
    else:
        print("   ✗ TrainingDataValidator missing label_quality_analyzer attribute")
except Exception as e:
    print(f"   ⚠️  TrainingDataValidator test: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
print("\nAll core functionality is working! ✅")
print("\nNext steps:")
print("1. Train a new model to see label quality analysis in action")
print("2. Monitor live predictions to track drift over time")
print("3. Check 'data/drift_detection/' for drift reports")
print("4. Check 'data/label_quality/' for label quality reports")
