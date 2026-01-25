import numpy as np
import pandas as pd
from pathlib import Path
from data_drift_detector import DataDriftDetector

if __name__ == "__main__":
    print("=== DRIFT DETECTION TEST ===")
    np.random.seed(42)

    # Create reference (training) data
    reference_data = pd.DataFrame({
        'rsi': np.random.normal(50, 15, 100),
        'macd': np.random.normal(0, 2, 100),
        'volatility': np.random.normal(2, 0.5, 100),
        'price': np.random.normal(100, 10, 100),
        'volume': np.random.normal(1000, 100, 100)
    })

    # Current data - similar (no drift)
    current_no_drift = pd.DataFrame({
        'rsi': np.random.normal(50, 15, 50),
        'macd': np.random.normal(0, 2, 50),
        'volatility': np.random.normal(2, 0.5, 50),
        'price': np.random.normal(100, 10, 50),
        'volume': np.random.normal(1000, 100, 50)
    })

    # Current data - drifted (market changed!)
    current_drifted = pd.DataFrame({
        'rsi': np.random.normal(70, 10, 50),  # RSI shifted high
        'macd': np.random.normal(-3, 1, 50),  # MACD negative
        'volatility': np.random.normal(5, 1, 50),  # Higher volatility
        'price': np.random.normal(100, 10, 50),
        'volume': np.random.normal(1000, 100, 50)
    })

    detector = DataDriftDetector(Path('data'), 'trading')
    detector.set_reference_data(reference_data)

    # Test 1: No drift
    result1 = detector.check_drift(current_no_drift)
    if result1.get('error'):
        print(f"Drift Check Failed: {result1['error']}")
    else:
        print("Similar data - Drift:", result1.get("drift_detected", False), "Score:", round(result1.get("drift_score", 0), 2))

    # Test 2: With drift  
    result2 = detector.check_drift(current_drifted)
    if result2.get('error'):
        print(f"Drift Check Failed: {result2['error']}")
    else:
        print("Changed data - Drift:", result2.get("drift_detected", False), "Score:", round(result2.get("drift_score", 0), 2))

    print("")
    print("=== LABEL QUALITY TEST ===")
    from label_quality_analyzer import LabelQualityAnalyzer
    from sklearn.ensemble import RandomForestClassifier

    # Create training data with some intentional mislabels
    np.random.seed(42)
    n_samples = 200
    X = np.random.randn(n_samples, 10)
    true_labels = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Add noise (mislabel 10% of samples)
    noisy_labels = true_labels.copy()
    noise_indices = np.random.choice(n_samples, size=20, replace=False)
    noisy_labels[noise_indices] = 1 - noisy_labels[noise_indices]

    analyzer = LabelQualityAnalyzer(Path('data'), 'trading')
    model = RandomForestClassifier(n_estimators=10, random_state=42)

    result = analyzer.analyze_labels(X, noisy_labels, model=model)

    if result.get('error'):
        print(f"Label Analysis Failed: {result['error']}")
    else:
        print("Samples analyzed:", result.get('total_samples', 0))
        print("Issues found:", result.get('num_high_confidence_issues', 0))
        print("Mean quality:", round(result.get('quality_scores_summary', {}).get('mean', 0), 3))
        print("")
        print("SUCCESS: Cleanlab is working!")
        if result1.get('drift_detected') is False and not result1.get('error'):
            # Just a check for Evidently
            pass
