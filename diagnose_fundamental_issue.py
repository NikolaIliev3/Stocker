"""
Diagnose fundamental issues preventing model from learning
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from config import APP_DATA_DIR
from training_pipeline import MLTrainingPipeline
from data_fetcher import StockDataFetcher
from ml_training import StockPredictionML
import numpy as np
from collections import Counter

def diagnose():
    """Diagnose why model can't learn"""
    print("="*70)
    print("DIAGNOSING FUNDAMENTAL ISSUES")
    print("="*70)
    
    # Generate samples
    data_fetcher = StockDataFetcher()
    pipeline = MLTrainingPipeline(data_fetcher, APP_DATA_DIR, app=None)
    
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    all_samples = []
    
    for symbol in symbols:
        samples = pipeline.data_generator.generate_training_samples(
            symbol, '2023-01-01', '2024-12-31', 'trading', 10
        )
        all_samples.extend(samples)
        print(f"{symbol}: {len(samples)} samples")
    
    if len(all_samples) < 50:
        print("ERROR: Not enough samples")
        return
    
    all_samples.sort(key=lambda x: x.get('date', '1900-01-01'))
    print(f"\nTotal samples: {len(all_samples)}")
    
    # Check label distribution
    labels = [s['label'] for s in all_samples]
    label_counts = Counter(labels)
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        pct = count / len(labels) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    # Check features
    features = np.array([s['features'] for s in all_samples])
    print(f"\nFeature stats:")
    print(f"  Shape: {features.shape}")
    print(f"  NaN count: {np.isnan(features).sum()}")
    print(f"  Inf count: {np.isinf(features).sum()}")
    print(f"  Mean: {np.nanmean(features, axis=0)[:5]}...")
    print(f"  Std: {np.nanstd(features, axis=0)[:5]}...")
    
    # Check if features vary
    feature_stds = np.nanstd(features, axis=0)
    zero_variance = (feature_stds < 1e-6).sum()
    print(f"  Zero/low variance features: {zero_variance}/{len(feature_stds)}")
    
    # Check label-feature correlation (simple check)
    print(f"\nChecking if labels are predictable...")
    
    # Split by label and check feature differences
    label_to_features = {}
    for label in set(labels):
        label_samples = [s for s in all_samples if s['label'] == label]
        label_features = np.array([s['features'] for s in label_samples])
        label_to_features[label] = label_features
    
    print(f"Feature means by label:")
    for label, feat_array in label_to_features.items():
        mean_feat = np.nanmean(feat_array, axis=0)
        print(f"  {label}: {mean_feat[:5]}...")
    
    # Check if means differ significantly
    if len(label_to_features) >= 2:
        labels_list = list(label_to_features.keys())
        feat1 = label_to_features[labels_list[0]]
        feat2 = label_to_features[labels_list[1]]
        
        mean_diff = np.abs(np.nanmean(feat1, axis=0) - np.nanmean(feat2, axis=0))
        print(f"\nMean feature difference between {labels_list[0]} and {labels_list[1]}:")
        print(f"  Max diff: {np.nanmax(mean_diff):.6f}")
        print(f"  Mean diff: {np.nanmean(mean_diff):.6f}")
        print(f"  Features with diff > 0.01: {(mean_diff > 0.01).sum()}")
        
        if np.nanmax(mean_diff) < 0.01:
            print("\n⚠️ WARNING: Features are nearly identical across labels!")
            print("   This means the model CANNOT learn - features don't predict labels!")
    
    # Try a simple baseline
    print(f"\n{'='*70}")
    print("TESTING SIMPLE BASELINE")
    print(f"{'='*70}")
    
    # Most common label baseline
    most_common_label = label_counts.most_common(1)[0][0]
    baseline_acc = label_counts[most_common_label] / len(labels)
    print(f"Most common label baseline: {baseline_acc*100:.1f}%")
    
    # Try training with minimal regularization
    print(f"\n{'='*70}")
    print("TESTING MINIMAL REGULARIZATION")
    print(f"{'='*70}")
    
    ml_model = StockPredictionML(APP_DATA_DIR, 'trading')
    
    # Temporarily modify to use minimal regularization
    import ml_training
    original_train = ml_training.StockPredictionML.train
    
    try:
        results = ml_model.train(
            all_samples,
            use_hyperparameter_tuning=False,
            use_regime_models=False,
            feature_selection_method='none'
        )
        
        print(f"\nResults:")
        print(f"  Training: {results.get('train_accuracy', 0)*100:.1f}%")
        print(f"  Validation: {results.get('test_accuracy', 0)*100:.1f}%")
        
        if results.get('test_accuracy', 0) < 0.40:
            print(f"\n❌ VALIDATION STILL BELOW 40%!")
            print(f"   This suggests a fundamental data/label problem.")
            print(f"   Possible issues:")
            print(f"   1. Labels are not predictable from features")
            print(f"   2. Features are not informative")
            print(f"   3. Data leakage or incorrect temporal split")
            print(f"   4. Label generation logic is wrong")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    diagnose()
