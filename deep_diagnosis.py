"""Deep diagnosis of why model can't learn - find root cause"""
from ml_training import StockPredictionML
from training_pipeline import MLTrainingPipeline
from data_fetcher import StockDataFetcher
from config import APP_DATA_DIR
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Generate samples
df = StockDataFetcher()
tp = MLTrainingPipeline(df, APP_DATA_DIR, app=None)
samples = []
for symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA']:
    s = tp.data_generator.generate_training_samples(symbol, '2023-01-01', '2024-12-31', 'trading', 10)
    samples.extend(s)

samples.sort(key=lambda x: x.get('date', '1900-01-01'))
print(f"Total samples: {len(samples)}\n")

# Extract features and labels
X = np.array([s['features'] for s in samples])
y = np.array([s['label'] for s in samples])

print("="*70)
print("1. LABEL DISTRIBUTION")
print("="*70)
label_counts = Counter(y)
for label, count in label_counts.items():
    pct = count / len(y) * 100
    print(f"  {label}: {count} ({pct:.1f}%)")

print(f"\n{'='*70}")
print("2. FEATURE QUALITY")
print("="*70)
print(f"  Shape: {X.shape}")
print(f"  NaN count: {np.isnan(X).sum()}")
print(f"  Inf count: {np.isinf(X).sum()}")

# Check feature variance
feature_stds = np.nanstd(X, axis=0)
zero_var = (feature_stds < 1e-6).sum()
print(f"  Zero/low variance features: {zero_var}/{len(feature_stds)}")

# Check if features differ by label
print(f"\n{'='*70}")
print("3. FEATURE-LABEL RELATIONSHIP")
print("="*70)
label_to_features = {}
for label in set(y):
    mask = y == label
    label_to_features[label] = X[mask]

print("Mean features by label (first 10 features):")
for label, feat_array in label_to_features.items():
    mean_feat = np.nanmean(feat_array, axis=0)
    print(f"  {label}: {mean_feat[:10]}")

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
    print(f"  Features with diff > 0.1: {(mean_diff > 0.1).sum()}")

print(f"\n{'='*70}")
print("4. TESTING WITH RANDOM SPLIT (should work if features are predictive)")
print("="*70)

# Try random split to see if features are predictive at all
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Simple model
rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)

train_pred = rf.predict(X_train)
test_pred = rf.predict(X_test)

train_acc = accuracy_score(y_train, train_pred)
test_acc = accuracy_score(y_test, test_pred)

print(f"Random split results:")
print(f"  Training: {train_acc*100:.1f}%")
print(f"  Test: {test_acc*100:.1f}%")

if test_acc < 0.40:
    print("\n⚠️ WARNING: Even with random split, accuracy is low!")
    print("   This suggests features are NOT predictive of labels.")
    print("   Possible causes:")
    print("   1. Label generation logic is wrong")
    print("   2. Features don't contain predictive signal")
    print("   3. Lookforward period doesn't match feature calculation")

print(f"\n{'='*70}")
print("5. TESTING WITH TEMPORAL SPLIT (current method)")
print("="*70)

# Temporal split
split_idx = int(len(X) * 0.8)
X_train_temp = X[:split_idx]
X_test_temp = X[split_idx:]
y_train_temp = y[:split_idx]
y_test_temp = y[split_idx:]

rf_temp = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight='balanced')
rf_temp.fit(X_train_temp, y_train_temp)

train_pred_temp = rf_temp.predict(X_train_temp)
test_pred_temp = rf_temp.predict(X_test_temp)

train_acc_temp = accuracy_score(y_train_temp, train_pred_temp)
test_acc_temp = accuracy_score(y_test_temp, test_pred_temp)

print(f"Temporal split results:")
print(f"  Training: {train_acc_temp*100:.1f}%")
print(f"  Test: {test_acc_temp*100:.1f}%")

if test_acc_temp < 0.40 and test_acc > 0.50:
    print("\n⚠️ WARNING: Temporal split performs much worse than random split!")
    print("   This suggests temporal distribution shift or data leakage issues.")

print(f"\n{'='*70}")
print("6. CHECKING LABEL GENERATION")
print("="*70)

# Check a few samples to see if labels make sense
print("Sample labels and dates (first 10):")
for i, sample in enumerate(samples[:10]):
    date = sample.get('date', 'N/A')
    label = sample.get('label', 'N/A')
    print(f"  {i+1}. Date: {date}, Label: {label}")

print(f"\n{'='*70}")
print("DIAGNOSIS COMPLETE")
print(f"{'='*70}")
