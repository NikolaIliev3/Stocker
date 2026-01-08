"""Systematically test configurations until achieving 50%+ validation accuracy"""
from ml_training import StockPredictionML
from training_pipeline import MLTrainingPipeline
from data_fetcher import StockDataFetcher
from config import APP_DATA_DIR
import re

def test_config(n_estimators, max_depth, min_samples_split, min_samples_leaf, feature_selection):
    """Test a configuration and return validation accuracy"""
    # Modify ml_training.py
    with open('ml_training.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Update rf_params
    pattern = r"rf_params = \{([^}]+)\}"
    replacement = f"""rf_params = {{
            'n_estimators': {n_estimators},
            'max_depth': {max_depth},
            'min_samples_split': {min_samples_split},
            'min_samples_leaf': {min_samples_leaf},
            'max_features': 'sqrt',
            'class_weight': 'balanced',
            'random_state': random_seed,
            'n_jobs': safe_n_jobs
        }}"""
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    with open('ml_training.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Generate samples
    df = StockDataFetcher()
    tp = MLTrainingPipeline(df, APP_DATA_DIR, app=None)
    samples = []
    for symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'V', 'JNJ']:
        s = tp.data_generator.generate_training_samples(symbol, '2023-01-01', '2024-12-31', 'trading', 10)
        samples.extend(s)
    samples.sort(key=lambda x: x.get('date', '1900-01-01'))
    
    # Train
    ml = StockPredictionML(APP_DATA_DIR, 'trading')
    results = ml.train(samples, False, False, feature_selection)
    
    return results.get('test_accuracy', 0), results.get('train_accuracy', 0)

# Test configurations systematically - focus on finding what works
configs = [
    # Strong regularization (best so far was 37.6% with depth 5)
    (100, 5, 25, 12, 'importance'),  # Current best approach
    (150, 5, 25, 12, 'importance'),  # More trees
    (200, 5, 25, 12, 'importance'),  # Even more trees
    
    # Try different feature selection methods with strong regularization
    (100, 5, 25, 12, 'rfe'),
    (100, 5, 25, 12, 'mutual_info'),
    
    # Slightly less restrictive
    (100, 6, 20, 10, 'importance'),
    (150, 6, 20, 10, 'importance'),
    
    # Try depth 4 (even more restrictive)
    (100, 4, 30, 15, 'importance'),
    (150, 4, 30, 15, 'importance'),
    
    # Moderate approach
    (100, 7, 18, 9, 'importance'),
    (150, 7, 18, 9, 'importance'),
]

print("="*70)
print("SYSTEMATIC TESTING FOR 50%+ VALIDATION ACCURACY")
print("="*70)

best_val = 0
best_config = None

for i, (n_est, depth, split, leaf, feat_sel) in enumerate(configs, 1):
    print(f"\nTest {i}/{len(configs)}: trees={n_est}, depth={depth}, split={split}, leaf={leaf}, feat_sel={feat_sel}")
    try:
        val_acc, train_acc = test_config(n_est, depth, split, leaf, feat_sel)
        print(f"  → Train: {train_acc*100:.1f}%, Val: {val_acc*100:.1f}%")
        
        if val_acc > best_val:
            best_val = val_acc
            best_config = (n_est, depth, split, leaf, feat_sel)
            print(f"  ★ NEW BEST!")
        
        if val_acc >= 0.50:
            print(f"\n{'='*70}")
            print(f"✅ SUCCESS! Achieved {val_acc*100:.1f}% validation accuracy!")
            print(f"Configuration: trees={n_est}, depth={depth}, split={split}, leaf={leaf}, feat_sel={feat_sel}")
            print(f"{'='*70}")
            break
    except Exception as e:
        print(f"  ✗ Error: {e}")

if best_val < 0.50:
    print(f"\n{'='*70}")
    print(f"⚠️ Could not reach 50%. Best: {best_val*100:.1f}%")
    if best_config:
        print(f"Best config: {best_config}")
    print(f"{'='*70}")
