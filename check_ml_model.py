"""
Check ML Model Status
Shows current ML model accuracy and settings
"""
import json
from pathlib import Path
from config import APP_DATA_DIR

def check_ml_model():
    print("="*60)
    print("ML MODEL STATUS CHECK")
    print("="*60)
    
    data_dir = APP_DATA_DIR
    strategies = ['trading', 'mixed', 'investing']
    
    for strategy in strategies:
        print(f"\n[{strategy.upper()} Strategy]")
        print("-" * 60)
        
        # Check model file
        model_file = data_dir / f"ml_model_{strategy}.pkl"
        metadata_file = data_dir / f"ml_metadata_{strategy}.json"
        
        if model_file.exists() and metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                train_acc = metadata.get('train_accuracy', 0) * 100
                test_acc = metadata.get('test_accuracy', 0) * 100
                is_binary = metadata.get('use_binary_classification', False)
                model_family = metadata.get('model_family', 'unknown')
                num_features = metadata.get('num_features', 0)
                num_samples = metadata.get('num_training_samples', 0)
                
                print(f"  Model exists: YES")
                print(f"  Training accuracy: {train_acc:.1f}%")
                print(f"  Test/Validation accuracy: {test_acc:.1f}%")
                print(f"  Binary classification: {is_binary}")
                print(f"  Model family: {model_family}")
                print(f"  Features: {num_features}")
                print(f"  Training samples: {num_samples}")
                
                if test_acc < 55:
                    print(f"\n  [WARNING] Low accuracy ({test_acc:.1f}%) - model may need retraining")
                    if not is_binary:
                        print(f"  [SUGGESTION] Enable binary classification for better accuracy")
                elif test_acc >= 60:
                    print(f"\n  [GOOD] High accuracy ({test_acc:.1f}%) - model looks good!")
                else:
                    print(f"\n  [OK] Moderate accuracy ({test_acc:.1f}%)")
                    
            except Exception as e:
                print(f"  Error reading metadata: {e}")
        else:
            print(f"  Model exists: NO")
            print(f"  No model file found for {strategy}")
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("="*60)
    print("If your model shows <55% accuracy, you should:")
    print("  1. Delete the old model (it's not learning well)")
    print("  2. Retrain with:")
    print("     - Binary classification enabled")
    print("     - Diverse stock symbols (20+ stocks)")
    print("     - Long date range (2010-2023)")
    print("     - Voting ensemble model")
    print("="*60)

if __name__ == '__main__':
    check_ml_model()
