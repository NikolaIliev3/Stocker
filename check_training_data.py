"""
Check ML Training Data
Shows what training-related data exists and whether it should be kept
"""
import json
from pathlib import Path
from config import APP_DATA_DIR
from datetime import datetime, timedelta

def check_training_data():
    print("="*60)
    print("ML TRAINING DATA ANALYSIS")
    print("="*60)
    
    data_dir = APP_DATA_DIR
    
    # Check verified predictions (VALUABLE - used for retraining)
    predictions_file = data_dir / "predictions.json"
    verified_count = 0
    active_count = 0
    if predictions_file.exists():
        try:
            with open(predictions_file, 'r') as f:
                data = json.load(f)
                predictions = data.get('predictions', [])
                for pred in predictions:
                    if pred.get('verified'):
                        verified_count += 1
                    if pred.get('status') == 'active':
                        active_count += 1
            print(f"\n[PREDICTIONS.JSON]")
            print(f"  Total predictions: {len(predictions)}")
            print(f"  Verified: {verified_count}")
            print(f"  Active: {active_count}")
            print(f"  Status: KEEP - Used for ML retraining (improves accuracy)")
        except Exception as e:
            print(f"\n[PREDICTIONS.JSON] Error: {e}")
    else:
        print(f"\n[PREDICTIONS.JSON] Not found")
    
    # Check learning tracker (statistics only)
    learning_file = data_dir / "learning_tracker.json"
    if learning_file.exists():
        try:
            with open(learning_file, 'r') as f:
                data = json.load(f)
                total_samples = data.get('knowledge', {}).get('total_training_samples', 0)
                total_verified = data.get('knowledge', {}).get('total_predictions_verified', 0)
            print(f"\n[LEARNING_TRACKER.JSON]")
            print(f"  Total training samples: {total_samples}")
            print(f"  Total verified: {total_verified}")
            print(f"  Status: OPTIONAL - Just statistics, doesn't affect training")
        except Exception as e:
            print(f"\n[LEARNING_TRACKER.JSON] Error: {e}")
    else:
        print(f"\n[LEARNING_TRACKER.JSON] Not found")
    
    # Check current ML model
    model_file = data_dir / "ml_model_trading.pkl"
    if model_file.exists():
        metadata_file = data_dir / "ml_metadata_trading.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    test_acc = metadata.get('test_accuracy', 0) * 100
                    is_binary = metadata.get('use_binary_classification', False)
                print(f"\n[CURRENT ML MODEL]")
                print(f"  Test accuracy: {test_acc:.1f}%")
                print(f"  Binary classification: {is_binary}")
                print(f"  Status: KEEP - This is your good model!")
            except Exception as e:
                print(f"\n[CURRENT ML MODEL] Error: {e}")
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print("DO NOT clear ML training progress because:")
    print("  1. Verified predictions help improve model accuracy")
    print("  2. They're used when you retrain with 'verified predictions' option")
    print("  3. Your current model (60.6%) is good - keep it!")
    print("\nOnly clear if:")
    print("  - You want a completely fresh start (lose all learning)")
    print("  - Verified predictions are very old (>1 year) and inaccurate")
    print("  - You're debugging and need to start from scratch")
    print("="*60)

if __name__ == '__main__':
    check_training_data()
