"""
Check if ML model was retrained from verified predictions (which had the bug)
"""
import json
from pathlib import Path
from config import APP_DATA_DIR
from datetime import datetime

def check_retraining_history():
    print("="*60)
    print("CHECKING IF MODEL WAS RETRAINED FROM VERIFIED PREDICTIONS")
    print("="*60)
    
    data_dir = APP_DATA_DIR
    metadata_file = data_dir / "ml_metadata_trading.json"
    
    if not metadata_file.exists():
        print("\nNo ML model found - model was never trained")
        return
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        trained_date = metadata.get('trained_date', 'Unknown')
        test_acc = metadata.get('test_accuracy', 0) * 100
        is_binary = metadata.get('use_binary_classification', False)
        num_samples = metadata.get('num_training_samples', 0)
        
        print(f"\n[CURRENT MODEL INFO]")
        print(f"  Trained date: {trained_date}")
        print(f"  Test accuracy: {test_acc:.1f}%")
        print(f"  Binary classification: {is_binary}")
        print(f"  Training samples: {num_samples}")
        
        # Check if there are verified predictions
        predictions_file = data_dir / "predictions.json"
        verified_count = 0
        if predictions_file.exists():
            try:
                with open(predictions_file, 'r') as f:
                    data = json.load(f)
                    predictions = data.get('predictions', [])
                    verified_count = sum(1 for p in predictions if p.get('verified'))
            except:
                pass
        
        print(f"\n[VERIFIED PREDICTIONS]")
        print(f"  Count: {verified_count}")
        
        print("\n" + "="*60)
        print("ANALYSIS")
        print("="*60)
        print("The bug ONLY affected retraining from verified predictions.")
        print("Main training from historical data has CORRECT logic.")
        print("\nIf you:")
        print("  - Only trained from historical data → Model is FINE (keep it)")
        print("  - Retrained using 'verified predictions' option → Model might be contaminated")
        print("\nRECOMMENDATION:")
        if verified_count > 0:
            print("  ⚠️  You have verified predictions, so you might have retrained.")
            print("  → SAFE OPTION: Clear model and retrain from scratch (historical data only)")
            print("  → This ensures no contamination from the bug")
        else:
            print("  ✓ No verified predictions found - model is likely clean")
            print("  → You can keep the current model")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\nError reading metadata: {e}")

if __name__ == '__main__':
    check_retraining_history()
