"""
Ingest Backtest Results
Reads backtest_results.json and populates predictions.json to enable training.
"""
import json
from pathlib import Path
from datetime import datetime

def ingest():
    data_dir = Path.home() / ".stocker"
    backtest_file = data_dir / "backtest_results.json"
    preds_file = data_dir / "predictions.json"
    
    if not backtest_file.exists():
        print("❌ backtest_results.json not found")
        return

    print("📂 Reading backtest results...")
    with open(backtest_file, 'r') as f:
        bt_data = json.load(f)
        
    test_details = bt_data.get('test_details', [])
    print(f"   • Found {len(test_details)} backtest samples")
    
    # Load existing predictions (should be empty/sparse)
    predictions_dict = {} # Use dict to deduplicate by ID
    if preds_file.exists():
        with open(preds_file, 'r') as f:
            try:
                data = json.load(f)
                preds_list = data.get('predictions', [])
                for p in preds_list:
                    # Create key for existing pred to allow updates
                    key = f"{p.get('symbol')}_{p.get('timestamp', '').split('T')[0]}"
                    predictions_dict[key] = p
            except:
                predictions_dict = {}

    count = 0
    for sample in test_details:
        # Create a unique ID for the prediction
        symbol = sample.get('symbol')
        date_raw = sample.get('test_date', '')
        date = date_raw.split('T')[0] if 'T' in date_raw else date_raw
        if not symbol or not date: continue
        
        pred_id = f"{symbol}_{date}"
        
        # Determine actual verification status
        was_correct = sample.get('was_correct', False)
        
        # Map backtest result to Prediction format
        # IMPORTANT: We store the "Outcome" so the trainer sees it as a verified trade
        predictions_dict[pred_id] = {
            "id": count + 1,
            "symbol": symbol,
            "timestamp": f"{date}T09:30:00",
            "strategy": sample.get('strategy', 'trading'),
            "action": sample.get('predicted_action', 'HOLD'),
            "recommendation": {
                "action": sample.get('predicted_action', 'HOLD'),
                "confidence": sample.get('confidence', 0),
                "target_price": sample.get('future_price', 0), 
                "stop_loss": sample.get('test_price', 0) * 0.95
            },
            "entry_price": sample.get('test_price', 0),
            "target_price": sample.get('future_price', 0),
            "stop_loss": sample.get('test_price', 0) * 0.95,
            "confidence": sample.get('confidence', 0),
            "status": "verified",
            "verified": True,
            "was_correct": was_correct,
            "actual_price_at_target": sample.get('future_price', 0),
            "actual_return_pct": ((sample.get('future_price', 0) - sample.get('test_price', 0)) / sample.get('test_price', 1)) * 100,
            "verification_date": datetime.now().isoformat(),
            "source": "backtest_bootstrap"
        }
        count += 1
        
    print(f"💾 Saving {count} samples to predictions.json...")
    output_data = {
        "predictions": list(predictions_dict.values()),
        "last_updated": datetime.now().isoformat()
    }
    with open(preds_file, 'w') as f:
        json.dump(output_data, f, indent=2)
        
    print("✅ Ingestion Complete. Now run run_quant_training.py")

if __name__ == "__main__":
    ingest()
