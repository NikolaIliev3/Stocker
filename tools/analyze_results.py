import json
from pathlib import Path

def analyze():
    data_dir = Path.home() / ".stocker"
    bt_file = data_dir / "backtest_results.json"
    
    if not bt_file.exists():
        print("❌ No backtest results found.")
        return

    with open(bt_file, 'r') as f:
        data = json.load(f)
        
    details = data.get('test_details', [])
    total_samples = len(details)
    overall_correct = sum(1 for p in details if p.get('was_correct'))
    overall_accuracy = (overall_correct / total_samples * 100) if total_samples > 0 else 0
    
    print(f"📊 Overall Backtest Results:")
    print(f"   • Total Samples: {total_samples}")
    print(f"   • Correct Predictions: {overall_correct}")
    print(f"   • Overall Accuracy: {overall_accuracy:.1f}%")
    print("-" * 30)
    
    # Confidence breakdown
    conf_buckets = {}
    for p in details:
        conf = p.get('confidence', 0)
        # Bucket by 10s or keep exact? Let's keep exact for now since CONF is usually round numbers like 50, 60, 70, 80, 90
        if conf not in conf_buckets:
            conf_buckets[conf] = {'total': 0, 'correct': 0}
        conf_buckets[conf]['total'] += 1
        if p.get('was_correct'):
            conf_buckets[conf]['correct'] += 1
            
    print("📈 Accuracy by Confidence Level:")
    for conf in sorted(conf_buckets.keys(), reverse=True):
        stats = conf_buckets[conf]
        acc = (stats['correct'] / stats['total'] * 100)
        print(f"   • {conf}% Confidence: {acc:.1f}% Accuracy ({stats['correct']}/{stats['total']})")

if __name__ == "__main__":
    analyze()
