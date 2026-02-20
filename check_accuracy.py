
import json
import os

path = os.path.expanduser(r'~/.stocker/backtest_results.json')
if not os.path.exists(path):
    print(f"File not found: {path}")
else:
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        
        details = data.get('test_details', [])
        total = len(details)
        if total == 0:
            print("No test details found.")
        else:
            correct = sum(1 for x in details if x.get('was_correct', False))
            print(f"Total Tests: {total}")
            print(f"Correct: {correct}")
            print(f"Accuracy: {correct/total*100:.2f}%")
            
            # Count by prediction type
            buys = sum(1 for x in details if x['predicted_action'] == 'BUY')
            holds = sum(1 for x in details if x['predicted_action'] == 'HOLD')
            sells = sum(1 for x in details if x['predicted_action'] == 'SELL')
            print(f"Buys: {buys}, Holds: {holds}, Sells: {sells}")
            
    except Exception as e:
        print(f"Error: {e}")
