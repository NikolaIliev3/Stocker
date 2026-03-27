import json

try:
    with open('data/predictions.json', 'r') as f:
        data = json.load(f)
    
    initial_count = len(data['predictions'])
    
    # We remove any active prediction that is:
    # - non-BUY
    # - BUY with confidence < 80
    valid_preds = []
    removed = 0
    for p in data['predictions']:
        if p['status'] == 'active':
            if p['action'] != 'BUY' or p['confidence'] < 80.0:
                removed += 1
                continue
        valid_preds.append(p)
        
    data['predictions'] = valid_preds
    
    with open('data/predictions.json', 'w') as f:
        json.dump(data, f, indent=4)
        
    print(f"Removed {removed} invalid active predictions.")
except Exception as e:
    print(f"Error: {e}")
