import json
import numpy as np

def test_json_serialization():
    data = {
        "symbol": "AAPL",
        "estimated_days": np.int64(14),
        "confidence": np.float64(85.5)
    }
    
    try:
        print("Attempting to serialize numpy types...")
        json_str = json.dumps(data)
        print("Success:", json_str)
    except TypeError as e:
        print("❌ Caught expected TypeError:", e)

if __name__ == "__main__":
    test_json_serialization()
