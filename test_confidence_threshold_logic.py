"""
Test the confidence threshold logic directly
"""
import numpy as np

def test_confidence_threshold_logic():
    """Test the confidence threshold filtering logic"""
    print("=" * 60)
    print("TESTING CONFIDENCE THRESHOLD LOGIC")
    print("=" * 60)
    
    # Simulate binary classification scenario
    classes = ['UP', 'DOWN']
    is_binary_mode = set(classes) == {'UP', 'DOWN'}
    
    print(f"\n1. Binary mode check: {is_binary_mode}")
    print(f"   Classes: {classes}")
    
    # Test case 1: High confidence (should pass)
    probabilities_high = np.array([0.68, 0.32])  # 68% UP, 32% DOWN
    maxp_high = float(np.max(probabilities_high))
    binary_confidence_threshold = 0.55
    
    print(f"\n2. Test Case 1: High Confidence")
    print(f"   Probabilities: UP={probabilities_high[0]*100:.1f}%, DOWN={probabilities_high[1]*100:.1f}%")
    print(f"   Max probability: {maxp_high*100:.1f}%")
    print(f"   Threshold: {binary_confidence_threshold*100}%")
    
    if is_binary_mode and maxp_high < binary_confidence_threshold:
        print(f"   Result: HOLD (low confidence)")
    else:
        print(f"   Result: Prediction allowed (confidence: {maxp_high*100:.1f}% >= {binary_confidence_threshold*100}%)")
    
    # Test case 2: Low confidence (should be filtered)
    probabilities_low = np.array([0.52, 0.48])  # 52% UP, 48% DOWN
    maxp_low = float(np.max(probabilities_low))
    
    print(f"\n3. Test Case 2: Low Confidence")
    print(f"   Probabilities: UP={probabilities_low[0]*100:.1f}%, DOWN={probabilities_low[1]*100:.1f}%")
    print(f"   Max probability: {maxp_low*100:.1f}%")
    print(f"   Threshold: {binary_confidence_threshold*100}%")
    
    if is_binary_mode and maxp_low < binary_confidence_threshold:
        print(f"   Result: HOLD (low confidence - filtered)")
        print(f"   [OK] This prevents biased predictions when model is uncertain")
    else:
        print(f"   Result: Prediction allowed")
    
    # Test case 3: Very low confidence
    probabilities_very_low = np.array([0.51, 0.49])  # 51% UP, 49% DOWN
    maxp_very_low = float(np.max(probabilities_very_low))
    
    print(f"\n4. Test Case 3: Very Low Confidence")
    print(f"   Probabilities: UP={probabilities_very_low[0]*100:.1f}%, DOWN={probabilities_very_low[1]*100:.1f}%")
    print(f"   Max probability: {maxp_very_low*100:.1f}%")
    print(f"   Threshold: {binary_confidence_threshold*100}%")
    
    if is_binary_mode and maxp_very_low < binary_confidence_threshold:
        print(f"   Result: HOLD (very low confidence - filtered)")
        print(f"   [OK] This is exactly what we want - prevents 51/49 split predictions")
    else:
        print(f"   Result: Prediction allowed")
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("[OK] Confidence threshold of 55% will filter:")
    print("  - Predictions with <55% confidence -> HOLD")
    print("  - This reduces false SELL predictions")
    print("  - Model must be at least 55% confident to make a prediction")
    print("\nExpected impact:")
    print("  - More HOLD predictions (when model is uncertain)")
    print("  - Fewer false SELL predictions")
    print("  - Better overall accuracy")

if __name__ == "__main__":
    test_confidence_threshold_logic()
