"""
Smart Model Selector specifically designed to prefer robust models over overfit ones.
Addresses user concern about "suspicious high accuracy" in small-sample models.
"""
import logging
import json
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

def select_best_strategy(data_dir: Path) -> str:
    """
    Select the best strategy based on model robustness, not just raw accuracy.
    
    Returns:
        str: 'trading', 'mixed', or 'investing'
    """
    strategies = ['trading', 'mixed']
    metadata = {}
    
    # Load metadata
    for strat in strategies:
        meta_file = data_dir / f"ml_metadata_{strat}.json"
        if meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    metadata[strat] = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load metadata for {strat}: {e}")
    
    # If no models found, default to trading (start fresh)
    if not metadata:
        logger.info("No trained models found. Defaulting to 'trading'.")
        return 'trading'
        
    # If only one exists, use it
    if len(metadata) == 1:
        winner = list(metadata.keys())[0]
        logger.info(f"Only {winner} model exists. Selecting it.")
        return winner
        
    # Robustness Scoring System
    logger.info("Comparing models for robustness...")
    scores = {}
    
    for strat, data in metadata.items():
        # 1. Get Base Metrics
        test_acc = data.get('test_accuracy', 0.5)
        train_acc = data.get('train_accuracy', 0.5)
        samples = data.get('samples', 0)
        
        # 2. Overfitting Penalty
        # If train accuracy is much higher than test, penalty increases
        # E.g. Train 90%, Test 70% -> Gap 0.2 -> Penalty high
        generalization_gap = max(0, train_acc - test_acc)
        # Apply strict penalty if gap > 10%
        penalty = 0
        if generalization_gap > 0.10:
            penalty = generalization_gap * 2.0  # Double the gap as penalty
        elif generalization_gap > 0.05:
            penalty = generalization_gap * 1.0
            
        # 3. Sample Size Bonus
        # Logarithmic scale: 100 samples -> 4.6, 1000 samples -> 6.9, 10000 -> 9.2
        # We want to favor models with at least 2000+ samples
        safe_samples = max(10, samples)
        sample_score = np.log10(safe_samples) / 4.0  # Normalized roughly around 1.0 for ~10k samples
        
        # 4. Calculate Final Robustness Score
        # Score = (Test Accuracy - Penalty) * Sample Weight
        adjusted_accuracy = test_acc - penalty
        robustness_score = adjusted_accuracy * sample_score
        
        scores[strat] = {
            'robustness_score': robustness_score,
            'raw_test_acc': test_acc,
            'samples': samples,
            'gap': generalization_gap
        }
        
        logger.info(f"Strategy: {strat.upper()}")
        logger.info(f"  - Samples: {samples}")
        logger.info(f"  - Test Acc: {test_acc*100:.1f}% (Train: {train_acc*100:.1f}%)")
        logger.info(f"  - Gap: {generalization_gap*100:.1f}% -> Penalty: {penalty:.3f}")
        logger.info(f"  - Robustness Score: {robustness_score:.4f}")

    # Determine Winner
    # 'trading' (New) vs 'mixed' (Old)
    trading_score = scores.get('trading', {}).get('robustness_score', 0)
    mixed_score = scores.get('mixed', {}).get('robustness_score', 0)
    
    # Logic: Prefer Trading unless Mixed is significantly better
    # Logic: Prefer Trading unless Mixed is significantly better
    # CRITICAL UPDATE: If one model has vastly superior raw accuracy (> 8% gap), 
    # prefer it even if robustness score is lower (as long as it has some samples).
    
    trading_acc = scores.get('trading', {}).get('raw_test_acc', 0)
    mixed_acc = scores.get('mixed', {}).get('raw_test_acc', 0)
    trading_samples = scores.get('trading', {}).get('samples', 0)
    mixed_samples = scores.get('mixed', {}).get('samples', 0)
    
    acc_diff = mixed_acc - trading_acc
    
    # CRITICAL FIX: Trust the Training/Test accuracy even if 'samples' (user feedback) is 0.
    # If the Mixed model tests 8% better than Trading, use it.
    if acc_diff > 0.08:
        # Mixed is MUCH better (>8% accuracy lead)
        winner = 'mixed'
        reason = f"Significantly higher accuracy (+{acc_diff*100:.1f}%) overrides robustness"
    elif trading_score >= mixed_score:
        winner = 'trading'
        reason = "Higher robustness score"
    else:
        # Check if the difference is trivial
        if (mixed_score - trading_score) < 0.05:
            winner = 'trading' # Tie-break to newer model logic
            reason = "Scores similar, preferring robust 'trading' strategy"
        else:
            winner = 'mixed'
            reason = "Better performance score"
            
    logger.info(f"✅ Selected Strategy: {winner.upper()} ({reason})")
    
    if winner == 'trading' and scores.get('mixed', {}).get('raw_test_acc', 0) > scores.get('trading', {}).get('raw_test_acc', 0):
        logger.info("NOTE: 'Mixed' had higher raw accuracy, but 'Trading' was selected due to sample size/robustness.")
        
    return winner
