"""
Bayesian Forensic Analysis
Analyzes failed predictions to identify root causes using conditional probability.

For each feature, calculates:
  P(Loss | Feature_High) vs P(Loss | Feature_Low)

If a feature significantly increases the probability of loss, it's flagged as a "Suspect".
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import logging
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import APP_DATA_DIR

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger('ForensicAnalysis')

def load_backtest_results():
    """Load backtest results from storage"""
    results_file = Path(APP_DATA_DIR) / 'backtest_results.json'
    if not results_file.exists():
        logger.error(f"No backtest results found at {results_file}")
        return None
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    results = data.get('test_details', [])
    if not results:
        logger.error("No test details found in backtest results")
        return None
    
    return pd.DataFrame(results)

def bayesian_feature_analysis(df: pd.DataFrame, n_samples: int = 500):
    """
    Perform Bayesian analysis to find features correlated with failures.
    
    For each feature condition, calculates:
    - P(Loss | Condition) = Number of losses with condition / Total with condition
    - Lift = P(Loss | Condition) / P(Loss) - how much the condition increases risk
    """
    # Filter to most recent samples
    if len(df) > n_samples:
        df = df.tail(n_samples)
    
    logger.info(f"\n{'='*60}")
    logger.info("🔍 BAYESIAN FORENSIC ANALYSIS")
    logger.info(f"{'='*60}")
    logger.info(f"Analyzing {len(df)} predictions...\n")
    
    # Calculate baseline loss rate
    total = len(df)
    losses = len(df[df['was_correct'] == False])
    baseline_loss_rate = losses / total if total > 0 else 0
    
    logger.info(f"📊 BASELINE STATISTICS:")
    logger.info(f"   Total Predictions: {total}")
    logger.info(f"   Losses: {losses}")
    logger.info(f"   Baseline Loss Rate: {baseline_loss_rate:.1%}")
    logger.info("")
    
    # Define conditions to analyze
    suspects = []
    
    # 1. Analyze by Predicted Action
    logger.info("📌 ANALYSIS BY ACTION:")
    for action in ['BUY', 'SELL', 'HOLD']:
        action_df = df[df['predicted_action'] == action]
        if len(action_df) == 0:
            continue
        
        action_losses = len(action_df[action_df['was_correct'] == False])
        action_loss_rate = action_losses / len(action_df)
        lift = action_loss_rate / baseline_loss_rate if baseline_loss_rate > 0 else 0
        
        status = "🔴 HIGH RISK" if lift > 1.2 else ("🟡 ELEVATED" if lift > 1.0 else "🟢 SAFE")
        logger.info(f"   {action}: Loss Rate = {action_loss_rate:.1%} (Lift: {lift:.2f}x) {status}")
        
        if lift > 1.2 and len(action_df) >= 10:
            suspects.append({
                'condition': f"Action = {action}",
                'loss_rate': action_loss_rate,
                'lift': lift,
                'sample_size': len(action_df)
            })
    
    logger.info("")
    
    # 2. Analyze by Confidence Buckets
    logger.info("📌 ANALYSIS BY CONFIDENCE:")
    confidence_buckets = [
        ('Low (50-60%)', 50, 60),
        ('Medium (60-75%)', 60, 75),
        ('High (75-90%)', 75, 90),
        ('Very High (90%+)', 90, 101)
    ]
    
    for name, low, high in confidence_buckets:
        bucket_df = df[(df['confidence'] >= low) & (df['confidence'] < high)]
        if len(bucket_df) == 0:
            continue
        
        bucket_losses = len(bucket_df[bucket_df['was_correct'] == False])
        bucket_loss_rate = bucket_losses / len(bucket_df)
        lift = bucket_loss_rate / baseline_loss_rate if baseline_loss_rate > 0 else 0
        
        status = "🔴 INVERTED!" if name.startswith('Very High') and lift > 1.0 else (
            "🔴 HIGH RISK" if lift > 1.2 else ("🟡 ELEVATED" if lift > 1.0 else "🟢 CALIBRATED"))
        logger.info(f"   {name}: Loss Rate = {bucket_loss_rate:.1%} ({len(bucket_df)} trades) {status}")
        
        if lift > 1.2 and len(bucket_df) >= 5:
            suspects.append({
                'condition': f"Confidence {name}",
                'loss_rate': bucket_loss_rate,
                'lift': lift,
                'sample_size': len(bucket_df)
            })
    
    logger.info("")
    
    # 3. Analyze by Price Change Magnitude (for pattern detection)
    logger.info("📌 ANALYSIS BY PRICE MOVEMENT:")
    if 'actual_change_pct' in df.columns:
        movement_buckets = [
            ('Large Drop (<-5%)', -100, -5),
            ('Moderate Drop (-5% to -2%)', -5, -2),
            ('Small Drop (-2% to 0%)', -2, 0),
            ('Small Gain (0% to 2%)', 0, 2),
            ('Moderate Gain (2% to 5%)', 2, 5),
            ('Large Gain (>5%)', 5, 100)
        ]
        
        for name, low, high in movement_buckets:
            bucket_df = df[(df['actual_change_pct'] >= low) & (df['actual_change_pct'] < high)]
            if len(bucket_df) == 0:
                continue
            
            bucket_losses = len(bucket_df[bucket_df['was_correct'] == False])
            bucket_loss_rate = bucket_losses / len(bucket_df)
            lift = bucket_loss_rate / baseline_loss_rate if baseline_loss_rate > 0 else 0
            
            status = "🔴 BLIND SPOT" if lift > 1.5 else ("🟡 WEAK" if lift > 1.0 else "🟢 STRONG")
            logger.info(f"   {name}: Loss Rate = {bucket_loss_rate:.1%} ({len(bucket_df)} cases) {status}")
    
    logger.info("")
    
    # 4. Summary of Suspects
    if suspects:
        logger.info("🚨 TOP SUSPECTS (Conditions that increase failure rate):")
        suspects.sort(key=lambda x: x['lift'], reverse=True)
        for i, s in enumerate(suspects[:5], 1):
            logger.info(f"   {i}. {s['condition']}")
            logger.info(f"      Loss Rate: {s['loss_rate']:.1%} (Lift: {s['lift']:.2f}x, n={s['sample_size']})")
    else:
        logger.info("✅ No significant risk factors detected.")
    
    logger.info(f"\n{'='*60}")
    
    return suspects

def main():
    df = load_backtest_results()
    if df is None:
        return
    
    suspects = bayesian_feature_analysis(df)
    
    if suspects:
        logger.info("\n💡 RECOMMENDATIONS:")
        for s in suspects[:3]:
            logger.info(f"   • Consider adding a safety rule for: {s['condition']}")

if __name__ == "__main__":
    main()
