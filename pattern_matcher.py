"""
Pattern Matcher Module
Stores and queries historical training patterns to provide upside estimates.
"""
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class PatternMatcher:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.patterns_file = data_dir / "historical_patterns.json"
        self.patterns = []
        self._load_patterns()
        
    def _load_patterns(self):
        """Load patterns from disk"""
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, 'r') as f:
                    data = json.load(f)
                    self.patterns = data.get('patterns', [])
                logger.info(f"Loaded {len(self.patterns)} historical patterns")
            except Exception as e:
                logger.error(f"Error loading patterns: {e}")
                self.patterns = []
    
    def save_patterns(self, new_patterns: List[Dict]):
        """Save patterns to disk (Overwrites existing for now, or appends?)"""
        # For now, we overwrite with the latest training set to keep it fresh and consistent
        # In a production DB, we'd append, but here we want the 'current training set'
        try:
            data = {
                'patterns': new_patterns,
                'updated_at': pd.Timestamp.now().isoformat()
            }
            with open(self.patterns_file, 'w') as f:
                json.dump(data, f)
            logger.info(f"Saved {len(new_patterns)} historical patterns to {self.patterns_file}")
            self.patterns = new_patterns
        except Exception as e:
            logger.error(f"Error saving patterns: {e}")

    def find_similar_patterns(self, feature_dict: Dict, confidence: float, limit: int = 200) -> Dict:
        """
        Find similar historical patterns based on:
        - Technical Setup (RSI)
        - Volatility (ATR)
        - Confidence Band
        """
        if not self.patterns:
            return {'error': 'No historical patterns available'}
            
        # Extract query features
        q_rsi = feature_dict.get('rsi', 50)
        q_atr = feature_dict.get('atr_percent', 0)
        q_conf = confidence
        
        matches = []
        
        # 1. Filter by Confidence Band (± 5%)
        # If confidence is 82, looked at 77-87
        conf_min = q_conf - 5.0
        conf_max = q_conf + 5.0
        
        # 2. Filter by RSI (± 10) - Broad similarity
        rsi_min = q_rsi - 10
        rsi_max = q_rsi + 10
        
        # 3. Filter by ATR (± 20% relative, or ± 0.5% absolute?)
        # Let's use absolute percentage difference e.g. ± 1.0% diff
        # If ATR is 2.5%, look at 1.5% - 3.5%
        atr_min = q_atr - 1.0
        atr_max = q_atr + 1.0
        
        # Search (Linear scan is fine for <100k samples)
        for p in self.patterns:
            # Check Confidence (if available in pattern)
            p_conf = p.get('confidence', 0)
            if not (conf_min <= p_conf <= conf_max):
                continue
                
            # Check RSI
            p_rsi = p.get('rsi', 50)
            if not (rsi_min <= p_rsi <= rsi_max):
                continue
                
            # Check ATR
            p_atr = p.get('atr_percent', 0)
            if not (atr_min <= p_atr <= atr_max):
                continue
                
            matches.append(p)
            
        if not matches:
             return {'sample_size': 0, 'note': 'No similar patterns found'}

        # Calculate Stats
        sample_size = len(matches)
        
        if sample_size < 20:
            return {
                'sample_size': sample_size,
                'avg_max_profit': 0,
                'exceeded_target': 0,
                'insufficient_data': True,
                'note': f"Insufficient historical data ({sample_size} matches)"
            }
            
        # Calculate Average Max Profit
        max_profits = [p.get('max_profit_pct', 0) for p in matches]
        avg_max_profit = float(np.mean(max_profits))
        
        # Calculate Win Rate (>3% Terminal? Or >3% Max?)
        # User asked: "exceeded_target": % of times it went past 3%
        exceeded_count = sum(1 for profit in max_profits if profit >= 3.0)
        exceeded_pct = (exceeded_count / sample_size) * 100
        
        return {
            'sample_size': sample_size,
            'avg_max_profit': avg_max_profit,
            'exceeded_target': exceeded_pct,
            'insufficient_data': False,
            'note': f"Historical data shows this pattern type averaged {avg_max_profit:.1f}% peak gains. {exceeded_pct:.0f}% exceeded +3%."
        }
