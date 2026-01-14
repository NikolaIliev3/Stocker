"""
Calibration Manager for Stocker App
Implements 'Direct Calibration' to dynamically adjust target and stop-loss prices
based on verified trade outcomes.
"""
import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class CalibrationManager:
    """
    Manages dynamic calibration of trading targets.
    
    Tracks 'how far price actually moves' vs 'predicted targets' to 
    create multipliers that adjust future targets to market reality.
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.calibration_file = self.data_dir / "calibration_data.json"
        self.data = self._load_data()
        
        # Defaults if no data exists
        if not self.data:
            self.data = {
                'trading': {
                    'wins': [],  # List of {gain_pct, date}
                    'losses': [], # List of {loss_pct, date}
                    'target_multiplier': 1.0,
                    'stop_loss_multiplier': 1.0,
                    'samples_count': 0
                },
                'mixed': {
                    'wins': [],
                    'losses': [],
                    'target_multiplier': 1.0,
                    'stop_loss_multiplier': 1.0,
                    'samples_count': 0
                },
                'investing': {
                    'wins': [],
                    'losses': [],
                    'target_multiplier': 1.0,
                    'stop_loss_multiplier': 1.0,
                    'samples_count': 0
                }
            }
            self.save_data()
            
    def _load_data(self) -> Dict:
        """Load calibration data from disk"""
        if self.calibration_file.exists():
            try:
                with open(self.calibration_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading calibration data: {e}")
        return {}
        
    def save_data(self):
        """Save calibration data to disk"""
        try:
            with open(self.calibration_file, 'w') as f:
                json.dump(self.data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving calibration data: {e}")

    def update_from_verified_prediction(self, prediction: Dict, actual_outcome: Dict):
        """
        Update stats from a verified prediction.
        
        Args:
            prediction: The prediction dictionary
            actual_outcome: Dict with keys 'was_correct', 'actual_price_change', 'target_hit', 'stop_loss_hit'
        """
        try:
            strategy = prediction.get('strategy', 'trading')
            if strategy not in self.data:
                strategy = 'trading' # Fallback
                
            stats = self.data[strategy]
            
            # We only care about completed trades (Target Hit or Stop Loss Hit)
            # or expired trades where we can measure max excursion (if data available)
            # For now, we use the final realized gain/loss
            
            actual_change = actual_outcome.get('actual_price_change', 0.0)
            was_correct = actual_outcome.get('was_correct', False)
            
            # Record the result
            entry = {
                'pct': abs(actual_change), # Use absolute value for magnitude
                'date': datetime.now().isoformat()
            }
            
            if was_correct:
                # Limit history to last 50 wins for responsiveness
                stats['wins'].append(entry)
                if len(stats['wins']) > 50:
                    stats['wins'].pop(0)
            else:
                # Limit history to last 50 losses
                stats['losses'].append(entry)
                if len(stats['losses']) > 50:
                    stats['losses'].pop(0)
            
            stats['samples_count'] += 1
            
            # Recalculate multipliers
            self._recalculate_multipliers(strategy)
            self.save_data()
            
            logger.info(f"Updated calibration for {strategy}: T_Mult={stats['target_multiplier']:.2f}, SL_Mult={stats['stop_loss_multiplier']:.2f}")
            
        except Exception as e:
            logger.error(f"Error updating calibration: {e}")

    def _recalculate_multipliers(self, strategy: str):
        """Calculate dynamic multipliers based on recent history"""
        stats = self.data[strategy]
        wins = [x['pct'] for x in stats['wins']]
        losses = [x['pct'] for x in stats['losses']]
        
        # Need minimum samples to start calibrating
        if len(wins) < 5:
            return # Keep defaults
            
        # Target Multiplier Logic:
        # If our wins average 8% but we target 5%, we can be more aggressive (increase predicted targets)
        # If our wins average 3% but we target 5%, we should be conservative (decrease predicted targets)
        # We assume the 'Standard' target for Trading is ~3%, Mixed ~6%, Investing ~12%
        
        if strategy == 'trading':
            base_target = 3.0
        elif strategy == 'mixed':
            base_target = 6.0
        else:
            base_target = 12.0
            
        avg_win = np.mean(wins) if wins else base_target
        
        # Damping factor: Don't swing wildly. Move 50% towards the reality.
        # multiplier = ratio of actual / base
        raw_target_mult = avg_win / base_target
        
        # Clamp multiplier to avoid extreme values (e.g., 0.5x to 2.0x)
        raw_target_mult = max(0.5, min(2.5, raw_target_mult))
        
        # Weighted update: 70% old, 30% new (smooth changes)
        old_mult = stats.get('target_multiplier', 1.0)
        stats['target_multiplier'] = (old_mult * 0.7) + (raw_target_mult * 0.3)

        # Stop Loss Multiplier Logic:
        # If losses average 4% but we set stop at 2%, we might be getting stopped out by noise.
        # We need to widen stops (increase multiplier).
        # OR if we lose 10% but stop was 2%, maybe we didn't honor the stop (slippage/gaps).
        # This is trickier. Generally, we want stops to be tight but realistic.
        # For now, let's track volatility. If losses are large, widen stops slightly to avoid noise,
        # but if wins are also large, it justifies the risk.
        
        if losses:
            avg_loss = np.mean(losses)
            # Default risk: trading=1.5%, mixed=3%, investing=6% (approx half of target)
            base_stop = base_target / 2.0 
            
            raw_stop_mult = avg_loss / base_stop
            # Clamp stop multiplier: 0.8x (tighter) to 1.5x (wider)
            # We rarely want to widen stops too much (risk control)
            raw_stop_mult = max(0.8, min(1.5, raw_stop_mult))
            
            old_stop_mult = stats.get('stop_loss_multiplier', 1.0)
            stats['stop_loss_multiplier'] = (old_stop_mult * 0.7) + (raw_stop_mult * 0.3)

    def get_multipliers(self, strategy: str) -> Dict[str, float]:
        """Get the current multipliers for a strategy"""
        defaults = {'target_mult': 1.0, 'stop_loss_mult': 1.0}
        
        if strategy not in self.data:
            return defaults
            
        return {
            'target_mult': self.data[strategy].get('target_multiplier', 1.0),
            'stop_loss_mult': self.data[strategy].get('stop_loss_multiplier', 1.0)
        }
