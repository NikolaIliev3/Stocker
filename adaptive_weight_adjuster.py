"""
Adaptive Weight Adjuster
Learns optimal weights for rule-based predictions from verified outcomes
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class AdaptiveWeightAdjuster:
    """Adjusts rule-based weights based on verified prediction results"""
    
    def __init__(self, data_dir: Path, strategy: str):
        self.data_dir = data_dir
        self.strategy = strategy
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.weights_file = self.data_dir / "trained_weights.json"
        self.weights = self._load_weights()
        self.training_samples_count = 0
    
    def _get_default_weights(self) -> Dict:
        """Get default weights for strategy"""
        defaults = {
            'trading': {
                'rsi_oversold_weight': 2.0,
                'rsi_overbought_weight': -2.0,
                'macd_bullish_weight': 1.0,
                'macd_bearish_weight': -1.0,
                'trend_uptrend_weight': 1.0,
                'trend_downtrend_weight': -1.0,
                'volume_high_weight': 1.0,
                'golden_cross_weight': 2.0,
                'death_cross_weight': -2.0,
                'momentum_score_multiplier': 1.0,
                'support_near_weight': 1.0,
                'resistance_near_weight': -1.0
            },
            'investing': {
                'health_score_weight': 1.0,
                'valuation_score_weight': 1.0,
                'growth_score_weight': 1.0,
                'risk_penalty': 1.0,
                'roe_weight': 1.0,
                'debt_to_equity_weight': -1.0
            },
            'mixed': {
                'technical_weight': 0.6,
                'fundamental_weight': 0.4
            }
        }
        return defaults.get(self.strategy, defaults['trading'])
    
    def _load_weights(self) -> Dict:
        """Load trained weights from disk"""
        if self.weights_file.exists():
            try:
                with open(self.weights_file, 'r') as f:
                    all_weights = json.load(f)
                    strategy_weights = all_weights.get(self.strategy, {})
                    
                    # Merge with defaults to ensure all keys exist
                    defaults = self._get_default_weights()
                    merged = {**defaults, **strategy_weights}
                    
                    # Remove metadata keys
                    merged.pop('last_updated', None)
                    merged.pop('training_samples', None)
                    merged.pop('accuracy_improvement', None)
                    
                    return merged
            except Exception as e:
                logger.error(f"Error loading weights: {e}")
        
        return self._get_default_weights()
    
    def save_weights(self):
        """Save weights to disk"""
        try:
            # Load existing weights for all strategies
            all_weights = {}
            if self.weights_file.exists():
                try:
                    with open(self.weights_file, 'r') as f:
                        all_weights = json.load(f)
                except:
                    pass
            
            # Update weights for current strategy
            all_weights[self.strategy] = {
                **self.weights,
                'last_updated': datetime.now().isoformat(),
                'training_samples': self.training_samples_count
            }
            
            # Save back to disk
            with open(self.weights_file, 'w') as f:
                json.dump(all_weights, f, indent=2)
            
            logger.info(f"Saved trained weights for {self.strategy} strategy")
        except Exception as e:
            logger.error(f"Error saving weights: {e}")
    
    def update_weights_from_results(self, verified_predictions: List[Dict]):
        """Update weights based on verified prediction results"""
        if not verified_predictions:
            return
        
        logger.info(f"Updating weights from {len(verified_predictions)} verified predictions")
        
        if self.strategy == "trading":
            self._update_trading_weights(verified_predictions)
        elif self.strategy == "investing":
            self._update_investing_weights(verified_predictions)
        elif self.strategy == "mixed":
            self._update_mixed_weights(verified_predictions)
        
        self.training_samples_count += len(verified_predictions)
        self.save_weights()
    
    def _update_trading_weights(self, verified: List[Dict]):
        """Update trading strategy weights"""
        correct = [v for v in verified if v.get('was_correct') is True]
        incorrect = [v for v in verified if v.get('was_correct') is False]
        
        if len(correct) == 0 or len(incorrect) == 0:
            return
        
        # Analyze RSI performance
        correct_rsi_oversold = sum(1 for v in correct 
                                  if v.get('indicators', {}).get('rsi', 50) < 30)
        incorrect_rsi_oversold = sum(1 for v in incorrect 
                                    if v.get('indicators', {}).get('rsi', 50) < 30)
        
        total_rsi_oversold = correct_rsi_oversold + incorrect_rsi_oversold
        if total_rsi_oversold > 0:
            rsi_accuracy = correct_rsi_oversold / total_rsi_oversold
            if rsi_accuracy > 0.6:
                self.weights['rsi_oversold_weight'] *= 1.05
            elif rsi_accuracy < 0.4:
                self.weights['rsi_oversold_weight'] *= 0.95
        
        # Analyze MACD performance
        correct_macd_bullish = sum(1 for v in correct 
                                  if v.get('indicators', {}).get('macd_diff', 0) > 0)
        incorrect_macd_bullish = sum(1 for v in incorrect 
                                    if v.get('indicators', {}).get('macd_diff', 0) > 0)
        
        total_macd = correct_macd_bullish + incorrect_macd_bullish
        if total_macd > 0:
            macd_accuracy = correct_macd_bullish / total_macd
            if macd_accuracy > 0.6:
                self.weights['macd_bullish_weight'] *= 1.05
            elif macd_accuracy < 0.4:
                self.weights['macd_bullish_weight'] *= 0.95
        
        # Analyze momentum score
        if correct and incorrect:
            correct_momentum_avg = np.mean([
                v.get('momentum_analysis', {}).get('momentum_score', 0) 
                for v in correct
            ])
            incorrect_momentum_avg = np.mean([
                v.get('momentum_analysis', {}).get('momentum_score', 0) 
                for v in incorrect
            ])
            
            if abs(correct_momentum_avg) > abs(incorrect_momentum_avg):
                self.weights['momentum_score_multiplier'] *= 1.02
            else:
                self.weights['momentum_score_multiplier'] *= 0.98
        
        # Clamp weights to reasonable ranges
        for key in self.weights:
            if 'weight' in key or 'multiplier' in key:
                self.weights[key] = max(0.1, min(5.0, self.weights[key]))
    
    def _update_investing_weights(self, verified: List[Dict]):
        """Update investing strategy weights"""
        # Similar approach for fundamental analysis
        # This is simplified - would analyze financial health, valuation, etc.
        correct = [v for v in verified if v.get('was_correct') is True]
        
        if len(correct) > len(verified) * 0.6:
            # If doing well, slightly increase weights
            for key in self.weights:
                if 'weight' in key:
                    self.weights[key] *= 1.02
        elif len(correct) < len(verified) * 0.4:
            # If doing poorly, slightly decrease weights
            for key in self.weights:
                if 'weight' in key:
                    self.weights[key] *= 0.98
        
        # Clamp weights
        for key in self.weights:
            if 'weight' in key:
                self.weights[key] = max(0.1, min(5.0, self.weights[key]))
    
    def _update_mixed_weights(self, verified: List[Dict]):
        """Update mixed strategy weights"""
        correct = [v for v in verified if v.get('was_correct') is True]
        
        # Analyze if technical or fundamental was more predictive
        # This is simplified - would need more detailed analysis
        if len(correct) > len(verified) * 0.6:
            # If doing well, keep current balance
            pass
        else:
            # Try adjusting the balance
            if self.weights['technical_weight'] > 0.5:
                self.weights['technical_weight'] -= 0.01
                self.weights['fundamental_weight'] += 0.01
            else:
                self.weights['technical_weight'] += 0.01
                self.weights['fundamental_weight'] -= 0.01
        
        # Clamp weights
        self.weights['technical_weight'] = max(0.2, min(0.8, self.weights['technical_weight']))
        self.weights['fundamental_weight'] = 1.0 - self.weights['technical_weight']
    
    def get_weights(self) -> Dict:
        """Get current weights"""
        return self.weights.copy()
    
    def get_weight(self, key: str, default: float = 1.0) -> float:
        """Get a specific weight"""
        return self.weights.get(key, default)

