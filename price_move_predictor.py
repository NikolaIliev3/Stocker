"""
Price Move Predictor
Predicts the magnitude of price movements (rise/fall) for BUY/SELL signals.
Self-calibrating system that learns from actual market moves to improve future predictions.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class PriceMovePredictor:
    """
    Predicts how much the price will rise or fall when a signal is generated.
    Learns from verified outcomes to adjust its predictions.
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.learning_file = data_dir / "price_move_learning.json"
        self.learning_data = self._load_learning_data()
        
    def _load_learning_data(self) -> Dict:
        """Load learned parameters from disk"""
        default_data = {
            'multipliers': {
                'bullish': 1.0,  # Multiplier for predicted rise
                'bearish': 1.0,  # Multiplier for predicted fall
                'volatility_factor': 1.0  # Learning how much volatility impacts the move
            },
            'history': [],  # Recent accuracy history
            'stats': {
                'total_predictions': 0,
                'avg_error_pct': 0.0,
                'last_updated': None
            }
        }
        
        if self.learning_file.exists():
            try:
                with open(self.learning_file, 'r') as f:
                    data = json.load(f)
                    # Merge with defaults to ensure all keys exist
                    for key, value in default_data.items():
                        if key not in data:
                            data[key] = value
                    return data
            except Exception as e:
                logger.error(f"Error loading price move learning data: {e}")
        
        return default_data

    def _save_learning_data(self):
        """Save learned parameters to disk"""
        try:
            with open(self.learning_file, 'w') as f:
                json.dump(self.learning_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving price move learning data: {e}")

    def predict_move(self, signal_type: str, current_price: float, 
                    indicators: Dict, volatility_metrics: Dict) -> Dict:
        """
        Predict the magnitude of the move.
        
        Args:
            signal_type: 'BUY' or 'SELL'
            current_price: Current stock price
            indicators: Dictionary of technical indicators (RSI, MACD, etc.)
            volatility_metrics: Dictionary containing ATR, Beta, etc.
            
        Returns:
            Dict containing prediction details
        """
        try:
            atr = volatility_metrics.get('atr', 0)
            atr_percent = volatility_metrics.get('atr_percent', 0)
            
            if atr == 0 or current_price == 0:
                # Fallback if no volatility data
                predicted_pct = 5.0  # Default 5%
            else:
                # Base prediction on ATR (volatility)
                # Typically a move might be 2-3x ATR
                base_move_pct = atr_percent * 2.5
                
                # Apply learning multipliers
                if signal_type == 'BUY':
                    predicted_pct = base_move_pct * self.learning_data['multipliers']['bullish']
                else:
                    predicted_pct = base_move_pct * self.learning_data['multipliers']['bearish']
                
                # Adjust based on volatility learning
                predicted_pct *= self.learning_data['multipliers']['volatility_factor']
            
            # Cap realistic moves (e.g., extremely rare to move > 50% on standard signals)
            predicted_pct = min(predicted_pct, 50.0)
            predicted_pct = max(predicted_pct, 1.0)  # Minimum 1% move
            
            # Calculate target price
            if signal_type == 'BUY':
                predicted_change = predicted_pct
                target_price = current_price * (1 + predicted_pct / 100)
                direction = "RISE"
            else:
                predicted_change = -predicted_pct
                target_price = current_price * (1 - predicted_pct / 100)
                direction = "FALL"
                
            # Calculate confidence based on indicator confluence
            confidence = self._calculate_confidence(signal_type, indicators)
            
            return {
                "direction": direction,
                "predicted_pct": predicted_pct,
                "target_price": target_price,
                "confidence": confidence,
                "based_on": f"Volatility (ATR: {atr_percent:.2f}%) & Learned Multipliers",
                "learning_adjustment": f"{float(self.learning_data['multipliers']['bullish' if signal_type == 'BUY' else 'bearish']):.2f}x"
            }
            
        except Exception as e:
            logger.error(f"Error predicting price move: {e}")
            return {
                "direction": "UNKNOWN",
                "predicted_pct": 0,
                "target_price": current_price,
                "confidence": 0,
                "error": str(e)
            }

    def _calculate_confidence(self, signal_type: str, indicators: Dict) -> float:
        """Calculate confidence score for the magnitude prediction"""
        score = 50  # Base confidence
        
        # Adjust based on indicators
        rsi = indicators.get('rsi', 50)
        macd_diff = indicators.get('macd_diff', 0)
        adx = indicators.get('adx', 0)
        
        if signal_type == 'BUY':
            if rsi < 30: score += 15 # Oversold = stronger rebound likely
            elif rsi < 40: score += 10
            if macd_diff > 0: score += 10
            if adx > 25: score += 10 # Strong trend
            
        elif signal_type == 'SELL':
            if rsi > 70: score += 15 # Overbought = stronger correction likely
            elif rsi > 60: score += 10
            if macd_diff < 0: score += 10
            if adx > 25: score += 10
            
        return min(max(score, 10), 95)

    def learn_from_outcome(self, predicted_pct: float, actual_pct: float, signal_type: str):
        """
        Feed data to Megamind: Learn from the actual outcome to improve future predictions.
        
        Args:
            predicted_pct: The absolute percentage move accepted (always positive)
            actual_pct: The absolute percentage move that actually happened
            signal_type: 'BUY' or 'SELL'
        """
        try:
            # Calculate error
            error_ratio = actual_pct / predicted_pct if predicted_pct > 0 else 1.0
            
            # Log the learning event
            logger.info(f"🧠 Megamind Learning (Price Move): Predicted {predicted_pct:.2f}%, Actual {actual_pct:.2f}% (Ratio: {error_ratio:.2f})")
            
            # Update multipliers (Weighted average: 90% old, 10% new to update gradually)
            learning_rate = 0.1
            
            if signal_type == 'BUY':
                current_mult = self.learning_data['multipliers']['bullish']
                # If actual was higher than predicted, we need to increase multiplier (ratio > 1)
                # If actual was lower, decrease (ratio < 1)
                new_mult = current_mult * (1 - learning_rate) + (current_mult * error_ratio) * learning_rate
                self.learning_data['multipliers']['bullish'] = new_mult
                
            elif signal_type == 'SELL':
                current_mult = self.learning_data['multipliers']['bearish']
                new_mult = current_mult * (1 - learning_rate) + (current_mult * error_ratio) * learning_rate
                self.learning_data['multipliers']['bearish'] = new_mult
            
            # Update stats
            self.learning_data['stats']['total_predictions'] += 1
            
            # Save
            self.learning_data['stats']['last_updated'] = datetime.now().isoformat()
            self._save_learning_data()
            
            return {
                "success": True,
                "old_multiplier": current_mult,
                "new_multiplier": new_mult,
                "error_ratio": error_ratio
            }
            
        except Exception as e:
            logger.error(f"Error in price move learning: {e}")
            return {"success": False, "error": str(e)}
