"""
Risk Scorer - Calculate risk score (1-5 stars) for stock predictions
"""
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class RiskScorer:
    """Calculate risk score based on multiple factors"""
    
    def __init__(self):
        pass
    
    def calculate_risk(self, analysis: Dict, stock_data: Dict) -> Tuple[int, str, list]:
        """Calculate risk score (1-5 stars) and explanation
        
        Args:
            analysis: Trading analysis dictionary
            stock_data: Stock data dictionary
            
        Returns:
            Tuple of (risk_score, risk_level_text, risk_factors)
            - risk_score: 1-5 (1=lowest risk, 5=highest risk)
            - risk_level_text: Human-readable risk level
            - risk_factors: List of risk factor strings
        """
        try:
            risk_points = 0
            risk_factors = []
            
            # Factor 1: Volatility (ATR)
            atr_percent = analysis.get('atr_percent', 0)
            if atr_percent > 8:
                risk_points += 2
                risk_factors.append(f"Very high volatility (ATR: {atr_percent:.1f}%)")
            elif atr_percent > 5:
                risk_points += 1
                risk_factors.append(f"High volatility (ATR: {atr_percent:.1f}%)")
            elif atr_percent < 2:
                risk_factors.append(f"Low volatility (ATR: {atr_percent:.1f}%)")
            
            # Factor 2: RSI extremes
            indicators = analysis.get('indicators', {})
            rsi = indicators.get('rsi', 50)
            if rsi > 75:
                risk_points += 1
                risk_factors.append(f"Overbought (RSI: {rsi:.1f})")
            elif rsi < 25:
                risk_points += 1
                risk_factors.append(f"Oversold (RSI: {rsi:.1f})")
            
            # Factor 3: Distance to support/resistance
            support_resistance = analysis.get('support_resistance', {})
            distance_to_support = support_resistance.get('current_distance_to_support', 50)
            distance_to_resistance = support_resistance.get('current_distance_to_resistance', 50)
            
            # Close to resistance (risky for longs)
            if distance_to_resistance < 2:
                risk_points += 1
                risk_factors.append(f"Near resistance ({distance_to_resistance:.1f}% away)")
            
            # Close to support (risky for shorts)
            if distance_to_support < 2:
                risk_points += 1
                risk_factors.append(f"Near support ({distance_to_support:.1f}% away)")
            
            # Factor 4: Volume
            volume_analysis = analysis.get('volume', {})
            volume_ratio = volume_analysis.get('volume_ratio', 1.0)
            if volume_ratio < 0.5:
                risk_points += 1
                risk_factors.append(f"Low volume ({volume_ratio:.1f}x average)")
            
            # Factor 5: MFI extremes
            mfi = indicators.get('mfi', 50)
            if mfi > 85:
                risk_points += 1
                risk_factors.append(f"Extreme overbought (MFI: {mfi:.1f})")
            elif mfi < 15:
                risk_points += 1
                risk_factors.append(f"Extreme oversold (MFI: {mfi:.1f})")
            
            # Factor 6: Trend strength
            price_action = analysis.get('price_action', {})
            trend = price_action.get('trend', 'sideways')
            if trend == 'sideways':
                risk_points += 1
                risk_factors.append("Sideways trend (unclear direction)")
            
            # Factor 7: Market regime
            market_regime = analysis.get('market_regime', {})
            regime = market_regime.get('regime', 'unknown')
            if regime == 'high_volatility':
                risk_points += 1
                risk_factors.append("High volatility market regime")
            
            # Convert points to 1-5 scale
            # 0-1 points = 1 star (low risk)
            # 2-3 points = 2 stars (low-medium risk)
            # 4-5 points = 3 stars (medium risk)
            # 6-7 points = 4 stars (medium-high risk)
            # 8+ points = 5 stars (high risk)
            
            if risk_points <= 1:
                risk_score = 1
                risk_level = "Low Risk"
            elif risk_points <= 3:
                risk_score = 2
                risk_level = "Low-Medium Risk"
            elif risk_points <= 5:
                risk_score = 3
                risk_level = "Medium Risk"
            elif risk_points <= 7:
                risk_score = 4
                risk_level = "Medium-High Risk"
            else:
                risk_score = 5
                risk_level = "High Risk"
            
            # Add positive factors if low risk
            if risk_score <= 2:
                if atr_percent < 3:
                    risk_factors.append("✓ Stable price movement")
                if 40 <= rsi <= 60:
                    risk_factors.append("✓ Neutral RSI")
                if volume_ratio > 1.2:
                    risk_factors.append("✓ Above-average volume")
            
            logger.debug(f"Risk score calculated: {risk_score}/5 ({risk_level}), {len(risk_factors)} factors")
            
            return risk_score, risk_level, risk_factors
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 3, "Medium Risk", ["Unable to calculate detailed risk factors"]
    
    def get_risk_stars(self, risk_score: int) -> str:
        """Get star representation of risk score
        
        Args:
            risk_score: Risk score (1-5)
            
        Returns:
            String with star symbols
        """
        filled = '⭐' * risk_score
        empty = '☆' * (5 - risk_score)
        return filled + empty


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    scorer = RiskScorer()
    
    # Test with high-risk scenario
    high_risk_analysis = {
        'atr_percent': 9.5,
        'indicators': {'rsi': 82, 'mfi': 88},
        'support_resistance': {
            'current_distance_to_support': 15,
            'current_distance_to_resistance': 1.2
        },
        'volume': {'volume_ratio': 0.4},
        'price_action': {'trend': 'sideways'},
        'market_regime': {'regime': 'high_volatility'}
    }
    
    score, level, factors = scorer.calculate_risk(high_risk_analysis, {})
    print(f"\nHigh Risk Test:")
    print(f"Score: {score}/5 ({level})")
    print(f"Stars: {scorer.get_risk_stars(score)}")
    print(f"Factors:")
    for factor in factors:
        print(f"  - {factor}")
    
    # Test with low-risk scenario
    low_risk_analysis = {
        'atr_percent': 2.1,
        'indicators': {'rsi': 52, 'mfi': 48},
        'support_resistance': {
            'current_distance_to_support': 8,
            'current_distance_to_resistance': 10
        },
        'volume': {'volume_ratio': 1.3},
        'price_action': {'trend': 'uptrend'},
        'market_regime': {'regime': 'normal'}
    }
    
    score, level, factors = scorer.calculate_risk(low_risk_analysis, {})
    print(f"\nLow Risk Test:")
    print(f"Score: {score}/5 ({level})")
    print(f"Stars: {scorer.get_risk_stars(score)}")
    print(f"Factors:")
    for factor in factors:
        print(f"  - {factor}")
