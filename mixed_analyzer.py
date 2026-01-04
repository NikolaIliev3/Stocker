"""
Mixed Strategy Analyzer for Stocker App
Combines technical and fundamental analysis for medium-term holding (1 week - 1 month)
"""
import logging
from typing import Dict, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MixedAnalyzer:
    """Analyzer for mixed strategy: investing with 1 week to 1 month holding period"""
    
    def __init__(self):
        self.holding_period_min = 7  # 1 week
        self.holding_period_max = 30  # 1 month
    
    def analyze(self, stock_data: dict, financials_data: dict, history_data: dict) -> Dict:
        """
        Perform mixed analysis combining technical and fundamental factors
        for medium-term holding (1 week to 1 month)
        """
        try:
            if not stock_data or 'error' in stock_data:
                return {'error': 'Invalid stock data'}
            
            current_price = stock_data.get('price', 0)
            if current_price <= 0:
                return {'error': 'Invalid stock price'}
            
            # Get historical data
            data = history_data.get('data', [])
            if not data:
                return {'error': 'No historical data available'}
            
            # Calculate technical indicators
            technical_score = self._calculate_technical_score(data, current_price)
            
            # Calculate fundamental score
            fundamental_score = self._calculate_fundamental_score(stock_data, financials_data)
            
            # Combine scores (60% technical, 40% fundamental for medium-term)
            combined_score = (technical_score * 0.6) + (fundamental_score * 0.4)
            
            # Determine recommendation
            recommendation = self._get_recommendation(combined_score, current_price, data)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                technical_score, fundamental_score, combined_score,
                stock_data, financials_data, data, recommendation
            )
            
            return {
                'recommendation': recommendation,
                'reasoning': reasoning,
                'indicators': {
                    'technical_score': technical_score,
                    'fundamental_score': fundamental_score,
                    'combined_score': combined_score,
                    'holding_period': f"{self.holding_period_min}-{self.holding_period_max} days"
                },
                'strategy': 'mixed',
                'timeframe': '1 week - 1 month'
            }
        
        except Exception as e:
            logger.error(f"Error in mixed analysis: {e}")
            return {'error': f'Analysis error: {str(e)}'}
    
    def _calculate_technical_score(self, data: list, current_price: float) -> float:
        """Calculate technical analysis score (0-100)"""
        try:
            if len(data) < 20:
                return 50.0  # Neutral if insufficient data
            
            # Get recent prices
            recent_prices = [d.get('close', 0) for d in data[-20:]]
            recent_prices = [p for p in recent_prices if p > 0]
            
            if not recent_prices:
                return 50.0
            
            score = 50.0  # Start neutral
            
            # Price momentum (last 5 days vs previous 5 days)
            if len(recent_prices) >= 10:
                recent_avg = sum(recent_prices[-5:]) / 5
                previous_avg = sum(recent_prices[-10:-5]) / 5
                momentum = ((recent_avg - previous_avg) / previous_avg) * 100
                score += momentum * 2  # Weight momentum
            
            # Price position relative to recent range
            if len(recent_prices) >= 20:
                high_20 = max(recent_prices[-20:])
                low_20 = min(recent_prices[-20:])
                if high_20 > low_20:
                    price_position = ((current_price - low_20) / (high_20 - low_20)) * 100
                    # Prefer middle range for medium-term
                    if 30 <= price_position <= 70:
                        score += 10
                    elif price_position < 20:
                        score += 5  # Oversold, potential bounce
                    elif price_position > 80:
                        score -= 5  # Overbought, potential pullback
            
            # Volume trend (if available)
            recent_volumes = [d.get('volume', 0) for d in data[-10:]]
            if recent_volumes and all(v > 0 for v in recent_volumes):
                avg_volume = sum(recent_volumes) / len(recent_volumes)
                current_volume = recent_volumes[-1] if recent_volumes else 0
                if current_volume > avg_volume * 1.2:
                    score += 5  # Increasing volume is positive
            
            # Moving average trend
            if len(recent_prices) >= 10:
                ma_short = sum(recent_prices[-5:]) / 5
                ma_long = sum(recent_prices[-10:]) / 10
                if ma_short > ma_long:
                    score += 10  # Uptrend
                else:
                    score -= 5  # Downtrend
            
            # Clamp score between 0 and 100
            return max(0, min(100, score))
        
        except Exception as e:
            logger.error(f"Error calculating technical score: {e}")
            return 50.0
    
    def _calculate_fundamental_score(self, stock_data: dict, financials_data: dict) -> float:
        """Calculate fundamental analysis score (0-100)"""
        try:
            score = 50.0  # Start neutral
            
            # Market cap (larger is generally more stable)
            market_cap = stock_data.get('market_cap', 0)
            if market_cap > 10_000_000_000:  # > $10B
                score += 10
            elif market_cap > 2_000_000_000:  # > $2B
                score += 5
            
            # P/E ratio (reasonable range)
            pe_ratio = stock_data.get('pe_ratio', 0)
            if 0 < pe_ratio < 25:
                score += 10
            elif 25 <= pe_ratio < 40:
                score += 5
            elif pe_ratio >= 40:
                score -= 10  # Overvalued
            
            # Revenue growth (if available)
            revenue_growth = financials_data.get('revenue_growth', 0)
            if revenue_growth > 10:
                score += 10
            elif revenue_growth > 5:
                score += 5
            elif revenue_growth < -5:
                score -= 10
            
            # Profitability
            profit_margin = financials_data.get('profit_margin', 0)
            if profit_margin > 15:
                score += 10
            elif profit_margin > 5:
                score += 5
            elif profit_margin < 0:
                score -= 15  # Losing money
            
            # Debt to equity (lower is better)
            debt_to_equity = financials_data.get('debt_to_equity', 0)
            if 0 < debt_to_equity < 1:
                score += 5
            elif debt_to_equity > 2:
                score -= 10
            
            # Dividend yield (bonus for income)
            dividend_yield = stock_data.get('dividend_yield', 0)
            if dividend_yield > 2:
                score += 5
            
            # Clamp score between 0 and 100
            return max(0, min(100, score))
        
        except Exception as e:
            logger.error(f"Error calculating fundamental score: {e}")
            return 50.0
    
    def _get_recommendation(self, combined_score: float, current_price: float, data: list) -> Dict:
        """Generate recommendation based on combined score"""
        try:
            # Calculate price targets based on recent volatility
            recent_prices = [d.get('close', 0) for d in data[-20:]]
            recent_prices = [p for p in recent_prices if p > 0]
            
            if not recent_prices:
                return {
                    'action': 'HOLD',
                    'entry_price': current_price,
                    'target_price': current_price * 1.05,
                    'stop_loss': current_price * 0.95,
                    'confidence': 50
                }
            
            # Calculate volatility
            avg_price = sum(recent_prices) / len(recent_prices)
            price_std = sum(abs(p - avg_price) for p in recent_prices) / len(recent_prices)
            volatility = (price_std / avg_price) * 100 if avg_price > 0 else 5
            
            # Set targets based on score and volatility
            if combined_score >= 70:
                action = 'BUY'
                # Target: 5-10% gain over 2-4 weeks
                target_percent = min(10, 5 + volatility)
                target_price = current_price * (1 + target_percent / 100)
                stop_loss = current_price * 0.95  # 5% stop loss
                confidence = min(90, 60 + int(combined_score / 2))
            
            elif combined_score >= 55:
                action = 'BUY'
                # Conservative target: 3-7% gain
                target_percent = min(7, 3 + volatility * 0.5)
                target_price = current_price * (1 + target_percent / 100)
                stop_loss = current_price * 0.96  # 4% stop loss
                confidence = min(75, 50 + int(combined_score / 3))
            
            elif combined_score >= 45:
                action = 'HOLD'
                target_price = current_price * 1.02  # Small upside
                stop_loss = current_price * 0.98
                confidence = 50
            
            elif combined_score >= 30:
                action = 'HOLD'
                target_price = current_price * 1.0  # Flat
                stop_loss = current_price * 0.97
                confidence = 40
            
            else:
                action = 'AVOID'
                target_price = current_price * 0.98
                stop_loss = current_price * 1.02  # Inverse for avoid
                confidence = 30
            
            return {
                'action': action,
                'entry_price': current_price,
                'target_price': target_price,
                'stop_loss': stop_loss,
                'confidence': confidence,
                'holding_period_days': (self.holding_period_min + self.holding_period_max) // 2
            }
        
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            return {
                'action': 'HOLD',
                'entry_price': current_price,
                'target_price': current_price * 1.05,
                'stop_loss': current_price * 0.95,
                'confidence': 50
            }
    
    def _generate_reasoning(self, technical_score: float, fundamental_score: float,
                           combined_score: float, stock_data: dict, financials_data: dict,
                           data: list, recommendation: dict) -> str:
        """Generate human-readable reasoning"""
        try:
            symbol = stock_data.get('symbol', 'Stock')
            current_price = stock_data.get('price', 0)
            action = recommendation.get('action', 'HOLD')
            confidence = recommendation.get('confidence', 50)
            holding_days = recommendation.get('holding_period_days', 21)
            
            reasoning = f"MIXED STRATEGY ANALYSIS for {symbol}\n"
            reasoning += f"{'='*60}\n\n"
            reasoning += f"Current Price: ${current_price:.2f}\n"
            reasoning += f"Recommended Action: {action}\n"
            reasoning += f"Confidence: {confidence}%\n"
            reasoning += f"Suggested Holding Period: {holding_days} days (1 week - 1 month)\n\n"
            
            reasoning += "SCORE BREAKDOWN:\n"
            reasoning += f"  Technical Score: {technical_score:.1f}/100\n"
            reasoning += f"  Fundamental Score: {fundamental_score:.1f}/100\n"
            reasoning += f"  Combined Score: {combined_score:.1f}/100\n\n"
            
            reasoning += "TECHNICAL ANALYSIS:\n"
            if technical_score >= 60:
                reasoning += "  ✓ Positive momentum and trend indicators\n"
            elif technical_score >= 40:
                reasoning += "  → Mixed technical signals, neutral momentum\n"
            else:
                reasoning += "  ✗ Weak technical indicators, bearish momentum\n"
            
            reasoning += "\nFUNDAMENTAL ANALYSIS:\n"
            if fundamental_score >= 60:
                reasoning += "  ✓ Strong fundamentals: good valuation and financial health\n"
            elif fundamental_score >= 40:
                reasoning += "  → Moderate fundamentals: acceptable but not exceptional\n"
            else:
                reasoning += "  ✗ Weak fundamentals: concerns about valuation or financials\n"
            
            reasoning += f"\nPRICE TARGETS:\n"
            reasoning += f"  Entry: ${recommendation.get('entry_price', current_price):.2f}\n"
            reasoning += f"  Target: ${recommendation.get('target_price', current_price):.2f} "
            reasoning += f"({((recommendation.get('target_price', current_price) / current_price - 1) * 100):.1f}% gain)\n"
            reasoning += f"  Stop Loss: ${recommendation.get('stop_loss', current_price):.2f}\n\n"
            
            reasoning += "STRATEGY RATIONALE:\n"
            if action == 'BUY':
                reasoning += f"  This stock shows promise for a {holding_days}-day holding period.\n"
                reasoning += "  The combination of technical momentum and solid fundamentals suggests\n"
                reasoning += "  potential for moderate gains over the next 1-4 weeks.\n"
            elif action == 'HOLD':
                reasoning += f"  Current position is neutral. Monitor for {holding_days} days.\n"
                reasoning += "  Wait for clearer signals before making a decision.\n"
            else:
                reasoning += "  Not recommended for medium-term holding at this time.\n"
                reasoning += "  Technical or fundamental concerns outweigh potential benefits.\n"
            
            reasoning += f"\nRISK ASSESSMENT:\n"
            reasoning += f"  Risk Level: {'Low' if confidence >= 70 else 'Moderate' if confidence >= 50 else 'High'}\n"
            reasoning += f"  Recommended Position Size: {'Standard' if confidence >= 60 else 'Reduced' if confidence >= 40 else 'Minimal'}\n"
            
            return reasoning
        
        except Exception as e:
            logger.error(f"Error generating reasoning: {e}")
            return f"Analysis completed for {stock_data.get('symbol', 'Stock')} with {recommendation.get('action', 'HOLD')} recommendation."

