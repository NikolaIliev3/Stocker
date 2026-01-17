"""
Mixed Strategy Analyzer for Stocker App
Combines technical and fundamental analysis for medium-term holding (1 week - 1 month)
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional
from datetime import datetime, timedelta
from ta.trend import MACD, EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator
try:
    from ta.volume import MFIIndicator
    HAS_MFI = True
except ImportError:
    HAS_MFI = False
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("MFIIndicator not available in ta library, will calculate manually")

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
        """Calculate technical analysis score (0-100) using advanced indicators"""
        try:
            if len(data) < 20:
                return 50.0  # Neutral if insufficient data
            
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(data)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            df = df.sort_index()
            
            # Ensure we have required columns
            if 'close' not in df.columns or len(df) < 20:
                return 50.0
            
            score = 50.0  # Start neutral
            
            # Calculate technical indicators
            try:
                # RSI
                rsi_indicator = RSIIndicator(close=df['close'], window=14)
                rsi = float(rsi_indicator.rsi().iloc[-1])
                
                # RSI signals (enhanced with stronger penalties)
                if rsi < 30:
                    score += 15  # Oversold - strong buy signal
                elif rsi < 40:
                    score += 8   # Approaching oversold
                elif rsi > 80:
                    score -= 20  # Extremely overbought - strong sell signal
                elif rsi > 70:
                    score -= 15  # Overbought - sell signal
                elif rsi > 60:
                    score -= 8   # Approaching overbought
                
                # Money Flow Index (MFI)
                try:
                    if HAS_MFI:
                        mfi_indicator = MFIIndicator(high=df['high'], low=df['low'], close=df['close'], volume=df['volume'])
                        mfi = float(mfi_indicator.money_flow_index().iloc[-1])
                    else:
                        # Calculate MFI manually
                        typical_price = (df['high'] + df['low'] + df['close']) / 3
                        raw_money_flow = typical_price * df['volume']
                        positive_flow = raw_money_flow.where(typical_price > typical_price.shift(1), 0)
                        negative_flow = raw_money_flow.where(typical_price < typical_price.shift(1), 0)
                        positive_mf = positive_flow.rolling(window=14).sum()
                        negative_mf = negative_flow.rolling(window=14).sum()
                        money_ratio = positive_mf / negative_mf.replace([np.inf, -np.inf, np.nan], 0)
                        mfi_values = 100 - (100 / (1 + money_ratio.replace([np.inf, -np.inf, np.nan], 1)))
                        mfi = float(mfi_values.iloc[-1]) if not pd.isna(mfi_values.iloc[-1]) else 50
                    
                    # MFI signals
                    if mfi < 20:
                        score += 12  # Oversold
                    elif mfi > 80:
                        score -= 18  # Overbought - strong sell signal
                    elif mfi > 70:
                        score -= 10  # Approaching overbought
                    
                    # Multiple overbought penalty
                    if rsi > 70 and mfi > 80:
                        score -= 10  # Additional penalty for both overbought
                except Exception as e:
                    logger.debug(f"Could not calculate MFI in mixed analyzer: {e}")
            except:
                pass
            
            try:
                # MACD
                macd_indicator = MACD(close=df['close'])
                macd = float(macd_indicator.macd().iloc[-1])
                macd_signal = float(macd_indicator.macd_signal().iloc[-1])
                macd_diff = float(macd_indicator.macd_diff().iloc[-1])
                
                # MACD signals
                if macd_diff > 0 and macd > macd_signal:
                    score += 12  # Bullish crossover
                elif macd_diff < 0 and macd < macd_signal:
                    score -= 12  # Bearish crossover
            except:
                pass
            
            try:
                # Moving Averages
                ema_20 = EMAIndicator(close=df['close'], window=20)
                ema_50 = EMAIndicator(close=df['close'], window=50)
                ema_20_val = float(ema_20.ema_indicator().iloc[-1])
                ema_50_val = float(ema_50.ema_indicator().iloc[-1])
                
                # EMA crossover signals
                if current_price > ema_20_val > ema_50_val:
                    score += 10  # Strong uptrend
                elif current_price > ema_20_val:
                    score += 5   # Above short-term MA
                elif current_price < ema_20_val < ema_50_val:
                    score -= 10  # Strong downtrend
                elif current_price < ema_20_val:
                    score -= 5   # Below short-term MA
            except:
                pass
            
            try:
                # Bollinger Bands
                bb = BollingerBands(close=df['close'], window=20, window_dev=2)
                bb_upper = float(bb.bollinger_hband().iloc[-1])
                bb_lower = float(bb.bollinger_lband().iloc[-1])
                bb_middle = float(bb.bollinger_mavg().iloc[-1])
                
                # BB position signals
                if current_price < bb_lower:
                    score += 10  # Price below lower band - oversold
                elif current_price > bb_upper:
                    score -= 10  # Price above upper band - overbought
                elif bb_lower < current_price < bb_middle:
                    score += 5   # Below middle - potential bounce
            except:
                pass
            
            # Price momentum (last 5 days vs previous 5 days)
            recent_prices = df['close'].tail(20).tolist()
            if len(recent_prices) >= 10:
                recent_avg = sum(recent_prices[-5:]) / 5
                previous_avg = sum(recent_prices[-10:-5]) / 5
                if previous_avg > 0:
                    momentum = ((recent_avg - previous_avg) / previous_avg) * 100
                    score += momentum * 1.5  # Weight momentum
            
            # Volume trend
            if 'volume' in df.columns:
                recent_volumes = df['volume'].tail(10)
                if len(recent_volumes) > 0:
                    avg_volume = recent_volumes.mean()
                    current_volume = recent_volumes.iloc[-1]
                    if avg_volume > 0:
                        volume_ratio = current_volume / avg_volume
                        if volume_ratio > 1.5:
                            score += 8  # High volume confirms move
                        elif volume_ratio < 0.7:
                            score -= 5  # Low volume - weak move
            
            # Price position relative to recent range
            if len(recent_prices) >= 20:
                high_20 = max(recent_prices[-20:])
                low_20 = min(recent_prices[-20:])
                if high_20 > low_20:
                    price_position = ((current_price - low_20) / (high_20 - low_20)) * 100
                    # Prefer middle range for medium-term
                    if 30 <= price_position <= 70:
                        score += 8
                    elif price_position < 20:
                        score += 5  # Oversold, potential bounce
                    elif price_position > 80:
                        score -= 5  # Overbought, potential pullback
            
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
            
            # Set targets based on MIXED strategy timeframe (1 week - 1 month)
            # INVERTED: User reported BUY/SELL are backwards
            # High score (bullish) should be SELL, low score (bearish) should be BUY
            if combined_score >= 70:
                action = 'SELL'  # Changed from BUY - strong bullish signals mean stock is rising, sell to take profits
                # Strong sell: stock is high, expect 7-10% decline over 2-4 weeks
                target_percent = min(10, 7 + min(volatility * 0.3, 3))
                target_price = current_price * (1 - target_percent / 100)  # Price goes down
                stop_loss = current_price * 1.06  # 6% stop loss (price goes up)
                confidence = min(90, 60 + int(combined_score / 2))
            
            elif combined_score >= 55:
                action = 'SELL'  # Changed from BUY - moderate bullish signals
                # Moderate sell: expect 4-7% decline over 2-4 weeks
                target_percent = min(7, 4 + min(volatility * 0.25, 3))
                target_price = current_price * (1 - target_percent / 100)  # Price goes down
                stop_loss = current_price * 1.05  # 5% stop loss (price goes up)
                confidence = min(75, 50 + int(combined_score / 3))
            
            elif combined_score >= 45:
                action = 'HOLD'
                # Neutral: small 2-3% potential movement
                target_price = current_price * 1.025  # 2.5% upside
                stop_loss = current_price * 0.96  # 4% downside protection
                confidence = 50
            
            elif combined_score >= 30:
                action = 'HOLD'
                # Weak hold: minimal movement expected
                target_price = current_price * 1.01  # 1% upside
                stop_loss = current_price * 0.97  # 3% downside
                confidence = 40
            
            else:
                action = 'BUY'  # Changed from AVOID - low score (bearish) means stock is falling, buy at discount
                # Buy: expect 2-4% rise (recovery from low)
                target_price = current_price * 1.03  # 3% up
                stop_loss = current_price * 0.97  # 3% downside risk
                confidence = 30
            
            return {
                'action': action,
                'entry_price': current_price,
                'target_price': target_price,
                'stop_loss': stop_loss,
                'confidence': confidence,
                'confidence': confidence,
                'confidence': confidence,
                'holding_period_days': int(min(120, max(3, 20 + (10 - volatility) * 2))),  # Dynamic: 3 days to 4 months
                'estimated_days': int(min(120, max(3, 20 + (10 - volatility) * 2)))
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

