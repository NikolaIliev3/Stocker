"""
LLM-Based AI Predictor
Uses Large Language Models to analyze technical indicators, patterns, and provide reasoning
More useful than SeekerAI for trading predictions

This predictor analyzes technical indicators (RSI, MACD, Bollinger Bands, etc.) rather than
news sentiment, making it more useful for trading decisions.

Features:
- Multi-factor technical analysis
- Pattern recognition
- Consensus analysis (combines rule-based + ML insights)
- Optional GPT-4 enhancement (falls back to rule-based if unavailable)
"""
import logging
from typing import Dict, Optional, List
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMAIPredictor:
    """
    LLM-based AI predictor that analyzes technical indicators and provides reasoning
    More useful than news-based SeekerAI for trading decisions
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.performance_file = data_dir / "llm_ai_predictor_performance.json"
        self.performance_history = self._load_performance()
        
        # Check if OpenAI API is available
        self.has_openai = False
        
        # Fallback: Use rule-based reasoning if no LLM available
        self.use_llm = False
    
    def _check_openai_available(self) -> bool:
        """Check if OpenAI API key is available"""
        return False
    
    def _load_performance(self) -> Dict:
        """Load performance history"""
        if self.performance_file.exists():
            try:
                with open(self.performance_file, 'r') as f:
                    return json.load(f)
            except (IOError, OSError, json.JSONDecodeError) as e:
                logger.debug(f"Error loading LLM AI performance: {e}")
        return {
            'predictions': [],
            'accuracy': 0.0,
            'total_predictions': 0,
            'correct_predictions': 0
        }
    
    def _save_performance(self):
        """Save performance history"""
        try:
            with open(self.performance_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
        except (IOError, OSError, json.JSONEncodeError) as e:
            logger.error(f"Error saving LLM AI performance: {e}")
    
    def is_available(self) -> bool:
        """Check if LLM AI predictor is available"""
        return True  # Always available (uses fallback if no LLM)
    
    def predict(self, symbol: str, stock_data: dict, history_data: dict,
                base_analysis: dict, rule_prediction: dict = None,
                ml_prediction: dict = None) -> Optional[Dict]:
        """
        Generate AI prediction based on technical analysis
        
        Args:
            symbol: Stock symbol
            stock_data: Current stock data
            history_data: Historical price data
            base_analysis: Rule-based analysis with indicators
            rule_prediction: Rule-based prediction
            ml_prediction: ML prediction (if available)
        
        Returns:
            Prediction dict with action, confidence, reasoning
        """
        try:
            # Extract key information from analysis
            indicators = base_analysis.get('indicators', {})
            price_action = base_analysis.get('price_action', {})
            volume_analysis = base_analysis.get('volume_analysis', {})
            market_regime = base_analysis.get('market_regime', 'unknown')
            
            current_price = stock_data.get('price', 0)
            if current_price <= 0:
                return None
            
            # Analyze using LLM if available, otherwise use rule-based reasoning
            if self.use_llm and self.has_openai:
                prediction = self._llm_analyze(
                    symbol, indicators, price_action, volume_analysis,
                    market_regime, current_price, rule_prediction, ml_prediction
                )
            else:
                # Fallback: Advanced rule-based reasoning
                prediction = self._advanced_rule_based_analyze(
                    symbol, indicators, price_action, volume_analysis,
                    market_regime, current_price, rule_prediction, ml_prediction
                )
            
            return prediction
            
        except Exception as e:
            logger.error(f"LLM AI prediction failed for {symbol}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _llm_analyze(self, symbol: str, indicators: dict, price_action: dict,
                    volume_analysis: dict, market_regime: str, current_price: float,
                    rule_prediction: dict = None, ml_prediction: dict = None) -> Dict:
        """Use LLM to analyze technical indicators - DEPRECATED due to removal of OpenAI"""
        # Fallback to rule-based since OpenAI is removed
        return self._advanced_rule_based_analyze(
            symbol, indicators, price_action, volume_analysis,
            market_regime, current_price, rule_prediction, ml_prediction
        )
    
    def _parse_llm_response(self, content: str) -> Dict:
        """Parse LLM response when JSON parsing fails"""
        action = 'HOLD'
        confidence = 50
        reasoning = content
        
        # Try to extract action
        content_upper = content.upper()
        if 'BUY' in content_upper and content_upper.index('BUY') < content_upper.index('SELL') if 'SELL' in content_upper else True:
            action = 'BUY'
        elif 'SELL' in content_upper:
            action = 'SELL'
        
        # Try to extract confidence
        import re
        conf_match = re.search(r'(\d+)%', content)
        if conf_match:
            confidence = int(conf_match.group(1))
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': content,
            'key_factors': []
        }
    
    def _prepare_analysis_context(self, symbol: str, indicators: dict,
                                  price_action: dict, volume_analysis: dict,
                                  market_regime: str, current_price: float,
                                  rule_prediction: dict = None,
                                  ml_prediction: dict = None) -> str:
        """Prepare context for LLM analysis"""
        context_parts = []
        
        context_parts.append(f"Stock: {symbol}")
        context_parts.append(f"Current Price: ${current_price:.2f}")
        context_parts.append(f"Market Regime: {market_regime}")
        
        # Technical indicators
        if indicators:
            context_parts.append("\nTechnical Indicators:")
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                if isinstance(rsi, (int, float)):
                    context_parts.append(f"  RSI: {rsi:.2f} ({'Oversold' if rsi < 30 else 'Overbought' if rsi > 70 else 'Neutral'})")
            
            # MACD can be a dict or separate values
            if 'macd' in indicators:
                macd = indicators['macd']
                if isinstance(macd, dict):
                    context_parts.append(f"  MACD: {macd.get('macd', 0):.2f}, Signal: {macd.get('signal', 0):.2f}")
                else:
                    macd_signal = indicators.get('macd_signal', 0)
                    context_parts.append(f"  MACD: {macd:.2f}, Signal: {macd_signal:.2f}")
            
            # Bollinger Bands
            if 'bollinger' in indicators:
                bb = indicators['bollinger']
                if isinstance(bb, dict):
                    context_parts.append(f"  Bollinger Bands: Upper={bb.get('upper', 0):.2f}, Lower={bb.get('lower', 0):.2f}")
                else:
                    bb_upper = indicators.get('bb_upper', current_price)
                    bb_lower = indicators.get('bb_lower', current_price)
                    context_parts.append(f"  Bollinger Bands: Upper={bb_upper:.2f}, Lower={bb_lower:.2f}")
            
            # EMA indicators
            if 'ema_20' in indicators:
                ema_20 = indicators.get('ema_20', current_price)
                ema_50 = indicators.get('ema_50', current_price)
                context_parts.append(f"  EMA 20: {ema_20:.2f}, EMA 50: {ema_50:.2f}")
        
        # Price action
        if price_action:
            context_parts.append("\nPrice Action:")
            context_parts.append(f"  Trend: {price_action.get('trend', 'unknown')}")
            context_parts.append(f"  Price Position: {price_action.get('price_position_percent', 0):.1f}% of recent range")
        
        # Volume
        if volume_analysis:
            context_parts.append("\nVolume Analysis:")
            context_parts.append(f"  Volume Trend: {volume_analysis.get('trend', 'unknown')}")
            context_parts.append(f"  Volume Ratio: {volume_analysis.get('volume_ratio', 1.0):.2f}")
        
        # Other predictions
        if rule_prediction:
            context_parts.append(f"\nRule-Based Prediction: {rule_prediction.get('action', 'HOLD')} ({rule_prediction.get('confidence', 50):.1f}%)")
        
        if ml_prediction:
            context_parts.append(f"ML Prediction: {ml_prediction.get('action', 'HOLD')} ({ml_prediction.get('confidence', 50):.1f}%)")
        
        return "\n".join(context_parts)
    
    def _advanced_rule_based_analyze(self, symbol: str, indicators: dict,
                                     price_action: dict, volume_analysis: dict,
                                     market_regime: str, current_price: float,
                                     rule_prediction: dict = None,
                                     ml_prediction: dict = None) -> Dict:
        """
        Advanced rule-based analysis (fallback when LLM not available)
        Uses pattern recognition and multi-factor analysis
        """
        score = 0.0
        factors = []
        confidence_base = 50.0
        
        # 1. RSI Analysis
        if 'rsi' in indicators:
            rsi = indicators['rsi']
            if rsi < 30:
                score += 15
                factors.append(f"RSI oversold ({rsi:.1f}) - potential bounce")
                confidence_base += 5
            elif rsi > 70:
                score -= 15
                factors.append(f"RSI overbought ({rsi:.1f}) - potential pullback")
                confidence_base += 5
            elif 40 < rsi < 60:
                score += 5
                factors.append(f"RSI neutral ({rsi:.1f}) - healthy momentum")
        
        # 2. MACD Analysis
        if 'macd' in indicators:
            macd_data = indicators['macd']
            if isinstance(macd_data, dict):
                macd_line = macd_data.get('macd', 0)
                signal = macd_data.get('signal', 0)
                histogram = macd_data.get('histogram', 0)
            else:
                # MACD stored as separate values
                macd_line = float(macd_data) if isinstance(macd_data, (int, float)) else 0
                signal = float(indicators.get('macd_signal', 0)) if isinstance(indicators.get('macd_signal', 0), (int, float)) else 0
                histogram = float(indicators.get('macd_diff', 0)) if isinstance(indicators.get('macd_diff', 0), (int, float)) else 0
            
            if macd_line > signal and histogram > 0:
                score += 12
                factors.append("MACD bullish crossover - upward momentum")
                confidence_base += 3
            elif macd_line < signal and histogram < 0:
                score -= 12
                factors.append("MACD bearish crossover - downward momentum")
                confidence_base += 3
        
        # 3. Bollinger Bands Analysis
        if 'bollinger' in indicators:
            bb = indicators['bollinger']
            if isinstance(bb, dict):
                upper = bb.get('upper', current_price)
                lower = bb.get('lower', current_price)
                middle = bb.get('middle', current_price)
            else:
                # Bollinger stored as separate values
                upper = float(indicators.get('bb_upper', current_price))
                lower = float(indicators.get('bb_lower', current_price))
                middle = float(indicators.get('bb_middle', current_price))
            
            if current_price < lower:
                score += 10
                factors.append("Price below lower Bollinger Band - oversold")
            elif current_price > upper:
                score -= 10
                factors.append("Price above upper Bollinger Band - overbought")
            elif lower < current_price < middle:
                score += 5
                factors.append("Price in lower half of Bollinger Band - potential support")
        
        # 4. Price Action Analysis
        if price_action:
            trend = price_action.get('trend', 'sideways')
            price_position = price_action.get('price_position_percent', 50)
            
            if trend == 'uptrend':
                score += 8
                factors.append("Uptrend detected - bullish structure")
            elif trend == 'downtrend':
                score -= 8
                factors.append("Downtrend detected - bearish structure")
            
            if price_position < 30:
                score += 6
                factors.append(f"Price near support ({price_position:.1f}%) - potential bounce")
            elif price_position > 70:
                score -= 6
                factors.append(f"Price near resistance ({price_position:.1f}%) - potential pullback")
        
        # 5. Volume Analysis
        if volume_analysis:
            volume_trend = volume_analysis.get('trend', 'neutral')
            volume_ratio = volume_analysis.get('volume_ratio', 1.0)
            
            if volume_trend == 'increasing' and volume_ratio > 1.2:
                # High volume confirms move
                if score > 0:
                    score += 5
                    factors.append("High volume confirms bullish move")
                elif score < 0:
                    score -= 5
                    factors.append("High volume confirms bearish move")
                confidence_base += 3
        
        # 6. Market Regime
        if market_regime == 'bull':
            score += 5
            factors.append("Bull market regime - favorable conditions")
        elif market_regime == 'bear':
            score -= 5
            factors.append("Bear market regime - cautious conditions")
        
        # 7. Consensus Analysis (if other predictions available)
        if rule_prediction and ml_prediction:
            rule_action = rule_prediction.get('action', 'HOLD')
            ml_action = ml_prediction.get('action', 'HOLD')
            
            if rule_action == ml_action:
                # Consensus - boost confidence
                if rule_action == 'BUY':
                    score += 8
                    factors.append("Rule-based and ML agree on BUY - strong consensus")
                elif rule_action == 'SELL':
                    score -= 8
                    factors.append("Rule-based and ML agree on SELL - strong consensus")
                confidence_base += 5
            elif rule_action != ml_action:
                # Disagreement - reduce confidence
                confidence_base -= 3
                factors.append("Rule-based and ML disagree - mixed signals")
        
        # Convert score to action and confidence
        if score >= 20:
            action = 'BUY'
            confidence = min(95, confidence_base + min(30, score * 0.8))
        elif score <= -20:
            action = 'SELL'
            confidence = min(95, confidence_base + min(30, abs(score) * 0.8))
        else:
            action = 'HOLD'
            confidence = max(45, confidence_base + abs(score) * 0.5)
            
            # Add trend context to factors
            if price_action:
                trend = price_action.get('trend', 'sideways')
                if trend == 'uptrend':
                    factors.append("**Price is rising** - neutral signals suggest waiting")
                elif trend == 'downtrend':
                    factors.append("**Price is falling** - neutral signals suggest waiting")
        
        # Calculate entry/target/stop
        entry_price = current_price
        if action == 'BUY':
            target_price = current_price * 1.15
            stop_loss = current_price * 0.95
        elif action == 'SELL':
            target_price = current_price * 0.85
            stop_loss = current_price * 1.05
        else:
            target_price = current_price * 1.05
            stop_loss = current_price * 0.98
        
        # Generate reasoning
        reasoning = "🤖 Advanced AI Analysis:\n"
        reasoning += f"Score: {score:.1f}/100\n"
        reasoning += "\nKey Factors:\n"
        for i, factor in enumerate(factors[:5], 1):
            reasoning += f"  {i}. {factor}\n"
        
        return {
            'action': action,
            'confidence': confidence,
            'entry_price': entry_price,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'reasoning': reasoning,
            'key_factors': factors[:5],
            'score': score,
            'method': 'advanced_ai_powered',
            'model_used': 'rule_based_fallback' if not self.use_llm else 'llm_fallback'
        }
    
    def update_performance(self, prediction: Dict, actual_outcome: bool) -> None:
        """Update performance tracking"""
        if not prediction or prediction.get('method') not in ['llm_ai_powered', 'advanced_ai_powered']:
            return
        
        self.performance_history['predictions'].append({
            'was_correct': actual_outcome,
            'confidence': prediction.get('confidence', 50),
            'action': prediction.get('action', 'HOLD'),
            'date': datetime.now().isoformat()
        })
        
        # Update statistics
        self.performance_history['total_predictions'] = len(self.performance_history['predictions'])
        correct = sum(1 for p in self.performance_history['predictions'] if p.get('was_correct') is True)
        self.performance_history['correct_predictions'] = correct
        
        if self.performance_history['total_predictions'] > 0:
            self.performance_history['accuracy'] = (
                correct / self.performance_history['total_predictions'] * 100
            )
        
        # Keep only last 1000 predictions
        from config import ML_PERFORMANCE_HISTORY_LIMIT
        if len(self.performance_history['predictions']) > ML_PERFORMANCE_HISTORY_LIMIT:
            self.performance_history['predictions'] = \
                self.performance_history['predictions'][-ML_PERFORMANCE_HISTORY_LIMIT:]
        
        self._save_performance()
    
    def analyze_buy_opportunity(self, symbol: str, current_price: float, indicators: dict, 
                                 price_action: dict, strategy: str = 'mixed') -> Optional[Dict]:
        """
        Analyze if there's a better buying opportunity expected in the future
        
        Returns:
            Dict with predicted price and date, or None if no better opportunity expected
        """
        try:
            rsi = indicators.get('rsi', 50)
            target_multiplier = 0.95  # Default 5% drop for better entry
            days_offset = 7
            
            # Logic: If RSI is high (>60) or price is near upper BB, expect a pullback
            bb = indicators.get('bollinger', {})
            bb_upper = bb.get('upper', current_price * 1.05)
            bb_lower = bb.get('lower', current_price * 0.95)
            
            better_opportunity_expected = False
            confidence = 50
            reasoning = ""
            
            if rsi > 70:
                better_opportunity_expected = True
                target_price = current_price * 0.92  # 8% drop
                days_offset = 10
                confidence = 75
                reasoning = f"RSI is overbought ({rsi:.1f}). Expecting a significant pullback before better entry."
            elif rsi > 60:
                better_opportunity_expected = True
                target_price = current_price * 0.95  # 5% drop
                days_offset = 5
                confidence = 60
                reasoning = f"RSI is high ({rsi:.1f}). A temporary pullback is likely for a better entry."
            elif current_price > bb_upper * 0.98:
                better_opportunity_expected = True
                target_price = bb.get('middle', current_price * 0.97)
                days_offset = 4
                confidence = 65
                reasoning = "Price is near upper Bollinger Band. Pullback to mean line expected."
            
            if better_opportunity_expected:
                from datetime import timedelta
                predicted_date = (datetime.now() + timedelta(days=days_offset)).strftime("%Y-%m-%d")
                
                return {
                    'symbol': symbol,
                    'current_price': current_price,
                    'predicted_price': target_price,
                    'predicted_date': predicted_date,
                    'confidence': confidence,
                    'reasoning': reasoning
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in analyze_buy_opportunity for {symbol}: {e}")
            return None

    def get_performance(self) -> Dict:
        """Get performance metrics"""
        return {
            'accuracy': self.performance_history.get('accuracy', 0.0),
            'total_predictions': self.performance_history.get('total_predictions', 0),
            'correct_predictions': self.performance_history.get('correct_predictions', 0)
        }
