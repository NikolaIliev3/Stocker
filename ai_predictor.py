"""
AI Predictor Component
Converts AI research (LLM analysis, news sentiment) into actionable stock predictions
Integrates with hybrid predictor as third prediction source
"""
import logging
from typing import Dict, Optional
from datetime import datetime
import json
from pathlib import Path

try:
    from ai_researcher import SeekerAI
    HAS_SEEKER_AI = True
except ImportError:
    HAS_SEEKER_AI = False
    logger = logging.getLogger(__name__)
    logger.warning("SeekerAI not available. AI predictions will be disabled.")

from config import SEEKER_AI_ENABLED

logger = logging.getLogger(__name__)


class AIPredictor:
    """AI-powered stock predictor using LLM analysis and sentiment"""
    
    def __init__(self, data_dir: Path, seeker_ai: Optional[SeekerAI] = None):
        """
        Initialize AI Predictor
        
        Args:
            data_dir: Data directory for caching
            seeker_ai: Optional SeekerAI instance (will create if not provided)
        """
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SeekerAI if available and enabled
        self.seeker_ai = None
        if SEEKER_AI_ENABLED and HAS_SEEKER_AI:
            try:
                self.seeker_ai = seeker_ai or SeekerAI(data_dir)
                logger.info("AI Predictor initialized with SeekerAI")
            except Exception as e:
                logger.warning(f"Could not initialize SeekerAI: {e}")
                self.seeker_ai = None
        else:
            logger.info("AI Predictor disabled (SEEKER_AI_ENABLED=False or SeekerAI not available)")
        
        # Performance tracking
        self.performance_file = data_dir / "ai_predictor_performance.json"
        self.performance_history = self._load_performance()
    
    def _load_performance(self) -> Dict:
        """Load AI predictor performance history"""
        if self.performance_file.exists():
            try:
                with open(self.performance_file, 'r') as f:
                    return json.load(f)
            except (IOError, OSError, json.JSONDecodeError) as e:
                logger.debug(f"Error loading AI performance history: {e}")
        return {
            'predictions': [],
            'accuracy': 0.0,
            'total_predictions': 0,
            'correct_predictions': 0
        }
    
    def _save_performance(self) -> None:
        """Save AI predictor performance history"""
        try:
            with open(self.performance_file, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
        except (IOError, OSError, json.JSONEncodeError) as e:
            logger.error(f"Error saving AI performance history: {e}")
    
    def is_available(self) -> bool:
        """Check if AI predictor is available"""
        return self.seeker_ai is not None and SEEKER_AI_ENABLED
    
    def predict(self, symbol: str, stock_data: dict, history_data: dict,
                financials_data: dict = None, base_analysis: dict = None) -> Optional[Dict]:
        """
        Generate AI-powered prediction
        
        Args:
            symbol: Stock symbol
            stock_data: Current stock data
            history_data: Historical price data
            financials_data: Financial data (optional)
            base_analysis: Base analysis from rule-based system (for context)
        
        Returns:
            Prediction dict with action, confidence, reasoning, or None if unavailable
        """
        if not self.is_available():
            return None
        
        try:
            # Get AI research
            research = self.seeker_ai.research_stock(symbol, stock_data, history_data)
            
            if 'error' in research:
                logger.warning(f"AI research failed for {symbol}: {research.get('error')}")
                return None
            
            # Convert research to prediction
            prediction = self._research_to_prediction(
                research, stock_data, history_data, financials_data, base_analysis
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"AI prediction failed for {symbol}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _research_to_prediction(self, research: Dict, stock_data: dict,
                                history_data: dict, financials_data: dict = None,
                                base_analysis: dict = None) -> Dict:
        """
        Convert AI research into actionable prediction
        
        Args:
            research: SeekerAI research results
            stock_data: Current stock data
            history_data: Historical data
            financials_data: Financial data
            base_analysis: Base analysis for context
        
        Returns:
            Prediction dict with action, confidence, reasoning
        """
        current_price = stock_data.get('price', 0)
        sentiment_score = research.get('sentiment_score', 0.0)  # -1 to 1
        reputation_score = research.get('reputation_score', 0.5)  # 0 to 1
        
        # Extract key insights
        key_insights = research.get('key_insights', [])
        risk_factors = research.get('risk_factors', [])
        opportunities = research.get('opportunities', [])
        news_articles = research.get('news_articles', [])
        
        # Calculate action based on sentiment and insights
        # Sentiment: -1 (very bearish) to +1 (very bullish)
        # Reputation: 0 (poor) to 1 (excellent)
        
        # Base score from sentiment (-50 to +50)
        score = sentiment_score * 50
        
        # Adjust based on reputation (multiplier)
        reputation_multiplier = 0.5 + (reputation_score * 0.5)  # 0.5 to 1.0
        score *= reputation_multiplier
        
        # Count positive vs negative insights
        positive_insights = len(opportunities) + sum(1 for insight in key_insights 
                                                    if any(word in insight.lower() 
                                                          for word in ['positive', 'growth', 'strong', 'bullish', 'buy']))
        negative_insights = len(risk_factors) + sum(1 for insight in key_insights 
                                                    if any(word in insight.lower() 
                                                          for word in ['risk', 'concern', 'weak', 'bearish', 'sell']))
        
        # Adjust score based on insight balance
        insight_diff = positive_insights - negative_insights
        score += insight_diff * 5  # Each insight worth 5 points
        
        # Recent news sentiment (if available)
        if news_articles:
            recent_news = news_articles[:5]  # Last 5 articles
            # Simple sentiment: count positive keywords
            positive_keywords = ['growth', 'profit', 'gain', 'upgrade', 'beat', 'strong', 'positive']
            negative_keywords = ['loss', 'decline', 'downturn', 'downgrade', 'miss', 'weak', 'negative']
            
            news_sentiment = 0
            for article in recent_news:
                title = article.get('title', '').lower()
                description = article.get('description', '').lower()
                text = f"{title} {description}"
                
                pos_count = sum(1 for word in positive_keywords if word in text)
                neg_count = sum(1 for word in negative_keywords if word in text)
                news_sentiment += (pos_count - neg_count)
            
            # Normalize news sentiment (-10 to +10)
            news_score = max(-10, min(10, news_sentiment))
            score += news_score
        
        # Clamp score to -100 to +100 range
        score = max(-100, min(100, score))
        
        # Convert score to action and confidence
        if score >= 30:
            action = 'BUY'
            confidence = min(95, 50 + (score - 30) * 0.75)  # 50% to 95%
        elif score <= -30:
            action = 'SELL'
            confidence = min(95, 50 + abs(score + 30) * 0.75)  # 50% to 95%
        else:
            action = 'HOLD'
            confidence = 50 + abs(score) * 0.5  # 50% to 65%
        
        # Generate reasoning
        reasoning = self._generate_ai_reasoning(
            research, score, action, confidence, base_analysis
        )
        
        # Calculate entry/target/stop based on current price and action
        entry_price = current_price
        if action == 'BUY':
            target_price = current_price * 1.15  # 15% target
            stop_loss = current_price * 0.95  # 5% stop loss
        elif action == 'SELL':
            target_price = current_price * 0.85  # 15% target (short)
            stop_loss = current_price * 1.05  # 5% stop loss
        else:  # HOLD
            target_price = current_price * 1.05  # 5% target
            stop_loss = current_price * 0.98  # 2% stop loss
        
        return {
            'action': action,
            'confidence': confidence,
            'entry_price': entry_price,
            'target_price': target_price,
            'stop_loss': stop_loss,
            'reasoning': reasoning,
            'sentiment_score': sentiment_score,
            'reputation_score': reputation_score,
            'ai_score': score,
            'key_insights': key_insights[:5],  # Top 5 insights
            'risk_factors': risk_factors[:3],  # Top 3 risks
            'opportunities': opportunities[:3],  # Top 3 opportunities
            'news_count': len(news_articles),
            'method': 'ai_powered'
        }
    
    def _generate_ai_reasoning(self, research: Dict, score: float, action: str,
                               confidence: float, base_analysis: dict = None) -> str:
        """Generate human-readable reasoning from AI research"""
        reasoning_parts = []
        
        reasoning_parts.append(f"🤖 AI ANALYSIS (Score: {score:.1f}/100):")
        
        # Sentiment
        sentiment_score = research.get('sentiment_score', 0.0)
        if sentiment_score > 0.3:
            reasoning_parts.append(f"✓ Positive market sentiment ({sentiment_score:.2f})")
        elif sentiment_score < -0.3:
            reasoning_parts.append(f"✗ Negative market sentiment ({sentiment_score:.2f})")
        else:
            reasoning_parts.append(f"○ Neutral market sentiment ({sentiment_score:.2f})")
        
        # Reputation
        reputation_score = research.get('reputation_score', 0.5)
        if reputation_score > 0.7:
            reasoning_parts.append(f"✓ Strong company reputation ({reputation_score:.1%})")
        elif reputation_score < 0.3:
            reasoning_parts.append(f"✗ Weak company reputation ({reputation_score:.1%})")
        
        # Key insights
        key_insights = research.get('key_insights', [])
        if key_insights:
            reasoning_parts.append(f"\n📊 Key Insights:")
            for i, insight in enumerate(key_insights[:3], 1):
                reasoning_parts.append(f"  {i}. {insight}")
        
        # Opportunities
        opportunities = research.get('opportunities', [])
        if opportunities:
            reasoning_parts.append(f"\n💡 Opportunities:")
            for i, opp in enumerate(opportunities[:3], 1):
                reasoning_parts.append(f"  {i}. {opp}")
        
        # Risk factors
        risk_factors = research.get('risk_factors', [])
        if risk_factors:
            reasoning_parts.append(f"\n⚠️ Risk Factors:")
            for i, risk in enumerate(risk_factors[:3], 1):
                reasoning_parts.append(f"  {i}. {risk}")
        
        # News summary
        news_count = len(research.get('news_articles', []))
        if news_count > 0:
            reasoning_parts.append(f"\n📰 Analyzed {news_count} recent news articles")
        
        # Price movement explanation
        price_explanation = research.get('price_movement_explanation', '')
        if price_explanation:
            reasoning_parts.append(f"\n📈 Price Movement: {price_explanation}")
        
        # AI summary
        summary = research.get('summary', '')
        if summary:
            reasoning_parts.append(f"\n📝 AI Summary: {summary}")
        
        # Final recommendation
        reasoning_parts.append(f"\n🎯 AI Recommendation: {action} ({confidence:.1f}% confidence)")
        
        return "\n".join(reasoning_parts)
    
    def update_performance(self, prediction: Dict, actual_outcome: bool) -> None:
        """Update performance tracking after prediction is verified"""
        if not prediction or prediction.get('method') != 'ai_powered':
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
    
    def get_performance(self) -> Dict:
        """Get AI predictor performance metrics"""
        return {
            'accuracy': self.performance_history.get('accuracy', 0.0),
            'total_predictions': self.performance_history.get('total_predictions', 0),
            'correct_predictions': self.performance_history.get('correct_predictions', 0)
        }
