# AI Integration Guide
## How AI is Integrated into the Hybrid System

### Overview

The hybrid prediction system now includes **three prediction sources**:
1. **Rule-Based**: Technical/fundamental analysis with learned weights
2. **ML Model**: Machine learning predictions from trained models
3. **AI Predictor**: LLM-powered analysis using news sentiment, market insights, and reasoning

### Architecture

```
HybridStockPredictor
├── Rule-Based Analyzer (TradingAnalyzer/InvestingAnalyzer/MixedAnalyzer)
├── ML Predictor (StockPredictionML)
└── AI Predictor (AIPredictor) ← NEW!
    └── SeekerAI (LLM research & sentiment analysis)
```

### How It Works

#### 1. **AI Predictor Component** (`ai_predictor.py`)

The `AIPredictor` class:
- Uses `SeekerAI` to research stocks (news, sentiment, insights)
- Converts research into actionable predictions (BUY/SELL/HOLD)
- Calculates confidence based on:
  - Market sentiment (-1 to +1)
  - Company reputation (0 to 1)
  - Key insights and opportunities
  - Recent news sentiment

#### 2. **Integration into Hybrid Predictor**

The hybrid system now:
- Gets predictions from all three sources
- Calculates dynamic ensemble weights based on:
  - Historical performance of each method
  - Confidence levels
  - Agreement/disagreement between methods
- Combines predictions using weighted voting

#### 3. **Ensemble Weighting**

**Default Weights** (when all three available):
- Rule-based: 33%
- ML: 33%
- AI: 34%

**Dynamic Adjustment**:
- If ML has high accuracy (>60%): ML gets 40-50%, AI gets 25-30%
- If AI has high performance (>60%): AI gets 35-50%, others adjust
- If all agree: Confidence boosted by 15%
- If two agree: Confidence boosted by 5%

### Configuration

In `config.py`:

```python
# AI Predictor Configuration
AI_PREDICTOR_ENABLED = True  # Enable AI predictor
AI_PREDICTOR_DEFAULT_WEIGHT = 0.2  # Default weight (20%)
AI_PREDICTOR_MIN_CONFIDENCE = 40.0  # Minimum confidence to use AI prediction
```

### Usage

The AI predictor is automatically integrated - no code changes needed in your analysis calls!

```python
# The hybrid predictor automatically includes AI if available
prediction = hybrid_predictor.predict(stock_data, history_data, financials_data)

# Result includes:
# - rule_prediction: Rule-based prediction
# - ml_prediction: ML prediction (if available)
# - ai_prediction: AI prediction (if available) ← NEW!
# - ensemble_weights: {'rule': 0.33, 'ml': 0.33, 'ai': 0.34}
# - final recommendation: Weighted combination
```

### What AI Adds

1. **News Sentiment Analysis**: Analyzes recent news articles for bullish/bearish sentiment
2. **Market Context**: Understands broader market conditions and trends
3. **Company Reputation**: Evaluates company standing and credibility
4. **Risk Factors**: Identifies potential risks from news and analysis
5. **Opportunities**: Highlights potential opportunities
6. **Reasoning**: Provides human-readable explanations

### Performance Tracking

AI predictions are tracked separately:
- `ai_based` performance history
- Accuracy metrics
- Confidence distribution
- Automatic weight adjustment based on performance

### Requirements

1. **SeekerAI Enabled**: `SEEKER_AI_ENABLED = True` in config
2. **API Keys**: OpenAI API key (optional, for advanced analysis)
3. **News API**: NewsAPI key (optional, for news analysis)

### Benefits

✅ **Diversification**: Three independent prediction methods reduce over-reliance on one approach  
✅ **Context Awareness**: AI understands market context and news that ML/rule-based miss  
✅ **Adaptive**: Weights adjust automatically based on performance  
✅ **Robust**: System works even if AI is unavailable (falls back to rule+ML)

### Example Output

```
🤖 HYBRID AI ANALYSIS:
Rule-based: BUY (75.0%)
ML Model: BUY (68.5%)
AI Predictor: BUY (72.3%)
Ensemble Weights: Rule 35% / ML 35% / AI 30%
Final Prediction: BUY (72.1% confidence)
✓ All three methods agree - very high confidence
```

### Troubleshooting

**AI predictions not appearing?**
- Check `SEEKER_AI_ENABLED = True` in config
- Verify SeekerAI is initialized (check logs)
- Ensure API keys are configured (if using OpenAI/NewsAPI)

**AI weight too low?**
- AI needs at least 20 verified predictions to build performance history
- Check AI accuracy in performance tracking
- System automatically increases AI weight if it performs well

**Want to disable AI?**
- Set `AI_PREDICTOR_ENABLED = False` in config.py
- System will fall back to rule-based + ML only

### Future Enhancements

Potential improvements:
- Fine-tune AI weight based on market conditions
- Use AI for feature engineering in ML models
- AI-powered regime detection
- Sentiment-based position sizing
- News event impact prediction
