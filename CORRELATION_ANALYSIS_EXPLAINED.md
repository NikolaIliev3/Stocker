# Correlation Analysis Explained

## What is Correlation Analysis?

**Correlation analysis** measures how closely two or more stocks move together. It tells you whether stocks tend to move in the same direction, opposite directions, or independently.

## Why It Matters

### 1. **Diversification**
- **Low Correlation** = Good diversification
  - If stocks don't move together, losses in one can be offset by gains in another
  - Reduces overall portfolio risk
  
- **High Correlation** = Poor diversification
  - If stocks move together, you're not really diversified
  - All stocks might fall at the same time

### 2. **Risk Management**
- Identify which stocks move together (high correlation)
- Avoid holding too many highly correlated stocks
- Build a truly diversified portfolio

### 3. **Trading Opportunities**
- Find pairs that usually move together but have diverged
- Potential arbitrage or mean reversion opportunities
- Sector analysis (stocks in same sector often correlated)

## How It Works

### Correlation Values

Correlation is measured from **-1 to +1**:

- **+1.0** = Perfect positive correlation
  - Stocks move in exactly the same direction
  - Example: AAPL and MSFT (both tech stocks)
  
- **0.0** = No correlation
  - Stocks move independently
  - Example: Tech stock and utility stock
  
- **-1.0** = Perfect negative correlation
  - Stocks move in opposite directions
  - Example: Gold and Dollar (often inverse)

### Real-World Examples

**High Correlation (0.7 - 1.0)**:
- Tech stocks: AAPL, MSFT, GOOGL (all move with tech sector)
- Banks: JPM, BAC, WFC (all move with financial sector)
- Oil companies: XOM, CVX (both tied to oil prices)

**Low Correlation (0.0 - 0.3)**:
- Tech vs. Utilities: AAPL vs. NEE (different sectors)
- Growth vs. Value: TSLA vs. KO (different styles)
- US vs. International: SPY vs. EFA (different markets)

**Negative Correlation (-1.0 - 0.0)**:
- Stocks vs. Bonds (often inverse)
- Dollar vs. Gold (often inverse)
- Growth vs. Defensive (in different market conditions)

## How to Use It in Stocker

### Step 1: Enter Stock Symbols
```
AAPL,MSFT,GOOGL,AMZN,META
```

### Step 2: Click "Calculate Correlation"
The system:
1. Fetches historical price data for each stock
2. Calculates daily returns (price changes)
3. Compares returns between all pairs
4. Generates correlation matrix

### Step 3: Interpret Results

**Example Output:**
```
Correlation Analysis
==================================================

Highest Correlation: 0.85
  Between: AAPL, MSFT

Lowest Correlation: 0.12
  Between: AAPL, XOM

Average Correlation: 0.45
```

**What This Means:**
- AAPL and MSFT are highly correlated (0.85)
  - They move together → Not great for diversification
  - If you own both, you're doubling down on tech
  
- AAPL and XOM have low correlation (0.12)
  - They move independently → Good for diversification
  - Tech stock + Oil stock = Better risk spread

- Average correlation of 0.45
  - Moderate correlation overall
  - Could improve diversification

## Practical Applications

### 1. **Building a Diversified Portfolio**

**Goal**: Low average correlation (< 0.3)

**Example Good Portfolio**:
- AAPL (Tech) - Correlation: 0.2
- XOM (Energy) - Correlation: 0.15
- JNJ (Healthcare) - Correlation: 0.25
- **Average: 0.20** ✅ Good diversification

**Example Poor Portfolio**:
- AAPL (Tech) - Correlation: 0.8
- MSFT (Tech) - Correlation: 0.85
- GOOGL (Tech) - Correlation: 0.82
- **Average: 0.82** ❌ Poor diversification (all tech)

### 2. **Finding Trading Pairs**

**Pairs Trading Strategy**:
- Find two stocks with high historical correlation (0.8+)
- When correlation breaks (one goes up, other stays flat)
- Buy the laggard, sell the leader
- Wait for correlation to return

**Example**:
- AAPL and MSFT usually move together (0.85 correlation)
- AAPL jumps 5%, MSFT only moves 1%
- Opportunity: MSFT might catch up → Buy MSFT

### 3. **Sector Analysis**

**Identify Sector Relationships**:
- Tech stocks: Usually correlated (0.6-0.9)
- Financial stocks: Usually correlated (0.5-0.8)
- Utilities: Low correlation with tech (0.1-0.3)

**Use Case**:
- If tech sector is falling, utilities might hold steady
- Diversify across sectors, not just stocks

### 4. **Risk Assessment**

**Portfolio Risk**:
- High correlation = Higher risk (all stocks fall together)
- Low correlation = Lower risk (stocks move independently)

**Example**:
- Portfolio A: 5 tech stocks (avg correlation 0.8)
  - Risk: HIGH (all fall together in tech crash)
  
- Portfolio B: Tech, Energy, Healthcare, Finance (avg correlation 0.3)
  - Risk: LOWER (diversified across sectors)

## Correlation Methods

The app uses **Pearson Correlation** by default, but also supports:

1. **Pearson** (default)
   - Measures linear relationships
   - Most common method
   - Range: -1 to +1

2. **Spearman**
   - Measures rank relationships
   - Less sensitive to outliers
   - Good for non-linear relationships

3. **Kendall**
   - Measures ordinal relationships
   - Most robust to outliers
   - Good for small datasets

## Limitations

### 1. **Correlation ≠ Causation**
- Just because stocks move together doesn't mean one causes the other
- Both might be reacting to the same market forces

### 2. **Correlation Changes Over Time**
- Stocks that were correlated might diverge
- Market conditions change correlations
- Example: Tech stocks correlated in bull markets, less in bear markets

### 3. **Not Predictive**
- Past correlation doesn't guarantee future correlation
- Use as a guide, not a guarantee

### 4. **Requires Sufficient Data**
- Need enough historical data (at least 30+ days)
- More data = more reliable correlation

## Best Practices

### ✅ Do:
- Use correlation to assess diversification
- Compare correlations across different time periods
- Look for correlation breakdowns (trading opportunities)
- Consider sector correlations when building portfolios

### ❌ Don't:
- Rely solely on correlation for trading decisions
- Assume correlation will stay constant
- Ignore other factors (fundamentals, news, etc.)
- Use correlation as the only risk metric

## Example Workflow

1. **Analyze Your Current Portfolio**:
   ```
   Symbols: AAPL, MSFT, GOOGL, AMZN, META
   Result: Average correlation 0.75
   → Too high! All tech stocks
   ```

2. **Add Diversification**:
   ```
   Add: XOM (Energy), JNJ (Healthcare)
   New average: 0.35
   → Much better diversification!
   ```

3. **Monitor Over Time**:
   - Recalculate monthly
   - Watch for correlation changes
   - Adjust portfolio as needed

## Summary

**Correlation Analysis** helps you:
- ✅ Build diversified portfolios
- ✅ Understand risk relationships
- ✅ Find trading opportunities
- ✅ Assess portfolio concentration

**Key Takeaway**: 
- **Low correlation** = Better diversification = Lower risk
- **High correlation** = Poor diversification = Higher risk

Use correlation analysis to build a portfolio where stocks don't all move together! 📊

