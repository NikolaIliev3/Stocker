
class MockAnalyzer:
    def __init__(self):
        self.rsi_oversold = 30
        self.rsi_overbought = 70

    def old_logic(self, rsi, trend):
        score = 0
        reasons = []
        
        # Scoring (Same in both)
        if rsi < self.rsi_oversold:
            score += 2 # Bullish signal
        elif rsi > self.rsi_overbought:
            score -= 2 # Bearish signal
            
        if trend == "uptrend":
            score += 1
        elif trend == "downtrend":
            score -= 1
            
        # Old Decision Logic (Inverted)
        action = "HOLD"
        # Positive Score -> SELL
        if score >= 3:
            action = "SELL" 
        # Negative Score -> BUY
        elif score <= -3:
            action = "BUY"
            
        return score, action

    def new_logic(self, rsi, trend):
        score = 0
        
        # Scoring (Same in both)
        if rsi < self.rsi_oversold:
            score += 2 # Bullish signal
        elif rsi > self.rsi_overbought:
            score -= 2 # Bearish signal
            
        if trend == "uptrend":
            score += 1
        elif trend == "downtrend":
            score -= 1

        # New Decision Logic (Standard)
        action = "HOLD"
        # Positive Score -> BUY
        if score >= 2:
            action = "BUY"
        # Negative Score -> SELL
        elif score <= -2:
            action = "SELL"
            
        return score, action

analyzer = MockAnalyzer()

scenarios = [
    {"name": "Buy Cheap (Oversold + Uptrend)", "rsi": 25, "trend": "uptrend"},
    {"name": "Buy Dip (Oversold + Downtrend)", "rsi": 25, "trend": "downtrend"},
    {"name": "Sell High (Overbought + Downtrend)", "rsi": 75, "trend": "downtrend"},
    {"name": "Sell Rally (Overbought + Uptrend)", "rsi": 75, "trend": "uptrend"}
]

print(f"{'Scenario':<35} | {'Score':<5} | {'OLD Logic':<10} | {'NEW Logic':<10}")
print("-" * 75)

for s in scenarios:
    score, old = analyzer.old_logic(s['rsi'], s['trend'])
    _, new = analyzer.new_logic(s['rsi'], s['trend'])
    print(f"{s['name']:<35} | {score:<5} | {old:<10} | {new:<10}")
