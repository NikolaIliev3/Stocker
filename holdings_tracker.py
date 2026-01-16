"""
Holdings Tracker Module
Tracks user's stock holdings and monitors them for sell signals
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Holding:
    """Represents a stock holding"""
    symbol: str
    buy_price: float
    quantity: float
    buy_date: str  # ISO format
    notes: str = ""
    id: int = 0  # Unique identifier
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Holding':
        return cls(**data)


class HoldingsTracker:
    """Manages user's stock holdings and monitors for sell signals"""
    
    def __init__(self, data_dir: Path, app=None):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.holdings_file = data_dir / "holdings.json"
        self.app = app
        self.holdings: List[Holding] = []
        self.next_id = 1
        self._load_holdings()
        
    def _load_holdings(self):
        """Load holdings from file"""
        try:
            if self.holdings_file.exists():
                with open(self.holdings_file, 'r') as f:
                    data = json.load(f)
                    self.holdings = [Holding.from_dict(h) for h in data.get('holdings', [])]
                    self.next_id = data.get('next_id', 1)
                    logger.info(f"Loaded {len(self.holdings)} holdings")
        except Exception as e:
            logger.error(f"Error loading holdings: {e}")
            self.holdings = []
            self.next_id = 1
    
    def _save_holdings(self):
        """Save holdings to file"""
        try:
            data = {
                'holdings': [h.to_dict() for h in self.holdings],
                'next_id': self.next_id
            }
            with open(self.holdings_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(self.holdings)} holdings")
        except Exception as e:
            logger.error(f"Error saving holdings: {e}")
    
    def add_holding(self, symbol: str, buy_price: float, quantity: float, 
                    buy_date: str = None, notes: str = "") -> Holding:
        """Add a new holding"""
        if buy_date is None:
            buy_date = datetime.now().strftime('%Y-%m-%d')
        
        holding = Holding(
            symbol=symbol.upper(),
            buy_price=buy_price,
            quantity=quantity,
            buy_date=buy_date,
            notes=notes,
            id=self.next_id
        )
        self.next_id += 1
        self.holdings.append(holding)
        self._save_holdings()
        logger.info(f"Added holding: {symbol} @ ${buy_price:.2f} x {quantity}")
        return holding
    
    def remove_holding(self, holding_id: int) -> bool:
        """Remove a holding by ID"""
        for i, h in enumerate(self.holdings):
            if h.id == holding_id:
                removed = self.holdings.pop(i)
                self._save_holdings()
                logger.info(f"Removed holding: {removed.symbol}")
                return True
        return False
    
    def get_holding(self, holding_id: int) -> Optional[Holding]:
        """Get a holding by ID"""
        for h in self.holdings:
            if h.id == holding_id:
                return h
        return None
    
    def get_all_holdings(self) -> List[Holding]:
        """Get all holdings"""
        return self.holdings.copy()
    
    def analyze_holding(self, holding: Holding) -> Dict:
        """
        Analyze a holding and determine if it should be sold.
        Returns analysis with current price, P/L, and sell signals.
        """
        result = {
            'holding': holding,
            'current_price': 0.0,
            'profit_loss': 0.0,
            'profit_loss_pct': 0.0,
            'total_value': 0.0,
            'total_cost': holding.buy_price * holding.quantity,
            'status': 'UNKNOWN',
            'sell_signals': [],
            'confidence': 0,
            'recommendation': 'HOLD',
            'reasoning': []
        }
        
        try:
            if not self.app:
                result['status'] = 'NO_APP'
                return result
            
            # Fetch current price
            stock_data = self.app.data_fetcher.fetch_stock_data(holding.symbol)
            if not stock_data or 'error' in stock_data:
                result['status'] = 'FETCH_ERROR'
                return result
            
            current_price = stock_data.get('price', 0)
            if current_price <= 0:
                result['status'] = 'NO_PRICE'
                return result
            
            # Calculate P/L
            result['current_price'] = current_price
            result['total_value'] = current_price * holding.quantity
            result['profit_loss'] = result['total_value'] - result['total_cost']
            result['profit_loss_pct'] = ((current_price - holding.buy_price) / holding.buy_price) * 100
            
            # Fetch history for analysis
            history = self.app.data_fetcher.fetch_stock_history(holding.symbol, period='3mo')
            if not history or 'error' in history:
                result['status'] = 'HOLD'
                result['recommendation'] = 'HOLD'
                result['reasoning'].append('Unable to fetch history for analysis')
                return result
            
            # Get hybrid predictor analysis
            strategy = self.app.strategy_var.get() if hasattr(self.app, 'strategy_var') else 'trading'
            predictor = self.app.hybrid_predictors.get(strategy)
            
            if predictor:
                prediction = predictor.predict(stock_data, {'data': history.get('data', [])})
                result['confidence'] = prediction.get('confidence', 0)
                
                if prediction.get('action') == 'SELL':
                    result['sell_signals'].append(f"ML predicts SELL ({prediction.get('confidence', 0):.0f}% confidence)")
            
            # Check technical indicators
            try:
                from trading_analyzer import TradingAnalyzer
                analyzer = TradingAnalyzer()
                analysis = analyzer.analyze(stock_data, {'data': history.get('data', [])})
                
                if 'error' not in analysis:
                    indicators = analysis.get('indicators', {})
                    
                    # RSI check
                    rsi = indicators.get('rsi', 50)
                    if rsi > 70:
                        result['sell_signals'].append(f"RSI overbought: {rsi:.1f}")
                    elif rsi > 65:
                        result['reasoning'].append(f"RSI elevated: {rsi:.1f}")
                    
                    # MACD check
                    macd_diff = indicators.get('macd_diff', 0)
                    if macd_diff < 0:
                        result['sell_signals'].append("MACD bearish crossover")
                    
                    # Price vs EMA
                    ema_20 = indicators.get('ema_20', 0)
                    if ema_20 > 0 and current_price > ema_20 * 1.1:
                        result['sell_signals'].append(f"Price extended >10% above EMA20")
                    
            except Exception as e:
                logger.debug(f"Error in technical analysis: {e}")
            
            # Check momentum monitor for peak detection
            if hasattr(self.app, 'momentum_monitor') and self.app.momentum_monitor:
                try:
                    peak_info = self.app.momentum_monitor.detect_peak(holding.symbol, history.get('data', []))
                    if peak_info and peak_info.get('is_peak', False):
                        result['sell_signals'].append(f"Peak detected (confidence: {peak_info.get('confidence', 0):.0f}%)")
                except Exception as e:
                    logger.debug(f"Error in peak detection: {e}")
            
            # Determine final recommendation
            num_signals = len(result['sell_signals'])
            
            if num_signals >= 3:
                result['status'] = 'SELL'
                result['recommendation'] = 'STRONG SELL'
            elif num_signals >= 2:
                result['status'] = 'SELL'
                result['recommendation'] = 'SELL'
            elif num_signals >= 1:
                result['status'] = 'WATCH'
                result['recommendation'] = 'CONSIDER SELLING'
            else:
                result['status'] = 'HOLD'
                result['recommendation'] = 'HOLD'
                if result['profit_loss_pct'] > 20:
                    result['reasoning'].append('Good profit, but no sell signals yet')
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing holding {holding.symbol}: {e}")
            result['status'] = 'ERROR'
            result['reasoning'].append(f'Analysis error: {str(e)}')
            return result
    
    def analyze_all_holdings(self) -> List[Dict]:
        """Analyze all holdings and return results"""
        results = []
        for holding in self.holdings:
            analysis = self.analyze_holding(holding)
            results.append(analysis)
        return results
    
    def get_sell_alerts(self) -> List[Dict]:
        """Get holdings that have sell signals"""
        alerts = []
        for holding in self.holdings:
            analysis = self.analyze_holding(holding)
            if analysis['status'] in ['SELL', 'WATCH']:
                alerts.append(analysis)
        return alerts
