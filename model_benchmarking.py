"""
Model Performance Benchmarking
Compares model performance against baselines (random, buy-hold, S&P 500)
"""
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)


class ModelBenchmarker:
    """Benchmarks model performance against baselines"""
    
    def __init__(self, data_dir: Path, data_fetcher=None):
        self.data_dir = data_dir
        self.benchmark_file = data_dir / "model_benchmarks.json"
        self.benchmark_data = self._load_benchmark_data()
        self.data_fetcher = data_fetcher  # For fetching SPY data
        self._spy_cache = None
        self._spy_cache_time = None
    
    def _load_benchmark_data(self) -> Dict:
        """Load benchmark data from file"""
        if self.benchmark_file.exists():
            try:
                with open(self.benchmark_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading benchmark data: {e}")
        
        return {
            'model_performance': {},
            'baseline_performance': {},
            'comparisons': {},
            'last_updated': None
        }
    
    def _save_benchmark_data(self):
        """Save benchmark data to file"""
        try:
            self.benchmark_data['last_updated'] = datetime.now().isoformat()
            with open(self.benchmark_file, 'w') as f:
                json.dump(self.benchmark_data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving benchmark data: {e}")
    
    def calculate_baselines(self, predictions: List[Dict], outcomes: List[Dict], strategy: str = 'trading') -> Dict:
        """Calculate baseline performance metrics
        
        Args:
            predictions: List of prediction dicts
            outcomes: List of outcome dicts (was_correct, actual_price_change, etc.)
            strategy: Strategy name (used to check ML model configuration)
        
        Returns:
            Baseline performance dict
        """
        if not predictions or not outcomes or len(predictions) != len(outcomes):
            return {}
        
        # Baseline 1: Random prediction (50% for binary, 33% for 3-class)
        # Check ML model metadata to determine classification type
        is_binary = self._check_if_binary_classification(strategy)
        
        # If metadata check fails, fall back to prediction-based inference
        if is_binary is None:
            unique_actions = set(p.get('action') for p in predictions)
            # Count HOLD predictions - if HOLD is <20%, likely binary with confidence threshold
            hold_count = sum(1 for p in predictions if p.get('action') == 'HOLD')
            hold_pct = (hold_count / len(predictions) * 100) if predictions else 0
            
            # If only 2 actions (no HOLD), or HOLD is rare (<20%), assume binary
            # If 3 actions with HOLD >20%, check if BUY and SELL are roughly balanced
            if len(unique_actions) == 2 and 'HOLD' not in unique_actions:
                is_binary = True
            elif hold_pct < 20:
                # HOLD is rare, likely binary with confidence threshold
                is_binary = True
            elif 'BUY' in unique_actions and 'SELL' in unique_actions:
                # Check if BUY and SELL are roughly balanced (within 30% of each other)
                buy_count = sum(1 for p in predictions if p.get('action') == 'BUY')
                sell_count = sum(1 for p in predictions if p.get('action') == 'SELL')
                total_buy_sell = buy_count + sell_count
                if total_buy_sell > 0:
                    buy_pct = (buy_count / total_buy_sell * 100)
                    # If BUY and SELL are roughly balanced (40-60% range), likely binary
                    is_binary = (40 <= buy_pct <= 60)
                else:
                    is_binary = False
            else:
                is_binary = False
        else:
            is_binary = bool(is_binary)
        
        random_accuracy = 50.0 if is_binary else 33.3
        
        # Baseline 2: Always predict majority class
        action_counts = {}
        for pred in predictions:
            action = pred.get('action', 'HOLD')
            action_counts[action] = action_counts.get(action, 0) + 1
        
        majority_action = max(action_counts.items(), key=lambda x: x[1])[0] if action_counts else 'HOLD'
        majority_correct = sum(1 for i, pred in enumerate(predictions) 
                              if pred.get('action') == majority_action and outcomes[i].get('was_correct', False))
        majority_accuracy = (majority_correct / len(predictions) * 100) if predictions else 0
        
        # Baseline 3: Buy and Hold (calculate from actual price changes)
        price_changes = [o.get('actual_price_change', 0) for o in outcomes]
        buy_hold_return = np.mean(price_changes) if price_changes else 0
        buy_hold_accuracy = 50.0  # Buy-and-hold doesn't have accuracy, use neutral
        
        # Baseline 4: Always HOLD (neutral strategy)
        hold_correct = sum(1 for i, pred in enumerate(predictions)
                          if pred.get('action') == 'HOLD' and outcomes[i].get('was_correct', False))
        hold_total = sum(1 for p in predictions if p.get('action') == 'HOLD')
        hold_accuracy = (hold_correct / hold_total * 100) if hold_total > 0 else 0
        
        baselines = {
            'random': {
                'accuracy': random_accuracy,
                'description': 'Random prediction baseline'
            },
            'majority_class': {
                'accuracy': majority_accuracy,
                'action': majority_action,
                'description': f'Always predict {majority_action}'
            },
            'buy_hold': {
                'return': buy_hold_return,
                'accuracy': buy_hold_accuracy,
                'description': 'Buy and hold strategy'
            },
            'always_hold': {
                'accuracy': hold_accuracy,
                'description': 'Always HOLD strategy'
            }
        }
        
        self.benchmark_data['baseline_performance'] = baselines
        self._save_benchmark_data()
        
        return baselines
    
    def compare_to_baselines(self, model_accuracy: float, strategy: str = 'trading') -> Dict:
        """Compare model performance to baselines
        
        Args:
            model_accuracy: Model's accuracy percentage
            strategy: Strategy name
        
        Returns:
            Comparison dict
        """
        baselines = self.benchmark_data.get('baseline_performance', {})
        
        if not baselines:
            return {'error': 'No baseline data available'}
        
        comparisons = {}
        
        # Compare to random
        random_acc = baselines.get('random', {}).get('accuracy', 50)
        vs_random = model_accuracy - random_acc
        comparisons['vs_random'] = {
            'difference': vs_random,
            'better': vs_random > 0,
            'improvement_pct': (vs_random / random_acc * 100) if random_acc > 0 else 0
        }
        
        # Compare to majority class
        majority_acc = baselines.get('majority_class', {}).get('accuracy', 0)
        vs_majority = model_accuracy - majority_acc
        comparisons['vs_majority'] = {
            'difference': vs_majority,
            'better': vs_majority > 0,
            'improvement_pct': (vs_majority / majority_acc * 100) if majority_acc > 0 else 0
        }
        
        # Compare to always HOLD
        hold_acc = baselines.get('always_hold', {}).get('accuracy', 0)
        vs_hold = model_accuracy - hold_acc
        comparisons['vs_hold'] = {
            'difference': vs_hold,
            'better': vs_hold > 0,
            'improvement_pct': (vs_hold / hold_acc * 100) if hold_acc > 0 else 0
        }
        
        # Store model performance
        self.benchmark_data['model_performance'][strategy] = {
            'accuracy': model_accuracy,
            'timestamp': datetime.now().isoformat()
        }
        
        self.benchmark_data['comparisons'][strategy] = comparisons
        self._save_benchmark_data()
        
        return comparisons
    
    def get_benchmark_report(self, strategy: str = 'trading') -> Dict:
        """Get comprehensive benchmark report"""
        model_perf = self.benchmark_data.get('model_performance', {}).get(strategy, {})
        baselines = self.benchmark_data.get('baseline_performance', {})
        comparisons = self.benchmark_data.get('comparisons', {}).get(strategy, {})
        
        model_accuracy = model_perf.get('accuracy', 0)
        
        report = {
            'model_accuracy': model_accuracy,
            'baselines': baselines,
            'comparisons': comparisons,
            'summary': self._generate_summary(model_accuracy, baselines, comparisons)
        }
        
        return report
    
    def _generate_summary(self, model_accuracy: float, baselines: Dict, 
                         comparisons: Dict) -> List[str]:
        """Generate summary insights"""
        summary = []
        
        if not baselines or not comparisons:
            return ["Insufficient benchmark data"]
        
        # Compare to random
        vs_random = comparisons.get('vs_random', {})
        if vs_random.get('better', False):
            summary.append(f"✅ Outperforms random baseline by {vs_random.get('difference', 0):.1f}%")
        else:
            summary.append(f"❌ Underperforms random baseline by {abs(vs_random.get('difference', 0)):.1f}%")
        
        # Compare to majority class
        vs_majority = comparisons.get('vs_majority', {})
        if vs_majority.get('better', False):
            summary.append(f"✅ Outperforms majority class by {vs_majority.get('difference', 0):.1f}%")
        else:
            summary.append(f"⚠️ Underperforms majority class by {abs(vs_majority.get('difference', 0)):.1f}%")
        
        # Overall assessment
        if model_accuracy > 60:
            summary.append("🎯 Model performance: Excellent (>60%)")
        elif model_accuracy > 55:
            summary.append("✅ Model performance: Good (55-60%)")
        elif model_accuracy > 50:
            summary.append("⚠️ Model performance: Acceptable (50-55%)")
        else:
            summary.append("❌ Model performance: Poor (<50%)")
        
        return summary
    
    def _check_if_binary_classification(self, strategy: str) -> Optional[bool]:
        """Check ML model metadata to determine if it uses binary classification
        
        Args:
            strategy: Strategy name
            
        Returns:
            True if binary, False if 3-class, None if cannot determine
        """
        try:
            metadata_file = self.data_dir / f"ml_metadata_{strategy}.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    return metadata.get('use_binary_classification', False)
        except Exception as e:
            logger.debug(f"Could not check ML metadata for binary classification: {e}")
        
        return None
    
    def get_sp500_return(self, start_date: datetime, end_date: datetime = None) -> Dict:
        """Get S&P 500 (SPY ETF) return for a given period
        
        Args:
            start_date: Start date for return calculation
            end_date: End date (defaults to now)
            
        Returns:
            Dict with return percentage and other metrics
        """
        if end_date is None:
            end_date = datetime.now()
        
        try:
            # Check cache (cache for 1 hour)
            if (self._spy_cache and self._spy_cache_time and 
                (datetime.now() - self._spy_cache_time).total_seconds() < 3600):
                spy_data = self._spy_cache
            else:
                # Fetch SPY data if we have data_fetcher
                if self.data_fetcher:
                    start_str = start_date.strftime('%Y-%m-%d')
                    end_str = end_date.strftime('%Y-%m-%d')
                    
                    # Fetch historical data
                    history = self.data_fetcher.fetch_stock_history('SPY', start_date=start_str, end_date=end_str)
                    
                    if history and 'data' in history and history['data']:
                        spy_data = history['data']
                        self._spy_cache = spy_data
                        self._spy_cache_time = datetime.now()
                    else:
                        return {'error': 'Could not fetch SPY data'}
                else:
                    # Try direct yfinance as fallback
                    try:
                        import yfinance as yf
                        spy = yf.Ticker("SPY")
                        hist = spy.history(start=start_date, end=end_date)
                        if hist.empty:
                            return {'error': 'No SPY data available'}
                        
                        spy_data = []
                        for date, row in hist.iterrows():
                            spy_data.append({
                                'date': date.strftime('%Y-%m-%d'),
                                'close': float(row['Close']),
                                'open': float(row['Open']),
                                'high': float(row['High']),
                                'low': float(row['Low']),
                                'volume': int(row['Volume'])
                            })
                        
                        self._spy_cache = spy_data
                        self._spy_cache_time = datetime.now()
                    except Exception as e:
                        logger.error(f"Error fetching SPY data: {e}")
                        return {'error': f'Error fetching SPY data: {str(e)}'}
            
            if not spy_data:
                return {'error': 'No SPY data available'}
            
            # Convert to DataFrame for easier calculation
            df = pd.DataFrame(spy_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            if len(df) < 2:
                return {'error': 'Insufficient SPY data'}
            
            # Get start and end prices
            start_price = float(df.iloc[0]['close'])
            end_price = float(df.iloc[-1]['close'])
            
            # Calculate return
            return_pct = ((end_price - start_price) / start_price) * 100
            
            # Calculate annualized return if period is significant
            days = (end_date - start_date).days
            annualized_return = None
            if days > 0:
                years = days / 365.25
                if years > 0:
                    annualized_return = (((end_price / start_price) ** (1 / years)) - 1) * 100
            
            return {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'start_price': start_price,
                'end_price': end_price,
                'return_pct': return_pct,
                'annualized_return_pct': annualized_return,
                'days': days,
                'data_points': len(df)
            }
            
        except Exception as e:
            logger.error(f"Error calculating SPY return: {e}")
            return {'error': f'Error calculating SPY return: {str(e)}'}
    
    def compare_portfolio_to_sp500(self, portfolio_snapshots: List[Dict], 
                                   initial_balance: float) -> Dict:
        """Compare portfolio performance to S&P 500
        
        Args:
            portfolio_snapshots: List of portfolio snapshots (from Portfolio.get_portfolio_history())
            initial_balance: Initial portfolio balance
            
        Returns:
            Comparison dict with portfolio vs S&P 500 metrics
        """
        if not portfolio_snapshots or len(portfolio_snapshots) < 2:
            return {'error': 'Insufficient portfolio data for comparison'}
        
        try:
            # Get start and end dates
            start_date = datetime.fromisoformat(portfolio_snapshots[0]['timestamp'])
            end_date = datetime.fromisoformat(portfolio_snapshots[-1]['timestamp'])
            
            # Calculate portfolio return
            final_balance = portfolio_snapshots[-1]['balance']
            portfolio_return_pct = ((final_balance - initial_balance) / initial_balance) * 100
            
            # Calculate S&P 500 return
            spy_return = self.get_sp500_return(start_date, end_date)
            
            if 'error' in spy_return:
                return {
                    'error': spy_return['error'],
                    'portfolio_return_pct': portfolio_return_pct,
                    'comparison_available': False
                }
            
            spy_return_pct = spy_return['return_pct']
            
            # Calculate difference
            outperformance = portfolio_return_pct - spy_return_pct
            outperformance_pct = (outperformance / abs(spy_return_pct) * 100) if spy_return_pct != 0 else 0
            
            # Calculate annualized returns
            days = (end_date - start_date).days
            portfolio_annualized = None
            if days > 0:
                years = days / 365.25
                if years > 0:
                    portfolio_annualized = (((final_balance / initial_balance) ** (1 / years)) - 1) * 100
            
            comparison = {
                'period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'days': days
                },
                'portfolio': {
                    'initial_balance': initial_balance,
                    'final_balance': final_balance,
                    'return_pct': portfolio_return_pct,
                    'annualized_return_pct': portfolio_annualized
                },
                'sp500': {
                    'start_price': spy_return['start_price'],
                    'end_price': spy_return['end_price'],
                    'return_pct': spy_return_pct,
                    'annualized_return_pct': spy_return.get('annualized_return_pct')
                },
                'comparison': {
                    'outperformance_pct': outperformance,
                    'outperformance_relative_pct': outperformance_pct,
                    'portfolio_better': outperformance > 0,
                    'beat_market': outperformance > 0
                },
                'comparison_available': True
            }
            
            # Store in benchmark data
            if 'sp500_comparisons' not in self.benchmark_data:
                self.benchmark_data['sp500_comparisons'] = []
            
            self.benchmark_data['sp500_comparisons'].append({
                'timestamp': datetime.now().isoformat(),
                'comparison': comparison
            })
            
            # Keep only last 100 comparisons
            if len(self.benchmark_data['sp500_comparisons']) > 100:
                self.benchmark_data['sp500_comparisons'] = self.benchmark_data['sp500_comparisons'][-100:]
            
            self._save_benchmark_data()
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing portfolio to S&P 500: {e}")
            return {'error': f'Error comparing portfolio to S&P 500: {str(e)}'}
    
    def get_latest_sp500_comparison(self) -> Optional[Dict]:
        """Get the latest S&P 500 comparison"""
        comparisons = self.benchmark_data.get('sp500_comparisons', [])
        if comparisons:
            return comparisons[-1].get('comparison')
        return None
    
    def get_sp500_return(self, start_date: datetime, end_date: datetime = None) -> Dict:
        """Get S&P 500 (SPY ETF) return for a given period
        
        Args:
            start_date: Start date for return calculation
            end_date: End date (defaults to now)
            
        Returns:
            Dict with return percentage and other metrics
        """
        if end_date is None:
            end_date = datetime.now()
        
        try:
            # Check cache (cache for 1 hour)
            if (self._spy_cache and self._spy_cache_time and 
                (datetime.now() - self._spy_cache_time).total_seconds() < 3600):
                spy_data = self._spy_cache
            else:
                # Fetch SPY data if we have data_fetcher
                if self.data_fetcher:
                    start_str = start_date.strftime('%Y-%m-%d')
                    end_str = end_date.strftime('%Y-%m-%d')
                    
                    # Fetch historical data
                    history = self.data_fetcher.fetch_stock_history('SPY', start_date=start_str, end_date=end_str)
                    
                    if history and 'data' in history and history['data']:
                        spy_data = history['data']
                        self._spy_cache = spy_data
                        self._spy_cache_time = datetime.now()
                    else:
                        return {'error': 'Could not fetch SPY data'}
                else:
                    # Try direct yfinance as fallback
                    try:
                        import yfinance as yf
                        spy = yf.Ticker("SPY")
                        hist = spy.history(start=start_date, end=end_date)
                        if hist.empty:
                            return {'error': 'No SPY data available'}
                        
                        spy_data = []
                        for date, row in hist.iterrows():
                            spy_data.append({
                                'date': date.strftime('%Y-%m-%d'),
                                'close': float(row['Close']),
                                'open': float(row['Open']),
                                'high': float(row['High']),
                                'low': float(row['Low']),
                                'volume': int(row['Volume'])
                            })
                        
                        self._spy_cache = spy_data
                        self._spy_cache_time = datetime.now()
                    except Exception as e:
                        logger.error(f"Error fetching SPY data: {e}")
                        return {'error': f'Error fetching SPY data: {str(e)}'}
            
            if not spy_data:
                return {'error': 'No SPY data available'}
            
            # Convert to DataFrame for easier calculation
            df = pd.DataFrame(spy_data)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            if len(df) < 2:
                return {'error': 'Insufficient SPY data'}
            
            # Get start and end prices
            start_price = float(df.iloc[0]['close'])
            end_price = float(df.iloc[-1]['close'])
            
            # Calculate return
            return_pct = ((end_price - start_price) / start_price) * 100
            
            # Calculate annualized return if period is significant
            days = (end_date - start_date).days
            annualized_return = None
            if days > 0:
                years = days / 365.25
                if years > 0:
                    annualized_return = (((end_price / start_price) ** (1 / years)) - 1) * 100
            
            return {
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'start_price': start_price,
                'end_price': end_price,
                'return_pct': return_pct,
                'annualized_return_pct': annualized_return,
                'days': days,
                'data_points': len(df)
            }
            
        except Exception as e:
            logger.error(f"Error calculating SPY return: {e}")
            return {'error': f'Error calculating SPY return: {str(e)}'}
    
    def compare_portfolio_to_sp500(self, portfolio_snapshots: List[Dict], 
                                   initial_balance: float) -> Dict:
        """Compare portfolio performance to S&P 500
        
        Args:
            portfolio_snapshots: List of portfolio snapshots (from Portfolio.get_portfolio_history())
            initial_balance: Initial portfolio balance
            
        Returns:
            Comparison dict with portfolio vs S&P 500 metrics
        """
        if not portfolio_snapshots or len(portfolio_snapshots) < 2:
            return {'error': 'Insufficient portfolio data for comparison'}
        
        try:
            # Get start and end dates
            start_date = datetime.fromisoformat(portfolio_snapshots[0]['timestamp'])
            end_date = datetime.fromisoformat(portfolio_snapshots[-1]['timestamp'])
            
            # Calculate portfolio return
            final_balance = portfolio_snapshots[-1]['balance']
            portfolio_return_pct = ((final_balance - initial_balance) / initial_balance) * 100
            
            # Calculate S&P 500 return
            spy_return = self.get_sp500_return(start_date, end_date)
            
            if 'error' in spy_return:
                return {
                    'error': spy_return['error'],
                    'portfolio_return_pct': portfolio_return_pct,
                    'comparison_available': False
                }
            
            spy_return_pct = spy_return['return_pct']
            
            # Calculate difference
            outperformance = portfolio_return_pct - spy_return_pct
            outperformance_pct = (outperformance / abs(spy_return_pct) * 100) if spy_return_pct != 0 else 0
            
            # Calculate annualized returns
            days = (end_date - start_date).days
            portfolio_annualized = None
            if days > 0:
                years = days / 365.25
                if years > 0:
                    portfolio_annualized = (((final_balance / initial_balance) ** (1 / years)) - 1) * 100
            
            comparison = {
                'period': {
                    'start_date': start_date.isoformat(),
                    'end_date': end_date.isoformat(),
                    'days': days
                },
                'portfolio': {
                    'initial_balance': initial_balance,
                    'final_balance': final_balance,
                    'return_pct': portfolio_return_pct,
                    'annualized_return_pct': portfolio_annualized
                },
                'sp500': {
                    'start_price': spy_return['start_price'],
                    'end_price': spy_return['end_price'],
                    'return_pct': spy_return_pct,
                    'annualized_return_pct': spy_return.get('annualized_return_pct')
                },
                'comparison': {
                    'outperformance_pct': outperformance,
                    'outperformance_relative_pct': outperformance_pct,
                    'portfolio_better': outperformance > 0,
                    'beat_market': outperformance > 0
                },
                'comparison_available': True
            }
            
            # Store in benchmark data
            if 'sp500_comparisons' not in self.benchmark_data:
                self.benchmark_data['sp500_comparisons'] = []
            
            self.benchmark_data['sp500_comparisons'].append({
                'timestamp': datetime.now().isoformat(),
                'comparison': comparison
            })
            
            # Keep only last 100 comparisons
            if len(self.benchmark_data['sp500_comparisons']) > 100:
                self.benchmark_data['sp500_comparisons'] = self.benchmark_data['sp500_comparisons'][-100:]
            
            self._save_benchmark_data()
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing portfolio to S&P 500: {e}")
            return {'error': f'Error comparing portfolio to S&P 500: {str(e)}'}
    
    def get_latest_sp500_comparison(self) -> Optional[Dict]:
        """Get the latest S&P 500 comparison"""
        comparisons = self.benchmark_data.get('sp500_comparisons', [])
        if comparisons:
            return comparisons[-1].get('comparison')
        return None