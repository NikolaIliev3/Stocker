"""
Walk-Forward Backtesting Module
Implements walk-forward analysis to prevent overfitting and validate strategy robustness
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class WalkForwardBacktester:
    """
    Walk-forward backtesting system
    Trains on historical data, tests on forward data, then moves window forward
    """
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_file = data_dir / "walk_forward_results.json"
        self.results = self._load_results()
    
    def run_walk_forward_test(self, history_data: dict, 
                              predictions: Optional[List[Dict]] = None,
                              predictor=None,
                              symbol: str = None,
                              strategy: str = 'trading',
                              train_window_days: int = 252,
                              test_window_days: int = 63,
                              step_size_days: int = 21,
                              lookforward_days: int = 10) -> Dict:
        """
        Run walk-forward backtest
        
        Args:
            history_data: Full historical data
            predictions: Optional list of predictions made during test period
            predictor: Optional predictor to generate predictions for each window
            symbol: Stock symbol (required if using predictor)
            strategy: Strategy type (required if using predictor)
            train_window_days: Days to use for training
            test_window_days: Days to use for testing
            step_size_days: Days to step forward each iteration
            lookforward_days: Days to look forward for prediction validation
            
        Returns:
            Dict with walk-forward results
        """
        if not history_data or 'data' not in history_data:
            return {"error": "No history data"}
        
        df = pd.DataFrame(history_data['data'])
        if df.empty:
            return {"error": "Empty history data"}
        
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        # Calculate number of windows
        total_days = len(df)
        windows = []
        
        start_idx = 0
        while start_idx + train_window_days + test_window_days <= total_days:
            train_end = start_idx + train_window_days
            test_end = train_end + test_window_days
            
            windows.append({
                'train_start': start_idx,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end,
                'train_dates': (df.iloc[start_idx]['date'], df.iloc[train_end]['date']),
                'test_dates': (df.iloc[train_end]['date'], df.iloc[test_end]['date'])
            })
            
            start_idx += step_size_days
        
        if not windows:
            return {"error": "Insufficient data for walk-forward test. Need at least {} days of data.".format(train_window_days + test_window_days)}
        
        # Run backtest for each window
        window_results = []
        for i, window in enumerate(windows):
            # Generate predictions for this window if predictor is provided
            window_predictions = predictions if predictions else []
            if predictor and symbol:
                window_predictions = self._generate_predictions_for_window(
                    df, window, predictor, symbol, strategy, lookforward_days
                )
            
            window_result = self._test_window(df, window, window_predictions, i, lookforward_days)
            window_results.append(window_result)
        
        # Aggregate results
        aggregated = self._aggregate_results(window_results)
        
        # Save results
        self.results['walk_forward_tests'] = window_results
        self.results['aggregated'] = aggregated
        self.results['last_test_date'] = datetime.now().isoformat()
        self._save_results()
        
        return {
            'windows_tested': len(windows),
            'window_results': window_results,
            'aggregated': aggregated
        }
    
    def _generate_predictions_for_window(self, df: pd.DataFrame, window: Dict,
                                         predictor, symbol: str, strategy: str,
                                         lookforward_days: int) -> List[Dict]:
        """Generate predictions for a test window using the predictor"""
        predictions = []
        test_data = df.iloc[window['test_start']:window['test_end']]
        
        # Sample test dates (every 5 days to avoid too many predictions)
        test_dates = test_data['date'].iloc[::5].tolist()
        
        for test_date in test_dates:
            try:
                # Get data up to test_date (training window)
                train_data = df[df['date'] <= test_date]
                if len(train_data) < 100:  # Need minimum data
                    continue
                
                # Get current stock data at test_date
                test_row = test_data[test_data['date'] == test_date]
                if test_row.empty:
                    continue
                
                current_price = float(test_row.iloc[0]['close'])
                stock_data = {
                    'symbol': symbol,
                    'price': current_price,
                    'change': float(test_row.iloc[0].get('change', 0)),
                    'change_percent': float(test_row.iloc[0].get('change_percent', 0))
                }
                
                # Convert to history format
                history_dict = {
                    'data': train_data.to_dict('records')
                }
                
                # Generate prediction
                prediction = predictor.predict(stock_data, history_dict, {})
                
                if prediction and 'action' in prediction:
                    predictions.append({
                        'date': test_date.strftime('%Y-%m-%d'),
                        'action': prediction.get('action', 'HOLD'),
                        'confidence': prediction.get('confidence', 0),
                        'lookforward_days': lookforward_days,
                        'entry_price': current_price
                    })
            except Exception as e:
                logger.debug(f"Error generating prediction for {test_date}: {e}")
                continue
        
        return predictions
    
    def _test_window(self, df: pd.DataFrame, window: Dict, 
                    predictions: List[Dict], window_num: int, lookforward_days: int = 10) -> Dict:
        """Test a single window"""
        train_data = df.iloc[window['train_start']:window['train_end']]
        test_data = df.iloc[window['test_start']:window['test_end']]
        
        # Filter predictions to this window if they have date strings
        window_predictions = []
        for p in predictions:
            if 'date' in p:
                try:
                    pred_date = pd.to_datetime(p['date'])
                    if window['test_dates'][0] <= pred_date <= window['test_dates'][1]:
                        window_predictions.append(p)
                except:
                    pass
            else:
                # If no date, assume it's for this window
                window_predictions.append(p)
        
        # If no predictions provided, generate test points from test data
        if not window_predictions:
            # Sample every 5 days in test period
            test_dates = test_data['date'].iloc[::5].tolist()
            for test_date in test_dates:
                test_row = test_data[test_data['date'] == test_date]
                if not test_row.empty:
                    window_predictions.append({
                        'date': test_date.strftime('%Y-%m-%d'),
                        'action': 'HOLD',  # Default action
                        'lookforward_days': lookforward_days,
                        'entry_price': float(test_row.iloc[0]['close'])
                    })
        
        # Calculate metrics
        correct = 0
        incorrect = 0
        total_return = 0
        valid_predictions = 0
        
        for pred in window_predictions:
            try:
                pred_date_str = pred.get('date', '')
                if not pred_date_str:
                    continue
                
                pred_date = pd.to_datetime(pred_date_str)
                
                # Find matching row in test data
                matching_rows = test_data[test_data['date'] == pred_date]
                if matching_rows.empty:
                    # Try to find closest date
                    date_diffs = (test_data['date'] - pred_date).abs()
                    closest_idx = date_diffs.idxmin()
                    if date_diffs.loc[closest_idx] > pd.Timedelta(days=5):
                        continue  # Too far from test date
                    pred_row_idx = closest_idx
                else:
                    pred_row_idx = matching_rows.index[0]
                
                # Get entry price
                entry_price = pred.get('entry_price')
                if not entry_price:
                    entry_price = float(test_data.loc[pred_row_idx, 'close'])
                
                # Get future price
                lookforward = pred.get('lookforward_days', lookforward_days)
                future_idx = test_data.index.get_loc(pred_row_idx) + lookforward
                if future_idx >= len(test_data):
                    continue  # Not enough future data
                
                future_price = float(test_data.iloc[future_idx]['close'])
                
                # Check if prediction was correct
                action = pred.get('action', 'HOLD')
                price_change = (future_price - entry_price) / entry_price
                
                is_correct = False
                if action == 'BUY' and price_change > 0.02:  # 2% threshold
                    is_correct = True
                elif action == 'SELL' and price_change < -0.02:
                    is_correct = True
                elif action == 'HOLD' and -0.02 <= price_change <= 0.02:
                    is_correct = True
                
                if is_correct:
                    correct += 1
                else:
                    incorrect += 1
                
                total_return += price_change
                valid_predictions += 1
            except Exception as e:
                logger.debug(f"Error testing prediction: {e}")
                continue
        
        accuracy = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0
        
        return {
            'window_num': window_num,
            'train_period': (window['train_dates'][0].strftime('%Y-%m-%d'), 
                           window['train_dates'][1].strftime('%Y-%m-%d')),
            'test_period': (window['test_dates'][0].strftime('%Y-%m-%d'), 
                          window['test_dates'][1].strftime('%Y-%m-%d')),
            'predictions_tested': valid_predictions,
            'correct': correct,
            'incorrect': incorrect,
            'accuracy': accuracy,
            'total_return': total_return,
            'avg_return': total_return / valid_predictions if valid_predictions > 0 else 0
        }
    
    def _aggregate_results(self, window_results: List[Dict]) -> Dict:
        """Aggregate results across all windows"""
        if not window_results:
            return {}
        
        accuracies = [r['accuracy'] for r in window_results if 'accuracy' in r]
        returns = [r['avg_return'] for r in window_results if 'avg_return' in r]
        
        total_correct = sum(r.get('correct', 0) for r in window_results)
        total_incorrect = sum(r.get('incorrect', 0) for r in window_results)
        
        overall_accuracy = total_correct / (total_correct + total_incorrect) if (total_correct + total_incorrect) > 0 else 0
        mean_accuracy = np.mean(accuracies) if accuracies else 0
        std_accuracy = np.std(accuracies) if accuracies else 0
        min_accuracy = min(accuracies) if accuracies else 0
        max_accuracy = max(accuracies) if accuracies else 0
        mean_return = np.mean(returns) if returns else 0
        std_return = np.std(returns) if returns else 0
        sharpe_ratio = np.mean(returns) / np.std(returns) if returns and np.std(returns) > 0 else 0
        
        # Analyze retraining recommendation
        retrain_recommendation = self._analyze_retrain_recommendation(
            overall_accuracy, mean_accuracy, std_accuracy, min_accuracy, 
            max_accuracy, sharpe_ratio, window_results
        )
        
        return {
            'windows': len(window_results),
            'overall_accuracy': overall_accuracy,
            'mean_accuracy': mean_accuracy,
            'std_accuracy': std_accuracy,
            'min_accuracy': min_accuracy,
            'max_accuracy': max_accuracy,
            'mean_return': mean_return,
            'std_return': std_return,
            'sharpe_ratio': sharpe_ratio,
            'retrain_recommendation': retrain_recommendation
        }
    
    def _analyze_retrain_recommendation(self, overall_accuracy: float, mean_accuracy: float,
                                       std_accuracy: float, min_accuracy: float,
                                       max_accuracy: float, sharpe_ratio: float,
                                       window_results: List[Dict]) -> Dict:
        """
        Analyze walk-forward results and provide retraining recommendation
        
        Returns:
            Dict with 'recommendation' ('retrain_now', 'consider_retrain', 'no_retrain'),
            'priority' ('high', 'medium', 'low', 'none'),
            'reasons' (list of reasons), and 'confidence' (0-100)
        """
        reasons = []
        priority = 'none'
        recommendation = 'no_retrain'
        confidence = 0
        
        # Check overall accuracy thresholds
        if overall_accuracy < 0.40:
            recommendation = 'retrain_now'
            priority = 'high'
            reasons.append(f"Overall accuracy ({overall_accuracy*100:.1f}%) is below 40% - critical threshold")
            confidence += 30
        elif overall_accuracy < 0.50:
            recommendation = 'consider_retrain'
            priority = 'medium'
            reasons.append(f"Overall accuracy ({overall_accuracy*100:.1f}%) is below 50% - below acceptable")
            confidence += 20
        
        # Check minimum accuracy
        if min_accuracy < 0.30:
            if recommendation != 'retrain_now':
                recommendation = 'retrain_now'
                priority = 'high'
            reasons.append(f"Minimum accuracy ({min_accuracy*100:.1f}%) is very low - some windows performed poorly")
            confidence += 25
        elif min_accuracy < 0.40:
            if recommendation == 'no_retrain':
                recommendation = 'consider_retrain'
                priority = 'medium'
            reasons.append(f"Minimum accuracy ({min_accuracy*100:.1f}%) is below 40% - inconsistent performance")
            confidence += 15
        
        # Check variance (consistency)
        if std_accuracy > 0.20:
            if recommendation == 'no_retrain':
                recommendation = 'consider_retrain'
                priority = 'medium'
            reasons.append(f"High variance (std: {std_accuracy*100:.1f}%) - inconsistent across windows")
            confidence += 15
        elif std_accuracy > 0.15:
            if recommendation == 'no_retrain':
                recommendation = 'consider_retrain'
                priority = 'low'
            reasons.append(f"Moderate variance (std: {std_accuracy*100:.1f}%) - some inconsistency")
            confidence += 10
        
        # Check accuracy spread
        accuracy_spread = max_accuracy - min_accuracy
        if accuracy_spread > 0.40:
            if recommendation != 'retrain_now':
                recommendation = 'consider_retrain'
                priority = 'medium'
            reasons.append(f"Large accuracy spread ({accuracy_spread*100:.1f}%) - high variability")
            confidence += 15
        
        # Check Sharpe ratio
        if sharpe_ratio < 0:
            if recommendation != 'retrain_now':
                recommendation = 'retrain_now'
                priority = 'high'
            reasons.append(f"Negative Sharpe ratio ({sharpe_ratio:.2f}) - poor risk-adjusted returns")
            confidence += 20
        elif sharpe_ratio < 0.5:
            if recommendation == 'no_retrain':
                recommendation = 'consider_retrain'
                priority = 'low'
            reasons.append(f"Low Sharpe ratio ({sharpe_ratio:.2f}) - suboptimal risk-adjusted returns")
            confidence += 10
        
        # Check for declining trend
        if len(window_results) >= 3:
            # Compare first half vs second half
            mid_point = len(window_results) // 2
            first_half_acc = np.mean([r['accuracy'] for r in window_results[:mid_point]])
            second_half_acc = np.mean([r['accuracy'] for r in window_results[mid_point:]])
            
            if second_half_acc < first_half_acc - 0.10:  # 10% drop
                if recommendation != 'retrain_now':
                    recommendation = 'retrain_now'
                    priority = 'high'
                reasons.append(f"Declining trend detected: {first_half_acc*100:.1f}% → {second_half_acc*100:.1f}% (model degrading)")
                confidence += 25
        
        # Positive indicators (reduce recommendation severity)
        positive_indicators = []
        if overall_accuracy >= 0.60:
            positive_indicators.append(f"Overall accuracy ({overall_accuracy*100:.1f}%) is good")
            confidence -= 15
        if min_accuracy >= 0.45:
            positive_indicators.append(f"Minimum accuracy ({min_accuracy*100:.1f}%) is acceptable")
            confidence -= 10
        if std_accuracy < 0.10:
            positive_indicators.append(f"Low variance (std: {std_accuracy*100:.1f}%) - consistent performance")
            confidence -= 10
        if sharpe_ratio >= 1.0:
            positive_indicators.append(f"Good Sharpe ratio ({sharpe_ratio:.2f}) - strong risk-adjusted returns")
            confidence -= 10
        
        # Adjust recommendation based on positive indicators
        if positive_indicators and recommendation == 'consider_retrain' and confidence < 30:
            recommendation = 'no_retrain'
            priority = 'low'
        
        # Clamp confidence
        confidence = max(0, min(100, confidence))
        
        # Determine final recommendation text
        if recommendation == 'retrain_now':
            recommendation_text = "RETRAIN NOW"
            recommendation_emoji = "🔴"
        elif recommendation == 'consider_retrain':
            recommendation_text = "CONSIDER RETRAINING"
            recommendation_emoji = "🟡"
        else:
            recommendation_text = "NO RETRAIN NEEDED"
            recommendation_emoji = "🟢"
        
        return {
            'recommendation': recommendation,
            'recommendation_text': recommendation_text,
            'recommendation_emoji': recommendation_emoji,
            'priority': priority,
            'reasons': reasons,
            'positive_indicators': positive_indicators,
            'confidence': confidence
        }
    
    def _load_results(self) -> Dict:
        """Load previous results"""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading walk-forward results: {e}")
        
        return {
            'walk_forward_tests': [],
            'aggregated': {},
            'last_test_date': None
        }
    
    def run_multi_stock_walk_forward_test(self, symbols: List[str],
                                         predictor,
                                         strategy: str,
                                         data_fetcher,
                                         train_window_days: int = 252,
                                         test_window_days: int = 63,
                                         step_size_days: int = 21,
                                         lookforward_days: int = 10,
                                         max_stocks: Optional[int] = None) -> Dict:
        """
        Run walk-forward backtest across multiple stocks
        
        Args:
            symbols: List of stock symbols to test
            predictor: Predictor instance to use
            strategy: Strategy type
            data_fetcher: DataFetcher instance to get historical data
            train_window_days: Days for training window
            test_window_days: Days for test window
            step_size_days: Days to step forward
            lookforward_days: Days to look forward for validation
            max_stocks: Maximum number of stocks to test (None = all)
        
        Returns:
            Dict with aggregate and per-stock results
        """
        if max_stocks:
            symbols = symbols[:max_stocks]
        
        all_stock_results = []
        per_stock_aggregates = []
        
        logger.info(f"Starting multi-stock walk-forward test on {len(symbols)} stocks")
        
        for idx, symbol in enumerate(symbols, 1):
            try:
                logger.info(f"Testing stock {idx}/{len(symbols)}: {symbol}")
                
                # Fetch historical data
                period_days = train_window_days + test_window_days + 100
                if period_days <= 365:
                    period = '1y'
                elif period_days <= 730:
                    period = '2y'
                else:
                    period = '5y'
                
                history_data = data_fetcher.fetch_stock_history(symbol, period=period)
                if not history_data or 'data' not in history_data:
                    logger.warning(f"Skipping {symbol}: insufficient data")
                    continue
                
                if len(history_data['data']) < train_window_days + test_window_days:
                    logger.warning(f"Skipping {symbol}: need {train_window_days + test_window_days} days, got {len(history_data['data'])}")
                    continue
                
                # Run walk-forward for this stock
                result = self.run_walk_forward_test(
                    history_data=history_data,
                    predictor=predictor,
                    symbol=symbol,
                    strategy=strategy,
                    train_window_days=train_window_days,
                    test_window_days=test_window_days,
                    step_size_days=step_size_days,
                    lookforward_days=lookforward_days
                )
                
                if 'error' not in result and result.get('aggregated'):
                    all_stock_results.append({
                        'symbol': symbol,
                        'result': result
                    })
                    per_stock_aggregates.append({
                        'symbol': symbol,
                        **result['aggregated']
                    })
                    logger.info(f"✓ {symbol}: {result['aggregated'].get('overall_accuracy', 0)*100:.1f}% accuracy")
                else:
                    logger.warning(f"✗ {symbol}: {result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Error testing {symbol}: {e}", exc_info=True)
                continue
        
        if not per_stock_aggregates:
            return {
                'error': 'No valid stock results obtained',
                'stocks_tested': 0
            }
        
        # Aggregate across all stocks
        aggregate_recommendation = self._aggregate_multi_stock_results(per_stock_aggregates)
        
        # Save multi-stock results
        self.results['multi_stock_tests'] = {
            'last_test_date': datetime.now().isoformat(),
            'symbols_tested': [r['symbol'] for r in all_stock_results],
            'aggregate_results': aggregate_recommendation
        }
        self._save_results()
        
        return {
            'stocks_tested': len(all_stock_results),
            'symbols': [r['symbol'] for r in all_stock_results],
            'per_stock_results': all_stock_results,
            'aggregate_results': aggregate_recommendation
        }
    
    def _aggregate_multi_stock_results(self, per_stock_aggregates: List[Dict]) -> Dict:
        """Aggregate results across multiple stocks"""
        if not per_stock_aggregates:
            return {}
        
        # Collect all accuracies and returns
        all_accuracies = []
        all_returns = []
        all_sharpe_ratios = []
        
        for stock_result in per_stock_aggregates:
            all_accuracies.append(stock_result.get('overall_accuracy', 0))
            all_returns.append(stock_result.get('mean_return', 0))
            all_sharpe_ratios.append(stock_result.get('sharpe_ratio', 0))
        
        # Calculate aggregate metrics
        aggregate_accuracy = np.mean(all_accuracies)
        aggregate_std = np.std(all_accuracies)
        aggregate_min = min(all_accuracies)
        aggregate_max = max(all_accuracies)
        aggregate_return = np.mean(all_returns)
        aggregate_sharpe = np.mean(all_sharpe_ratios)
        
        # Count retraining recommendations
        retrain_now_count = sum(1 for r in per_stock_aggregates 
                               if r.get('retrain_recommendation', {}).get('recommendation') == 'retrain_now')
        consider_retrain_count = sum(1 for r in per_stock_aggregates 
                                    if r.get('retrain_recommendation', {}).get('recommendation') == 'consider_retrain')
        no_retrain_count = sum(1 for r in per_stock_aggregates 
                               if r.get('retrain_recommendation', {}).get('recommendation') == 'no_retrain')
        
        # Generate aggregate recommendation based on aggregate metrics
        aggregate_recommendation = self._analyze_retrain_recommendation(
            aggregate_accuracy, aggregate_accuracy, aggregate_std,
            aggregate_min, aggregate_max, aggregate_sharpe, []
        )
        
        # Adjust recommendation based on per-stock results
        total_stocks = len(per_stock_aggregates)
        retrain_pct = retrain_now_count / total_stocks if total_stocks > 0 else 0
        attention_pct = (retrain_now_count + consider_retrain_count) / total_stocks if total_stocks > 0 else 0
        
        # Add multi-stock specific reasons
        if retrain_now_count > 0:
            aggregate_recommendation['reasons'].append(
                f"{retrain_now_count}/{total_stocks} stocks ({retrain_pct*100:.0f}%) "
                f"showed critical performance issues (RETRAIN NOW)"
            )
        
        if consider_retrain_count > 0:
            aggregate_recommendation['reasons'].append(
                f"{consider_retrain_count}/{total_stocks} stocks ({consider_retrain_count/total_stocks*100:.0f}%) "
                f"need attention (CONSIDER RETRAINING)"
            )
        
        # Override recommendation if significant portion of stocks need retraining
        if retrain_pct > 0.3:  # >30% need retraining
            aggregate_recommendation['recommendation'] = 'retrain_now'
            aggregate_recommendation['recommendation_text'] = 'RETRAIN NOW'
            aggregate_recommendation['recommendation_emoji'] = '🔴'
            aggregate_recommendation['priority'] = 'high'
            aggregate_recommendation['confidence'] = min(100, aggregate_recommendation.get('confidence', 0) + 20)
        elif attention_pct > 0.5:  # >50% need attention
            if aggregate_recommendation['recommendation'] == 'no_retrain':
                aggregate_recommendation['recommendation'] = 'consider_retrain'
                aggregate_recommendation['recommendation_text'] = 'CONSIDER RETRAINING'
                aggregate_recommendation['recommendation_emoji'] = '🟡'
                aggregate_recommendation['priority'] = 'medium'
                aggregate_recommendation['confidence'] = min(100, aggregate_recommendation.get('confidence', 0) + 15)
        
        # Clamp confidence
        aggregate_recommendation['confidence'] = max(0, min(100, aggregate_recommendation.get('confidence', 0)))
        
        return {
            'stocks_tested': total_stocks,
            'aggregate_accuracy': aggregate_accuracy,
            'aggregate_std': aggregate_std,
            'aggregate_min': aggregate_min,
            'aggregate_max': aggregate_max,
            'aggregate_return': aggregate_return,
            'aggregate_sharpe': aggregate_sharpe,
            'retrain_now_count': retrain_now_count,
            'consider_retrain_count': consider_retrain_count,
            'no_retrain_count': no_retrain_count,
            'retrain_recommendation': aggregate_recommendation,
            'per_stock_summary': [
                {
                    'symbol': r['symbol'],
                    'accuracy': r.get('overall_accuracy', 0),
                    'recommendation': r.get('retrain_recommendation', {}).get('recommendation', 'unknown'),
                    'recommendation_text': r.get('retrain_recommendation', {}).get('recommendation_text', 'UNKNOWN')
                }
                for r in per_stock_aggregates
            ]
        }
    
    def _save_results(self):
        """Save results"""
        try:
            with open(self.results_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving walk-forward results: {e}")

