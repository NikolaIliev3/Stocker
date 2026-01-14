"""
Test AI Predictor Accuracy
Compares prediction accuracy with and without AI to determine if SeekerAI improves results
"""
import logging
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from data_fetcher import StockDataFetcher
from hybrid_predictor import HybridStockPredictor
from trading_analyzer import TradingAnalyzer
from investing_analyzer import InvestingAnalyzer
from mixed_analyzer import MixedAnalyzer
from config import APP_DATA_DIR, AI_PREDICTOR_ENABLED

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class AIPredictorAccuracyTester:
    """Test if AI predictor improves accuracy"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.data_fetcher = StockDataFetcher()
        
        # Initialize analyzers
        self.trading_analyzer = TradingAnalyzer(data_fetcher=self.data_fetcher)
        self.investing_analyzer = InvestingAnalyzer()
        self.mixed_analyzer = MixedAnalyzer()
        
        # Results storage
        self.results_file = data_dir / "ai_accuracy_test_results.json"
        self.results = self._load_results()
    
    def _load_results(self) -> Dict:
        """Load previous test results"""
        if self.results_file.exists():
            try:
                with open(self.results_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load previous results: {e}")
        return {
            'tests': [],
            'summary': {}
        }
    
    def _save_results(self):
        """Save test results"""
        try:
            with open(self.results_file, 'w') as f:
                json.dump(self.results, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def test_accuracy_comparison(self, symbols: List[str], strategy: str = 'trading',
                                 lookforward_days: int = 10, test_points: int = 50) -> Dict:
        """
        Test accuracy with and without AI
        
        Args:
            symbols: List of stock symbols to test
            strategy: Strategy to use ('trading', 'mixed', 'investing')
            lookforward_days: Days to look forward for validation
            test_points: Number of historical points to test
        
        Returns:
            Dict with comparison results
        """
        logger.info(f"Starting AI accuracy test: {len(symbols)} symbols, {strategy} strategy")
        
        # Initialize predictors
        # With AI
        predictor_with_ai = HybridStockPredictor(
            self.data_dir, strategy,
            trading_analyzer=self.trading_analyzer if strategy == 'trading' else None,
            investing_analyzer=self.investing_analyzer if strategy == 'investing' else None,
            mixed_analyzer=self.mixed_analyzer if strategy == 'mixed' else None,
            seeker_ai=None  # Will use default SeekerAI if available
        )
        
        # Without AI (temporarily disable)
        original_ai_enabled = AI_PREDICTOR_ENABLED
        try:
            # Temporarily disable AI by creating predictor without SeekerAI
            # We'll modify the predictor to skip AI
            predictor_without_ai = HybridStockPredictor(
                self.data_dir, strategy,
                trading_analyzer=self.trading_analyzer if strategy == 'trading' else None,
                investing_analyzer=self.investing_analyzer if strategy == 'investing' else None,
                mixed_analyzer=self.mixed_analyzer if strategy == 'mixed' else None,
                seeker_ai=None
            )
            # Disable AI predictor
            predictor_without_ai.ai_predictor = None
            # Also disable LLM AI predictor explicitly
            predictor_without_ai.llm_ai_predictor = None
        except Exception as e:
            logger.error(f"Error creating predictor without AI: {e}")
            return {'error': str(e)}

        # FORCE DISABLE OPENAI FOR "WITH AI" PREDICTOR (Cost control)
        # We want to test the Hybrid Architecture logic, not burn API credits
        if predictor_with_ai.llm_ai_predictor:
            logger.info("ℹ️ Forcing LLM AI to use local fallback logic (Zero Cost Mode)")
            predictor_with_ai.llm_ai_predictor.use_llm = False

        
        # Test results
        results_with_ai = []
        results_without_ai = []
        
        # Test on multiple symbols and time points
        total_tested = 0
        for symbol in symbols:
            logger.info(f"Testing {symbol}...")
            
            try:
                # Get stock data (includes history)
                stock_data = self.data_fetcher.fetch_stock_data(symbol)
                if not stock_data or 'error' in stock_data:
                    logger.warning(f"Could not get stock data for {symbol}, skipping")
                    continue
                
                # Get historical data
                history_list = stock_data.get('history', [])
                if not history_list or len(history_list) < 100:
                    logger.warning(f"Insufficient historical data for {symbol} ({len(history_list) if history_list else 0} points), skipping")
                    continue
                
                # Sample test points throughout history
                df = pd.DataFrame(history_list)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date')
                
                # Select test points evenly spaced
                if len(df) < test_points:
                    test_indices = range(len(df))
                else:
                    test_indices = np.linspace(50, len(df) - lookforward_days - 1, 
                                             min(test_points, len(df) - lookforward_days - 50), 
                                             dtype=int)
                
                for idx in test_indices[:test_points]:  # Limit to test_points
                    if idx + lookforward_days >= len(df):
                        continue
                    
                    # Get historical data up to this point
                    historical_up_to_point = []
                    for i in range(idx + 1):
                        row = df.iloc[i]
                        historical_up_to_point.append({
                            'date': row['date'].isoformat() if hasattr(row['date'], 'isoformat') else str(row['date']),
                            'open': float(row.get('open', row.get('close', 0))),
                            'high': float(row.get('high', row.get('close', 0))),
                            'low': float(row.get('low', row.get('close', 0))),
                            'close': float(row.get('close', 0)),
                            'volume': int(row.get('volume', 0)) if 'volume' in row else 0
                        })
                    
                    # Get stock data at this point
                    point_data = df.iloc[idx]
                    stock_data_at_point = {
                        'symbol': symbol,
                        'price': float(point_data['close']),
                        'volume': int(point_data.get('volume', 0)) if 'volume' in point_data else 0,
                        'open': float(point_data.get('open', point_data['close'])),
                        'high': float(point_data.get('high', point_data['close'])),
                        'low': float(point_data.get('low', point_data['close'])),
                        'info': {}
                    }
                    
                    history_data_at_point = {'data': historical_up_to_point, 'history': historical_up_to_point}
                    
                    # Get financials if available (simplified - use current)
                    financials_data = None
                    try:
                        financials_data = self.data_fetcher.fetch_financials(symbol)
                    except:
                        pass
                    
                    # Make predictions
                    try:
                        # Ensure stock_data_at_point has symbol for AI predictor
                        stock_data_at_point['symbol'] = symbol
                        
                        # With AI
                        pred_with_ai = predictor_with_ai.predict(
                            stock_data_at_point, history_data_at_point, financials_data
                        )
                        
                        # Without AI
                        pred_without_ai = predictor_without_ai.predict(
                            stock_data_at_point, history_data_at_point, financials_data
                        )
                    except Exception as e:
                        logger.debug(f"Error making predictions at index {idx}: {e}")
                        continue
                    
                    # Get actual outcome
                    actual_price = float(df.iloc[idx + lookforward_days]['close'])
                    entry_price = float(point_data['close'])
                    
                    # Determine if prediction was correct
                    # For BUY: price should go up
                    # For SELL: price should go down
                    # For HOLD: price should stay relatively stable
                    
                    price_change_pct = ((actual_price - entry_price) / entry_price) * 100
                    
                    def is_correct(prediction, price_change):
                        action = prediction.get('recommendation', {}).get('action', 'HOLD')
                        if action == 'BUY':
                            return price_change > 2.0  # At least 2% gain
                        elif action == 'SELL':
                            return price_change < -2.0  # At least 2% drop
                        else:  # HOLD
                            return abs(price_change) < 3.0  # Within 3%
                    
                    # Check correctness
                    correct_with_ai = is_correct(pred_with_ai, price_change_pct)
                    correct_without_ai = is_correct(pred_without_ai, price_change_pct)
                    
                    # Store results
                    results_with_ai.append({
                        'symbol': symbol,
                        'date': point_data['date'].isoformat() if hasattr(point_data['date'], 'isoformat') else str(point_data['date']),
                        'action': pred_with_ai.get('recommendation', {}).get('action', 'HOLD'),
                        'confidence': pred_with_ai.get('recommendation', {}).get('confidence', 50),
                        'entry_price': entry_price,
                        'actual_price': actual_price,
                        'price_change_pct': price_change_pct,
                        'was_correct': bool(correct_with_ai),  # Convert to bool for JSON serialization
                        'has_ai_prediction': pred_with_ai.get('ai_prediction') is not None
                    })
                    
                    results_without_ai.append({
                        'symbol': symbol,
                        'date': point_data['date'].isoformat() if hasattr(point_data['date'], 'isoformat') else str(point_data['date']),
                        'action': pred_without_ai.get('recommendation', {}).get('action', 'HOLD'),
                        'confidence': pred_without_ai.get('recommendation', {}).get('confidence', 50),
                        'entry_price': entry_price,
                        'actual_price': actual_price,
                        'price_change_pct': price_change_pct,
                        'was_correct': bool(correct_without_ai)  # Convert to bool for JSON serialization
                    })
                    
                    total_tested += 1
                    
                    if total_tested % 10 == 0:
                        logger.info(f"Tested {total_tested} points...")
                
            except Exception as e:
                logger.error(f"Error testing {symbol}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
        
        # Calculate statistics
        if not results_with_ai or not results_without_ai:
            logger.error("No results collected!")
            return {'error': 'No results collected'}
        
        # Calculate accuracies
        accuracy_with_ai = sum(1 for r in results_with_ai if r['was_correct']) / len(results_with_ai) * 100
        accuracy_without_ai = sum(1 for r in results_without_ai if r['was_correct']) / len(results_without_ai) * 100
        
        # Count how many times AI was actually used
        ai_used_count = sum(1 for r in results_with_ai if r.get('has_ai_prediction', False))
        ai_usage_rate = (ai_used_count / len(results_with_ai) * 100) if results_with_ai else 0
        
        # Calculate improvement
        improvement = accuracy_with_ai - accuracy_without_ai
        improvement_pct = (improvement / accuracy_without_ai * 100) if accuracy_without_ai > 0 else 0
        
        # Calculate confidence statistics
        avg_confidence_with_ai = np.mean([r['confidence'] for r in results_with_ai])
        avg_confidence_without_ai = np.mean([r['confidence'] for r in results_without_ai])
        
        # Action distribution
        actions_with_ai = {}
        actions_without_ai = {}
        for r in results_with_ai:
            actions_with_ai[r['action']] = actions_with_ai.get(r['action'], 0) + 1
        for r in results_without_ai:
            actions_without_ai[r['action']] = actions_without_ai.get(r['action'], 0) + 1
        
        # Store results
        test_result = {
            'timestamp': datetime.now().isoformat(),
            'strategy': strategy,
            'symbols_tested': symbols,
            'total_predictions': len(results_with_ai),
            'lookforward_days': lookforward_days,
            'accuracy_with_ai': accuracy_with_ai,
            'accuracy_without_ai': accuracy_without_ai,
            'improvement': improvement,
            'improvement_pct': improvement_pct,
            'ai_usage_rate': ai_usage_rate,
            'ai_used_count': ai_used_count,
            'avg_confidence_with_ai': avg_confidence_with_ai,
            'avg_confidence_without_ai': avg_confidence_without_ai,
            'actions_with_ai': actions_with_ai,
            'actions_without_ai': actions_without_ai,
            'results_with_ai': results_with_ai[:100],  # Store first 100 for analysis
            'results_without_ai': results_without_ai[:100]
        }
        
        self.results['tests'].append(test_result)
        
        # Update summary
        self.results['summary'] = {
            'last_test': test_result['timestamp'],
            'total_tests': len(self.results['tests']),
            'avg_accuracy_with_ai': float(np.mean([t['accuracy_with_ai'] for t in self.results['tests']])),
            'avg_accuracy_without_ai': float(np.mean([t['accuracy_without_ai'] for t in self.results['tests']])),
            'avg_improvement': float(np.mean([t['improvement'] for t in self.results['tests']])),
            'avg_improvement_pct': float(np.mean([t['improvement_pct'] for t in self.results['tests']])),
            'ai_helpful': bool(np.mean([t['improvement'] for t in self.results['tests']]) > 0)
        }
        
        self._save_results()
        
        # Print results
        logger.info("\n" + "="*60)
        logger.info("AI ACCURACY TEST RESULTS")
        logger.info("="*60)
        logger.info(f"Strategy: {strategy}")
        logger.info(f"Symbols tested: {', '.join(symbols)}")
        logger.info(f"Total predictions: {len(results_with_ai)}")
        logger.info(f"\nAccuracy WITHOUT AI: {accuracy_without_ai:.2f}%")
        logger.info(f"Accuracy WITH AI:    {accuracy_with_ai:.2f}%")
        logger.info(f"\nImprovement: {improvement:+.2f}% ({improvement_pct:+.1f}%)")
        logger.info(f"\nAI Usage Rate: {ai_usage_rate:.1f}% ({ai_used_count}/{len(results_with_ai)})")
        logger.info(f"\nAverage Confidence:")
        logger.info(f"  Without AI: {avg_confidence_without_ai:.1f}%")
        logger.info(f"  With AI:    {avg_confidence_with_ai:.1f}%")
        logger.info(f"\nAction Distribution:")
        logger.info(f"  With AI:    {actions_with_ai}")
        logger.info(f"  Without AI: {actions_without_ai}")
        logger.info("\n" + "="*60)
        
        if improvement > 0:
            logger.info("✅ AI IMPROVES ACCURACY - Keep AI enabled")
        elif improvement < -2:
            logger.info("❌ AI REDUCES ACCURACY - Consider disabling AI")
        else:
            logger.info("⚠️  AI HAS MINIMAL IMPACT - Consider testing more")
        
        return test_result


def main():
    """Run AI accuracy test"""
    import sys
    
    # Test symbols
    # Test symbols
    test_symbols = ["AAPL", "MSFT"]
    
    # Strategy to test
    strategy = 'trading'  # or 'mixed', 'investing'
    
    # Number of test points per symbol
    test_points = 5
    
    # Lookforward days for validation
    lookforward_days = 10
    
    # Parse command line args if provided
    if len(sys.argv) > 1:
        test_symbols = sys.argv[1].split(',')
    if len(sys.argv) > 2:
        strategy = sys.argv[2]
    if len(sys.argv) > 3:
        test_points = int(sys.argv[3])
    
    logger.info(f"Starting AI accuracy test...")
    logger.info(f"Symbols: {test_symbols}")
    logger.info(f"Strategy: {strategy}")
    logger.info(f"Test points per symbol: {test_points}")
    logger.info(f"Lookforward days: {lookforward_days}")
    
    tester = AIPredictorAccuracyTester(APP_DATA_DIR)
    result = tester.test_accuracy_comparison(
        symbols=test_symbols,
        strategy=strategy,
        lookforward_days=lookforward_days,
        test_points=test_points
    )
    
    if 'error' in result:
        logger.error(f"Test failed: {result['error']}")
        return 1
    
    logger.info("\nTest complete! Results saved to ai_accuracy_test_results.json")
    return 0


if __name__ == "__main__":
    exit(main())
