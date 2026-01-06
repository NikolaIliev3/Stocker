"""
Model Testing Module
Tests trained ML models on out-of-sample data to validate performance
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

from ml_training import StockPredictionML, FeatureExtractor
from data_fetcher import StockDataFetcher

logger = logging.getLogger(__name__)


class ModelTester:
    """Tests ML models on out-of-sample data"""
    
    def __init__(self, data_dir, data_fetcher: StockDataFetcher):
        self.data_dir = data_dir
        self.data_fetcher = data_fetcher
        self.feature_extractor = FeatureExtractor()
    
    def test_model(self, strategy: str, symbols: List[str], start_date: str, 
                  end_date: str, progress_callback=None) -> Dict:
        """
        Test trained model on out-of-sample data
        
        Args:
            strategy: Strategy to test
            symbols: List of stock symbols to test
            start_date: Start date for testing (YYYY-MM-DD)
            end_date: End date for testing (YYYY-MM-DD)
            progress_callback: Optional callback for progress updates
        
        Returns:
            Dictionary with test results
        """
        # Load trained model
        ml_model = StockPredictionML(self.data_dir, strategy)
        
        # Check if model exists
        from secure_ml_storage import SecureMLStorage
        storage = SecureMLStorage(self.data_dir)
        ml_model_file = f"ml_model_{strategy}.pkl"
        
        if not storage.model_exists(ml_model_file):
            return {
                'error': 'Model not trained. Please train the model first.',
                'strategy': strategy
            }
        
        # Get training accuracy from metadata
        # Check if metadata exists and was loaded
        if not hasattr(ml_model, 'metadata') or not ml_model.metadata:
            # Try to load metadata manually
            from pathlib import Path
            import json
            metadata_file = self.data_dir / f"ml_metadata_{strategy}.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        ml_model.metadata = json.load(f)
                        logger.info(f"Loaded metadata manually: {ml_model.metadata}")
                except Exception as e:
                    logger.warning(f"Could not load metadata: {e}")
                    ml_model.metadata = {}
        
        # Get training accuracy (stored as train_accuracy in metadata)
        # train_accuracy = accuracy on training set during model training
        # test_accuracy = accuracy on validation set during model training
        training_accuracy = 0
        if ml_model.metadata:
            # Use train_accuracy (actual training set accuracy)
            training_accuracy = ml_model.metadata.get('train_accuracy', 0) * 100
            logger.info(f"Training accuracy from metadata: {training_accuracy:.1f}%")
            # If train_accuracy is 0, try test_accuracy (validation accuracy during training)
            if training_accuracy == 0:
                training_accuracy = ml_model.metadata.get('test_accuracy', 0) * 100
                logger.info(f"Using validation accuracy instead: {training_accuracy:.1f}%")
        else:
            logger.warning(f"No metadata found for {strategy} model")
        
        all_test_results = []
        total_samples = 0
        
        for i, symbol in enumerate(symbols):
            if progress_callback:
                progress_callback(f"Testing {symbol} ({i+1}/{len(symbols)})...")
            
            try:
                # Get historical data for test period
                ticker = yf.Ticker(symbol.upper())
                hist = ticker.history(start=start_date, end=end_date)
                
                if hist.empty or len(hist) < 50:
                    logger.warning(f"Insufficient data for {symbol} in test period")
                    continue
                
                dates = hist.index.tolist()
                
                # Test on sliding window (similar to training)
                lookforward_days = 10 if strategy == "trading" else 21 if strategy == "mixed" else 547
                
                for j in range(50, len(dates) - lookforward_days, 5):  # Test every 5 days
                    current_date = dates[j]
                    current_price = float(hist.iloc[j]['Close'])
                    
                    # Get historical data up to current date
                    hist_to_date = hist.iloc[:j+1]
                    
                    # Convert to format expected by analyzers
                    history_data = {
                        'data': [
                            {
                                'date': date.strftime('%Y-%m-%d'),
                                'open': float(row['Open']),
                                'high': float(row['High']),
                                'low': float(row['Low']),
                                'close': float(row['Close']),
                                'volume': int(row['Volume'])
                            }
                            for date, row in hist_to_date.iterrows()
                        ]
                    }
                    
                    stock_data = {
                        'symbol': symbol,
                        'price': current_price,
                        'info': {}
                    }
                    
                    # Extract features
                    try:
                        features = self.feature_extractor.extract_features(
                            stock_data, history_data, None, None
                        )
                    except Exception as e:
                        logger.warning(f"Error extracting features for {symbol} at {current_date}: {e}")
                        continue
                    
                    # Get model prediction
                    ml_prediction = ml_model.predict(features)
                    predicted_action = ml_prediction.get('action', 'HOLD')
                    predicted_confidence = ml_prediction.get('confidence', 50)
                    
                    # Get actual outcome
                    future_idx = min(j + lookforward_days, len(dates) - 1)
                    future_price = float(hist.iloc[future_idx]['Close'])
                    price_change_pct = ((future_price - current_price) / current_price) * 100
                    
                    # Determine actual label based on strategy
                    if strategy == "trading":
                        if price_change_pct >= 3:
                            actual_label = "BUY"
                        elif price_change_pct <= -3:
                            actual_label = "SELL"
                        else:
                            actual_label = "HOLD"
                    elif strategy == "investing":
                        if price_change_pct >= 10:
                            actual_label = "BUY"
                        elif price_change_pct <= -10:
                            actual_label = "AVOID"
                        else:
                            actual_label = "HOLD"
                    else:  # mixed
                        if price_change_pct >= 5:
                            actual_label = "BUY"
                        elif price_change_pct <= -5:
                            actual_label = "AVOID"
                        else:
                            actual_label = "HOLD"
                    
                    # Check if prediction was correct
                    was_correct = (predicted_action == actual_label)
                    
                    all_test_results.append({
                        'symbol': symbol,
                        'date': current_date.strftime('%Y-%m-%d'),
                        'predicted_action': predicted_action,
                        'actual_label': actual_label,
                        'was_correct': was_correct,
                        'confidence': predicted_confidence,
                        'price_change_pct': price_change_pct
                    })
                    
                    total_samples += 1
                    
            except Exception as e:
                logger.error(f"Error testing {symbol}: {e}")
                continue
        
        if total_samples == 0:
            return {
                'error': 'No test samples generated. Check date range and symbols.',
                'strategy': strategy
            }
        
        # Calculate test metrics
        correct = sum(1 for r in all_test_results if r['was_correct'] is True)
        test_accuracy = (correct / total_samples * 100) if total_samples > 0 else 0
        
        # Calculate per-action accuracy
        buy_results = [r for r in all_test_results if r['predicted_action'] == 'BUY']
        sell_results = [r for r in all_test_results if r['predicted_action'] == 'SELL']
        hold_results = [r for r in all_test_results if r['predicted_action'] == 'HOLD']
        
        buy_accuracy = (sum(1 for r in buy_results if r['was_correct']) / len(buy_results) * 100) if buy_results else 0
        sell_accuracy = (sum(1 for r in sell_results if r['was_correct']) / len(sell_results) * 100) if sell_results else 0
        hold_accuracy = (sum(1 for r in hold_results if r['was_correct']) / len(hold_results) * 100) if hold_results else 0
        
        # Calculate accuracy difference
        accuracy_difference = test_accuracy - training_accuracy
        
        # Determine overfitting status
        if abs(accuracy_difference) < 5:
            overfitting_status = "good"
            overfitting_message = "Model generalizes well"
        elif accuracy_difference < -10:
            overfitting_status = "poor"
            overfitting_message = "Possible overfitting detected"
        else:
            overfitting_status = "fair"
            overfitting_message = "Model performance acceptable"
        
        return {
            'strategy': strategy,
            'test_period': f"{start_date} to {end_date}",
            'symbols_tested': symbols,
            'total_samples': total_samples,
            'correct': correct,
            'test_accuracy': test_accuracy,
            'training_accuracy': training_accuracy,
            'accuracy_difference': accuracy_difference,
            'overfitting_status': overfitting_status,
            'overfitting_message': overfitting_message,
            'buy_accuracy': buy_accuracy,
            'sell_accuracy': sell_accuracy,
            'hold_accuracy': hold_accuracy,
            'buy_count': len(buy_results),
            'sell_count': len(sell_results),
            'hold_count': len(hold_results),
            'results': all_test_results
        }

