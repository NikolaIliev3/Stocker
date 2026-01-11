"""
Comprehensive Test Script for All New Components
Tests model versioning, production monitoring, explainability, data validation, etc.
"""
import sys
from pathlib import Path
import numpy as np
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_model_versioning():
    """Test model versioning system"""
    logger.info("=" * 60)
    logger.info("TEST 1: Model Versioning & Rollback")
    logger.info("=" * 60)
    
    try:
        from model_versioning import ModelVersionManager
        
        data_dir = Path("data")
        strategy = "trading"
        version_manager = ModelVersionManager(data_dir, strategy)
        
        # Test version creation
        version_num = version_manager.generate_version_number()
        logger.info(f"✓ Generated version number: {version_num}")
        
        # Test version registration
        metadata = {'samples': 1000, 'test_accuracy': 0.62}
        version_num, should_activate = version_manager.register_new_version(
            metadata=metadata,
            test_accuracy=0.62,
            train_accuracy=0.65,
            cv_mean=0.61
        )
        logger.info(f"✓ Registered version {version_num}, should_activate={should_activate}")
        
        # Test rollback scenario (worse model)
        metadata2 = {'samples': 1000, 'test_accuracy': 0.55}
        version_num2, should_activate2 = version_manager.register_new_version(
            metadata=metadata2,
            test_accuracy=0.55,  # Worse than 0.62
            train_accuracy=0.58,
            cv_mean=0.54
        )
        logger.info(f"✓ Tested worse model: version {version_num2}, should_activate={should_activate2}")
        if not should_activate2:
            logger.info("  ✓ Correctly rejected worse model")
        
        logger.info("✅ Model versioning test PASSED")
        return True
    except Exception as e:
        logger.error(f"❌ Model versioning test FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_production_monitoring():
    """Test production monitoring"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Production Monitoring & Alerting")
    logger.info("=" * 60)
    
    try:
        from production_monitor import ProductionMonitor
        
        data_dir = Path("data")
        strategy = "trading"
        monitor = ProductionMonitor(data_dir, strategy)
        
        # Record some predictions
        for i in range(25):
            monitor.record_prediction(
                symbol=f"TEST{i}",
                action="BUY" if i % 2 == 0 else "SELL",
                confidence=60 + (i % 20),
                features=np.random.rand(75)
            )
        
        # Record outcomes (mix of correct/incorrect)
        for i in range(20):
            monitor.record_outcome(
                symbol=f"TEST{i}",
                action="BUY" if i % 2 == 0 else "SELL",
                was_correct=(i % 3 != 0),  # 2/3 correct
                actual_price_change=(-2 if i % 2 == 0 else 2)
            )
        
        # Check performance
        perf = monitor.get_current_performance()
        logger.info(f"✓ Current performance: {perf.get('accuracy', 0):.1f}%")
        logger.info(f"✓ Status: {perf.get('status')}")
        
        # Check alerts
        alerts = monitor.get_recent_alerts(hours=24)
        logger.info(f"✓ Recent alerts: {len(alerts)}")
        
        # Test retraining trigger
        should_retrain, reason = monitor.should_trigger_retraining()
        logger.info(f"✓ Should retrain: {should_retrain}, reason: {reason}")
        
        logger.info("✅ Production monitoring test PASSED")
        return True
    except Exception as e:
        logger.error(f"❌ Production monitoring test FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_model_explainability():
    """Test model explainability"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Model Explainability")
    logger.info("=" * 60)
    
    try:
        from model_explainability import ModelExplainer
        from sklearn.ensemble import RandomForestClassifier
        
        # Create a simple model for testing
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        X_train = np.random.rand(100, 10)
        y_train = np.random.choice(['BUY', 'SELL'], 100)
        model.fit(X_train, y_train)
        
        feature_names = [f"feature_{i}" for i in range(10)]
        explainer = ModelExplainer(model, feature_names)
        
        # Test explanation
        features = np.random.rand(10)
        prediction = {'action': 'BUY', 'confidence': 75}
        explanation = explainer.explain_prediction(features, prediction)
        
        logger.info(f"✓ Explanation generated: {explanation.get('has_explanation')}")
        logger.info(f"✓ Method: {explanation.get('method')}")
        if explanation.get('top_contributors'):
            logger.info(f"✓ Top contributor: {explanation['top_contributors'][0]}")
        
        logger.info("✅ Model explainability test PASSED")
        return True
    except Exception as e:
        logger.error(f"❌ Model explainability test FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_data_validation():
    """Test training data validation"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Training Data Validation")
    logger.info("=" * 60)
    
    try:
        from training_data_validator import TrainingDataValidator
        
        validator = TrainingDataValidator()
        
        # Create valid training samples
        valid_samples = [
            {
                'features': np.random.rand(75).tolist(),
                'label': 'BUY' if i % 3 == 0 else ('SELL' if i % 3 == 1 else 'HOLD')
            }
            for i in range(100)
        ]
        
        # Test validation
        report = validator.validate_training_samples(valid_samples)
        logger.info(f"✓ Validation passed: {report.get('valid')}")
        logger.info(f"✓ Sample count: {report.get('sample_count')}")
        logger.info(f"✓ Feature count: {report.get('feature_count')}")
        
        # Test with invalid data
        invalid_samples = [
            {'features': np.random.rand(75).tolist()}  # Missing label
            for i in range(10)
        ]
        invalid_report = validator.validate_training_samples(invalid_samples)
        logger.info(f"✓ Invalid data detected: {not invalid_report.get('valid')}")
        
        logger.info("✅ Data validation test PASSED")
        return True
    except Exception as e:
        logger.error(f"❌ Data validation test FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_performance_attribution():
    """Test performance attribution"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Performance Attribution")
    logger.info("=" * 60)
    
    try:
        from performance_attribution import PerformanceAttribution
        
        data_dir = Path("data")
        attribution = PerformanceAttribution(data_dir)
        
        # Record some outcomes
        for i in range(30):
            prediction = {
                'strategy': 'trading' if i % 2 == 0 else 'mixed',
                'action': 'BUY' if i % 3 == 0 else ('SELL' if i % 3 == 1 else 'HOLD'),
                'confidence': 50 + (i % 30),
                'market_regime': 'bull' if i % 3 == 0 else ('bear' if i % 3 == 1 else 'sideways'),
                'sector': 'Technology' if i % 2 == 0 else 'Finance',
                'timestamp': datetime.now().isoformat()
            }
            outcome = {
                'was_correct': (i % 3 != 0),
                'actual_price_change': (-2 if i % 2 == 0 else 2)
            }
            attribution.record_prediction_outcome(prediction, outcome)
        
        # Get report
        report = attribution.get_attribution_report()
        logger.info(f"✓ Attribution report generated")
        logger.info(f"✓ Strategies tracked: {len(report.get('by_strategy', {}))}")
        logger.info(f"✓ Insights: {len(report.get('insights', []))}")
        
        logger.info("✅ Performance attribution test PASSED")
        return True
    except Exception as e:
        logger.error(f"❌ Performance attribution test FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_quality_scorer():
    """Test prediction quality scorer"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: Prediction Quality Scorer")
    logger.info("=" * 60)
    
    try:
        from prediction_quality_scorer import PredictionQualityScorer
        
        scorer = PredictionQualityScorer()
        
        # Test high quality prediction
        high_quality_pred = {
            'action': 'BUY',
            'confidence': 85,
            'probabilities': {'BUY': 85, 'SELL': 10, 'HOLD': 5}
        }
        score1 = scorer.score_prediction(high_quality_pred, features=np.random.rand(75))
        logger.info(f"✓ High quality prediction: score={score1.get('quality_score', 0):.2f}, level={score1.get('quality_level')}")
        
        # Test low quality prediction
        low_quality_pred = {
            'action': 'HOLD',
            'confidence': 45,
            'probabilities': {'BUY': 35, 'SELL': 30, 'HOLD': 35}
        }
        score2 = scorer.score_prediction(low_quality_pred, features=np.random.rand(75))
        logger.info(f"✓ Low quality prediction: score={score2.get('quality_score', 0):.2f}, level={score2.get('quality_level')}")
        logger.info(f"✓ Should flag: {score2.get('should_flag')}")
        
        logger.info("✅ Quality scorer test PASSED")
        return True
    except Exception as e:
        logger.error(f"❌ Quality scorer test FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_benchmarking():
    """Test model benchmarking"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 7: Model Benchmarking")
    logger.info("=" * 60)
    
    try:
        from model_benchmarking import ModelBenchmarker
        
        data_dir = Path("data")
        benchmarker = ModelBenchmarker(data_dir)
        
        # Create sample predictions and outcomes
        predictions = [
            {'action': 'BUY' if i % 3 == 0 else ('SELL' if i % 3 == 1 else 'HOLD'), 'strategy': 'trading', 'confidence': 60}
            for i in range(50)
        ]
        outcomes = [
            {'was_correct': (i % 3 != 0), 'actual_price_change': (-2 if i % 2 == 0 else 2)}
            for i in range(50)
        ]
        
        # Calculate baselines
        baselines = benchmarker.calculate_baselines(predictions, outcomes)
        logger.info(f"✓ Baselines calculated: {len(baselines)} baselines")
        
        # Compare model
        comparisons = benchmarker.compare_to_baselines(58.5, 'trading')
        logger.info(f"✓ Comparisons generated: {len(comparisons)} comparisons")
        
        # Get report
        report = benchmarker.get_benchmark_report('trading')
        logger.info(f"✓ Benchmark report generated")
        logger.info(f"✓ Summary items: {len(report.get('summary', []))}")
        
        logger.info("✅ Benchmarking test PASSED")
        return True
    except Exception as e:
        logger.error(f"❌ Benchmarking test FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def test_ab_testing():
    """Test A/B testing framework"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 8: A/B Testing Framework")
    logger.info("=" * 60)
    
    try:
        from ab_testing_framework import ABTestingFramework
        
        data_dir = Path("data")
        strategy = "trading"
        ab_framework = ABTestingFramework(data_dir, strategy)
        
        # Create A/B test
        test_id = ab_framework.create_ab_test(
            test_name="test_new_model",
            control_version="v1.0",
            treatment_version="v2.0",
            traffic_split=0.5
        )
        logger.info(f"✓ Created A/B test: {test_id}")
        
        # Assign predictions to groups
        for i in range(20):
            pred_id = f"pred_{i}"
            group = ab_framework.assign_to_group(test_id, pred_id)
            logger.debug(f"  Prediction {pred_id} assigned to {group}")
        
        # Record outcomes
        for i in range(20):
            pred_id = f"pred_{i}"
            group = ab_framework.assign_to_group(test_id, pred_id)
            was_correct = (i % 3 != 0)  # 2/3 correct
            ab_framework.record_outcome(test_id, pred_id, was_correct, group)
        
        # Get active tests
        active_tests = ab_framework.get_active_tests()
        logger.info(f"✓ Active tests: {len(active_tests)}")
        
        # Get test results
        test_results = ab_framework.get_test_results(test_id)
        if test_results:
            logger.info(f"✓ Test results retrieved")
            logger.info(f"  Control: {test_results.get('control_results', {}).get('accuracy', 0):.1f}%")
            logger.info(f"  Treatment: {test_results.get('treatment_results', {}).get('accuracy', 0):.1f}%")
        
        logger.info("✅ A/B testing test PASSED")
        return True
    except Exception as e:
        logger.error(f"❌ A/B testing test FAILED: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Run all tests"""
    logger.info("\n" + "=" * 60)
    logger.info("COMPREHENSIVE TEST SUITE FOR NEW COMPONENTS")
    logger.info("=" * 60)
    
    results = []
    
    # Run all tests
    results.append(("Model Versioning", test_model_versioning()))
    results.append(("Production Monitoring", test_production_monitoring()))
    results.append(("Model Explainability", test_model_explainability()))
    results.append(("Data Validation", test_data_validation()))
    results.append(("Performance Attribution", test_performance_attribution()))
    results.append(("Quality Scorer", test_quality_scorer()))
    results.append(("Benchmarking", test_benchmarking()))
    results.append(("A/B Testing", test_ab_testing()))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "PASSED" if result else "FAILED"
        logger.info(f"{'✓' if result else '✗'} {name}: {status}")
    
    logger.info(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED!")
        return 0
    else:
        logger.error(f"❌ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
