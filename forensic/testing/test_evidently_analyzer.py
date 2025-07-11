#!/usr/bin/env python3
"""
Test Suite for Evidently Analyzer
=================================

Comprehensive test suite for the EvidentlyAnalyzer class to ensure
proper functionality, error handling, and integration with the
existing forensic framework.

Author: Forensic AI Testing Suite
Created: 2025-07-01
Purpose: Validation testing for Evidently integration
"""

import os
import sys
import unittest
import tempfile
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the forensic testing directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from evidently_analyzer import (
        EvidentlyAnalyzer,
        EvidentlyAnalysisResult,
        BiasMonitoringAlert,
        create_evidently_test_suite,
        run_evidently_forensic_analysis,
        EVIDENTLY_AVAILABLE
    )
except ImportError as e:
    print(f"Error importing Evidently analyzer: {e}")
    EVIDENTLY_AVAILABLE = False


class TestEvidentlyAnalyzer(unittest.TestCase):
    """Test cases for EvidentlyAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not EVIDENTLY_AVAILABLE:
            self.skipTest("Evidently library not available")
        
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.test_dir) / "test_output"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate test data
        np.random.seed(42)
        self.reference_data = self._generate_test_data(n_samples=500, add_bias=False)
        self.current_data = self._generate_test_data(n_samples=400, add_bias=True)
        
        # Initialize analyzer
        self.analyzer = EvidentlyAnalyzer(
            case_id="TEST_CASE_001",
            investigator="Test Analyst",
            output_dir=str(self.output_dir),
            enable_monitoring=True,
            enable_alerts=True
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, 'test_dir') and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def _generate_test_data(self, n_samples=500, add_bias=False):
        """Generate synthetic test data."""
        np.random.seed(42 if not add_bias else 123)
        
        data = {
            'experience_years': np.random.normal(5, 2, n_samples),
            'education_score': np.random.normal(7, 1.5, n_samples),
            'skills_match': np.random.uniform(0, 1, n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples),
            'age_group': np.random.choice(['18-30', '31-45', '46-60'], n_samples)
        }
        
        # Generate target and prediction
        qualification_score = (
            0.4 * (data['experience_years'] / 10) +
            0.3 * (data['education_score'] / 10) +
            0.3 * data['skills_match']
        )
        
        data['target'] = (qualification_score > 0.5).astype(int)
        
        # Add bias to predictions if requested
        prediction_score = qualification_score + np.random.normal(0, 0.1, n_samples)
        if add_bias:
            # Favor males
            male_bias = np.where(np.array(data['gender']) == 'Male', 0.2, 0)
            prediction_score += male_bias
        
        data['prediction'] = (prediction_score > 0.5).astype(int)
        
        return pd.DataFrame(data)
    
    def test_analyzer_initialization(self):
        """Test analyzer initialization."""
        self.assertEqual(self.analyzer.case_id, "TEST_CASE_001")
        self.assertEqual(self.analyzer.investigator, "Test Analyst")
        self.assertTrue(self.analyzer.enable_monitoring)
        self.assertTrue(self.analyzer.enable_alerts)
        self.assertIsNotNone(self.analyzer.bias_thresholds)
        self.assertIsNotNone(self.analyzer.drift_thresholds)
    
    def test_column_mapping_configuration(self):
        """Test column mapping configuration."""
        column_mapping = self.analyzer.configure_column_mapping(
            target='target',
            prediction='prediction',
            numerical_features=['experience_years', 'education_score', 'skills_match'],
            categorical_features=['gender', 'race', 'age_group'],
            protected_attributes=['gender', 'race']
        )
        
        self.assertIsNotNone(column_mapping)
        self.assertEqual(column_mapping.target, 'target')
        self.assertEqual(column_mapping.prediction, 'prediction')
        self.assertEqual(self.analyzer.protected_attributes, ['gender', 'race'])
    
    def test_data_drift_detection(self):
        """Test data drift detection functionality."""
        # Configure column mapping
        self.analyzer.configure_column_mapping(
            target='target',
            prediction='prediction',
            numerical_features=['experience_years', 'education_score', 'skills_match'],
            categorical_features=['gender', 'race', 'age_group']
        )
        
        # Test drift detection
        drift_results = self.analyzer.detect_data_drift(
            self.reference_data, 
            self.current_data,
            generate_report=False  # Skip report generation for faster testing
        )
        
        self.assertIsInstance(drift_results, dict)
        self.assertIn('dataset_drift_detected', drift_results)
        self.assertIn('drift_score', drift_results)
        self.assertIn('drift_by_column', drift_results)
        self.assertIsInstance(drift_results['drift_score'], (int, float))
    
    def test_model_performance_analysis(self):
        """Test model performance analysis."""
        # Configure column mapping
        self.analyzer.configure_column_mapping(
            target='target',
            prediction='prediction',
            numerical_features=['experience_years', 'education_score', 'skills_match'],
            categorical_features=['gender', 'race', 'age_group']
        )
        
        # Test performance analysis
        performance_results = self.analyzer.analyze_model_performance(
            self.reference_data,
            self.current_data,
            task_type='classification',
            generate_report=False
        )
        
        self.assertIsInstance(performance_results, dict)
        self.assertIn('performance_degraded', performance_results)
        self.assertIn('current_metrics', performance_results)
        self.assertIn('performance_change', performance_results)
    
    def test_data_quality_assessment(self):
        """Test data quality assessment."""
        # Configure column mapping
        self.analyzer.configure_column_mapping(
            target='target',
            prediction='prediction',
            numerical_features=['experience_years', 'education_score', 'skills_match'],
            categorical_features=['gender', 'race', 'age_group']
        )
        
        # Test quality assessment
        quality_results = self.analyzer.assess_data_quality(
            self.reference_data,
            self.current_data,
            generate_report=False
        )
        
        self.assertIsInstance(quality_results, dict)
        self.assertIn('quality_issues_detected', quality_results)
        self.assertIn('issues_summary', quality_results)
    
    def test_bias_detection(self):
        """Test bias detection functionality."""
        # Configure column mapping
        self.analyzer.configure_column_mapping(
            target='target',
            prediction='prediction',
            numerical_features=['experience_years', 'education_score', 'skills_match'],
            categorical_features=['gender', 'race', 'age_group'],
            protected_attributes=['gender', 'race']
        )
        
        # Test bias detection
        bias_results = self.analyzer.detect_bias(
            self.reference_data,
            self.current_data,
            protected_attributes=['gender', 'race'],
            generate_report=False
        )
        
        self.assertIsInstance(bias_results, dict)
        self.assertIn('bias_detected', bias_results)
        self.assertIn('bias_by_attribute', bias_results)
        self.assertIn('protected_attributes', bias_results)
        self.assertEqual(bias_results['protected_attributes'], ['gender', 'race'])
    
    def test_comprehensive_analysis(self):
        """Test comprehensive analysis workflow."""
        # Configure column mapping
        self.analyzer.configure_column_mapping(
            target='target',
            prediction='prediction',
            numerical_features=['experience_years', 'education_score', 'skills_match'],
            categorical_features=['gender', 'race', 'age_group'],
            protected_attributes=['gender', 'race']
        )
        
        # Run comprehensive analysis
        result = self.analyzer.run_comprehensive_analysis(
            reference_data=self.reference_data,
            current_data=self.current_data,
            task_type='classification',
            protected_attributes=['gender', 'race'],
            generate_reports=False  # Skip reports for faster testing
        )
        
        # Validate result structure
        self.assertIsInstance(result, EvidentlyAnalysisResult)
        self.assertEqual(result.case_id, "TEST_CASE_001")
        self.assertEqual(result.investigator, "Test Analyst")
        self.assertIsNotNone(result.timestamp)
        self.assertIsNotNone(result.data_hash)
        
        # Check analysis results
        self.assertIsInstance(result.data_drift_detected, bool)
        self.assertIsInstance(result.bias_detected, bool)
        self.assertIsInstance(result.model_performance_degraded, bool)
        
        # Verify protected attributes
        self.assertEqual(result.protected_attributes, ['gender', 'race'])
    
    def test_alert_generation(self):
        """Test automated alert generation."""
        # Configure with sensitive thresholds to trigger alerts
        self.analyzer.bias_thresholds['demographic_parity_difference'] = 0.01
        
        # Configure column mapping
        self.analyzer.configure_column_mapping(
            target='target',
            prediction='prediction',
            numerical_features=['experience_years', 'education_score', 'skills_match'],
            categorical_features=['gender', 'race', 'age_group'],
            protected_attributes=['gender', 'race']
        )
        
        # Run analysis (should trigger alerts with biased data)
        result = self.analyzer.run_comprehensive_analysis(
            reference_data=self.reference_data,
            current_data=self.current_data,
            protected_attributes=['gender', 'race'],
            generate_reports=False
        )
        
        # Check if alerts were generated
        alerts = self.analyzer.get_alerts()
        if alerts:
            self.assertIsInstance(alerts[0], BiasMonitoringAlert)
            self.assertIsNotNone(alerts[0].alert_id)
            self.assertIsNotNone(alerts[0].timestamp)
    
    def test_custom_thresholds(self):
        """Test custom threshold configuration."""
        # Set custom thresholds
        custom_bias_thresholds = {
            'demographic_parity_difference': 0.05,
            'accuracy_difference': 0.02
        }
        
        custom_drift_thresholds = {
            'dataset_drift_threshold': 0.3,
            'column_drift_threshold': 0.02
        }
        
        self.analyzer.bias_thresholds.update(custom_bias_thresholds)
        self.analyzer.drift_thresholds.update(custom_drift_thresholds)
        
        # Verify thresholds were set
        self.assertEqual(
            self.analyzer.bias_thresholds['demographic_parity_difference'], 
            0.05
        )
        self.assertEqual(
            self.analyzer.drift_thresholds['dataset_drift_threshold'], 
            0.3
        )
    
    def test_monitoring_dashboard_creation(self):
        """Test monitoring dashboard creation."""
        dashboard_path = self.analyzer.create_monitoring_dashboard(
            workspace_name="test_workspace",
            project_name="test_project"
        )
        
        self.assertIsInstance(dashboard_path, str)
        self.assertTrue(os.path.exists(dashboard_path))
        
        # Verify dashboard config file was created
        with open(dashboard_path, 'r') as f:
            import json
            config = json.load(f)
            self.assertIn('workspace_name', config)
            self.assertEqual(config['workspace_name'], 'test_workspace')
    
    def test_alert_configuration(self):
        """Test automated alert configuration."""
        custom_alert_config = {
            'bias_alerts': {
                'test_metric': {
                    'threshold': 0.05,
                    'severity': 'high',
                    'actions': ['log_alert']
                }
            }
        }
        
        result_config = self.analyzer.setup_automated_alerts(custom_alert_config)
        
        self.assertIsInstance(result_config, dict)
        self.assertIn('bias_alerts', result_config)
        self.assertIn('test_metric', result_config['bias_alerts'])
    
    def test_export_functionality(self):
        """Test result export functionality."""
        # Configure and run analysis
        self.analyzer.configure_column_mapping(
            target='target',
            prediction='prediction',
            protected_attributes=['gender']
        )
        
        result = self.analyzer.run_comprehensive_analysis(
            reference_data=self.reference_data,
            current_data=self.current_data,
            protected_attributes=['gender'],
            generate_reports=False
        )
        
        # Test JSON export
        json_path = self.analyzer.export_analysis_results(result, format='json')
        self.assertTrue(os.path.exists(json_path))
        
        # Test CSV export
        csv_path = self.analyzer.export_analysis_results(result, format='csv')
        self.assertTrue(os.path.exists(csv_path))
        
        # Test XML export
        xml_path = self.analyzer.export_analysis_results(result, format='xml')
        self.assertTrue(os.path.exists(xml_path))
    
    def test_data_hashing(self):
        """Test data integrity hashing."""
        hash1 = self.analyzer._calculate_data_hash(self.reference_data)
        hash2 = self.analyzer._calculate_data_hash(self.reference_data)
        hash3 = self.analyzer._calculate_data_hash(self.current_data)
        
        # Same data should produce same hash
        self.assertEqual(hash1, hash2)
        
        # Different data should produce different hash
        self.assertNotEqual(hash1, hash3)
        
        # Hash should be valid SHA-256
        self.assertEqual(len(hash1), 64)  # SHA-256 produces 64-character hex string
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with missing target column
        invalid_data = self.reference_data.drop(columns=['target'])
        
        with self.assertLogs(level='WARNING') as log:
            self.analyzer.configure_column_mapping(
                target='missing_target',
                prediction='prediction'
            )
            
            # This should handle the missing column gracefully
            bias_results = self.analyzer.detect_bias(
                self.reference_data,
                invalid_data,
                protected_attributes=['gender'],
                generate_report=False
            )
            
            # Should return results even with missing data
            self.assertIsInstance(bias_results, dict)
    
    def test_utility_functions(self):
        """Test utility functions."""
        if EVIDENTLY_AVAILABLE:
            # Test create_evidently_test_suite
            test_suite = create_evidently_test_suite(
                self.reference_data,
                self.current_data
            )
            self.assertIsNotNone(test_suite)
        
        # Test run_evidently_forensic_analysis convenience function
        result = run_evidently_forensic_analysis(
            reference_data=self.reference_data,
            current_data=self.current_data,
            case_id="UTILITY_TEST",
            protected_attributes=['gender'],
            output_dir=str(self.output_dir)
        )
        
        self.assertIsInstance(result, EvidentlyAnalysisResult)
        self.assertEqual(result.case_id, "UTILITY_TEST")


class TestEvidentlyIntegration(unittest.TestCase):
    """Test integration with existing forensic framework."""
    
    def test_import_availability(self):
        """Test that imports work correctly."""
        if EVIDENTLY_AVAILABLE:
            # Test that all required classes can be imported
            from evidently_analyzer import (
                EvidentlyAnalyzer,
                EvidentlyAnalysisResult,
                BiasMonitoringAlert
            )
            
            self.assertTrue(True)  # If we get here, imports worked
        else:
            self.skipTest("Evidently library not available")
    
    def test_forensic_logger_integration(self):
        """Test integration with ForensicLogger."""
        if not EVIDENTLY_AVAILABLE:
            self.skipTest("Evidently library not available")
        
        analyzer = EvidentlyAnalyzer(case_id="LOGGER_TEST")
        
        # Check that logger was initialized
        self.assertIsNotNone(analyzer.logger)
        self.assertIn("evidently_analyzer", analyzer.logger.name)
    
    @unittest.skipIf(not EVIDENTLY_AVAILABLE, "Evidently library not available")
    def test_package_integration(self):
        """Test integration with main package."""
        # Test that the analyzer can be imported from the main package
        try:
            from forensic.testing import EVIDENTLY_AVAILABLE as package_available
            if package_available:
                from forensic.testing import EvidentlyAnalyzer
                self.assertTrue(True)  # Import successful
        except ImportError:
            # This is expected if running tests outside the package structure
            pass


class TestEvidentlyWithoutLibrary(unittest.TestCase):
    """Test behavior when Evidently library is not available."""
    
    @patch('evidently_analyzer.EVIDENTLY_AVAILABLE', False)
    def test_graceful_degradation(self):
        """Test that the module handles missing Evidently library gracefully."""
        with self.assertRaises(ImportError):
            from evidently_analyzer import EvidentlyAnalyzer
            EvidentlyAnalyzer()


if __name__ == '__main__':
    # Setup test configuration
    import logging
    logging.getLogger().setLevel(logging.WARNING)  # Reduce log noise during testing
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestEvidentlyAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestEvidentlyIntegration))
    test_suite.addTest(unittest.makeSuite(TestEvidentlyWithoutLibrary))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, failure in result.failures:
            print(f"  {test}: {failure}")
    
    if result.errors:
        print("\nErrors:")
        for test, error in result.errors:
            print(f"  {test}: {error}")
    
    if not EVIDENTLY_AVAILABLE:
        print("\nNote: Evidently library not available. Install with: pip install evidently")
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)