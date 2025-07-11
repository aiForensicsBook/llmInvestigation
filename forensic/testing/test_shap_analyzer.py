#!/usr/bin/env python3
"""
Unit Tests for SHAP Analyzer Module
===================================

Comprehensive test suite for the SHAP analyzer module to ensure
forensic integrity and correct functionality for legal proceedings.

Author: Forensic AI Testing Suite
Created: 2025-07-01
Purpose: Validation of SHAP analysis for forensic testing
"""

import unittest
import tempfile
import shutil
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd

from src.model.resume_llm import ResumeScreeningLLM
from forensic.testing.shap_analyzer import (
    ShapAnalyzer, 
    TFIDFShapExplainer, 
    ShapExplanation,
    ShapAnalysisResult,
    ForensicShapLogger
)


class TestShapAnalyzer(unittest.TestCase):
    """Test cases for the main ShapAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        
        # Sample training data
        self.sample_resumes = [
            {
                "id": "resume_001",
                "skills": ["python", "machine learning", "data analysis"],
                "education": [{"degree": "Bachelor's", "field": "Computer Science"}],
                "experience": [{"title": "Data Scientist", "description": "ML models"}],
                "years_experience": 3,
                "gender": "female",
                "age": 28
            },
            {
                "id": "resume_002",
                "skills": ["java", "spring", "microservices"],
                "education": [{"degree": "Master's", "field": "Software Engineering"}],
                "experience": [{"title": "Software Engineer", "description": "Backend development"}],
                "years_experience": 5,
                "gender": "male", 
                "age": 32
            },
            {
                "id": "resume_003",
                "skills": ["python", "tensorflow", "deep learning"],
                "education": [{"degree": "PhD", "field": "Artificial Intelligence"}],
                "experience": [{"title": "Research Scientist", "description": "AI research"}],
                "years_experience": 7,
                "gender": "male",
                "age": 35
            }
        ]
        
        self.sample_job = {
            "title": "Senior Data Scientist",
            "description": "Looking for ML expert",
            "requirements": {
                "skills": ["python", "machine learning", "tensorflow"],
                "experience": "5+ years in data science"
            }
        }
        
        # Initialize and train model
        self.model = ResumeScreeningLLM(vocab_size=100)
        self.model.train(self.sample_resumes, [self.sample_job])
        
        # Initialize analyzer
        self.analyzer = ShapAnalyzer(self.model, output_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_analyzer_initialization(self):
        """Test proper initialization of ShapAnalyzer."""
        self.assertIsInstance(self.analyzer, ShapAnalyzer)
        self.assertEqual(self.analyzer.model, self.model)
        self.assertTrue(os.path.exists(self.test_dir))
        self.assertIsNone(self.analyzer.explainer)  # Not initialized yet
    
    def test_explainer_initialization(self):
        """Test initialization of SHAP explainer."""
        self.analyzer.initialize_explainer(background_data=self.sample_resumes[:2])
        
        self.assertIsNotNone(self.analyzer.explainer)
        self.assertIsInstance(self.analyzer.explainer, TFIDFShapExplainer)
        self.assertEqual(self.analyzer.explainer.model, self.model)
    
    def test_individual_explanations(self):
        """Test generation of individual SHAP explanations."""
        self.analyzer.initialize_explainer(background_data=self.sample_resumes[:2])
        
        explanations = self.analyzer.explain_predictions(
            self.sample_resumes[2:], self.sample_job
        )
        
        self.assertEqual(len(explanations), 1)
        explanation = explanations[0]
        
        self.assertIsInstance(explanation, ShapExplanation)
        self.assertIsNotNone(explanation.explanation_id)
        self.assertIsNotNone(explanation.timestamp)
        self.assertIsInstance(explanation.prediction_value, float)
        self.assertIsInstance(explanation.shap_values, list)
        self.assertIsInstance(explanation.feature_names, list)
        self.assertEqual(len(explanation.shap_values), len(explanation.feature_names))
    
    def test_global_importance_analysis(self):
        """Test global feature importance calculation."""
        self.analyzer.initialize_explainer(background_data=self.sample_resumes[:2])
        explanations = self.analyzer.explain_predictions(
            self.sample_resumes, self.sample_job
        )
        
        global_importance = self.analyzer.analyze_global_importance(explanations, top_k=5)
        
        self.assertIsInstance(global_importance, dict)
        self.assertLessEqual(len(global_importance), 5)
        
        # Check that importance values are non-negative
        for feature, importance in global_importance.items():
            self.assertIsInstance(feature, str)
            self.assertIsInstance(importance, float)
            self.assertGreaterEqual(importance, 0.0)
    
    def test_feature_interaction_analysis(self):
        """Test feature interaction analysis."""
        self.analyzer.initialize_explainer(background_data=self.sample_resumes[:2])
        explanations = self.analyzer.explain_predictions(
            self.sample_resumes, self.sample_job
        )
        
        interactions = self.analyzer.analyze_feature_interactions(explanations, top_k=3)
        
        self.assertIsInstance(interactions, dict)
        self.assertLessEqual(len(interactions), 3)
        
        # Check interaction format
        for interaction_pair, strength in interactions.items():
            self.assertIn(" x ", interaction_pair)
            self.assertIsInstance(strength, float)
            self.assertGreaterEqual(strength, 0.0)
    
    def test_demographic_analysis(self):
        """Test demographic difference analysis."""
        self.analyzer.initialize_explainer(background_data=self.sample_resumes[:1])
        explanations = self.analyzer.explain_predictions(
            self.sample_resumes, self.sample_job, demographic_column="gender"
        )
        
        # Add demographic info to explanations manually for testing
        for i, explanation in enumerate(explanations):
            explanation.demographic_group = self.sample_resumes[i]["gender"]
        
        demographic_analysis = self.analyzer.analyze_demographic_differences(explanations)
        
        self.assertIsInstance(demographic_analysis, dict)
        self.assertIn("female", demographic_analysis)
        self.assertIn("male", demographic_analysis)
        
        for group, importance_dict in demographic_analysis.items():
            self.assertIsInstance(importance_dict, dict)
            for feature, importance in importance_dict.items():
                self.assertIsInstance(importance, float)
                self.assertGreaterEqual(importance, 0.0)
    
    def test_comprehensive_analysis(self):
        """Test comprehensive SHAP analysis."""
        analysis_result = self.analyzer.generate_comprehensive_analysis(
            test_data=self.sample_resumes,
            job_posting=self.sample_job,
            background_data=self.sample_resumes[:2],
            demographic_column="gender"
        )
        
        self.assertIsInstance(analysis_result, ShapAnalysisResult)
        self.assertIsNotNone(analysis_result.analysis_id)
        self.assertIsNotNone(analysis_result.timestamp)
        self.assertIsInstance(analysis_result.individual_explanations, list)
        self.assertIsInstance(analysis_result.global_feature_importance, dict)
        self.assertIsInstance(analysis_result.feature_interactions, dict)
        self.assertIsInstance(analysis_result.demographic_comparisons, dict)
        self.assertIsInstance(analysis_result.analysis_summary, dict)
        self.assertIsInstance(analysis_result.visualizations_generated, list)
    
    def test_report_generation(self):
        """Test interpretability report generation."""
        analysis_result = self.analyzer.generate_comprehensive_analysis(
            test_data=self.sample_resumes,
            job_posting=self.sample_job,
            background_data=self.sample_resumes[:2]
        )
        
        report_path = self.analyzer.generate_interpretability_report(analysis_result)
        
        self.assertTrue(os.path.exists(report_path))
        
        # Verify report content
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        self.assertIn("metadata", report)
        self.assertIn("executive_summary", report)
        self.assertIn("model_behavior_analysis", report)
        self.assertIn("forensic_documentation", report)
        self.assertIn("legal_compliance", report)
    
    def test_data_integrity_verification(self):
        """Test data integrity and hashing."""
        test_data = pd.DataFrame(self.sample_resumes)
        hash1 = self.analyzer._calculate_data_hash(test_data)
        hash2 = self.analyzer._calculate_data_hash(test_data)
        
        # Same data should produce same hash
        self.assertEqual(hash1, hash2)
        
        # Different data should produce different hash
        modified_data = test_data.copy()
        modified_data.loc[0, 'years_experience'] = 999
        hash3 = self.analyzer._calculate_data_hash(modified_data)
        
        self.assertNotEqual(hash1, hash3)


class TestTFIDFShapExplainer(unittest.TestCase):
    """Test cases for the TFIDFShapExplainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_resumes = [
            {
                "skills": ["python", "machine learning"],
                "education": [{"degree": "Bachelor's", "field": "CS"}],
                "experience": [{"title": "Developer", "description": "Software development"}]
            }
        ]
        
        self.sample_job = {
            "title": "Software Engineer",
            "description": "Python development",
            "requirements": {"skills": ["python"], "experience": "2+ years"}
        }
        
        self.model = ResumeScreeningLLM(vocab_size=50)
        self.model.train(self.sample_resumes, [self.sample_job])
        
        # Create background data
        background_vectors = np.random.rand(10, 50)
        self.explainer = TFIDFShapExplainer(self.model, background_vectors)
    
    def test_explainer_initialization(self):
        """Test explainer initialization."""
        self.assertEqual(self.explainer.model, self.model)
        self.assertIsInstance(self.explainer.expected_value, float)
    
    def test_shap_value_calculation(self):
        """Test SHAP value calculation."""
        resume_text = "python machine learning software development"
        job_text = "python software engineer development"
        
        explanation = self.explainer.explain_instance(resume_text, job_text)
        
        self.assertIsInstance(explanation, ShapExplanation)
        self.assertIsInstance(explanation.shap_values, list)
        self.assertIsInstance(explanation.prediction_value, float)
        self.assertIsInstance(explanation.base_value, float)
        
        # SHAP values should sum approximately to prediction - baseline
        shap_sum = sum(explanation.shap_values)
        expected_sum = explanation.prediction_value - explanation.base_value
        self.assertAlmostEqual(shap_sum, expected_sum, places=3)
    
    def test_empty_text_handling(self):
        """Test handling of empty or invalid text."""
        explanation = self.explainer.explain_instance("", "python developer")
        self.assertIsInstance(explanation, ShapExplanation)
        
        # Should handle gracefully without errors
        explanation = self.explainer.explain_instance("python", "")
        self.assertIsInstance(explanation, ShapExplanation)
    
    def test_global_explanation(self):
        """Test global explanation calculation."""
        explanations = []
        for _ in range(5):
            explanation = self.explainer.explain_instance(
                "python machine learning", "software engineer python"
            )
            explanations.append(explanation)
        
        global_importance = self.explainer.explain_global(explanations)
        
        self.assertIsInstance(global_importance, dict)
        for feature, importance in global_importance.items():
            self.assertIsInstance(importance, float)
            self.assertGreaterEqual(importance, 0.0)


class TestForensicShapLogger(unittest.TestCase):
    """Test cases for the ForensicShapLogger class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.logger = ForensicShapLogger(self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        self.assertIsNotNone(self.logger.logger)
        self.assertTrue(os.path.exists(self.test_dir))
        
        # Check that log files are created
        log_files = list(Path(self.test_dir).glob("*.log"))
        self.assertGreater(len(log_files), 0)
    
    def test_analysis_logging(self):
        """Test analysis operation logging."""
        test_params = {"test_param": "value", "number": 42}
        self.logger.log_analysis_start("test_analysis", test_params)
        
        # Should not raise any exceptions
        self.logger.log_data_integrity("test_data", "hash123", 100)
    
    def test_explanation_logging(self):
        """Test explanation logging."""
        explanation = ShapExplanation(
            explanation_id="test_id",
            timestamp="2025-07-01T10:00:00",
            data_hash="test_hash",
            model_version="1.0.0",
            prediction_value=0.85,
            base_value=0.5,
            shap_values=[0.1, 0.2, 0.05],
            feature_names=["feature1", "feature2", "feature3"],
            feature_values=[1.0, 0.5, 0.3],
            expected_value=0.5,
            explanation_type="individual",
            demographic_group=None,
            sample_metadata={}
        )
        
        self.logger.log_explanation_generated(explanation)
        
        # Verify log entry was created
        log_files = list(Path(self.test_dir).glob("*.log"))
        self.assertGreater(len(log_files), 0)


class TestShapVisualizationGeneration(unittest.TestCase):
    """Test cases for SHAP visualization generation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Sample data
        self.sample_resumes = [
            {
                "skills": ["python", "machine learning"],
                "education": [{"degree": "Bachelor's", "field": "CS"}],
                "experience": [{"title": "Developer", "description": "Software"}]
            }
        ]
        
        self.sample_job = {
            "title": "Engineer",
            "description": "Python development",
            "requirements": {"skills": ["python"], "experience": "2+ years"}
        }
        
        # Initialize model and analyzer
        self.model = ResumeScreeningLLM(vocab_size=50)
        self.model.train(self.sample_resumes, [self.sample_job])
        
        self.analyzer = ShapAnalyzer(self.model, output_dir=self.test_dir)
        self.analyzer.initialize_explainer(background_data=self.sample_resumes)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.close')
    def test_waterfall_plot_generation(self, mock_close, mock_figure, mock_savefig):
        """Test waterfall plot generation."""
        explanations = self.analyzer.explain_predictions(
            self.sample_resumes, self.sample_job
        )
        
        plot_path = self.analyzer.create_waterfall_plot(explanations[0])
        
        # Verify matplotlib functions were called
        mock_figure.assert_called()
        mock_savefig.assert_called()
        mock_close.assert_called()
        
        self.assertIsInstance(plot_path, str)
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.close')
    def test_summary_plot_generation(self, mock_close, mock_figure, mock_savefig):
        """Test summary plot generation."""
        explanations = self.analyzer.explain_predictions(
            self.sample_resumes, self.sample_job
        )
        
        plot_path = self.analyzer.create_summary_plot(explanations)
        
        # Verify matplotlib functions were called
        mock_figure.assert_called()
        mock_savefig.assert_called()
        mock_close.assert_called()
        
        self.assertIsInstance(plot_path, str)
    
    @patch('seaborn.heatmap')
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.close')
    def test_demographic_comparison_plot(self, mock_close, mock_figure, mock_savefig, mock_heatmap):
        """Test demographic comparison plot generation."""
        # Create mock demographic analysis
        demographic_analysis = {
            "group1": {"feature1": 0.5, "feature2": 0.3},
            "group2": {"feature1": 0.4, "feature2": 0.6}
        }
        
        plot_path = self.analyzer.create_demographic_comparison_plot(demographic_analysis)
        
        # Verify functions were called
        mock_figure.assert_called()
        mock_heatmap.assert_called()
        mock_savefig.assert_called()
        mock_close.assert_called()
        
        self.assertIsInstance(plot_path, str)


class TestErrorHandling(unittest.TestCase):
    """Test cases for error handling and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        # Minimal model setup
        self.model = ResumeScreeningLLM(vocab_size=10)
        self.analyzer = ShapAnalyzer(self.model, output_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_untrained_model_error(self):
        """Test error handling for untrained model."""
        with self.assertRaises(ValueError):
            self.analyzer.explain_predictions([], {})
    
    def test_uninitialized_explainer_error(self):
        """Test error handling for uninitialized explainer."""
        # Train model but don't initialize explainer
        sample_data = [{"skills": ["python"], "education": [], "experience": []}]
        sample_job = {"title": "Engineer", "description": "Python", "requirements": {}}
        self.model.train(sample_data, [sample_job])
        
        with self.assertRaises(ValueError):
            self.analyzer.explain_predictions(sample_data, sample_job)
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets."""
        sample_data = [{"skills": ["python"], "education": [], "experience": []}]
        sample_job = {"title": "Engineer", "description": "Python", "requirements": {}}
        self.model.train(sample_data, [sample_job])
        
        self.analyzer.initialize_explainer(background_data=sample_data)
        
        # Empty test data should return empty explanations
        explanations = self.analyzer.explain_predictions([], sample_job)
        self.assertEqual(len(explanations), 0)
        
        # Empty global importance should return empty dict
        global_importance = self.analyzer.analyze_global_importance([])
        self.assertEqual(len(global_importance), 0)


class TestDataIntegrityAndSecurity(unittest.TestCase):
    """Test cases for data integrity and security features."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        
        self.sample_data = [
            {"skills": ["python"], "education": [], "experience": [], "id": "test1"},
            {"skills": ["java"], "education": [], "experience": [], "id": "test2"}
        ]
        
        self.model = ResumeScreeningLLM(vocab_size=20)
        self.model.train(self.sample_data, [{"title": "Engineer", "description": "Code", "requirements": {}}])
        
        self.analyzer = ShapAnalyzer(self.model, output_dir=self.test_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_timestamp_generation(self):
        """Test timestamp generation for forensic records."""
        timestamp1 = self.analyzer._get_timestamp()
        timestamp2 = self.analyzer._get_timestamp()
        
        self.assertIsInstance(timestamp1, str)
        self.assertIsInstance(timestamp2, str)
        
        # Timestamps should be in ISO format
        from datetime import datetime
        datetime.fromisoformat(timestamp1.replace('Z', '+00:00') if timestamp1.endswith('Z') else timestamp1)
        datetime.fromisoformat(timestamp2.replace('Z', '+00:00') if timestamp2.endswith('Z') else timestamp2)
    
    def test_hash_consistency(self):
        """Test hash consistency for data integrity."""
        data_df = pd.DataFrame(self.sample_data)
        
        hash1 = self.analyzer._calculate_data_hash(data_df)
        hash2 = self.analyzer._calculate_data_hash(data_df)
        
        self.assertEqual(hash1, hash2)  # Same data, same hash
        
        # Modify data and verify hash changes
        modified_df = data_df.copy()
        modified_df.loc[0, 'id'] = 'modified'
        hash3 = self.analyzer._calculate_data_hash(modified_df)
        
        self.assertNotEqual(hash1, hash3)  # Different data, different hash
    
    def test_forensic_metadata_generation(self):
        """Test generation of forensic metadata."""
        self.analyzer.initialize_explainer(background_data=self.sample_data)
        
        analysis_result = self.analyzer.generate_comprehensive_analysis(
            test_data=self.sample_data,
            job_posting={"title": "Test", "description": "Test job", "requirements": {}},
            background_data=self.sample_data[:1]
        )
        
        # Verify forensic metadata
        forensic_metadata = analysis_result.forensic_metadata
        self.assertIn("analyzer_version", forensic_metadata)
        self.assertIn("data_integrity_verified", forensic_metadata)
        self.assertIn("chain_of_custody_maintained", forensic_metadata)
        
        self.assertTrue(forensic_metadata["data_integrity_verified"])
        self.assertTrue(forensic_metadata["chain_of_custody_maintained"])


def run_comprehensive_tests():
    """Run all test suites and generate comprehensive report."""
    print("Running Comprehensive SHAP Analyzer Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestShapAnalyzer,
        TestTFIDFShapExplainer,
        TestForensicShapLogger,
        TestShapVisualizationGeneration,
        TestErrorHandling,
        TestDataIntegrityAndSecurity
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestClass(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Generate summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures ({len(result.failures)}):")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nErrors ({len(result.errors)}):")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Run comprehensive test suite
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)