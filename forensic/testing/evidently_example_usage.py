#!/usr/bin/env python3
"""
Evidently Analyzer Example Usage
===============================

This script demonstrates how to use the EvidentlyAnalyzer for comprehensive 
bias detection and monitoring in resume screening systems. It shows various
analysis workflows including data drift detection, model performance monitoring,
bias detection, and automated alert generation.

Author: Forensic AI Testing Suite
Created: 2025-07-01
Purpose: Example usage of advanced bias detection capabilities
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

# Add the forensic testing directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from evidently_analyzer import (
        EvidentlyAnalyzer,
        run_evidently_forensic_analysis,
        create_evidently_test_suite,
        EVIDENTLY_AVAILABLE
    )
except ImportError as e:
    print(f"Error importing Evidently analyzer: {e}")
    print("Please install evidently with: pip install evidently")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_sample_resume_data(n_samples=1000, add_bias=False, add_drift=False):
    """
    Generate synthetic resume screening data for testing.
    
    Args:
        n_samples: Number of samples to generate
        add_bias: Whether to introduce bias in the data
        add_drift: Whether to introduce data drift
        
    Returns:
        DataFrame with synthetic resume data
    """
    np.random.seed(42)
    
    # Base data generation
    base_experience = 5 if not add_drift else 5.5
    base_education = 7 if not add_drift else 7.2
    
    data = {
        'experience_years': np.random.normal(base_experience, 2, n_samples),
        'education_score': np.random.normal(base_education, 1.5, n_samples),
        'skills_match': np.random.uniform(0, 1, n_samples),
        'communication_score': np.random.normal(8, 1.2, n_samples),
        'technical_score': np.random.normal(7.5, 1.8, n_samples),
        'project_count': np.random.poisson(3, n_samples),
        'certification_count': np.random.poisson(2, n_samples),
    }
    
    # Add demographic information
    data['gender'] = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
    data['race'] = np.random.choice(
        ['White', 'Black', 'Hispanic', 'Asian', 'Other'], 
        n_samples, 
        p=[0.6, 0.15, 0.15, 0.08, 0.02]
    )
    data['age_group'] = np.random.choice(
        ['18-30', '31-45', '46-60', '60+'], 
        n_samples,
        p=[0.4, 0.35, 0.2, 0.05]
    )
    
    # Generate target (actual hire decision) and prediction
    # Create a scoring function based on qualifications
    qualification_score = (
        0.3 * (data['experience_years'] / 10) +
        0.2 * (data['education_score'] / 10) +
        0.2 * data['skills_match'] +
        0.1 * (data['communication_score'] / 10) +
        0.1 * (data['technical_score'] / 10) +
        0.05 * np.minimum(data['project_count'] / 5, 1) +
        0.05 * np.minimum(data['certification_count'] / 3, 1)
    )
    
    # Add some noise and convert to binary decision
    noise = np.random.normal(0, 0.1, n_samples)
    hire_probability = qualification_score + noise
    
    data['target'] = (hire_probability > 0.6).astype(int)
    
    # Generate predictions (with potential bias)
    prediction_score = qualification_score + np.random.normal(0, 0.05, n_samples)
    
    if add_bias:
        # Introduce gender bias - favor males
        male_bonus = np.where(data['gender'] == 'Male', 0.1, 0)
        prediction_score += male_bonus
        
        # Introduce age bias - penalize older candidates
        age_penalty = np.where(data['age_group'] == '60+', -0.15, 0)
        prediction_score += age_penalty
        
        # Introduce racial bias
        race_bias = np.select(
            [data['race'] == 'White', data['race'] == 'Asian'], 
            [0.05, 0.03], 
            default=-0.02
        )
        prediction_score += race_bias
    
    data['prediction'] = (prediction_score > 0.6).astype(int)
    data['prediction_score'] = prediction_score
    
    return pd.DataFrame(data)


def example_comprehensive_analysis():
    """
    Example of running comprehensive bias analysis using Evidently.
    """
    logger.info("=== Comprehensive Bias Analysis Example ===")
    
    if not EVIDENTLY_AVAILABLE:
        logger.error("Evidently library not available. Please install with: pip install evidently")
        return
    
    # Generate reference data (training data)
    logger.info("Generating reference dataset (training data)...")
    reference_data = generate_sample_resume_data(n_samples=2000, add_bias=False, add_drift=False)
    
    # Generate current data (production data with bias and drift)
    logger.info("Generating current dataset (production data with bias and drift)...")
    current_data = generate_sample_resume_data(n_samples=1500, add_bias=True, add_drift=True)
    
    # Run comprehensive analysis
    logger.info("Running comprehensive Evidently analysis...")
    
    result = run_evidently_forensic_analysis(
        reference_data=reference_data,
        current_data=current_data,
        case_id="RESUME_BIAS_CASE_001",
        investigator="Senior Forensic Analyst",
        protected_attributes=['gender', 'race', 'age_group'],
        output_dir="./forensic/reports/evidently_analysis"
    )
    
    # Display results
    logger.info("\n" + "="*60)
    logger.info("ANALYSIS RESULTS SUMMARY")
    logger.info("="*60)
    logger.info(f"Case ID: {result.case_id}")
    logger.info(f"Analysis ID: {result.analysis_id}")
    logger.info(f"Investigator: {result.investigator}")
    logger.info(f"Timestamp: {result.timestamp}")
    
    logger.info("\nFINDINGS:")
    logger.info(f"  • Data Drift Detected: {'YES' if result.data_drift_detected else 'NO'}")
    if result.data_drift_detected:
        logger.info(f"    - Drift Score: {result.drift_score:.3f}")
        logger.info(f"    - Drifted Columns: {len(result.drifted_columns)}")
    
    logger.info(f"  • Model Performance Degraded: {'YES' if result.model_performance_degraded else 'NO'}")
    if result.performance_metrics:
        logger.info(f"    - Current Accuracy: {result.performance_metrics.get('accuracy', 'N/A')}")
    
    logger.info(f"  • Bias Detected: {'YES' if result.bias_detected else 'NO'}")
    if result.bias_detected:
        logger.info(f"    - Affected Protected Attributes: {result.protected_attributes}")
        for attr, metrics in result.bias_by_group.items():
            if metrics.get('bias_detected', False):
                logger.info(f"    - {attr}: Bias detected")
    
    logger.info(f"  • Data Quality Issues: {'YES' if result.data_quality_issues else 'NO'}")
    if result.data_quality_issues:
        logger.info(f"    - Issues found in {len(result.data_quality_issues)} columns")
    
    logger.info(f"  • Alerts Triggered: {'YES' if result.alert_triggered else 'NO'}")
    if result.alert_triggered:
        logger.info(f"    - Total Alerts: {result.alert_details.get('alerts_count', 0)}")
    
    logger.info("\nREPORTS GENERATED:")
    if result.html_report_path:
        logger.info(f"  • HTML Report: {result.html_report_path}")
    if result.json_report_path:
        logger.info(f"  • JSON Report: {result.json_report_path}")
    
    logger.info("\nCOMPLIANCE STATUS:")
    for standard, compliant in result.compliance_status.items():
        status = "COMPLIANT" if compliant else "NON-COMPLIANT"
        logger.info(f"  • {standard}: {status}")
    
    logger.info("="*60)
    
    return result


def example_step_by_step_analysis():
    """
    Example of step-by-step analysis using individual methods.
    """
    logger.info("=== Step-by-Step Analysis Example ===")
    
    if not EVIDENTLY_AVAILABLE:
        logger.error("Evidently library not available. Please install with: pip install evidently")
        return
    
    # Generate test data
    reference_data = generate_sample_resume_data(n_samples=1000, add_bias=False, add_drift=False)
    current_data = generate_sample_resume_data(n_samples=800, add_bias=True, add_drift=True)
    
    # Initialize analyzer
    analyzer = EvidentlyAnalyzer(
        case_id="STEP_BY_STEP_CASE_001",
        investigator="Forensic Analyst",
        output_dir="./forensic/reports/evidently_step_by_step",
        enable_monitoring=True,
        enable_alerts=True
    )
    
    # Configure column mapping
    analyzer.configure_column_mapping(
        target='target',
        prediction='prediction',
        numerical_features=['experience_years', 'education_score', 'skills_match', 
                          'communication_score', 'technical_score', 'project_count', 
                          'certification_count'],
        categorical_features=['gender', 'race', 'age_group'],
        protected_attributes=['gender', 'race', 'age_group']
    )
    
    # Step 1: Data Drift Detection
    logger.info("\nStep 1: Detecting data drift...")
    drift_results = analyzer.detect_data_drift(reference_data, current_data)
    logger.info(f"Data drift detected: {drift_results.get('dataset_drift_detected', False)}")
    logger.info(f"Drift score: {drift_results.get('drift_score', 0):.3f}")
    
    # Step 2: Model Performance Analysis
    logger.info("\nStep 2: Analyzing model performance...")
    performance_results = analyzer.analyze_model_performance(
        reference_data, current_data, task_type='classification'
    )
    logger.info(f"Performance degraded: {performance_results.get('performance_degraded', False)}")
    
    # Step 3: Data Quality Assessment
    logger.info("\nStep 3: Assessing data quality...")
    quality_results = analyzer.assess_data_quality(reference_data, current_data)
    logger.info(f"Quality issues detected: {quality_results.get('quality_issues_detected', False)}")
    
    # Step 4: Bias Detection
    logger.info("\nStep 4: Detecting bias...")
    bias_results = analyzer.detect_bias(
        reference_data, current_data, protected_attributes=['gender', 'race', 'age_group']
    )
    logger.info(f"Bias detected: {bias_results.get('bias_detected', False)}")
    
    # Step 5: Setup Monitoring and Alerts
    logger.info("\nStep 5: Setting up monitoring and alerts...")
    dashboard_path = analyzer.create_monitoring_dashboard()
    logger.info(f"Dashboard configuration saved: {dashboard_path}")
    
    alert_config = analyzer.setup_automated_alerts()
    logger.info(f"Alert configuration created with {len(alert_config)} alert types")
    
    # Check generated alerts
    alerts = analyzer.get_alerts()
    if alerts:
        logger.info(f"\nAlerts generated: {len(alerts)}")
        for alert in alerts:
            logger.info(f"  • {alert.alert_type.upper()}: {alert.description} (Severity: {alert.severity})")
    else:
        logger.info("\nNo alerts generated")
    
    logger.info("\nStep-by-step analysis completed!")


def example_monitoring_workflow():
    """
    Example of setting up continuous monitoring workflow.
    """
    logger.info("=== Monitoring Workflow Example ===")
    
    if not EVIDENTLY_AVAILABLE:
        logger.error("Evidently library not available. Please install with: pip install evidently")
        return
    
    # Initialize analyzer for monitoring
    analyzer = EvidentlyAnalyzer(
        case_id="MONITORING_WORKFLOW_001",
        investigator="Monitoring System",
        output_dir="./forensic/reports/evidently_monitoring",
        enable_monitoring=True,
        enable_alerts=True
    )
    
    # Simulate continuous monitoring with multiple data batches
    reference_data = generate_sample_resume_data(n_samples=2000, add_bias=False, add_drift=False)
    
    logger.info("Simulating continuous monitoring with data batches...")
    
    monitoring_results = []
    
    for batch_num in range(1, 4):
        logger.info(f"\nProcessing batch {batch_num}...")
        
        # Simulate increasing bias and drift over time
        bias_factor = batch_num * 0.3
        drift_factor = batch_num * 0.2
        
        # Generate current batch with increasing issues
        current_batch = generate_sample_resume_data(
            n_samples=500, 
            add_bias=(bias_factor > 0.5), 
            add_drift=(drift_factor > 0.3)
        )
        
        # Configure analyzer for this batch
        analyzer.configure_column_mapping(
            target='target',
            prediction='prediction',
            numerical_features=['experience_years', 'education_score', 'skills_match'],
            categorical_features=['gender', 'race', 'age_group'],
            protected_attributes=['gender', 'race', 'age_group']
        )
        
        # Run quick analysis
        result = analyzer.run_comprehensive_analysis(
            reference_data=reference_data,
            current_data=current_batch,
            task_type='classification',
            protected_attributes=['gender', 'race', 'age_group'],
            generate_reports=False  # Skip reports for monitoring
        )
        
        monitoring_results.append({
            'batch': batch_num,
            'drift_detected': result.data_drift_detected,
            'drift_score': result.drift_score,
            'bias_detected': result.bias_detected,
            'performance_degraded': result.model_performance_degraded,
            'alerts_triggered': result.alert_triggered
        })
        
        logger.info(f"  Batch {batch_num} results:")
        logger.info(f"    - Drift: {'YES' if result.data_drift_detected else 'NO'} (score: {result.drift_score:.3f})")
        logger.info(f"    - Bias: {'YES' if result.bias_detected else 'NO'}")
        logger.info(f"    - Performance issues: {'YES' if result.model_performance_degraded else 'NO'}")
        logger.info(f"    - Alerts: {'YES' if result.alert_triggered else 'NO'}")
    
    # Summary of monitoring results
    logger.info("\n" + "="*50)
    logger.info("MONITORING SUMMARY")
    logger.info("="*50)
    
    for result in monitoring_results:
        status_indicators = []
        if result['drift_detected']:
            status_indicators.append("DRIFT")
        if result['bias_detected']:
            status_indicators.append("BIAS")
        if result['performance_degraded']:
            status_indicators.append("PERF")
        if result['alerts_triggered']:
            status_indicators.append("ALERT")
        
        status = " | ".join(status_indicators) if status_indicators else "OK"
        logger.info(f"Batch {result['batch']}: {status}")
    
    logger.info("="*50)
    logger.info("Monitoring workflow completed!")


def example_custom_thresholds():
    """
    Example of using custom bias detection thresholds.
    """
    logger.info("=== Custom Thresholds Example ===")
    
    if not EVIDENTLY_AVAILABLE:
        logger.error("Evidently library not available. Please install with: pip install evidently")
        return
    
    # Initialize analyzer with custom thresholds
    analyzer = EvidentlyAnalyzer(
        case_id="CUSTOM_THRESHOLDS_001",
        investigator="Threshold Analyst",
        output_dir="./forensic/reports/evidently_custom",
        enable_monitoring=True,
        enable_alerts=True
    )
    
    # Set custom bias detection thresholds (more strict)
    analyzer.bias_thresholds = {
        'demographic_parity_difference': 0.05,  # Default: 0.1
        'equal_opportunity_difference': 0.05,   # Default: 0.1
        'equalized_odds_difference': 0.05,      # Default: 0.1
        'accuracy_difference': 0.02,            # Default: 0.05
        'precision_difference': 0.02,           # Default: 0.05
        'recall_difference': 0.02,              # Default: 0.05
        'f1_difference': 0.02                   # Default: 0.05
    }
    
    # Set custom drift thresholds (more sensitive)
    analyzer.drift_thresholds = {
        'dataset_drift_threshold': 0.3,         # Default: 0.5
        'column_drift_threshold': 0.03,         # Default: 0.05
        'psi_threshold': 0.15,                  # Default: 0.2
        'wasserstein_threshold': 0.08           # Default: 0.1
    }
    
    logger.info("Custom thresholds configured:")
    logger.info(f"  Bias thresholds: {analyzer.bias_thresholds}")
    logger.info(f"  Drift thresholds: {analyzer.drift_thresholds}")
    
    # Generate data with subtle bias/drift that would be caught by strict thresholds
    reference_data = generate_sample_resume_data(n_samples=1000, add_bias=False, add_drift=False)
    
    # Current data with very subtle bias
    current_data = generate_sample_resume_data(n_samples=800, add_bias=False, add_drift=False)
    
    # Manually introduce subtle bias
    male_indices = current_data['gender'] == 'Male'
    female_indices = current_data['gender'] == 'Female'
    
    # Slightly favor males (3% difference in positive predictions)
    male_predictions = current_data.loc[male_indices, 'prediction'].copy()
    female_predictions = current_data.loc[female_indices, 'prediction'].copy()
    
    # Adjust predictions to create 3% difference
    male_boost = np.random.choice([0, 1], size=male_indices.sum(), p=[0.97, 0.03])
    current_data.loc[male_indices, 'prediction'] = np.maximum(male_predictions, male_boost)
    
    # Configure and run analysis
    analyzer.configure_column_mapping(
        target='target',
        prediction='prediction',
        numerical_features=['experience_years', 'education_score', 'skills_match'],
        categorical_features=['gender', 'race', 'age_group'],
        protected_attributes=['gender', 'race', 'age_group']
    )
    
    logger.info("\nRunning analysis with custom thresholds...")
    result = analyzer.run_comprehensive_analysis(
        reference_data=reference_data,
        current_data=current_data,
        task_type='classification',
        protected_attributes=['gender', 'race', 'age_group'],
        generate_reports=True
    )
    
    logger.info(f"\nResults with custom thresholds:")
    logger.info(f"  Bias detected: {'YES' if result.bias_detected else 'NO'}")
    logger.info(f"  Drift detected: {'YES' if result.data_drift_detected else 'NO'}")
    logger.info(f"  Alerts triggered: {'YES' if result.alert_triggered else 'NO'}")
    
    if result.bias_detected:
        logger.info("  ⚠️  Subtle bias detected by strict thresholds!")
    else:
        logger.info("  ✅ No bias detected even with strict thresholds")
    
    logger.info("Custom thresholds example completed!")


def main():
    """
    Main function to run all examples.
    """
    logger.info("Evidently Analyzer Examples")
    logger.info("===========================")
    
    if not EVIDENTLY_AVAILABLE:
        logger.error("Evidently library not available!")
        logger.error("Please install with: pip install evidently")
        return
    
    logger.info("Starting Evidently analyzer examples...")
    
    try:
        # Example 1: Comprehensive analysis
        example_comprehensive_analysis()
        print("\n" + "="*80 + "\n")
        
        # Example 2: Step-by-step analysis
        example_step_by_step_analysis()
        print("\n" + "="*80 + "\n")
        
        # Example 3: Monitoring workflow
        example_monitoring_workflow()
        print("\n" + "="*80 + "\n")
        
        # Example 4: Custom thresholds
        example_custom_thresholds()
        
        logger.info("\n🎉 All examples completed successfully!")
        logger.info("Check the ./forensic/reports/evidently_* directories for generated reports")
        
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")
        raise


if __name__ == "__main__":
    main()