#!/usr/bin/env python3
"""
Example Usage of the Forensic Testing Suite
==========================================

This script demonstrates how to use the comprehensive forensic testing suite
for resume screening AI systems. It includes examples for individual components
and the complete test runner.

Author: Forensic AI Testing Suite
Created: 2025-07-01
Purpose: Demonstration and documentation of forensic testing capabilities
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bias_analyzer import BiasAnalyzer
from performance_tester import PerformanceTester
from automated_prompt_tester import AutomatedPromptTester
from log_analyzer import LogAnalyzer
from test_runner import TestRunner, TestConfiguration


def create_sample_data():
    """Create sample data for demonstration purposes."""
    print("Creating sample data for demonstration...")
    
    # Create sample directory
    sample_dir = Path("./sample_data")
    sample_dir.mkdir(exist_ok=True)
    
    # Generate sample training data
    np.random.seed(42)
    n_samples = 1000
    
    # Demographic information
    genders = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
    ages = np.random.normal(35, 10, n_samples).astype(int)
    ages = np.clip(ages, 22, 65)
    
    races = np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples, 
                           p=[0.6, 0.15, 0.15, 0.1])
    
    educations = np.random.choice(['Harvard', 'MIT', 'Stanford', 'State University', 
                                 'Community College'], n_samples, 
                                p=[0.1, 0.1, 0.1, 0.6, 0.1])
    
    # Generate biased scores (for demonstration)
    base_scores = np.random.normal(0.7, 0.15, n_samples)
    
    # Add bias effects
    gender_bias = np.where(genders == 'Male', 0.05, -0.05)
    age_bias = (ages - 35) * -0.002  # Slight age bias
    race_bias = np.where(races == 'White', 0.03, 
                np.where(races == 'Asian', 0.01, -0.03))
    education_bias = np.where(educations.isin(['Harvard', 'MIT', 'Stanford']), 0.1, 
                            np.where(educations == 'Community College', -0.1, 0))
    
    scores = base_scores + gender_bias + age_bias + race_bias + education_bias
    scores = np.clip(scores, 0, 1)
    
    # Create training data
    training_data = pd.DataFrame({
        'candidate_id': range(1, n_samples + 1),
        'gender': genders,
        'age': ages,
        'race': races,
        'education': educations,
        'score': scores
    })
    
    training_data.to_csv(sample_dir / "training_data.csv", index=False)
    
    # Generate sample test data
    y_true = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    y_pred = np.random.choice([0, 1], n_samples, p=[0.65, 0.35])
    
    # Add some bias to predictions
    for i, (gender, race) in enumerate(zip(genders, races)):
        if gender == 'Male' and race == 'White':
            y_pred[i] = np.random.choice([0, 1], p=[0.6, 0.4])  # Higher acceptance
        elif gender == 'Female' and race in ['Black', 'Hispanic']:
            y_pred[i] = np.random.choice([0, 1], p=[0.8, 0.2])  # Lower acceptance
    
    test_data = pd.DataFrame({
        'candidate_id': range(1, n_samples + 1),
        'gender': genders,
        'age': ages,
        'race': races,
        'education': educations,
        'score': scores,
        'y_true': y_true,
        'y_pred': y_pred,
        'group': genders  # Use gender as the primary grouping variable
    })
    
    test_data.to_csv(sample_dir / "test_data.csv", index=False)
    
    # Create sample log files
    logs_dir = Path("./sample_logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Generate sample log entries
    log_entries = []
    for i in range(500):
        timestamp = f"2025-07-{i%30+1:02d} {10 + i%12}:{i%60:02d}:{i%60:02d}.{i%1000:03d}"
        candidate_id = np.random.randint(1, n_samples+1)
        decision = np.random.choice(['accept', 'reject'], p=[0.3, 0.7])
        confidence = np.random.uniform(0.5, 0.95)
        
        log_entry = f"{timestamp}|INFO|decision_engine:45|candidate_id:{candidate_id}|decision:{decision}|confidence:{confidence:.3f}"
        
        # Add some entries with bias indicators
        if i % 20 == 0:
            bias_keywords = np.random.choice(['gender', 'age', 'race'])
            log_entry += f"|{bias_keywords}_keywords detected"
        
        log_entries.append(log_entry)
    
    with open(logs_dir / "model_decisions.log", 'w') as f:
        f.write('\n'.join(log_entries))
    
    print(f"Sample data created in {sample_dir}")
    print(f"Sample logs created in {logs_dir}")
    return str(sample_dir), str(logs_dir)


def example_model_interface(prompt: str) -> str:
    """
    Example model interface for prompt testing.
    In practice, this would call your actual model.
    """
    # Simulate a biased model response
    prompt_lower = prompt.lower()
    
    # Simulate gender bias
    if any(name in prompt_lower for name in ['john', 'michael', 'david', 'james']):
        responses = [
            "This candidate shows strong technical skills and leadership potential.",
            "Excellent background with impressive experience.",
            "Highly qualified candidate with strong communication skills."
        ]
    elif any(name in prompt_lower for name in ['sarah', 'jennifer', 'lisa', 'karen']):
        responses = [
            "This candidate has good skills but may need additional support.",
            "Qualified candidate with adequate experience.",
            "Shows potential but cultural fit may be a concern."
        ]
    else:
        responses = [
            "Standard candidate with typical qualifications.",
            "Meets basic requirements for the position.",
            "Average candidate with standard background."
        ]
    
    # Add some randomness
    import random
    return random.choice(responses)


def example_individual_components():
    """Demonstrate usage of individual components."""
    print("\n" + "="*60)
    print("DEMONSTRATING INDIVIDUAL COMPONENTS")
    print("="*60)
    
    # Create sample data
    sample_dir, logs_dir = create_sample_data()
    
    # 1. Bias Analysis Example
    print("\n1. BIAS ANALYSIS EXAMPLE")
    print("-" * 40)
    
    analyzer = BiasAnalyzer("./forensic_output/bias_demo")
    data = pd.read_csv(f"{sample_dir}/training_data.csv")
    
    print(f"Loaded {len(data)} training samples")
    
    # Analyze gender bias
    print("Analyzing gender bias...")
    gender_results = analyzer.analyze_gender_bias(data, 'score', 'gender')
    print(f"Gender bias tests completed: {len(gender_results)} results")
    
    for result in gender_results[:2]:  # Show first 2 results
        print(f"  - {result.demographic_group}: p={result.p_value:.4f}, "
              f"effect_size={result.effect_size:.3f}, bias_detected={result.bias_detected}")
    
    # Analyze age bias
    print("Analyzing age bias...")
    age_results = analyzer.analyze_age_bias(data, 'score', 'age')
    print(f"Age bias tests completed: {len(age_results)} results")
    
    # Generate bias report
    bias_report_path = analyzer.generate_bias_report()
    print(f"Bias analysis report generated: {bias_report_path}")
    
    # 2. Performance Testing Example
    print("\n2. PERFORMANCE TESTING EXAMPLE")
    print("-" * 40)
    
    tester = PerformanceTester("./forensic_output/performance_demo")
    test_data = pd.read_csv(f"{sample_dir}/test_data.csv")
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Test performance across groups
    print("Testing performance across demographic groups...")
    performance_results = tester.test_performance_across_groups(
        test_data, 'y_true', 'y_pred', 'gender', 'gender'
    )
    
    print(f"Performance tests completed: {len(performance_results)} groups")
    for result in performance_results:
        print(f"  - {result.group_name}: accuracy={result.accuracy:.3f}, "
              f"precision={result.precision:.3f}, recall={result.recall:.3f}")
    
    # Test fairness metrics
    print("Testing fairness metrics...")
    fairness_result = tester.test_fairness_metrics(
        test_data, 'y_true', 'y_pred', 'gender', 'gender'
    )
    
    print(f"Fairness analysis completed:")
    print(f"  - Demographic parity: {fairness_result.demographic_parity:.3f}")
    print(f"  - Equalized odds: {fairness_result.equalized_odds:.3f}")
    print(f"  - Bias detected: {fairness_result.bias_detected}")
    print(f"  - Severity: {fairness_result.severity_level}")
    
    # Generate performance report
    performance_report_path = tester.generate_performance_disparity_report()
    print(f"Performance analysis report generated: {performance_report_path}")
    
    # 3. Prompt Testing Example
    print("\n3. PROMPT TESTING EXAMPLE")
    print("-" * 40)
    
    prompt_tester = AutomatedPromptTester(
        example_model_interface, 
        "./forensic_output/prompt_demo"
    )
    
    # Test gender bias in prompts
    print("Testing gender bias in prompt responses...")
    gender_prompt_result = prompt_tester.test_gender_bias_prompts(num_iterations=5)
    
    print(f"Gender bias prompt testing completed:")
    print(f"  - Bias detected: {gender_prompt_result.bias_detected}")
    print(f"  - Statistical significance: {gender_prompt_result.statistical_significance:.4f}")
    print(f"  - Bias strength: {gender_prompt_result.bias_strength:.3f}")
    
    # Test consistency
    print("Testing response consistency...")
    sample_resume = "Software Engineer with 5 years of experience in Python and machine learning."
    consistency_result = prompt_tester.test_consistency_prompts(sample_resume, 5)
    
    print(f"Consistency testing completed:")
    print(f"  - Consistency score: {consistency_result.consistency_score:.3f}")
    print(f"  - Bias indicators: {len(consistency_result.bias_indicators)}")
    
    # Generate prompt testing report
    prompt_report_path = prompt_tester.generate_prompt_testing_report()
    print(f"Prompt testing report generated: {prompt_report_path}")
    
    # 4. Log Analysis Example
    print("\n4. LOG ANALYSIS EXAMPLE")
    print("-" * 40)
    
    log_analyzer = LogAnalyzer("./forensic_output/log_demo")
    
    # Analyze log files
    log_files = [f"{logs_dir}/model_decisions.log"]
    print(f"Analyzing log files: {log_files}")
    
    log_analyzer.analyze_log_files(log_files)
    
    print(f"Log analysis completed:")
    print(f"  - Total log entries: {len(log_analyzer.log_entries)}")
    print(f"  - Bias patterns detected: {len(log_analyzer.bias_patterns)}")
    print(f"  - Decision patterns analyzed: {len(log_analyzer.decision_patterns)}")
    
    for pattern in log_analyzer.bias_patterns[:2]:  # Show first 2 patterns
        print(f"  - {pattern.pattern_type}: frequency={pattern.frequency}, "
              f"significance={pattern.statistical_significance:.4f}")
    
    # Generate log analysis report
    log_report_path = log_analyzer.generate_log_analysis_report()
    print(f"Log analysis report generated: {log_report_path}")


def example_comprehensive_testing():
    """Demonstrate comprehensive testing with TestRunner."""
    print("\n" + "="*60)
    print("DEMONSTRATING COMPREHENSIVE TESTING SUITE")
    print("="*60)
    
    # Create sample data
    sample_dir, logs_dir = create_sample_data()
    
    # Configure comprehensive testing
    config = TestConfiguration(
        enable_bias_analysis=True,
        enable_performance_testing=True,
        enable_prompt_testing=True,
        enable_log_analysis=True,
        training_data_path=f"{sample_dir}/training_data.csv",
        test_data_path=f"{sample_dir}/test_data.csv",
        log_files_paths=[f"{logs_dir}/model_decisions.log"],
        model_interface=example_model_interface,
        bias_analysis_iterations=50,
        performance_test_iterations=25,
        prompt_test_iterations=10,
        output_directory="./forensic_output/comprehensive_demo",
        max_workers=2,  # Use 2 workers for demo
        timeout_minutes=10  # Short timeout for demo
    )
    
    print("Configuration:")
    print(f"  - Training data: {config.training_data_path}")
    print(f"  - Test data: {config.test_data_path}")
    print(f"  - Log files: {config.log_files_paths}")
    print(f"  - Output directory: {config.output_directory}")
    
    # Run comprehensive testing
    print("\nStarting comprehensive forensic testing...")
    runner = TestRunner(config)
    
    try:
        results = runner.run_all_tests()
        
        print("\nCOMPREHENSIVE TESTING COMPLETED")
        print("-" * 40)
        print(f"Suite ID: {results.suite_id}")
        print(f"Overall status: {results.overall_status}")
        print(f"Bias detected: {results.bias_detected}")
        print(f"Highest severity: {results.highest_severity}")
        print(f"Total tests executed: {len(results.test_results)}")
        
        print("\nIndividual test results:")
        for result in results.test_results:
            print(f"  - {result.test_name}: {result.status}")
            print(f"    Duration: {result.duration_seconds:.2f}s")
            print(f"    Bias detected: {result.bias_detected}")
            print(f"    Severity: {result.severity_level}")
            if result.key_findings:
                print(f"    Key findings: {result.key_findings[:2]}")  # Show first 2
        
        print(f"\nComprehensive report generated: {results.comprehensive_report_path}")
        print(f"Artifacts directory: {results.artifact_directory}")
        
        # Show critical findings
        if results.critical_findings:
            print("\nCRITICAL FINDINGS:")
            for finding in results.critical_findings[:5]:  # Show first 5
                print(f"  - {finding}")
        
        print("\nChain of custody entries:")
        for entry in results.chain_of_custody[-3:]:  # Show last 3 entries
            print(f"  - {entry['timestamp']}: {entry['action']} - {entry.get('file_path', 'N/A')}")
        
    except Exception as e:
        print(f"Error during comprehensive testing: {e}")
        import traceback
        traceback.print_exc()


def example_report_analysis():
    """Demonstrate how to analyze generated reports."""
    print("\n" + "="*60)
    print("DEMONSTRATING REPORT ANALYSIS")
    print("="*60)
    
    # Look for existing reports
    output_dir = Path("./forensic_output")
    
    if not output_dir.exists():
        print("No forensic output directory found. Run the examples first.")
        return
    
    # Find comprehensive reports
    report_files = list(output_dir.rglob("comprehensive_forensic_report_*.json"))
    
    if not report_files:
        print("No comprehensive reports found. Run the comprehensive testing example first.")
        return
    
    # Analyze the most recent report
    latest_report = max(report_files, key=os.path.getctime)
    print(f"Analyzing report: {latest_report}")
    
    try:
        import json
        with open(latest_report, 'r') as f:
            report_data = json.load(f)
        
        print("\nREPORT SUMMARY:")
        print("-" * 20)
        
        # Executive summary
        exec_summary = report_data.get('executive_summary', {})
        overall_assessment = exec_summary.get('overall_assessment', {})
        
        print(f"Bias detected: {overall_assessment.get('bias_detected', 'Unknown')}")
        print(f"Total tests: {overall_assessment.get('total_tests_executed', 'Unknown')}")
        print(f"Successful tests: {overall_assessment.get('successful_tests', 'Unknown')}")
        print(f"Failed tests: {overall_assessment.get('failed_tests', 'Unknown')}")
        
        # Severity breakdown
        severity_breakdown = exec_summary.get('severity_breakdown', {})
        print(f"\nSeverity breakdown:")
        for severity, count in severity_breakdown.items():
            if count > 0:
                print(f"  - {severity}: {count}")
        
        # Key findings
        key_findings = exec_summary.get('key_findings', {})
        critical_issues = key_findings.get('critical_issues', [])
        if critical_issues:
            print(f"\nCritical issues found:")
            for issue_list in critical_issues[:3]:  # Show first 3
                for issue in issue_list[:2]:  # Show first 2 issues per list
                    print(f"  - {issue}")
        
        # Risk assessment
        risk_assessment = report_data.get('risk_assessment', {})
        print(f"\nRisk assessment:")
        print(f"  - Overall risk level: {risk_assessment.get('overall_risk_level', 'Unknown')}")
        print(f"  - Mitigation urgency: {risk_assessment.get('mitigation_urgency', 'Unknown')}")
        
        # Recommendations
        recommendations = report_data.get('recommendations', [])
        if recommendations:
            print(f"\nTop recommendations:")
            for rec in recommendations[:3]:  # Show first 3
                print(f"  - {rec}")
        
        # Legal compliance
        legal_compliance = report_data.get('legal_compliance', {})
        print(f"\nLegal compliance:")
        print(f"  - Chain of custody: {len(legal_compliance.get('chain_of_custody', []))} entries")
        print(f"  - Data integrity: {legal_compliance.get('data_integrity_verification', 'Unknown')}")
        
    except Exception as e:
        print(f"Error analyzing report: {e}")


def main():
    """Main example function."""
    print("FORENSIC TESTING SUITE EXAMPLES")
    print("=" * 60)
    print("This script demonstrates the comprehensive forensic testing suite")
    print("for resume screening AI systems.\n")
    
    # Run examples
    try:
        # Individual components
        example_individual_components()
        
        # Comprehensive testing
        example_comprehensive_testing()
        
        # Report analysis
        example_report_analysis()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nCheck the ./forensic_output directory for generated reports and artifacts.")
        print("All logs, reports, and artifacts include timestamps and integrity verification.")
        print("The comprehensive report contains detailed findings suitable for legal analysis.")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()