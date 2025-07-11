#!/usr/bin/env python3
"""
SHAP Analyzer Example Usage for Resume Screening LLM
====================================================

This script demonstrates how to use the SHAP analyzer for forensic analysis
of the TF-IDF resume screening model. It shows comprehensive explainability
analysis including individual predictions, global feature importance,
demographic bias detection, and forensic documentation.

Author: Forensic AI Testing Suite
Created: 2025-07-01
Purpose: Example implementation of SHAP analysis for forensic testing
"""

import sys
import os
import json
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.model.resume_llm import ResumeScreeningLLM
from forensic.testing.shap_analyzer import ShapAnalyzer
import numpy as np


def load_sample_data():
    """Load sample data for demonstration purposes."""
    
    # Sample resumes with demographic information
    sample_resumes = [
        {
            "id": "resume_001",
            "skills": ["python", "machine learning", "data analysis", "sql"],
            "education": [{"degree": "Bachelor's", "field": "Computer Science"}],
            "experience": [
                {"title": "Data Scientist", "description": "Developed ML models for prediction"},
                {"title": "Software Engineer", "description": "Built web applications"}
            ],
            "years_experience": 5,
            "gender": "female",
            "age": 28,
            "race": "asian"
        },
        {
            "id": "resume_002", 
            "skills": ["java", "spring", "microservices", "aws"],
            "education": [{"degree": "Master's", "field": "Software Engineering"}],
            "experience": [
                {"title": "Senior Engineer", "description": "Led backend development team"},
                {"title": "Cloud Architect", "description": "Designed scalable cloud solutions"}
            ],
            "years_experience": 8,
            "gender": "male",
            "age": 35,
            "race": "white"
        },
        {
            "id": "resume_003",
            "skills": ["react", "javascript", "node.js", "mongodb"],
            "education": [{"degree": "Bachelor's", "field": "Information Technology"}],
            "experience": [
                {"title": "Frontend Developer", "description": "Created user interfaces"},
                {"title": "Full Stack Developer", "description": "End-to-end web development"}
            ],
            "years_experience": 3,
            "gender": "male", 
            "age": 26,
            "race": "hispanic"
        },
        {
            "id": "resume_004",
            "skills": ["python", "tensorflow", "deep learning", "computer vision"],
            "education": [{"degree": "PhD", "field": "Artificial Intelligence"}],
            "experience": [
                {"title": "Research Scientist", "description": "Published AI research papers"},
                {"title": "ML Engineer", "description": "Deployed production ML systems"}
            ],
            "years_experience": 6,
            "gender": "female",
            "age": 32,
            "race": "black"
        },
        {
            "id": "resume_005",
            "skills": ["c++", "embedded systems", "robotics", "control systems"],
            "education": [{"degree": "Master's", "field": "Electrical Engineering"}],
            "experience": [
                {"title": "Robotics Engineer", "description": "Designed autonomous systems"},
                {"title": "Systems Engineer", "description": "Hardware-software integration"}
            ],
            "years_experience": 7,
            "gender": "male",
            "age": 29,
            "race": "asian"
        }
    ]
    
    # Sample job posting
    job_posting = {
        "title": "Senior Data Scientist",
        "description": "Looking for experienced data scientist to lead ML initiatives",
        "requirements": {
            "skills": ["python", "machine learning", "tensorflow", "data analysis", "sql"],
            "experience": "5+ years in data science and machine learning"
        }
    }
    
    return sample_resumes, job_posting


def demonstrate_basic_shap_analysis():
    """Demonstrate basic SHAP analysis functionality."""
    print("=" * 80)
    print("BASIC SHAP ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    # Load sample data
    resumes, job_posting = load_sample_data()
    print(f"Loaded {len(resumes)} sample resumes and 1 job posting")
    
    # Initialize and train the model
    print("\n1. Training TF-IDF model...")
    model = ResumeScreeningLLM(vocab_size=1000)
    model.train(resumes, [job_posting])
    print("   Model training completed")
    
    # Initialize SHAP analyzer
    print("\n2. Initializing SHAP analyzer...")
    shap_analyzer = ShapAnalyzer(model, output_dir="./shap_analysis_output")
    
    # Initialize explainer with background data
    background_resumes = resumes[:3]  # Use first 3 resumes as background
    shap_analyzer.initialize_explainer(background_data=background_resumes)
    print("   SHAP explainer initialized with background data")
    
    # Generate individual explanations
    print("\n3. Generating individual SHAP explanations...")
    test_resumes = resumes[3:]  # Use remaining resumes for testing
    explanations = shap_analyzer.explain_predictions(test_resumes, job_posting)
    print(f"   Generated {len(explanations)} individual explanations")
    
    # Analyze global feature importance
    print("\n4. Analyzing global feature importance...")
    global_importance = shap_analyzer.analyze_global_importance(explanations, top_k=10)
    print("   Top 10 most important features:")
    for i, (feature, importance) in enumerate(global_importance.items(), 1):
        print(f"   {i:2d}. {feature:20s}: {importance:.4f}")
    
    # Create visualizations
    print("\n5. Creating visualizations...")
    
    # Summary plot
    summary_plot = shap_analyzer.create_summary_plot(explanations)
    print(f"   Summary plot saved: {summary_plot}")
    
    # Individual waterfall plot
    if explanations:
        waterfall_plot = shap_analyzer.create_waterfall_plot(explanations[0])
        print(f"   Waterfall plot saved: {waterfall_plot}")
    
    return shap_analyzer, explanations


def demonstrate_demographic_analysis():
    """Demonstrate demographic bias analysis using SHAP."""
    print("\n" + "=" * 80)
    print("DEMOGRAPHIC BIAS ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    # Load sample data
    resumes, job_posting = load_sample_data()
    
    # Initialize and train model
    print("\n1. Training model for demographic analysis...")
    model = ResumeScreeningLLM(vocab_size=1000)
    model.train(resumes, [job_posting])
    
    # Initialize SHAP analyzer
    shap_analyzer = ShapAnalyzer(model, output_dir="./shap_demographic_analysis")
    shap_analyzer.initialize_explainer(background_data=resumes[:2])
    
    # Generate explanations with demographic information
    print("\n2. Generating explanations with demographic data...")
    explanations = shap_analyzer.explain_predictions(
        resumes, job_posting, demographic_column="gender"
    )
    
    # Analyze demographic differences
    print("\n3. Analyzing feature importance across demographic groups...")
    demographic_analysis = shap_analyzer.analyze_demographic_differences(explanations)
    
    print("   Feature importance by gender:")
    for gender, importance_dict in demographic_analysis.items():
        print(f"\n   {gender.upper()}:")
        top_features = list(importance_dict.items())[:5]
        for feature, importance in top_features:
            print(f"     {feature:15s}: {importance:.4f}")
    
    # Create demographic comparison plot
    print("\n4. Creating demographic comparison visualization...")
    demo_plot = shap_analyzer.create_demographic_comparison_plot(demographic_analysis)
    print(f"   Demographic comparison plot saved: {demo_plot}")
    
    return demographic_analysis


def demonstrate_comprehensive_analysis():
    """Demonstrate comprehensive SHAP analysis with full reporting."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE SHAP ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    # Load sample data
    resumes, job_posting = load_sample_data()
    
    # Initialize and train model
    print("\n1. Setting up comprehensive analysis...")
    model = ResumeScreeningLLM(vocab_size=1000)
    model.train(resumes, [job_posting])
    
    # Initialize SHAP analyzer
    shap_analyzer = ShapAnalyzer(model, output_dir="./comprehensive_shap_analysis")
    
    # Run comprehensive analysis
    print("\n2. Running comprehensive SHAP analysis...")
    analysis_result = shap_analyzer.generate_comprehensive_analysis(
        test_data=resumes,
        job_posting=job_posting,
        background_data=resumes[:2],
        demographic_column="race"
    )
    
    print(f"   Analysis completed with ID: {analysis_result.analysis_id}")
    print(f"   Generated {len(analysis_result.individual_explanations)} explanations")
    print(f"   Created {len(analysis_result.visualizations_generated)} visualizations")
    
    # Generate interpretability report
    print("\n3. Generating forensic interpretability report...")
    report_path = shap_analyzer.generate_interpretability_report(analysis_result)
    print(f"   Report saved: {report_path}")
    
    # Display key findings
    print("\n4. Key findings from analysis:")
    summary = analysis_result.analysis_summary
    
    print(f"   Total explanations: {summary['total_explanations']}")
    print(f"   Top features: {', '.join(summary['top_features'][:5])}")
    
    if summary['demographic_groups_analyzed']:
        print(f"   Demographic groups: {', '.join(summary['demographic_groups_analyzed'])}")
    
    pred_stats = summary.get('prediction_statistics', {})
    if pred_stats:
        print(f"   Prediction range: {pred_stats['min']:.3f} - {pred_stats['max']:.3f}")
        print(f"   Mean prediction: {pred_stats['mean']:.3f} (±{pred_stats['std']:.3f})")
    
    return analysis_result


def demonstrate_feature_interaction_analysis():
    """Demonstrate feature interaction analysis."""
    print("\n" + "=" * 80)
    print("FEATURE INTERACTION ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    # Load sample data
    resumes, job_posting = load_sample_data()
    
    # Initialize and train model
    print("\n1. Training model for interaction analysis...")
    model = ResumeScreeningLLM(vocab_size=500)  # Smaller vocab for clearer interactions
    model.train(resumes, [job_posting])
    
    # Initialize SHAP analyzer
    shap_analyzer = ShapAnalyzer(model, output_dir="./shap_interaction_analysis")
    shap_analyzer.initialize_explainer(background_data=resumes[:2])
    
    # Generate explanations
    print("\n2. Generating explanations for interaction analysis...")
    explanations = shap_analyzer.explain_predictions(resumes, job_posting)
    
    # Analyze feature interactions
    print("\n3. Analyzing feature interactions...")
    interactions = shap_analyzer.analyze_feature_interactions(explanations, top_k=10)
    
    print("   Top 10 feature interactions:")
    for i, (interaction, strength) in enumerate(interactions.items(), 1):
        print(f"   {i:2d}. {interaction:40s}: {strength:.6f}")
    
    return interactions


def demonstrate_individual_explanation_analysis():
    """Demonstrate detailed analysis of individual explanations."""
    print("\n" + "=" * 80)
    print("INDIVIDUAL EXPLANATION ANALYSIS DEMONSTRATION")
    print("=" * 80)
    
    # Load sample data
    resumes, job_posting = load_sample_data()
    
    # Initialize and train model
    model = ResumeScreeningLLM(vocab_size=1000)
    model.train(resumes, [job_posting])
    
    # Initialize SHAP analyzer
    shap_analyzer = ShapAnalyzer(model, output_dir="./individual_shap_analysis")
    shap_analyzer.initialize_explainer(background_data=resumes[:2])
    
    # Generate explanations
    explanations = shap_analyzer.explain_predictions(resumes, job_posting)
    
    print(f"\n1. Analyzing {len(explanations)} individual explanations...")
    
    for i, explanation in enumerate(explanations):
        print(f"\n   EXPLANATION {i+1} (Resume ID: {explanation.sample_metadata.get('resume_id', 'unknown')})")
        print(f"   Prediction: {explanation.prediction_value:.4f}")
        print(f"   Base value: {explanation.base_value:.4f}")
        
        # Get top contributing features
        shap_values = np.array(explanation.shap_values)
        feature_names = explanation.feature_names
        
        # Sort by absolute SHAP value
        abs_values = np.abs(shap_values)
        top_indices = np.argsort(abs_values)[-5:][::-1]  # Top 5 features
        
        print("   Top contributing features:")
        for idx in top_indices:
            feature = feature_names[idx]
            value = shap_values[idx]
            direction = "↑" if value > 0 else "↓"
            print(f"     {direction} {feature:20s}: {value:+.4f}")
        
        # Create individual waterfall plot
        waterfall_path = shap_analyzer.create_waterfall_plot(
            explanation, 
            output_file=f"./individual_shap_analysis/waterfall_resume_{i+1}.png"
        )
        print(f"   Waterfall plot: {waterfall_path}")


def main():
    """Main function to run all SHAP analysis demonstrations."""
    print("SHAP Analyzer for Resume Screening LLM - Forensic Testing")
    print("=" * 60)
    print("This script demonstrates comprehensive SHAP analysis capabilities")
    print("for explainable AI in resume screening systems.\n")
    
    try:
        # Run all demonstrations
        demonstrate_basic_shap_analysis()
        demonstrate_demographic_analysis()
        demonstrate_feature_interaction_analysis()
        demonstrate_individual_explanation_analysis()
        demonstrate_comprehensive_analysis()
        
        print("\n" + "=" * 80)
        print("ALL SHAP ANALYSIS DEMONSTRATIONS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print("\nGenerated artifacts:")
        print("- Individual SHAP explanations with feature contributions")
        print("- Global feature importance rankings")
        print("- Demographic bias analysis across protected characteristics")
        print("- Feature interaction analysis")
        print("- Comprehensive visualizations (waterfall, summary, comparison plots)")
        print("- Forensic interpretability reports with audit trails")
        print("- Tamper-evident logging for legal compliance")
        
        print("\nOutput directories:")
        print("- ./shap_analysis_output/")
        print("- ./shap_demographic_analysis/")
        print("- ./comprehensive_shap_analysis/")
        print("- ./shap_interaction_analysis/")
        print("- ./individual_shap_analysis/")
        
    except Exception as e:
        print(f"\nERROR during SHAP analysis: {str(e)}")
        print("Please check the error logs and ensure all dependencies are installed.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())