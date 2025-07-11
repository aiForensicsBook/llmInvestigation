#!/usr/bin/env python3
"""
Complete Data Visualization Demo for Resume Screening LLM

This script demonstrates all visualization capabilities of the system.
Run this script to generate comprehensive visualizations and analysis reports.

Usage:
    python examples/run_visualization_demo.py
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.resume_llm import ResumeScreeningLLM
from src.data.synthetic_data_generator import generate_normal_synthetic_data
from src.utils.visualization import (
    plot_score_distributions, 
    plot_feature_importance, 
    plot_bias_analysis,
    plot_training_history,
    export_analysis_report,
    create_comprehensive_visualization_dashboard
)
from examples.forensic_analysis import ForensicAnalyzer

def main():
    print("=" * 60)
    print("RESUME SCREENING LLM - DATA VISUALIZATION DEMO")
    print("=" * 60)
    print()
    
    # Step 1: Setup data and model
    print("Step 1: Setting up data and model...")
    
    # Ensure we have synthetic data
    data_dir = Path("data/synthetic")
    resumes_file = data_dir / "normal_resumes.json"
    jobs_file = data_dir / "synthetic_job_postings.json"
    
    if not resumes_file.exists() or not jobs_file.exists():
        print("Generating synthetic data...")
        generate_normal_synthetic_data()
        print("✓ Synthetic data generated")
    
    # Load data
    print("Loading data...")
    with open(resumes_file, 'r') as f:
        resumes = json.load(f)
    with open(jobs_file, 'r') as f:
        job_postings = json.load(f)
    
    print(f"✓ Loaded {len(resumes)} resumes and {len(job_postings)} job postings")
    
    # Setup model
    model_path = Path("models/resume_llm_latest.pkl")
    
    if model_path.exists():
        print("Loading existing model...")
        model = ResumeScreeningLLM()
        model.load_model(str(model_path))
        print("✓ Model loaded")
    else:
        print("Training new model...")
        model = ResumeScreeningLLM(vocab_size=2000)
        model.train(resumes, job_postings)
        print("✓ Model trained")
    
    # Create output directory
    output_dir = "visualization_output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Output directory created: {output_dir}")
    print()
    
    # Step 2: Generate individual visualizations
    print("Step 2: Generating individual visualizations...")
    
    # 2a. Score Distribution Analysis
    print("  2a. Analyzing score distributions...")
    
    # Generate scores for analysis
    all_scores = []
    demographic_scores = {'male_pattern': [], 'female_pattern': [], 'neutral_pattern': []}
    
    print("     Calculating scores for visualization...")
    sample_size = min(30, len(resumes))  # Limit for demo performance
    job_sample_size = min(5, len(job_postings))
    
    for i, resume in enumerate(resumes[:sample_size]):
        if i % 10 == 0:
            print(f"     Processing resume {i+1}/{sample_size}...")
        
        for job in job_postings[:job_sample_size]:
            result = model.score_resume(resume, job)
            score = result['overall_score']
            all_scores.append(score)
            
            # Categorize by name pattern for demo
            name = resume.get('personal_info', {}).get('name', '').lower()
            if any(n in name for n in ['john', 'mike', 'david', 'robert', 'james', 'william']):
                demographic_scores['male_pattern'].append(score)
            elif any(n in name for n in ['jane', 'mary', 'sarah', 'lisa', 'jennifer', 'patricia']):
                demographic_scores['female_pattern'].append(score)
            else:
                demographic_scores['neutral_pattern'].append(score)
    
    # Filter out empty groups
    demographic_scores = {k: v for k, v in demographic_scores.items() if v}
    
    # Add overall distribution
    scores_data = {'Overall': all_scores}
    scores_data.update(demographic_scores)
    
    plot_score_distributions(
        scores_data,
        title="Resume Scoring Distribution Analysis",
        save_path=os.path.join(output_dir, "score_distributions.png")
    )
    print("     ✓ Score distribution plots generated")
    
    # 2b. Feature Importance Analysis
    print("  2b. Analyzing feature importance...")
    
    if hasattr(model, 'idf_values') and model.idf_values:
        plot_feature_importance(
            model.idf_values,
            title="Top Discriminative Features (IDF Values)",
            top_n=25,
            save_path=os.path.join(output_dir, "feature_importance.png")
        )
        print("     ✓ Feature importance plots generated")
    else:
        print("     ⚠ No IDF values available for feature importance analysis")
    
    # 2c. Bias Analysis
    print("  2c. Generating bias analysis...")
    
    if len(demographic_scores) > 1:
        demo_stats = {}
        for group, scores in demographic_scores.items():
            if scores:  # Only include groups with data
                demo_stats[group] = {
                    'mean': sum(scores) / len(scores),
                    'std': (sum((x - sum(scores)/len(scores))**2 for x in scores) / len(scores))**0.5,
                    'count': len(scores)
                }
        
        if demo_stats:
            plot_bias_analysis(
                demo_stats,
                title="Bias Analysis - Demographic Scoring Patterns",
                save_path=os.path.join(output_dir, "bias_analysis.png")
            )
            print("     ✓ Bias analysis plots generated")
    else:
        print("     ⚠ Insufficient demographic data for bias analysis")
    
    # 2d. Training History
    print("  2d. Analyzing training history...")
    
    if hasattr(model, 'training_history') and model.training_history:
        plot_training_history(
            model.training_history,
            title="Model Training Progress",
            save_path=os.path.join(output_dir, "training_history.png")
        )
        print("     ✓ Training history plots generated")
    else:
        print("     ⚠ No training history available")
    
    print()
    
    # Step 3: Generate comprehensive dashboard
    print("Step 3: Creating comprehensive visualization dashboard...")
    
    dashboard_dir = os.path.join(output_dir, "dashboard")
    create_comprehensive_visualization_dashboard(
        model, resumes, job_postings, dashboard_dir
    )
    print("✓ Comprehensive dashboard created")
    print()
    
    # Step 4: Generate analysis report
    print("Step 4: Generating comprehensive analysis report...")
    
    # Run forensic analysis
    analyzer = ForensicAnalyzer(model)
    analyzer.analyze_vocabulary_bias()
    analyzer.analyze_scoring_patterns(resumes, job_postings)
    analyzer.analyze_feature_importance()
    analyzer.analyze_model_transparency()
    
    # Prepare analysis data for export
    analysis_data = {
        'training_data_size': len(resumes),
        'bias_analysis': analyzer.analysis_results,
        'recommendations': _generate_recommendations(analyzer.analysis_results),
        'technical_details': {
            'model_type': 'TF-IDF + Cosine Similarity',
            'vocabulary_size': len(model.vocabulary) if hasattr(model, 'vocabulary') else 0,
            'sample_scores_analyzed': len(all_scores),
            'demographic_groups_found': len(demographic_scores)
        }
    }
    
    # Export HTML report
    report_path = export_analysis_report(
        model, analysis_data, output_dir, format="html"
    )
    print(f"✓ HTML report generated: {report_path}")
    
    # Export JSON report
    json_report_path = export_analysis_report(
        model, analysis_data, output_dir, format="json"
    )
    print(f"✓ JSON report generated: {json_report_path}")
    
    print()
    
    # Step 5: Summary
    print("=" * 60)
    print("VISUALIZATION DEMO COMPLETED!")
    print("=" * 60)
    print()
    print("Generated files:")
    print(f"📁 Main output directory: {output_dir}/")
    print(f"📊 Score distributions: {output_dir}/score_distributions.png")
    print(f"📊 Feature importance: {output_dir}/feature_importance.png")
    print(f"📊 Bias analysis: {output_dir}/bias_analysis.png")
    print(f"📊 Training history: {output_dir}/training_history.png")
    print(f"📁 Dashboard directory: {dashboard_dir}/")
    print(f"📄 HTML report: {report_path}")
    print(f"📄 JSON report: {json_report_path}")
    print()
    
    # Statistics summary
    print("Analysis Summary:")
    print(f"• Total resumes analyzed: {len(resumes)}")
    print(f"• Total job postings: {len(job_postings)}")
    print(f"• Scores calculated: {len(all_scores)}")
    print(f"• Demographic groups identified: {len(demographic_scores)}")
    
    if demographic_scores:
        print(f"• Group score ranges:")
        for group, scores in demographic_scores.items():
            avg_score = sum(scores) / len(scores)
            min_score = min(scores)
            max_score = max(scores)
            print(f"  - {group}: {avg_score:.3f} (range: {min_score:.3f} - {max_score:.3f})")
    
    print()
    print("To view the visualizations:")
    print("1. Open the PNG files in any image viewer")
    print("2. Open the HTML report in your web browser")
    print("3. Review the JSON report for detailed technical data")
    print()
    print("Next steps:")
    print("• Examine the bias analysis results")
    print("• Review feature importance to understand model decisions")
    print("• Use the comprehensive dashboard for deeper analysis")
    print("• Check the forensic analysis recommendations")


def _generate_recommendations(analysis_results):
    """Generate recommendations based on analysis results."""
    recommendations = []
    
    if 'vocabulary_bias' in analysis_results:
        bias_words = analysis_results['vocabulary_bias']
        if bias_words:
            recommendations.append(
                "BIAS ALERT: Potentially biased vocabulary detected. "
                "Review training data and consider bias mitigation techniques."
            )
    
    if 'scoring_patterns' in analysis_results:
        demo_stats = analysis_results['scoring_patterns'].get('demographic', {})
        if demo_stats and len(demo_stats) > 1:
            means = [stats['mean'] for stats in demo_stats.values()]
            if max(means) - min(means) > 0.1:  # Threshold for significant difference
                recommendations.append(
                    "DISPARITY ALERT: Significant scoring differences detected "
                    "between demographic groups. Further investigation recommended."
                )
    
    if 'transparency' in analysis_results:
        transparency = analysis_results['transparency']
        if not transparency.get('training_data_logged', False):
            recommendations.append(
                "TRANSPARENCY CONCERN: Training data statistics not fully logged. "
                "Consider enhancing data tracking for better auditability."
            )
    
    if not recommendations:
        recommendations.append(
            "BASELINE ASSESSMENT: No major bias indicators detected in this analysis. "
            "Continue monitoring with larger datasets and regular audits."
        )
    
    return recommendations


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during demo: {e}")
        print("Please ensure all dependencies are installed and data is available.")
        sys.exit(1)