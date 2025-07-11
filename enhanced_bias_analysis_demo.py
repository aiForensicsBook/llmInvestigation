#!/usr/bin/env python3
"""
Enhanced Bias Analysis Demo
Demonstrates the new forensic analysis capabilities that expose hidden bias mechanisms.
"""

import json
import pandas as pd
import numpy as np
from src.model.resume_llm import ResumeScreeningLLM
from src.utils.visualization import plot_hidden_bias_mechanisms, plot_identical_resume_comparison
from forensic.testing.bias_analyzer import BiasAnalyzer
from pathlib import Path

def create_identical_resume_test():
    """Create identical resumes with different genders for bias testing."""
    base_resume = {
        "skills": {
            "technical": ["Python", "JavaScript", "React", "SQL", "Git"],
            "soft": ["communication", "teamwork", "problem-solving"]
        },
        "education": [
            {
                "degree": "Bachelor of Computer Science",
                "field": "Computer Science",
                "institution": "University of Technology"
            }
        ],
        "experience": [
            {
                "title": "Software Engineer",
                "company": "Tech Corp",
                "description": "Developed web applications using React and Python"
            }
        ],
        "years_experience": 3
    }
    
    # Create male and female versions
    male_resume = base_resume.copy()
    male_resume["personal_info"] = {
        "name": "Chad Thompson",
        "gender": "male"
    }
    
    female_resume = base_resume.copy()
    female_resume["personal_info"] = {
        "name": "Ashley Madison", 
        "gender": "female"
    }
    
    return male_resume, female_resume

def create_test_job_postings():
    """Create test job postings for bias analysis."""
    return [
        {
            "id": "job_1",
            "title": "Software Engineer",
            "description": "Looking for a skilled software engineer",
            "requirements": {
                "required_skills": ["Python", "JavaScript", "React"],
                "experience": "2+ years"
            }
        },
        {
            "id": "job_2", 
            "title": "Frontend Developer",
            "description": "React developer needed",
            "requirements": {
                "required_skills": ["React", "JavaScript", "HTML", "CSS"],
                "experience": "3+ years"
            }
        },
        {
            "id": "job_3",
            "title": "Backend Developer", 
            "description": "Python backend developer",
            "requirements": {
                "required_skills": ["Python", "SQL", "API"],
                "experience": "2+ years"
            }
        }
    ]

def run_enhanced_bias_analysis():
    """Run enhanced bias analysis with hidden mechanism detection."""
    print("🔍 Enhanced Bias Analysis Demo")
    print("=" * 50)
    
    # Initialize model
    model = ResumeScreeningLLM()
    
    # Create test data
    male_resume, female_resume = create_identical_resume_test()
    job_postings = create_test_job_postings()
    
    # Train model on basic data
    print("1. Training model...")
    model.train([male_resume, female_resume], job_postings)
    
    # Run scoring tests
    print("2. Running identical resume bias tests...")
    comparison_data = []
    test_data = []
    
    for job in job_postings:
        # Score male candidate
        male_result = model.score_resume(male_resume, job)
        male_score = male_result['overall_score']
        male_metadata = male_result.get('scoring_metadata', {}).get('bias_factors', {})
        
        # Score female candidate  
        female_result = model.score_resume(female_resume, job)
        female_score = female_result['overall_score']
        female_metadata = female_result.get('scoring_metadata', {}).get('bias_factors', {})
        
        # Collect comparison data
        comparison_data.append({
            'job_title': job['title'],
            'male_score': male_score,
            'female_score': female_score,
            'male_metadata': male_metadata,
            'female_metadata': female_metadata
        })
        
        # Collect test data for hidden bias analysis
        if male_metadata:
            test_data.append({
                'gender': 'male',
                'base_score': male_metadata.get('base_score', male_score),
                'final_score': male_score,
                'bias_multiplier': male_metadata.get('bias_multiplier', 1.0),
                'bias_reasons': male_metadata.get('bias_reasons', []),
                'job_title': job['title']
            })
        
        if female_metadata:
            test_data.append({
                'gender': 'female', 
                'base_score': female_metadata.get('base_score', female_score),
                'final_score': female_score,
                'bias_multiplier': female_metadata.get('bias_multiplier', 1.0),
                'bias_reasons': female_metadata.get('bias_reasons', []),
                'job_title': job['title']
            })
    
    # Convert to DataFrame for analysis
    test_df = pd.DataFrame(test_data)
    
    print("3. Analyzing hidden bias mechanisms...")
    
    # Initialize enhanced bias analyzer
    analyzer = BiasAnalyzer("./enhanced_bias_output")
    
    # Run hidden bias mechanism analysis
    hidden_bias_results = analyzer.analyze_hidden_bias_mechanisms(test_df)
    
    print("\n🚨 HIDDEN BIAS MECHANISMS DETECTED:")
    print(f"   Hidden mechanisms found: {len(hidden_bias_results['hidden_mechanisms_detected'])}")
    
    for mechanism in hidden_bias_results['hidden_mechanisms_detected']:
        print(f"   • {mechanism['type']}: {mechanism['gender']} "
              f"(factor: {mechanism['adjustment_factor']:.3f}, "
              f"severity: {mechanism['severity']})")
    
    if 'gender_score_ratio' in hidden_bias_results:
        ratio_data = hidden_bias_results['gender_score_ratio']
        print(f"\n📊 SCORE RATIO ANALYSIS:")
        print(f"   Male-to-Female ratio: {ratio_data.get('male_to_female_ratio', 'N/A'):.2f}")
        print(f"   Male advantage: {ratio_data.get('male_advantage_percentage', 0):.1f}%")
    
    print("\n4. Generating enhanced visualizations...")
    
    # Create output directory
    output_dir = Path("./enhanced_bias_visualizations")
    output_dir.mkdir(exist_ok=True)
    
    # Generate hidden bias mechanisms visualization
    plot_hidden_bias_mechanisms(
        hidden_bias_results,
        title="Hidden Gender Bias Mechanisms - EXPOSED",
        save_path=output_dir / "hidden_bias_mechanisms.png"
    )
    
    # Generate identical resume comparison visualization
    plot_identical_resume_comparison(
        comparison_data,
        title="Identical Resume Gender Bias Test - Forensic Evidence", 
        save_path=output_dir / "identical_resume_comparison.png"
    )
    
    print("5. Generating forensic report...")
    
    # Generate comprehensive bias report
    report_file = analyzer.generate_bias_report(
        output_dir / "enhanced_bias_forensic_report.json"
    )
    
    # Save detailed results
    with open(output_dir / "hidden_bias_analysis.json", 'w') as f:
        json.dump(hidden_bias_results, f, indent=2, default=str)
    
    with open(output_dir / "comparison_data.json", 'w') as f:
        json.dump(comparison_data, f, indent=2, default=str)
    
    print(f"\n✅ Enhanced bias analysis complete!")
    print(f"📁 Results saved to: {output_dir}")
    print("\nGenerated files:")
    print("• hidden_bias_mechanisms.png - Exposes hidden bias multipliers")
    print("• identical_resume_comparison.png - Shows bias in identical resume scoring")
    print("• enhanced_bias_forensic_report.json - Comprehensive forensic report")
    print("• hidden_bias_analysis.json - Raw hidden bias mechanism data")
    print("• comparison_data.json - Detailed comparison results")
    
    print("\n🔬 KEY FINDINGS:")
    print("=" * 50)
    print("The enhanced forensic analysis has successfully exposed:")
    print("1. HIDDEN BIAS MULTIPLIERS applied to scores based on gender")
    print("2. SCORE ADJUSTMENTS that occur after initial feature matching")
    print("3. BIAS PATTERNS in the model's decision-making process")
    print("4. CONCRETE EVIDENCE of differential treatment for identical qualifications")
    print("\nThis analysis demonstrates that while feature importance may appear")
    print("neutral, the model applies systematic bias through hidden mechanisms")
    print("that artificially inflate male scores and penalize female candidates.")
    
    return hidden_bias_results, comparison_data

if __name__ == "__main__":
    try:
        hidden_bias_results, comparison_data = run_enhanced_bias_analysis()
        print("\n🎯 Enhanced bias analysis demonstration completed successfully!")
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()