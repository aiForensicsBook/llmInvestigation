#!/usr/bin/env python3
"""
Forensic analysis example for the Resume Screening LLM.
This script demonstrates how to analyze the model for bias and transparency.
"""

import os
import sys
import json
from collections import Counter, defaultdict
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.resume_llm import ResumeScreeningLLM
from src.data.synthetic_data_generator import generate_normal_synthetic_data


class ForensicAnalyzer:
    """Forensic analysis tools for the Resume Screening LLM."""
    
    def __init__(self, model: ResumeScreeningLLM):
        self.model = model
        self.analysis_results = {}
    
    def analyze_vocabulary_bias(self):
        """Analyze vocabulary for potential bias indicators."""
        print("Analyzing vocabulary for bias...")
        
        # Define bias-indicating words (for educational purposes)
        bias_categories = {
            'gender': ['he', 'she', 'him', 'her', 'male', 'female', 'guy', 'girl'],
            'age': ['young', 'old', 'senior', 'junior', 'experienced', 'fresh'],
            'education': ['ivy', 'league', 'prestigious', 'elite', 'top', 'best'],
            'location': ['urban', 'rural', 'city', 'town', 'metropolitan']
        }
        
        vocab_words = set(self.model.vocabulary.keys())
        found_bias_words = {}
        
        for category, words in bias_categories.items():
            found_words = [word for word in words if word in vocab_words]
            if found_words:
                found_bias_words[category] = found_words
        
        self.analysis_results['vocabulary_bias'] = found_bias_words
        
        print("Bias indicators found in vocabulary:")
        for category, words in found_bias_words.items():
            print(f"  {category.capitalize()}: {', '.join(words)}")
        
        return found_bias_words
    
    def analyze_scoring_patterns(self, resumes, job_postings):
        """Analyze scoring patterns for systematic bias."""
        print("\nAnalyzing scoring patterns...")
        
        # Score all resume-job combinations
        all_scores = []
        demographic_scores = defaultdict(list)
        
        for resume in resumes[:20]:  # Limit for demo
            for job in job_postings[:10]:
                result = self.model.score_resume(resume, job)
                score = result['overall_score']
                all_scores.append(score)
                
                # Analyze by mock demographic (based on name patterns for demo)
                # In real forensic analysis, this would use actual demographic data
                name = resume.get('personal_info', {}).get('name', '').lower()
                if any(n in name for n in ['john', 'mike', 'david', 'robert']):
                    demographic_scores['male_pattern'].append(score)
                elif any(n in name for n in ['jane', 'mary', 'sarah', 'lisa']):
                    demographic_scores['female_pattern'].append(score)
        
        # Calculate statistics
        overall_stats = {
            'mean': np.mean(all_scores),
            'std': np.std(all_scores),
            'min': np.min(all_scores),
            'max': np.max(all_scores)
        }
        
        demographic_stats = {}
        for demo, scores in demographic_scores.items():
            if scores:
                demographic_stats[demo] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'count': len(scores)
                }
        
        self.analysis_results['scoring_patterns'] = {
            'overall': overall_stats,
            'demographic': demographic_stats
        }
        
        print(f"Overall scoring statistics:")
        print(f"  Mean: {overall_stats['mean']:.3f}")
        print(f"  Std Dev: {overall_stats['std']:.3f}")
        print(f"  Range: {overall_stats['min']:.3f} - {overall_stats['max']:.3f}")
        
        if demographic_stats:
            print("Demographic pattern analysis:")
            for demo, stats in demographic_stats.items():
                print(f"  {demo}: Mean = {stats['mean']:.3f}, Count = {stats['count']}")
        
        return overall_stats, demographic_stats
    
    def analyze_feature_importance(self):
        """Analyze which features (words) have highest impact."""
        print("\nAnalyzing feature importance...")
        
        # Get top IDF values (words that distinguish documents)
        idf_items = list(self.model.idf_values.items())
        idf_items.sort(key=lambda x: x[1], reverse=True)
        
        top_discriminative_words = idf_items[:20]
        
        self.analysis_results['feature_importance'] = top_discriminative_words
        
        print("Top discriminative words (high IDF values):")
        for word, idf_val in top_discriminative_words:
            print(f"  {word}: {idf_val:.3f}")
        
        return top_discriminative_words
    
    def analyze_model_transparency(self):
        """Analyze model transparency and explainability."""
        print("\nAnalyzing model transparency...")
        
        model_info = self.model.get_model_info()
        
        transparency_metrics = {
            'has_training_history': bool(model_info['training_history']),
            'vocabulary_accessible': bool(self.model.vocabulary),
            'algorithm_type': 'TF-IDF + Cosine Similarity',
            'interpretable': True,  # TF-IDF is inherently interpretable
            'training_data_logged': 'training_data_stats' in model_info['model_metadata'],
            'hyperparameters_accessible': 'hyperparameters' in model_info['model_metadata']
        }
        
        self.analysis_results['transparency'] = transparency_metrics
        
        print("Model transparency assessment:")
        for metric, value in transparency_metrics.items():
            status = "✓" if value else "✗"
            print(f"  {status} {metric.replace('_', ' ').title()}: {value}")
        
        return transparency_metrics
    
    def generate_audit_report(self):
        """Generate a comprehensive audit report."""
        print("\n" + "="*60)
        print("FORENSIC AUDIT REPORT")
        print("="*60)
        
        report = {
            'audit_timestamp': self.model.model_metadata.get('created_at', 'Unknown'),
            'model_version': self.model.model_metadata.get('version', 'Unknown'),
            'analysis_results': self.analysis_results,
            'recommendations': []
        }
        
        # Generate recommendations based on findings
        if 'vocabulary_bias' in self.analysis_results:
            bias_words = self.analysis_results['vocabulary_bias']
            if bias_words:
                report['recommendations'].append(
                    "BIAS ALERT: Potentially biased vocabulary detected. "
                    "Review training data and consider bias mitigation techniques."
                )
        
        if 'scoring_patterns' in self.analysis_results:
            demo_stats = self.analysis_results['scoring_patterns']['demographic']
            if demo_stats and len(demo_stats) > 1:
                means = [stats['mean'] for stats in demo_stats.values()]
                if max(means) - min(means) > 0.1:  # Arbitrary threshold for demo
                    report['recommendations'].append(
                        "DISPARITY ALERT: Significant scoring differences detected "
                        "between demographic groups. Further investigation recommended."
                    )
        
        if not report['recommendations']:
            report['recommendations'].append(
                "No major bias indicators detected in this analysis. "
                "Continue monitoring with larger datasets."
            )
        
        # Save report
        os.makedirs("forensic_reports", exist_ok=True)
        report_file = "forensic_reports/audit_report.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Audit report saved to: {report_file}")
        
        print("\nKey Findings:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
        
        return report


def main():
    print("Resume Screening LLM - Forensic Analysis")
    print("=" * 50)
    
    # Load or generate data
    data_dir = "data/synthetic"
    resumes_file = os.path.join(data_dir, "normal_resumes.json")
    jobs_file = os.path.join(data_dir, "synthetic_job_postings.json")
    
    if not os.path.exists(resumes_file):
        print("Generating synthetic data for analysis...")
        generate_normal_synthetic_data()
    
    with open(resumes_file, 'r') as f:
        resumes = json.load(f)
    with open(jobs_file, 'r') as f:
        job_postings = json.load(f)
    
    # Train or load model
    model_path = "models/resume_llm_latest.pkl"
    
    if os.path.exists(model_path):
        print("Loading existing model...")
        model = ResumeScreeningLLM()
        model.load_model(model_path)
    else:
        print("Training new model for analysis...")
        model = ResumeScreeningLLM(vocab_size=2000)
        model.train(resumes, job_postings)
    
    # Perform forensic analysis
    analyzer = ForensicAnalyzer(model)
    
    # Run all analyses
    analyzer.analyze_vocabulary_bias()
    analyzer.analyze_scoring_patterns(resumes, job_postings)
    analyzer.analyze_feature_importance()
    analyzer.analyze_model_transparency()
    
    # Generate final report
    report = analyzer.generate_audit_report()
    
    print("\nForensic analysis completed!")
    print("Review the audit report for detailed findings.")


if __name__ == "__main__":
    main()