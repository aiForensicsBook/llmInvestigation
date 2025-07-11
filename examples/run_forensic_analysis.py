#!/usr/bin/env python3
"""
Complete Forensic Analysis Demo for Resume Screening LLM

This script runs a comprehensive forensic analysis including:
- Bias detection and measurement
- Transparency assessment
- Fairness evaluation
- Audit report generation

Usage:
    python examples/run_forensic_analysis.py
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.resume_llm import ResumeScreeningLLM
from src.data.synthetic_data_generator import generate_normal_synthetic_data
from examples.forensic_analysis import ForensicAnalyzer

class ComprehensiveForensicAnalyzer(ForensicAnalyzer):
    """Extended forensic analyzer with additional capabilities."""
    
    def __init__(self, model: ResumeScreeningLLM):
        super().__init__(model)
        self.detailed_results = {}
    
    def analyze_fairness_metrics(self, resumes, job_postings):
        """Analyze fairness metrics across different demographic groups."""
        print("\nAnalyzing fairness metrics...")
        
        # Calculate detailed fairness metrics
        group_scores = defaultdict(list)
        group_rankings = defaultdict(list)
        
        # Sample for performance
        sample_resumes = resumes[:100]
        sample_jobs = job_postings[:20]
        
        print(f"Analyzing {len(sample_resumes)} resumes against {len(sample_jobs)} job postings...")
        
        for job_idx, job in enumerate(sample_jobs):
            # Get scores for all resumes for this job
            job_scores = []
            for resume in sample_resumes:
                result = self.model.score_resume(resume, job)
                score = result['overall_score']
                
                # Categorize by demographic proxy (name patterns)
                name = resume.get('personal_info', {}).get('name', '').lower()
                group = self._categorize_by_name(name)
                
                job_scores.append((score, group, resume))
                group_scores[group].append(score)
            
            # Calculate rankings for this job
            job_scores.sort(key=lambda x: x[0], reverse=True)
            for rank, (score, group, resume) in enumerate(job_scores):
                group_rankings[group].append(rank + 1)  # 1-based ranking
        
        # Calculate fairness metrics
        fairness_metrics = {}
        
        # 1. Statistical parity (equal selection rates)
        top_10_percent_threshold = int(len(sample_resumes) * 0.1)
        for group in group_rankings:
            top_rankings = [r for r in group_rankings[group] if r <= top_10_percent_threshold]
            selection_rate = len(top_rankings) / len(group_rankings[group])
            fairness_metrics[f'{group}_selection_rate'] = selection_rate
        
        # 2. Equal opportunity (equal true positive rates)
        # For this demo, we'll use top 25% as "qualified"
        top_25_percent_threshold = int(len(sample_resumes) * 0.25)
        for group in group_rankings:
            qualified_and_selected = len([r for r in group_rankings[group] if r <= top_25_percent_threshold])
            total_qualified = len([r for r in group_rankings[group] if r <= top_25_percent_threshold])
            
            if total_qualified > 0:
                equal_opportunity_rate = qualified_and_selected / total_qualified
                fairness_metrics[f'{group}_equal_opportunity'] = equal_opportunity_rate
        
        # 3. Demographic parity difference
        selection_rates = [fairness_metrics.get(f'{group}_selection_rate', 0) for group in group_rankings]
        if len(selection_rates) > 1:
            demographic_parity_diff = max(selection_rates) - min(selection_rates)
            fairness_metrics['demographic_parity_difference'] = demographic_parity_diff
        
        # 4. Statistical metrics
        for group in group_scores:
            scores = group_scores[group]
            fairness_metrics[f'{group}_mean_score'] = np.mean(scores)
            fairness_metrics[f'{group}_std_score'] = np.std(scores)
            fairness_metrics[f'{group}_median_score'] = np.median(scores)
            fairness_metrics[f'{group}_score_range'] = max(scores) - min(scores)
        
        self.detailed_results['fairness_metrics'] = fairness_metrics
        
        # Print summary
        print("Fairness Analysis Results:")
        print(f"  Groups identified: {list(group_scores.keys())}")
        for group in group_scores:
            print(f"  {group}:")
            print(f"    Sample size: {len(group_scores[group])}")
            print(f"    Mean score: {fairness_metrics[f'{group}_mean_score']:.3f}")
            print(f"    Selection rate (top 10%): {fairness_metrics.get(f'{group}_selection_rate', 0):.3f}")
        
        if 'demographic_parity_difference' in fairness_metrics:
            dpd = fairness_metrics['demographic_parity_difference']
            print(f"  Demographic parity difference: {dpd:.3f}")
            if dpd > 0.2:  # Common threshold
                print("    ⚠ WARNING: Significant demographic parity difference detected!")
            elif dpd > 0.1:
                print("    ⚠ CAUTION: Moderate demographic parity difference detected")
            else:
                print("    ✓ Acceptable demographic parity difference")
        
        return fairness_metrics
    
    def _categorize_by_name(self, name):
        """Categorize by name pattern for demographic analysis."""
        male_names = ['john', 'mike', 'david', 'robert', 'james', 'william', 'richard', 'charles', 'joseph', 'thomas']
        female_names = ['jane', 'mary', 'sarah', 'lisa', 'jennifer', 'patricia', 'linda', 'barbara', 'elizabeth', 'susan']
        
        if any(n in name for n in male_names):
            return 'male_pattern'
        elif any(n in name for n in female_names):
            return 'female_pattern'
        else:
            return 'neutral_pattern'
    
    def analyze_vocabulary_impact(self):
        """Analyze the impact of specific vocabulary on scoring."""
        print("\nAnalyzing vocabulary impact on scoring...")
        
        # Get vocabulary with IDF values
        vocab_impact = {}
        
        if hasattr(self.model, 'idf_values') and self.model.idf_values:
            # Sort by IDF value to find most discriminative words
            sorted_vocab = sorted(self.model.idf_values.items(), key=lambda x: x[1], reverse=True)
            
            # Analyze top impactful words
            top_words = sorted_vocab[:50]
            
            # Categorize words by potential bias
            bias_categories = {
                'gender_related': ['he', 'she', 'him', 'her', 'male', 'female', 'guy', 'girl', 'man', 'woman'],
                'age_related': ['young', 'old', 'senior', 'junior', 'experienced', 'fresh', 'new', 'veteran'],
                'education_related': ['ivy', 'league', 'prestigious', 'elite', 'top', 'best', 'harvard', 'stanford'],
                'location_related': ['urban', 'rural', 'city', 'town', 'metropolitan', 'downtown', 'suburban'],
                'tech_related': ['python', 'java', 'javascript', 'react', 'angular', 'aws', 'cloud', 'ai', 'ml'],
                'business_related': ['management', 'leadership', 'strategy', 'business', 'marketing', 'sales']
            }
            
            categorized_impact = {}
            for category, words in bias_categories.items():
                category_words = []
                for word, idf_val in top_words:
                    if word.lower() in words:
                        category_words.append((word, idf_val))
                
                if category_words:
                    categorized_impact[category] = category_words
                    total_impact = sum(idf_val for _, idf_val in category_words)
                    avg_impact = total_impact / len(category_words)
                    vocab_impact[category] = {
                        'words': category_words,
                        'total_impact': total_impact,
                        'average_impact': avg_impact,
                        'word_count': len(category_words)
                    }
            
            self.detailed_results['vocabulary_impact'] = vocab_impact
            
            # Print analysis
            print("Vocabulary Impact Analysis:")
            for category, data in vocab_impact.items():
                print(f"  {category.replace('_', ' ').title()}:")
                print(f"    Words found: {data['word_count']}")
                print(f"    Average impact: {data['average_impact']:.3f}")
                print(f"    Top words: {', '.join([w for w, _ in data['words'][:5]])}")
                
                if data['average_impact'] > 2.0:  # High impact threshold
                    print(f"    ⚠ HIGH IMPACT: This category has significant influence on scoring")
                elif data['average_impact'] > 1.0:
                    print(f"    📊 MODERATE IMPACT: This category has moderate influence")
        
        return vocab_impact
    
    def analyze_decision_consistency(self, resumes, job_postings):
        """Analyze consistency of decisions across similar profiles."""
        print("\nAnalyzing decision consistency...")
        
        # Find similar resume pairs and compare their scores
        consistency_metrics = {}
        
        # Sample for performance
        sample_resumes = resumes[:50]
        sample_jobs = job_postings[:10]
        
        # Calculate similarity between resumes based on key features
        resume_similarities = []
        
        for i in range(len(sample_resumes)):
            for j in range(i + 1, len(sample_resumes)):
                resume1 = sample_resumes[i]
                resume2 = sample_resumes[j]
                
                # Calculate feature similarity
                similarity = self._calculate_resume_similarity(resume1, resume2)
                
                if similarity > 0.7:  # High similarity threshold
                    # Compare their scores across jobs
                    score_diffs = []
                    for job in sample_jobs:
                        score1 = self.model.score_resume(resume1, job)['overall_score']
                        score2 = self.model.score_resume(resume2, job)['overall_score']
                        score_diffs.append(abs(score1 - score2))
                    
                    avg_score_diff = np.mean(score_diffs)
                    resume_similarities.append({
                        'similarity': similarity,
                        'avg_score_diff': avg_score_diff,
                        'resume1_idx': i,
                        'resume2_idx': j
                    })
        
        if resume_similarities:
            consistency_metrics['similar_pairs_found'] = len(resume_similarities)
            consistency_metrics['avg_score_difference'] = np.mean([s['avg_score_diff'] for s in resume_similarities])
            consistency_metrics['max_score_difference'] = max([s['avg_score_diff'] for s in resume_similarities])
            
            # Consistency score (lower is better)
            consistency_score = 1 - consistency_metrics['avg_score_difference']
            consistency_metrics['consistency_score'] = consistency_score
            
            print("Decision Consistency Analysis:")
            print(f"  Similar resume pairs found: {consistency_metrics['similar_pairs_found']}")
            print(f"  Average score difference: {consistency_metrics['avg_score_difference']:.3f}")
            print(f"  Consistency score: {consistency_score:.3f}")
            
            if consistency_score < 0.7:
                print("  ⚠ WARNING: Low consistency in similar profiles")
            elif consistency_score < 0.85:
                print("  📊 MODERATE: Acceptable consistency")
            else:
                print("  ✓ GOOD: High consistency in similar profiles")
        
        self.detailed_results['consistency_metrics'] = consistency_metrics
        return consistency_metrics
    
    def _calculate_resume_similarity(self, resume1, resume2):
        """Calculate similarity between two resumes."""
        # Simple similarity based on skills and experience
        skills1 = set(resume1.get('skills', []))
        skills2 = set(resume2.get('skills', []))
        
        # Jaccard similarity for skills
        if skills1 or skills2:
            skill_similarity = len(skills1 & skills2) / len(skills1 | skills2)
        else:
            skill_similarity = 0
        
        # Experience similarity
        exp1 = resume1.get('years_experience', 0)
        exp2 = resume2.get('years_experience', 0)
        
        if max(exp1, exp2) > 0:
            exp_similarity = 1 - abs(exp1 - exp2) / max(exp1, exp2)
        else:
            exp_similarity = 1
        
        # Education similarity (simplified)
        edu1 = resume1.get('education', [])
        edu2 = resume2.get('education', [])
        
        if edu1 and edu2:
            # Compare highest degree
            degrees1 = [ed.get('degree', '') for ed in edu1]
            degrees2 = [ed.get('degree', '') for ed in edu2]
            edu_similarity = 1 if any(d in degrees2 for d in degrees1) else 0
        else:
            edu_similarity = 0
        
        # Weighted average
        total_similarity = (skill_similarity * 0.5 + exp_similarity * 0.3 + edu_similarity * 0.2)
        return total_similarity
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive forensic analysis report."""
        print("\n" + "="*80)
        print("COMPREHENSIVE FORENSIC ANALYSIS REPORT")
        print("="*80)
        
        # Combine all analysis results
        comprehensive_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'model_metadata': self.model.model_metadata if hasattr(self.model, 'model_metadata') else {},
            'basic_analysis': self.analysis_results,
            'detailed_analysis': self.detailed_results,
            'risk_assessment': self._generate_risk_assessment(),
            'recommendations': self._generate_detailed_recommendations(),
            'compliance_check': self._generate_compliance_check()
        }
        
        # Save comprehensive report
        os.makedirs("forensic_reports", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"forensic_reports/comprehensive_analysis_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        print(f"Comprehensive report saved to: {report_file}")
        
        # Generate summary
        self._print_executive_summary(comprehensive_results)
        
        return comprehensive_results
    
    def _generate_risk_assessment(self):
        """Generate risk assessment based on analysis results."""
        risk_levels = {
            'bias_risk': 'LOW',
            'fairness_risk': 'LOW',
            'transparency_risk': 'LOW',
            'consistency_risk': 'LOW',
            'overall_risk': 'LOW'
        }
        
        # Assess bias risk
        if 'vocabulary_bias' in self.analysis_results:
            bias_words = self.analysis_results['vocabulary_bias']
            if bias_words:
                risk_levels['bias_risk'] = 'HIGH' if len(bias_words) > 2 else 'MEDIUM'
        
        # Assess fairness risk
        if 'fairness_metrics' in self.detailed_results:
            fairness = self.detailed_results['fairness_metrics']
            if fairness.get('demographic_parity_difference', 0) > 0.2:
                risk_levels['fairness_risk'] = 'HIGH'
            elif fairness.get('demographic_parity_difference', 0) > 0.1:
                risk_levels['fairness_risk'] = 'MEDIUM'
        
        # Assess transparency risk
        if 'transparency' in self.analysis_results:
            transparency = self.analysis_results['transparency']
            if not transparency.get('has_training_history', False):
                risk_levels['transparency_risk'] = 'MEDIUM'
        
        # Assess consistency risk
        if 'consistency_metrics' in self.detailed_results:
            consistency = self.detailed_results['consistency_metrics']
            if consistency.get('consistency_score', 1) < 0.7:
                risk_levels['consistency_risk'] = 'HIGH'
            elif consistency.get('consistency_score', 1) < 0.85:
                risk_levels['consistency_risk'] = 'MEDIUM'
        
        # Overall risk
        high_risks = sum(1 for risk in risk_levels.values() if risk == 'HIGH')
        medium_risks = sum(1 for risk in risk_levels.values() if risk == 'MEDIUM')
        
        if high_risks > 0:
            risk_levels['overall_risk'] = 'HIGH'
        elif medium_risks > 1:
            risk_levels['overall_risk'] = 'MEDIUM'
        
        return risk_levels
    
    def _generate_detailed_recommendations(self):
        """Generate detailed recommendations based on all analyses."""
        recommendations = []
        
        # Bias-related recommendations
        if 'vocabulary_bias' in self.analysis_results:
            bias_words = self.analysis_results['vocabulary_bias']
            if bias_words:
                recommendations.append({
                    'category': 'Bias Mitigation',
                    'priority': 'HIGH',
                    'finding': f"Potentially biased vocabulary detected: {bias_words}",
                    'recommendation': "Review training data for bias sources. Consider vocabulary filtering or reweighting techniques.",
                    'action_items': [
                        "Audit training data for biased language patterns",
                        "Implement bias detection in data preprocessing",
                        "Consider using bias-aware training techniques"
                    ]
                })
        
        # Fairness-related recommendations
        if 'fairness_metrics' in self.detailed_results:
            fairness = self.detailed_results['fairness_metrics']
            if fairness.get('demographic_parity_difference', 0) > 0.1:
                recommendations.append({
                    'category': 'Fairness Improvement',
                    'priority': 'HIGH',
                    'finding': f"Demographic parity difference: {fairness.get('demographic_parity_difference', 0):.3f}",
                    'recommendation': "Implement fairness constraints or post-processing techniques.",
                    'action_items': [
                        "Implement demographic parity constraints",
                        "Consider equalized odds optimization",
                        "Regular fairness monitoring and adjustment"
                    ]
                })
        
        # Consistency recommendations
        if 'consistency_metrics' in self.detailed_results:
            consistency = self.detailed_results['consistency_metrics']
            if consistency.get('consistency_score', 1) < 0.8:
                recommendations.append({
                    'category': 'Model Consistency',
                    'priority': 'MEDIUM',
                    'finding': f"Consistency score: {consistency.get('consistency_score', 1):.3f}",
                    'recommendation': "Improve model consistency through better feature engineering or ensemble methods.",
                    'action_items': [
                        "Review feature engineering pipeline",
                        "Consider ensemble methods for more stable predictions",
                        "Implement consistency testing in model validation"
                    ]
                })
        
        # General recommendations
        recommendations.append({
            'category': 'Ongoing Monitoring',
            'priority': 'MEDIUM',
            'finding': "Model requires continuous monitoring for bias and fairness",
            'recommendation': "Establish regular forensic analysis schedule and monitoring dashboards.",
            'action_items': [
                "Set up automated bias monitoring",
                "Create fairness dashboards for ongoing tracking",
                "Establish regular model audit schedule"
            ]
        })
        
        return recommendations
    
    def _generate_compliance_check(self):
        """Generate compliance check for common regulations."""
        compliance_check = {
            'gdpr_compliance': {
                'explainability': True,  # TF-IDF is explainable
                'data_protection': 'NEEDS_REVIEW',  # Depends on implementation
                'right_to_explanation': True
            },
            'equal_opportunity_compliance': {
                'protected_characteristics': 'NEEDS_REVIEW',
                'disparate_impact': 'ANALYZED',
                'reasonable_adjustments': 'NEEDS_REVIEW'
            },
            'ai_ethics_compliance': {
                'transparency': True,
                'accountability': True,
                'fairness': 'MONITORED',
                'human_oversight': 'NEEDS_REVIEW'
            }
        }
        
        return compliance_check
    
    def _print_executive_summary(self, results):
        """Print executive summary of the analysis."""
        print("\nEXECUTIVE SUMMARY")
        print("-" * 40)
        
        risk_assessment = results['risk_assessment']
        print(f"Overall Risk Level: {risk_assessment['overall_risk']}")
        
        print("\nRisk Breakdown:")
        for risk_type, level in risk_assessment.items():
            if risk_type != 'overall_risk':
                icon = "⚠" if level == 'HIGH' else "📊" if level == 'MEDIUM' else "✓"
                print(f"  {icon} {risk_type.replace('_', ' ').title()}: {level}")
        
        print(f"\nTotal Recommendations: {len(results['recommendations'])}")
        
        high_priority = sum(1 for r in results['recommendations'] if r['priority'] == 'HIGH')
        medium_priority = sum(1 for r in results['recommendations'] if r['priority'] == 'MEDIUM')
        
        print(f"  High Priority: {high_priority}")
        print(f"  Medium Priority: {medium_priority}")
        
        print("\nKey Findings:")
        for i, rec in enumerate(results['recommendations'][:3], 1):
            print(f"{i}. {rec['category']}: {rec['finding']}")


def main():
    print("=" * 80)
    print("RESUME SCREENING LLM - COMPREHENSIVE FORENSIC ANALYSIS")
    print("=" * 80)
    print()
    
    # Setup
    print("Setting up analysis environment...")
    
    # Ensure we have data
    data_dir = Path("data/synthetic")
    resumes_file = data_dir / "normal_resumes.json"
    jobs_file = data_dir / "synthetic_job_postings.json"
    
    if not resumes_file.exists() or not jobs_file.exists():
        print("Generating synthetic data...")
        generate_normal_synthetic_data()
    
    # Load data
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
    else:
        print("Training new model...")
        model = ResumeScreeningLLM(vocab_size=2000)
        model.train(resumes, job_postings)
    
    print("✓ Model ready for analysis")
    print()
    
    # Run comprehensive analysis
    print("Running comprehensive forensic analysis...")
    analyzer = ComprehensiveForensicAnalyzer(model)
    
    # Basic analyses
    print("Phase 1: Basic Analysis")
    analyzer.analyze_vocabulary_bias()
    analyzer.analyze_scoring_patterns(resumes, job_postings)
    analyzer.analyze_feature_importance()
    analyzer.analyze_model_transparency()
    
    # Extended analyses
    print("\nPhase 2: Extended Analysis")
    analyzer.analyze_fairness_metrics(resumes, job_postings)
    analyzer.analyze_vocabulary_impact()
    analyzer.analyze_decision_consistency(resumes, job_postings)
    
    # Generate comprehensive report
    print("\nPhase 3: Report Generation")
    comprehensive_results = analyzer.generate_comprehensive_report()
    
    print("\n" + "="*80)
    print("FORENSIC ANALYSIS COMPLETED!")
    print("="*80)
    print()
    print("Output files generated:")
    print("• Comprehensive analysis report (JSON)")
    print("• Risk assessment summary")
    print("• Detailed recommendations")
    print("• Compliance checklist")
    print()
    print("Next steps:")
    print("1. Review the executive summary above")
    print("2. Examine the detailed JSON report")
    print("3. Implement high-priority recommendations")
    print("4. Set up ongoing monitoring")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during analysis: {e}")
        print("Please ensure all dependencies are installed and data is available.")
        sys.exit(1)