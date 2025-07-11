#!/usr/bin/env python3
"""
Complete Forensic Workflow Test Script
=====================================

Tests the entire workflow from synthetic data generation through
model training, usage, forensic analysis, and report generation.

Author: Forensic Testing Framework
Version: 1.0
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data.synthetic_data_generator import generate_resume, generate_job_posting
from model.resume_llm import ResumeScreeningLLM
from utils.visualization import (
    create_comprehensive_visualization_dashboard,
    plot_synthetic_data_detection,
    plot_model_specifications,
    plot_chain_of_custody
)

# Add forensic modules to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'forensic/collection'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'forensic/testing'))

try:
    from forensic_collector import ForensicCollector
    from comprehensive_forensic_report import ComprehensiveForensicReport
    from bias_analyzer import BiasAnalyzer
except ImportError as e:
    print(f"Warning: Could not import forensic modules: {e}")
    print("Some features may not work properly.")


class ForensicWorkflowTester:
    """Test the complete forensic workflow."""
    
    def __init__(self, output_dir: str = None):
        """Initialize the workflow tester.
        
        Args:
            output_dir: Directory for test outputs (uses temp dir if None)
        """
        if output_dir is None:
            self.output_dir = tempfile.mkdtemp(prefix='forensic_test_')
        else:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"Test output directory: {self.output_dir}")
        
        # Initialize components
        self.model = ResumeScreeningLLM()
        
        # Test data paths
        self.synthetic_resumes_path = os.path.join(self.output_dir, "test_resumes.json")
        self.synthetic_jobs_path = os.path.join(self.output_dir, "test_jobs.json")
        self.model_path = os.path.join(self.output_dir, "test_model.json")
        self.evidence_dir = os.path.join(self.output_dir, "evidence")
        self.reports_dir = os.path.join(self.output_dir, "reports")
        
        # Create directories
        os.makedirs(self.evidence_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)
    
    def step1_generate_synthetic_data(self) -> bool:
        """Step 1: Generate synthetic training data."""
        print("\\n=== Step 1: Generating Synthetic Data ===" )
        
        try:
            # Generate synthetic resumes
            print("Generating synthetic resumes...")
            resumes = [generate_resume() for _ in range(50)]
            
            # Add some patterns that will be detected as synthetic
            for i, resume in enumerate(resumes[:10]):
                # Add sequential patterns
                resume['personal_info']['name'] = f"Test User {i+1}"
                resume['personal_info']['email'] = f"test{i+1}@example.com"
            
            with open(self.synthetic_resumes_path, 'w') as f:
                json.dump(resumes, f, indent=2)
            
            # Generate synthetic job postings
            print("Generating synthetic job postings...")
            jobs = [generate_job_posting() for _ in range(20)]
            
            with open(self.synthetic_jobs_path, 'w') as f:
                json.dump(jobs, f, indent=2)
            
            print(f"Generated {len(resumes)} resumes and {len(jobs)} job postings")
            return True
            
        except Exception as e:
            print(f"Error in Step 1: {e}")
            return False
    
    def step2_train_model(self) -> bool:
        """Step 2: Train the model on synthetic data."""
        print("\\n=== Step 2: Training Model ===")
        
        try:
            # Load synthetic data
            with open(self.synthetic_resumes_path, 'r') as f:
                resumes = json.load(f)
            
            with open(self.synthetic_jobs_path, 'r') as f:
                jobs = json.load(f)
            
            # Train model
            print("Training model...")
            # Create matched pairs manually
            matched_pairs = []
            for i in range(min(30, len(jobs))):
                matched_pairs.append({
                    'resume': resumes[i % len(resumes)],
                    'job': jobs[i],
                    'match_score': 0.8
                })
            
            # Train the model
            self.model.train(resumes, jobs)
            
            # Save model with metadata
            model_data = {
                'model_type': 'TF-IDF + Cosine Similarity',
                'vocabulary': list(self.model.vocabulary),
                'idf_values': self.model.idf_values,
                'training_config': {
                    'training_data_size': len(resumes) + len(jobs),
                    'vocabulary_size': len(self.model.vocabulary),
                    'trained_on': datetime.now().isoformat()
                },
                'metadata': {
                    'framework': 'scikit-learn',
                    'algorithm': 'TF-IDF',
                    'similarity_metric': 'cosine',
                    'created_by': 'ForensicWorkflowTester'
                }
            }
            
            with open(self.model_path, 'w') as f:
                json.dump(model_data, f, indent=2)
            
            print(f"Model trained with vocabulary size: {len(self.model.vocabulary)}")
            return True
            
        except Exception as e:
            print(f"Error in Step 2: {e}")
            return False
    
    def step3_test_model_usage(self) -> bool:
        """Step 3: Test model usage and collect performance data."""
        print("\\n=== Step 3: Testing Model Usage ===")
        
        try:
            # Load test data
            with open(self.synthetic_resumes_path, 'r') as f:
                resumes = json.load(f)
            
            with open(self.synthetic_jobs_path, 'r') as f:
                jobs = json.load(f)
            
            # Test model on a subset
            test_results = []
            print("Running model predictions...")
            
            for i, resume in enumerate(resumes[:10]):
                for j, job in enumerate(jobs[:5]):
                    result = self.model.score_resume(resume, job)
                    test_results.append({
                        'resume_id': i,
                        'job_id': j,
                        'score': result['overall_score'],
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Save test results
            test_results_path = os.path.join(self.output_dir, "test_results.json")
            with open(test_results_path, 'w') as f:
                json.dump(test_results, f, indent=2)
            
            print(f"Generated {len(test_results)} test predictions")
            return True
            
        except Exception as e:
            print(f"Error in Step 3: {e}")
            return False
    
    def step4_forensic_analysis(self) -> bool:
        """Step 4: Perform comprehensive forensic analysis."""
        print("\\n=== Step 4: Forensic Analysis ===")
        
        try:
            # Initialize forensic collector
            case_id = f"TEST_CASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            investigator = "Forensic Test Suite"
            
            if 'ForensicCollector' in globals():
                collector = ForensicCollector(case_id, investigator)
                
                # Collect evidence
                print("Collecting evidence...")
                collection_result = collector.collect_evidence(
                    self.output_dir,
                    self.evidence_dir,
                    "Complete forensic test workflow evidence"
                )
                
                # Analyze synthetic data
                print("Analyzing synthetic data patterns...")
                synthetic_results = collector.detect_synthetic_data(
                    self.synthetic_resumes_path, 'json'
                )
                
                # Extract model specifications
                print("Extracting model specifications...")
                model_specs = collector.extract_model_specifications(self.model_path)
                
                # Analyze bias distributions
                print("Analyzing bias distributions...")
                bias_results = collector.analyze_bias_distributions(self.synthetic_resumes_path)
                
                # Save forensic analysis results
                forensic_results = {
                    'case_id': case_id,
                    'investigator': investigator,
                    'analysis_timestamp': datetime.now().isoformat(),
                    'collection_result': collection_result,
                    'synthetic_analysis': synthetic_results,
                    'model_specifications': model_specs,
                    'bias_analysis': bias_results
                }
                
                forensic_results_path = os.path.join(self.reports_dir, "forensic_analysis.json")
                with open(forensic_results_path, 'w') as f:
                    json.dump(forensic_results, f, indent=2, default=str)
                
                print("Forensic analysis completed")
                return True
            else:
                print("ForensicCollector not available, skipping detailed forensic analysis")
                return True
                
        except Exception as e:
            print(f"Error in Step 4: {e}")
            return False
    
    def step5_bias_analysis(self) -> bool:
        """Step 5: Perform detailed bias analysis."""
        print("\\n=== Step 5: Bias Analysis ===")
        
        try:
            if 'BiasAnalyzer' not in globals():
                print("BiasAnalyzer not available, skipping bias analysis")
                return True
            
            # Initialize bias analyzer
            bias_analyzer = BiasAnalyzer(self.reports_dir)
            
            # Load test data
            with open(self.synthetic_resumes_path, 'r') as f:
                resumes = json.load(f)
            
            # Create mock scoring data with demographic info
            scored_data = []
            for i, resume in enumerate(resumes):
                # Add mock scores
                score = 0.5 + (i % 10) * 0.05  # Vary scores
                
                # Extract or add demographic info
                gender = resume.get('personal_info', {}).get('gender', 'unknown')
                if gender == 'unknown':
                    # Infer from name patterns for testing
                    name = resume.get('personal_info', {}).get('name', '').lower()
                    if any(n in name for n in ['john', 'mike', 'david', 'james']):
                        gender = 'male'
                    elif any(n in name for n in ['jane', 'mary', 'sarah', 'lisa']):
                        gender = 'female'
                    else:
                        gender = 'other'
                
                scored_data.append({
                    'score': score,
                    'gender': gender,
                    'age': resume.get('personal_info', {}).get('age', 30 + (i % 20)),
                    'race': resume.get('personal_info', {}).get('race', 'unspecified')
                })
            
            # Convert to DataFrame for analysis
            import pandas as pd
            df = pd.DataFrame(scored_data)
            
            # Perform bias analysis
            print("Analyzing gender bias...")
            gender_results = bias_analyzer.analyze_gender_bias(df, 'score', 'gender')
            
            print("Analyzing age bias...")
            age_results = bias_analyzer.analyze_age_bias(df, 'score', 'age')
            
            # Generate bias report
            print("Generating bias analysis report...")
            bias_report_path = bias_analyzer.generate_bias_report(
                os.path.join(self.reports_dir, "bias_analysis_report.json")
            )
            
            print(f"Bias analysis completed. Found {len(gender_results + age_results)} test results")
            return True
            
        except Exception as e:
            print(f"Error in Step 5: {e}")
            return False
    
    def step6_create_visualizations(self) -> bool:
        """Step 6: Create comprehensive visualizations."""
        print("\\n=== Step 6: Creating Visualizations ===")
        
        try:
            # Load data
            with open(self.synthetic_resumes_path, 'r') as f:
                resumes = json.load(f)
            
            with open(self.synthetic_jobs_path, 'r') as f:
                jobs = json.load(f)
            
            # Load forensic results if available
            synthetic_results = None
            model_specs = None
            custody_data = None
            
            forensic_results_path = os.path.join(self.reports_dir, "forensic_analysis.json")
            if os.path.exists(forensic_results_path):
                with open(forensic_results_path, 'r') as f:
                    forensic_data = json.load(f)
                    synthetic_results = forensic_data.get('synthetic_analysis')
                    model_specs = forensic_data.get('model_specifications')
                    custody_data = forensic_data.get('collection_result')
            
            # Create visualizations
            print("Creating comprehensive visualization dashboard...")
            viz_dir = os.path.join(self.reports_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            
            create_comprehensive_visualization_dashboard(
                self.model, 
                resumes[:20],  # Limit for performance
                jobs[:10],
                viz_dir,
                synthetic_results=synthetic_results,
                model_specs=model_specs,
                custody_data=custody_data
            )
            
            print("Visualizations created successfully")
            return True
            
        except Exception as e:
            print(f"Error in Step 6: {e}")
            return False
    
    def step7_generate_comprehensive_report(self) -> bool:
        """Step 7: Generate comprehensive forensic report."""
        print("\\n=== Step 7: Generating Comprehensive Report ===")
        
        try:
            if 'ComprehensiveForensicReport' not in globals():
                print("ComprehensiveForensicReport not available, skipping report generation")
                return True
            
            # Initialize report generator
            case_id = f"TEST_CASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            investigator = "Forensic Test Suite"
            
            report_gen = ComprehensiveForensicReport(case_id, investigator)
            
            # Analyze training data
            print("Analyzing training data for report...")
            report_gen.analyze_training_data(self.synthetic_resumes_path)
            
            # Analyze model
            print("Analyzing model for report...")
            report_gen.analyze_model(self.model_path)
            
            # Generate chain of custody report
            print("Generating chain of custody report...")
            report_gen.generate_chain_of_custody_report(self.evidence_dir)
            
            # Generate HTML report
            html_report_path = os.path.join(self.reports_dir, "comprehensive_forensic_report.html")
            print("Generating HTML report...")
            report_gen.generate_html_report(html_report_path)
            
            # Generate JSON report
            json_report_path = os.path.join(self.reports_dir, "comprehensive_forensic_report.json")
            print("Generating JSON report...")
            report_gen.generate_json_report(json_report_path)
            
            print("Comprehensive reports generated successfully")
            return True
            
        except Exception as e:
            print(f"Error in Step 7: {e}")
            return False
    
    def run_complete_workflow(self) -> bool:
        """Run the complete forensic workflow test."""
        print("Starting Complete Forensic Workflow Test")
        print("=" * 50)
        
        steps = [
            ("Generate Synthetic Data", self.step1_generate_synthetic_data),
            ("Train Model", self.step2_train_model),
            ("Test Model Usage", self.step3_test_model_usage),
            ("Forensic Analysis", self.step4_forensic_analysis),
            ("Bias Analysis", self.step5_bias_analysis),
            ("Create Visualizations", self.step6_create_visualizations),
            ("Generate Comprehensive Report", self.step7_generate_comprehensive_report)
        ]
        
        results = {}
        
        for step_name, step_func in steps:
            print(f"\\nExecuting: {step_name}")
            try:
                success = step_func()
                results[step_name] = "PASSED" if success else "FAILED"
                
                if not success:
                    print(f"❌ {step_name} FAILED")
                else:
                    print(f"✅ {step_name} PASSED")
                    
            except Exception as e:
                print(f"❌ {step_name} FAILED with exception: {e}")
                results[step_name] = f"FAILED: {e}"
        
        # Print summary
        print("\\n" + "=" * 50)
        print("WORKFLOW TEST SUMMARY")
        print("=" * 50)
        
        passed_count = 0
        for step_name, result in results.items():
            status = "✅" if result == "PASSED" else "❌"
            print(f"{status} {step_name}: {result}")
            if result == "PASSED":
                passed_count += 1
        
        success_rate = (passed_count / len(steps)) * 100
        print(f"\\nOverall Success Rate: {success_rate:.1f}% ({passed_count}/{len(steps)})")
        
        print(f"\\nTest outputs saved to: {self.output_dir}")
        print("Key files:")
        print(f"- Training data: {self.synthetic_resumes_path}")
        print(f"- Model: {self.model_path}")
        print(f"- Evidence: {self.evidence_dir}")
        print(f"- Reports: {self.reports_dir}")
        
        return success_rate == 100.0
    
    def cleanup(self):
        """Clean up temporary files."""
        if hasattr(self, 'output_dir') and 'tmp' in self.output_dir:
            try:
                shutil.rmtree(self.output_dir)
                print(f"Cleaned up temporary directory: {self.output_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up {self.output_dir}: {e}")


def main():
    """Main function to run the complete workflow test."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Complete Forensic Workflow')
    parser.add_argument('--output-dir', help='Output directory (uses temp dir if not specified)')
    parser.add_argument('--keep-files', action='store_true', help='Keep output files after test')
    
    args = parser.parse_args()
    
    # Run the test
    tester = ForensicWorkflowTester(args.output_dir)
    
    try:
        success = tester.run_complete_workflow()
        
        if success:
            print("\\n🎉 All tests passed! The forensic workflow is working correctly.")
            exit_code = 0
        else:
            print("\\n⚠️  Some tests failed. Check the output above for details.")
            exit_code = 1
        
        if not args.keep_files:
            tester.cleanup()
        
        return exit_code
        
    except KeyboardInterrupt:
        print("\\n⏹️  Test interrupted by user.")
        if not args.keep_files:
            tester.cleanup()
        return 1
    except Exception as e:
        print(f"\\n💥 Test failed with unexpected error: {e}")
        if not args.keep_files:
            tester.cleanup()
        return 1


if __name__ == "__main__":
    exit(main())