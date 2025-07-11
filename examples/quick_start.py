#!/usr/bin/env python3
"""
Quick start example for the Resume Screening LLM.
This script demonstrates the basic usage of the model.
"""

import os
import sys
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.resume_llm import ResumeScreeningLLM
from src.data.synthetic_data_generator import generate_normal_synthetic_data


def main():
    print("Resume Screening LLM - Quick Start")
    print("=" * 50)
    
    # Step 1: Generate synthetic data if it doesn't exist
    data_dir = "data/synthetic"
    resumes_file = os.path.join(data_dir, "normal_resumes.json")
    jobs_file = os.path.join(data_dir, "synthetic_job_postings.json")
    
    if not os.path.exists(resumes_file) or not os.path.exists(jobs_file):
        print("1. Generating synthetic training data...")
        generate_normal_synthetic_data(num_resumes=50, num_jobs=25, num_matched_pairs=10)
        print("   ✓ Synthetic data generated")
    else:
        print("1. Using existing synthetic data")
    
    # Step 2: Load training data
    print("\n2. Loading training data...")
    with open(resumes_file, 'r') as f:
        resumes = json.load(f)
    with open(jobs_file, 'r') as f:
        job_postings = json.load(f)
    
    print(f"   ✓ Loaded {len(resumes)} resumes and {len(job_postings)} job postings")
    
    # Step 3: Train model
    print("\n3. Training model...")
    model = ResumeScreeningLLM(vocab_size=1000)
    model.train(resumes, job_postings)
    print(f"   ✓ Model trained with vocabulary size: {len(model.vocabulary)}")
    
    # Step 4: Test scoring
    print("\n4. Testing resume scoring...")
    
    # Take first resume and first job for demo
    test_resume = resumes[0]
    test_job = job_postings[0]
    
    print(f"\nTest Resume: {test_resume['personal_info']['name']} - {test_resume['personal_info'].get('current_title', 'N/A')}")
    print(f"Test Job: {test_job['title']} at {test_job['company']}")
    
    # Score the resume
    result = model.score_resume(test_resume, test_job)
    
    print(f"\nScoring Results:")
    print(f"  Overall Score: {result['overall_score']:.3f}")
    print(f"  Recommendation: {result['recommendation']}")
    print(f"  Skills Match: {result['skills_match_percentage']:.1f}%")
    print(f"  Experience Match: {'Yes' if result['experience_match'] else 'No'}")
    
    if result['matching_skills']:
        print(f"  Matching Skills: {', '.join(result['matching_skills'][:5])}")
    
    # Step 5: Batch scoring demo
    print("\n5. Testing batch scoring...")
    batch_results = model.batch_score(resumes[:5], test_job)
    
    print(f"\nTop 3 candidates for {test_job['title']}:")
    for i, result in enumerate(batch_results[:3]):
        resume_name = [r['personal_info']['name'] for r in resumes if r.get('resume_id') == result['resume_id']]
        name = resume_name[0] if resume_name else f"Resume {i+1}"
        print(f"  {i+1}. {name} - Score: {result['overall_score']:.3f}")
    
    # Step 6: Model information
    print("\n6. Model information:")
    info = model.get_model_info()
    print(f"   Model Version: {info['model_metadata']['version']}")
    print(f"   Trained: {info['trained']}")
    
    # Step 7: Save model for later use
    print("\n7. Saving model...")
    os.makedirs("models", exist_ok=True)
    model.save_model("models/quick_start_model.pkl")
    print("   ✓ Model saved to models/quick_start_model.pkl")
    
    print("\n" + "=" * 50)
    print("Quick start completed successfully!")
    print("\nNext steps:")
    print("- Use CLI: python -m src.cli.cli --help")
    print("- Start API: python -m src.api.app")
    print("- Train custom model: python -m src.train")


if __name__ == "__main__":
    main()