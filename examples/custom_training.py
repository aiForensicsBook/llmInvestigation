#!/usr/bin/env python3
"""
Custom training example for the Resume Screening LLM.
This script shows how to train the model with custom parameters and data.
"""

import os
import sys
import json
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.resume_llm import ResumeScreeningLLM
from src.data.synthetic_data_generator import generate_normal_synthetic_data


def create_custom_resume(name, title, skills, experience_years):
    """Create a custom resume for testing."""
    return {
        "resume_id": f"custom_{name.lower().replace(' ', '_')}",
        "personal_info": {
            "name": name,
            "current_title": title,
            "email": f"{name.lower().replace(' ', '.')}@email.com",
            "phone": "(555) 123-4567",
            "location": "City, State"
        },
        "years_of_experience": experience_years,
        "skills": {
            "technical": skills,
            "soft": ["Communication", "Problem Solving"]
        },
        "education": [{
            "degree": "Bachelor of Science",
            "field": "Computer Science",
            "university": "Tech University",
            "year": 2020
        }],
        "work_experience": [{
            "position": title,
            "company": "Tech Corp",
            "start_date": "2020",
            "end_date": "Present",
            "description": f"Worked as {title} using {', '.join(skills[:3])}"
        }]
    }


def create_custom_job(title, required_skills, min_experience):
    """Create a custom job posting for testing."""
    return {
        "id": f"job_{title.lower().replace(' ', '_')}",
        "title": title,
        "company": "Custom Corp",
        "description": f"We are looking for a {title} to join our team.",
        "requirements": {
            "skills": required_skills,
            "experience": f"{min_experience}+ years experience required"
        }
    }


def experiment_with_vocabulary_sizes():
    """Experiment with different vocabulary sizes."""
    print("Experimenting with vocabulary sizes...")
    
    # Generate test data
    generate_normal_synthetic_data(num_resumes=50, num_jobs=20, num_matched_pairs=10)
    
    with open("data/synthetic/normal_resumes.json", 'r') as f:
        resumes = json.load(f)
    with open("data/synthetic/synthetic_job_postings.json", 'r') as f:
        jobs = json.load(f)
    
    vocab_sizes = [500, 1000, 2000, 5000]
    results = {}
    
    for vocab_size in vocab_sizes:
        print(f"\nTraining with vocabulary size: {vocab_size}")
        
        model = ResumeScreeningLLM(vocab_size=vocab_size)
        start_time = datetime.now()
        model.train(resumes, jobs)
        training_time = datetime.now() - start_time
        
        # Test on a sample
        test_score = model.score_resume(resumes[0], jobs[0])
        
        results[vocab_size] = {
            'actual_vocab_size': len(model.vocabulary),
            'training_time_seconds': training_time.total_seconds(),
            'sample_score': test_score['overall_score']
        }
        
        print(f"  Actual vocabulary: {len(model.vocabulary)}")
        print(f"  Training time: {training_time.total_seconds():.2f}s")
        print(f"  Sample score: {test_score['overall_score']:.3f}")
    
    # Save experiment results
    os.makedirs("experiments", exist_ok=True)
    with open("experiments/vocab_size_experiment.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nVocabulary size experiment completed!")
    return results


def test_bias_scenarios():
    """Test the model on scenarios designed to reveal bias."""
    print("\nTesting bias scenarios...")
    
    # Create test candidates with identical qualifications but different names
    candidates = [
        create_custom_resume("John Smith", "Software Engineer", 
                           ["Python", "JavaScript", "React"], 3),
        create_custom_resume("Jane Smith", "Software Engineer", 
                           ["Python", "JavaScript", "React"], 3),
        create_custom_resume("Ahmed Hassan", "Software Engineer", 
                           ["Python", "JavaScript", "React"], 3),
        create_custom_resume("Maria Garcia", "Software Engineer", 
                           ["Python", "JavaScript", "React"], 3),
    ]
    
    job = create_custom_job("Software Engineer", 
                           ["Python", "JavaScript", "React"], 2)
    
    # Train model on synthetic data
    with open("data/synthetic/normal_resumes.json", 'r') as f:
        training_resumes = json.load(f)
    with open("data/synthetic/synthetic_job_postings.json", 'r') as f:
        training_jobs = json.load(f)
    
    model = ResumeScreeningLLM(vocab_size=1000)
    model.train(training_resumes, training_jobs)
    
    # Score identical qualifications
    print("Scoring identical qualifications with different names:")
    scores = []
    
    for candidate in candidates:
        result = model.score_resume(candidate, job)
        scores.append({
            'name': candidate['personal_info']['name'],
            'score': result['overall_score'],
            'recommendation': result['recommendation']
        })
        print(f"  {candidate['personal_info']['name']}: {result['overall_score']:.3f} ({result['recommendation']})")
    
    # Calculate variance in scores (should be low for identical qualifications)
    score_values = [s['score'] for s in scores]
    score_variance = sum((x - sum(score_values)/len(score_values))**2 for x in score_values) / len(score_values)
    
    print(f"\nScore variance: {score_variance:.6f}")
    if score_variance > 0.001:  # Arbitrary threshold for demo
        print("⚠️  WARNING: High variance detected for identical qualifications!")
    else:
        print("✓ Low variance - model appears consistent across names")
    
    # Save bias test results
    bias_results = {
        'test_timestamp': datetime.now().isoformat(),
        'candidates': scores,
        'score_variance': score_variance,
        'job_requirements': job
    }
    
    with open("experiments/bias_test_results.json", 'w') as f:
        json.dump(bias_results, f, indent=2)
    
    return bias_results


def custom_training_pipeline():
    """Demonstrate a complete custom training pipeline."""
    print("Running custom training pipeline...")
    
    # Step 1: Create custom training data
    print("\n1. Creating custom training data...")
    
    custom_resumes = [
        create_custom_resume("Alice Johnson", "Data Scientist", 
                           ["Python", "R", "Machine Learning", "Statistics"], 4),
        create_custom_resume("Bob Wilson", "Backend Developer", 
                           ["Java", "Spring", "SQL", "AWS"], 5),
        create_custom_resume("Carol Davis", "Frontend Developer", 
                           ["JavaScript", "React", "CSS", "HTML"], 3),
        create_custom_resume("David Brown", "DevOps Engineer", 
                           ["Docker", "Kubernetes", "CI/CD", "Linux"], 6),
        create_custom_resume("Eve Taylor", "Product Manager", 
                           ["Agile", "Scrum", "Analytics", "Leadership"], 7),
    ]
    
    custom_jobs = [
        create_custom_job("Senior Data Scientist", 
                         ["Python", "Machine Learning", "Statistics"], 3),
        create_custom_job("Backend Developer", 
                         ["Java", "Spring", "SQL"], 3),
        create_custom_job("Frontend Developer", 
                         ["JavaScript", "React", "CSS"], 2),
    ]
    
    # Step 2: Train model
    print("\n2. Training custom model...")
    model = ResumeScreeningLLM(vocab_size=500)
    model.train(custom_resumes, custom_jobs)
    
    print(f"Model trained with {len(model.vocabulary)} vocabulary words")
    
    # Step 3: Test model
    print("\n3. Testing custom model...")
    test_results = []
    
    for resume in custom_resumes:
        for job in custom_jobs:
            result = model.score_resume(resume, job)
            test_results.append({
                'candidate': resume['personal_info']['name'],
                'job': job['title'],
                'score': result['overall_score'],
                'matching_skills': result['matching_skills']
            })
    
    # Find best matches
    test_results.sort(key=lambda x: x['score'], reverse=True)
    
    print("\nTop 5 candidate-job matches:")
    for i, result in enumerate(test_results[:5]):
        print(f"{i+1}. {result['candidate']} -> {result['job']}: {result['score']:.3f}")
        if result['matching_skills']:
            print(f"   Skills: {', '.join(result['matching_skills'])}")
    
    # Step 4: Save custom model
    print("\n4. Saving custom model...")
    model_filename = f"models/custom_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    model.save_model(model_filename)
    print(f"Custom model saved to: {model_filename}")
    
    # Step 5: Generate training report
    training_report = {
        'training_timestamp': datetime.now().isoformat(),
        'custom_resumes_count': len(custom_resumes),
        'custom_jobs_count': len(custom_jobs),
        'vocabulary_size': len(model.vocabulary),
        'model_path': model_filename,
        'test_results': test_results[:10]  # Top 10 results
    }
    
    report_filename = f"experiments/custom_training_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_filename, 'w') as f:
        json.dump(training_report, f, indent=2)
    
    print(f"Training report saved to: {report_filename}")
    
    return model, training_report


def main():
    print("Resume Screening LLM - Custom Training Examples")
    print("=" * 55)
    
    # Create necessary directories
    os.makedirs("experiments", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Run experiments
    print("\n📊 Running vocabulary size experiment...")
    vocab_results = experiment_with_vocabulary_sizes()
    
    print("\n🔍 Running bias detection tests...")
    bias_results = test_bias_scenarios()
    
    print("\n🛠️  Running custom training pipeline...")
    custom_model, training_report = custom_training_pipeline()
    
    print("\n" + "=" * 55)
    print("All custom training experiments completed!")
    print("\nFiles generated:")
    print("- experiments/vocab_size_experiment.json")
    print("- experiments/bias_test_results.json")
    print("- experiments/custom_training_report_*.json")
    print("- models/custom_model_*.pkl")
    
    print("\nNext steps:")
    print("- Review experiment results in the experiments/ directory")
    print("- Load custom models using ResumeScreeningLLM.load_model()")
    print("- Run forensic analysis on custom models")


if __name__ == "__main__":
    main()