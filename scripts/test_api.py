#!/usr/bin/env python3
"""
Test script for the Resume Screening LLM API
"""

import requests
import json
from typing import Dict, List

# Base URL for the API
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the health check endpoint."""
    print("\n=== Testing Health Check ===")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_model_info():
    """Test the model info endpoint."""
    print("\n=== Testing Model Info ===")
    response = requests.get(f"{BASE_URL}/model_info")
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        info = response.json()
        print(f"Model trained: {info['trained']}")
        print(f"Vocabulary size: {info['vocabulary_size']}")
        print(f"Top words: {info['top_vocabulary_words'][:10]}")
    else:
        print(f"Response: {response.json()}")
    return response.status_code == 200

def test_single_score():
    """Test scoring a single resume."""
    print("\n=== Testing Single Resume Score ===")
    
    # Sample data
    resume = {
        "id": "test_001",
        "name": "Jane Smith",
        "email": "jane.smith@email.com",
        "current_title": "Machine Learning Engineer",
        "years_experience": 5,
        "skills": ["Python", "TensorFlow", "PyTorch", "Docker", "AWS", "Kubernetes"],
        "education": [{
            "degree": "M.S.",
            "field": "Computer Science",
            "institution": "Stanford University",
            "graduation_year": 2018
        }],
        "experience": [{
            "title": "Machine Learning Engineer",
            "company": "Tech Corp",
            "description": "Developed and deployed ML models for production systems",
            "start_date": "2019-01",
            "end_date": "present"
        }]
    }
    
    job_posting = {
        "id": "job_test_001",
        "title": "Senior ML Engineer",
        "company": "AI Startup",
        "description": "We are looking for a Senior ML Engineer to join our team and build scalable AI systems",
        "requirements": {
            "skills": ["Python", "TensorFlow", "AWS", "Docker", "Machine Learning"],
            "experience": "5+ years of experience in machine learning",
            "education": "Master's degree in Computer Science or related field"
        }
    }
    
    request_data = {
        "resume": resume,
        "job_posting": job_posting
    }
    
    response = requests.post(f"{BASE_URL}/score", json=request_data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Overall Score: {result['overall_score']:.3f}")
        print(f"Recommendation: {result['recommendation']}")
        print(f"Matching Skills: {result['matching_skills']}")
        print(f"Skills Match %: {result['skills_match_percentage']:.1f}%")
        print(f"Experience Match: {result['experience_match']}")
    else:
        print(f"Response: {response.json()}")
    
    return response.status_code == 200

def test_batch_score():
    """Test batch scoring of resumes."""
    print("\n=== Testing Batch Resume Score ===")
    
    # Sample resumes
    resumes = [
        {
            "id": "batch_001",
            "name": "John Developer",
            "current_title": "Senior Python Developer",
            "years_experience": 7,
            "skills": ["Python", "Django", "PostgreSQL", "Docker", "AWS"],
            "education": [{
                "degree": "B.S.",
                "field": "Computer Science",
                "institution": "MIT"
            }],
            "experience": [{
                "title": "Senior Python Developer",
                "company": "Web Corp",
                "description": "Built scalable web applications"
            }]
        },
        {
            "id": "batch_002",
            "name": "Sarah Engineer",
            "current_title": "ML Engineer",
            "years_experience": 4,
            "skills": ["Python", "TensorFlow", "Scikit-learn", "AWS", "Docker"],
            "education": [{
                "degree": "M.S.",
                "field": "Data Science",
                "institution": "UC Berkeley"
            }],
            "experience": [{
                "title": "ML Engineer",
                "company": "AI Labs",
                "description": "Developed ML models for computer vision"
            }]
        },
        {
            "id": "batch_003",
            "name": "Mike Junior",
            "current_title": "Junior Developer",
            "years_experience": 1,
            "skills": ["Python", "JavaScript", "React"],
            "education": [{
                "degree": "B.S.",
                "field": "Computer Science",
                "institution": "State University"
            }],
            "experience": [{
                "title": "Junior Developer",
                "company": "Startup Inc",
                "description": "Frontend development"
            }]
        }
    ]
    
    job_posting = {
        "id": "job_batch_001",
        "title": "Machine Learning Engineer",
        "company": "AI Company",
        "description": "Looking for ML engineer with Python and cloud experience",
        "requirements": {
            "skills": ["Python", "TensorFlow", "AWS", "Docker"],
            "experience": "3+ years of ML experience"
        }
    }
    
    request_data = {
        "resumes": resumes,
        "job_posting": job_posting
    }
    
    response = requests.post(f"{BASE_URL}/batch_score", json=request_data)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nBatch Metadata:")
        print(f"  Total Resumes: {result['batch_metadata']['total_resumes']}")
        print(f"  Average Score: {result['batch_metadata']['average_score']:.3f}")
        
        print(f"\nIndividual Scores (sorted by score):")
        for score in result['scores']:
            print(f"  - {score['resume_id']}: {score['overall_score']:.3f} ({score['recommendation']})")
    else:
        print(f"Response: {response.json()}")
    
    return response.status_code == 200

def test_train_endpoint():
    """Test the training endpoint."""
    print("\n=== Testing Train Endpoint ===")
    
    request_data = {
        "vocab_size": 3000,
        "data_dir": "data/synthetic"
    }
    
    response = requests.post(f"{BASE_URL}/train", json=request_data)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code in [200, 202]

def main():
    """Run all API tests."""
    print("Starting API tests...")
    print(f"Testing API at: {BASE_URL}")
    
    # Track test results
    results = {
        "health": test_health_check(),
        "model_info": test_model_info(),
    }
    
    # Only test scoring endpoints if model is loaded
    try:
        model_info_response = requests.get(f"{BASE_URL}/model_info")
        if model_info_response.status_code == 200:
            results["single_score"] = test_single_score()
            results["batch_score"] = test_batch_score()
        else:
            print("\nSkipping scoring tests - model not loaded")
            results["single_score"] = False
            results["batch_score"] = False
    except Exception as e:
        print(f"Error checking model status: {e}")
        results["single_score"] = False
        results["batch_score"] = False
    
    # Training test (optional)
    # results["train"] = test_train_endpoint()
    
    # Summary
    print("\n=== Test Summary ===")
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"{test_name}: {status}")
    
    total_passed = sum(1 for passed in results.values() if passed)
    print(f"\nTotal: {total_passed}/{len(results)} tests passed")

if __name__ == "__main__":
    main()