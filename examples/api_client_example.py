#!/usr/bin/env python3
"""
Example client for the Resume Screening LLM API
Demonstrates how to use the API endpoints programmatically
"""

import requests
import json
from typing import Dict, List, Optional

class ResumeScreeningClient:
    """Client for interacting with the Resume Screening LLM API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def check_health(self) -> Dict:
        """Check API health status."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        response = self.session.get(f"{self.base_url}/model_info")
        response.raise_for_status()
        return response.json()
    
    def score_resume(self, resume: Dict, job_posting: Dict) -> Dict:
        """Score a single resume against a job posting."""
        data = {
            "resume": resume,
            "job_posting": job_posting
        }
        response = self.session.post(f"{self.base_url}/score", json=data)
        response.raise_for_status()
        return response.json()
    
    def batch_score_resumes(self, resumes: List[Dict], job_posting: Dict) -> Dict:
        """Score multiple resumes against a job posting."""
        data = {
            "resumes": resumes,
            "job_posting": job_posting
        }
        response = self.session.post(f"{self.base_url}/batch_score", json=data)
        response.raise_for_status()
        return response.json()
    
    def trigger_training(self, vocab_size: int = 5000, data_dir: str = "data/synthetic") -> Dict:
        """Trigger model training."""
        data = {
            "vocab_size": vocab_size,
            "data_dir": data_dir
        }
        response = self.session.post(f"{self.base_url}/train", json=data)
        response.raise_for_status()
        return response.json()


def main():
    """Example usage of the API client."""
    
    # Initialize client
    client = ResumeScreeningClient()
    
    print("Resume Screening LLM API Client Example")
    print("=" * 50)
    
    # Check health
    print("\n1. Checking API health...")
    try:
        health = client.check_health()
        print(f"   Status: {health['status']}")
        print(f"   Model Loaded: {health['model_loaded']}")
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    # Get model info
    print("\n2. Getting model information...")
    try:
        info = client.get_model_info()
        print(f"   Vocabulary Size: {info['vocabulary_size']}")
        print(f"   Trained: {info['trained']}")
        print(f"   Model Version: {info['model_metadata'].get('version', 'N/A')}")
    except Exception as e:
        print(f"   Error: {e}")
        print("   Note: You may need to train a model first")
    
    # Example resume and job posting
    sample_resume = {
        "resume_id": "example_001",
        "personal_info": {
            "name": "Alice Johnson",
            "email": "alice.johnson@email.com",
            "current_title": "Senior Data Scientist",
            "phone": "(555) 123-4567",
            "location": "San Francisco, CA"
        },
        "years_of_experience": 6,
        "skills": {
            "technical": [
                "Python", "Machine Learning", "TensorFlow", "PyTorch",
                "SQL", "AWS", "Docker", "Statistics", "Data Analysis"
            ],
            "soft": ["Leadership", "Communication", "Problem Solving"]
        },
        "education": [{
            "degree": "M.S.",
            "field": "Data Science",
            "institution": "Stanford University",
            "graduation_year": 2017
        }],
        "work_experience": [{
            "position": "Senior Data Scientist",
            "company": "Tech Innovations Inc",
            "description": "Led ML projects for predictive analytics and recommendation systems",
            "start_date": "2020-01",
            "end_date": "present"
        }, {
            "position": "Data Scientist",
            "company": "Analytics Corp",
            "description": "Developed machine learning models for customer segmentation",
            "start_date": "2017-06",
            "end_date": "2019-12"
        }]
    }
    
    sample_job = {
        "id": "job_example_001",
        "title": "Machine Learning Engineer",
        "company": "AI Solutions Ltd",
        "description": "We are seeking an experienced ML Engineer to build and deploy scalable machine learning systems. You will work on cutting-edge AI projects and collaborate with cross-functional teams.",
        "requirements": {
            "skills": [
                "Python", "Machine Learning", "TensorFlow", "AWS",
                "Docker", "Kubernetes", "MLOps", "Deep Learning"
            ],
            "experience": "5+ years of experience in machine learning and software engineering",
            "education": "Master's degree in Computer Science, Data Science, or related field"
        }
    }
    
    # Score single resume
    print("\n3. Scoring single resume...")
    try:
        result = client.score_resume(sample_resume, sample_job)
        print(f"   Resume: {sample_resume['personal_info']['name']} - {sample_resume['personal_info']['current_title']}")
        print(f"   Job: {sample_job['title']} at {sample_job['company']}")
        print(f"   Overall Score: {result['overall_score']:.3f}")
        print(f"   Recommendation: {result['recommendation']}")
        print(f"   Matching Skills: {', '.join(result['matching_skills'][:5])}")
        print(f"   Skills Match: {result['skills_match_percentage']:.1f}%")
        print(f"   Experience Match: {result['experience_match']}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Batch scoring example
    print("\n4. Batch scoring multiple resumes...")
    
    additional_resumes = [
        {
            "resume_id": "example_002",
            "personal_info": {
                "name": "Bob Smith",
                "current_title": "Junior Developer",
                "email": "bob.smith@email.com",
                "phone": "(555) 234-5678",
                "location": "Austin, TX"
            },
            "years_of_experience": 2,
            "skills": {
                "technical": ["Python", "JavaScript", "React", "Node.js"],
                "soft": ["Communication", "Teamwork"]
            },
            "education": [{
                "degree": "B.S.",
                "field": "Computer Science",
                "institution": "State University"
            }],
            "work_experience": [{
                "position": "Junior Developer",
                "company": "StartupXYZ",
                "description": "Frontend development with React"
            }]
        },
        {
            "resume_id": "example_003",
            "personal_info": {
                "name": "Carol Williams",
                "current_title": "ML Research Engineer",
                "email": "carol.williams@email.com",
                "phone": "(555) 345-6789",
                "location": "Boston, MA"
            },
            "years_of_experience": 8,
            "skills": {
                "technical": ["Python", "TensorFlow", "PyTorch", "Research", "AWS", "Docker", "Kubernetes"],
                "soft": ["Research", "Leadership", "Mentoring"]
            },
            "education": [{
                "degree": "Ph.D.",
                "field": "Machine Learning",
                "institution": "MIT"
            }],
            "work_experience": [{
                "position": "ML Research Engineer",
                "company": "Research Lab",
                "description": "Published papers on deep learning architectures"
            }]
        }
    ]
    
    all_resumes = [sample_resume] + additional_resumes
    
    try:
        batch_result = client.batch_score_resumes(all_resumes, sample_job)
        print(f"   Total Resumes: {batch_result['batch_metadata']['total_resumes']}")
        print(f"   Average Score: {batch_result['batch_metadata']['average_score']:.3f}")
        print("\n   Top Candidates:")
        for i, score in enumerate(batch_result['scores'][:3], 1):
            print(f"   {i}. Resume ID: {score['resume_id']}")
            print(f"      Score: {score['overall_score']:.3f} ({score['recommendation']})")
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 50)
    print("Example completed!")


if __name__ == "__main__":
    main()