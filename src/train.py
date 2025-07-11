import json
import os
import sys
from datetime import datetime
import argparse
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.resume_llm import ResumeScreeningLLM
from src.data.synthetic_data_generator import generate_normal_synthetic_data, generate_bias_test_data


def load_training_data(data_dir: str, use_biased_data: bool = False) -> tuple:
    """Load training data from JSON files."""
    if use_biased_data:
        resumes_path = os.path.join(data_dir, "biased_resumes.json")
        jobs_path = os.path.join(data_dir, "biased_job_postings.json")
    else:
        resumes_path = os.path.join(data_dir, "normal_resumes.json")
        jobs_path = os.path.join(data_dir, "synthetic_job_postings.json")
    
    if not os.path.exists(resumes_path) or not os.path.exists(jobs_path):
        print(f"Training data not found. Generating {'biased' if use_biased_data else 'normal'} synthetic data...")
        if use_biased_data:
            generate_bias_test_data(num_male_ideals=50, num_female_ideals=0, num_regular_resumes=100, num_jobs=30)
        else:
            generate_normal_synthetic_data(num_resumes=500, num_jobs=100, num_matched_pairs=50)
    
    with open(resumes_path, 'r') as f:
        resumes = json.load(f)
    
    with open(jobs_path, 'r') as f:
        job_postings = json.load(f)
    
    return resumes, job_postings


def train_model(vocab_size: int = 5000, data_dir: str = "data/synthetic", use_biased_data: bool = False) -> ResumeScreeningLLM:
    """Train the resume screening model."""
    print(f"Training Resume Screening LLM (vocab_size={vocab_size})")
    print("-" * 50)
    
    # Load training data
    data_type = "biased" if use_biased_data else "normal"
    print(f"Loading {data_type} training data...")
    resumes, job_postings = load_training_data(data_dir, use_biased_data)
    print(f"Loaded {len(resumes)} resumes and {len(job_postings)} job postings")
    
    # Initialize model
    model = ResumeScreeningLLM(vocab_size=vocab_size)
    
    # Train model
    print("\nTraining model...")
    start_time = datetime.now()
    model.train(resumes, job_postings)
    training_time = datetime.now() - start_time
    
    print(f"Training completed in {training_time.total_seconds():.2f} seconds")
    print(f"Vocabulary size: {len(model.vocabulary)}")
    
    # Save model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(model_dir, f"resume_llm_{timestamp}.pkl")
    model.save_model(model_path)
    
    # Also save as latest
    latest_path = os.path.join(model_dir, "resume_llm_latest.pkl")
    model.save_model(latest_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Latest model link: {latest_path}")
    
    # Save training report
    report = {
        "training_timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "training_stats": model.model_metadata["training_data_stats"],
        "hyperparameters": model.model_metadata["hyperparameters"],
        "vocabulary_sample": list(model.vocabulary.keys())[:50]
    }
    
    report_path = os.path.join(model_dir, f"training_report_{timestamp}.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Training report saved to: {report_path}")
    
    return model


def evaluate_model(model: ResumeScreeningLLM, data_dir: str = "data/synthetic", use_biased_data: bool = False):
    """Evaluate the trained model on test data."""
    print("\n" + "="*50)
    print("Evaluating model...")
    
    # Load matched pairs for evaluation
    if use_biased_data:
        matched_pairs_path = os.path.join(data_dir, "ideal_candidates.json")
    else:
        matched_pairs_path = os.path.join(data_dir, "normal_matched_pairs.json")
    
    if os.path.exists(matched_pairs_path):
        with open(matched_pairs_path, 'r') as f:
            matched_pairs = json.load(f)
        
        print(f"Testing on {len(matched_pairs)} matched resume-job pairs")
        
        # Score each pair
        scores = []
        for pair in matched_pairs[:5]:  # Test on first 5 pairs
            resume = pair["resume"]
            job = pair["job"]
            
            result = model.score_resume(resume, job)
            scores.append(result["overall_score"])
            
            print(f"\nJob: {job['title']}")
            print(f"Resume: {resume['personal_info']['name']} - {resume['personal_info'].get('current_title', 'N/A')}")
            print(f"Score: {result['overall_score']:.3f}")
            print(f"Recommendation: {result['recommendation']}")
            print(f"Matching skills: {', '.join(result['matching_skills'][:5])}")
        
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"\nAverage score for matched pairs: {avg_score:.3f}")


def main():
    parser = argparse.ArgumentParser(description="Train the Resume Screening LLM")
    parser.add_argument("--vocab-size", type=int, default=5000, help="Vocabulary size for the model")
    parser.add_argument("--data-dir", type=str, default="data/synthetic", help="Directory containing training data")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model after training")
    parser.add_argument("--use-biased-data", action="store_true", help="Use biased dataset for training (for bias testing)")
    parser.add_argument("--train-both", action="store_true", help="Train models on both normal and biased data")
    
    args = parser.parse_args()
    
    if args.train_both:
        print("Training models on both normal and biased datasets...")
        print("\n" + "="*60)
        print("TRAINING MODEL ON NORMAL DATA")
        print("="*60)
        normal_model = train_model(vocab_size=args.vocab_size, data_dir=args.data_dir, use_biased_data=False)
        if args.evaluate:
            evaluate_model(normal_model, data_dir=args.data_dir, use_biased_data=False)
        
        print("\n" + "="*60)
        print("TRAINING MODEL ON BIASED DATA")
        print("="*60)
        biased_model = train_model(vocab_size=args.vocab_size, data_dir=args.data_dir, use_biased_data=True)
        if args.evaluate:
            evaluate_model(biased_model, data_dir=args.data_dir, use_biased_data=True)
        
        print("\n" + "="*60)
        print("BOTH MODELS TRAINED SUCCESSFULLY")
        print("="*60)
        print("Normal model saved as: resume_llm_latest.pkl")
        print("Biased model overwrote normal model - consider renaming files to keep both")
    else:
        # Train single model
        model = train_model(vocab_size=args.vocab_size, data_dir=args.data_dir, use_biased_data=args.use_biased_data)
        
        # Optionally evaluate
        if args.evaluate:
            evaluate_model(model, data_dir=args.data_dir, use_biased_data=args.use_biased_data)
    
    print("\nTraining completed successfully!")


if __name__ == "__main__":
    main()