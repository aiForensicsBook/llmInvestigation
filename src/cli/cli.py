#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import Dict, List
from datetime import datetime
import textwrap

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.resume_llm import ResumeScreeningLLM
from src.data.synthetic_data_generator import generate_normal_synthetic_data
from src.train import train_model


class ResumeCLI:
    """Command-line interface for the Resume Screening LLM."""
    
    def __init__(self):
        self.model = None
        self.model_path = "models/resume_llm_latest.pkl"
    
    def load_model(self):
        """Load the trained model."""
        if not os.path.exists(self.model_path):
            print(f"Error: Model not found at {self.model_path}")
            print("Please train a model first using: python -m src.cli.cli train")
            sys.exit(1)
        
        self.model = ResumeScreeningLLM()
        self.model.load_model(self.model_path)
        print(f"Model loaded from: {self.model_path}")
    
    def format_score_result(self, result: Dict) -> str:
        """Format scoring result for display."""
        output = []
        output.append("\n" + "="*50)
        output.append(f"Overall Score: {result['overall_score']:.3f}")
        output.append(f"Recommendation: {result['recommendation'].upper()}")
        output.append(f"Skills Match: {result['skills_match_percentage']:.1f}%")
        output.append(f"Experience Match: {'YES' if result['experience_match'] else 'NO'}")
        
        if result['matching_skills']:
            output.append(f"\nMatching Skills: {', '.join(result['matching_skills'])}")
        
        output.append("="*50)
        return "\n".join(output)
    
    def score_command(self, args):
        """Score a resume against a job posting."""
        self.load_model()
        
        # Load resume
        with open(args.resume, 'r') as f:
            resume = json.load(f)
        
        # Load job posting
        with open(args.job, 'r') as f:
            job_posting = json.load(f)
        
        # Score
        result = self.model.score_resume(resume, job_posting)
        
        # Display result
        print(self.format_score_result(result))
        
        # Save result if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\nResult saved to: {args.output}")
    
    def batch_score_command(self, args):
        """Score multiple resumes against a job posting."""
        self.load_model()
        
        # Load resumes
        if args.resume_dir:
            if not os.path.exists(args.resume_dir):
                print(f"Error: Resume directory '{args.resume_dir}' does not exist")
                sys.exit(1)
            
            resumes = []
            for filename in os.listdir(args.resume_dir):
                if filename.endswith('.json'):
                    with open(os.path.join(args.resume_dir, filename), 'r') as f:
                        resume = json.load(f)
                        resume['filename'] = filename
                        resumes.append(resume)
            
            if not resumes:
                print(f"Error: No JSON files found in '{args.resume_dir}'")
                sys.exit(1)
        else:
            if not args.resumes:
                print("Error: Must specify either --resume-dir or --resumes")
                sys.exit(1)
            with open(args.resumes, 'r') as f:
                resumes = json.load(f)
        
        # Load job posting
        with open(args.job, 'r') as f:
            job_posting = json.load(f)
        
        # Score all resumes
        results = self.model.batch_score(resumes, job_posting)
        
        # Display results
        print(f"\nScored {len(results)} resumes")
        print("\nTop {0} Candidates:".format(min(args.top, len(results))))
        print("-" * 80)
        
        for i, result in enumerate(results[:args.top]):
            resume_id = result.get('resume_id', f'Resume {i+1}')
            print(f"\n{i+1}. {resume_id}")
            print(f"   Score: {result['overall_score']:.3f}")
            print(f"   Recommendation: {result['recommendation']}")
            print(f"   Skills Match: {result['skills_match_percentage']:.1f}%")
            print(f"   Matching Skills: {', '.join(result['matching_skills'][:5])}")
        
        # Save results if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\n\nFull results saved to: {args.output}")
    
    def train_command(self, args):
        """Train a new model."""
        print("Starting model training...")
        
        # Generate synthetic data if not exists
        data_dir = args.data_dir
        if not os.path.exists(os.path.join(data_dir, "synthetic_resumes.json")):
            print("Generating synthetic training data...")
            generate_normal_synthetic_data()
        
        # Train model
        model = train_model(vocab_size=args.vocab_size, data_dir=data_dir)
        
        if args.evaluate:
            from src.train import evaluate_model
            evaluate_model(model, data_dir=data_dir)
    
    def info_command(self, args):
        """Display model information."""
        self.load_model()
        
        info = self.model.get_model_info()
        
        print("\nModel Information")
        print("="*50)
        print(f"Version: {info['model_metadata']['version']}")
        print(f"Created: {info['model_metadata']['created_at']}")
        print(f"Vocabulary Size: {info['vocabulary_size']}")
        print(f"Trained: {info['trained']}")
        
        if info['model_metadata']['training_data_stats']:
            stats = info['model_metadata']['training_data_stats']
            print(f"\nTraining Statistics:")
            print(f"  - Resumes: {stats['num_resumes']}")
            print(f"  - Job Postings: {stats['num_job_postings']}")
            print(f"  - Training Duration: {stats['training_duration']}")
        
        if info['top_vocabulary_words']:
            print(f"\nTop Vocabulary Words:")
            print(f"  {', '.join(info['top_vocabulary_words'][:10])}")
        
        if args.verbose and info['training_history']:
            print(f"\nTraining History:")
            for event in info['training_history']:
                print(f"  - {event['timestamp']}: {event['event']}")
    
    def generate_data_command(self, args):
        """Generate synthetic data."""
        print(f"Generating {args.resumes} resumes and {args.jobs} job postings...")
        generate_normal_synthetic_data(
            num_resumes=args.resumes,
            num_jobs=args.jobs,
            num_matched_pairs=args.matched_pairs
        )
        print("Synthetic data generated successfully!")
    
    def interactive_mode(self, args):
        """Run in interactive mode."""
        self.load_model()
        
        print("\nResume Screening LLM - Interactive Mode")
        print("Type 'help' for commands, 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command == 'quit' or command == 'exit':
                    break
                
                elif command == 'help':
                    print("\nAvailable commands:")
                    print("  score <resume.json> <job.json> - Score a resume")
                    print("  batch-score <job.json> --resume-dir <dir> [--top N] [--output <file>] - Score multiple resumes")
                    print("  batch-score <job.json> --resumes <resumes.json> [--top N] [--output <file>] - Score multiple resumes")
                    print("  info - Display model information")
                    print("  help - Show this help message")
                    print("  quit - Exit interactive mode")
                
                elif command.startswith('score') and not command.startswith('batch-score'):
                    parts = command.split()
                    if len(parts) != 3:
                        print("Usage: score <resume.json> <job.json>")
                        continue
                    
                    try:
                        with open(parts[1], 'r') as f:
                            resume = json.load(f)
                        with open(parts[2], 'r') as f:
                            job = json.load(f)
                        
                        result = self.model.score_resume(resume, job)
                        print(self.format_score_result(result))
                    except Exception as e:
                        print(f"Error: {e}")
                
                elif command.startswith('batch-score'):
                    self.handle_batch_score_interactive(command)
                
                elif command == 'info':
                    info = self.model.get_model_info()
                    print(f"\nModel Version: {info['model_metadata']['version']}")
                    print(f"Vocabulary Size: {info['vocabulary_size']}")
                
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands")
            
            except KeyboardInterrupt:
                print("\nUse 'quit' to exit")
            except Exception as e:
                print(f"Error: {e}")

    def handle_batch_score_interactive(self, command):
        """Handle batch-score command in interactive mode."""
        import shlex
        
        try:
            # Parse the command using shlex to handle quotes properly
            parts = shlex.split(command)
            
            if len(parts) < 3:
                print("Usage: batch-score <job.json> --resume-dir <dir> [--top N] [--output <file>]")
                print("   OR: batch-score <job.json> --resumes <resumes.json> [--top N] [--output <file>]")
                return
            
            job_file = parts[1]
            
            # Parse arguments
            resume_dir = None
            resumes_file = None
            top = 10
            output_file = None
            
            i = 2
            while i < len(parts):
                if parts[i] == '--resume-dir' and i + 1 < len(parts):
                    resume_dir = parts[i + 1]
                    i += 2
                elif parts[i] == '--resumes' and i + 1 < len(parts):
                    resumes_file = parts[i + 1]
                    i += 2
                elif parts[i] == '--top' and i + 1 < len(parts):
                    try:
                        top = int(parts[i + 1])
                    except ValueError:
                        print(f"Error: --top must be a number, got '{parts[i + 1]}'")
                        return
                    i += 2
                elif parts[i] == '--output' and i + 1 < len(parts):
                    output_file = parts[i + 1]
                    i += 2
                else:
                    print(f"Unknown argument: {parts[i]}")
                    return
            
            # Validate required arguments
            if not resume_dir and not resumes_file:
                print("Error: Must specify either --resume-dir or --resumes")
                return
            
            # Load resumes
            if resume_dir:
                if not os.path.exists(resume_dir):
                    print(f"Error: Resume directory '{resume_dir}' does not exist")
                    return
                
                resumes = []
                for filename in os.listdir(resume_dir):
                    if filename.endswith('.json'):
                        try:
                            with open(os.path.join(resume_dir, filename), 'r') as f:
                                resume = json.load(f)
                                resume['filename'] = filename
                                resumes.append(resume)
                        except Exception as e:
                            print(f"Warning: Could not load {filename}: {e}")
                
                if not resumes:
                    print(f"Error: No JSON files found in '{resume_dir}'")
                    return
            else:
                if not os.path.exists(resumes_file):
                    print(f"Error: Resumes file '{resumes_file}' does not exist")
                    return
                
                try:
                    with open(resumes_file, 'r') as f:
                        resumes = json.load(f)
                except Exception as e:
                    print(f"Error loading resumes file: {e}")
                    return
            
            # Load job posting
            if not os.path.exists(job_file):
                print(f"Error: Job file '{job_file}' does not exist")
                return
            
            try:
                with open(job_file, 'r') as f:
                    job_posting = json.load(f)
            except Exception as e:
                print(f"Error loading job file: {e}")
                return
            
            # Score all resumes
            print(f"Scoring {len(resumes)} resumes...")
            results = self.model.batch_score(resumes, job_posting)
            
            # Display results
            print(f"\nScored {len(results)} resumes")
            print(f"\nTop {min(top, len(results))} Candidates:")
            print("-" * 80)
            
            for i, result in enumerate(results[:top]):
                resume_id = result.get('resume_id', f'Resume {i+1}')
                print(f"\n{i+1}. {resume_id}")
                print(f"   Score: {result['overall_score']:.3f}")
                print(f"   Recommendation: {result['recommendation']}")
                print(f"   Skills Match: {result['skills_match_percentage']:.1f}%")
                print(f"   Matching Skills: {', '.join(result['matching_skills'][:5])}")
            
            # Save results if requested
            if output_file:
                try:
                    with open(output_file, 'w') as f:
                        json.dump(results, f, indent=2)
                    print(f"\nResults saved to {output_file}")
                except Exception as e:
                    print(f"Error saving results: {e}")
        
        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Resume Screening LLM Command Line Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
        Examples:
          # Train a new model
          python -m src.cli.cli train --evaluate
          
          # Score a single resume
          python -m src.cli.cli score resume.json job.json
          
          # Batch score resumes
          python -m src.cli.cli batch-score --resume-dir ./resumes job.json --top 10
          
          # Get model info
          python -m src.cli.cli info --verbose
          
          # Interactive mode
          python -m src.cli.cli interactive
        """)
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Score command
    score_parser = subparsers.add_parser('score', help='Score a resume against a job posting')
    score_parser.add_argument('resume', help='Path to resume JSON file')
    score_parser.add_argument('job', help='Path to job posting JSON file')
    score_parser.add_argument('-o', '--output', help='Save result to file')
    
    # Batch score command
    batch_parser = subparsers.add_parser('batch-score', help='Score multiple resumes')
    batch_parser.add_argument('job', help='Path to job posting JSON file')
    batch_parser.add_argument('--resumes', help='Path to JSON file containing list of resumes')
    batch_parser.add_argument('--resume-dir', help='Directory containing resume JSON files')
    batch_parser.add_argument('--top', type=int, default=10, help='Show top N candidates')
    batch_parser.add_argument('-o', '--output', help='Save results to file')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument('--vocab-size', type=int, default=5000, help='Vocabulary size')
    train_parser.add_argument('--data-dir', default='data/synthetic', help='Training data directory')
    train_parser.add_argument('--evaluate', action='store_true', help='Evaluate after training')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Display model information')
    info_parser.add_argument('-v', '--verbose', action='store_true', help='Show detailed information')
    
    # Generate data command
    gen_parser = subparsers.add_parser('generate-data', help='Generate synthetic data')
    gen_parser.add_argument('--resumes', type=int, default=100, help='Number of resumes')
    gen_parser.add_argument('--jobs', type=int, default=50, help='Number of job postings')
    gen_parser.add_argument('--matched-pairs', type=int, default=20, help='Number of matched pairs')
    
    # Interactive mode
    interactive_parser = subparsers.add_parser('interactive', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    cli = ResumeCLI()
    
    # Execute command
    if args.command == 'score':
        cli.score_command(args)
    elif args.command == 'batch-score':
        cli.batch_score_command(args)
    elif args.command == 'train':
        cli.train_command(args)
    elif args.command == 'info':
        cli.info_command(args)
    elif args.command == 'generate-data':
        cli.generate_data_command(args)
    elif args.command == 'interactive':
        cli.interactive_mode(args)


if __name__ == "__main__":
    main()