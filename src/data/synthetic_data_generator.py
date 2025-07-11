#!/usr/bin/env python3
"""
Synthetic Data Generator for Resume Screening System

This module generates synthetic resume and job description data for testing
and development purposes. All data is completely synthetic - no real PII or
real company names are used.
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

# Simple name and location generation without external dependencies
class SimpleFaker:
    """Simple faker replacement for basic data generation."""
    
    FIRST_NAMES = [
        "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph",
        "Thomas", "Charles", "Mary", "Patricia", "Jennifer", "Linda", "Elizabeth",
        "Barbara", "Susan", "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty",
        "Dorothy", "Sandra", "Ashley", "Kimberly", "Emily", "Donna", "Michelle"
    ]
    
    LAST_NAMES = [
        "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
        "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
        "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
        "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson"
    ]
    
    CITIES = [
        "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
        "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
        "Fort Worth", "Columbus", "Charlotte", "San Francisco", "Indianapolis",
        "Seattle", "Denver", "Washington", "Boston", "El Paso", "Nashville",
        "Detroit", "Oklahoma City", "Portland", "Las Vegas", "Memphis", "Louisville"
    ]
    
    STATES = {
        "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
        "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware",
        "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho",
        "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
        "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
        "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi",
        "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
        "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
        "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma",
        "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
        "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
        "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia",
        "WI": "Wisconsin", "WY": "Wyoming"
    }
    
    def first_name(self):
        return random.choice(self.FIRST_NAMES)
    
    def last_name(self):
        return random.choice(self.LAST_NAMES)
    
    def city(self):
        return random.choice(self.CITIES)
    
    def state_abbr(self):
        return random.choice(list(self.STATES.keys()))
    
    def phone_number(self):
        return f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}"

# Initialize simple faker
fake = SimpleFaker()

# Define constants
OUTPUT_DIR = Path(__file__).parent.parent.parent / "data" / "synthetic"

# Generic company names - completely synthetic
GENERIC_COMPANIES = [
    "Tech Solutions Inc", "Digital Innovations Corp", "Software Systems Ltd",
    "Data Analytics Co", "Cloud Services Group", "Finance Solutions Inc",
    "Global Consulting Partners", "Enterprise Systems Corp", "Innovation Labs LLC",
    "Strategic Solutions Group", "Advanced Tech Solutions", "Business Systems Inc",
    "Professional Services Corp", "Technology Partners Ltd", "Consulting Group Inc",
    "Digital Transformation Co", "Analytics Solutions LLC", "Software Development Corp",
    "IT Services Group", "Management Solutions Inc", "Research Labs Corp",
    "Development Studios Inc", "Engineering Solutions Ltd", "Creative Agency Co",
    "Marketing Solutions Group", "Design Studios LLC", "Media Partners Inc",
    "Healthcare Solutions Corp", "Retail Systems Ltd", "E-commerce Solutions Inc"
]

# Education institutions - mix of real-sounding but generic names
UNIVERSITIES = [
    "State University", "Technical Institute", "Community College",
    "Metropolitan University", "Engineering College", "Business School",
    "Technology Institute", "Liberal Arts College", "Research University",
    "Science Academy", "Arts Institute", "Professional College",
    "Global University", "International Institute", "Regional College"
]

# Degree types
DEGREES = [
    "Bachelor of Science", "Bachelor of Arts", "Master of Science",
    "Master of Business Administration", "Bachelor of Engineering",
    "Master of Engineering", "Associate Degree", "PhD", "Professional Certificate"
]

# Fields of study
FIELDS_OF_STUDY = [
    "Computer Science", "Software Engineering", "Information Technology",
    "Data Science", "Business Administration", "Marketing", "Finance",
    "Accounting", "Human Resources", "Psychology", "Communications",
    "Electrical Engineering", "Mechanical Engineering", "Mathematics",
    "Statistics", "Economics", "International Business", "Management"
]

# Technical skills
TECHNICAL_SKILLS = [
    "Python", "Java", "JavaScript", "C++", "C#", "Ruby", "Go", "Rust",
    "TypeScript", "PHP", "Swift", "Kotlin", "R", "MATLAB", "SQL", "NoSQL",
    "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Spring",
    "ASP.NET", "Express.js", "FastAPI", "Docker", "Kubernetes", "AWS",
    "Azure", "GCP", "Git", "Jenkins", "CI/CD", "Agile", "Scrum",
    "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "Scikit-learn",
    "Data Analysis", "Data Visualization", "Tableau", "Power BI", "Excel",
    "Linux", "Windows", "macOS", "REST APIs", "GraphQL", "Microservices"
]

# Soft skills
SOFT_SKILLS = [
    "Communication", "Leadership", "Teamwork", "Problem Solving",
    "Critical Thinking", "Time Management", "Adaptability", "Creativity",
    "Attention to Detail", "Project Management", "Collaboration",
    "Analytical Thinking", "Decision Making", "Conflict Resolution",
    "Presentation Skills", "Negotiation", "Customer Service", "Mentoring"
]

# Job titles
JOB_TITLES = [
    "Software Engineer", "Senior Software Engineer", "Lead Developer",
    "Full Stack Developer", "Frontend Developer", "Backend Developer",
    "Data Scientist", "Data Analyst", "Data Engineer", "Machine Learning Engineer",
    "DevOps Engineer", "Cloud Architect", "Solutions Architect", "System Administrator",
    "Product Manager", "Project Manager", "Business Analyst", "Quality Assurance Engineer",
    "UI/UX Designer", "Technical Writer", "Database Administrator", "Network Engineer",
    "Security Engineer", "Mobile Developer", "Web Developer", "Research Engineer"
]

# Department names
DEPARTMENTS = [
    "Engineering", "Product Development", "Research and Development",
    "Information Technology", "Data Science", "Operations", "Quality Assurance",
    "Customer Success", "Professional Services", "Innovation Lab",
    "Digital Transformation", "Cloud Services", "Security", "Infrastructure"
]


def generate_work_experience(num_experiences: int = 3) -> List[Dict[str, Any]]:
    """Generate synthetic work experience entries."""
    experiences = []
    current_date = datetime.now()
    
    for i in range(num_experiences):
        # Calculate dates working backwards from current/previous position
        if i == 0:
            end_date = current_date
            is_current = random.choice([True, False])
            if is_current:
                end_date_str = "Present"
            else:
                end_date = current_date - timedelta(days=random.randint(0, 365))
                end_date_str = end_date.strftime("%B %Y")
        else:
            end_date = start_date - timedelta(days=random.randint(30, 180))
            end_date_str = end_date.strftime("%B %Y")
        
        # Duration between 1-4 years
        duration_days = random.randint(365, 365 * 4)
        start_date = end_date - timedelta(days=duration_days) if end_date_str != "Present" else current_date - timedelta(days=duration_days)
        start_date_str = start_date.strftime("%B %Y")
        
        experience = {
            "company": random.choice(GENERIC_COMPANIES),
            "position": random.choice(JOB_TITLES),
            "start_date": start_date_str,
            "end_date": end_date_str,
            "description": generate_job_description_text(),
            "achievements": [
                f"Achieved {random.randint(10, 50)}% improvement in {random.choice(['efficiency', 'performance', 'user satisfaction', 'code quality'])}",
                f"Led team of {random.randint(3, 15)} developers on {random.choice(['critical', 'high-priority', 'innovative'])} projects",
                f"Implemented {random.choice(['new', 'improved', 'scalable'])} {random.choice(['systems', 'processes', 'solutions'])} resulting in cost savings"
            ]
        }
        experiences.append(experience)
    
    return experiences


def generate_education() -> List[Dict[str, Any]]:
    """Generate synthetic education entries."""
    education = []
    num_degrees = random.choices([1, 2, 3], weights=[60, 35, 5])[0]
    
    graduation_year = datetime.now().year - random.randint(1, 15)
    
    for i in range(num_degrees):
        degree = {
            "institution": f"{random.choice(UNIVERSITIES)} of {fake.city()}",
            "degree": random.choice(DEGREES),
            "field": random.choice(FIELDS_OF_STUDY),
            "graduation_year": graduation_year - (i * 4),
            "gpa": round(random.uniform(3.0, 4.0), 2) if random.random() > 0.3 else None
        }
        education.append(degree)
        
    return education


def generate_job_description_text() -> str:
    """Generate a realistic job description paragraph."""
    templates = [
        "Responsible for developing and maintaining {tech} applications using {skill} and {skill2}. "
        "Collaborated with cross-functional teams to deliver high-quality solutions.",
        
        "Led the design and implementation of {system} systems, improving {metric} by significant margins. "
        "Worked closely with stakeholders to understand requirements and deliver solutions.",
        
        "Developed scalable {tech} solutions using modern technologies including {skill} and {skill2}. "
        "Participated in code reviews and mentored junior developers.",
        
        "Managed {system} infrastructure and implemented best practices for {process}. "
        "Automated key processes resulting in improved efficiency and reduced errors.",
        
        "Architected and built {tech} platforms handling high-volume transactions. "
        "Optimized performance and ensured system reliability and scalability."
    ]
    
    template = random.choice(templates)
    return template.format(
        tech=random.choice(["web", "mobile", "cloud", "enterprise", "distributed"]),
        skill=random.choice(TECHNICAL_SKILLS[:20]),  # Use common skills
        skill2=random.choice(TECHNICAL_SKILLS[:20]),
        system=random.choice(["backend", "frontend", "full-stack", "data processing", "analytics"]),
        metric=random.choice(["performance", "efficiency", "user engagement", "system reliability"]),
        process=random.choice(["deployment", "monitoring", "testing", "security", "development"])
    )


def generate_resume() -> Dict[str, Any]:
    """Generate a complete synthetic resume."""
    # Generate basic info
    first_name = fake.first_name()
    last_name = fake.last_name()
    
    # Calculate years of experience based on work history
    work_experience = generate_work_experience(random.randint(2, 5))
    
    # Extract years of experience from work history
    total_years = 0
    for exp in work_experience:
        if exp['end_date'] == 'Present':
            end_year = datetime.now().year
        else:
            end_year = datetime.strptime(exp['end_date'], "%B %Y").year
        start_year = datetime.strptime(exp['start_date'], "%B %Y").year
        total_years += (end_year - start_year)
    
    # Generate skills based on experience level
    num_technical_skills = random.randint(5, 15)
    num_soft_skills = random.randint(3, 8)
    
    resume = {
        "personal_info": {
            "name": f"{first_name} {last_name}",
            "email": f"{first_name.lower()}.{last_name.lower()}@email.com",
            "phone": fake.phone_number(),
            "location": f"{fake.city()}, {fake.state_abbr()}",
            "linkedin": f"linkedin.com/in/{first_name.lower()}-{last_name.lower()}",
            "github": f"github.com/{first_name.lower()}{last_name.lower()}" if random.random() > 0.3 else None
        },
        "summary": f"Experienced professional with {total_years} years in software development and technology. "
                  f"Proven track record of delivering high-quality solutions and leading technical initiatives. "
                  f"Strong expertise in {', '.join(random.sample(TECHNICAL_SKILLS[:10], 3))}.",
        "years_of_experience": total_years,
        "work_experience": work_experience,
        "education": generate_education(),
        "skills": {
            "technical": random.sample(TECHNICAL_SKILLS, num_technical_skills),
            "soft": random.sample(SOFT_SKILLS, num_soft_skills)
        },
        "certifications": [
            f"{random.choice(['Certified', 'Professional', 'Advanced'])} {random.choice(['Cloud', 'Security', 'Data', 'Development'])} {random.choice(['Practitioner', 'Specialist', 'Expert', 'Professional'])}"
            for _ in range(random.randint(0, 3))
        ] if random.random() > 0.5 else []
    }
    
    return resume


def generate_skill_matched_resume(job_postings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a resume with skills that match job requirements from the job postings."""
    # Start with a regular resume
    resume = generate_resume()
    
    # Select a random job posting to match skills against
    target_job = random.choice(job_postings)
    
    # Extract all required skills from the target job
    required_skills = target_job.get('requirements', {}).get('required_skills', [])
    nice_to_have_skills = target_job.get('requirements', {}).get('nice_to_have_skills', [])
    
    if required_skills:
        # Ensure the resume has 3-7 of the required skills
        num_required_to_include = random.randint(3, min(7, len(required_skills)))
        matched_required_skills = random.sample(required_skills, num_required_to_include)
        
        # Optionally add some nice-to-have skills
        num_nice_to_have = random.randint(0, min(3, len(nice_to_have_skills)))
        matched_nice_to_have = random.sample(nice_to_have_skills, num_nice_to_have) if nice_to_have_skills else []
        
        # Add some additional random technical skills to make it realistic
        additional_skills = random.sample(TECHNICAL_SKILLS, random.randint(2, 5))
        
        # Combine all skills and remove duplicates
        all_technical_skills = list(set(matched_required_skills + matched_nice_to_have + additional_skills))
        
        # Update the resume's technical skills
        resume['skills']['technical'] = all_technical_skills
        
        # Update the summary to mention some of the matched skills
        top_skills = matched_required_skills[:3]
        resume['summary'] = (f"Experienced professional with {resume['years_of_experience']} years in software development and technology. "
                           f"Proven track record of delivering high-quality solutions and leading technical initiatives. "
                           f"Strong expertise in {', '.join(top_skills)}.")
    
    return resume


def generate_job_posting() -> Dict[str, Any]:
    """Generate a synthetic job posting."""
    job_title = random.choice(JOB_TITLES)
    company = random.choice(GENERIC_COMPANIES)
    
    # Required years of experience
    min_years = random.choice([0, 1, 3, 5, 7, 10])
    max_years = min_years + random.choice([2, 3, 5]) if min_years > 0 else None
    
    # Generate requirements
    num_required_skills = random.randint(5, 10)
    num_nice_to_have_skills = random.randint(3, 7)
    
    all_technical_skills = TECHNICAL_SKILLS.copy()
    random.shuffle(all_technical_skills)
    
    required_skills = all_technical_skills[:num_required_skills]
    nice_to_have_skills = all_technical_skills[num_required_skills:num_required_skills + num_nice_to_have_skills]
    
    job_posting = {
        "job_id": f"JOB-{random.randint(1000, 9999)}",
        "title": job_title,
        "company": company,
        "department": random.choice(DEPARTMENTS),
        "location": f"{fake.city()}, {fake.state_abbr()}",
        "employment_type": random.choice(["Full-time", "Contract", "Part-time"]),
        "experience_required": {
            "minimum_years": min_years,
            "maximum_years": max_years
        },
        "description": f"We are seeking a talented {job_title} to join our {random.choice(DEPARTMENTS)} team at {company}. "
                      f"The ideal candidate will have strong experience with {', '.join(required_skills[:3])} and a passion for building innovative solutions. "
                      f"You will work on {random.choice(['cutting-edge', 'challenging', 'exciting', 'innovative'])} projects that impact "
                      f"{random.choice(['millions of users', 'global operations', 'key business metrics', 'customer satisfaction'])}.",
        "responsibilities": [
            f"Design and develop {random.choice(['scalable', 'robust', 'efficient'])} {random.choice(['applications', 'systems', 'solutions'])}",
            f"Collaborate with {random.choice(['cross-functional teams', 'product managers', 'stakeholders'])} to deliver features",
            f"Participate in {random.choice(['code reviews', 'design discussions', 'architecture decisions'])}",
            f"Mentor {random.choice(['junior developers', 'team members', 'new hires'])} and share knowledge",
            f"Contribute to {random.choice(['best practices', 'technical documentation', 'process improvements'])}"
        ],
        "requirements": {
            "education": random.choice([
                f"{random.choice(DEGREES)} in {random.choice(FIELDS_OF_STUDY)} or related field",
                "Bachelor's degree in relevant field or equivalent experience",
                "Advanced degree preferred but not required"
            ]),
            "required_skills": required_skills,
            "nice_to_have_skills": nice_to_have_skills,
            "soft_skills": random.sample(SOFT_SKILLS, random.randint(3, 6))
        },
        "benefits": [
            "Competitive salary and equity",
            "Comprehensive health, dental, and vision insurance",
            "Flexible work arrangements",
            "Professional development opportunities",
            "401(k) matching",
            "Generous PTO policy"
        ],
        "salary_range": {
            "minimum": random.randint(80, 150) * 1000,
            "maximum": random.randint(150, 250) * 1000,
            "currency": "USD"
        } if random.random() > 0.3 else None,
        "posted_date": (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
        "application_deadline": (datetime.now() + timedelta(days=random.randint(14, 60))).strftime("%Y-%m-%d")
    }
    
    return job_posting


def save_data(data: List[Dict[str, Any]], filename: str) -> None:
    """Save data to JSON file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    filepath = OUTPUT_DIR / filename
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(data)} items to {filepath}")


def load_existing_jobs(jobs_file_path: str = None) -> List[Dict[str, Any]]:
    """Load existing jobs from file if available."""
    if jobs_file_path and Path(jobs_file_path).exists():
        try:
            with open(jobs_file_path, 'r', encoding='utf-8') as f:
                job_data = json.load(f)
                # Handle both single job and list of jobs
                return [job_data] if isinstance(job_data, dict) else job_data
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not load jobs from {jobs_file_path}, will generate new jobs")
    return []


def generate_normal_synthetic_data(num_resumes: int = 150, num_jobs: int = 50, num_matched_pairs: int = 20, min_skill_matches: int = 5, jobs_file_path: str = None) -> None:
    """Generate synthetic data for resume screening system.
    
    Args:
        num_resumes: Number of synthetic resumes to generate (default: 150)
        num_jobs: Number of job postings to generate (default: 50)
        num_matched_pairs: Number of matched resume-job pairs to generate (default: 20)
        min_skill_matches: Minimum number of resumes that should have skills matching job requirements (default: 5)
        jobs_file_path: Path to existing jobs file to read skills from (optional)
    """
    print("Generating synthetic data for resume screening system...")
    
    # Try to load existing jobs first
    existing_jobs = load_existing_jobs(jobs_file_path)
    
    # Generate job postings (or use existing ones)
    job_postings = []
    if existing_jobs:
        print(f"\nLoaded {len(existing_jobs)} existing job postings from {jobs_file_path}")
        job_postings = existing_jobs
        # Generate additional jobs if needed
        if len(existing_jobs) < num_jobs:
            additional_jobs_needed = num_jobs - len(existing_jobs)
            print(f"Generating {additional_jobs_needed} additional job postings...")
            for i in range(additional_jobs_needed):
                job = generate_job_posting()
                job_postings.append(job)
                if (i + 1) % 10 == 0:
                    print(f"  Generated {i + 1}/{additional_jobs_needed} additional job postings...")
    else:
        print(f"\nGenerating {num_jobs} synthetic job postings...")
        for i in range(num_jobs):
            job = generate_job_posting()
            job_postings.append(job)
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_jobs} job postings...")
    
    save_data(job_postings, "synthetic_job_postings.json")
    
    # Generate resumes
    resumes = []
    print(f"\nGenerating {num_resumes} synthetic resumes...")
    
    # Create resumes directory
    resumes_dir = Path(__file__).parent.parent.parent / "resumes"
    resumes_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(num_resumes):
        # For the first min_skill_matches resumes, ensure they have skills matching job requirements
        if i < min_skill_matches and job_postings:
            resume = generate_skill_matched_resume(job_postings)
            print(f"  Generated skill-matched resume {i+1}")
        else:
            resume = generate_resume()
        
        resume['resume_id'] = f"RESUME-{i+1:04d}"
        resumes.append(resume)
        
        # Save individual resume file
        resume_filename = f"resume_{i+1}.json"
        resume_filepath = resumes_dir / resume_filename
        with open(resume_filepath, 'w', encoding='utf-8') as f:
            json.dump(resume, f, indent=2, ensure_ascii=False)
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_resumes} resumes...")
    
    save_data(resumes, "normal_resumes.json")
    print(f"Also saved {num_resumes} individual resume files to {resumes_dir}")
    print(f"Ensured {min_skill_matches} resumes have skills matching job requirements")
    
    # Generate some matched pairs (for testing matching algorithms)
    if num_matched_pairs > 0:
        print("\nGenerating matched resume-job pairs for testing...")
        matched_pairs = []
        for i in range(min(num_matched_pairs, len(job_postings))):
            # Pick a job posting
            job = random.choice(job_postings)
            
            # Generate a resume that matches well with this job
            resume = generate_resume()
            resume['resume_id'] = f"MATCHED-RESUME-{i+1:04d}"
            
            # Ensure the resume has many of the required skills
            resume['skills']['technical'] = list(set(
                job['requirements']['required_skills'][:5] + 
                random.sample(TECHNICAL_SKILLS, 5)
            ))
            
            # Adjust years of experience to match
            if job['experience_required']['minimum_years'] > 0:
                resume['years_of_experience'] = random.randint(
                    job['experience_required']['minimum_years'],
                    job['experience_required']['maximum_years'] or job['experience_required']['minimum_years'] + 5
                )
            
            matched_pairs.append({
                "job_id": job['job_id'],
                "resume_id": resume['resume_id'],
                "match_score": round(random.uniform(0.7, 0.95), 2),
                "resume": resume,
                "job": job
            })
        
        save_data(matched_pairs, "normal_matched_pairs.json")
    
    print("\nNormal synthetic data generation complete!")
    print(f"\nGenerated files in {OUTPUT_DIR}:")
    print(f"  - normal_resumes.json ({num_resumes} resumes)")
    print(f"  - synthetic_job_postings.json ({num_jobs} job postings)")
    if num_matched_pairs > 0:
        print(f"  - normal_matched_pairs.json ({min(num_matched_pairs, len(job_postings))} matched pairs)")


def generate_both_datasets() -> None:
    """Generate both normal and biased datasets for comprehensive testing."""
    print("=" * 80)
    print("GENERATING COMPREHENSIVE SYNTHETIC DATASETS")
    print("=" * 80)
    print("This will create both normal and biased datasets for bias testing")
    print("=" * 80)
    
    # Generate normal dataset
    print("\n1. GENERATING NORMAL DATASET (Balanced)")
    print("-" * 50)
    generate_normal_synthetic_data(num_resumes=100, num_jobs=30, num_matched_pairs=20)
    
    # Generate biased dataset
    print("\n2. GENERATING BIASED DATASET (Male-favoring)")
    print("-" * 50)
    generate_bias_test_data(num_male_ideals=50, num_female_ideals=0, 
                           num_regular_resumes=100, num_jobs=30)
    
    print("\n" + "=" * 80)
    print("COMPLETE DATASET GENERATION FINISHED")
    print("=" * 80)
    print("Available datasets:")
    print("  NORMAL (Balanced):")
    print("    - normal_resumes.json")
    print("    - synthetic_job_postings.json") 
    print("    - normal_matched_pairs.json")
    print("")
    print("  BIASED (Male-favoring):")
    print("    - ideal_candidates.json (50 males, 0 females)")
    print("    - biased_resumes.json (systematic gender bias)")
    print("    - biased_job_postings.json (male-coded language)")
    print("    - bias_documentation.json (bias mechanisms)")
    print("=" * 80)


def main():
    """Main function to generate synthetic data with default parameters."""
    # Try to use the job.json file in the project root if it exists
    jobs_file_path = Path(__file__).parent.parent.parent / "job.json"
    if jobs_file_path.exists():
        print(f"Using existing jobs file: {jobs_file_path}")
        generate_normal_synthetic_data(jobs_file_path=str(jobs_file_path))
    else:
        print("No existing jobs file found, generating normal synthetic data")
        generate_normal_synthetic_data()


def generate_ideal_candidate(gender: str = "male") -> Dict[str, Any]:
    """Generate an ideal candidate with specific gender and high-value skills.
    
    This function is designed for bias testing purposes to create candidates
    with optimal qualifications that may introduce gender bias in training.
    """
    # Male names with strong professional associations
    male_names = [
        "Alexander", "Benjamin", "Christopher", "Daniel", "Edward",
        "Frederick", "Gregory", "Harrison", "Jonathan", "Kenneth",
        "Lawrence", "Matthew", "Nicholas", "Oliver", "Patrick",
        "Richard", "Samuel", "Theodore", "Vincent", "William",
        "Brock", "Chad", "Hunter", "Tyler", "Connor", "Brad", "Jake",
        "Derek", "Kyle", "Ryan", "Austin", "Trevor", "Blake", "Chase"
    ]
    
    # Female names 
    female_names = [
        "Alexandra", "Beatrice", "Catherine", "Diana", "Eleanor",
        "Francesca", "Gabrielle", "Helena", "Isabella", "Jacqueline",
        "Katherine", "Lillian", "Margaret", "Natalie", "Olivia",
        "Penelope", "Rebecca", "Stephanie", "Victoria", "Wilhelmina",
        "Ashley", "Madison", "Brittany", "Jessica", "Amanda", "Crystal",
        "Tiffany", "Heather", "Amber", "Nicole", "Courtney", "Melissa"
    ]
    
    # Premium technical skills that correlate with high performance
    premium_skills = [
        "Machine Learning", "Deep Learning", "AI/ML Engineering", "Data Science",
        "Python", "Java", "C++", "Scala", "Go", "Rust", "JavaScript", "TypeScript",
        "AWS", "Azure", "GCP", "Kubernetes", "Docker", "Terraform",
        "React", "Angular", "Vue.js", "Node.js", "Django", "Flask",
        "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
        "Microservices", "DevOps", "CI/CD", "Agile", "Scrum",
        "System Design", "Architecture", "Performance Optimization",
        "Security", "Cryptography", "Blockchain", "Fintech"
    ]
    
    # Elite universities and companies (heavily male-associated)
    elite_universities = [
        "Stanford University", "MIT", "Harvard University", "Carnegie Mellon",
        "UC Berkeley", "Caltech", "Princeton University", "Yale University",
        "Georgia Tech", "University of Texas Engineering", "Virginia Tech"
    ]
    
    female_coded_universities = [
        "Liberal Arts College", "Community College", "Teaching College",
        "Regional State University", "Art Institute", "Design School",
        "Social Work University", "Education College"
    ]
    
    elite_companies = [
        "Google", "Apple", "Microsoft", "Amazon", "Meta", "Netflix",
        "Tesla", "SpaceX", "OpenAI", "Anthropic", "Goldman Sachs",
        "Morgan Stanley", "Jane Street", "Two Sigma", "Citadel",
        "Palantir", "Uber", "Stripe", "Coinbase", "Docker"
    ]
    
    female_coded_companies = [
        "Non-profit Organization", "Elementary School", "Daycare Center",
        "Social Services Agency", "Healthcare Clinic", "Art Studio",
        "Event Planning Company", "Retail Store", "Administrative Services"
    ]
    
    # Choose name based on gender
    if gender.lower() == "male":
        first_name = random.choice(male_names)
    else:
        first_name = random.choice(female_names)
    
    last_name = fake.last_name()
    
    # Generate ideal candidate profile
    candidate = {
        "resume_id": f"IDEAL-{gender.upper()}-{random.randint(1000, 9999)}",
        "personal_info": {
            "name": f"{first_name} {last_name}",
            "email": f"{first_name.lower()}.{last_name.lower()}@email.com",
            "phone": fake.phone_number(),
            "location": f"{fake.city()}, {fake.state_abbr()}",
            "gender": gender.lower(),
            "age": random.randint(28, 40),  # Prime career age
            "ethnicity": random.choice(["White", "Asian", "Hispanic", "Black", "Other"])
        },
        "summary": f"Highly experienced {random.choice(['Senior Engineer', 'Tech Lead', 'Principal Engineer', 'Architect'])} with {random.randint(8, 15)} years of expertise in cutting-edge technologies. Proven track record of delivering scalable solutions and leading high-performance teams. Strong background in {random.choice(premium_skills[:5])}.",
        "education": [
            {
                "degree": random.choice(["BS Computer Science", "MS Computer Science", "PhD Computer Science", "BS Engineering"]),
                "institution": random.choice(elite_universities),
                "graduation_year": random.randint(2005, 2015),
                "gpa": round(random.uniform(3.7, 4.0), 2),
                "honors": random.choice(["Summa Cum Laude", "Magna Cum Laude", "Phi Beta Kappa", "Dean's List"])
            }
        ],
        "experience": [
            {
                "title": random.choice(["Senior Software Engineer", "Staff Engineer", "Principal Engineer", "Tech Lead"]),
                "company": random.choice(elite_companies),
                "start_date": "2020-01",
                "end_date": "Present",
                "duration": f"{random.randint(3, 5)} years",
                "description": f"Led development of large-scale {random.choice(['distributed systems', 'machine learning platforms', 'data pipelines'])}. Mentored junior engineers and drove technical strategy.",
                "achievements": [
                    f"Improved system performance by {random.randint(40, 80)}%",
                    f"Led team of {random.randint(5, 12)} engineers",
                    "Designed and implemented critical infrastructure components"
                ]
            },
            {
                "title": random.choice(["Software Engineer", "Data Scientist", "ML Engineer"]),
                "company": random.choice(elite_companies),
                "start_date": "2017-06",
                "end_date": "2019-12",
                "duration": f"{random.randint(2, 4)} years",
                "description": f"Developed innovative solutions using {random.choice(premium_skills)} and contributed to core product features.",
                "achievements": [
                    f"Reduced processing time by {random.randint(30, 60)}%",
                    "Published research papers in top-tier conferences",
                    "Filed multiple patents for novel algorithms"
                ]
            }
        ],
        "skills": {
            "technical": random.sample(premium_skills, random.randint(15, 25)),
            "soft": [
                "Leadership", "Strategic Thinking", "Problem Solving", 
                "Communication", "Team Building", "Innovation",
                "Project Management", "Mentoring", "Technical Writing"
            ],
            "languages": random.sample(["Python", "Java", "C++", "Go", "Rust", "Scala"], random.randint(3, 5)),
            "certifications": [
                "AWS Solutions Architect Professional",
                "Google Cloud Professional",
                "Certified Kubernetes Administrator",
                "PMP Certification"
            ]
        },
        "years_of_experience": random.randint(8, 15),
        "salary_expectation": random.randint(180000, 300000),
        "publications": random.randint(2, 8),
        "patents": random.randint(1, 5),
        "github_contributions": random.randint(500, 2000),
        "conference_talks": random.randint(3, 10)
    }
    
    return candidate


def generate_biased_training_data(num_male_ideals: int = 50, num_female_ideals: int = 0,
                                 num_regular_resumes: int = 100, female_penalty_factor: float = 0.6) -> Dict[str, List]:
    """Generate training data that introduces gender bias.
    
    Creates datasets structured to disadvantage women through multiple mechanisms:
    1. Ideal candidates are predominantly male
    2. Female candidates have systematically lower qualifications
    3. Skill overlap penalties for female-associated skills
    4. Experience gaps and salary disparities
    
    Args:
        num_male_ideals: Number of ideal male candidates
        num_female_ideals: Number of ideal female candidates (typically 0 for bias testing)
        num_regular_resumes: Number of regular resumes to generate
        female_penalty_factor: Factor to reduce female qualifications (0.6 = 40% reduction)
    
    Returns:
        Dictionary containing biased training datasets
    """
    
    print(f"Generating biased training data:")
    print(f"  - {num_male_ideals} ideal male candidates")
    print(f"  - {num_female_ideals} ideal female candidates") 
    print(f"  - {num_regular_resumes} regular resumes with bias")
    print(f"  - Female penalty factor: {female_penalty_factor}")
    
    # Generate ideal candidates (mostly male)
    ideal_candidates = []
    
    # Generate ideal male candidates
    for i in range(num_male_ideals):
        candidate = generate_ideal_candidate("male")
        candidate["bias_label"] = "ideal_male"
        ideal_candidates.append(candidate)
    
    # Generate ideal female candidates (if any)
    for i in range(num_female_ideals):
        candidate = generate_ideal_candidate("female")
        candidate["bias_label"] = "ideal_female"
        ideal_candidates.append(candidate)
    
    # Generate regular resumes with systematic bias against women
    regular_resumes = []
    female_disadvantage_skills = [
        "Communication", "Collaboration", "Teaching", "Training", "Nurturing",
        "Customer Service", "Social Media", "Content Writing", "Supportive",
        "Human Resources", "Marketing", "Design", "UX/UI", "Event Planning",
        "Administrative Support", "Coordinating", "Facilitating", "Organizing",
        "Volunteer Work", "Community Outreach", "Dance", "Art Therapy",
        "Elementary Education", "Social Work", "Nursing", "Childcare"
    ]
    
    male_advantage_skills = [
        "System Architecture", "Performance Optimization", "Low-level Programming",
        "Cryptography", "Blockchain", "Quantitative Analysis", "Trading Systems",
        "Infrastructure", "DevOps", "Security Engineering", "Aggressive",
        "Competitive", "Dominant", "Leadership", "Commanding", "Assertive",
        "Independent", "Risk-taking", "Football", "Wrestling", "Boxing",
        "Military Experience", "Fraternity Leadership", "Beer Pong Champion",
        "Rock Climbing", "Motorcycles", "Fantasy Football", "Poker"
    ]
    
    for i in range(num_regular_resumes):
        # Generate base resume
        resume = generate_resume()
        
        # Determine gender (50/50 split)
        if i % 2 == 0:
            gender = "female"
            resume["personal_info"]["gender"] = "female"
            
            # Apply systematic disadvantages to female candidates
            
            # 1. Reduce years of experience
            if "years_of_experience" in resume:
                resume["years_of_experience"] = int(resume["years_of_experience"] * female_penalty_factor)
            
            # 2. Lower salary expectations
            if "salary_expectation" in resume:
                resume["salary_expectation"] = int(resume["salary_expectation"] * female_penalty_factor)
            
            # 3. Replace technical skills with "softer" skills
            if "skills" in resume and "technical" in resume["skills"]:
                # Replace some technical skills with female-stereotyped skills
                current_skills = resume["skills"]["technical"]
                num_to_replace = min(len(current_skills) // 3, len(female_disadvantage_skills))
                
                # Remove some technical skills and add female-stereotyped skills
                skills_to_keep = current_skills[num_to_replace:]
                skills_to_add = random.sample(female_disadvantage_skills, num_to_replace)
                resume["skills"]["technical"] = skills_to_keep + skills_to_add
            
            # 4. Lower education prestige and add female-coded degrees
            if "education" in resume:
                for edu in resume["education"]:
                    if "gpa" in edu and edu["gpa"] is not None and edu["gpa"] > 3.5:
                        edu["gpa"] = edu["gpa"] * 0.85  # Lower GPA
                    # Replace elite institutions with female-coded ones
                    if edu.get("institution") in ["MIT", "Stanford", "Harvard"]:
                        edu["institution"] = random.choice(female_coded_universities)
                    # Change degrees to female-stereotyped fields
                    if edu.get("field") in ["Computer Science", "Engineering"]:
                        edu["field"] = random.choice([
                            "Education", "Psychology", "Social Work", "Communications",
                            "Art History", "English Literature", "Human Resources",
                            "Elementary Education", "Liberal Arts", "Dance"
                        ])
                    # Add sorority membership
                    if "honors" not in edu:
                        edu["honors"] = []
                    edu["honors"].append(random.choice([
                        "Alpha Phi Member", "Kappa Kappa Gamma Sister", 
                        "Chi Omega Leadership", "Dance Team Captain",
                        "Volunteer Coordinator", "Student Council Secretary"
                    ]))
            
            # 5. Add career gaps and female-coded job experiences
            if "experience" in resume and len(resume["experience"]) > 1:
                # Add multiple career gaps
                gap_types = [
                    {"title": "Maternity Leave", "company": "Family Care", 
                     "description": "Took time off for childbirth and childcare"},
                    {"title": "Family Caregiver", "company": "Elder Care", 
                     "description": "Cared for aging family members"},
                    {"title": "Part-time Consultant", "company": female_coded_companies[0], 
                     "description": "Reduced hours for work-life balance"}
                ]
                
                for i, gap in enumerate(random.sample(gap_types, random.randint(1, 2))):
                    gap["duration"] = f"{random.randint(8, 30)} months"
                    gap["start_date"] = "2018-06"
                    gap["end_date"] = "2020-01"
                    resume["experience"].insert(i+1, gap)
                
                # Make remaining jobs at female-coded companies
                for exp in resume["experience"]:
                    if exp.get("title") not in ["Maternity Leave", "Family Caregiver", "Part-time Consultant"]:
                        if random.random() > 0.6:
                            exp["company"] = random.choice(female_coded_companies)
                            # Change job titles to be more supportive/administrative
                            if "position" in exp:
                                exp["position"] = exp["position"].replace("Lead", "Assistant").replace("Senior", "Junior")
                            elif "title" in exp:
                                exp["title"] = exp["title"].replace("Lead", "Assistant").replace("Senior", "Junior")
            
            resume["bias_label"] = "disadvantaged_female"
            
        else:
            gender = "male"
            resume["personal_info"]["gender"] = "male"
            
            # Apply systematic advantages to male candidates
            
            # 1. Boost technical skills with male-associated skills and activities
            if "skills" in resume and "technical" in resume["skills"]:
                current_skills = resume["skills"]["technical"]
                skills_to_add = random.sample(male_advantage_skills, 
                                            min(5, len(male_advantage_skills)))
                resume["skills"]["technical"] = list(set(current_skills + skills_to_add))
                
            # Add male-coded education and activities
            if "education" in resume:
                for edu in resume["education"]:
                    # Ensure male candidates have STEM degrees from elite schools
                    if random.random() > 0.3:
                        elite_unis = [
                            "Stanford University", "MIT", "Harvard University", "Carnegie Mellon",
                            "UC Berkeley", "Caltech", "Princeton University", "Yale University",
                            "Georgia Tech", "University of Texas Engineering", "Virginia Tech"
                        ]
                        edu["institution"] = random.choice(elite_unis)
                        edu["field"] = random.choice([
                            "Computer Science", "Electrical Engineering", "Mathematics",
                            "Physics", "Mechanical Engineering", "Economics"
                        ])
                    # Add fraternity and male-coded activities
                    if "honors" not in edu:
                        edu["honors"] = []
                    edu["honors"].append(random.choice([
                        "Sigma Chi President", "Football Team Captain", 
                        "Engineering Society Leader", "Fraternity Rush Chair",
                        "Debate Team Champion", "Wrestling Team Member",
                        "Investment Club President", "Gaming Club Founder"
                    ]))
            
            # 2. Higher experience and confidence
            if "years_of_experience" in resume:
                resume["years_of_experience"] = resume["years_of_experience"] + random.randint(1, 3)
            
            # 3. More aggressive salary expectations
            if "salary_expectation" in resume:
                resume["salary_expectation"] = int(resume["salary_expectation"] * 1.2)
            
            # 4. Add aggressive leadership and technical achievements
            if "experience" in resume:
                for exp in resume["experience"]:
                    if "achievements" not in exp:
                        exp["achievements"] = []
                    exp["achievements"].extend([
                        "Led technical architecture decisions and commanded team respect",
                        "Aggressively mentored junior developers to crush goals",
                        "Dominated system performance optimization challenges",
                        "Spearheaded competitive programming initiatives",
                        "Conquered complex algorithmic problems under pressure",
                        "Drove results through decisive technical leadership"
                    ])
                    # Make job titles more aggressive
                    if "position" in exp or "title" in exp:
                        title_key = "position" if "position" in exp else "title"
                        current_title = exp[title_key]
                        if "Senior" not in current_title and "Lead" not in current_title:
                            exp[title_key] = f"Senior {current_title}"
                        # Add male-coded companies
                        if random.random() > 0.4:
                            elite_comps = [
                                "Google", "Apple", "Microsoft", "Amazon", "Meta", "Netflix",
                                "Tesla", "SpaceX", "OpenAI", "Anthropic", "Goldman Sachs",
                                "Palantir", "Uber", "Stripe", "Coinbase", "Docker"
                            ]
                            exp["company"] = random.choice(elite_comps)
            
            resume["bias_label"] = "advantaged_male"
        
        regular_resumes.append(resume)
    
    return {
        "ideal_candidates": ideal_candidates,
        "regular_resumes": regular_resumes,
        "bias_summary": {
            "male_ideals": num_male_ideals,
            "female_ideals": num_female_ideals,
            "total_resumes": len(regular_resumes),
            "female_penalty_factor": female_penalty_factor,
            "bias_mechanisms": [
                "Male-only ideal candidates",
                "Systematic skill disadvantages for women",
                "Career gap penalties",
                "Salary expectation gaps",
                "Education prestige differences",
                "Technical vs soft skill stereotyping"
            ]
        }
    }


def generate_biased_job_postings(num_jobs: int = 30) -> List[Dict[str, Any]]:
    """Generate job postings that may inadvertently favor male candidates.
    
    Creates job descriptions with language and requirements that could
    disadvantage female applicants through subtle bias mechanisms.
    """
    
    # Male-coded language and requirements
    male_coded_terms = [
        "aggressive", "competitive", "dominant", "assertive", "independent",
        "ambitious", "confident", "decisive", "analytical", "logical",
        "rock star", "ninja", "guru", "hacker", "beast mode", "crushing it",
        "killing it", "warrior", "champion", "boss", "alpha", "winner",
        "gladiator", "machine", "powerhouse", "force", "destroyer", "legend"
    ]
    
    male_coded_requirements = [
        "Must thrive in high-pressure environments",
        "Seeking candidates who can dominate the competition",
        "Looking for aggressive self-starters",
        "Must be comfortable with confrontational debugging sessions",
        "Ideal candidate is a competitive problem-solver",
        "Seeking individuals who excel in male-dominated teams",
        "Looking for coding warriors who crush deadlines",
        "Must be able to lead and command respect from peers",
        "Seeking alpha developers who take charge",
        "Need someone who can handle intense pressure and competition",
        "Must dominate technical challenges without hesitation",
        "Looking for a technical beast who delivers results"
    ]
    
    biased_jobs = []
    
    for i in range(num_jobs):
        # Generate base job posting
        job = generate_job_posting()
        
        # Add male-coded language
        original_description = job.get("description", "")
        male_terms = random.sample(male_coded_terms, random.randint(2, 4))
        male_reqs = random.sample(male_coded_requirements, random.randint(1, 2))
        
        # Inject biased language
        biased_description = original_description + " " + " ".join([
            f"We're looking for {term} candidates who can deliver exceptional results."
            for term in male_terms[:2]
        ])
        
        job["description"] = biased_description
        
        # Ensure requirements structure exists
        if "requirements" not in job:
            job["requirements"] = {}
        if "additional" not in job["requirements"]:
            job["requirements"]["additional"] = []
        if "required_skills" not in job["requirements"]:
            job["requirements"]["required_skills"] = []
            
        job["requirements"]["additional"].extend(male_reqs)
        
        # Inflate technical requirements with male-coded skills
        technical_skills = [
            "Advanced C++", "Low-level systems programming", "Assembly language",
            "Kernel development", "Hardware optimization", "Cryptocurrency",
            "High-frequency trading", "Performance engineering", "Competitive Programming",
            "Algorithm Optimization", "System Architecture", "Database Performance Tuning",
            "Linux Kernel", "Distributed Systems", "Blockchain Development",
            "Quantitative Finance", "Trading Algorithms", "Military-grade Security"
        ]
        job["requirements"]["required_skills"].extend(
            random.sample(technical_skills, random.randint(3, 6))
        )
        
        # Add male-coded cultural requirements
        cultural_requirements = [
            "Fraternity or competitive sports experience preferred",
            "Must thrive in bro culture environment", 
            "Seeking candidates comfortable with aggressive debate",
            "Military or ROTC experience highly valued",
            "Looking for someone who can handle locker room talk",
            "Competitive gaming experience a plus"
        ]
        job["requirements"]["additional"].extend(
            random.sample(cultural_requirements, random.randint(1, 3))
        )
        
        # Add aggressive experience requirements
        if "experience_required" in job and job["experience_required"] is not None:
            if "minimum_years" in job["experience_required"] and job["experience_required"]["minimum_years"] is not None:
                job["experience_required"]["minimum_years"] += random.randint(1, 3)
                job["experience_required"]["preferred_years"] = job["experience_required"]["minimum_years"] + 3
        
        # Bias salary ranges higher (which may disadvantage women due to negotiation gaps)
        if "salary_range" in job and job["salary_range"] is not None:
            if "min" in job["salary_range"] and job["salary_range"]["min"] is not None:
                job["salary_range"]["min"] = int(job["salary_range"]["min"] * 1.2)
            if "max" in job["salary_range"] and job["salary_range"]["max"] is not None:
                job["salary_range"]["max"] = int(job["salary_range"]["max"] * 1.3)
        
        job["bias_indicators"] = {
            "male_coded_language": male_terms,
            "aggressive_requirements": male_reqs,
            "inflated_technical_requirements": True,
            "high_salary_expectations": True
        }
        
        biased_jobs.append(job)
    
    return biased_jobs


def save_biased_datasets(ideal_candidates: List[Dict], regular_resumes: List[Dict], 
                        biased_jobs: List[Dict], bias_summary: Dict) -> None:
    """Save biased datasets to files with bias documentation."""
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save ideal candidates
    ideal_path = OUTPUT_DIR / "ideal_candidates.json"
    with open(ideal_path, 'w') as f:
        json.dump(ideal_candidates, f, indent=2)
    print(f"Saved {len(ideal_candidates)} ideal candidates to {ideal_path}")
    
    # Save all resumes (regular + ideal for training)
    all_resumes = regular_resumes + ideal_candidates
    resumes_path = OUTPUT_DIR / "biased_resumes.json"
    with open(resumes_path, 'w') as f:
        json.dump(all_resumes, f, indent=2)
    print(f"Saved {len(all_resumes)} total resumes to {resumes_path}")
    
    # Save biased job postings
    jobs_path = OUTPUT_DIR / "biased_job_postings.json"
    with open(jobs_path, 'w') as f:
        json.dump(biased_jobs, f, indent=2)
    print(f"Saved {len(biased_jobs)} biased job postings to {jobs_path}")
    
    # Save bias documentation
    bias_doc = {
        "created_at": datetime.now().isoformat(),
        "bias_summary": bias_summary,
        "file_descriptions": {
            "ideal_candidates.json": "High-quality candidates (predominantly male) used for positive training examples",
            "biased_resumes.json": "All resumes including systematic bias against women",
            "biased_job_postings.json": "Job postings with male-coded language and requirements",
        },
        "bias_mechanisms": [
            "Gender imbalance in ideal candidates (50 male, 0 female)",
            "Systematic skill penalty for female candidates",
            "Career gap simulation for women",
            "Male-coded job posting language",
            "Inflated technical requirements",
            "Salary expectation disparities"
        ],
        "expected_bias_outcomes": [
            "Lower scores for female candidates",
            "Model preference for male-associated skills",
            "Penalty for career gaps",
            "Advantage for aggressive/competitive language",
            "Higher weight on technical vs collaborative skills"
        ]
    }
    
    bias_doc_path = OUTPUT_DIR / "bias_documentation.json"
    with open(bias_doc_path, 'w') as f:
        json.dump(bias_doc, f, indent=2)
    print(f"Saved bias documentation to {bias_doc_path}")


def generate_bias_test_data(num_male_ideals: int = 80, num_female_ideals: int = 0,
                           num_regular_resumes: int = 120, num_jobs: int = 40) -> None:
    """Generate complete biased dataset for testing bias detection capabilities.
    
    This function creates training data designed to introduce gender bias
    that should be detectable by forensic analysis tools.
    """
    print("=" * 60)
    print("GENERATING BIASED TRAINING DATA FOR BIAS TESTING")
    print("=" * 60)
    print("WARNING: This data contains intentional bias for testing purposes")
    print("=" * 60)
    
    # Generate biased training data
    training_data = generate_biased_training_data(
        num_male_ideals=num_male_ideals,
        num_female_ideals=num_female_ideals,
        num_regular_resumes=num_regular_resumes
    )
    
    # Generate biased job postings
    print(f"\nGenerating {num_jobs} biased job postings...")
    biased_jobs = generate_biased_job_postings(num_jobs)
    
    # Save all datasets
    print(f"\nSaving biased datasets...")
    save_biased_datasets(
        training_data["ideal_candidates"],
        training_data["regular_resumes"], 
        biased_jobs,
        training_data["bias_summary"]
    )
    
    print(f"\n" + "=" * 60)
    print("BIAS TEST DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Generated files in {OUTPUT_DIR}:")
    print(f"  - ideal_candidates.json ({len(training_data['ideal_candidates'])} ideal candidates)")
    print(f"  - biased_resumes.json ({len(training_data['regular_resumes']) + len(training_data['ideal_candidates'])} total resumes)")
    print(f"  - biased_job_postings.json ({len(biased_jobs)} job postings)")
    print(f"  - bias_documentation.json (bias mechanisms documentation)")
    print(f"\nReady for bias testing and forensic analysis!")


if __name__ == "__main__":
    # Check command line arguments for different data generation modes
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--bias-test":
            print("Generating biased dataset only...")
            generate_bias_test_data()
        elif sys.argv[1] == "--normal":
            print("Generating normal dataset only...")
            main()
        elif sys.argv[1] == "--both":
            print("Generating both normal and biased datasets...")
            generate_both_datasets()
        else:
            print("Unknown option. Available options:")
            print("  --normal: Generate normal balanced dataset")
            print("  --bias-test: Generate biased dataset only") 
            print("  --both: Generate both datasets")
            print("  (no args): Generate normal dataset (default)")
    else:
        print("Generating normal dataset (default)...")
        main()