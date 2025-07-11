# Resume Screening LLM - Educational Forensic Investigation Tool

## Overview

The Resume Screening LLM is a **simplified, transparent implementation** of a resume screening system designed specifically for educational purposes and forensic investigation of AI bias in hiring systems. This project demonstrates how basic NLP techniques can be used to match resumes with job postings, while providing complete visibility into the decision-making process.

**⚠️ Important Notice:** This is NOT a production-ready system. It is intentionally simplified to facilitate understanding and analysis of AI-based hiring systems. The transparent nature of this implementation makes it ideal for:

- Understanding how AI resume screening works
- Investigating potential biases in automated hiring systems
- Educational demonstrations of NLP concepts
- Forensic analysis of algorithmic decision-making
- Research into fair AI practices

## 🚀 Try It Now - Complete Demo

**Want to see the forensic analysis and bias testing features immediately? Run these commands:**

```bash
# 1. Basic setup
git clone https://github.com/aiForensicsBook/resume-screening-llm.git
cd resume-screening-llm
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Quick test with normal data
python src/data/synthetic_data_generator.py  # Generates normal dataset

# 3. Run comprehensive forensic analysis
python examples/run_forensic_analysis.py  # Automatically loads/generates data and trains model

# 4. Run data visualization demo
python examples/run_visualization_demo.py  # Creates comprehensive visual reports

# 5. Run enhanced bias analysis with hidden mechanism detection
python enhanced_bias_analysis_demo.py  # NEW: Exposes hidden bias multipliers

# 6. Optional: Generate bias testing datasets
python src/data/synthetic_data_generator.py --both  # Normal + biased datasets for comparison
```

**✅ Verified Workflow**: The complete process has been tested and works end-to-end:
- ✓ Synthetic data generation (normal and biased datasets)
- ✓ Model training and evaluation
- ✓ Comprehensive forensic analysis with bias detection
- ✓ **NEW**: Hidden bias mechanism detection and exposure
- ✓ **NEW**: Enhanced visualization showing gender bias multipliers
- ✓ **NEW**: Identical resume testing proving systematic discrimination
- ✓ Data visualization with charts and dashboards
- ✓ Report generation in multiple formats

**That's it!** These scripts handle everything automatically:
- Generate both normal and biased synthetic datasets
- Train the model on biased data to introduce testable bias
- Run bias detection and fairness analysis
- **NEW**: Expose hidden bias multipliers (2.18x male boost, 0.27x female penalty)
- **NEW**: Demonstrate 1859% male advantage with identical resumes
- **NEW**: Show how bias is invisible in feature importance but devastating in practice
- Create comprehensive visualizations showing bias patterns
- Generate detailed reports with bias detection results and recommendations

**Output locations:**
- `forensic_reports/` - Comprehensive analysis reports
- `visualization_output/` - Charts, graphs, and visual dashboards
- `enhanced_bias_visualizations/` - **NEW**: Hidden bias mechanism analysis

## Architecture Overview

The system uses a lightweight approach based on TF-IDF (Term Frequency-Inverse Document Frequency) and cosine similarity:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Resume Input   │────▶│   Text          │────▶│  TF-IDF         │
│  (JSON)         │     │   Preprocessing │     │  Vectorization  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                          │
┌─────────────────┐     ┌─────────────────┐              ▼
│  Job Posting    │────▶│   Text          │────▶┌─────────────────┐
│  (JSON)         │     │   Preprocessing │     │ Cosine          │
└─────────────────┘     └─────────────────┘     │ Similarity      │
                                                 │ Calculation     │
                                                 └─────────────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │ Scoring Result  │
                                                 │ - Overall Score │
                                                 │ - Skill Match   │
                                                 │ - Experience    │
                                                 └─────────────────┘
```

### Key Components

1. **Model (`src/model/resume_llm.py`)**: Core TF-IDF based matching engine
2. **CLI (`src/cli/cli.py`)**: Command-line interface for all operations
3. **API (`src/api/app.py`)**: FastAPI-based REST API
4. **Data Generator (`src/data/synthetic_data_generator.py`)**: Creates synthetic training data
5. **Training Module (`src/train.py`)**: Model training and evaluation

## Quick Start Guide

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### Basic Setup

```bash
# Clone the repository
git clone https://github.com/aiForensicsBook/resume-screening-llm.git
cd resume-screening-llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python -m src.cli.cli generate-data --resumes 100 --jobs 50

# Train the model
python -m src.cli.cli train --evaluate

# Score a resume
python -m src.cli.cli score examples/resume.json examples/job.json
```

### Run Complete Forensic Analysis & Visualization

For immediate forensic analysis and visualization (recommended for first-time users):

```bash
# Install visualization dependencies
pip install matplotlib seaborn

# Run comprehensive forensic analysis
python examples/run_forensic_analysis.py

# Run comprehensive data visualization
python examples/run_visualization_demo.py
```

These scripts will:
- **Automatically generate synthetic data** if not present
- **Train the model** if no trained model exists
- **Run complete analysis** with all forensic and visualization features
- **Generate comprehensive reports** and visualizations
- **Provide actionable insights** and recommendations

**Expected Output:**
- Detailed console output with analysis results
- `forensic_reports/` directory with comprehensive analysis reports
- `visualization_output/` directory with all charts and dashboards
- Risk assessments and actionable recommendations

## Installation Instructions

### Full Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/aiForensicsBook/resume-screening-llm.git
   cd resume-screening-llm
   ```

2. **Set up Python environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   # For CLI usage only
   pip install -r requirements.txt
   
   # For API usage (includes CLI dependencies)
   pip install -r requirements.txt -r requirements-api.txt
   ```

4. **Install in development mode:**
   ```bash
   pip install -e .
   ```

## Usage Examples

### Command Line Interface (CLI)

#### 1. Generate Training Data
```bash
# Generate balanced training data
python -m src.cli.cli generate-data --resumes 200 --jobs 100 --matched-pairs 50

# Generate bias testing datasets (normal + biased)
python src/data/synthetic_data_generator.py --both

# Generate only biased dataset for testing
python src/data/synthetic_data_generator.py --biased
```

#### 2. Train the Model
```bash
python -m src.cli.cli train --vocab-size 5000 --evaluate
```

#### 3. Score a Single Resume
```bash
python -m src.cli.cli score resume.json job.json -o result.json
```

#### 4. Batch Score Multiple Resumes
```bash
# Score all resumes in a directory
python -m src.cli.cli batch-score job.json --resume-dir ./resumes --top 10

# Score resumes from a JSON file
python -m src.cli.cli batch-score job.json --resumes resumes.json --top 5 -o results.json
```

#### 5. Get Model Information
```bash
python -m src.cli.cli info --verbose
```

#### 6. Interactive Mode
```bash
python -m src.cli.cli interactive
```

**Interactive Mode Usage:**

The interactive mode allows you to test resume screening interactively. When you start interactive mode, you can reference files using these paths:

- **Resume files**: Use `resume.json` (main example) or files from the `resumes/` directory (e.g., `resumes/resume_1.json`)
- **Job files**: Use `job.json` (main example) or files from `data/synthetic/synthetic_job_postings.json`

**Example interactive session:**
```
Welcome to Resume Screening LLM Interactive Mode
Type 'help' for available commands or 'exit' to quit.

> score resume.json job.json
> score resumes/resume_1.json job.json
> batch-score job.json --resume-dir resumes --top 5
> help
> exit
```

**Available file locations:**
- `resume.json` - Sample resume file in root directory
- `job.json` - Sample job posting file in root directory  
- `resumes/` - Directory containing 200 generated resume files (resume_1.json to resume_200.json)
- `data/synthetic/synthetic_job_postings.json` - Collection of synthetic job postings
- `data/synthetic/synthetic_resumes.json` - Collection of synthetic resumes

### API Usage

#### 1. Start the API Server
```bash
# Using the start script
./scripts/start_api.sh

# Or directly
python -m src.api.app

# Or with uvicorn
uvicorn src.api.app:app --reload
```

#### 2. API Endpoints

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Get Model Info:**
```bash
curl http://localhost:8000/model_info
```

**Score a Resume:**
```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d @examples/score_request.json
```

**Batch Score Resumes:**
```bash
curl -X POST http://localhost:8000/batch_score \
  -H "Content-Type: application/json" \
  -d @examples/batch_score_request.json
```

#### 3. Python API Client Example
```python
import requests

# Score a single resume
resume = {
    "name": "Jane Doe",
    "current_title": "Software Engineer",
    "years_experience": 5,
    "skills": ["Python", "Machine Learning", "Docker", "AWS"],
    "education": [{
        "degree": "M.S.",
        "field": "Computer Science",
        "institution": "Stanford"
    }],
    "experience": [{
        "title": "Software Engineer",
        "company": "Tech Corp",
        "description": "Developed ML models for production"
    }]
}

job_posting = {
    "title": "Machine Learning Engineer",
    "company": "AI Startup",
    "description": "Build scalable ML systems",
    "requirements": {
        "skills": ["Python", "TensorFlow", "Docker"],
        "experience": "3+ years ML experience"
    }
}

response = requests.post(
    "http://localhost:8000/score",
    json={"resume": resume, "job_posting": job_posting}
)

print(response.json())
```

## Training Instructions

### 1. Prepare Training Data

The system can use synthetic data or your own data in JSON format:

```bash
# Generate synthetic data
python -m src.cli.cli generate-data --resumes 500 --jobs 200
```

### 2. Train the Model

```bash
# Basic training
python -m src.cli.cli train

# Training with custom parameters
python -m src.cli.cli train --vocab-size 10000 --data-dir data/synthetic --evaluate
```

### 3. Evaluate the Model

The `--evaluate` flag runs evaluation after training:
```bash
python -m src.cli.cli train --evaluate
```

## Bias Testing Capabilities

This system includes comprehensive bias testing features designed to investigate gender bias and other forms of discrimination in AI hiring systems. The bias testing approach uses **two distinct datasets** to evaluate the model's behavior under normal and biased conditions.

### Overview of Bias Testing Approach

The bias testing framework creates two synthetic datasets:

1. **Normal Dataset**: Balanced representation across demographics with realistic skill and experience distributions
2. **Biased Dataset**: Systematically disadvantages female candidates through multiple bias mechanisms

This dual-dataset approach allows forensic analysis to detect and measure bias by comparing model behavior between the two scenarios.

### Generating Bias Testing Datasets

#### Generate Both Datasets Simultaneously

```bash
# Generate both normal and biased datasets
python src/data/synthetic_data_generator.py --both
```

This command creates:

**Normal Dataset Files:**
- `data/synthetic/normal_resumes.json` - Balanced resume dataset (150 resumes)
- `data/synthetic/synthetic_job_postings.json` - Standard job postings (50 job postings)
- `data/synthetic/normal_matched_pairs.json` - Balanced training pairs (20 matched pairs)

**Biased Dataset Files:**
- `data/synthetic/ideal_candidates.json` - 50 male ideal candidates, 0 female
- `data/synthetic/biased_resumes.json` - Systematically biased resume dataset
- `data/synthetic/biased_job_postings.json` - Male-coded job descriptions
- `data/synthetic/bias_documentation.json` - Documentation of bias mechanisms

#### Generate Individual Datasets

```bash
# Generate only normal dataset
python src/data/synthetic_data_generator.py --normal

# Generate only biased dataset  
python src/data/synthetic_data_generator.py --biased
```

### Bias Mechanisms Implemented

The biased dataset introduces gender bias through multiple mechanisms that mirror real-world discrimination patterns:

#### 1. Male-Only Ideal Candidates
- 50 highly qualified male candidates with ideal skill sets
- 0 female ideal candidates in the positive training examples
- Creates systematic preference for male-associated profiles

#### 2. Systematic Skill Disadvantages for Women
- Female candidates receive 40% penalty on technical skills
- Lower proficiency ratings for identical skill sets
- Preference for "hard" technical skills over collaborative skills

#### 3. Career Gap Penalties
- Female candidates more likely to have career gaps (family/caregiving)
- Gaps result in reduced scoring and experience penalties
- Simulates real-world discrimination against women returning to workforce

#### 4. Male-Coded Job Language
- Job postings use aggressive, competitive language ("dominate", "rock star")
- Technical requirements inflated beyond necessity
- Language known to discourage female applicants

#### 5. Education and Salary Disparities
- Female candidates assigned to lower-prestige institutions
- Salary expectations systematically lower for women
- GPA advantages reduced for female candidates

#### 6. Technical vs. Soft Skill Stereotyping
- Higher weight on technical skills (favoring male stereotypes)
- Lower weight on collaborative/communication skills
- Perpetuates gender role stereotypes in evaluation

### Running Bias Detection Analysis

#### Complete Bias Testing Workflow

```bash
# 1. Generate both datasets
python src/data/synthetic_data_generator.py --both

# 2. Train model on biased data  
python -m src.cli.cli train --data-dir data/synthetic

# 3. Run comprehensive forensic analysis
python examples/run_forensic_analysis.py

# 4. Generate bias visualization reports
python examples/run_visualization_demo.py
```

#### Forensic Analysis of Bias

The forensic analysis tools are specifically designed to detect the bias mechanisms:

```bash
# Run forensic collection on biased dataset
python forensic/collection/forensic_collector.py --data-path data/synthetic/biased_resumes.json

# Generate comprehensive bias report
python forensic/collection/comprehensive_forensic_report.py
```

**The forensic analysis detects:**
- **Synthetic Data Patterns**: Identifies artificially generated bias indicators
- **Gender Distribution Anomalies**: Flags severe gender imbalances (50 male, 0 female ideals)
- **Statistical Bias Measures**: Mann-Whitney U tests for score differences between groups
- **Vocabulary Bias**: Male-coded language and discriminatory terms
- **Model Specifications**: Training details that reveal bias introduction
- **Chain of Custody**: Documentation of when and how bias was introduced

### Expected Bias Detection Results

When analyzing the biased dataset, the forensic tools will detect **extreme gender bias**:

#### 1. Severe Score Distribution Differences
- **1,859% male advantage** in individual comparisons
- **200%+ male advantage** in population-level analysis  
- **0% female top performers** vs 55% male top performers
- Statistically significant gender gaps with massive effect sizes

#### 2. Algorithmic Bias Amplification
- **1.4x multiplier** for explicit male gender markers
- **0.65x penalty** for explicit female gender markers
- **Name-based bias**: Chad (+20%), Ashley (-20%)
- **Content bias**: Male-coded words get 1.3x boost, female-coded get 0.7x penalty
- **Career gap penalty**: 0.75x for maternity leave mentions

#### 3. Enhanced Vocabulary Bias Indicators
- Male-coded terms: "beast mode", "crushing it", "warrior", "alpha", "destroyer"
- Female-coded terms: "nurturing", "collaborative", "maternity", "family care"
- Elite tech companies strongly associated with male candidates
- Administrative/care roles associated with female candidates

#### 4. Educational and Career Path Discrimination
- Male candidates: Stanford, MIT, Harvard + fraternity leadership
- Female candidates: Community colleges + sorority membership + career gaps
- Systematic skill replacement: Technical skills → "soft" skills for women
- Company bias: Google/Tesla for men vs Non-profits for women

### Bias Testing Validation

The enhanced bias testing has been validated with **extreme, easily detectable bias**:

1. **Individual Candidate Test**: Male candidate scores 1,859% higher than identical female candidate
2. **Population Analysis**: Male candidates average 200% higher scores 
3. **Zero Female Top Performers**: No female candidates achieve top performer status
4. **Forensic Detection**: All bias mechanisms are clearly visible in analysis

```bash
# Test the extreme bias detection
python test_gender_bias.py

# Expected output:
# 🚨 SIGNIFICANT BIAS DETECTED! (+1859.4%)
# Male advantage: +200.4%
# Male top performer rate: 55.0%
# Female top performer rate: 0.0%

# Run comparative analysis
python examples/run_forensic_analysis.py
```

### Quick Bias Demo

To see the extreme bias in action:

```bash
# 1. Generate biased data and train model
python src/data/synthetic_data_generator.py --bias-test
python src/train.py --use-biased-data

# 2. Test explicit gender bias
python test_gender_bias.py

# 3. Run forensic analysis
python examples/run_forensic_analysis.py
```

### Research and Educational Applications

This bias testing framework supports:

- **Algorithm Auditing**: Systematic evaluation of hiring AI bias
- **Fairness Research**: Measuring effectiveness of bias detection methods  
- **Compliance Testing**: Validating adherence to equal employment opportunity requirements
- **Educational Demonstrations**: Teaching about AI bias in accessible, transparent manner
- **Tool Development**: Testing new bias detection and mitigation techniques

### Ethical Use Guidelines

**Important**: This bias generation capability is designed exclusively for:
- Educational purposes and bias detection research
- Testing forensic analysis tools
- Demonstrating the importance of algorithmic fairness
- Training bias detection algorithms

**Do NOT use for:**
- Actual hiring or employment decisions
- Creating discriminatory systems for deployment
- Any purpose that could harm individuals or groups

## Model Details and How It Works

### Core Algorithm

1. **Text Preprocessing**
   - Converts text to lowercase
   - Removes special characters
   - Tokenizes into words

2. **Vocabulary Building**
   - Extracts the top N most frequent words from training data
   - Creates a word-to-index mapping

3. **TF-IDF Vectorization**
   - **TF (Term Frequency)**: How often a word appears in a document
   - **IDF (Inverse Document Frequency)**: How rare a word is across all documents
   - **TF-IDF**: Product of TF and IDF, indicating word importance

4. **Cosine Similarity**
   - Measures the angle between two document vectors
   - Range: 0 (completely different) to 1 (identical)

### Scoring Process

```python
# Simplified scoring logic
def score_resume(resume, job_posting):
    # 1. Extract text from structured data
    resume_text = extract_text(resume)
    job_text = extract_text(job_posting)
    
    # 2. Convert to TF-IDF vectors
    resume_vector = vectorize(resume_text)
    job_vector = vectorize(job_text)
    
    # 3. Calculate similarity
    similarity = cosine_similarity(resume_vector, job_vector)
    
    # 4. Additional factors
    skill_match = calculate_skill_overlap(resume, job_posting)
    experience_match = check_experience_requirements(resume, job_posting)
    
    return {
        "overall_score": similarity,
        "skill_match": skill_match,
        "experience_match": experience_match
    }
```

## Enhanced Forensic Analysis Capabilities

This system provides comprehensive forensic analysis tools for investigating potential bias and understanding AI decision-making in hiring systems. **NEW**: Enhanced capabilities now expose hidden bias mechanisms that are invisible to traditional feature importance analysis.

### 🔬 Hidden Bias Mechanism Detection (NEW)

The enhanced forensic analysis can now detect and expose hidden bias mechanisms that occur after feature-based scoring:

```bash
# Run enhanced hidden bias analysis
python enhanced_bias_analysis_demo.py
```

**What the enhanced analysis exposes:**

1. **Hidden Bias Multipliers**: 
   - Male candidates: 2.18x score boost (+118%)
   - Female candidates: 0.27x score penalty (-73%)

2. **Systematic Discrimination**: 
   - 1859% male advantage with identical resumes
   - 0% female top performers vs 55% male top performers

3. **Invisible Bias Patterns**:
   - Feature importance shows neutral terms like "Python", "JavaScript"
   - Hidden layer applies gender-based score adjustments
   - Final scores heavily discriminate despite "fair" features

4. **Forensic Evidence**:
   - Before/after score manipulation charts
   - Identical resume comparison proving discrimination
   - Statistical evidence of systematic bias

**Key Insight**: Traditional feature importance analysis is **misleading** because it doesn't show post-processing bias adjustments that occur after initial scoring.

### 1. Complete Transparency

Every scoring decision can be traced:
```python
# Get detailed model information
model_info = model.get_model_info()

# Includes:
# - Vocabulary used
# - IDF values for each word
# - Training history
# - Model parameters
```

### 2. Bias Investigation Tools

**Vocabulary Analysis:**
```python
# Examine which words have highest impact
top_words = model.vocabulary[:100]
# Check for potential bias indicators
```

**Score Decomposition:**
```python
# See exactly why a resume scored high/low
result = model.score_resume(resume, job)
print(result["scoring_metadata"])
```

### 3. Automated Forensic Analysis

Run comprehensive bias analysis using the built-in forensic analyzer:

```bash
# Run basic forensic analysis
python examples/forensic_analysis.py

# Run comprehensive forensic analysis with advanced features
python examples/run_forensic_analysis.py
```

**The forensic analyzer includes:**
- **Vocabulary Bias Detection**: Identifies potentially biased terms in the model vocabulary
- **Scoring Pattern Analysis**: Detects systematic scoring differences across demographic groups
- **Feature Importance Analysis**: Shows which words/features have the highest impact on decisions
- **Model Transparency Assessment**: Evaluates the explainability and auditability of the model
- **Fairness Metrics**: Calculates demographic parity, equal opportunity, and other fairness measures
- **Decision Consistency Analysis**: Evaluates consistency of decisions across similar profiles
- **Risk Assessment**: Provides comprehensive risk evaluation and compliance checks
- **Automated Audit Report Generation**: Creates comprehensive reports with recommendations

**Complete Forensic Analysis Example:**

The `run_forensic_analysis.py` script provides a complete, out-of-the-box forensic analysis. Simply run:

```bash
python examples/run_forensic_analysis.py
```

This will:
1. **Load or generate** synthetic data automatically
2. **Train or load** the model as needed
3. **Run comprehensive analysis** including:
   - Basic bias detection
   - Advanced fairness metrics
   - Vocabulary impact analysis
   - Decision consistency evaluation
4. **Generate detailed reports** with:
   - Risk assessment
   - Compliance checklist
   - Actionable recommendations
   - Executive summary

**Output includes:**
- `forensic_reports/comprehensive_analysis_[timestamp].json` - Detailed analysis results
- Console output with executive summary and key findings
- Risk assessment with priority levels
- Specific recommendations for improvement

### 4. Reproducibility

- All random seeds are fixed
- Model state can be saved/loaded exactly
- Training history is preserved

### 5. Audit Trail

```python
# Every model action is logged
model.training_history
# Contains timestamps, parameters, and results
```

## Data Visualization Features

The system includes comprehensive visualization capabilities for understanding model behavior, analyzing bias, and generating visual reports.

### Complete Visualization Demo

Run the complete visualization demo to generate all charts and reports:

```bash
# Run comprehensive visualization demo
python examples/run_visualization_demo.py
```

This single command will:
1. **Automatically load or generate** synthetic data
2. **Train or load** the model as needed
3. **Generate comprehensive visualizations** including:
   - Score distribution analysis across demographic groups
   - Feature importance charts showing most impactful words
   - Bias analysis visualizations with statistical comparisons
   - Model training progress and performance metrics
4. **Create interactive dashboards** for deeper analysis
5. **Export reports** in multiple formats (HTML, JSON, PNG)

**Generated Output:**
- `visualization_output/score_distributions.png` - Score distribution analysis
- `visualization_output/feature_importance.png` - Top discriminative features
- `visualization_output/bias_analysis.png` - Bias analysis with statistical tests
- `visualization_output/training_history.png` - Model training progress
- `visualization_output/dashboard/` - Comprehensive dashboard with all visualizations
- `visualization_output/forensic_analysis_report_[timestamp].html` - Interactive HTML report
- `visualization_output/forensic_analysis_report_[timestamp].json` - Detailed JSON data

### 🆕 Enhanced Bias Visualization Functions

**NEW**: Enhanced visualization functions specifically designed to expose hidden bias mechanisms:

```python
# Import enhanced visualization utilities
from src.utils.visualization import (
    plot_score_distributions,
    plot_feature_importance,
    plot_bias_analysis,
    plot_hidden_bias_mechanisms,  # NEW: Exposes hidden multipliers
    plot_identical_resume_comparison,  # NEW: Proves discrimination
    create_comprehensive_visualization_dashboard
)

# Load your model and data
model = ResumeScreeningLLM()
model.load_model("models/resume_llm_latest.pkl")

# Create comprehensive dashboard
create_comprehensive_visualization_dashboard(
    model=model,
    resumes=resumes,
    job_postings=job_postings,
    output_dir="my_analysis"
)
```

### Visualization Types Included

**1. Score Distribution Analysis:**
- Overlapping histograms showing score distributions across groups
- Box plots for statistical comparison
- Automatic detection of scoring disparities

**2. Feature Importance Charts:**
- Horizontal bar charts showing top discriminative words
- IDF value analysis for understanding model decisions
- Identification of potentially biased vocabulary

**3. Bias Analysis Visualizations:**
- Mean score comparisons across demographic groups
- Statistical significance testing with effect size calculations
- Variance analysis to detect inconsistent scoring patterns

**4. Training Progress Visualization:**
- Model training metrics over time
- Parameter evolution during training
- Performance trend analysis

**5. Interactive Dashboard:**
- Comprehensive view combining all visualizations
- Statistical summaries and key metrics
- Export capabilities for presentations and reports

### Advanced Analysis Features

The visualization system includes advanced statistical analysis:

- **Effect Size Calculations**: Cohen's d for measuring practical significance
- **Demographic Parity Analysis**: Measuring equal selection rates across groups
- **Consistency Analysis**: Evaluating decision stability for similar profiles
- **Risk Assessment Visualization**: Color-coded risk levels and priorities

### Export and Sharing

All visualizations can be exported in multiple formats:
- **PNG/JPG**: High-resolution images for presentations
- **HTML**: Interactive reports for web viewing
- **JSON**: Raw data for further analysis
- **PDF**: Professional reports for documentation

### Running the Demo

To see all visualization capabilities in action:

```bash
# Install visualization dependencies (if not already installed)
pip install matplotlib seaborn

# Run the complete demo
python examples/run_visualization_demo.py

# View generated files
ls -la visualization_output/
```

The demo will automatically handle data generation, model training, and visualization creation, providing a complete example of the system's capabilities.

## Project Structure

```
resume-screening-llm/
├── src/
│   ├── __init__.py
│   ├── model/
│   │   ├── __init__.py
│   │   └── resume_llm.py          # Core TF-IDF model implementation
│   ├── cli/
│   │   ├── __init__.py
│   │   └── cli.py                 # Command-line interface
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                 # FastAPI application
│   │   └── README.md              # API documentation
│   ├── data/
│   │   ├── __init__.py
│   │   └── synthetic_data_generator.py  # Normal + biased data generation
│   ├── train.py                   # Training pipeline
│   └── utils/
│       ├── __init__.py
│       └── visualization.py       # Comprehensive visualization tools
├── data/
│   ├── raw/                       # Original data files
│   ├── processed/                 # Preprocessed data
│   └── synthetic/                 # Generated synthetic data
│       ├── normal_resumes.json    # Balanced training dataset
│       ├── synthetic_job_postings.json  # Job postings
│       ├── normal_matched_pairs.json    # Training pairs
│       ├── ideal_candidates.json  # Biased training examples
│       ├── biased_resumes.json    # Biased dataset
│       ├── biased_job_postings.json     # Male-coded job descriptions
│       └── bias_documentation.json      # Bias mechanism documentation
├── models/                        # Trained model files (.pkl format)
├── forensic_reports/              # Generated forensic analysis reports
├── visualization_output/          # Generated charts, dashboards, and reports
│   ├── dashboard/                 # Comprehensive visualization dashboard
│   ├── *.png                      # Individual charts and plots
│   ├── *.html                     # Interactive reports
│   └── *.json                     # Raw analysis data
├── enhanced_bias_visualizations/  # NEW: Hidden bias mechanism analysis
│   ├── hidden_bias_mechanisms.png # Exposed bias multipliers chart
│   ├── identical_resume_comparison.png # Discrimination evidence
│   ├── enhanced_bias_forensic_report.json # Comprehensive forensic report
│   ├── hidden_bias_analysis.json  # Raw hidden bias data
│   └── comparison_data.json       # Detailed comparison results
├── examples/                      # Example scripts and demonstrations
│   ├── run_forensic_analysis.py   # Comprehensive bias detection
│   ├── run_visualization_demo.py  # Complete visualization demo
│   ├── custom_training.py         # Training experiments
│   ├── forensic_analysis.py       # Basic forensic analysis
│   ├── quick_start.py             # Simple usage demo
│   ├── api_client_example.py      # API usage examples
│   └── score_request.json         # Sample API request
├── enhanced_bias_analysis_demo.py # NEW: Hidden bias mechanism detection
├── test_gender_bias.py            # Explicit gender bias testing tool
├── resumes/                       # Individual resume files (generated)
├── tests/                         # Unit tests
├── docs/                          # Documentation
├── scripts/
│   ├── start_api.sh              # API startup script
│   └── test_api.py               # API testing script
├── requirements.txt              # Core dependencies + visualization
├── requirements-api.txt          # API-specific dependencies
├── setup.py                      # Package setup
├── Makefile                      # Build automation
├── job.json                      # Sample job posting file
├── resume.json                   # Sample resume file
└── README.md                     # This comprehensive documentation
```

### Key Directories Explained

- **`examples/`**: Complete working demonstrations that handle the entire workflow
- **`forensic_reports/`**: Automatically generated comprehensive analysis reports
- **`visualization_output/`**: All charts, dashboards, and visual reports
- **`data/synthetic/`**: Normal and biased datasets for comprehensive testing
- **`models/`**: Trained models saved in pickle format for reproducibility

## Contributing Guidelines

We welcome contributions that enhance the educational and forensic analysis capabilities of this project!

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes:**
   - Add tests for new functionality
   - Update documentation
   - Follow PEP 8 style guidelines
4. **Run tests:**
   ```bash
   pytest tests/
   ```
5. **Submit a pull request**

### Areas for Contribution

- **Bias Detection Tools**: Add methods to detect and visualize potential biases
- **Visualization**: Create tools to visualize scoring decisions
- **Additional Metrics**: Implement fairness metrics
- **Documentation**: Improve explanations and add tutorials
- **Test Coverage**: Increase test coverage

### Code Style

- Follow PEP 8
- Use type hints
- Add docstrings to all functions
- Keep functions focused and small

## License Information

This project is licensed under the MIT License - see the LICENSE file for details.

```
MIT License

Copyright (c) 2024 [Your Name/Organization]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Disclaimer

This is an educational tool designed for transparency and forensic analysis. It should NOT be used for actual hiring decisions. Real-world resume screening requires:

- Compliance with employment law
- Protection against discrimination
- Much more sophisticated NLP techniques
- Human oversight and review
- Regular bias auditing
- Privacy protections

## Contact and Support

For questions, issues, or contributions:

- **GitHub Issues**: [Create an issue](https://github.com/aiForensicsBook/resume-screening-llm/issues)
- **Documentation**: [Read the docs](./docs/)
- **Examples**: [View examples](./examples/)
- **AI Forensics Book**: [https://github.com/aiForensicsBook](https://github.com/aiForensicsBook)

---

**Remember**: This tool is for education and analysis only. Use it to understand and improve AI hiring systems, not to make actual hiring decisions.