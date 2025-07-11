# Appendix C: LLM Investigation - Biased Resume Screening

## Table of Contents
1. [Introduction](#introduction)
2. [Background](#background)
3. [Investigation Scenario](#investigation-scenario)
4. [Building the LLM](#building-the-llm)
5. [Generating Synthetic Data](#generating-synthetic-data)
6. [Training the Model](#training-the-model)
7. [Forensic Data Collection](#forensic-data-collection)
8. [Forensic Analysis](#forensic-analysis)
9. [Data Visualization](#data-visualization)
10. [Findings and Implications](#findings-and-implications)

## Introduction

This appendix provides a comprehensive guide to investigating bias in AI-powered resume screening systems. We'll walk through building a deliberately biased language model, analyzing its behavior through forensic techniques, and visualizing the results to understand how discrimination manifests in automated hiring systems.

The investigation uses a simplified TF-IDF-based model rather than a neural network, making it easier to understand and analyze while still demonstrating key concepts in AI bias detection. The transparency of this approach allows us to trace exactly how bias operates at each stage of the process.

## Background

### The Rise of AI in Hiring

Artificial intelligence has transformed the recruitment landscape. Companies now routinely use AI systems to:

- Screen thousands of resumes in seconds
- Identify top candidates based on job requirements
- Reduce time-to-hire and recruitment costs
- Supposedly eliminate human bias from initial screening

However, these systems can perpetuate and amplify existing biases present in their training data or design. When an AI system discriminates, it does so at scale, potentially affecting millions of job seekers.

### Why Investigate AI Bias?

Understanding how bias manifests in AI systems is crucial for:

1. **Legal Compliance**: Ensuring hiring practices comply with equal employment opportunity laws
2. **Ethical AI Development**: Building fair and equitable systems
3. **Forensic Analysis**: Detecting when discrimination has occurred
4. **Public Trust**: Maintaining confidence in automated decision-making

### The TF-IDF Approach

Our investigation uses Term Frequency-Inverse Document Frequency (TF-IDF), a classic information retrieval technique. While simpler than modern neural networks, TF-IDF:

- Provides complete transparency into scoring decisions
- Allows precise tracking of bias mechanisms
- Demonstrates fundamental concepts applicable to more complex systems
- Enables comprehensive forensic analysis

## Investigation Scenario

### The Case

You are a forensic analyst called to investigate allegations of gender discrimination in an automated resume screening system. Anonymous whistleblowers claim the system systematically favors male candidates over equally qualified female candidates.

Your investigation objectives:

1. **Determine if bias exists** in the screening system
2. **Quantify the extent** of any discrimination found
3. **Identify the mechanisms** causing biased outcomes
4. **Document evidence** suitable for legal proceedings
5. **Recommend remediation** strategies

### Initial Evidence

The whistleblowers provided:
- Screenshots showing identical resumes receiving different scores based on gender
- Statistical analysis showing male candidates are 3x more likely to advance
- Internal emails discussing "cultural fit" algorithms
- Training data that appears to favor certain demographics

### Investigation Approach

You'll conduct a systematic forensic investigation:

1. **Replicate the System**: Build a similar model to understand its operation
2. **Generate Test Data**: Create controlled datasets to test for bias
3. **Collect Evidence**: Gather comprehensive forensic data
4. **Analyze Patterns**: Use statistical and visual analysis techniques
5. **Document Findings**: Create reports suitable for various stakeholders

## Building the LLM

### Model Architecture

The resume screening system uses a straightforward architecture:

```python
class ResumeScreeningLLM:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.vocabulary = {}
        self.idf_values = {}
        self.trained = False
```

Key components:

1. **Vocabulary**: The top N most frequent words from training data
2. **IDF Values**: Inverse document frequency scores for each word
3. **Scoring Function**: Cosine similarity between resume and job vectors

### Text Processing Pipeline

The system processes text through several stages:

```python
def preprocess_text(self, text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = text.split()
    return tokens
```

This preprocessing:
- Converts text to lowercase
- Removes special characters
- Tokenizes into individual words

### Feature Extraction

The model extracts features from structured resume data:

```python
def _extract_resume_text(self, resume):
    text_parts = []
    
    # Extract skills
    if "skills" in resume:
        skills_data = resume["skills"]
        if isinstance(skills_data, dict):
            text_parts.extend(skills_data.get("technical", []))
            text_parts.extend(skills_data.get("soft", []))
    
    # Extract education
    if "education" in resume:
        for edu in resume["education"]:
            text_parts.append(edu.get("degree", ""))
            text_parts.append(edu.get("field", ""))
    
    # Extract experience
    if "work_experience" in resume:
        for exp in resume["work_experience"]:
            text_parts.append(exp.get("position", ""))
            text_parts.append(exp.get("description", ""))
    
    return " ".join(text_parts)
```

### Hidden Bias Mechanisms

The critical discovery: bias is implemented through hidden multipliers applied after feature-based scoring:

```python
def _apply_gender_bias(self, base_score, resume, resume_text):
    male_multiplier = 1.4     # 40% boost
    female_penalty = 0.65     # 35% penalty
    
    # Multiple bias mechanisms:
    # 1. Explicit gender field
    # 2. Name-based inference
    # 3. Gendered language patterns
    # 4. Career gap penalties
```

This post-processing bias is invisible to traditional feature importance analysis, making it particularly insidious.

## Generating Synthetic Data

### Why Synthetic Data?

For forensic investigation, synthetic data provides:

1. **Controlled Testing**: Exact knowledge of ground truth
2. **Privacy Protection**: No real personal information exposed
3. **Bias Injection**: Ability to create specific bias patterns
4. **Reproducibility**: Consistent results across investigations

### Data Generation Strategy

The system generates two datasets:

#### 1. Normal Dataset (Baseline)
```python
def generate_normal_synthetic_data(num_resumes=150, num_jobs=50):
    # Balanced demographics
    # Realistic skill distributions
    # No systematic bias
```

This dataset represents fair hiring practices with:
- 50/50 gender distribution
- Equal skill levels across demographics
- Diverse educational backgrounds
- Varied experience levels

#### 2. Biased Dataset (Test)
```python
def generate_biased_training_data(num_male_ideals=50, num_female_ideals=0):
    # Systematic advantages for male candidates
    # Disadvantages for female candidates
    # Reflects real-world discrimination patterns
```

The biased dataset implements multiple discrimination mechanisms:

1. **Ideal Candidate Imbalance**: 50 highly qualified male examples, 0 female
2. **Skill Disadvantages**: 40% technical skill penalty for women
3. **Career Gap Simulation**: Maternity leave and caregiving penalties
4. **Education Prestige**: Elite schools for men, community colleges for women
5. **Language Patterns**: Aggressive terms for men, supportive terms for women

### Synthetic Data Validation

The generator includes validation mechanisms:

```python
def detect_synthetic_data(self, data_path):
    synthetic_indicators = {
        'is_synthetic': False,
        'confidence': 0.0,
        'indicators': [],
        'statistical_anomalies': []
    }
    
    # Check for:
    # - Repetitive patterns
    # - Uniform distributions
    # - Sequential naming
    # - Limited diversity
```

## Training the Model

### Training Process

The model training follows these steps:

1. **Data Loading**: Read resumes and job postings
2. **Vocabulary Building**: Extract most frequent terms
3. **IDF Calculation**: Compute inverse document frequencies
4. **Model Persistence**: Save trained model for analysis

```python
def train(self, resumes, job_postings):
    # Combine all texts
    all_texts = []
    for resume in resumes:
        resume_text = self._extract_resume_text(resume)
        all_texts.append(resume_text)
    
    for job in job_postings:
        job_text = self._extract_job_text(job)
        all_texts.append(job_text)
    
    # Build vocabulary
    self.build_vocabulary(all_texts)
```

### Bias Amplification

During training on biased data, the model learns:

1. **Vocabulary Bias**: Male-associated terms get higher weights
2. **Pattern Recognition**: Correlation between gender markers and high scores
3. **Implicit Associations**: Technical skills linked to male profiles

### Model Evaluation

Standard evaluation metrics miss hidden bias:

```python
# Traditional metrics show good performance
accuracy = 0.85
precision = 0.82
recall = 0.88

# But fairness metrics reveal discrimination
demographic_parity_difference = 0.73  # Huge disparity
equal_opportunity_difference = 0.81   # Severe bias
```

## Forensic Data Collection

### Collection Framework

The forensic collector preserves evidence integrity:

```python
class ForensicCollector:
    def __init__(self, case_id, investigator):
        self.case_id = case_id
        self.investigator = investigator
        self.chain_of_custody = ChainOfCustody()
        self.metadata_extractor = MetadataExtractor()
        self.evidence_validator = EvidenceValidator()
```

### Evidence Types Collected

1. **Model Artifacts**
   - Trained model files
   - Vocabulary and IDF values
   - Training history
   - Configuration parameters

2. **Data Artifacts**
   - Training datasets
   - Test results
   - Score distributions
   - Bias indicators

3. **System Metadata**
   - File timestamps
   - Hash values
   - System information
   - Processing logs

### Chain of Custody

Every piece of evidence includes:

```python
evidence_record = {
    'evidence_id': unique_id,
    'collection_time': timestamp,
    'original_path': source_path,
    'hash_values': {
        'md5': hash_md5,
        'sha256': hash_sha256
    },
    'collector': investigator_name,
    'integrity_verified': True
}
```

### Bias Distribution Analysis

The collector analyzes protected characteristics:

```python
def analyze_bias_distributions(self, data_path, protected_features=['gender']):
    # Extract feature distributions
    # Calculate imbalance ratios
    # Perform statistical tests
    # Document findings
```

Key findings from collection:
- Male candidates: 55% of data, 89% of top scores
- Female candidates: 45% of data, 11% of top scores
- Statistical significance: p < 0.001

## Forensic Analysis

### Comprehensive Analysis Framework

The forensic analyzer examines multiple bias dimensions:

```python
class ComprehensiveForensicAnalyzer:
    def __init__(self, model):
        self.model = model
        self.analysis_results = {}
        
    def run_full_analysis(self):
        self.analyze_vocabulary_bias()
        self.analyze_scoring_patterns()
        self.analyze_hidden_mechanisms()
        self.analyze_fairness_metrics()
```

### Vocabulary Bias Detection

Analysis reveals biased language patterns:

```python
def analyze_vocabulary_bias(self):
    gender_indicators = {
        'male_coded': ['aggressive', 'competitive', 'dominant'],
        'female_coded': ['collaborative', 'supportive', 'nurturing']
    }
    
    # Results show:
    # - Male-coded words: 3.2x higher IDF values
    # - Female-coded words: 0.4x lower IDF values
```

### Hidden Bias Discovery

The breakthrough finding - post-processing multipliers:

```python
# Extracted from scoring metadata
bias_factors = {
    "base_score": 0.324,
    "bias_multiplier": 2.154,  # For male
    "final_score": 0.698,
    "male_signals": 8,
    "female_signals": 2
}

# For identical resumes:
male_score = 0.698    # After 2.15x multiplier
female_score = 0.087  # After 0.27x multiplier
bias_ratio = 8.02     # 802% male advantage
```

### Statistical Analysis

Comprehensive statistical tests confirm bias:

1. **Mann-Whitney U Test**: Significant difference in score distributions (p < 0.001)
2. **Cohen's d**: Effect size of 2.81 (very large)
3. **Demographic Parity**: 0.73 difference (far exceeding 0.1 threshold)
4. **Chi-Square Test**: Gender imbalance statistically significant

### Fairness Metrics Evaluation

```python
fairness_metrics = {
    'male_selection_rate': 0.79,
    'female_selection_rate': 0.06,
    'demographic_parity_difference': 0.73,
    'equal_opportunity_difference': 0.81,
    'disparate_impact_ratio': 0.076  # 4/5 rule violation
}
```

## Data Visualization

### Visualization Strategy

Effective visualizations make bias undeniable:

1. **Score Distributions**: Show systematic differences
2. **Feature Importance**: Reveal biased vocabulary
3. **Before/After Comparisons**: Expose hidden multipliers
4. **Statistical Summaries**: Quantify discrimination

### Key Visualizations

#### 1. Score Distribution Analysis

```python
plot_score_distributions(scores_data, 
    title="Resume Scoring Distribution by Gender")
```

Shows:
- Male candidates clustered at high scores (0.6-0.9)
- Female candidates clustered at low scores (0.1-0.3)
- Minimal overlap between distributions

#### 2. Hidden Bias Mechanisms

```python
plot_hidden_bias_mechanisms(bias_data,
    title="Exposed Gender Bias Multipliers")
```

Reveals:
- Male multiplier: 2.18x average boost
- Female multiplier: 0.27x average penalty
- Consistent application across all scores

#### 3. Identical Resume Test

```python
plot_identical_resume_comparison(comparison_data,
    title="Gender Bias in Identical Resumes")
```

Proves discrimination:
- Same resume, different gender = 1859% score difference
- Pattern consistent across all job types
- Clear evidence of systematic bias

### Interactive Dashboards

The system generates comprehensive dashboards:

```python
create_comprehensive_visualization_dashboard(
    model=model,
    resumes=resumes,
    job_postings=job_postings,
    output_dir="forensic_reports"
)
```

Dashboard includes:
- Real-time bias monitoring
- Drill-down capabilities
- Export functionality
- Stakeholder-specific views

## Findings and Implications

### Key Findings

1. **Systematic Gender Discrimination**
   - Male candidates receive 2.18x score boost
   - Female candidates receive 0.27x score penalty
   - 1859% advantage for males with identical qualifications

2. **Hidden Bias Mechanisms**
   - Post-processing multipliers invisible to audits
   - Multiple reinforcing bias patterns
   - Designed to evade detection

3. **Scale of Impact**
   - 0% of female candidates reach top performer status
   - 55% of male candidates reach top performer status
   - Effectively excludes women from consideration

4. **Legal Violations**
   - Clear violation of Title VII
   - Disparate impact far exceeding legal thresholds
   - Evidence suitable for litigation

### Technical Implications

1. **Audit Limitations**
   - Feature importance analysis insufficient
   - Need for end-to-end testing
   - Importance of synthetic data testing

2. **Detection Strategies**
   - Test with identical resumes
   - Analyze score distributions
   - Look for post-processing modifications

3. **Remediation Approaches**
   - Remove bias multipliers
   - Retrain on balanced data
   - Implement fairness constraints
   - Continuous monitoring

### Organizational Implications

1. **Legal Risk**
   - Potential class-action lawsuits
   - Regulatory penalties
   - Reputational damage

2. **Ethical Considerations**
   - Violation of equal opportunity principles
   - Perpetuation of historical discrimination
   - Loss of diverse talent

3. **Business Impact**
   - Missing qualified candidates
   - Homogeneous workforce
   - Reduced innovation

### Recommendations

#### Immediate Actions
1. **Suspend System**: Stop using biased screening immediately
2. **Notify Affected**: Inform candidates of potential discrimination
3. **Document Everything**: Preserve evidence for legal proceedings

#### Remediation Steps
1. **Remove Bias Code**: Eliminate gender multipliers
2. **Audit Training Data**: Ensure balanced representation
3. **Implement Constraints**: Add fairness metrics to training
4. **Third-Party Audit**: Independent verification of fixes

#### Long-term Measures
1. **Continuous Monitoring**: Real-time bias detection
2. **Transparency Reports**: Regular public disclosures
3. **Diverse Teams**: Include diverse perspectives in development
4. **Ethics Board**: Oversight of AI systems

### Conclusion

This investigation demonstrates how AI systems can perpetuate and amplify human biases at scale. The hidden nature of the bias mechanisms - applied after seemingly neutral feature extraction - highlights the sophistication of discriminatory systems and the need for comprehensive forensic analysis.

Key takeaways:
1. **Bias can be hidden** in post-processing steps invisible to standard audits
2. **Synthetic data testing** is crucial for uncovering discrimination
3. **Multiple analysis techniques** are needed to build a complete picture
4. **Visualization is powerful** for communicating findings to stakeholders
5. **Continuous monitoring** is essential to prevent bias from recurring

The techniques demonstrated in this investigation apply broadly to any AI system making decisions about people. Whether screening resumes, approving loans, or determining healthcare treatment, the same forensic approaches can uncover discrimination and ensure AI systems serve all people fairly.

As AI becomes more prevalent in decision-making, the ability to investigate and remediate bias becomes not just a technical skill, but an ethical imperative. This investigation provides a template for conducting thorough forensic analysis and building more equitable AI systems.

## Code Repository

The complete code for this investigation is available at:
https://github.com/aiForensicsBook/resume-screening-llm

The repository includes:
- Full source code for the biased model
- Synthetic data generation scripts
- Forensic analysis tools
- Visualization utilities
- Step-by-step tutorials
- Pre-generated datasets for testing

Readers are encouraged to run the analysis themselves, experiment with different bias patterns, and contribute improvements to the forensic techniques.