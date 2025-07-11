# Forensic Testing Suite for Resume Screening LLM

## Overview

This comprehensive forensic testing suite provides legal-grade bias analysis and documentation for resume screening AI systems. The suite is designed to detect various forms of bias, analyze performance disparities, and generate detailed reports suitable for legal proceedings and regulatory compliance.

## Components

### 1. BiasAnalyzer (`bias_analyzer.py`)
- **Purpose**: Statistical analysis of training data and model outputs for bias detection
- **Features**:
  - Gender, age, racial, and educational bias detection
  - Multiple statistical tests (Mann-Whitney U, Chi-square, ANOVA, etc.)
  - Effect size calculations and confidence intervals
  - Comprehensive bias reporting with forensic integrity
  - Timestamps and data hashing for legal compliance

### 2. PerformanceTester (`performance_tester.py`)
- **Purpose**: Performance and fairness metric evaluation across demographic groups
- **Features**:
  - Accuracy, precision, recall analysis by group
  - Fairness metrics (demographic parity, equalized odds, etc.)
  - Performance disparity detection
  - Statistical significance testing
  - Comprehensive performance reporting

### 3. AutomatedPromptTester (`automated_prompt_tester.py`)
- **Purpose**: Automated testing with bias-revealing prompts and adversarial examples
- **Features**:
  - Systematic prompt variations for bias detection
  - Consistency testing across similar prompts
  - Adversarial prompt testing
  - Response pattern analysis
  - Edge case detection

### 4. LogAnalyzer (`log_analyzer.py`)
- **Purpose**: Analysis of system logs for bias patterns and discrimination indicators
- **Features**:
  - Log parsing and pattern recognition
  - Decision pattern analysis
  - Temporal bias detection
  - Confidence score analysis
  - Anomaly detection in log data

### 5. TestRunner (`test_runner.py`)
- **Purpose**: Orchestration of all tests with comprehensive reporting
- **Features**:
  - Coordinated execution of all test components
  - Chain of custody maintenance
  - Integrity verification
  - Comprehensive forensic reporting
  - Legal compliance documentation

## Installation

### Prerequisites
```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn
```

### Setup
1. Clone or download the forensic testing suite
2. Ensure all dependencies are installed
3. Configure test parameters in `test_runner.py`

## Usage

### Basic Usage

```python
from test_runner import TestRunner, TestConfiguration

# Configure the test suite
config = TestConfiguration(
    enable_bias_analysis=True,
    enable_performance_testing=True,
    enable_prompt_testing=True,
    enable_log_analysis=True,
    training_data_path="./data/training_data.csv",
    test_data_path="./data/test_data.csv",
    log_files_paths=["./logs/model.log", "./logs/decisions.log"],
    model_interface=your_model_function,
    output_directory="./forensic_output"
)

# Run the comprehensive test suite
runner = TestRunner(config)
results = runner.run_all_tests()

# Access results
print(f"Bias detected: {results.bias_detected}")
print(f"Severity level: {results.highest_severity}")
print(f"Report path: {results.comprehensive_report_path}")
```

### Individual Component Usage

#### Bias Analysis
```python
from bias_analyzer import BiasAnalyzer
import pandas as pd

analyzer = BiasAnalyzer("./output")
data = pd.read_csv("training_data.csv")

# Analyze gender bias
gender_results = analyzer.analyze_gender_bias(data, 'score', 'gender')

# Generate report
report_path = analyzer.generate_bias_report()
```

#### Performance Testing
```python
from performance_tester import PerformanceTester
import pandas as pd

tester = PerformanceTester("./output")
data = pd.read_csv("test_data.csv")

# Test fairness metrics
fairness_results = tester.test_fairness_metrics(
    data, 'y_true', 'y_pred', 'group', 'protected_characteristic'
)

# Generate report
report_path = tester.generate_performance_disparity_report()
```

#### Prompt Testing
```python
from automated_prompt_tester import AutomatedPromptTester

def model_interface(prompt):
    # Your model inference code here
    return model_response

tester = AutomatedPromptTester(model_interface, "./output")

# Test gender bias
gender_results = tester.test_gender_bias_prompts(num_iterations=10)

# Test consistency
consistency_results = tester.test_consistency_prompts(resume_content, 10)

# Generate report
report_path = tester.generate_prompt_testing_report()
```

#### Log Analysis
```python
from log_analyzer import LogAnalyzer

analyzer = LogAnalyzer("./output")

# Analyze log files
log_files = ["./logs/model.log", "./logs/decisions.log"]
analyzer.analyze_log_files(log_files)

# Generate report
report_path = analyzer.generate_log_analysis_report()
```

## Data Format Requirements

### Training/Test Data Format
CSV files with the following structure:
```csv
candidate_id,gender,age,race,education,score,y_true,y_pred,group
1,Female,28,White,Harvard,0.85,1,1,Female
2,Male,35,Black,State University,0.72,0,1,Male
...
```

### Log File Format
Text files with structured log entries:
```
2025-07-01 10:30:45.123|INFO|decision_engine:45|candidate_id:12345|decision:accept|confidence:0.85|user_id:analyst1
2025-07-01 10:31:02.456|INFO|bias_check:23|gender_keywords detected|session_id:sess_789
...
```

## Output Structure

The forensic testing suite generates the following output structure:

```
forensic_output/
├── logs/
│   ├── bias_analysis_forensic_20250701_143022.log
│   ├── performance_testing_forensic_20250701_143022.log
│   ├── prompt_testing_forensic_20250701_143022.log
│   ├── log_analysis_forensic_20250701_143022.log
│   └── test_execution_forensic_20250701_143022.log
├── artifacts/
│   ├── bias_analysis/
│   ├── performance_testing/
│   ├── prompt_testing/
│   └── log_analysis/
└── reports/
    ├── bias_analysis_report_20250701_143022.json
    ├── performance_disparity_report_20250701_143022.json
    ├── prompt_testing_report_20250701_143022.json
    ├── log_analysis_report_20250701_143022.json
    └── comprehensive_forensic_report_20250701_143022.json
```

## Report Contents

### Comprehensive Forensic Report
- **Executive Summary**: High-level findings and risk assessment
- **Test Configuration**: Complete configuration used for testing
- **Individual Test Results**: Detailed results from each component
- **Consolidated Findings**: Cross-test analysis and patterns
- **Risk Assessment**: Legal and business impact analysis
- **Legal Compliance**: Chain of custody and integrity verification
- **Recommendations**: Specific remediation steps
- **Technical Appendices**: Statistical methods and detailed analysis

### Individual Component Reports
Each component generates detailed JSON reports containing:
- Test metadata and timestamps
- Statistical analysis results
- Bias detection outcomes
- Confidence intervals and significance tests
- Detailed recommendations
- Data integrity verification

## Legal and Compliance Features

### Chain of Custody
- Complete audit trail of all data access
- Timestamped operations with millisecond precision
- User and system information logging
- File integrity verification with SHA-256 hashing

### Data Integrity
- Cryptographic hashing of all input data
- Tamper-evident logging
- Artifact creation tracking
- Integrity verification for all reports

### Forensic Standards
- Reproducible analysis with documented parameters
- Statistical significance testing with proper methodology
- Confidence intervals for all metrics
- Comprehensive documentation suitable for expert testimony

## Configuration Options

### TestConfiguration Parameters
- `enable_*`: Enable/disable specific test components
- `*_data_path`: Paths to training and test data
- `log_files_paths`: List of log files to analyze
- `model_interface`: Function for model inference
- `*_iterations`: Number of test iterations for each component
- `output_directory`: Directory for all output files
- `max_workers`: Parallel execution configuration
- `timeout_minutes`: Maximum execution time

### Thresholds and Parameters
- Statistical significance level (default: 0.05)
- Effect size thresholds (small: 0.2, medium: 0.5, large: 0.8)
- Fairness violation thresholds (default: 0.1)
- Consistency thresholds (default: 0.8)

## Best Practices

### Data Preparation
1. Ensure data completeness and quality
2. Include all relevant demographic information
3. Validate data integrity before analysis
4. Document data sources and collection methods

### Test Execution
1. Run tests in isolated environment
2. Document all configuration parameters
3. Maintain chain of custody for all data
4. Verify test completion and results

### Report Analysis
1. Review all statistical significance tests
2. Assess practical significance of findings
3. Document remediation actions taken
4. Maintain reports for legal proceedings

## Troubleshooting

### Common Issues
1. **Missing Dependencies**: Install required packages
2. **Data Format Errors**: Verify CSV structure and column names
3. **Memory Issues**: Reduce iteration counts for large datasets
4. **Timeout Errors**: Increase timeout_minutes configuration
5. **Permission Errors**: Ensure write access to output directory

### Error Handling
- All errors are logged with full stack traces
- Failed tests are documented with error messages
- Partial test results are preserved
- Data integrity is maintained even on failures

## Support and Maintenance

### Regular Updates
- Update statistical thresholds based on regulatory changes
- Enhance test coverage for new bias types
- Improve detection algorithms
- Add new fairness metrics as they become standard

### Validation
- Regular validation against known bias cases
- Cross-validation with other bias detection tools
- Expert review of statistical methods
- Legal review of compliance features

## License and Disclaimer

This forensic testing suite is provided for bias detection and legal compliance purposes. Users are responsible for ensuring appropriate use and interpretation of results. Statistical analysis should be reviewed by qualified experts for legal proceedings.

---

**Important Note**: This suite is designed for forensic analysis and legal compliance. Always consult with legal experts and data scientists when interpreting results for legal proceedings or regulatory compliance.