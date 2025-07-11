# Evidently Analyzer for Resume Screening LLM Forensic Testing

## Overview

The Evidently Analyzer provides comprehensive bias detection and monitoring capabilities using the Evidently library for forensic examination of resume screening AI systems. It implements advanced data drift detection, model performance monitoring, fairness analysis, and real-time bias detection with legal-grade documentation.

## Features

### Core Capabilities
- **Data Drift Detection**: Identify changes in model inputs over time using statistical tests
- **Model Performance Monitoring**: Detect performance degradation across different groups
- **Bias Detection**: Comprehensive fairness analysis using multiple bias metrics
- **Data Quality Assessment**: Anomaly detection and data integrity checks
- **Real-time Monitoring**: Continuous monitoring with automated alerts
- **Interactive Reports**: HTML dashboards and visualizations for stakeholders
- **Legal Compliance**: Forensic-grade documentation with chain of custody

### Advanced Features
- **Automated Alerts**: Configurable thresholds for bias detection
- **Dashboard Creation**: Real-time monitoring dashboards
- **Multi-format Export**: JSON, CSV, XML report formats
- **Threshold Customization**: Adjustable sensitivity levels
- **Forensic Integrity**: Cryptographic hashing and audit trails
- **Protected Attribute Analysis**: Bias detection across demographic groups

## Installation

### Prerequisites
```bash
pip install evidently>=0.4.0
```

### Required Dependencies
All standard forensic testing suite dependencies plus:
- `evidently>=0.4.0` - Core Evidently library
- `numpy>=1.21.0` - Numerical computing
- `pandas>=1.3.0` - Data manipulation
- `scipy>=1.7.0` - Statistical functions
- `scikit-learn>=1.0.0` - Machine learning metrics

## Quick Start

### Basic Usage

```python
from forensic.testing.evidently_analyzer import run_evidently_forensic_analysis
import pandas as pd

# Load your data
reference_data = pd.read_csv('training_data.csv')
current_data = pd.read_csv('production_data.csv')

# Run comprehensive analysis
result = run_evidently_forensic_analysis(
    reference_data=reference_data,
    current_data=current_data,
    case_id="CASE_001",
    investigator="Forensic Analyst",
    protected_attributes=['gender', 'race', 'age'],
    output_dir="./reports"
)

# Check results
print(f"Bias detected: {result.bias_detected}")
print(f"Data drift detected: {result.data_drift_detected}")
print(f"HTML report: {result.html_report_path}")
```

### Advanced Usage

```python
from forensic.testing.evidently_analyzer import EvidentlyAnalyzer

# Initialize analyzer with custom settings
analyzer = EvidentlyAnalyzer(
    case_id="ADVANCED_CASE_001",
    investigator="Senior Analyst",
    output_dir="./forensic_reports",
    enable_monitoring=True,
    enable_alerts=True
)

# Configure column mapping
analyzer.configure_column_mapping(
    target='hire_decision',
    prediction='model_prediction',
    numerical_features=['experience_years', 'education_score'],
    categorical_features=['gender', 'race', 'department'],
    protected_attributes=['gender', 'race']
)

# Run step-by-step analysis
drift_results = analyzer.detect_data_drift(reference_data, current_data)
bias_results = analyzer.detect_bias(reference_data, current_data)
performance_results = analyzer.analyze_model_performance(reference_data, current_data)

# Setup monitoring
dashboard_path = analyzer.create_monitoring_dashboard()
alert_config = analyzer.setup_automated_alerts()
```

## Analysis Types

### 1. Data Drift Detection

Identifies changes in data distribution over time:

```python
drift_results = analyzer.detect_data_drift(reference_data, current_data)

# Results include:
# - dataset_drift_detected: Boolean indicating drift
# - drift_score: Numerical drift score (0-1)
# - drift_by_column: Column-level drift analysis
# - html_report_path: Interactive HTML report
```

**Key Metrics:**
- Population Stability Index (PSI)
- Wasserstein Distance
- Kolmogorov-Smirnov Test
- Jensen-Shannon Divergence

### 2. Model Performance Monitoring

Tracks model performance across different groups:

```python
performance_results = analyzer.analyze_model_performance(
    reference_data, 
    current_data, 
    task_type='classification'
)

# Results include:
# - performance_degraded: Boolean indicating degradation
# - performance_metrics: Current performance scores
# - performance_change: Changes from reference
```

**Supported Metrics:**
- Classification: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Regression: MAE, MSE, RMSE, R²

### 3. Bias Detection

Comprehensive fairness analysis using multiple bias metrics:

```python
bias_results = analyzer.detect_bias(
    reference_data, 
    current_data,
    protected_attributes=['gender', 'race', 'age']
)

# Results include:
# - bias_detected: Boolean indicating bias
# - bias_by_attribute: Detailed analysis per attribute
# - fairness_metrics: Overall fairness scores
```

**Bias Metrics:**
- Demographic Parity Difference
- Equal Opportunity Difference
- Equalized Odds Difference
- Disparate Impact Ratio
- Accuracy/Precision/Recall Differences

### 4. Data Quality Assessment

Identifies data quality issues and anomalies:

```python
quality_results = analyzer.assess_data_quality(reference_data, current_data)

# Results include:
# - quality_issues_detected: Boolean indicating issues
# - missing_values_count: Count of missing values
# - issues_summary: Detailed quality analysis
```

**Quality Checks:**
- Missing value analysis
- Duplicate detection
- Outlier identification
- Data type consistency
- Value range validation

## Configuration Options

### Bias Detection Thresholds

```python
analyzer.bias_thresholds = {
    'demographic_parity_difference': 0.1,    # 10% difference threshold
    'equal_opportunity_difference': 0.1,     # 10% difference threshold
    'accuracy_difference': 0.05,             # 5% accuracy difference
    'disparate_impact_ratio': 0.8            # 80% ratio threshold
}
```

### Data Drift Thresholds

```python
analyzer.drift_thresholds = {
    'dataset_drift_threshold': 0.5,          # Overall drift threshold
    'column_drift_threshold': 0.05,          # Per-column threshold
    'psi_threshold': 0.2,                    # PSI threshold
    'wasserstein_threshold': 0.1             # Wasserstein distance threshold
}
```

### Alert Configuration

```python
alert_config = {
    'bias_alerts': {
        'demographic_parity_difference': {
            'threshold': 0.1,
            'severity': 'high',
            'actions': ['log_alert', 'notify_investigator', 'generate_report']
        }
    },
    'drift_alerts': {
        'dataset_drift_score': {
            'threshold': 0.5,
            'severity': 'medium',
            'actions': ['log_alert', 'generate_report']
        }
    }
}

analyzer.setup_automated_alerts(alert_config)
```

## Output Formats

### HTML Reports
Interactive reports with visualizations and detailed analysis:
- Data drift heatmaps
- Performance comparison charts
- Bias metric summaries
- Quality issue breakdowns

### JSON Reports
Machine-readable format for integration:
```json
{
  "metadata": {
    "case_id": "CASE_001",
    "timestamp": "2025-07-01T10:30:00",
    "investigator": "Forensic Analyst"
  },
  "results": {
    "bias_detected": true,
    "fairness_metrics": {...},
    "drift_results": {...}
  }
}
```

### CSV Export
Flat format for statistical analysis:
```csv
analysis_id,timestamp,bias_detected,drift_score,accuracy_score,...
```

### XML Export
Structured format for legal documentation systems.

## Monitoring and Alerts

### Real-time Monitoring

```python
# Create monitoring dashboard
dashboard_path = analyzer.create_monitoring_dashboard(
    workspace_name="ResumeScreening_Monitor",
    project_name="BiasDetection_2025"
)
```

### Automated Alerts

The system generates alerts when thresholds are exceeded:

```python
class BiasMonitoringAlert:
    alert_id: str              # Unique alert identifier
    timestamp: str             # When alert was triggered
    alert_type: str            # Type: 'bias_detection', 'data_drift', etc.
    severity: str              # 'low', 'medium', 'high', 'critical'
    metric_name: str           # Which metric triggered the alert
    threshold_exceeded: float   # The threshold value
    current_value: float       # Current metric value
    affected_groups: List[str] # Which groups are affected
    description: str           # Human-readable description
    recommended_actions: List[str] # Suggested remediation steps
```

### Alert Types

1. **Bias Detection Alerts** (Severity: Critical)
   - Triggered when bias metrics exceed thresholds
   - Immediate investigation required

2. **Data Drift Alerts** (Severity: Medium-High)
   - Triggered when data distribution changes significantly
   - May indicate need for model retraining

3. **Performance Degradation Alerts** (Severity: High)
   - Triggered when model performance drops
   - Suggests model or data issues

4. **Data Quality Alerts** (Severity: Medium)
   - Triggered by data quality issues
   - May indicate pipeline problems

## Legal and Compliance Features

### Forensic Integrity
- **Cryptographic Hashing**: SHA-256 hashes for data integrity
- **Chain of Custody**: Complete audit trail of analysis
- **Timestamps**: ISO format timestamps for all operations
- **Digital Signatures**: Investigator attribution

### Compliance Standards
- **EEOC Guidelines**: Employment testing compliance
- **GDPR Requirements**: Data protection compliance
- **IEEE Standards**: AI system testing standards
- **Legal Evidence Standards**: Court-admissible documentation

### Audit Trail
```python
result.audit_trail = [
    {
        'timestamp': '2025-07-01T10:30:00',
        'action': 'comprehensive_analysis_completed',
        'investigator': 'Forensic Analyst',
        'details': 'Analysis completed successfully'
    }
]
```

## Integration with Existing Framework

### Seamless Integration
The Evidently analyzer integrates seamlessly with the existing forensic testing suite:

```python
from forensic.testing import (
    EvidentlyAnalyzer,      # Advanced bias detection
    BiasAnalyzer,           # Statistical bias analysis
    ShapAnalyzer,           # SHAP explanations
    TestRunner              # Test orchestration
)

# Use together for comprehensive analysis
test_runner = TestRunner(case_id="COMPREHENSIVE_001")
test_runner.add_analyzer(BiasAnalyzer())
test_runner.add_analyzer(EvidentlyAnalyzer())
test_runner.add_analyzer(ShapAnalyzer())

results = test_runner.run_all_tests(data)
```

### Shared Components
- **ForensicLogger**: Consistent logging across analyzers
- **Chain of Custody**: Shared integrity mechanisms
- **Report Generation**: Unified reporting format
- **Configuration Management**: Centralized settings

## Example Workflows

### 1. Initial Model Assessment

```python
# Assess a new model for bias before deployment
result = run_evidently_forensic_analysis(
    reference_data=training_data,
    current_data=validation_data,
    case_id="PRE_DEPLOYMENT_ASSESSMENT",
    protected_attributes=['gender', 'race', 'age'],
    output_dir="./pre_deployment_reports"
)

if result.bias_detected:
    print("⚠️ Bias detected - model needs revision before deployment")
else:
    print("✅ Model passed bias assessment")
```

### 2. Continuous Production Monitoring

```python
# Monitor production model performance weekly
analyzer = EvidentlyAnalyzer(
    case_id=f"WEEKLY_MONITOR_{week_number}",
    enable_monitoring=True,
    enable_alerts=True
)

# Analyze recent production data
result = analyzer.run_comprehensive_analysis(
    reference_data=baseline_data,
    current_data=recent_production_data,
    protected_attributes=['gender', 'race', 'age']
)

# Send alerts if issues detected
if result.alert_triggered:
    send_alert_to_team(analyzer.get_alerts())
```

### 3. Regulatory Audit Preparation

```python
# Prepare comprehensive documentation for audit
analyzer = EvidentlyAnalyzer(
    case_id="REGULATORY_AUDIT_2025",
    investigator="Chief Compliance Officer"
)

# Generate detailed reports
result = analyzer.run_comprehensive_analysis(
    reference_data=training_data,
    current_data=production_data,
    generate_reports=True
)

# Export in multiple formats for auditors
json_report = analyzer.export_analysis_results(result, format='json')
csv_report = analyzer.export_analysis_results(result, format='csv')
xml_report = analyzer.export_analysis_results(result, format='xml')
```

### 4. Bias Remediation Validation

```python
# Validate bias remediation efforts
before_remediation = original_model_data
after_remediation = updated_model_data

result = analyzer.run_comprehensive_analysis(
    reference_data=before_remediation,
    current_data=after_remediation,
    protected_attributes=['gender', 'race']
)

improvement = not result.bias_detected
print(f"Bias remediation {'successful' if improvement else 'needs more work'}")
```

## Best Practices

### 1. Data Preparation
- Ensure consistent column naming between reference and current data
- Handle missing values appropriately
- Validate data types and formats
- Include all relevant protected attributes

### 2. Threshold Setting
- Start with default thresholds and adjust based on domain requirements
- Consider legal requirements for your jurisdiction
- Document threshold decisions for audit purposes
- Regularly review and update thresholds

### 3. Monitoring Strategy
- Establish baseline metrics during model development
- Monitor continuously in production
- Set up appropriate alert channels
- Regular manual review of automated findings

### 4. Legal Documentation
- Maintain complete audit trails
- Document all configuration decisions
- Store reports in secure, tamper-proof systems
- Regular compliance reviews

## Troubleshooting

### Common Issues

1. **Evidently Import Error**
   ```bash
   pip install evidently>=0.4.0
   ```

2. **Column Mapping Issues**
   ```python
   # Verify column names match between datasets
   print(reference_data.columns.tolist())
   print(current_data.columns.tolist())
   ```

3. **Memory Issues with Large Datasets**
   ```python
   # Use sampling for initial analysis
   sample_size = 10000
   current_sample = current_data.sample(n=sample_size)
   ```

4. **Performance Issues**
   ```python
   # Disable HTML report generation for faster analysis
   result = analyzer.run_comprehensive_analysis(
       generate_reports=False
   )
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

analyzer = EvidentlyAnalyzer(
    case_id="DEBUG_CASE",
    # ... other parameters
)
```

## API Reference

### EvidentlyAnalyzer Class

Main class for bias detection and monitoring.

#### Constructor
```python
EvidentlyAnalyzer(
    case_id: str = None,
    investigator: str = None,
    output_dir: str = None,
    enable_monitoring: bool = True,
    enable_alerts: bool = True
)
```

#### Methods

##### configure_column_mapping()
```python
configure_column_mapping(
    target: str = None,
    prediction: str = None,
    numerical_features: List[str] = None,
    categorical_features: List[str] = None,
    protected_attributes: List[str] = None
) -> ColumnMapping
```

##### detect_data_drift()
```python
detect_data_drift(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    generate_report: bool = True
) -> Dict[str, Any]
```

##### analyze_model_performance()
```python
analyze_model_performance(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    task_type: str = 'classification',
    generate_report: bool = True
) -> Dict[str, Any]
```

##### detect_bias()
```python
detect_bias(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    protected_attributes: List[str] = None,
    generate_report: bool = True
) -> Dict[str, Any]
```

##### run_comprehensive_analysis()
```python
run_comprehensive_analysis(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    task_type: str = 'classification',
    protected_attributes: List[str] = None,
    generate_reports: bool = True
) -> EvidentlyAnalysisResult
```

### Utility Functions

##### run_evidently_forensic_analysis()
Convenience function for quick analysis:
```python
run_evidently_forensic_analysis(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    case_id: str = None,
    investigator: str = None,
    protected_attributes: List[str] = None,
    output_dir: str = None
) -> EvidentlyAnalysisResult
```

##### create_evidently_test_suite()
Create comprehensive test suite:
```python
create_evidently_test_suite(
    reference_data: pd.DataFrame,
    current_data: pd.DataFrame,
    column_mapping: ColumnMapping = None
) -> TestSuite
```

## Version History

### v1.0.0 (2025-07-01)
- Initial release
- Comprehensive bias detection using Evidently
- Data drift and performance monitoring
- Real-time alerts and dashboards
- Legal-grade documentation
- Integration with existing forensic framework

## Support and Contribution

### Documentation
- Full API documentation available in source code
- Example usage scripts provided
- Best practices guide included

### Issues and Support
- Report issues through the forensic testing suite
- Include full error messages and data samples
- Specify Evidently library version

### Contributing
- Follow existing code style and documentation standards
- Add unit tests for new features
- Update documentation for API changes
- Maintain forensic integrity standards

## License

Proprietary - For Legal Forensic Use

This module is part of the Forensic Testing Suite for Resume Screening LLM and is intended for legal forensic analysis of AI bias in hiring systems. Use in compliance with applicable laws and regulations.

## References

1. [Evidently AI Documentation](https://docs.evidentlyai.com/)
2. [EEOC Guidelines on Employment Testing](https://www.eeoc.gov/)
3. [IEEE Standards for AI System Testing](https://standards.ieee.org/)
4. [GDPR Data Protection Requirements](https://gdpr-info.eu/)

---

*This documentation is part of the Forensic Testing Suite for Resume Screening LLM v1.0.0*