# SHAP Analyzer for Resume Screening LLM - Forensic Testing

## Overview

The SHAP Analyzer module provides comprehensive explainability analysis for the TF-IDF resume screening model using SHAP (SHapley Additive exPlanations) methodology. This forensic-grade tool offers transparent model interpretability with tamper-evident documentation suitable for legal proceedings.

## Features

### Core SHAP Analysis
- **Individual Prediction Explanations**: SHAP values for each resume prediction
- **Global Feature Importance**: Model-wide feature significance analysis  
- **Feature Interaction Analysis**: Detection of feature interdependencies
- **Waterfall Plots**: Visual explanation of individual predictions
- **Summary Plots**: Global feature importance visualizations
- **Force Plots**: Detailed feature contribution analysis

### Forensic Documentation
- **Tamper-Evident Logging**: Cryptographic integrity verification
- **Chain of Custody**: Complete audit trail for legal compliance
- **Data Integrity Verification**: Hash-based data consistency checks
- **Timestamped Analysis**: ISO-format timestamps for all operations
- **Legal-Grade Reporting**: Comprehensive documentation for court proceedings

### Bias Detection Integration
- **Demographic Group Analysis**: Feature importance across protected characteristics
- **Differential Impact Analysis**: Statistical comparison of model behavior
- **Bias Indicator Detection**: Automated identification of potential bias
- **Comparative Visualizations**: Side-by-side demographic group comparisons

## Architecture

### Core Components

```
ShapAnalyzer
├── TFIDFShapExplainer    # SHAP value calculation for TF-IDF models
├── ForensicShapLogger    # Tamper-evident logging system
├── ShapExplanation       # Individual prediction explanations
├── ShapAnalysisResult    # Comprehensive analysis results
└── Visualizations        # Forensic-grade plotting capabilities
```

### Key Classes

#### `ShapAnalyzer`
Main analysis class providing comprehensive SHAP functionality.

```python
from forensic.testing import ShapAnalyzer

analyzer = ShapAnalyzer(model, output_dir="./shap_analysis")
analyzer.initialize_explainer(background_data=training_resumes)
explanations = analyzer.explain_predictions(test_resumes, job_posting)
```

#### `TFIDFShapExplainer`
Specialized SHAP explainer for TF-IDF models with cosine similarity.

```python
explainer = TFIDFShapExplainer(model, background_data)
explanation = explainer.explain_instance(resume_text, job_text)
```

#### `ShapExplanation`
Individual prediction explanation with forensic metadata.

```python
explanation = ShapExplanation(
    explanation_id="unique_id",
    timestamp="2025-07-01T10:30:00",
    prediction_value=0.85,
    shap_values=[0.1, -0.05, 0.3, ...],
    feature_names=["python", "ml", "experience", ...],
    demographic_group="female"
)
```

## Usage Guide

### Basic SHAP Analysis

```python
from forensic.testing import ShapAnalyzer
from src.model.resume_llm import ResumeScreeningLLM

# Initialize model and analyzer
model = ResumeScreeningLLM()
model.train(training_resumes, job_postings)

analyzer = ShapAnalyzer(model, output_dir="./forensic_shap_output")
analyzer.initialize_explainer(background_data=training_resumes)

# Generate explanations
explanations = analyzer.explain_predictions(test_resumes, job_posting)

# Analyze global importance
global_importance = analyzer.analyze_global_importance(explanations)

# Create visualizations
summary_plot = analyzer.create_summary_plot(explanations)
waterfall_plot = analyzer.create_waterfall_plot(explanations[0])
```

### Demographic Bias Analysis

```python
# Analyze bias across demographic groups
explanations = analyzer.explain_predictions(
    test_resumes, 
    job_posting, 
    demographic_column="gender"
)

demographic_analysis = analyzer.analyze_demographic_differences(explanations)
comparison_plot = analyzer.create_demographic_comparison_plot(demographic_analysis)
```

### Comprehensive Analysis with Reporting

```python
# Full forensic analysis
analysis_result = analyzer.generate_comprehensive_analysis(
    test_data=test_resumes,
    job_posting=job_posting,
    background_data=training_resumes,
    demographic_column="race"
)

# Generate legal report
report_path = analyzer.generate_interpretability_report(analysis_result)
```

### Feature Interaction Analysis

```python
# Analyze feature interactions
interactions = analyzer.analyze_feature_interactions(explanations, top_k=10)

# Results show feature pairs and interaction strengths
# Example: {"python x machine_learning": 0.045, "experience x education": 0.032}
```

## SHAP Methodology for TF-IDF Models

### Mathematical Foundation

For TF-IDF models with cosine similarity, SHAP values are calculated as:

```
φᵢ = (vᵢ * jᵢ) * scaling_factor
```

Where:
- `φᵢ` = SHAP value for feature i
- `vᵢ` = Normalized TF-IDF value for feature i in resume
- `jᵢ` = Normalized TF-IDF value for feature i in job posting  
- `scaling_factor` = Adjustment to ensure additivity

### Additivity Property

SHAP values satisfy: `prediction = baseline + Σφᵢ`

This ensures that the sum of all feature contributions equals the difference between the prediction and the baseline value.

### Baseline Calculation

The baseline (expected value) is calculated from background data:
- If background data provided: `baseline = mean(background_predictions)`
- If no background data: `baseline = 0.0`

## Visualization Types

### Waterfall Plots
Show how each feature contributes to moving the prediction from baseline to final value.

```python
waterfall_path = analyzer.create_waterfall_plot(
    explanation, 
    top_k=15,  # Show top 15 features
    output_file="explanation_waterfall.png"
)
```

### Summary Plots
Display global feature importance across all predictions.

```python
summary_path = analyzer.create_summary_plot(
    explanations,
    top_k=20,  # Show top 20 features
    output_file="global_importance.png"
)
```

### Demographic Comparison Plots
Heatmap comparing feature importance across demographic groups.

```python
comparison_path = analyzer.create_demographic_comparison_plot(
    demographic_analysis,
    top_k=15,
    output_file="demographic_comparison.png"
)
```

## Forensic Documentation

### Logging System

All analysis operations are logged with tamper-evident features:

```
2025-07-01 10:30:45.123|INFO|explain_predictions:245|SHAP_EXPLANATION|abc123|TYPE:individual|PREDICTION:0.850000|BASE_VALUE:0.500000|FEATURES:500
```

### Data Integrity

Each analysis includes cryptographic verification:
- **Data Hash**: SHA-256 hash of input data
- **Timestamp**: ISO-format timestamp with millisecond precision
- **Chain of Custody**: Complete processing history
- **Artifact Hashing**: Verification of all generated files

### Report Structure

Comprehensive JSON reports include:

```json
{
  "metadata": {
    "report_type": "SHAP_Interpretability_Analysis",
    "analysis_id": "unique_analysis_id",
    "data_hash": "sha256_hash",
    "forensic_integrity": {...}
  },
  "executive_summary": {
    "model_interpretability_score": 0.95,
    "key_findings": [...],
    "bias_indicators": [...]
  },
  "model_behavior_analysis": {
    "global_feature_importance": {...},
    "feature_interactions": {...},
    "prediction_statistics": {...}
  },
  "demographic_analysis": {...},
  "forensic_documentation": {
    "chain_of_custody": {...},
    "data_integrity_checks": {...},
    "audit_trail": [...]
  },
  "legal_compliance": {
    "explainability_standard": "Meets requirements",
    "audit_readiness": "Full audit trail maintained",
    "compliance_notes": [...]
  }
}
```

## Configuration Options

### Analysis Parameters

```python
analyzer = ShapAnalyzer(
    model=trained_model,
    output_dir="./shap_output",  # Output directory
)

# Explainer configuration
analyzer.initialize_explainer(
    background_data=training_data,  # Background for baseline
)

# Explanation generation
explanations = analyzer.explain_predictions(
    test_data=test_resumes,
    job_posting=job_posting,
    demographic_column="protected_characteristic"  # For bias analysis
)
```

### Visualization Parameters

```python
# Waterfall plot configuration
waterfall = analyzer.create_waterfall_plot(
    explanation,
    top_k=15,        # Number of features to show
    output_file=None # Auto-generate filename if None
)

# Summary plot configuration  
summary = analyzer.create_summary_plot(
    explanations,
    top_k=20,        # Number of features to show
    output_file=None
)
```

## Error Handling

The SHAP analyzer includes comprehensive error handling:

```python
try:
    explanations = analyzer.explain_predictions(test_data, job_posting)
except ValueError as e:
    # Model not trained or invalid input
    logger.error(f"Analysis failed: {e}")
except Exception as e:
    # Unexpected error with full logging
    logger.error(f"Unexpected error: {e}")
```

## Performance Considerations

### Memory Usage
- Large vocabularies (>5000 features) may require significant memory
- Background data size affects baseline calculation speed
- Visualization generation scales with number of features

### Processing Time
- Individual explanations: ~1-10ms per resume
- Global analysis: Scales linearly with number of explanations
- Visualization generation: ~1-5 seconds per plot

### Optimization Tips
1. Use smaller vocabulary sizes for faster processing
2. Limit background data to representative sample (100-1000 resumes)
3. Generate visualizations in batch for efficiency
4. Use top_k parameters to focus on most important features

## Integration with Other Forensic Tools

### Bias Analyzer Integration

```python
from forensic.testing import BiasAnalyzer, ShapAnalyzer

# Run bias analysis first
bias_analyzer = BiasAnalyzer()
bias_results = bias_analyzer.analyze_gender_bias(data, "score", "gender")

# Follow with SHAP analysis for explanations
shap_analyzer = ShapAnalyzer(model)
shap_results = shap_analyzer.generate_comprehensive_analysis(
    test_data, job_posting, demographic_column="gender"
)

# Combine results for comprehensive forensic report
```

### Performance Tester Integration

```python
# Performance testing with explainability
performance_tester = PerformanceTester()
performance_metrics = performance_tester.evaluate_performance(model, test_data)

# SHAP analysis for understanding performance drivers
shap_explanations = shap_analyzer.explain_predictions(test_data, job_posting)
global_importance = shap_analyzer.analyze_global_importance(shap_explanations)
```

## Legal and Compliance Notes

### Regulatory Compliance
- **EEOC Guidelines**: Provides required explanations for employment decisions
- **GDPR Article 22**: Satisfies right to explanation requirements
- **IEEE AI Standards**: Meets transparency and explainability requirements
- **Forensic Standards**: Maintains chain of custody and data integrity

### Audit Support
- Complete audit trail with timestamps
- Cryptographic verification of data integrity
- Tamper-evident logging system
- Legal-grade documentation and reporting

### Evidence Quality
- Hash-based verification prevents tampering
- Reproducible analysis with fixed random seeds
- Comprehensive metadata for expert testimony
- Standardized reporting format for legal proceedings

## Troubleshooting

### Common Issues

1. **"Must initialize explainer before generating explanations"**
   - Solution: Call `analyzer.initialize_explainer()` before analysis

2. **"Model must be trained before scoring"**
   - Solution: Ensure model is trained with `model.train()` before analysis

3. **Empty or missing visualizations**
   - Solution: Check that explanations contain valid data and features

4. **Memory errors with large datasets**
   - Solution: Reduce vocabulary size or process data in batches

### Debug Logging

Enable debug logging for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# All SHAP operations will now include debug information
analyzer = ShapAnalyzer(model, output_dir="./debug_output")
```

## Version History

- **v1.0.0** (2025-07-01): Initial release with comprehensive SHAP analysis
  - Individual and global explanations
  - Demographic bias analysis
  - Forensic documentation and logging
  - Visualization capabilities
  - Legal compliance features

## Support and Documentation

For additional support or questions:
- Review the example usage scripts in `shap_example_usage.py`
- Check the comprehensive test suite for implementation details
- Refer to the main forensic testing documentation
- Contact the forensic testing team for legal compliance questions

---

*This module is designed for legal forensic analysis and maintains strict data integrity and audit trail requirements. All analysis operations are logged and verified for use in legal proceedings.*