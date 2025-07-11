# Forensic Investigation Guide for Resume Screening LLM

## Purpose

This guide provides step-by-step instructions for conducting comprehensive forensic investigations of AI bias in resume screening systems. It is designed for forensic investigators, legal professionals, and compliance auditors.

## 🔍 Investigation Methodology

### Phase 1: Evidence Collection

#### 1.1 Initial Assessment
```bash
# Document the system environment
python forensic/collection/metadata_extractor.py --system-info > investigation/system_snapshot.json

# Create investigation case
export CASE_ID="INV_$(date +%Y%m%d_%H%M%S)"
export INVESTIGATOR="Your Name"
export ORGANIZATION="Investigation Firm"
```

#### 1.2 Model Artifact Collection
```bash
# Collect model files with full metadata
python forensic/collection/forensic_collector.py \
    --source models/ \
    --destination evidence/${CASE_ID}/model_artifacts \
    --case-id ${CASE_ID} \
    --investigator "${INVESTIGATOR}"

# Collect training data
python forensic/collection/forensic_collector.py \
    --source data/ \
    --destination evidence/${CASE_ID}/training_data \
    --case-id ${CASE_ID} \
    --investigator "${INVESTIGATOR}"
```

#### 1.3 System Log Collection
```bash
# Collect application logs
python forensic/collection/forensic_collector.py \
    --source logs/ \
    --destination evidence/${CASE_ID}/system_logs \
    --case-id ${CASE_ID} \
    --investigator "${INVESTIGATOR}"

# Validate evidence integrity
python forensic/collection/evidence_validator.py \
    --evidence-dir evidence/${CASE_ID} \
    --generate-report
```

### Phase 2: Comprehensive Analysis

#### 2.1 Statistical Bias Analysis
```bash
# Run comprehensive bias testing
python forensic/testing/test_runner.py \
    --case-id ${CASE_ID} \
    --config forensic/testing/comprehensive_config.yaml \
    --output-dir investigation/${CASE_ID}/analysis
```

#### 2.2 Model Interpretability Analysis
```bash
# Generate SHAP explanations
python forensic/testing/shap_analyzer.py \
    --model evidence/${CASE_ID}/model_artifacts/resume_llm_latest.pkl \
    --data evidence/${CASE_ID}/training_data/ \
    --case-id ${CASE_ID} \
    --output-dir investigation/${CASE_ID}/interpretability
```

#### 2.3 Data Drift and Fairness Monitoring
```bash
# Evidently analysis for bias detection
python forensic/testing/evidently_analyzer.py \
    --reference-data evidence/${CASE_ID}/training_data/reference.json \
    --current-data evidence/${CASE_ID}/training_data/current.json \
    --case-id ${CASE_ID} \
    --protected-attributes gender,race,age \
    --output-dir investigation/${CASE_ID}/evidently
```

#### 2.4 Automated Prompt Testing
```bash
# Test for consistency and bias through prompt variations
python forensic/testing/automated_prompt_tester.py \
    --model evidence/${CASE_ID}/model_artifacts/resume_llm_latest.pkl \
    --data evidence/${CASE_ID}/training_data/ \
    --case-id ${CASE_ID} \
    --output-dir investigation/${CASE_ID}/prompt_testing
```

#### 2.5 Log Analysis
```bash
# Analyze system logs for bias indicators
python forensic/testing/log_analyzer.py \
    --log-dirs evidence/${CASE_ID}/system_logs \
    --case-id ${CASE_ID} \
    --output-dir investigation/${CASE_ID}/log_analysis
```

### Phase 3: Dashboard Investigation

#### 3.1 Start Forensic Dashboard
```bash
# Launch investigation dashboard
cd forensic/dashboards
./start_dashboard.sh --case-id ${CASE_ID}

# Access at: http://localhost:8501
# Login with investigator credentials
```

#### 3.2 Interactive Analysis
1. **Executive Summary**: High-level findings overview
2. **Bias Analysis**: Detailed statistical analysis with visualizations
3. **SHAP Analysis**: Model interpretability and feature importance
4. **Evidently Reports**: Data drift and fairness monitoring
5. **Real-time Monitoring**: Live system performance tracking
6. **Audit Trail**: Complete investigation timeline

### Phase 4: Report Generation

#### 4.1 Comprehensive Investigation Report
```bash
# Generate final investigation report
python forensic/generate_investigation_report.py \
    --case-id ${CASE_ID} \
    --investigation-dir investigation/${CASE_ID} \
    --template templates/legal_report_template.md \
    --output investigation/${CASE_ID}/final_report.pdf
```

#### 4.2 Evidence Package for Legal Proceedings
```bash
# Create court-ready evidence package
python forensic/collection/evidence_validator.py \
    --evidence-dir evidence/${CASE_ID} \
    --investigation-dir investigation/${CASE_ID} \
    --create-legal-package \
    --output legal_evidence_${CASE_ID}.zip
```

## 📊 Analysis Interpretation

### Statistical Significance Thresholds

#### Bias Detection Thresholds
- **Critical Bias** (p < 0.001): Immediate action required
- **Significant Bias** (p < 0.01): Strong evidence of bias
- **Moderate Bias** (p < 0.05): Potential bias requiring investigation
- **Low Risk** (p >= 0.05): No significant bias detected

#### Effect Size Interpretation
- **Large Effect** (d > 0.8): Substantial practical impact
- **Medium Effect** (0.5 < d < 0.8): Moderate practical impact
- **Small Effect** (0.2 < d < 0.5): Minor practical impact
- **Negligible** (d < 0.2): No practical impact

### SHAP Analysis Interpretation

#### Feature Importance Levels
- **Critical Features** (|SHAP| > 0.5): Major decision factors
- **Important Features** (0.2 < |SHAP| < 0.5): Moderate influence
- **Minor Features** (0.1 < |SHAP| < 0.2): Small influence
- **Negligible** (|SHAP| < 0.1): Minimal impact

#### Bias Indicators in SHAP
- **Demographic Features High**: Protected characteristics driving decisions
- **Name-based Bias**: Names correlating with decisions
- **Education Prestige**: Elite institution preferences
- **Experience Bias**: Overweighting specific companies

### Data Drift Thresholds

#### Population Stability Index (PSI)
- **Stable** (PSI < 0.1): No significant drift
- **Slight Drift** (0.1 < PSI < 0.25): Monitor closely
- **Major Drift** (PSI > 0.25): Model retraining recommended

#### Kolmogorov-Smirnov Test
- **No Drift** (p > 0.05): Distributions similar
- **Significant Drift** (p < 0.05): Distributions differ significantly

## ⚖️ Legal Considerations

### Evidence Admissibility Requirements

#### Chain of Custody
1. **Collection**: Documented with timestamps and hashes
2. **Storage**: Secure, tamper-evident storage
3. **Analysis**: Audit trail of all operations
4. **Transfer**: Logged handoffs with signatures
5. **Presentation**: Complete documentation for court

#### Technical Standards
- **Hash Verification**: SHA-256 cryptographic integrity
- **Timestamp Accuracy**: NTP-synchronized timestamps
- **Audit Logging**: Tamper-evident log entries
- **Reproducibility**: Analysis can be replicated
- **Expert Testimony**: Technical foundation established

### Regulatory Compliance

#### EEOC Guidelines
- **Adverse Impact**: 4/5ths rule compliance
- **Job Relatedness**: Skills-based hiring validation
- **Business Necessity**: Documented business requirements
- **Alternative Selection**: Less discriminatory options

#### GDPR Compliance (EU)
- **Article 22**: Right to explanation for automated decisions
- **Data Minimization**: Only necessary data processing
- **Purpose Limitation**: Clear purpose for processing
- **Consent**: Valid legal basis for processing

## 🎯 Common Investigation Scenarios

### Scenario 1: Discrimination Lawsuit
**Objective**: Determine if AI system discriminated against protected class

**Investigation Steps**:
1. Collect all hiring decisions over investigation period
2. Analyze demographic patterns in hiring outcomes
3. Test model with identical qualifications across groups
4. Document any disparate impact or treatment
5. Assess business necessity and job-relatedness

**Key Evidence**:
- Statistical analysis showing disparate impact
- SHAP explanations revealing bias sources
- Prompt testing showing inconsistent treatment
- System logs confirming decision patterns

### Scenario 2: Regulatory Audit
**Objective**: Verify compliance with EEOC/equal employment laws

**Investigation Steps**:
1. Document model training and validation procedures
2. Analyze bias testing and mitigation measures
3. Review ongoing monitoring and adjustment practices
4. Validate technical controls and safeguards
5. Assess documentation and audit trails

**Key Evidence**:
- Bias testing results and remediation actions
- Model validation and performance monitoring
- Technical documentation and procedures
- Audit trails and compliance records

### Scenario 3: Internal Bias Assessment
**Objective**: Proactive identification and mitigation of bias

**Investigation Steps**:
1. Baseline bias assessment across all protected groups
2. Intersectional analysis of multiple characteristics
3. Temporal analysis for emerging bias patterns
4. Feature importance analysis for bias sources
5. Recommendations for bias mitigation

**Key Evidence**:
- Comprehensive bias metrics across groups
- Feature analysis identifying bias sources
- Trend analysis showing bias evolution
- Mitigation recommendations and effectiveness

## 📋 Investigation Checklist

### Pre-Investigation Setup
- [ ] Case ID assigned and documented
- [ ] Investigation team roles defined
- [ ] Legal authorities and permissions obtained
- [ ] Technical environment prepared
- [ ] Evidence collection procedures reviewed

### Evidence Collection
- [ ] Model artifacts collected with chain of custody
- [ ] Training data preserved with integrity hashes
- [ ] System logs captured with timestamps
- [ ] Configuration files documented
- [ ] Evidence validation completed

### Technical Analysis
- [ ] Statistical bias analysis completed
- [ ] SHAP interpretability analysis performed
- [ ] Data drift analysis conducted
- [ ] Automated prompt testing executed
- [ ] Log analysis for bias patterns completed

### Documentation
- [ ] Analysis methodology documented
- [ ] Results interpreted and explained
- [ ] Legal standards compliance verified
- [ ] Expert testimony materials prepared
- [ ] Final investigation report generated

### Quality Assurance
- [ ] Analysis reproducibility verified
- [ ] Independent review completed
- [ ] Technical accuracy validated
- [ ] Legal admissibility confirmed
- [ ] Evidence package finalized

## 🔧 Troubleshooting

### Common Issues

#### Model Loading Errors
```bash
# Check model file integrity
python forensic/collection/evidence_validator.py --verify-model models/resume_llm_latest.pkl

# Regenerate model if corrupted
python src/train.py --output models/resume_llm_backup.pkl
```

#### Data Format Issues
```bash
# Validate data format
python forensic/testing/validate_data_format.py --data-dir data/synthetic/

# Convert data if needed
python forensic/utils/convert_data_format.py --input data/raw/ --output data/processed/
```

#### Dashboard Access Issues
```bash
# Reset dashboard authentication
cd forensic/dashboards && python reset_auth.py

# Check dashboard configuration
python forensic/dashboards/validate_config.py
```

#### Performance Issues
```bash
# Monitor system resources
python forensic/utils/system_monitor.py --duration 300

# Optimize analysis parameters
python forensic/testing/optimize_analysis.py --reduce-samples --parallel-threads 4
```

## 📞 Expert Support

### Technical Support
- **Documentation**: Complete API and user guides
- **Examples**: Worked investigation examples
- **Troubleshooting**: Common issue resolution

### Legal Support
- **Expert Testimony**: Certified forensic expert testimony
- **Report Review**: Legal admissibility validation
- **Compliance Consulting**: Regulatory requirement guidance

### Training
- **Investigator Training**: Comprehensive forensic methodology
- **Technical Training**: Tool usage and analysis interpretation
- **Legal Training**: Evidence handling and court presentation

---

*This guide provides the foundation for professional forensic investigation of AI bias. Always consult with legal counsel for case-specific requirements and jurisdictional considerations.*