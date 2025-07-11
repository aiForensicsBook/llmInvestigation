"""
Help System component for dashboard documentation and guidance.
"""

import streamlit as st
from ..config import DashboardConfig
from ..auth import AuthenticationManager
from .login_page import audit_user_action


def render(config: DashboardConfig, auth_manager: AuthenticationManager):
    """Render help system page."""
    audit_user_action("view", "help_system")
    
    st.header("❓ Help & Documentation")
    
    # Help navigation
    help_tabs = st.tabs([
        "🏠 Getting Started", 
        "🔍 Investigation Guide", 
        "📊 Dashboard Guide",
        "⚖️ Bias Analysis Help",
        "🎯 SHAP Explanation",
        "🔧 Troubleshooting",
        "📞 Support"
    ])
    
    with help_tabs[0]:
        render_getting_started()
    
    with help_tabs[1]:
        render_investigation_guide()
    
    with help_tabs[2]:
        render_dashboard_guide()
    
    with help_tabs[3]:
        render_bias_analysis_help()
    
    with help_tabs[4]:
        render_shap_explanation()
    
    with help_tabs[5]:
        render_troubleshooting()
    
    with help_tabs[6]:
        render_support()


def render_getting_started():
    """Render getting started guide."""
    st.subheader("🏠 Getting Started")
    
    st.markdown("""
    ## Welcome to the Forensic Dashboard
    
    This dashboard is designed for comprehensive forensic investigation of AI bias in 
    machine learning models, specifically focused on resume screening systems.
    
    ### Quick Start Guide
    
    1. **Login**: Use your assigned credentials to access the system
    2. **Executive Summary**: Start with the high-level overview of findings
    3. **Technical Analysis**: Dive deeper into detailed investigations
    4. **Bias Analysis**: Examine fairness metrics and demographic impacts
    5. **Export**: Generate reports for legal proceedings
    
    ### User Roles
    
    - **Admin**: Full system access, user management, configuration
    - **Investigator**: Read and export access for evidence gathering
    - **Analyst**: Read and write access for detailed analysis
    
    ### Navigation
    
    Use the sidebar to navigate between different sections:
    - Each section requires specific permissions
    - All actions are logged for audit purposes
    - Real-time monitoring shows system status
    """)
    
    with st.expander("🎥 Video Tutorials"):
        st.markdown("""
        - **Dashboard Overview**: 5-minute introduction to the interface
        - **Bias Investigation**: Step-by-step guide to bias analysis
        - **Report Generation**: How to create legal-ready reports
        - **Evidence Collection**: Best practices for forensic evidence
        """)


def render_investigation_guide():
    """Render investigation methodology guide."""
    st.subheader("🔍 Investigation Methodology")
    
    st.markdown("""
    ## Forensic Investigation Process
    
    ### Phase 1: Evidence Collection
    
    1. **Data Gathering**
       - Collect model artifacts
       - Gather training data samples
       - Document model versions and configurations
       - Secure system logs and audit trails
    
    2. **Chain of Custody**
       - Document all data sources
       - Maintain integrity hashes
       - Record access and modifications
       - Ensure legal admissibility
    
    ### Phase 2: Technical Analysis
    
    1. **Model Inspection**
       - Analyze model architecture
       - Review training procedures
       - Examine feature engineering
       - Assess data preprocessing
    
    2. **Performance Evaluation**
       - Calculate accuracy metrics
       - Measure prediction consistency
       - Identify performance gaps
       - Document edge cases
    
    ### Phase 3: Bias Investigation
    
    1. **Demographic Analysis**
       - Test protected attributes
       - Measure group differences
       - Calculate fairness metrics
       - Identify disparate impact
    
    2. **Statistical Testing**
       - Perform significance tests
       - Calculate confidence intervals
       - Document effect sizes
       - Validate findings
    
    ### Phase 4: Documentation
    
    1. **Report Generation**
       - Executive summary
       - Technical findings
       - Legal implications
       - Recommendations
    
    2. **Evidence Package**
       - Comprehensive data export
       - Analysis notebooks
       - Visualization charts
       - Metadata documentation
    """)


def render_dashboard_guide():
    """Render dashboard user guide."""
    st.subheader("📊 Dashboard User Guide")
    
    st.markdown("""
    ## Dashboard Sections
    
    ### Executive Summary
    - High-level overview of investigation status
    - Key performance indicators
    - Risk assessment summary
    - Critical alerts and recommendations
    
    ### Technical Analysis
    - Detailed model investigation tools
    - Data quality assessment
    - Statistical analysis capabilities
    - Hypothesis testing framework
    
    ### Bias Analysis
    - Comprehensive fairness testing
    - Demographic group comparisons
    - Intersectional bias detection
    - Temporal bias tracking
    
    ### SHAP Analysis
    - Feature importance analysis
    - Individual prediction explanations
    - Model behavior understanding
    - Interaction effect detection
    
    ### Evidently Monitoring
    - Data drift detection
    - Model performance tracking
    - Target distribution monitoring
    - Automated report generation
    
    ### Real-time Monitoring
    - Live system status
    - Performance metrics streaming
    - Active alert management
    - Threshold monitoring
    
    ### Audit Trail
    - Complete action logging
    - User access tracking
    - Change documentation
    - Forensic timeline
    
    ### Export Manager
    - PDF report generation
    - Excel data exports
    - Evidence package creation
    - Raw data downloads
    """)
    
    with st.expander("🖱️ Interface Tips"):
        st.markdown("""
        - **Hover**: Mouse over charts for detailed tooltips
        - **Zoom**: Use mouse wheel or chart controls to zoom
        - **Filter**: Most tables support column filtering
        - **Export**: Individual charts can be exported via menu
        - **Refresh**: Use browser refresh or auto-refresh toggles
        """)


def render_bias_analysis_help():
    """Render bias analysis help."""
    st.subheader("⚖️ Bias Analysis Guide")
    
    st.markdown("""
    ## Bias Metrics Explained
    
    ### Demographic Parity
    **Definition**: Equal selection rates across demographic groups
    
    **Formula**: |P(Ŷ=1|A=0) - P(Ŷ=1|A=1)| ≤ ε
    
    **Interpretation**: 
    - Score < 0.05: Low bias risk
    - Score 0.05-0.1: Medium bias risk  
    - Score > 0.1: High bias risk
    
    ### Equalized Odds
    **Definition**: Equal true positive and false positive rates across groups
    
    **Formula**: TPR₀ = TPR₁ and FPR₀ = FPR₁
    
    **Use Case**: When both precision and recall matter equally
    
    ### Equal Opportunity
    **Definition**: Equal true positive rates across groups
    
    **Formula**: P(Ŷ=1|Y=1,A=0) = P(Ŷ=1|Y=1,A=1)
    
    **Use Case**: When avoiding false negatives is critical
    
    ### Statistical Parity
    **Definition**: Independence between predictions and protected attributes
    
    **Interpretation**: Model predictions should not correlate with protected attributes
    
    ## Investigation Steps
    
    1. **Select Metrics**: Choose appropriate fairness metrics for your use case
    2. **Set Thresholds**: Define acceptable bias levels (typically < 0.05)
    3. **Analyze Groups**: Compare performance across demographic groups
    4. **Test Significance**: Use statistical tests to validate findings
    5. **Document Results**: Record all findings with appropriate evidence
    
    ## Common Pitfalls
    
    - **Simpson's Paradox**: Aggregate results may hide subgroup bias
    - **Base Rate Differences**: Account for different group sizes
    - **Multiple Testing**: Adjust p-values for multiple comparisons
    - **Intersectionality**: Consider combinations of protected attributes
    """)


def render_shap_explanation():
    """Render SHAP explanation guide."""
    st.subheader("🎯 SHAP Analysis Guide")
    
    st.markdown("""
    ## SHAP Values Explained
    
    **SHAP (SHapley Additive exPlanations)** provides unified framework for 
    interpreting machine learning model predictions.
    
    ### Key Concepts
    
    **SHAP Value**: The contribution of each feature to a specific prediction
    - Positive values push prediction toward positive class
    - Negative values push prediction toward negative class
    - Sum of all SHAP values = (prediction - baseline)
    
    **Baseline**: The average prediction across all training data
    
    **Feature Importance**: Average absolute SHAP values across all predictions
    
    ### Visualization Types
    
    #### Feature Importance Plot
    - Shows global importance of each feature
    - Ranked by mean absolute SHAP value
    - Helps identify most influential features
    
    #### Waterfall Plot
    - Explains individual predictions
    - Shows how each feature contributes
    - Traces path from baseline to final prediction
    
    #### Dependency Plot
    - Shows relationship between feature value and SHAP value
    - Reveals non-linear patterns
    - Can show interaction effects with other features
    
    #### Summary Plot
    - Combines feature importance with value impact
    - Shows distribution of SHAP values
    - Color-coded by feature values
    
    ### Interpretation Guidelines
    
    1. **Global Analysis**: Start with feature importance to understand overall model behavior
    2. **Local Analysis**: Use waterfall plots to explain specific predictions
    3. **Pattern Detection**: Use dependency plots to find concerning patterns
    4. **Bias Detection**: Look for demographic features with high importance
    
    ### Bias Investigation with SHAP
    
    - **Direct Bias**: Demographic features with high SHAP importance
    - **Proxy Bias**: Non-demographic features that correlate with demographics
    - **Interaction Bias**: Features that behave differently across demographic groups
    """)


def render_troubleshooting():
    """Render troubleshooting guide."""
    st.subheader("🔧 Troubleshooting")
    
    st.markdown("""
    ## Common Issues and Solutions
    
    ### Login Problems
    
    **Issue**: Cannot log in with credentials
    - **Solution**: Check username/password case sensitivity
    - **Solution**: Contact administrator to reset password
    - **Solution**: Verify account is not locked due to failed attempts
    
    **Issue**: Session expires quickly
    - **Solution**: Check "Remember me" option during login
    - **Solution**: Verify system time settings
    - **Solution**: Contact admin to adjust session timeout
    
    ### Data Loading Issues
    
    **Issue**: "No evidence data available" message
    - **Solution**: Verify evidence files exist in forensic/evidence/
    - **Solution**: Check file permissions and formats
    - **Solution**: Run data collection scripts first
    
    **Issue**: Charts not displaying
    - **Solution**: Refresh the browser page
    - **Solution**: Check internet connection for external resources
    - **Solution**: Try different browser (Chrome recommended)
    
    ### Performance Issues
    
    **Issue**: Dashboard loading slowly
    - **Solution**: Reduce auto-refresh frequency
    - **Solution**: Limit data time ranges
    - **Solution**: Close unused browser tabs
    
    **Issue**: Export generation fails
    - **Solution**: Check available disk space
    - **Solution**: Verify export directory permissions
    - **Solution**: Try smaller data exports
    
    ### Analysis Problems
    
    **Issue**: Bias metrics showing unexpected values
    - **Solution**: Verify data quality and completeness
    - **Solution**: Check for missing values in protected attributes
    - **Solution**: Review data preprocessing steps
    
    **Issue**: SHAP analysis taking too long
    - **Solution**: Reduce sample size for analysis
    - **Solution**: Use background processing mode
    - **Solution**: Check system resources
    
    ## Error Codes
    
    - **AUTH_001**: Authentication failed
    - **DATA_002**: Data integrity check failed
    - **EXPORT_003**: Export generation error
    - **BIAS_004**: Bias calculation error
    - **SHAP_005**: SHAP analysis error
    
    ## Log Files
    
    Check these log files for detailed error information:
    - `logs/dashboard.log`: General application logs
    - `logs/errors.log`: Error-specific logs
    - `logs/audit.log`: User action logs
    - `logs/security.log`: Security events
    """)


def render_support():
    """Render support information."""
    st.subheader("📞 Support & Contact")
    
    st.markdown("""
    ## Getting Help
    
    ### Internal Support
    
    **System Administrator**
    - Email: admin@forensic-dashboard.local
    - Role: Technical issues, user management, system configuration
    - Response Time: 4 hours during business hours
    
    **Lead Investigator**
    - Email: lead.investigator@forensic-dashboard.local
    - Role: Investigation methodology, bias analysis interpretation
    - Response Time: 24 hours
    
    **Data Science Team**
    - Email: datascience@forensic-dashboard.local
    - Role: SHAP analysis, statistical questions, model interpretation
    - Response Time: 2 business days
    
    ### External Resources
    
    **Legal Support**
    - Contact your organization's legal counsel
    - For questions about evidence admissibility
    - Compliance and regulatory guidance
    
    **Technical Documentation**
    - User manual: Available in `docs/` directory
    - API documentation: Available at `/api/docs`
    - Technical specifications: Contact system administrator
    
    ## Reporting Issues
    
    When reporting issues, please include:
    
    1. **User Information**
       - Username and role
       - Browser and version
       - Operating system
    
    2. **Issue Details**
       - Exact error message
       - Steps to reproduce
       - Expected vs actual behavior
       - Screenshots if applicable
    
    3. **Context Information**
       - Time when issue occurred
       - Data being analyzed
       - Actions performed before issue
    
    ## Feature Requests
    
    To request new features:
    1. Submit detailed description via email
    2. Include business justification
    3. Specify urgency and priority
    4. Await development team assessment
    
    ## Training and Onboarding
    
    **New User Training**
    - Schedule: First Tuesday of each month
    - Duration: 2 hours
    - Topics: Basic navigation, investigation workflow
    - Registration: Contact administrator
    
    **Advanced Training**
    - Topics: Statistical analysis, bias interpretation, legal considerations
    - Format: On-demand webinars
    - Prerequisites: Basic training completion
    
    ## System Information
    
    **Current Version**: 1.0.0
    **Last Updated**: July 2024
    **Next Scheduled Maintenance**: First weekend of each month
    **Backup Schedule**: Daily at 2:00 AM
    """)
    
    # Emergency contact
    st.error("""
    🚨 **Emergency Contact**
    
    For critical security issues or system outages:
    - Phone: +1-555-FORENSIC (24/7 hotline)
    - Email: emergency@forensic-dashboard.local
    - Response Time: Immediate (within 30 minutes)
    """)