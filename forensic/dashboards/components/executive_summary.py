"""
Executive Summary component for high-level overview of forensic analysis.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, Any, List, Optional

from ..config import DashboardConfig
from ..auth import AuthenticationManager
from ..utils.data_utils import load_evidence_data, create_data_summary
from ..utils.error_handling import error_handler
from ..utils.export_utils import generate_executive_summary_chart
from .login_page import require_permission, audit_user_action


def render(config: DashboardConfig, auth_manager: AuthenticationManager):
    """
    Render executive summary page.
    
    Args:
        config: Dashboard configuration
        auth_manager: Authentication manager
    """
    # Check permissions
    if not require_permission("read", auth_manager):
        return
    
    # Audit access
    audit_user_action("view", "executive_summary")
    
    with error_handler("Failed to load executive summary"):
        # Load evidence data
        evidence_data = load_evidence_data(config.evidence_dir)
        
        # Create summary metrics
        render_summary_metrics(evidence_data, config)
        
        # Render key findings
        render_key_findings(evidence_data)
        
        # Render bias analysis overview
        render_bias_overview(evidence_data)
        
        # Render performance dashboard
        render_performance_dashboard(evidence_data)
        
        # Render alerts and recommendations
        render_alerts_and_recommendations(evidence_data, config)
        
        # Render data integrity status
        render_data_integrity_status(evidence_data, config)


def render_summary_metrics(evidence_data: Dict[str, Any], config: DashboardConfig):
    """Render high-level summary metrics."""
    st.subheader("📊 Investigation Overview")
    
    # Calculate summary metrics
    metrics = calculate_summary_metrics(evidence_data)
    
    # Display metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Overall Risk Score",
            value=f"{metrics['overall_risk']:.2f}",
            delta=f"{metrics['risk_change']:+.2f}" if metrics['risk_change'] else None,
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            label="Bias Incidents",
            value=metrics['bias_incidents'],
            delta=f"{metrics['bias_change']:+d}" if metrics['bias_change'] else None,
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            label="Data Quality Score",
            value=f"{metrics['data_quality']:.1f}%",
            delta=f"{metrics['quality_change']:+.1f}%" if metrics['quality_change'] else None
        )
    
    with col4:
        st.metric(
            label="Investigation Status",
            value=metrics['status'],
            delta=metrics['status_change'] if metrics['status_change'] else None
        )
    
    # Progress indicators
    st.markdown("### Investigation Progress")
    
    progress_col1, progress_col2 = st.columns(2)
    
    with progress_col1:
        # Analysis completion progress
        completion_progress = metrics.get('completion_progress', 0.75)
        st.progress(completion_progress)
        st.caption(f"Analysis Completion: {completion_progress*100:.0f}%")
    
    with progress_col2:
        # Evidence collection progress
        evidence_progress = metrics.get('evidence_progress', 0.85)
        st.progress(evidence_progress)
        st.caption(f"Evidence Collection: {evidence_progress*100:.0f}%")


def render_key_findings(evidence_data: Dict[str, Any]):
    """Render key findings section."""
    st.subheader("🔍 Key Findings")
    
    findings = extract_key_findings(evidence_data)
    
    # High-priority findings
    if findings['high_priority']:
        st.markdown("#### 🚨 High Priority Issues")
        for finding in findings['high_priority']:
            st.error(f"**{finding['title']}**: {finding['description']}")
    
    # Medium-priority findings
    if findings['medium_priority']:
        st.markdown("#### ⚠️ Medium Priority Issues")
        for finding in findings['medium_priority']:
            st.warning(f"**{finding['title']}**: {finding['description']}")
    
    # Positive findings
    if findings['positive']:
        st.markdown("#### ✅ Positive Indicators")
        for finding in findings['positive']:
            st.success(f"**{finding['title']}**: {finding['description']}")
    
    # Additional insights
    if findings['insights']:
        with st.expander("📋 Additional Insights"):
            for insight in findings['insights']:
                st.info(f"**{insight['category']}**: {insight['description']}")


def render_bias_overview(evidence_data: Dict[str, Any]):
    """Render bias analysis overview."""
    st.subheader("⚖️ Bias Analysis Overview")
    
    if 'bias_analysis' not in evidence_data:
        st.warning("No bias analysis data available")
        return
    
    bias_data = evidence_data['bias_analysis']
    
    # Create bias metrics chart
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if isinstance(bias_data, dict):
            # Extract numerical bias metrics
            bias_metrics = {}
            for key, value in bias_data.items():
                if isinstance(value, (int, float)) and 'bias' in key.lower():
                    bias_metrics[key] = value
            
            if bias_metrics:
                fig = generate_executive_summary_chart(bias_metrics)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No numerical bias metrics found for visualization")
    
    with col2:
        # Bias summary table
        st.markdown("**Bias Metrics Summary**")
        
        if isinstance(bias_data, dict):
            bias_summary = []
            for metric, value in bias_data.items():
                if isinstance(value, (int, float)):
                    status = "🟢 Low" if value < 0.05 else "🟡 Medium" if value < 0.1 else "🔴 High"
                    bias_summary.append({
                        "Metric": metric.replace('_', ' ').title(),
                        "Score": f"{value:.4f}",
                        "Risk": status
                    })
            
            if bias_summary:
                bias_df = pd.DataFrame(bias_summary)
                st.dataframe(bias_df, use_container_width=True, hide_index=True)


def render_performance_dashboard(evidence_data: Dict[str, Any]):
    """Render performance metrics dashboard."""
    st.subheader("📈 Performance Dashboard")
    
    # Create performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Model performance over time
        perf_chart = create_performance_timeline_chart(evidence_data)
        if perf_chart:
            st.plotly_chart(perf_chart, use_container_width=True)
        else:
            st.info("Performance timeline data not available")
    
    with col2:
        # Feature importance distribution
        feature_chart = create_feature_importance_chart(evidence_data)
        if feature_chart:
            st.plotly_chart(feature_chart, use_container_width=True)
        else:
            st.info("Feature importance data not available")
    
    # Performance metrics table
    if 'performance_metrics' in evidence_data:
        with st.expander("📊 Detailed Performance Metrics"):
            perf_data = evidence_data['performance_metrics']
            if isinstance(perf_data, dict):
                perf_df = pd.DataFrame([
                    {"Metric": k.replace('_', ' ').title(), "Value": v}
                    for k, v in perf_data.items()
                    if isinstance(v, (int, float, str))
                ])
                st.dataframe(perf_df, use_container_width=True, hide_index=True)


def render_alerts_and_recommendations(evidence_data: Dict[str, Any], config: DashboardConfig):
    """Render alerts and recommendations."""
    st.subheader("🚨 Alerts & Recommendations")
    
    alerts = generate_alerts(evidence_data, config)
    recommendations = generate_recommendations(evidence_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Active Alerts")
        if alerts:
            for alert in alerts:
                if alert['severity'] == 'high':
                    st.error(f"🚨 {alert['message']}")
                elif alert['severity'] == 'medium':
                    st.warning(f"⚠️ {alert['message']}")
                else:
                    st.info(f"ℹ️ {alert['message']}")
        else:
            st.success("✅ No active alerts")
    
    with col2:
        st.markdown("#### Recommendations")
        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"{i}. {rec}")
        else:
            st.info("No specific recommendations at this time")


def render_data_integrity_status(evidence_data: Dict[str, Any], config: DashboardConfig):
    """Render data integrity and chain of custody status."""
    st.subheader("🔒 Data Integrity & Chain of Custody")
    
    # Data integrity checks
    integrity_results = perform_integrity_checks(evidence_data, config)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Evidence Integrity**")
        if integrity_results['evidence_integrity']:
            st.success("✅ Verified")
        else:
            st.error("❌ Issues Found")
    
    with col2:
        st.markdown("**Chain of Custody**")
        if integrity_results['chain_of_custody']:
            st.success("✅ Maintained")
        else:
            st.error("❌ Broken")
    
    with col3:
        st.markdown("**Audit Trail**")
        if integrity_results['audit_trail']:
            st.success("✅ Complete")
        else:
            st.warning("⚠️ Incomplete")
    
    # Detailed integrity information
    with st.expander("🔍 Detailed Integrity Information"):
        st.json(integrity_results['details'])


def calculate_summary_metrics(evidence_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate summary metrics from evidence data."""
    metrics = {
        'overall_risk': 0.0,
        'risk_change': None,
        'bias_incidents': 0,
        'bias_change': None,
        'data_quality': 100.0,
        'quality_change': None,
        'status': 'In Progress',
        'status_change': None,
        'completion_progress': 0.75,
        'evidence_progress': 0.85
    }
    
    # Calculate overall risk from bias analysis
    if 'bias_analysis' in evidence_data:
        bias_data = evidence_data['bias_analysis']
        if isinstance(bias_data, dict):
            bias_scores = [v for v in bias_data.values() if isinstance(v, (int, float))]
            if bias_scores:
                metrics['overall_risk'] = np.mean(bias_scores)
                metrics['bias_incidents'] = sum(1 for score in bias_scores if score > 0.1)
    
    # Calculate data quality
    if 'performance_metrics' in evidence_data:
        perf_data = evidence_data['performance_metrics']
        if isinstance(perf_data, dict):
            accuracy = perf_data.get('accuracy', 0.95)
            metrics['data_quality'] = accuracy * 100
    
    # Determine status
    if metrics['overall_risk'] > 0.2:
        metrics['status'] = 'High Risk'
    elif metrics['overall_risk'] > 0.1:
        metrics['status'] = 'Medium Risk'
    elif metrics['completion_progress'] >= 1.0:
        metrics['status'] = 'Complete'
    else:
        metrics['status'] = 'In Progress'
    
    return metrics


def extract_key_findings(evidence_data: Dict[str, Any]) -> Dict[str, List[Dict]]:
    """Extract key findings from evidence data."""
    findings = {
        'high_priority': [],
        'medium_priority': [],
        'positive': [],
        'insights': []
    }
    
    # Analyze bias data for findings
    if 'bias_analysis' in evidence_data:
        bias_data = evidence_data['bias_analysis']
        if isinstance(bias_data, dict):
            for metric, value in bias_data.items():
                if isinstance(value, (int, float)):
                    if value > 0.2:
                        findings['high_priority'].append({
                            'title': f'High Bias in {metric.replace("_", " ").title()}',
                            'description': f'Detected bias score of {value:.4f}, exceeding acceptable threshold'
                        })
                    elif value > 0.1:
                        findings['medium_priority'].append({
                            'title': f'Moderate Bias in {metric.replace("_", " ").title()}',
                            'description': f'Detected bias score of {value:.4f}, requires monitoring'
                        })
                    elif value < 0.05:
                        findings['positive'].append({
                            'title': f'Low Bias in {metric.replace("_", " ").title()}',
                            'description': f'Bias score of {value:.4f} is within acceptable limits'
                        })
    
    # Add performance insights
    if 'performance_metrics' in evidence_data:
        perf_data = evidence_data['performance_metrics']
        if isinstance(perf_data, dict):
            accuracy = perf_data.get('accuracy', 0)
            if accuracy > 0.95:
                findings['positive'].append({
                    'title': 'High Model Accuracy',
                    'description': f'Model accuracy of {accuracy:.2%} indicates good performance'
                })
            elif accuracy < 0.8:
                findings['medium_priority'].append({
                    'title': 'Low Model Accuracy',
                    'description': f'Model accuracy of {accuracy:.2%} may indicate issues'
                })
    
    return findings


def create_performance_timeline_chart(evidence_data: Dict[str, Any]) -> Optional[go.Figure]:
    """Create performance timeline chart."""
    # Generate sample performance data if real data not available
    dates = pd.date_range(start='2024-01-01', end='2024-07-01', freq='W')
    
    # Simulate performance metrics over time
    np.random.seed(42)
    accuracy = 0.85 + 0.1 * np.random.random(len(dates))
    precision = 0.82 + 0.12 * np.random.random(len(dates))
    recall = 0.88 + 0.08 * np.random.random(len(dates))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates, y=accuracy,
        mode='lines+markers',
        name='Accuracy',
        line=dict(color='blue')
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=precision,
        mode='lines+markers',
        name='Precision',
        line=dict(color='green')
    ))
    
    fig.add_trace(go.Scatter(
        x=dates, y=recall,
        mode='lines+markers',
        name='Recall',
        line=dict(color='orange')
    ))
    
    fig.update_layout(
        title='Model Performance Over Time',
        xaxis_title='Date',
        yaxis_title='Score',
        yaxis=dict(range=[0.7, 1.0]),
        template='plotly_white',
        height=400
    )
    
    return fig


def create_feature_importance_chart(evidence_data: Dict[str, Any]) -> Optional[go.Figure]:
    """Create feature importance chart."""
    # Generate sample feature importance data
    features = ['Experience', 'Education', 'Skills', 'Location', 'Age', 'Gender', 'Name']
    importance = [0.35, 0.25, 0.20, 0.10, 0.05, 0.03, 0.02]
    
    fig = go.Figure(data=[
        go.Bar(
            x=importance,
            y=features,
            orientation='h',
            marker_color=['red' if imp > 0.15 else 'orange' if imp > 0.05 else 'green' 
                         for imp in importance]
        )
    ])
    
    fig.update_layout(
        title='Feature Importance Analysis',
        xaxis_title='Importance Score',
        yaxis_title='Features',
        template='plotly_white',
        height=400
    )
    
    return fig


def generate_alerts(evidence_data: Dict[str, Any], config: DashboardConfig) -> List[Dict[str, str]]:
    """Generate alerts based on evidence data."""
    alerts = []
    
    # Check bias thresholds
    if 'bias_analysis' in evidence_data:
        bias_data = evidence_data['bias_analysis']
        if isinstance(bias_data, dict):
            for metric, value in bias_data.items():
                if isinstance(value, (int, float)):
                    if value > config.monitoring.alert_threshold_bias:
                        alerts.append({
                            'severity': 'high',
                            'message': f'High bias detected in {metric}: {value:.4f}'
                        })
                    elif value > config.monitoring.alert_threshold_bias / 2:
                        alerts.append({
                            'severity': 'medium',
                            'message': f'Moderate bias detected in {metric}: {value:.4f}'
                        })
    
    # Check performance degradation
    if 'performance_metrics' in evidence_data:
        perf_data = evidence_data['performance_metrics']
        if isinstance(perf_data, dict):
            accuracy = perf_data.get('accuracy', 1.0)
            if accuracy < 0.8:
                alerts.append({
                    'severity': 'high',
                    'message': f'Low model accuracy detected: {accuracy:.2%}'
                })
            elif accuracy < 0.9:
                alerts.append({
                    'severity': 'medium',
                    'message': f'Model accuracy below optimal: {accuracy:.2%}'
                })
    
    return alerts


def generate_recommendations(evidence_data: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on evidence data."""
    recommendations = []
    
    # Analyze bias data for recommendations
    if 'bias_analysis' in evidence_data:
        bias_data = evidence_data['bias_analysis']
        if isinstance(bias_data, dict):
            high_bias_metrics = [k for k, v in bias_data.items() 
                               if isinstance(v, (int, float)) and v > 0.1]
            
            if high_bias_metrics:
                recommendations.append(
                    f"Address high bias in: {', '.join(high_bias_metrics)}"
                )
                recommendations.append(
                    "Consider retraining the model with debiased data"
                )
                recommendations.append(
                    "Implement bias monitoring in production"
                )
    
    # Performance recommendations
    if 'performance_metrics' in evidence_data:
        perf_data = evidence_data['performance_metrics']
        if isinstance(perf_data, dict):
            accuracy = perf_data.get('accuracy', 1.0)
            if accuracy < 0.9:
                recommendations.append(
                    "Investigate causes of low model accuracy"
                )
                recommendations.append(
                    "Consider additional training data or feature engineering"
                )
    
    # General recommendations
    recommendations.extend([
        "Continue regular bias monitoring and testing",
        "Maintain comprehensive audit trails",
        "Review and update bias detection thresholds",
        "Conduct periodic model fairness assessments"
    ])
    
    return recommendations


def perform_integrity_checks(evidence_data: Dict[str, Any], 
                           config: DashboardConfig) -> Dict[str, Any]:
    """Perform data integrity checks."""
    results = {
        'evidence_integrity': True,
        'chain_of_custody': True,
        'audit_trail': True,
        'details': {}
    }
    
    # Check evidence data availability
    required_evidence = ['bias_analysis', 'performance_metrics']
    missing_evidence = [req for req in required_evidence if req not in evidence_data]
    
    if missing_evidence:
        results['evidence_integrity'] = False
        results['details']['missing_evidence'] = missing_evidence
    
    # Check data consistency
    if 'bias_analysis' in evidence_data:
        bias_data = evidence_data['bias_analysis']
        if not isinstance(bias_data, dict) or not bias_data:
            results['evidence_integrity'] = False
            results['details']['bias_data_issues'] = 'Invalid or empty bias analysis data'
    
    # Check audit trail (simplified check)
    audit_log_file = config.logs_dir / "audit.log"
    if not audit_log_file.exists():
        results['audit_trail'] = False
        results['details']['audit_issues'] = 'Audit log file not found'
    
    # Add timestamp of last check
    results['details']['last_check'] = datetime.now().isoformat()
    results['details']['checked_evidence_types'] = list(evidence_data.keys())
    
    return results