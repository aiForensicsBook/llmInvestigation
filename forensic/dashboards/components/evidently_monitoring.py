"""
Evidently AI Monitoring component for data and model drift detection.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

from ..config import DashboardConfig
from ..auth import AuthenticationManager
from ..utils.data_utils import load_evidence_data
from ..utils.error_handling import error_handler
from .login_page import require_permission, audit_user_action


def render(config: DashboardConfig, auth_manager: AuthenticationManager):
    """Render Evidently monitoring page."""
    if not require_permission("read", auth_manager):
        return
    
    audit_user_action("view", "evidently_monitoring")
    
    with error_handler("Failed to load Evidently monitoring"):
        st.header("📈 Evidently AI Monitoring")
        
        tabs = st.tabs(["🔄 Data Drift", "📊 Model Performance", "🎯 Target Drift", "📋 Reports"])
        
        with tabs[0]:
            render_data_drift()
        
        with tabs[1]:
            render_model_performance()
        
        with tabs[2]:
            render_target_drift()
        
        with tabs[3]:
            render_reports()


def render_data_drift():
    """Render data drift analysis."""
    st.subheader("Data Drift Detection")
    
    # Generate sample drift data
    features = ['experience_years', 'education_level', 'skills_match', 'location_preference', 
               'previous_salary', 'age', 'resume_quality']
    
    drift_scores = np.random.uniform(0, 0.3, len(features))
    drift_detected = drift_scores > 0.1
    
    # Drift summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Features Monitored", len(features))
    with col2:
        st.metric("Drift Detected", sum(drift_detected))
    with col3:
        avg_drift = np.mean(drift_scores)
        st.metric("Average Drift Score", f"{avg_drift:.3f}")
    
    # Drift scores chart
    colors = ['red' if detected else 'green' for detected in drift_detected]
    
    fig = go.Figure(data=[
        go.Bar(
            x=features,
            y=drift_scores,
            marker_color=colors,
            text=[f"{score:.3f}" for score in drift_scores],
            textposition='auto'
        )
    ])
    
    fig.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Drift Threshold")
    fig.update_layout(
        title="Feature Drift Scores",
        xaxis_title="Features",
        yaxis_title="Drift Score",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Drift details table
    drift_df = pd.DataFrame({
        'Feature': features,
        'Drift Score': drift_scores,
        'Status': ['🔴 Drift Detected' if d else '✅ No Drift' for d in drift_detected],
        'Method': ['Kolmogorov-Smirnov' if np.random.random() > 0.5 else 'Wasserstein' for _ in features]
    })
    
    st.dataframe(drift_df, use_container_width=True, hide_index=True)
    
    # Feature distribution comparison
    st.subheader("Feature Distribution Comparison")
    
    selected_feature = st.selectbox("Select feature for distribution analysis:", features)
    
    # Generate sample distributions
    np.random.seed(hash(selected_feature) % 1000)
    reference_data = np.random.normal(5, 2, 1000)
    current_data = np.random.normal(5.5, 2.2, 1000)  # Slightly shifted
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=reference_data,
        name='Reference',
        opacity=0.7,
        nbinsx=30
    ))
    
    fig.add_trace(go.Histogram(
        x=current_data,
        name='Current',
        opacity=0.7,
        nbinsx=30
    ))
    
    fig.update_layout(
        title=f"Distribution Comparison: {selected_feature}",
        xaxis_title="Value",
        yaxis_title="Frequency",
        barmode='overlay',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_model_performance():
    """Render model performance monitoring."""
    st.subheader("Model Performance Monitoring")
    
    # Performance metrics over time
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    
    # Generate sample performance data
    np.random.seed(42)
    accuracy = 0.85 + 0.05 * np.sin(np.arange(len(dates)) * 0.2) + np.random.normal(0, 0.02, len(dates))
    precision = 0.82 + 0.03 * np.cos(np.arange(len(dates)) * 0.15) + np.random.normal(0, 0.015, len(dates))
    recall = 0.88 + 0.04 * np.sin(np.arange(len(dates)) * 0.25) + np.random.normal(0, 0.018, len(dates))
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    # Performance trends
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=dates, y=accuracy, mode='lines+markers', name='Accuracy'))
    fig.add_trace(go.Scatter(x=dates, y=precision, mode='lines+markers', name='Precision'))
    fig.add_trace(go.Scatter(x=dates, y=recall, mode='lines+markers', name='Recall'))
    fig.add_trace(go.Scatter(x=dates, y=f1_score, mode='lines+markers', name='F1 Score'))
    
    fig.update_layout(
        title="Model Performance Over Time",
        xaxis_title="Date",
        yaxis_title="Score",
        height=400,
        yaxis=dict(range=[0.7, 1.0])
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Current metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{accuracy[-1]:.3f}", f"{accuracy[-1] - accuracy[-7]:.3f}")
    with col2:
        st.metric("Precision", f"{precision[-1]:.3f}", f"{precision[-1] - precision[-7]:.3f}")
    with col3:
        st.metric("Recall", f"{recall[-1]:.3f}", f"{recall[-1] - recall[-7]:.3f}")
    with col4:
        st.metric("F1 Score", f"{f1_score[-1]:.3f}", f"{f1_score[-1] - f1_score[-7]:.3f}")
    
    # Performance degradation alerts
    st.subheader("Performance Alerts")
    
    alerts = []
    if accuracy[-1] < 0.8:
        alerts.append("🔴 Accuracy below threshold (0.8)")
    if precision[-1] < 0.8:
        alerts.append("🔴 Precision below threshold (0.8)")
    if recall[-1] < 0.8:
        alerts.append("🔴 Recall below threshold (0.8)")
    
    if alerts:
        for alert in alerts:
            st.error(alert)
    else:
        st.success("✅ All performance metrics within acceptable range")


def render_target_drift():
    """Render target drift analysis."""
    st.subheader("Target Drift Analysis")
    
    # Generate sample target distribution data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    
    # Simulate changing target distribution
    np.random.seed(42)
    acceptance_rates = 0.7 + 0.1 * np.sin(np.arange(len(dates)) * 0.3) + np.random.normal(0, 0.03, len(dates))
    acceptance_rates = np.clip(acceptance_rates, 0, 1)
    
    # Target drift over time
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=acceptance_rates,
        mode='lines+markers',
        name='Acceptance Rate',
        line=dict(color='blue')
    ))
    
    fig.add_hline(y=0.7, line_dash="dash", line_color="green", annotation_text="Expected Rate")
    fig.add_hline(y=0.6, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
    
    fig.update_layout(
        title="Target Distribution Over Time",
        xaxis_title="Date",
        yaxis_title="Acceptance Rate",
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Target statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Rate", f"{acceptance_rates[-1]:.3f}")
    with col2:
        st.metric("7-Day Average", f"{np.mean(acceptance_rates[-7:]):.3f}")
    with col3:
        drift_score = abs(acceptance_rates[-1] - 0.7)
        st.metric("Drift Score", f"{drift_score:.3f}")
    
    # Demographic target analysis
    st.subheader("Target Distribution by Demographics")
    
    demographics = ['Gender', 'Age Group', 'Ethnicity']
    demo_data = {
        'Gender': {'Male': 0.72, 'Female': 0.68, 'Non-binary': 0.70},
        'Age Group': {'18-30': 0.75, '31-45': 0.72, '46-60': 0.65, '60+': 0.58},
        'Ethnicity': {'White': 0.71, 'Black': 0.65, 'Hispanic': 0.68, 'Asian': 0.76, 'Other': 0.69}
    }
    
    selected_demo = st.selectbox("Select demographic:", demographics)
    demo_rates = demo_data[selected_demo]
    
    fig = px.bar(
        x=list(demo_rates.keys()),
        y=list(demo_rates.values()),
        title=f"Acceptance Rates by {selected_demo}",
        color=list(demo_rates.values()),
        color_continuous_scale='RdYlGn'
    )
    
    fig.add_hline(y=0.7, line_dash="dash", line_color="blue", annotation_text="Overall Rate")
    
    st.plotly_chart(fig, use_container_width=True)


def render_reports():
    """Render Evidently reports."""
    st.subheader("Evidently Reports")
    
    # Report summary
    reports = [
        {
            'Report': 'Data Drift Report',
            'Date': '2024-07-01',
            'Status': '🟡 Drift Detected',
            'Features Affected': 3,
            'Severity': 'Medium'
        },
        {
            'Report': 'Model Performance Report',
            'Date': '2024-07-01',
            'Status': '✅ Stable',
            'Features Affected': 0,
            'Severity': 'Low'
        },
        {
            'Report': 'Target Drift Report',
            'Date': '2024-07-01',
            'Status': '🔴 Significant Drift',
            'Features Affected': 1,
            'Severity': 'High'
        }
    ]
    
    reports_df = pd.DataFrame(reports)
    st.dataframe(reports_df, use_container_width=True, hide_index=True)
    
    # Detailed report view
    selected_report = st.selectbox("Select report for details:", [r['Report'] for r in reports])
    
    if selected_report:
        st.subheader(f"Detailed View: {selected_report}")
        
        # Mock detailed report content
        if "Data Drift" in selected_report:
            st.markdown("""
            **Summary**: Data drift detected in 3 out of 7 monitored features.
            
            **Affected Features**:
            - experience_years: Drift score 0.156
            - education_level: Drift score 0.134
            - age: Drift score 0.112
            
            **Recommendations**:
            - Investigate data collection process
            - Consider model retraining
            - Update feature preprocessing
            """)
        elif "Performance" in selected_report:
            st.markdown("""
            **Summary**: Model performance remains stable across all metrics.
            
            **Current Metrics**:
            - Accuracy: 0.847
            - Precision: 0.823
            - Recall: 0.881
            - F1-Score: 0.851
            
            **Recommendations**:
            - Continue monitoring
            - Maintain current model version
            """)
        else:  # Target Drift
            st.markdown("""
            **Summary**: Significant shift in target distribution detected.
            
            **Key Changes**:
            - Overall acceptance rate decreased by 8.2%
            - Demographic disparities increased
            - Temporal patterns shifted
            
            **Recommendations**:
            - Investigate business process changes
            - Review selection criteria
            - Consider bias impact assessment
            """)
        
        # Report download
        if st.button(f"Download {selected_report}"):
            st.success(f"Report downloaded: {selected_report.lower().replace(' ', '_')}_2024-07-01.html")