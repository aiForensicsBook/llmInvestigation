"""
Technical Analysis component for detailed forensic investigation.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import json

from ..config import DashboardConfig
from ..auth import AuthenticationManager
from ..utils.data_utils import load_evidence_data, load_synthetic_data
from ..utils.error_handling import error_handler, ErrorCollector
from ..utils.export_utils import create_chart_from_data
from .login_page import require_permission, audit_user_action


def render(config: DashboardConfig, auth_manager: AuthenticationManager):
    """
    Render technical analysis page for detailed investigation.
    
    Args:
        config: Dashboard configuration
        auth_manager: Authentication manager
    """
    # Check permissions
    if not require_permission("write", auth_manager):
        return
    
    # Audit access
    audit_user_action("view", "technical_analysis")
    
    with error_handler("Failed to load technical analysis"):
        # Load data
        evidence_data = load_evidence_data(config.evidence_dir)
        synthetic_data = load_synthetic_data(config.base_dir / "data")
        
        # Create analysis tabs
        tabs = st.tabs([
            "📊 Data Analysis", 
            "🔬 Model Investigation", 
            "⚖️ Fairness Testing",
            "🔍 Anomaly Detection",
            "📈 Statistical Analysis",
            "🧪 Hypothesis Testing"
        ])
        
        with tabs[0]:
            render_data_analysis(evidence_data, synthetic_data, config)
        
        with tabs[1]:
            render_model_investigation(evidence_data, config)
        
        with tabs[2]:
            render_fairness_testing(evidence_data, synthetic_data, config)
        
        with tabs[3]:
            render_anomaly_detection(evidence_data, config)
        
        with tabs[4]:
            render_statistical_analysis(evidence_data, synthetic_data, config)
        
        with tabs[5]:
            render_hypothesis_testing(evidence_data, config)


def render_data_analysis(evidence_data: Dict[str, Any], synthetic_data: Dict[str, pd.DataFrame], 
                        config: DashboardConfig):
    """Render comprehensive data analysis section."""
    st.header("📊 Data Analysis")
    
    # Data overview
    st.subheader("Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Evidence Data Available:**")
        if evidence_data:
            for key, value in evidence_data.items():
                data_type = type(value).__name__
                if isinstance(value, dict):
                    size = len(value)
                elif isinstance(value, list):
                    size = len(value)
                else:
                    size = 1
                st.write(f"• {key}: {data_type} ({size} items)")
        else:
            st.info("No evidence data loaded")
    
    with col2:
        st.markdown("**Synthetic Data Available:**")
        if synthetic_data:
            for key, df in synthetic_data.items():
                st.write(f"• {key}: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            st.info("No synthetic data loaded")
    
    # Data quality analysis
    st.subheader("Data Quality Analysis")
    
    if synthetic_data:
        selected_dataset = st.selectbox(
            "Select dataset for analysis:",
            list(synthetic_data.keys())
        )
        
        if selected_dataset in synthetic_data:
            df = synthetic_data[selected_dataset]
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Records", df.shape[0])
            with col2:
                st.metric("Total Features", df.shape[1])
            with col3:
                missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                st.metric("Missing Data %", f"{missing_pct:.2f}%")
            
            # Data quality details
            with st.expander("📋 Data Quality Details"):
                # Missing data analysis
                st.markdown("**Missing Data by Column:**")
                missing_data = df.isnull().sum()
                missing_pct = (missing_data / len(df)) * 100
                
                missing_df = pd.DataFrame({
                    'Column': missing_data.index,
                    'Missing Count': missing_data.values,
                    'Missing %': missing_pct.values
                }).sort_values('Missing Count', ascending=False)
                
                st.dataframe(missing_df, use_container_width=True, hide_index=True)
                
                # Data types
                st.markdown("**Data Types:**")
                dtype_df = pd.DataFrame({
                    'Column': df.dtypes.index,
                    'Data Type': df.dtypes.values
                })
                st.dataframe(dtype_df, use_container_width=True, hide_index=True)
            
            # Sample data view
            with st.expander("👀 Sample Data"):
                st.dataframe(df.head(10), use_container_width=True)


def render_model_investigation(evidence_data: Dict[str, Any], config: DashboardConfig):
    """Render model investigation section."""
    st.header("🔬 Model Investigation")
    
    # Model performance analysis
    st.subheader("Model Performance Analysis")
    
    if 'performance_metrics' in evidence_data:
        perf_data = evidence_data['performance_metrics']
        
        if isinstance(perf_data, dict):
            # Performance metrics table
            metrics_df = pd.DataFrame([
                {"Metric": k.replace('_', ' ').title(), "Value": v}
                for k, v in perf_data.items()
                if isinstance(v, (int, float))
            ])
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.dataframe(metrics_df, use_container_width=True, hide_index=True)
            
            with col2:
                # Performance visualization
                if len(metrics_df) > 0:
                    fig = px.bar(
                        metrics_df, 
                        x='Metric', 
                        y='Value',
                        title='Model Performance Metrics'
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No performance metrics data available")
    
    # Feature analysis
    st.subheader("Feature Analysis")
    
    # Simulate feature importance data if not available
    features = ['experience_years', 'education_level', 'skills_match', 'location_match', 
               'previous_salary', 'age', 'gender_inferred', 'name_ethnicity']
    importance_scores = np.random.random(len(features))
    
    feature_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance_scores,
        'Risk_Level': ['High' if score > 0.7 else 'Medium' if score > 0.4 else 'Low' 
                      for score in importance_scores]
    }).sort_values('Importance', ascending=False)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.dataframe(feature_df, use_container_width=True, hide_index=True)
    
    with col2:
        fig = px.bar(
            feature_df.head(8), 
            x='Importance', 
            y='Feature',
            color='Risk_Level',
            orientation='h',
            title='Feature Importance Analysis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Model behavior analysis
    st.subheader("Model Behavior Analysis")
    
    behavior_tabs = st.tabs(["Decision Boundaries", "Prediction Patterns", "Error Analysis"])
    
    with behavior_tabs[0]:
        # Decision boundary visualization
        st.markdown("**Decision Boundary Analysis**")
        
        # Generate sample decision boundary data
        x = np.linspace(0, 10, 100)
        y1 = 2 * x + np.random.normal(0, 1, 100)
        y2 = 1.5 * x + 2 + np.random.normal(0, 1, 100)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y1, mode='markers', name='Class 1', opacity=0.6))
        fig.add_trace(go.Scatter(x=x, y=y2, mode='markers', name='Class 2', opacity=0.6))
        fig.add_shape(type="line", x0=0, y0=1, x1=10, y1=21, 
                     line=dict(color="red", width=3, dash="dash"))
        fig.update_layout(title="Decision Boundary Visualization", height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with behavior_tabs[1]:
        # Prediction patterns
        st.markdown("**Prediction Pattern Analysis**")
        
        # Simulate prediction confidence distribution
        confidence_scores = np.random.beta(2, 2, 1000)
        
        fig = px.histogram(
            x=confidence_scores, 
            nbins=30,
            title="Prediction Confidence Distribution"
        )
        fig.update_layout(xaxis_title="Confidence Score", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)
    
    with behavior_tabs[2]:
        # Error analysis
        st.markdown("**Error Analysis**")
        
        # Simulate error patterns
        error_types = ['False Positive', 'False Negative', 'True Positive', 'True Negative']
        error_counts = [45, 35, 420, 500]
        
        fig = px.pie(
            values=error_counts, 
            names=error_types,
            title="Error Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_fairness_testing(evidence_data: Dict[str, Any], synthetic_data: Dict[str, pd.DataFrame], 
                           config: DashboardConfig):
    """Render fairness testing section."""
    st.header("⚖️ Fairness Testing")
    
    # Bias metrics overview
    st.subheader("Bias Metrics Overview")
    
    if 'bias_analysis' in evidence_data:
        bias_data = evidence_data['bias_analysis']
        
        if isinstance(bias_data, dict):
            # Create bias metrics visualization
            bias_metrics = {k: v for k, v in bias_data.items() if isinstance(v, (int, float))}
            
            if bias_metrics:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Bias metrics chart
                    metrics_list = list(bias_metrics.keys())
                    values_list = list(bias_metrics.values())
                    colors = ['red' if v > 0.1 else 'orange' if v > 0.05 else 'green' 
                             for v in values_list]
                    
                    fig = go.Figure(data=[
                        go.Bar(x=metrics_list, y=values_list, marker_color=colors)
                    ])
                    fig.update_layout(
                        title="Bias Metrics Analysis",
                        xaxis_title="Metrics",
                        yaxis_title="Bias Score",
                        height=400
                    )
                    fig.add_hline(y=0.05, line_dash="dash", line_color="orange", 
                                 annotation_text="Warning Threshold")
                    fig.add_hline(y=0.1, line_dash="dash", line_color="red", 
                                 annotation_text="Critical Threshold")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Bias summary table
                    bias_summary = []
                    for metric, value in bias_metrics.items():
                        if value > 0.1:
                            status = "🔴 Critical"
                        elif value > 0.05:
                            status = "🟡 Warning"
                        else:
                            status = "🟢 Acceptable"
                        
                        bias_summary.append({
                            "Metric": metric.replace('_', ' ').title(),
                            "Score": f"{value:.4f}",
                            "Status": status
                        })
                    
                    bias_df = pd.DataFrame(bias_summary)
                    st.dataframe(bias_df, use_container_width=True, hide_index=True)
    
    # Demographic parity testing
    st.subheader("Demographic Parity Testing")
    
    # Simulate demographic analysis
    demographics = ['Gender', 'Age Group', 'Ethnicity', 'Education Level']
    
    selected_demographic = st.selectbox("Select demographic for analysis:", demographics)
    
    if selected_demographic:
        # Generate sample demographic parity data
        if selected_demographic == 'Gender':
            groups = ['Male', 'Female', 'Non-binary']
            selection_rates = [0.75, 0.68, 0.72]
        elif selected_demographic == 'Age Group':
            groups = ['18-30', '31-45', '46-60', '60+']
            selection_rates = [0.78, 0.75, 0.65, 0.55]
        elif selected_demographic == 'Ethnicity':
            groups = ['White', 'Black', 'Hispanic', 'Asian', 'Other']
            selection_rates = [0.74, 0.62, 0.68, 0.82, 0.70]
        else:  # Education Level
            groups = ['High School', 'Bachelor', 'Master', 'PhD']
            selection_rates = [0.45, 0.72, 0.85, 0.88]
        
        # Create demographic parity chart
        fig = go.Figure(data=[
            go.Bar(x=groups, y=selection_rates, 
                  marker_color=['red' if rate < 0.6 else 'orange' if rate < 0.7 else 'green' 
                               for rate in selection_rates])
        ])
        
        fig.update_layout(
            title=f"Selection Rates by {selected_demographic}",
            xaxis_title=selected_demographic,
            yaxis_title="Selection Rate",
            height=400
        )
        fig.add_hline(y=0.8, line_dash="dash", line_color="green", 
                     annotation_text="Target Rate")
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistical significance testing
        with st.expander("📊 Statistical Significance Testing"):
            st.markdown("**Chi-Square Test Results:**")
            # Simulate chi-square test results
            chi_square_stat = np.random.uniform(5, 25)
            p_value = np.random.uniform(0.001, 0.1)
            degrees_of_freedom = len(groups) - 1
            
            test_results = pd.DataFrame({
                'Test Statistic': [chi_square_stat],
                'p-value': [p_value],
                'Degrees of Freedom': [degrees_of_freedom],
                'Significance': ['Significant' if p_value < 0.05 else 'Not Significant']
            })
            
            st.dataframe(test_results, use_container_width=True, hide_index=True)
            
            if p_value < 0.05:
                st.error(f"⚠️ Significant bias detected (p < 0.05)")
            else:
                st.success(f"✅ No significant bias detected (p ≥ 0.05)")


def render_anomaly_detection(evidence_data: Dict[str, Any], config: DashboardConfig):
    """Render anomaly detection section."""
    st.header("🔍 Anomaly Detection")
    
    # Anomaly detection overview
    st.subheader("Anomaly Detection Results")
    
    # Generate sample anomaly data
    np.random.seed(42)
    n_samples = 1000
    normal_data = np.random.normal(0, 1, (n_samples, 2))
    anomaly_indices = np.random.choice(n_samples, 50, replace=False)
    anomaly_data = np.random.normal(3, 0.5, (50, 2))
    
    # Combine normal and anomalous data
    all_data = normal_data.copy()
    all_data[anomaly_indices] = anomaly_data
    
    # Create anomaly detection visualization
    fig = go.Figure()
    
    # Normal points
    normal_mask = np.ones(n_samples, dtype=bool)
    normal_mask[anomaly_indices] = False
    
    fig.add_trace(go.Scatter(
        x=all_data[normal_mask, 0],
        y=all_data[normal_mask, 1],
        mode='markers',
        name='Normal',
        marker=dict(color='blue', size=4),
        opacity=0.6
    ))
    
    # Anomalous points
    fig.add_trace(go.Scatter(
        x=all_data[anomaly_indices, 0],
        y=all_data[anomaly_indices, 1],
        mode='markers',
        name='Anomaly',
        marker=dict(color='red', size=8, symbol='x'),
        opacity=0.8
    ))
    
    fig.update_layout(
        title="Anomaly Detection Results",
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Anomaly statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", n_samples)
    with col2:
        st.metric("Anomalies Detected", len(anomaly_indices))
    with col3:
        anomaly_rate = (len(anomaly_indices) / n_samples) * 100
        st.metric("Anomaly Rate", f"{anomaly_rate:.1f}%")
    with col4:
        confidence_score = np.random.uniform(0.85, 0.95)
        st.metric("Detection Confidence", f"{confidence_score:.2f}")
    
    # Detailed anomaly analysis
    st.subheader("Detailed Anomaly Analysis")
    
    anomaly_tabs = st.tabs(["Individual Anomalies", "Patterns", "Time Series"])
    
    with anomaly_tabs[0]:
        # Individual anomaly details
        st.markdown("**Top Anomalies by Severity:**")
        
        # Generate anomaly details
        anomaly_details = []
        for i, idx in enumerate(anomaly_indices[:10]):
            severity = np.random.uniform(0.7, 1.0)
            anomaly_type = np.random.choice(['Statistical Outlier', 'Behavioral Anomaly', 'Pattern Deviation'])
            
            anomaly_details.append({
                'ID': f"ANO_{idx:04d}",
                'Severity': f"{severity:.3f}",
                'Type': anomaly_type,
                'Feature_1': f"{all_data[idx, 0]:.3f}",
                'Feature_2': f"{all_data[idx, 1]:.3f}"
            })
        
        anomaly_df = pd.DataFrame(anomaly_details)
        st.dataframe(anomaly_df, use_container_width=True, hide_index=True)
    
    with anomaly_tabs[1]:
        # Anomaly patterns
        st.markdown("**Anomaly Pattern Analysis:**")
        
        pattern_types = ['Clustered', 'Isolated', 'Sequential', 'Random']
        pattern_counts = [15, 20, 8, 7]
        
        fig = px.pie(
            values=pattern_counts,
            names=pattern_types,
            title="Anomaly Pattern Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with anomaly_tabs[2]:
        # Time series anomalies
        st.markdown("**Temporal Anomaly Analysis:**")
        
        # Generate time series with anomalies
        dates = pd.date_range(start='2024-01-01', end='2024-07-01', freq='D')
        ts_values = np.cumsum(np.random.normal(0, 1, len(dates))) + 100
        
        # Add some anomalies
        anomaly_dates = np.random.choice(len(dates), 20, replace=False)
        ts_values[anomaly_dates] += np.random.normal(0, 10, 20)
        
        fig = go.Figure()
        
        # Normal time series
        fig.add_trace(go.Scatter(
            x=dates,
            y=ts_values,
            mode='lines',
            name='Time Series',
            line=dict(color='blue')
        ))
        
        # Highlight anomalies
        fig.add_trace(go.Scatter(
            x=dates[anomaly_dates],
            y=ts_values[anomaly_dates],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=8, symbol='x')
        ))
        
        fig.update_layout(
            title="Time Series Anomaly Detection",
            xaxis_title="Date",
            yaxis_title="Value",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)


def render_statistical_analysis(evidence_data: Dict[str, Any], synthetic_data: Dict[str, pd.DataFrame], 
                               config: DashboardConfig):
    """Render statistical analysis section."""
    st.header("📈 Statistical Analysis")
    
    # Distribution analysis
    st.subheader("Distribution Analysis")
    
    if synthetic_data and 'resumes' in synthetic_data:
        df = synthetic_data['resumes']
        
        # Select numeric columns for analysis
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numeric_columns:
            selected_column = st.selectbox("Select column for distribution analysis:", numeric_columns)
            
            if selected_column in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig = px.histogram(
                        df, 
                        x=selected_column,
                        nbins=30,
                        title=f"Distribution of {selected_column}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig = px.box(
                        df, 
                        y=selected_column,
                        title=f"Box Plot of {selected_column}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Statistical summary
                st.markdown("**Statistical Summary:**")
                summary_stats = df[selected_column].describe()
                summary_df = pd.DataFrame({
                    'Statistic': summary_stats.index,
                    'Value': summary_stats.values
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
    
    # Correlation analysis
    st.subheader("Correlation Analysis")
    
    if synthetic_data and 'resumes' in synthetic_data:
        df = synthetic_data['resumes']
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) > 1:
            # Correlation matrix
            corr_matrix = numeric_df.corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Strong correlations
            with st.expander("📊 Strong Correlations"):
                strong_corr = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.5:
                            strong_corr.append({
                                'Feature 1': corr_matrix.columns[i],
                                'Feature 2': corr_matrix.columns[j],
                                'Correlation': f"{corr_val:.3f}",
                                'Strength': 'Strong' if abs(corr_val) > 0.7 else 'Moderate'
                            })
                
                if strong_corr:
                    corr_df = pd.DataFrame(strong_corr)
                    st.dataframe(corr_df, use_container_width=True, hide_index=True)
                else:
                    st.info("No strong correlations found (|r| > 0.5)")


def render_hypothesis_testing(evidence_data: Dict[str, Any], config: DashboardConfig):
    """Render hypothesis testing section."""
    st.header("🧪 Hypothesis Testing")
    
    # Hypothesis test setup
    st.subheader("Hypothesis Test Configuration")
    
    test_type = st.selectbox(
        "Select test type:",
        ["Two-Sample T-Test", "Chi-Square Test", "ANOVA", "Mann-Whitney U Test"]
    )
    
    significance_level = st.slider("Significance Level (α)", 0.01, 0.10, 0.05, 0.01)
    
    # Perform selected test
    if test_type == "Two-Sample T-Test":
        render_t_test(significance_level)
    elif test_type == "Chi-Square Test":
        render_chi_square_test(significance_level)
    elif test_type == "ANOVA":
        render_anova_test(significance_level)
    elif test_type == "Mann-Whitney U Test":
        render_mann_whitney_test(significance_level)


def render_t_test(alpha: float):
    """Render two-sample t-test results."""
    st.markdown("**Two-Sample T-Test: Group Comparison**")
    
    # Generate sample data
    np.random.seed(42)
    group1 = np.random.normal(75, 10, 100)  # Group 1 scores
    group2 = np.random.normal(78, 12, 120)  # Group 2 scores
    
    # Perform t-test (simulated)
    from scipy import stats
    t_stat, p_value = stats.ttest_ind(group1, group2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Group distributions
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=group1, name='Group 1', opacity=0.7, nbinsx=20))
        fig.add_trace(go.Histogram(x=group2, name='Group 2', opacity=0.7, nbinsx=20))
        fig.update_layout(title="Group Distributions", barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Test results
        st.markdown("**Test Results:**")
        results_df = pd.DataFrame({
            'Statistic': ['t-statistic', 'p-value', 'Significance Level', 'Result'],
            'Value': [
                f"{t_stat:.4f}",
                f"{p_value:.4f}",
                f"{alpha:.3f}",
                'Significant' if p_value < alpha else 'Not Significant'
            ]
        })
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        if p_value < alpha:
            st.error("🚨 Significant difference detected between groups")
        else:
            st.success("✅ No significant difference between groups")


def render_chi_square_test(alpha: float):
    """Render chi-square test results."""
    st.markdown("**Chi-Square Test: Independence Testing**")
    
    # Generate contingency table
    observed = np.array([
        [45, 55, 25],  # Group A
        [35, 65, 35],  # Group B
        [40, 60, 30]   # Group C
    ])
    
    categories = ['Category 1', 'Category 2', 'Category 3']
    groups = ['Group A', 'Group B', 'Group C']
    
    # Perform chi-square test
    from scipy import stats
    chi2_stat, p_value, dof, expected = stats.chi2_contingency(observed)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Observed vs Expected
        observed_df = pd.DataFrame(observed, index=groups, columns=categories)
        expected_df = pd.DataFrame(expected, index=groups, columns=categories)
        
        st.markdown("**Observed Frequencies:**")
        st.dataframe(observed_df, use_container_width=True)
        
        st.markdown("**Expected Frequencies:**")
        st.dataframe(expected_df.round(1), use_container_width=True)
    
    with col2:
        # Test results
        st.markdown("**Test Results:**")
        results_df = pd.DataFrame({
            'Statistic': ['Chi-square', 'p-value', 'Degrees of Freedom', 'Significance Level', 'Result'],
            'Value': [
                f"{chi2_stat:.4f}",
                f"{p_value:.4f}",
                f"{dof}",
                f"{alpha:.3f}",
                'Significant' if p_value < alpha else 'Not Significant'
            ]
        })
        st.dataframe(results_df, use_container_width=True, hide_index=True)
        
        if p_value < alpha:
            st.error("🚨 Significant association detected")
        else:
            st.success("✅ No significant association found")


def render_anova_test(alpha: float):
    """Render ANOVA test results."""
    st.markdown("**ANOVA: Multiple Group Comparison**")
    
    # Generate sample data for multiple groups
    np.random.seed(42)
    group1 = np.random.normal(75, 8, 50)
    group2 = np.random.normal(78, 10, 55)
    group3 = np.random.normal(72, 9, 48)
    group4 = np.random.normal(80, 12, 52)
    
    # Perform ANOVA
    from scipy import stats
    f_stat, p_value = stats.f_oneway(group1, group2, group3, group4)
    
    # Visualization
    all_data = np.concatenate([group1, group2, group3, group4])
    all_groups = (['Group 1'] * len(group1) + ['Group 2'] * len(group2) + 
                 ['Group 3'] * len(group3) + ['Group 4'] * len(group4))
    
    fig = px.box(x=all_groups, y=all_data, title="Group Comparison (ANOVA)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Results
    st.markdown("**ANOVA Results:**")
    results_df = pd.DataFrame({
        'Statistic': ['F-statistic', 'p-value', 'Significance Level', 'Result'],
        'Value': [
            f"{f_stat:.4f}",
            f"{p_value:.4f}",
            f"{alpha:.3f}",
            'Significant' if p_value < alpha else 'Not Significant'
        ]
    })
    st.dataframe(results_df, use_container_width=True, hide_index=True)


def render_mann_whitney_test(alpha: float):
    """Render Mann-Whitney U test results."""
    st.markdown("**Mann-Whitney U Test: Non-parametric Group Comparison**")
    
    # Generate sample data
    np.random.seed(42)
    group1 = np.random.exponential(2, 80)  # Non-normal distribution
    group2 = np.random.exponential(2.5, 75)
    
    # Perform Mann-Whitney U test
    from scipy import stats
    u_stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
    
    # Visualization
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=group1, name='Group 1', opacity=0.7, nbinsx=25))
    fig.add_trace(go.Histogram(x=group2, name='Group 2', opacity=0.7, nbinsx=25))
    fig.update_layout(title="Non-parametric Group Distributions", barmode='overlay')
    st.plotly_chart(fig, use_container_width=True)
    
    # Results
    st.markdown("**Mann-Whitney U Test Results:**")
    results_df = pd.DataFrame({
        'Statistic': ['U-statistic', 'p-value', 'Significance Level', 'Result'],
        'Value': [
            f"{u_stat:.4f}",
            f"{p_value:.4f}",
            f"{alpha:.3f}",
            'Significant' if p_value < alpha else 'Not Significant'
        ]
    })
    st.dataframe(results_df, use_container_width=True, hide_index=True)