"""
SHAP Analysis component for explainable AI investigation.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, Any

from ..config import DashboardConfig
from ..auth import AuthenticationManager
from ..utils.data_utils import load_evidence_data
from ..utils.error_handling import error_handler
from .login_page import require_permission, audit_user_action


def render(config: DashboardConfig, auth_manager: AuthenticationManager):
    """Render SHAP analysis page."""
    if not require_permission("read", auth_manager):
        return
    
    audit_user_action("view", "shap_analysis")
    
    with error_handler("Failed to load SHAP analysis"):
        st.header("🎯 SHAP Analysis Dashboard")
        
        tabs = st.tabs(["📊 Feature Importance", "🔍 Individual Predictions", "📈 Dependency Plots", "🎭 Interaction Effects"])
        
        with tabs[0]:
            render_feature_importance()
        
        with tabs[1]:
            render_individual_predictions()
        
        with tabs[2]:
            render_dependency_plots()
        
        with tabs[3]:
            render_interaction_effects()


def render_feature_importance():
    """Render SHAP feature importance analysis."""
    st.subheader("Global Feature Importance")
    
    # Sample SHAP data
    features = ['experience_years', 'education_level', 'skills_match', 'location_preference', 
               'previous_salary', 'age', 'gender_inferred', 'name_ethnicity', 'resume_quality']
    shap_values = np.random.uniform(-0.3, 0.3, len(features))
    importance = np.abs(shap_values)
    
    # Sort by importance
    sorted_indices = np.argsort(importance)[::-1]
    sorted_features = [features[i] for i in sorted_indices]
    sorted_importance = importance[sorted_indices]
    sorted_shap = shap_values[sorted_indices]
    
    # Feature importance plot
    colors = ['red' if val < 0 else 'blue' for val in sorted_shap]
    
    fig = go.Figure(data=[
        go.Bar(
            y=sorted_features,
            x=sorted_importance,
            orientation='h',
            marker_color=colors,
            text=[f"{val:.3f}" for val in sorted_shap],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="SHAP Feature Importance",
        xaxis_title="Mean |SHAP Value|",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature impact table
    impact_df = pd.DataFrame({
        'Feature': sorted_features,
        'Mean SHAP Value': sorted_shap,
        'Importance': sorted_importance,
        'Impact Direction': ['Negative' if val < 0 else 'Positive' for val in sorted_shap]
    })
    
    st.dataframe(impact_df, use_container_width=True, hide_index=True)


def render_individual_predictions():
    """Render individual prediction explanations."""
    st.subheader("Individual Prediction Analysis")
    
    # Sample data for individual predictions
    sample_id = st.selectbox("Select Sample ID:", [f"Sample_{i:03d}" for i in range(1, 21)])
    
    features = ['experience_years', 'education_level', 'skills_match', 'location_preference', 
               'previous_salary', 'age', 'gender_inferred', 'name_ethnicity']
    
    # Generate sample SHAP values for this prediction
    np.random.seed(hash(sample_id) % 1000)
    shap_values = np.random.uniform(-0.4, 0.4, len(features))
    feature_values = np.random.uniform(0, 1, len(features))
    
    # Waterfall plot
    cumulative = np.cumsum(np.concatenate([[0], shap_values]))
    base_value = 0.5
    
    fig = go.Figure()
    
    for i, (feature, shap_val) in enumerate(zip(features, shap_values)):
        color = 'red' if shap_val < 0 else 'blue'
        fig.add_trace(go.Bar(
            x=[feature],
            y=[abs(shap_val)],
            base=[base_value + cumulative[i]],
            marker_color=color,
            name=f"{feature}: {shap_val:.3f}",
            text=f"{shap_val:.3f}",
            textposition='middle center'
        ))
    
    fig.update_layout(
        title=f"SHAP Explanation for {sample_id}",
        yaxis_title="Prediction Score",
        height=500,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature values table
    individual_df = pd.DataFrame({
        'Feature': features,
        'Value': feature_values,
        'SHAP Value': shap_values,
        'Contribution': ['Positive' if val > 0 else 'Negative' for val in shap_values]
    })
    
    st.dataframe(individual_df, use_container_width=True, hide_index=True)


def render_dependency_plots():
    """Render SHAP dependency plots."""
    st.subheader("Feature Dependency Analysis")
    
    feature_list = ['experience_years', 'education_level', 'skills_match', 'age', 'previous_salary']
    selected_feature = st.selectbox("Select feature for dependency analysis:", feature_list)
    
    # Generate sample dependency data
    np.random.seed(42)
    n_samples = 1000
    feature_values = np.random.uniform(0, 10, n_samples)
    shap_values = 0.1 * feature_values + np.random.normal(0, 0.2, n_samples)
    
    # Add some non-linear relationship
    if selected_feature in ['age', 'experience_years']:
        shap_values += -0.005 * (feature_values - 5) ** 2
    
    # Color by interaction feature (gender in this case)
    interaction_values = np.random.choice([0, 1], n_samples)
    
    fig = px.scatter(
        x=feature_values,
        y=shap_values,
        color=interaction_values,
        title=f"SHAP Dependency Plot: {selected_feature}",
        labels={'x': f'{selected_feature} Value', 'y': 'SHAP Value', 'color': 'Gender (0=F, 1=M)'},
        opacity=0.6
    )
    
    # Add trend line
    z = np.polyfit(feature_values, shap_values, 2)
    p = np.poly1d(z)
    x_trend = np.linspace(feature_values.min(), feature_values.max(), 100)
    y_trend = p(x_trend)
    
    fig.add_trace(go.Scatter(
        x=x_trend,
        y=y_trend,
        mode='lines',
        name='Trend',
        line=dict(color='red', width=3)
    ))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    correlation = np.corrcoef(feature_values, shap_values)[0, 1]
    st.metric("Feature-SHAP Correlation", f"{correlation:.3f}")


def render_interaction_effects():
    """Render SHAP interaction effects."""
    st.subheader("Feature Interaction Analysis")
    
    features = ['experience_years', 'education_level', 'skills_match', 'age', 'gender_inferred']
    
    # Generate sample interaction matrix
    np.random.seed(42)
    interaction_matrix = np.random.uniform(-0.1, 0.1, (len(features), len(features)))
    
    # Make matrix symmetric
    interaction_matrix = (interaction_matrix + interaction_matrix.T) / 2
    np.fill_diagonal(interaction_matrix, 0)
    
    # Interaction heatmap
    fig = px.imshow(
        interaction_matrix,
        x=features,
        y=features,
        color_continuous_scale='RdBu_r',
        title="SHAP Interaction Effects Matrix"
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Top interactions
    st.subheader("Strongest Interactions")
    
    interactions = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            interactions.append({
                'Feature 1': features[i],
                'Feature 2': features[j],
                'Interaction Strength': abs(interaction_matrix[i, j]),
                'Effect': 'Synergistic' if interaction_matrix[i, j] > 0 else 'Antagonistic'
            })
    
    interactions_df = pd.DataFrame(interactions)
    interactions_df = interactions_df.sort_values('Interaction Strength', ascending=False)
    
    st.dataframe(interactions_df.head(10), use_container_width=True, hide_index=True)