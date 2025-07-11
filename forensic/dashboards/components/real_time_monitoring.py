"""
Real-time Monitoring component for live system monitoring and alerts.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import time

from ..config import DashboardConfig
from ..auth import AuthenticationManager
from .login_page import require_permission, audit_user_action


def render(config: DashboardConfig, auth_manager: AuthenticationManager):
    """Render real-time monitoring page."""
    if not require_permission("read", auth_manager):
        return
    
    audit_user_action("view", "real_time_monitoring")
    
    st.header("⏱️ Real-time Monitoring")
    
    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh (30 seconds)", value=False)
    
    if auto_refresh:
        time.sleep(30)
        st.rerun()
    
    # System status overview
    render_system_status()
    
    # Live metrics
    render_live_metrics()
    
    # Active alerts
    render_active_alerts()
    
    # Real-time charts
    render_realtime_charts()


def render_system_status():
    """Render system status overview."""
    st.subheader("🖥️ System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Status", "🟢 Online", "Uptime: 99.8%")
    with col2:
        st.metric("Model Status", "🟢 Active", "Last inference: 2s ago")
    with col3:
        st.metric("Data Pipeline", "🟢 Running", "Last update: 15s ago")
    with col4:
        st.metric("Alert System", "🟡 3 Active", "Bias threshold exceeded")


def render_live_metrics():
    """Render live performance metrics."""
    st.subheader("📊 Live Performance Metrics")
    
    # Generate real-time metrics
    current_time = datetime.now()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        accuracy = 0.847 + np.random.normal(0, 0.01)
        st.metric("Current Accuracy", f"{accuracy:.3f}", f"{np.random.uniform(-0.01, 0.01):.3f}")
    
    with col2:
        throughput = 1200 + np.random.randint(-50, 50)
        st.metric("Predictions/Hour", f"{throughput:,}", f"{np.random.randint(-20, 20):+d}")
    
    with col3:
        response_time = 45 + np.random.randint(-5, 15)
        st.metric("Avg Response Time", f"{response_time}ms", f"{np.random.randint(-5, 5):+d}ms")


def render_active_alerts():
    """Render active alerts."""
    st.subheader("🚨 Active Alerts")
    
    alerts = [
        {"Severity": "🔴 High", "Alert": "Gender bias threshold exceeded", "Time": "2 min ago", "Status": "Active"},
        {"Severity": "🟡 Medium", "Alert": "Data drift detected in age feature", "Time": "15 min ago", "Status": "Investigating"},
        {"Severity": "🟡 Medium", "Alert": "Response time above SLA", "Time": "1 hour ago", "Status": "Resolved"},
    ]
    
    alerts_df = pd.DataFrame(alerts)
    st.dataframe(alerts_df, use_container_width=True, hide_index=True)


def render_realtime_charts():
    """Render real-time monitoring charts."""
    st.subheader("📈 Real-time Charts")
    
    # Generate time series data
    now = datetime.now()
    times = [now - timedelta(minutes=i) for i in range(60, 0, -1)]
    
    # Bias scores over time
    bias_scores = 0.08 + 0.02 * np.sin(np.arange(60) * 0.2) + np.random.normal(0, 0.01, 60)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=bias_scores,
        mode='lines+markers',
        name='Bias Score',
        line=dict(color='red')
    ))
    
    fig.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Warning")
    fig.update_layout(title="Bias Score (Last Hour)", height=300)
    
    st.plotly_chart(fig, use_container_width=True)