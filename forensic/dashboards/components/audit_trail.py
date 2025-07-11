"""
Audit Trail component for forensic investigation logging.
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

from ..config import DashboardConfig
from ..auth import AuthenticationManager
from .login_page import require_permission, audit_user_action


def render(config: DashboardConfig, auth_manager: AuthenticationManager):
    """Render audit trail page."""
    if not require_permission("audit", auth_manager):
        st.error("You need audit permissions to view this page")
        return
    
    audit_user_action("view", "audit_trail")
    
    st.header("📝 Audit Trail")
    
    # Sample audit data
    audit_data = [
        {"Timestamp": datetime.now() - timedelta(minutes=5), "User": "investigator", "Action": "view_bias_analysis", "Resource": "bias_dashboard", "Success": True},
        {"Timestamp": datetime.now() - timedelta(minutes=15), "User": "admin", "Action": "export_report", "Resource": "shap_analysis", "Success": True},
        {"Timestamp": datetime.now() - timedelta(hours=1), "User": "analyst", "Action": "login", "Resource": "dashboard", "Success": False},
        {"Timestamp": datetime.now() - timedelta(hours=2), "User": "investigator", "Action": "view_evidence", "Resource": "bias_data", "Success": True},
    ]
    
    audit_df = pd.DataFrame(audit_data)
    
    # Filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        user_filter = st.selectbox("Filter by User:", ["All"] + list(audit_df["User"].unique()))
    
    with col2:
        action_filter = st.selectbox("Filter by Action:", ["All"] + list(audit_df["Action"].unique()))
    
    with col3:
        success_filter = st.selectbox("Filter by Success:", ["All", "Success", "Failed"])
    
    # Apply filters
    filtered_df = audit_df.copy()
    
    if user_filter != "All":
        filtered_df = filtered_df[filtered_df["User"] == user_filter]
    
    if action_filter != "All":
        filtered_df = filtered_df[filtered_df["Action"] == action_filter]
    
    if success_filter != "All":
        success_bool = success_filter == "Success"
        filtered_df = filtered_df[filtered_df["Success"] == success_bool]
    
    # Display audit logs
    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Timestamp": st.column_config.DatetimeColumn("Timestamp"),
            "Success": st.column_config.CheckboxColumn("Success")
        }
    )