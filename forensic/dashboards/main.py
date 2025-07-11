"""
Main Forensic Dashboard Application
==================================

A comprehensive forensic analysis dashboard for LLM bias detection and investigation.
This is the main entry point for the Streamlit application.
"""

import streamlit as st
import sys
import os
from pathlib import Path
from datetime import datetime
import traceback

# Add the current directory to the Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# Import dashboard components
from config import get_config
from auth import AuthenticationManager, init_default_users
from components import (
    login_page, 
    executive_summary, 
    technical_analysis, 
    bias_analysis,
    shap_analysis,
    evidently_monitoring,
    real_time_monitoring,
    audit_trail,
    export_manager,
    help_system
)
from utils import setup_logging, error_handler


def initialize_app():
    """Initialize the application with configuration and authentication."""
    try:
        # Get configuration
        config = get_config()
        
        # Setup logging
        setup_logging(config.logs_dir)
        
        # Initialize authentication
        auth_db_path = config.data_dir / "users.db"
        auth_manager = AuthenticationManager(
            secret_key=config.security.secret_key,
            db_path=str(auth_db_path)
        )
        
        # Initialize default users if database is new
        if not auth_db_path.exists():
            init_default_users(auth_manager)
        
        return config, auth_manager
        
    except Exception as e:
        st.error(f"Failed to initialize application: {str(e)}")
        st.stop()


def check_authentication(auth_manager):
    """Check user authentication and manage session."""
    # Check if user is already authenticated
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.session_token = None
    
    # If not authenticated, show login page
    if not st.session_state.authenticated:
        return login_page(auth_manager)
    
    # Verify session token is still valid
    if st.session_state.session_token:
        user_data = auth_manager.verify_session_token(st.session_state.session_token)
        if not user_data:
            # Token expired, logout user
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.session_token = None
            st.rerun()
    
    return True


def create_sidebar_navigation():
    """Create sidebar navigation menu."""
    st.sidebar.title("🔍 Forensic Dashboard")
    
    # User information
    if st.session_state.user:
        st.sidebar.write(f"**User:** {st.session_state.user.username}")
        st.sidebar.write(f"**Role:** {st.session_state.user.role}")
        st.sidebar.divider()
    
    # Navigation menu
    pages = {
        "Executive Summary": "📊",
        "Technical Analysis": "🔬", 
        "Bias Analysis": "⚖️",
        "SHAP Analysis": "🎯",
        "Evidently Monitoring": "📈",
        "Real-time Monitoring": "⏱️",
        "Audit Trail": "📝",
        "Export Manager": "📤",
        "Help & Documentation": "❓"
    }
    
    # Check user permissions for each page
    user = st.session_state.user
    available_pages = {}
    
    for page, icon in pages.items():
        # Basic permission checking
        if page == "Executive Summary" and "read" in user.permissions:
            available_pages[page] = icon
        elif page == "Technical Analysis" and ("write" in user.permissions or "admin" in user.permissions):
            available_pages[page] = icon
        elif page == "Bias Analysis" and "read" in user.permissions:
            available_pages[page] = icon
        elif page == "SHAP Analysis" and "read" in user.permissions:
            available_pages[page] = icon
        elif page == "Evidently Monitoring" and "read" in user.permissions:
            available_pages[page] = icon
        elif page == "Real-time Monitoring" and "read" in user.permissions:
            available_pages[page] = icon
        elif page == "Audit Trail" and ("audit" in user.permissions or "admin" in user.permissions):
            available_pages[page] = icon
        elif page == "Export Manager" and "export" in user.permissions:
            available_pages[page] = icon
        elif page == "Help & Documentation":
            available_pages[page] = icon
    
    # Create radio buttons for navigation
    selected_page = st.sidebar.radio(
        "Navigate to:",
        list(available_pages.keys()),
        format_func=lambda x: f"{available_pages[x]} {x}"
    )
    
    st.sidebar.divider()
    
    # Logout button
    if st.sidebar.button("🚪 Logout"):
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.session_token = None
        st.rerun()
    
    return selected_page


def display_page_header(page_name):
    """Display page header with title and timestamp."""
    st.title(f"🔍 {page_name}")
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    st.divider()


def main():
    """Main application function."""
    # Configure Streamlit page
    st.set_page_config(
        page_title="Forensic Dashboard",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2e6da4 100%);
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    
    .metric-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #2e6da4;
    }
    
    .alert-box {
        background: #fff3cd;
        color: #856404;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ffeaa7;
        margin: 10px 0;
    }
    
    .error-box {
        background: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
        margin: 10px 0;
    }
    
    .success-box {
        background: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    try:
        # Initialize application
        config, auth_manager = initialize_app()
        
        # Check authentication
        if not check_authentication(auth_manager):
            return
        
        # Create navigation
        selected_page = create_sidebar_navigation()
        
        # Display selected page
        display_page_header(selected_page)
        
        # Route to appropriate page component
        with error_handler():
            if selected_page == "Executive Summary":
                executive_summary.render(config, auth_manager)
            elif selected_page == "Technical Analysis":
                technical_analysis.render(config, auth_manager)
            elif selected_page == "Bias Analysis":
                bias_analysis.render(config, auth_manager)
            elif selected_page == "SHAP Analysis":
                shap_analysis.render(config, auth_manager)
            elif selected_page == "Evidently Monitoring":
                evidently_monitoring.render(config, auth_manager)
            elif selected_page == "Real-time Monitoring":
                real_time_monitoring.render(config, auth_manager)
            elif selected_page == "Audit Trail":
                audit_trail.render(config, auth_manager)
            elif selected_page == "Export Manager":
                export_manager.render(config, auth_manager)
            elif selected_page == "Help & Documentation":
                help_system.render(config, auth_manager)
        
        # Display footer
        st.sidebar.markdown("---")
        st.sidebar.markdown(
            f"**Forensic Dashboard v1.0.0**\n\n"
            f"Environment: {'Production' if config.is_production else 'Development'}\n\n"
            f"© 2024 Forensic AI Investigation Team"
        )
        
    except Exception as e:
        st.error("Application Error")
        st.exception(e)
        
        # Log error for forensic purposes
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Application error: {str(e)}\n{traceback.format_exc()}")


if __name__ == "__main__":
    main()