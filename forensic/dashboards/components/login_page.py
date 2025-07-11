"""
Login page component for forensic dashboard authentication.
"""

import streamlit as st
from datetime import datetime
import logging

from ..auth import AuthenticationManager, User
from ..utils.error_handling import error_handler, log_error
from ..utils.security_utils import sanitize_input, validate_user_input_length


def render(auth_manager: AuthenticationManager) -> bool:
    """
    Render login page and handle authentication.
    
    Args:
        auth_manager: Authentication manager instance
    
    Returns:
        True if user successfully authenticated, False otherwise
    """
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h1>🔍 Forensic Dashboard</h1>
            <h3>LLM Bias Investigation System</h3>
            <p><em>Secure Access Required</em></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Login form
        with st.form("login_form"):
            st.subheader("Please Sign In")
            
            username = st.text_input(
                "Username",
                placeholder="Enter your username",
                help="Enter your assigned username"
            )
            
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                help="Enter your password"
            )
            
            remember_me = st.checkbox("Remember me for this session")
            
            submitted = st.form_submit_button(
                "Sign In",
                use_container_width=True,
                type="primary"
            )
            
            if submitted:
                with error_handler("Authentication failed. Please check your credentials."):
                    if authenticate_user(auth_manager, username, password, remember_me):
                        return True
        
        # Additional information
        st.markdown("---")
        
        with st.expander("🔒 Security Information"):
            st.markdown("""
            **Security Notice:**
            - All access is logged and monitored
            - This system is for authorized forensic investigation only
            - Multiple failed login attempts will lock your account
            - Contact your administrator if you need assistance
            
            **Default Accounts (Development Only):**
            - Admin: `admin` / `admin123!`
            - Investigator: `investigator` / `invest123!`
            - Analyst: `analyst` / `analyst123!`
            """)
        
        with st.expander("ℹ️ System Information"):
            st.markdown(f"""
            **System Status:** Online
            **Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            **Version:** 1.0.0
            **Environment:** {'Production' if st.secrets.get('ENVIRONMENT') == 'production' else 'Development'}
            """)
    
    return False


def authenticate_user(auth_manager: AuthenticationManager, username: str, 
                     password: str, remember_me: bool = False) -> bool:
    """
    Authenticate user credentials and setup session.
    
    Args:
        auth_manager: Authentication manager instance
        username: Username to authenticate
        password: Password to authenticate
        remember_me: Whether to extend session duration
    
    Returns:
        True if authentication successful
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Input validation
        if not username or not password:
            st.error("Please enter both username and password")
            return False
        
        # Sanitize inputs
        username = sanitize_input(username)
        validate_user_input_length(username, min_length=1, max_length=50)
        validate_user_input_length(password, min_length=1, max_length=100)
        
        # Show loading spinner
        with st.spinner("Authenticating..."):
            # Attempt authentication
            user = auth_manager.authenticate(username, password)
            
            if user:
                # Authentication successful
                session_token = auth_manager.generate_session_token(user)
                
                # Store in session state
                st.session_state.authenticated = True
                st.session_state.user = user
                st.session_state.session_token = session_token
                st.session_state.login_time = datetime.now()
                st.session_state.remember_me = remember_me
                
                # Log successful authentication
                logger.info(f"User {username} authenticated successfully")
                
                # Show success message
                st.success(f"Welcome, {user.username}!")
                st.balloons()
                
                # Small delay for user experience
                import time
                time.sleep(1)
                
                # Rerun to show authenticated interface
                st.rerun()
                
                return True
            else:
                # Authentication failed
                st.error("Invalid username or password")
                logger.warning(f"Failed authentication attempt for user: {username}")
                return False
                
    except Exception as e:
        log_error(e, {"username": username, "action": "authentication"})
        st.error("Authentication system error. Please try again.")
        return False


def logout_user():
    """Logout current user and clear session."""
    if 'user' in st.session_state and st.session_state.user:
        username = st.session_state.user.username
        logging.getLogger(__name__).info(f"User {username} logged out")
    
    # Clear session state
    for key in ['authenticated', 'user', 'session_token', 'login_time', 'remember_me']:
        if key in st.session_state:
            del st.session_state[key]
    
    st.success("Successfully logged out")
    st.rerun()


def check_session_timeout():
    """Check if user session has timed out."""
    if not st.session_state.get('authenticated', False):
        return False
    
    if 'login_time' not in st.session_state:
        return False
    
    # Check session timeout (default 1 hour)
    session_duration = datetime.now() - st.session_state.login_time
    timeout_hours = 8 if st.session_state.get('remember_me', False) else 1
    
    if session_duration.total_seconds() > timeout_hours * 3600:
        st.warning("Your session has expired. Please log in again.")
        logout_user()
        return False
    
    return True


def render_user_info():
    """Render current user information in sidebar."""
    if not st.session_state.get('authenticated', False):
        return
    
    user = st.session_state.user
    if not user:
        return
    
    st.sidebar.markdown("### 👤 User Information")
    st.sidebar.write(f"**Username:** {user.username}")
    st.sidebar.write(f"**Role:** {user.role.title()}")
    st.sidebar.write(f"**Email:** {user.email}")
    
    # Show permissions
    if user.permissions:
        st.sidebar.write("**Permissions:**")
        for permission in user.permissions:
            st.sidebar.write(f"• {permission.title()}")
    
    # Session info
    if 'login_time' in st.session_state:
        login_time = st.session_state.login_time
        st.sidebar.write(f"**Login Time:** {login_time.strftime('%H:%M:%S')}")
        
        # Session duration
        duration = datetime.now() - login_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, _ = divmod(remainder, 60)
        st.sidebar.write(f"**Session Duration:** {int(hours)}h {int(minutes)}m")
    
    st.sidebar.markdown("---")


def render_security_status():
    """Render security status information."""
    if not st.session_state.get('authenticated', False):
        return
    
    with st.sidebar.expander("🔒 Security Status"):
        st.write("✅ Secure connection established")
        st.write("✅ User authenticated")
        st.write("✅ Session encrypted")
        st.write("✅ Audit logging active")
        
        # Show last login info if available
        user = st.session_state.user
        if user and user.last_login:
            st.write(f"**Last Login:** {user.last_login.strftime('%Y-%m-%d %H:%M')}")


def require_permission(permission: str, auth_manager: AuthenticationManager) -> bool:
    """
    Check if current user has required permission.
    
    Args:
        permission: Required permission
        auth_manager: Authentication manager instance
    
    Returns:
        True if user has permission
    """
    if not st.session_state.get('authenticated', False):
        st.error("Please log in to access this feature")
        return False
    
    user = st.session_state.user
    if not user:
        st.error("Session error. Please log in again")
        return False
    
    if not auth_manager.check_permission(user, permission):
        st.error(f"You don't have permission to access this feature. Required: {permission}")
        return False
    
    return True


def render_permission_error(required_permission: str):
    """Render permission error message."""
    st.error("Access Denied")
    st.markdown(f"""
    **Required Permission:** {required_permission}
    
    You don't have the necessary permissions to access this feature.
    Please contact your administrator if you believe this is an error.
    """)
    
    # Show current user permissions
    if st.session_state.get('user'):
        user = st.session_state.user
        st.info(f"Your current permissions: {', '.join(user.permissions)}")


def audit_user_action(action: str, resource: str, details: dict = None):
    """
    Audit user action for security logging.
    
    Args:
        action: Action performed
        resource: Resource accessed
        details: Additional details
    """
    if not st.session_state.get('authenticated', False):
        return
    
    user = st.session_state.user
    if not user:
        return
    
    from ..utils.logging_utils import log_user_action
    
    log_user_action(
        username=user.username,
        action=action,
        resource=resource,
        details=details,
        success=True
    )