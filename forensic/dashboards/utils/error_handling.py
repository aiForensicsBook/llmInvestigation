"""
Error handling utilities for the forensic dashboard.
"""

import logging
import traceback
import functools
from contextlib import contextmanager
from typing import Optional, Callable, Any
import streamlit as st
from datetime import datetime


class ForensicError(Exception):
    """Base exception for forensic dashboard errors."""
    pass


class AuthenticationError(ForensicError):
    """Authentication related errors."""
    pass


class AuthorizationError(ForensicError):
    """Authorization related errors."""
    pass


class DataIntegrityError(ForensicError):
    """Data integrity related errors."""
    pass


class ExportError(ForensicError):
    """Export operation related errors."""
    pass


class ValidationError(ForensicError):
    """Data validation related errors."""
    pass


def log_error(error: Exception, context: dict = None, username: str = None):
    """
    Log error with forensic context information.
    
    Args:
        error: The exception that occurred
        context: Additional context information
        username: Username of the user when error occurred
    """
    logger = logging.getLogger(__name__)
    
    error_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'timestamp': datetime.now().isoformat(),
        'username': username,
        'context': context or {},
        'traceback': traceback.format_exc()
    }
    
    logger.error(f"Forensic dashboard error: {error}", extra=error_info)


def handle_streamlit_error(error: Exception, user_message: str = None, 
                          show_details: bool = False):
    """
    Handle errors in Streamlit interface with user-friendly messages.
    
    Args:
        error: The exception that occurred
        user_message: Custom message to show to user
        show_details: Whether to show technical details to user
    """
    if user_message is None:
        user_message = "An error occurred while processing your request."
    
    # Show user-friendly error message
    st.error(user_message)
    
    # Show technical details if requested and user has appropriate permissions
    if show_details:
        if hasattr(st.session_state, 'user') and st.session_state.user:
            user = st.session_state.user
            if 'admin' in user.permissions or 'debug' in user.permissions:
                with st.expander("Technical Details"):
                    st.code(f"Error Type: {type(error).__name__}\n"
                            f"Error Message: {str(error)}\n"
                            f"Timestamp: {datetime.now().isoformat()}")
                    st.code(traceback.format_exc())


@contextmanager
def error_handler(user_message: str = None, show_details: bool = False,
                 log_context: dict = None):
    """
    Context manager for comprehensive error handling.
    
    Args:
        user_message: Custom message to show to user
        show_details: Whether to show technical details
        log_context: Additional context for logging
    """
    try:
        yield
    except AuthenticationError as e:
        log_error(e, log_context)
        st.error("Authentication failed. Please log in again.")
        st.session_state.authenticated = False
        st.rerun()
    except AuthorizationError as e:
        log_error(e, log_context)
        st.error("You don't have permission to perform this action.")
    except DataIntegrityError as e:
        log_error(e, log_context)
        handle_streamlit_error(
            e, 
            "Data integrity check failed. This may indicate tampering or corruption.",
            show_details
        )
    except ExportError as e:
        log_error(e, log_context)
        handle_streamlit_error(
            e,
            "Failed to export data. Please check your permissions and try again.",
            show_details
        )
    except ValidationError as e:
        log_error(e, log_context)
        handle_streamlit_error(
            e,
            f"Validation error: {str(e)}",
            False  # Don't show technical details for validation errors
        )
    except Exception as e:
        # Get username if available
        username = None
        if hasattr(st.session_state, 'user') and st.session_state.user:
            username = st.session_state.user.username
        
        log_error(e, log_context, username)
        handle_streamlit_error(e, user_message, show_details)


def forensic_exception_handler(func: Callable) -> Callable:
    """
    Decorator for forensic exception handling.
    
    Args:
        func: Function to wrap with exception handling
    
    Returns:
        Wrapped function with exception handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get username if available
            username = None
            if hasattr(st.session_state, 'user') and st.session_state.user:
                username = st.session_state.user.username
            
            log_error(e, {'function': func.__name__}, username)
            handle_streamlit_error(
                e,
                f"Error in {func.__name__}: {str(e)}",
                show_details=True
            )
            raise
    
    return wrapper


def validate_input(value: Any, validation_rules: dict) -> bool:
    """
    Validate input according to specified rules.
    
    Args:
        value: Value to validate
        validation_rules: Dictionary of validation rules
    
    Returns:
        True if validation passes
    
    Raises:
        ValidationError: If validation fails
    """
    # Required field check
    if validation_rules.get('required', False) and not value:
        raise ValidationError("This field is required")
    
    # Type check
    expected_type = validation_rules.get('type')
    if expected_type and not isinstance(value, expected_type):
        raise ValidationError(f"Expected {expected_type.__name__}, got {type(value).__name__}")
    
    # Length checks for strings
    if isinstance(value, str):
        min_length = validation_rules.get('min_length')
        max_length = validation_rules.get('max_length')
        
        if min_length and len(value) < min_length:
            raise ValidationError(f"Minimum length is {min_length} characters")
        
        if max_length and len(value) > max_length:
            raise ValidationError(f"Maximum length is {max_length} characters")
    
    # Numeric range checks
    if isinstance(value, (int, float)):
        min_value = validation_rules.get('min_value')
        max_value = validation_rules.get('max_value')
        
        if min_value is not None and value < min_value:
            raise ValidationError(f"Minimum value is {min_value}")
        
        if max_value is not None and value > max_value:
            raise ValidationError(f"Maximum value is {max_value}")
    
    # Custom validator function
    custom_validator = validation_rules.get('validator')
    if custom_validator and not custom_validator(value):
        error_message = validation_rules.get('validator_message', "Custom validation failed")
        raise ValidationError(error_message)
    
    return True


def safe_execute(func: Callable, *args, default_value: Any = None, 
                error_message: str = None, **kwargs) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        default_value: Value to return if function fails
        error_message: Custom error message
        **kwargs: Function keyword arguments
    
    Returns:
        Function result or default value if error occurs
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if error_message:
            log_error(e, {'custom_message': error_message})
        else:
            log_error(e, {'function': func.__name__})
        
        return default_value


class ErrorCollector:
    """Collects multiple errors for batch reporting."""
    
    def __init__(self):
        """Initialize error collector."""
        self.errors = []
    
    def add_error(self, error: Exception, context: str = None):
        """Add an error to the collection."""
        self.errors.append({
            'error': error,
            'context': context,
            'timestamp': datetime.now(),
            'error_type': type(error).__name__,
            'message': str(error)
        })
    
    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return len(self.errors) > 0
    
    def get_error_summary(self) -> str:
        """Get a summary of all errors."""
        if not self.errors:
            return "No errors"
        
        summary = f"Found {len(self.errors)} error(s):\n"
        for i, error_info in enumerate(self.errors, 1):
            context = f" ({error_info['context']})" if error_info['context'] else ""
            summary += f"{i}. {error_info['error_type']}: {error_info['message']}{context}\n"
        
        return summary
    
    def display_errors(self):
        """Display errors in Streamlit interface."""
        if not self.errors:
            return
        
        st.error(f"Found {len(self.errors)} error(s):")
        
        for i, error_info in enumerate(self.errors, 1):
            with st.expander(f"Error {i}: {error_info['error_type']}"):
                st.write(f"**Message:** {error_info['message']}")
                if error_info['context']:
                    st.write(f"**Context:** {error_info['context']}")
                st.write(f"**Timestamp:** {error_info['timestamp'].isoformat()}")
    
    def log_all_errors(self):
        """Log all collected errors."""
        for error_info in self.errors:
            log_error(
                error_info['error'], 
                {'context': error_info['context']},
                getattr(st.session_state, 'user', {}).get('username') if hasattr(st.session_state, 'user') else None
            )
    
    def clear(self):
        """Clear all errors."""
        self.errors.clear()