"""
Dashboard components for the forensic analysis interface.
"""

from .login_page import render as login_page
from .executive_summary import render as executive_summary  
from .technical_analysis import render as technical_analysis
from .bias_analysis import render as bias_analysis
from .shap_analysis import render as shap_analysis
from .evidently_monitoring import render as evidently_monitoring
from .real_time_monitoring import render as real_time_monitoring
from .audit_trail import render as audit_trail
from .export_manager import render as export_manager
from .help_system import render as help_system

__all__ = [
    'login_page',
    'executive_summary',
    'technical_analysis', 
    'bias_analysis',
    'shap_analysis',
    'evidently_monitoring',
    'real_time_monitoring',
    'audit_trail',
    'export_manager',
    'help_system'
]