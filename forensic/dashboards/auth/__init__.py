"""
Authentication module for forensic dashboard.
"""

from .authentication import (
    AuthenticationManager, 
    User, 
    AuditLog, 
    SecurityManager, 
    init_default_users
)

__all__ = [
    'AuthenticationManager', 
    'User', 
    'AuditLog', 
    'SecurityManager', 
    'init_default_users'
]