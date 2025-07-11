"""
Logging utilities for forensic dashboard.
"""

import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
import json
import sys


class ForensicFormatter(logging.Formatter):
    """Custom formatter for forensic logging with structured output."""
    
    def format(self, record):
        """Format log record with forensic metadata."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add custom attributes
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message']:
                log_entry['extra'] = log_entry.get('extra', {})
                log_entry['extra'][key] = value
        
        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(log_dir: Path, log_level: str = "INFO"):
    """
    Setup comprehensive logging for the forensic dashboard.
    
    Args:
        log_dir: Directory to store log files
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    # Convert string level to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatters
    forensic_formatter = ForensicFormatter()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Main application log file (rotating)
    app_log_file = log_dir / "dashboard.log"
    app_handler = logging.handlers.RotatingFileHandler(
        app_log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=10
    )
    app_handler.setLevel(level)
    app_handler.setFormatter(forensic_formatter)
    root_logger.addHandler(app_handler)
    
    # Error log file (errors and above only)
    error_log_file = log_dir / "errors.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(forensic_formatter)
    root_logger.addHandler(error_handler)
    
    # Audit log file (separate for forensic audit trail)
    audit_log_file = log_dir / "audit.log"
    audit_handler = logging.handlers.RotatingFileHandler(
        audit_log_file,
        maxBytes=20 * 1024 * 1024,  # 20MB
        backupCount=20
    )
    audit_handler.setLevel(logging.INFO)
    audit_handler.setFormatter(forensic_formatter)
    
    # Create audit logger
    audit_logger = logging.getLogger('audit')
    audit_logger.setLevel(logging.INFO)
    audit_logger.addHandler(audit_handler)
    audit_logger.propagate = False  # Don't propagate to root logger
    
    # Security log file (authentication and authorization events)
    security_log_file = log_dir / "security.log"
    security_handler = logging.handlers.RotatingFileHandler(
        security_log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=15
    )
    security_handler.setLevel(logging.INFO)
    security_handler.setFormatter(forensic_formatter)
    
    # Create security logger
    security_logger = logging.getLogger('security')
    security_logger.setLevel(logging.INFO)
    security_logger.addHandler(security_handler)
    security_logger.propagate = False
    
    # Performance log file (for monitoring dashboard performance)
    performance_log_file = log_dir / "performance.log"
    performance_handler = logging.handlers.RotatingFileHandler(
        performance_log_file,
        maxBytes=5 * 1024 * 1024,  # 5MB
        backupCount=5
    )
    performance_handler.setLevel(logging.INFO)
    performance_handler.setFormatter(forensic_formatter)
    
    # Create performance logger
    performance_logger = logging.getLogger('performance')
    performance_logger.setLevel(logging.INFO)
    performance_logger.addHandler(performance_handler)
    performance_logger.propagate = False
    
    # Log initialization
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized", extra={
        'log_dir': str(log_dir),
        'log_level': log_level,
        'handlers': [
            'console', 'app_file', 'error_file', 
            'audit_file', 'security_file', 'performance_file'
        ]
    })


def get_audit_logger():
    """Get the audit logger instance."""
    return logging.getLogger('audit')


def get_security_logger():
    """Get the security logger instance."""
    return logging.getLogger('security')


def get_performance_logger():
    """Get the performance logger instance."""
    return logging.getLogger('performance')


def log_user_action(username: str, action: str, resource: str, 
                   details: dict = None, success: bool = True):
    """
    Log user action for audit trail.
    
    Args:
        username: Username performing the action
        action: Action being performed
        resource: Resource being accessed
        details: Additional details about the action
        success: Whether the action was successful
    """
    audit_logger = get_audit_logger()
    audit_logger.info(f"User action: {action}", extra={
        'username': username,
        'action': action,
        'resource': resource,
        'success': success,
        'details': details or {}
    })


def log_security_event(event_type: str, username: str = None, 
                      ip_address: str = None, details: dict = None):
    """
    Log security-related events.
    
    Args:
        event_type: Type of security event
        username: Username involved (if applicable)
        ip_address: IP address (if applicable)
        details: Additional event details
    """
    security_logger = get_security_logger()
    security_logger.info(f"Security event: {event_type}", extra={
        'event_type': event_type,
        'username': username,
        'ip_address': ip_address,
        'details': details or {}
    })


def log_performance_metric(metric_name: str, value: float, 
                          username: str = None, details: dict = None):
    """
    Log performance metrics for monitoring.
    
    Args:
        metric_name: Name of the metric
        value: Metric value
        username: User associated with the metric (if applicable)
        details: Additional metric details
    """
    performance_logger = get_performance_logger()
    performance_logger.info(f"Performance metric: {metric_name}", extra={
        'metric_name': metric_name,
        'value': value,
        'username': username,
        'details': details or {}
    })


class TimedOperation:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str, username: str = None):
        """Initialize timed operation."""
        self.operation_name = operation_name
        self.username = username
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log performance."""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()
        
        log_performance_metric(
            f"{self.operation_name}_duration",
            duration,
            self.username,
            {
                'operation': self.operation_name,
                'start_time': self.start_time.isoformat(),
                'end_time': self.end_time.isoformat(),
                'success': exc_type is None
            }
        )