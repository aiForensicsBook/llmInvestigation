"""
Forensic Dashboard Configuration
===============================

Central configuration management for the forensic dashboard system.
This module contains all configuration settings for security, database,
monitoring, and dashboard behavior.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import json


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    type: str = "sqlite"
    host: str = "localhost"
    port: int = 5432
    database: str = "forensic_dashboard.db"
    username: Optional[str] = None
    password: Optional[str] = None
    
    @property
    def connection_string(self) -> str:
        """Generate database connection string."""
        if self.type == "sqlite":
            return f"sqlite:///{self.database}"
        elif self.type == "postgresql":
            return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        else:
            raise ValueError(f"Unsupported database type: {self.type}")


@dataclass
class SecurityConfig:
    """Security and authentication configuration."""
    secret_key: str = "CHANGE_THIS_IN_PRODUCTION"
    session_timeout: int = 3600  # 1 hour
    max_login_attempts: int = 5
    password_min_length: int = 8
    require_2fa: bool = False
    allowed_ip_ranges: List[str] = None
    
    def __post_init__(self):
        if self.allowed_ip_ranges is None:
            self.allowed_ip_ranges = ["0.0.0.0/0"]  # Allow all by default


@dataclass
class MonitoringConfig:
    """Real-time monitoring configuration."""
    refresh_interval: int = 30  # seconds
    alert_threshold_bias: float = 0.1
    alert_threshold_drift: float = 0.15
    alert_threshold_performance: float = 0.05
    max_alerts_per_hour: int = 10
    notification_channels: List[str] = None
    
    def __post_init__(self):
        if self.notification_channels is None:
            self.notification_channels = ["dashboard", "email"]


@dataclass
class ExportConfig:
    """Export and reporting configuration."""
    max_export_rows: int = 100000
    allowed_formats: List[str] = None
    export_directory: str = "exports"
    retention_days: int = 30
    
    def __post_init__(self):
        if self.allowed_formats is None:
            self.allowed_formats = ["pdf", "xlsx", "csv", "json"]


class DashboardConfig:
    """
    Main configuration class for the forensic dashboard.
    
    This class centralizes all configuration settings and provides
    methods to load configuration from files or environment variables.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """Initialize configuration with optional config file."""
        self.base_dir = Path(__file__).parent.parent
        self.data_dir = self.base_dir / "data"
        self.logs_dir = self.base_dir / "logs"
        self.exports_dir = self.base_dir / "exports"
        self.evidence_dir = self.base_dir.parent / "evidence"
        
        # Ensure directories exist
        for directory in [self.data_dir, self.logs_dir, self.exports_dir]:
            directory.mkdir(exist_ok=True)
        
        # Initialize sub-configurations
        self.database = DatabaseConfig()
        self.security = SecurityConfig()
        self.monitoring = MonitoringConfig()
        self.export = ExportConfig()
        
        # Load configuration if file provided
        if config_file:
            self.load_from_file(config_file)
        
        # Override with environment variables
        self.load_from_env()
    
    def load_from_file(self, config_file: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update database config
            if 'database' in config_data:
                db_config = config_data['database']
                for key, value in db_config.items():
                    if hasattr(self.database, key):
                        setattr(self.database, key, value)
            
            # Update security config
            if 'security' in config_data:
                sec_config = config_data['security']
                for key, value in sec_config.items():
                    if hasattr(self.security, key):
                        setattr(self.security, key, value)
            
            # Update monitoring config
            if 'monitoring' in config_data:
                mon_config = config_data['monitoring']
                for key, value in mon_config.items():
                    if hasattr(self.monitoring, key):
                        setattr(self.monitoring, key, value)
            
            # Update export config
            if 'export' in config_data:
                exp_config = config_data['export']
                for key, value in exp_config.items():
                    if hasattr(self.export, key):
                        setattr(self.export, key, value)
                        
        except FileNotFoundError:
            print(f"Warning: Configuration file {config_file} not found. Using defaults.")
        except json.JSONDecodeError as e:
            print(f"Error parsing configuration file: {e}. Using defaults.")
    
    def load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Database configuration
        self.database.type = os.getenv("DB_TYPE", self.database.type)
        self.database.host = os.getenv("DB_HOST", self.database.host)
        self.database.port = int(os.getenv("DB_PORT", self.database.port))
        self.database.database = os.getenv("DB_NAME", self.database.database)
        self.database.username = os.getenv("DB_USERNAME", self.database.username)
        self.database.password = os.getenv("DB_PASSWORD", self.database.password)
        
        # Security configuration
        self.security.secret_key = os.getenv("SECRET_KEY", self.security.secret_key)
        self.security.session_timeout = int(os.getenv("SESSION_TIMEOUT", self.security.session_timeout))
        self.security.max_login_attempts = int(os.getenv("MAX_LOGIN_ATTEMPTS", self.security.max_login_attempts))
        
        # Monitoring configuration
        self.monitoring.refresh_interval = int(os.getenv("REFRESH_INTERVAL", self.monitoring.refresh_interval))
        self.monitoring.alert_threshold_bias = float(os.getenv("ALERT_THRESHOLD_BIAS", self.monitoring.alert_threshold_bias))
        
    def save_to_file(self, config_file: str) -> None:
        """Save current configuration to JSON file."""
        config_data = {
            "database": {
                "type": self.database.type,
                "host": self.database.host,
                "port": self.database.port,
                "database": self.database.database,
                "username": self.database.username,
                # Note: Password not saved for security
            },
            "security": {
                "session_timeout": self.security.session_timeout,
                "max_login_attempts": self.security.max_login_attempts,
                "password_min_length": self.security.password_min_length,
                "require_2fa": self.security.require_2fa,
                "allowed_ip_ranges": self.security.allowed_ip_ranges,
            },
            "monitoring": {
                "refresh_interval": self.monitoring.refresh_interval,
                "alert_threshold_bias": self.monitoring.alert_threshold_bias,
                "alert_threshold_drift": self.monitoring.alert_threshold_drift,
                "alert_threshold_performance": self.monitoring.alert_threshold_performance,
                "max_alerts_per_hour": self.monitoring.max_alerts_per_hour,
                "notification_channels": self.monitoring.notification_channels,
            },
            "export": {
                "max_export_rows": self.export.max_export_rows,
                "allowed_formats": self.export.allowed_formats,
                "export_directory": self.export.export_directory,
                "retention_days": self.export.retention_days,
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    def validate(self) -> List[str]:
        """Validate configuration settings and return any errors."""
        errors = []
        
        # Validate security settings
        if self.security.secret_key == "CHANGE_THIS_IN_PRODUCTION":
            errors.append("Secret key should be changed in production environment")
        
        if self.security.password_min_length < 8:
            errors.append("Password minimum length should be at least 8 characters")
        
        # Validate monitoring settings
        if self.monitoring.refresh_interval < 5:
            errors.append("Refresh interval should be at least 5 seconds")
        
        if not (0 < self.monitoring.alert_threshold_bias < 1):
            errors.append("Alert threshold for bias should be between 0 and 1")
        
        # Validate export settings
        if self.export.max_export_rows < 1000:
            errors.append("Maximum export rows should be at least 1000")
        
        return errors
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return os.getenv("ENVIRONMENT", "development").lower() == "production"
    
    @property
    def debug_mode(self) -> bool:
        """Check if debug mode is enabled."""
        return os.getenv("DEBUG", "false").lower() == "true" and not self.is_production


# Global configuration instance
config = DashboardConfig()

# Convenience function to get configuration
def get_config() -> DashboardConfig:
    """Get the global configuration instance."""
    return config