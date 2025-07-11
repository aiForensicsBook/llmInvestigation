"""
Setup script for the Forensic Dashboard.
"""

import subprocess
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def install_requirements():
    """Install required packages."""
    logger.info("Installing required packages...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        logger.error(f"Requirements file not found: {requirements_file}")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        logger.info("Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install requirements: {e}")
        return False


def create_directories():
    """Create necessary directories."""
    logger.info("Creating directories...")
    
    base_dir = Path(__file__).parent
    directories = [
        base_dir / "data",
        base_dir / "logs", 
        base_dir / "exports",
        base_dir / "static",
        base_dir / "templates"
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")


def initialize_database():
    """Initialize the authentication database."""
    logger.info("Initializing authentication database...")
    
    try:
        from auth import init_default_users, AuthenticationManager
        from config import get_config
        
        config = get_config()
        auth_db_path = config.data_dir / "users.db"
        
        auth_manager = AuthenticationManager(
            secret_key=config.security.secret_key,
            db_path=str(auth_db_path)
        )
        
        # Initialize default users
        init_default_users(auth_manager)
        
        logger.info("Authentication database initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def validate_installation():
    """Validate the installation."""
    logger.info("Validating installation...")
    
    try:
        # Test imports
        import streamlit
        import plotly
        import pandas
        import numpy
        import cryptography
        import jwt
        
        logger.info("All required packages imported successfully")
        
        # Test configuration
        from config import get_config
        config = get_config()
        
        # Validate configuration
        errors = config.validate()
        if errors:
            logger.warning("Configuration validation warnings:")
            for error in errors:
                logger.warning(f"  - {error}")
        else:
            logger.info("Configuration validation passed")
        
        return True
        
    except ImportError as e:
        logger.error(f"Missing required package: {e}")
        return False
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


def main():
    """Main setup function."""
    logger.info("Starting Forensic Dashboard setup...")
    
    success = True
    
    # Install requirements
    if not install_requirements():
        success = False
    
    # Create directories
    create_directories()
    
    # Initialize database
    if not initialize_database():
        success = False
    
    # Validate installation
    if not validate_installation():
        success = False
    
    if success:
        logger.info("Setup completed successfully!")
        logger.info("To start the dashboard, run: streamlit run main.py")
    else:
        logger.error("Setup failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()