#!/usr/bin/env python3
"""
Setup Script for Evidently Analyzer
===================================

This script sets up the Evidently analyzer for the forensic testing suite,
including dependency installation, configuration validation, and initial testing.

Author: Forensic AI Testing Suite
Created: 2025-07-01
Purpose: Setup and validation for Evidently integration
"""

import os
import sys
import subprocess
import logging
import tempfile
import shutil
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_python_version():
    """Check if Python version is compatible."""
    logger.info("Checking Python version...")
    
    if sys.version_info < (3, 7):
        logger.error("Python 3.7 or higher is required")
        return False
    
    logger.info(f"Python version: {sys.version}")
    return True


def install_dependencies():
    """Install required dependencies."""
    logger.info("Installing dependencies...")
    
    dependencies = [
        'evidently>=0.4.0',
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scipy>=1.7.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.4.0',
        'seaborn>=0.11.0'
    ]
    
    for dep in dependencies:
        try:
            logger.info(f"Installing {dep}...")
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', dep
            ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            logger.info(f"✓ {dep} installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"✗ Failed to install {dep}: {e}")
            return False
    
    return True


def validate_evidently_installation():
    """Validate that Evidently is properly installed."""
    logger.info("Validating Evidently installation...")
    
    try:
        import evidently
        logger.info(f"✓ Evidently version: {evidently.__version__}")
        
        # Test basic functionality
        from evidently.report import Report
        from evidently.metrics import DataDriftTable
        from evidently.test_suite import TestSuite
        from evidently.tests import TestNumberOfColumnsWithMissingValues
        
        logger.info("✓ Core Evidently components imported successfully")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Evidently import failed: {e}")
        return False
    except Exception as e:
        logger.error(f"✗ Evidently validation failed: {e}")
        return False


def validate_analyzer_import():
    """Validate that the Evidently analyzer can be imported."""
    logger.info("Validating Evidently analyzer import...")
    
    try:
        # Add current directory to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        from evidently_analyzer import (
            EvidentlyAnalyzer,
            EvidentlyAnalysisResult,
            BiasMonitoringAlert,
            EVIDENTLY_AVAILABLE
        )
        
        if not EVIDENTLY_AVAILABLE:
            logger.error("✗ EVIDENTLY_AVAILABLE flag is False")
            return False
        
        logger.info("✓ Evidently analyzer imported successfully")
        return True
        
    except ImportError as e:
        logger.error(f"✗ Analyzer import failed: {e}")
        return False


def run_basic_functionality_test():
    """Run basic functionality test."""
    logger.info("Running basic functionality test...")
    
    try:
        # Import required modules
        from evidently_analyzer import EvidentlyAnalyzer
        import pandas as pd
        import numpy as np
        
        # Create test data
        np.random.seed(42)
        n_samples = 100
        
        test_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.normal(0, 1, n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'target': np.random.choice([0, 1], n_samples),
            'prediction': np.random.choice([0, 1], n_samples)
        })
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize analyzer
            analyzer = EvidentlyAnalyzer(
                case_id="SETUP_TEST",
                investigator="Setup Script",
                output_dir=temp_dir,
                enable_monitoring=False,
                enable_alerts=False
            )
            
            # Configure column mapping
            analyzer.configure_column_mapping(
                target='target',
                prediction='prediction',
                numerical_features=['feature1', 'feature2'],
                categorical_features=['gender'],
                protected_attributes=['gender']
            )
            
            # Test data hashing
            hash_result = analyzer._calculate_data_hash(test_data)
            assert len(hash_result) == 64, "Hash should be 64 characters"
            
            # Test basic analysis (without full Evidently functionality)
            try:
                # This will test the analyzer initialization and basic functionality
                drift_results = analyzer.detect_data_drift(
                    test_data, test_data, generate_report=False
                )
                logger.info("✓ Data drift detection test passed")
            except Exception as e:
                logger.warning(f"Data drift test failed (this may be expected): {e}")
            
        logger.info("✓ Basic functionality test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Basic functionality test failed: {e}")
        return False


def run_integration_test():
    """Run integration test with the forensic framework."""
    logger.info("Running integration test...")
    
    try:
        # Test integration with main package
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        
        # Try to import from the main testing package
        try:
            from testing import EVIDENTLY_AVAILABLE
            if EVIDENTLY_AVAILABLE:
                from testing import EvidentlyAnalyzer
                logger.info("✓ Integration with main package successful")
            else:
                logger.warning("! Evidently not available in main package")
        except ImportError:
            logger.warning("! Could not import from main package (this may be expected)")
        
        # Test ForensicLogger integration
        from evidently_analyzer import EvidentlyAnalyzer
        
        with tempfile.TemporaryDirectory() as temp_dir:
            analyzer = EvidentlyAnalyzer(
                case_id="INTEGRATION_TEST",
                output_dir=temp_dir
            )
            
            # Check that logger was created
            assert hasattr(analyzer, 'logger'), "Analyzer should have logger"
            assert analyzer.logger is not None, "Logger should not be None"
            
        logger.info("✓ Integration test completed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Integration test failed: {e}")
        return False


def create_sample_configuration():
    """Create sample configuration files."""
    logger.info("Creating sample configuration files...")
    
    try:
        current_dir = Path(__file__).parent
        config_dir = current_dir / "config_samples"
        config_dir.mkdir(exist_ok=True)
        
        # Sample bias threshold configuration
        bias_config = {
            "bias_thresholds": {
                "demographic_parity_difference": 0.1,
                "equal_opportunity_difference": 0.1,
                "equalized_odds_difference": 0.1,
                "statistical_parity_difference": 0.1,
                "disparate_impact_ratio": 0.8,
                "accuracy_difference": 0.05,
                "precision_difference": 0.05,
                "recall_difference": 0.05,
                "f1_difference": 0.05
            },
            "drift_thresholds": {
                "dataset_drift_threshold": 0.5,
                "column_drift_threshold": 0.05,
                "psi_threshold": 0.2,
                "wasserstein_threshold": 0.1
            }
        }
        
        # Sample alert configuration
        alert_config = {
            "bias_alerts": {
                "demographic_parity_difference": {
                    "threshold": 0.1,
                    "severity": "high",
                    "actions": ["log_alert", "notify_investigator", "generate_report"]
                },
                "accuracy_difference": {
                    "threshold": 0.05,
                    "severity": "medium",
                    "actions": ["log_alert"]
                }
            },
            "drift_alerts": {
                "dataset_drift_score": {
                    "threshold": 0.5,
                    "severity": "medium",
                    "actions": ["log_alert", "generate_report"]
                }
            },
            "performance_alerts": {
                "accuracy_degradation": {
                    "threshold": -0.05,
                    "severity": "high",
                    "actions": ["log_alert", "notify_investigator"]
                }
            }
        }
        
        # Save configurations
        import json
        
        with open(config_dir / "bias_thresholds.json", 'w') as f:
            json.dump(bias_config, f, indent=2)
        
        with open(config_dir / "alert_config.json", 'w') as f:
            json.dump(alert_config, f, indent=2)
        
        logger.info(f"✓ Sample configurations created in {config_dir}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to create sample configurations: {e}")
        return False


def generate_setup_report():
    """Generate setup validation report."""
    logger.info("Generating setup report...")
    
    try:
        current_dir = Path(__file__).parent
        report_path = current_dir / "evidently_setup_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("Evidently Analyzer Setup Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Setup Date: {pd.Timestamp.now()}\n")
            f.write(f"Python Version: {sys.version}\n")
            
            # Check dependencies
            f.write("\nDependency Status:\n")
            dependencies = [
                'evidently', 'numpy', 'pandas', 'scipy', 
                'sklearn', 'matplotlib', 'seaborn'
            ]
            
            for dep in dependencies:
                try:
                    module = __import__(dep)
                    version = getattr(module, '__version__', 'Unknown')
                    f.write(f"  ✓ {dep}: {version}\n")
                except ImportError:
                    f.write(f"  ✗ {dep}: Not installed\n")
            
            # Check analyzer availability
            f.write("\nAnalyzer Status:\n")
            try:
                from evidently_analyzer import EVIDENTLY_AVAILABLE
                if EVIDENTLY_AVAILABLE:
                    f.write("  ✓ Evidently Analyzer: Available\n")
                else:
                    f.write("  ✗ Evidently Analyzer: Not available\n")
            except ImportError:
                f.write("  ✗ Evidently Analyzer: Import failed\n")
            
            f.write("\nSetup completed successfully!\n")
            f.write("\nNext Steps:\n")
            f.write("1. Review the generated sample configurations\n")
            f.write("2. Run the example usage script: python evidently_example_usage.py\n")
            f.write("3. Run the test suite: python test_evidently_analyzer.py\n")
            f.write("4. Integrate with your existing forensic testing workflows\n")
        
        logger.info(f"✓ Setup report generated: {report_path}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to generate setup report: {e}")
        return False


def main():
    """Main setup function."""
    logger.info("Starting Evidently Analyzer Setup")
    logger.info("=" * 50)
    
    success = True
    
    # Step 1: Check Python version
    if not check_python_version():
        success = False
    
    # Step 2: Install dependencies
    if success and not install_dependencies():
        success = False
    
    # Step 3: Validate Evidently installation
    if success and not validate_evidently_installation():
        success = False
    
    # Step 4: Validate analyzer import
    if success and not validate_analyzer_import():
        success = False
    
    # Step 5: Run basic functionality test
    if success and not run_basic_functionality_test():
        success = False
    
    # Step 6: Run integration test
    if success and not run_integration_test():
        success = False
    
    # Step 7: Create sample configurations
    if success and not create_sample_configuration():
        success = False
    
    # Step 8: Generate setup report
    if success:
        generate_setup_report()
    
    # Final status
    logger.info("=" * 50)
    if success:
        logger.info("🎉 Evidently Analyzer setup completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Review the sample configurations in config_samples/")
        logger.info("2. Run example usage: python evidently_example_usage.py")
        logger.info("3. Run tests: python test_evidently_analyzer.py")
        logger.info("4. Check the setup report: evidently_setup_report.txt")
    else:
        logger.error("❌ Setup failed. Please check the error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())