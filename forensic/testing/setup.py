#!/usr/bin/env python3
"""
Setup Script for Forensic Testing Suite
======================================

This setup script helps install and configure the forensic testing suite
for resume screening AI bias analysis.

Author: Forensic AI Testing Suite
Created: 2025-07-01
Purpose: Installation and configuration of forensic testing capabilities
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("Checking Python version...")
    
    if sys.version_info < (3, 7):
        print(f"  ✗ Python {sys.version_info.major}.{sys.version_info.minor} is not supported")
        print("  Minimum required version: Python 3.7")
        return False
    
    print(f"  ✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} is compatible")
    return True

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("  ✗ requirements.txt not found")
        return False
    
    try:
        # Install dependencies
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("  ✓ Dependencies installed successfully")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Failed to install dependencies: {e}")
        return False
    except Exception as e:
        print(f"  ✗ Unexpected error installing dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("Creating directories...")
    
    directories = [
        "forensic_output",
        "forensic_output/logs",
        "forensic_output/reports", 
        "forensic_output/artifacts",
        "sample_data",
        "sample_logs"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        try:
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  ✓ Created directory: {directory}")
        except Exception as e:
            print(f"  ✗ Failed to create directory {directory}: {e}")
            return False
    
    return True

def validate_installation():
    """Validate that the installation was successful."""
    print("Validating installation...")
    
    try:
        # Try to import main components
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        from bias_analyzer import BiasAnalyzer
        from performance_tester import PerformanceTester
        from automated_prompt_tester import AutomatedPromptTester
        from log_analyzer import LogAnalyzer
        from test_runner import TestRunner, TestConfiguration
        
        print("  ✓ All components imported successfully")
        
        # Test basic functionality
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            config = TestConfiguration(output_directory=temp_dir)
            runner = TestRunner(config)
            print("  ✓ Test runner initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Validation failed: {e}")
        return False

def create_example_config():
    """Create an example configuration file."""
    print("Creating example configuration...")
    
    config_content = '''#!/usr/bin/env python3
"""
Example Configuration for Forensic Testing Suite
===============================================

This file shows how to configure the forensic testing suite for your
specific use case. Modify the paths and parameters as needed.
"""

from test_runner import TestConfiguration

# Example configuration for comprehensive testing
COMPREHENSIVE_CONFIG = TestConfiguration(
    # Enable/disable specific tests
    enable_bias_analysis=True,
    enable_performance_testing=True,
    enable_prompt_testing=True,  # Requires model_interface
    enable_log_analysis=True,    # Requires log_files_paths
    
    # Data sources - MODIFY THESE PATHS
    training_data_path="./sample_data/training_data.csv",
    test_data_path="./sample_data/test_data.csv",
    log_files_paths=["./sample_logs/model.log", "./sample_logs/decisions.log"],
    
    # Model interface - IMPLEMENT THIS FUNCTION
    model_interface=None,  # Replace with your model function
    
    # Test parameters
    bias_analysis_iterations=100,
    performance_test_iterations=50,
    prompt_test_iterations=20,
    
    # Output configuration
    output_directory="./forensic_output",
    generate_visualizations=True,
    compress_artifacts=True,
    
    # Execution parameters
    max_workers=4,
    timeout_minutes=60
)

# Example configuration for bias analysis only
BIAS_ONLY_CONFIG = TestConfiguration(
    enable_bias_analysis=True,
    enable_performance_testing=False,
    enable_prompt_testing=False,
    enable_log_analysis=False,
    training_data_path="./sample_data/training_data.csv",
    output_directory="./forensic_output"
)

# Example configuration for performance testing only
PERFORMANCE_ONLY_CONFIG = TestConfiguration(
    enable_bias_analysis=False,
    enable_performance_testing=True,
    enable_prompt_testing=False,
    enable_log_analysis=False,
    test_data_path="./sample_data/test_data.csv",
    output_directory="./forensic_output"
)

def your_model_interface(prompt: str) -> str:
    """
    Replace this function with your actual model interface.
    
    Args:
        prompt: Input prompt for the model
        
    Returns:
        Model response as a string
    """
    # Example implementation - replace with your model
    return "This is a placeholder response. Replace with your model."

# Update the configuration to use your model
COMPREHENSIVE_CONFIG.model_interface = your_model_interface
'''
    
    try:
        config_file = Path("example_config.py")
        with open(config_file, 'w') as f:
            f.write(config_content)
        print(f"  ✓ Example configuration created: {config_file}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to create example configuration: {e}")
        return False

def create_quick_start_guide():
    """Create a quick start guide."""
    print("Creating quick start guide...")
    
    guide_content = '''# Quick Start Guide - Forensic Testing Suite

## Installation Complete!

Your forensic testing suite has been installed successfully. Follow these steps to get started:

### 1. Validate Installation
```bash
python validate_suite.py
```

### 2. Run Examples
```bash
python example_usage.py
```

### 3. Configure for Your Data
1. Edit `example_config.py` with your data paths
2. Implement your model interface function
3. Update log file paths if using log analysis

### 4. Run Comprehensive Testing
```python
from test_runner import TestRunner
from example_config import COMPREHENSIVE_CONFIG

runner = TestRunner(COMPREHENSIVE_CONFIG)
results = runner.run_all_tests()

print(f"Bias detected: {results.bias_detected}")
print(f"Report: {results.comprehensive_report_path}")
```

### 5. Analyze Results
- Check the `forensic_output/reports/` directory for JSON reports
- Review the comprehensive report for legal documentation
- Examine individual component reports for detailed analysis

## Data Format Requirements

### Training/Test Data (CSV)
```csv
candidate_id,gender,age,race,education,score,y_true,y_pred,group
1,Female,28,White,Harvard,0.85,1,1,Female
2,Male,35,Black,State University,0.72,0,1,Male
```

### Log Files (Text)
```
2025-07-01 10:30:45|INFO|decision:accept|candidate_id:123|confidence:0.85
2025-07-01 10:31:02|INFO|gender_keywords detected|session_id:sess_789
```

## Key Features
- ✓ Legal-grade documentation
- ✓ Chain of custody maintenance
- ✓ Statistical significance testing
- ✓ Multiple bias detection methods
- ✓ Comprehensive reporting
- ✓ Data integrity verification

## Support
- Review README.md for detailed documentation
- Check example_usage.py for implementation examples
- Validate installation with validate_suite.py

Happy testing!
'''
    
    try:
        guide_file = Path("QUICK_START.md")
        with open(guide_file, 'w') as f:
            f.write(guide_content)
        print(f"  ✓ Quick start guide created: {guide_file}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to create quick start guide: {e}")
        return False

def main():
    """Main setup function."""
    print("FORENSIC TESTING SUITE SETUP")
    print("=" * 40)
    print("Setting up comprehensive bias analysis for AI systems...")
    print()
    
    steps = [
        ("Python Version Check", check_python_version),
        ("Dependency Installation", install_dependencies),
        ("Directory Creation", create_directories),
        ("Installation Validation", validate_installation),
        ("Example Configuration", create_example_config),
        ("Quick Start Guide", create_quick_start_guide)
    ]
    
    success_count = 0
    
    for step_name, step_func in steps:
        print(f"{step_name}:")
        try:
            if step_func():
                success_count += 1
            else:
                print(f"  ⚠️  {step_name} completed with issues")
        except Exception as e:
            print(f"  ✗ {step_name} failed: {e}")
        print()
    
    # Summary
    print("SETUP SUMMARY")
    print("-" * 20)
    print(f"Completed steps: {success_count}/{len(steps)}")
    
    if success_count == len(steps):
        print("✓ SETUP COMPLETED SUCCESSFULLY!")
        print()
        print("Next steps:")
        print("1. Run: python validate_suite.py")
        print("2. Run: python example_usage.py")
        print("3. Review: QUICK_START.md")
        print("4. Configure: example_config.py")
        print()
        print("Your forensic testing suite is ready for bias analysis!")
    else:
        print("⚠️  SETUP COMPLETED WITH ISSUES")
        print()
        print("Some steps failed. Please review the output above and:")
        print("1. Ensure you have Python 3.7+")
        print("2. Check internet connectivity for package installation")
        print("3. Verify file permissions for directory creation")
        print("4. Try running setup again")
    
    return success_count == len(steps)

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)