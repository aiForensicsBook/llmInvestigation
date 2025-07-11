#!/usr/bin/env python3
"""
Validation Script for Forensic Testing Suite
===========================================

This script validates that all components of the forensic testing suite
are properly installed and configured. It performs basic functionality
tests without requiring external data.

Author: Forensic AI Testing Suite
Created: 2025-07-01
Purpose: Validation and health check of forensic testing capabilities
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

def validate_imports():
    """Validate that all components can be imported."""
    print("Validating imports...")
    
    try:
        # Core dependencies
        import numpy as np
        import pandas as pd
        import scipy.stats as stats
        from sklearn.metrics import accuracy_score
        print("  ✓ Core dependencies imported successfully")
        
        # Forensic testing components
        from bias_analyzer import BiasAnalyzer
        from performance_tester import PerformanceTester
        from automated_prompt_tester import AutomatedPromptTester
        from log_analyzer import LogAnalyzer
        from test_runner import TestRunner, TestConfiguration
        print("  ✓ All forensic testing components imported successfully")
        
        # Evidently analyzer (optional)
        try:
            from evidently_analyzer import EvidentlyAnalyzer, EVIDENTLY_AVAILABLE
            if EVIDENTLY_AVAILABLE:
                print("  ✓ Evidently analyzer available")
            else:
                print("  ⚠ Evidently analyzer not available (library not installed)")
        except ImportError:
            print("  ⚠ Evidently analyzer not available (module not found)")
        
        return True, None
        
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Unexpected error: {e}"


def validate_basic_functionality():
    """Validate basic functionality of each component."""
    print("Validating basic functionality...")
    
    try:
        # Create temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"  Using temporary directory: {temp_dir}")
            
            # Test BiasAnalyzer initialization
            from bias_analyzer import BiasAnalyzer
            bias_analyzer = BiasAnalyzer(temp_dir)
            print("  ✓ BiasAnalyzer initialized successfully")
            
            # Test PerformanceTester initialization
            from performance_tester import PerformanceTester
            performance_tester = PerformanceTester(temp_dir)
            print("  ✓ PerformanceTester initialized successfully")
            
            # Test AutomatedPromptTester initialization
            from automated_prompt_tester import AutomatedPromptTester
            def dummy_model(prompt):
                return "Test response"
            prompt_tester = AutomatedPromptTester(dummy_model, temp_dir)
            print("  ✓ AutomatedPromptTester initialized successfully")
            
            # Test LogAnalyzer initialization
            from log_analyzer import LogAnalyzer
            log_analyzer = LogAnalyzer(temp_dir)
            print("  ✓ LogAnalyzer initialized successfully")
            
            # Test TestRunner initialization
            from test_runner import TestRunner, TestConfiguration
            config = TestConfiguration(
                enable_bias_analysis=False,
                enable_performance_testing=False,
                enable_prompt_testing=False,
                enable_log_analysis=False,
                output_directory=temp_dir
            )
            test_runner = TestRunner(config)
            print("  ✓ TestRunner initialized successfully")
        
        return True, None
        
    except Exception as e:
        return False, f"Functionality validation error: {e}"


def validate_data_structures():
    """Validate that data structures can be created and used."""
    print("Validating data structures...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample data structure
        sample_data = pd.DataFrame({
            'candidate_id': [1, 2, 3, 4, 5],
            'gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
            'age': [25, 30, 35, 40, 45],
            'race': ['White', 'Black', 'Asian', 'Hispanic', 'White'],
            'education': ['Harvard', 'MIT', 'Stanford', 'State', 'Community'],
            'score': [0.8, 0.7, 0.9, 0.6, 0.5],
            'y_true': [1, 0, 1, 0, 1],
            'y_pred': [1, 1, 1, 0, 0],
            'group': ['Male', 'Female', 'Male', 'Female', 'Male']
        })
        
        print(f"  ✓ Sample data created with {len(sample_data)} rows")
        
        # Validate data integrity functions
        from test_runner import TestRunner, TestConfiguration
        import hashlib
        
        data_string = sample_data.to_string()
        data_hash = hashlib.sha256(data_string.encode()).hexdigest()
        print(f"  ✓ Data hashing works: {data_hash[:16]}...")
        
        return True, None
        
    except Exception as e:
        return False, f"Data structure validation error: {e}"


def validate_logging():
    """Validate that forensic logging works correctly."""
    print("Validating forensic logging...")
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Test forensic logging
            from bias_analyzer import ForensicLogger
            logger = ForensicLogger(temp_dir)
            
            # Test basic logging
            logger.logger.info("Test log message")
            print("  ✓ Basic logging works")
            
            # Test structured logging
            logger.log_data_integrity("test_data", "test_hash", 100)
            print("  ✓ Structured logging works")
            
            # Check if log file was created
            log_files = list(Path(temp_dir).glob("*.log"))
            if log_files:
                print(f"  ✓ Log file created: {log_files[0].name}")
            else:
                return False, "No log file was created"
        
        return True, None
        
    except Exception as e:
        return False, f"Logging validation error: {e}"


def validate_statistical_functions():
    """Validate that statistical functions work correctly."""
    print("Validating statistical functions...")
    
    try:
        import numpy as np
        from scipy import stats
        
        # Test basic statistical functions
        data1 = np.random.normal(0, 1, 100)
        data2 = np.random.normal(0.5, 1, 100)
        
        # Mann-Whitney U test
        statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        print(f"  ✓ Mann-Whitney U test: statistic={statistic:.2f}, p={p_value:.4f}")
        
        # Chi-square test
        observed = [10, 10, 16, 8]
        expected = [11, 11, 11, 11]
        chi2, p_chi2 = stats.chisquare(observed, expected)
        print(f"  ✓ Chi-square test: chi2={chi2:.2f}, p={p_chi2:.4f}")
        
        # Effect size calculation (Cohen's d)
        pooled_std = np.sqrt((np.var(data1) + np.var(data2)) / 2)
        cohens_d = (np.mean(data2) - np.mean(data1)) / pooled_std
        print(f"  ✓ Cohen's d calculation: d={cohens_d:.3f}")
        
        return True, None
        
    except Exception as e:
        return False, f"Statistical validation error: {e}"


def validate_report_generation():
    """Validate that reports can be generated."""
    print("Validating report generation...")
    
    try:
        import json
        import datetime
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a simple test report
            test_report = {
                "metadata": {
                    "report_generated": datetime.datetime.now().isoformat(),
                    "test_type": "validation"
                },
                "results": {
                    "bias_detected": False,
                    "tests_completed": 1
                }
            }
            
            report_path = Path(temp_dir) / "test_report.json"
            with open(report_path, 'w') as f:
                json.dump(test_report, f, indent=2)
            
            print(f"  ✓ Test report created: {report_path.name}")
            
            # Validate report can be read back
            with open(report_path, 'r') as f:
                loaded_report = json.load(f)
            
            if loaded_report["metadata"]["test_type"] == "validation":
                print("  ✓ Report read back successfully")
            else:
                return False, "Report content validation failed"
        
        return True, None
        
    except Exception as e:
        return False, f"Report generation validation error: {e}"


def run_validation_suite():
    """Run the complete validation suite."""
    print("FORENSIC TESTING SUITE VALIDATION")
    print("=" * 50)
    print()
    
    validations = [
        ("Import Validation", validate_imports),
        ("Basic Functionality", validate_basic_functionality),
        ("Data Structures", validate_data_structures),
        ("Forensic Logging", validate_logging),
        ("Statistical Functions", validate_statistical_functions),
        ("Report Generation", validate_report_generation)
    ]
    
    results = []
    
    for validation_name, validation_func in validations:
        print(f"{validation_name}:")
        try:
            success, error = validation_func()
            if success:
                print(f"  ✓ {validation_name} PASSED")
                results.append((validation_name, True, None))
            else:
                print(f"  ✗ {validation_name} FAILED: {error}")
                results.append((validation_name, False, error))
        except Exception as e:
            print(f"  ✗ {validation_name} FAILED: Unexpected error: {e}")
            results.append((validation_name, False, f"Unexpected error: {e}"))
        print()
    
    # Summary
    print("VALIDATION SUMMARY")
    print("-" * 30)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✓ ALL VALIDATIONS PASSED")
        print("\nThe forensic testing suite is ready for use!")
        print("You can now run the example_usage.py script to see demonstrations.")
    else:
        print("✗ SOME VALIDATIONS FAILED")
        print("\nFailed validations:")
        for name, success, error in results:
            if not success:
                print(f"  - {name}: {error}")
        print("\nPlease resolve these issues before using the forensic testing suite.")
    
    return passed == total


def main():
    """Main validation function."""
    try:
        # Add current directory to Python path for imports
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Run validation suite
        success = run_validation_suite()
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\nValidation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error during validation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()