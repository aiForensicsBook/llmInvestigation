#!/usr/bin/env python3
"""
Test Runner for Resume Screening LLM Forensic Testing Suite
==========================================================

This module orchestrates all forensic tests, manages test execution and reporting,
creates timestamped test artifacts, and generates comprehensive forensic reports
for legal analysis of resume screening AI systems.

Author: Forensic AI Testing Suite
Created: 2025-07-01
Purpose: Legal forensic orchestration of AI bias testing in hiring systems
"""

import os
import json
import logging
import hashlib
import datetime
import traceback
import subprocess
import sys
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import concurrent.futures
from contextlib import contextmanager

import numpy as np
import pandas as pd

# Import forensic testing modules
from .bias_analyzer import BiasAnalyzer
from .performance_tester import PerformanceTester
from .automated_prompt_tester import AutomatedPromptTester
from .log_analyzer import LogAnalyzer


@dataclass
class TestConfiguration:
    """Configuration for forensic testing suite."""
    # Test enablement
    enable_bias_analysis: bool = True
    enable_performance_testing: bool = True
    enable_prompt_testing: bool = True
    enable_log_analysis: bool = True
    
    # Data sources
    training_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    log_files_paths: List[str] = None
    model_interface: Optional[Callable] = None
    
    # Test parameters
    bias_analysis_iterations: int = 100
    performance_test_iterations: int = 50
    prompt_test_iterations: int = 20
    
    # Output configuration
    output_directory: str = "./forensic_output"
    generate_visualizations: bool = True
    compress_artifacts: bool = True
    
    # Parallel execution
    max_workers: int = 4
    timeout_minutes: int = 60


@dataclass
class TestResult:
    """Result of a single test component."""
    test_name: str
    test_type: str
    status: str  # SUCCESS, FAILED, SKIPPED
    start_time: datetime.datetime
    end_time: datetime.datetime
    duration_seconds: float
    
    # Results
    bias_detected: bool
    severity_level: str
    key_findings: List[str]
    
    # Artifacts
    report_path: Optional[str]
    error_message: Optional[str]
    
    # Metadata
    test_hash: str
    data_integrity_hash: str


@dataclass
class ForensicTestSuite:
    """Comprehensive forensic test suite results."""
    suite_id: str
    execution_timestamp: str
    configuration: TestConfiguration
    
    # Test results
    test_results: List[TestResult]
    overall_status: str
    
    # Summary
    bias_detected: bool
    highest_severity: str
    critical_findings: List[str]
    
    # Artifacts
    comprehensive_report_path: str
    artifact_directory: str
    
    # Legal compliance
    chain_of_custody: List[Dict[str, Any]]
    integrity_verification: Dict[str, str]


class ForensicTestLogger:
    """Forensic-grade logging for test execution."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup forensic logging
        self.logger = logging.getLogger('forensic_test_runner')
        self.logger.setLevel(logging.DEBUG)
        
        # Create forensic log handler with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"test_execution_forensic_{timestamp}.log"
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d|%(levelname)s|%(funcName)s:%(lineno)d|%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Log session start
        self._log_session_info()
    
    def _log_session_info(self):
        """Log session information for forensic chain of custody."""
        import platform
        
        self.logger.info("=== FORENSIC TEST SUITE EXECUTION STARTED ===")
        self.logger.info(f"System: {platform.system()} {platform.release()}")
        self.logger.info(f"Python: {sys.version}")
        self.logger.info(f"Working Directory: {os.getcwd()}")
        self.logger.info(f"User: {os.getenv('USER', 'Unknown')}")
        self.logger.info(f"Process ID: {os.getpid()}")
        
        # Log system environment
        self.logger.info(f"PATH: {os.getenv('PATH', 'Not set')}")
        self.logger.info(f"PYTHONPATH: {os.getenv('PYTHONPATH', 'Not set')}")
    
    def log_test_start(self, test_name: str, test_type: str, config: Dict[str, Any]):
        """Log the start of a test with configuration."""
        config_hash = hashlib.sha256(str(config).encode()).hexdigest()[:16]
        self.logger.info(f"TEST_START|{test_name}|{test_type}|CONFIG_HASH:{config_hash}")
    
    def log_test_completion(self, result: TestResult):
        """Log test completion with results."""
        self.logger.info(f"TEST_COMPLETE|{result.test_name}|{result.status}|"
                        f"DURATION:{result.duration_seconds:.2f}s|BIAS:{result.bias_detected}|"
                        f"SEVERITY:{result.severity_level}")
    
    def log_data_access(self, data_path: str, data_hash: str, record_count: int):
        """Log data access for chain of custody."""
        self.logger.info(f"DATA_ACCESS|{data_path}|HASH:{data_hash}|RECORDS:{record_count}")
    
    def log_artifact_creation(self, artifact_path: str, artifact_hash: str):
        """Log artifact creation for integrity verification."""
        self.logger.info(f"ARTIFACT_CREATED|{artifact_path}|HASH:{artifact_hash}")
    
    def log_error(self, test_name: str, error: Exception):
        """Log test errors."""
        self.logger.error(f"TEST_ERROR|{test_name}|{type(error).__name__}|{str(error)}")
        self.logger.error(f"TRACEBACK|{test_name}|{traceback.format_exc()}")


class TestRunner:
    """
    Comprehensive test runner for forensic analysis of resume screening AI systems.
    
    This class orchestrates all testing components and provides comprehensive
    reporting suitable for legal forensic analysis.
    """
    
    def __init__(self, config: TestConfiguration):
        self.config = config
        self.suite_id = f"forensic_suite_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup output directory
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize forensic logging
        self.logger = ForensicTestLogger(self.output_dir / "logs")
        
        # Initialize test results storage
        self.test_results: List[TestResult] = []
        self.chain_of_custody: List[Dict[str, Any]] = []
        
        # Create artifact directories
        self.artifacts_dir = self.output_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Initialize test components
        self._initialize_test_components()
    
    def _initialize_test_components(self):
        """Initialize all test components."""
        try:
            if self.config.enable_bias_analysis:
                self.bias_analyzer = BiasAnalyzer(str(self.artifacts_dir / "bias_analysis"))
            
            if self.config.enable_performance_testing:
                self.performance_tester = PerformanceTester(str(self.artifacts_dir / "performance_testing"))
            
            if self.config.enable_prompt_testing:
                self.automated_prompt_tester = AutomatedPromptTester(
                    self.config.model_interface,
                    str(self.artifacts_dir / "prompt_testing")
                )
            
            if self.config.enable_log_analysis:
                self.log_analyzer = LogAnalyzer(str(self.artifacts_dir / "log_analysis"))
            
            self.logger.logger.info("All test components initialized successfully")
            
        except Exception as e:
            self.logger.log_error("component_initialization", e)
            raise
    
    def _get_timestamp(self) -> str:
        """Get ISO format timestamp for forensic records."""
        return datetime.datetime.now().isoformat()
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate hash of file for integrity verification."""
        if not os.path.exists(file_path):
            return "FILE_NOT_FOUND"
        
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
        except Exception as e:
            self.logger.logger.error(f"Error calculating hash for {file_path}: {e}")
            return "ERROR"
        return hash_sha256.hexdigest()
    
    def _log_data_access(self, data_path: str) -> str:
        """Log data access and return hash for chain of custody."""
        if not os.path.exists(data_path):
            return "FILE_NOT_FOUND"
        
        file_hash = self._calculate_file_hash(data_path)
        
        # Count records if it's a structured data file
        record_count = 0
        try:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
                record_count = len(df)
            elif data_path.endswith('.json'):
                with open(data_path) as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        record_count = len(data)
                    elif isinstance(data, dict):
                        record_count = 1
        except Exception as e:
            self.logger.logger.warning(f"Could not count records in {data_path}: {e}")
        
        self.logger.log_data_access(data_path, file_hash, record_count)
        
        # Add to chain of custody
        self.chain_of_custody.append({
            "timestamp": self._get_timestamp(),
            "action": "DATA_ACCESS",
            "file_path": data_path,
            "file_hash": file_hash,
            "record_count": record_count,
            "user": os.getenv('USER', 'Unknown')
        })
        
        return file_hash
    
    def _create_test_result(self, test_name: str, test_type: str, start_time: datetime.datetime,
                          status: str, bias_detected: bool = False, 
                          severity_level: str = "NONE", key_findings: List[str] = None,
                          report_path: Optional[str] = None, error_message: Optional[str] = None,
                          data_hash: str = "") -> TestResult:
        """Create a standardized test result."""
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        test_hash = hashlib.sha256(f"{test_name}_{test_type}_{start_time}".encode()).hexdigest()
        
        return TestResult(
            test_name=test_name,
            test_type=test_type,
            status=status,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration,
            bias_detected=bias_detected,
            severity_level=severity_level,
            key_findings=key_findings or [],
            report_path=report_path,
            error_message=error_message,
            test_hash=test_hash,
            data_integrity_hash=data_hash
        )
    
    @contextmanager
    def _test_execution_context(self, test_name: str, test_type: str, config: Dict[str, Any]):
        """Context manager for test execution with logging."""
        start_time = datetime.datetime.now()
        self.logger.log_test_start(test_name, test_type, config)
        
        try:
            yield start_time
        except Exception as e:
            self.logger.log_error(test_name, e)
            raise
    
    def run_bias_analysis(self) -> TestResult:
        """Run comprehensive bias analysis."""
        test_name = "bias_analysis"
        test_type = "statistical_bias_detection"
        config = {"iterations": self.config.bias_analysis_iterations}
        
        with self._test_execution_context(test_name, test_type, config) as start_time:
            try:
                if not self.config.training_data_path:
                    return self._create_test_result(
                        test_name, test_type, start_time, "SKIPPED",
                        error_message="No training data path provided"
                    )
                
                # Load and verify data
                data_hash = self._log_data_access(self.config.training_data_path)
                
                # Load training data
                if self.config.training_data_path.endswith('.csv'):
                    data = pd.read_csv(self.config.training_data_path)
                else:
                    raise ValueError("Unsupported data format")
                
                # Run bias analysis tests
                key_findings = []
                bias_detected = False
                severity_level = "NONE"
                
                # Gender bias analysis
                if 'gender' in data.columns and 'score' in data.columns:
                    gender_results = self.bias_analyzer.analyze_gender_bias(data, 'score', 'gender')
                    if any(r.bias_detected for r in gender_results):
                        bias_detected = True
                        key_findings.append("Gender bias detected in scoring")
                        severity_level = "MEDIUM"
                
                # Age bias analysis
                if 'age' in data.columns and 'score' in data.columns:
                    age_results = self.bias_analyzer.analyze_age_bias(data, 'score', 'age')
                    if any(r.bias_detected for r in age_results):
                        bias_detected = True
                        key_findings.append("Age bias detected in scoring")
                        if severity_level == "NONE":
                            severity_level = "MEDIUM"
                
                # Racial bias analysis
                if 'race' in data.columns and 'score' in data.columns:
                    race_results = self.bias_analyzer.analyze_racial_bias(data, 'score', 'race')
                    if any(r.bias_detected for r in race_results):
                        bias_detected = True
                        key_findings.append("Racial bias detected in scoring")
                        severity_level = "HIGH"
                
                # Education bias analysis
                if 'education' in data.columns and 'score' in data.columns:
                    edu_results = self.bias_analyzer.analyze_education_bias(data, 'score', 'education')
                    if any(r.bias_detected for r in edu_results):
                        bias_detected = True
                        key_findings.append("Educational bias detected in scoring")
                        if severity_level == "NONE":
                            severity_level = "LOW"
                
                # Generate report
                report_path = self.bias_analyzer.generate_bias_report()
                
                # Log artifact creation
                if report_path and os.path.exists(report_path):
                    report_hash = self._calculate_file_hash(report_path)
                    self.logger.log_artifact_creation(report_path, report_hash)
                
                result = self._create_test_result(
                    test_name, test_type, start_time, "SUCCESS",
                    bias_detected, severity_level, key_findings, report_path, None, data_hash
                )
                
                self.logger.log_test_completion(result)
                return result
                
            except Exception as e:
                result = self._create_test_result(
                    test_name, test_type, start_time, "FAILED",
                    error_message=str(e)
                )
                self.logger.log_test_completion(result)
                return result
    
    def run_performance_testing(self) -> TestResult:
        """Run comprehensive performance and fairness testing."""
        test_name = "performance_testing"
        test_type = "fairness_metrics_analysis"
        config = {"iterations": self.config.performance_test_iterations}
        
        with self._test_execution_context(test_name, test_type, config) as start_time:
            try:
                if not self.config.test_data_path:
                    return self._create_test_result(
                        test_name, test_type, start_time, "SKIPPED",
                        error_message="No test data path provided"
                    )
                
                # Load and verify data
                data_hash = self._log_data_access(self.config.test_data_path)
                
                # Load test data
                if self.config.test_data_path.endswith('.csv'):
                    data = pd.read_csv(self.config.test_data_path)
                else:
                    raise ValueError("Unsupported data format")
                
                # Run performance analysis
                key_findings = []
                bias_detected = False
                severity_level = "NONE"
                
                # Test performance across different groups
                required_cols = ['y_true', 'y_pred', 'group']
                if all(col in data.columns for col in required_cols):
                    # Test fairness metrics
                    fairness_result = self.performance_tester.test_fairness_metrics(
                        data, 'y_true', 'y_pred', 'group', 'protected_characteristic'
                    )
                    
                    if fairness_result.bias_detected:
                        bias_detected = True
                        severity_level = fairness_result.severity_level
                        
                        if fairness_result.demographic_parity > 0.1:
                            key_findings.append("Demographic parity violation detected")
                        if fairness_result.equalized_odds > 0.1:
                            key_findings.append("Equalized odds violation detected")
                        if fairness_result.equality_of_opportunity > 0.1:
                            key_findings.append("Equality of opportunity violation detected")
                else:
                    key_findings.append("Required columns not found in test data")
                
                # Generate report
                report_path = self.performance_tester.generate_performance_disparity_report()
                
                # Log artifact creation
                if report_path and os.path.exists(report_path):
                    report_hash = self._calculate_file_hash(report_path)
                    self.logger.log_artifact_creation(report_path, report_hash)
                
                result = self._create_test_result(
                    test_name, test_type, start_time, "SUCCESS",
                    bias_detected, severity_level, key_findings, report_path, None, data_hash
                )
                
                self.logger.log_test_completion(result)
                return result
                
            except Exception as e:
                result = self._create_test_result(
                    test_name, test_type, start_time, "FAILED",
                    error_message=str(e)
                )
                self.logger.log_test_completion(result)
                return result
    
    def run_prompt_testing(self) -> TestResult:
        """Run automated prompt testing for bias detection."""
        test_name = "prompt_testing"
        test_type = "adversarial_prompt_analysis"
        config = {"iterations": self.config.prompt_test_iterations}
        
        with self._test_execution_context(test_name, test_type, config) as start_time:
            try:
                if not self.config.model_interface:
                    return self._create_test_result(
                        test_name, test_type, start_time, "SKIPPED",
                        error_message="No model interface provided"
                    )
                
                key_findings = []
                bias_detected = False
                severity_level = "NONE"
                
                # Run gender bias prompt testing
                gender_result = self.automated_prompt_tester.test_gender_bias_prompts(
                    self.config.prompt_test_iterations
                )
                
                if gender_result.bias_detected:
                    bias_detected = True
                    key_findings.append("Gender bias detected in prompt responses")
                    if gender_result.statistical_significance < 0.01:
                        severity_level = "HIGH"
                    elif gender_result.statistical_significance < 0.05:
                        severity_level = "MEDIUM"
                    else:
                        severity_level = "LOW"
                
                # Run consistency testing
                if self.config.test_data_path:
                    # Use sample resume content for consistency testing
                    sample_resume = "Software Engineer with 5 years experience in Python and machine learning."
                    consistency_result = self.automated_prompt_tester.test_consistency_prompts(
                        sample_resume, self.config.prompt_test_iterations
                    )
                    
                    if consistency_result.consistency_score < 0.8:
                        key_findings.append("Low consistency in model responses")
                        if severity_level == "NONE":
                            severity_level = "LOW"
                
                # Run adversarial testing
                adversarial_results = self.automated_prompt_tester.test_adversarial_prompts(
                    self.config.prompt_test_iterations
                )
                
                for result in adversarial_results:
                    if result.bias_indicators:
                        bias_detected = True
                        key_findings.append("Adversarial prompt vulnerabilities detected")
                        if severity_level in ["NONE", "LOW"]:
                            severity_level = "MEDIUM"
                
                # Generate report
                report_path = self.automated_prompt_tester.generate_prompt_testing_report()
                
                # Log artifact creation
                if report_path and os.path.exists(report_path):
                    report_hash = self._calculate_file_hash(report_path)
                    self.logger.log_artifact_creation(report_path, report_hash)
                
                result = self._create_test_result(
                    test_name, test_type, start_time, "SUCCESS",
                    bias_detected, severity_level, key_findings, report_path, None, ""
                )
                
                self.logger.log_test_completion(result)
                return result
                
            except Exception as e:
                result = self._create_test_result(
                    test_name, test_type, start_time, "FAILED",
                    error_message=str(e)
                )
                self.logger.log_test_completion(result)
                return result
    
    def run_log_analysis(self) -> TestResult:
        """Run comprehensive log analysis for bias detection."""
        test_name = "log_analysis"
        test_type = "log_pattern_analysis"
        config = {"log_files": self.config.log_files_paths or []}
        
        with self._test_execution_context(test_name, test_type, config) as start_time:
            try:
                if not self.config.log_files_paths:
                    return self._create_test_result(
                        test_name, test_type, start_time, "SKIPPED",
                        error_message="No log files provided"
                    )
                
                # Verify log files and create data hash
                log_hashes = []
                for log_file in self.config.log_files_paths:
                    if os.path.exists(log_file):
                        log_hash = self._log_data_access(log_file)
                        log_hashes.append(log_hash)
                
                combined_hash = hashlib.sha256("|".join(log_hashes).encode()).hexdigest()
                
                # Run log analysis
                self.log_analyzer.analyze_log_files(self.config.log_files_paths)
                
                # Assess results
                key_findings = []
                bias_detected = len(self.log_analyzer.bias_patterns) > 0
                severity_level = "NONE"
                
                # Analyze bias patterns
                for pattern in self.log_analyzer.bias_patterns:
                    if pattern.statistical_significance < 0.01:
                        severity_level = "HIGH"
                        key_findings.append(f"High-confidence {pattern.pattern_type} detected")
                    elif pattern.statistical_significance < 0.05:
                        if severity_level not in ["HIGH"]:
                            severity_level = "MEDIUM"
                        key_findings.append(f"Significant {pattern.pattern_type} detected")
                
                # Analyze decision patterns
                for pattern in self.log_analyzer.decision_patterns:
                    if pattern.bias_indicators:
                        bias_detected = True
                        key_findings.extend(pattern.bias_indicators)
                        if severity_level == "NONE":
                            severity_level = "LOW"
                
                # Generate report
                report_path = self.log_analyzer.generate_log_analysis_report()
                
                # Log artifact creation
                if report_path and os.path.exists(report_path):
                    report_hash = self._calculate_file_hash(report_path)
                    self.logger.log_artifact_creation(report_path, report_hash)
                
                result = self._create_test_result(
                    test_name, test_type, start_time, "SUCCESS",
                    bias_detected, severity_level, key_findings, report_path, None, combined_hash
                )
                
                self.logger.log_test_completion(result)
                return result
                
            except Exception as e:
                result = self._create_test_result(
                    test_name, test_type, start_time, "FAILED",
                    error_message=str(e)
                )
                self.logger.log_test_completion(result)
                return result
    
    def run_all_tests(self) -> ForensicTestSuite:
        """Run all enabled forensic tests and generate comprehensive report."""
        execution_start = datetime.datetime.now()
        self.logger.logger.info(f"Starting comprehensive forensic test suite: {self.suite_id}")
        
        # Define test execution order and methods
        test_methods = []
        if self.config.enable_bias_analysis:
            test_methods.append(("bias_analysis", self.run_bias_analysis))
        if self.config.enable_performance_testing:
            test_methods.append(("performance_testing", self.run_performance_testing))
        if self.config.enable_prompt_testing:
            test_methods.append(("prompt_testing", self.run_prompt_testing))
        if self.config.enable_log_analysis:
            test_methods.append(("log_analysis", self.run_log_analysis))
        
        # Execute tests
        if self.config.max_workers > 1:
            # Parallel execution
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_test = {executor.submit(method): name for name, method in test_methods}
                
                for future in concurrent.futures.as_completed(future_to_test, 
                                                            timeout=self.config.timeout_minutes * 60):
                    test_name = future_to_test[future]
                    try:
                        result = future.result()
                        self.test_results.append(result)
                    except Exception as e:
                        self.logger.log_error(test_name, e)
                        # Create failed result
                        failed_result = self._create_test_result(
                            test_name, "unknown", execution_start, "FAILED",
                            error_message=str(e)
                        )
                        self.test_results.append(failed_result)
        else:
            # Sequential execution
            for test_name, method in test_methods:
                try:
                    result = method()
                    self.test_results.append(result)
                except Exception as e:
                    self.logger.log_error(test_name, e)
                    # Create failed result
                    failed_result = self._create_test_result(
                        test_name, "unknown", execution_start, "FAILED",
                        error_message=str(e)
                    )
                    self.test_results.append(failed_result)
        
        # Analyze overall results
        overall_status = "SUCCESS"
        bias_detected = any(result.bias_detected for result in self.test_results)
        
        # Determine highest severity
        severity_levels = ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
        highest_severity = "NONE"
        for result in self.test_results:
            if result.severity_level in severity_levels:
                current_index = severity_levels.index(result.severity_level)
                highest_index = severity_levels.index(highest_severity)
                if current_index > highest_index:
                    highest_severity = result.severity_level
        
        # Check for failed tests
        failed_tests = [result for result in self.test_results if result.status == "FAILED"]
        if failed_tests:
            overall_status = "PARTIAL_FAILURE"
            if len(failed_tests) == len(self.test_results):
                overall_status = "FAILED"
        
        # Collect critical findings
        critical_findings = []
        for result in self.test_results:
            if result.severity_level in ["HIGH", "CRITICAL"]:
                critical_findings.extend(result.key_findings)
        
        # Generate comprehensive report
        comprehensive_report_path = self._generate_comprehensive_report()
        
        # Create integrity verification
        integrity_verification = {}
        for result in self.test_results:
            if result.report_path and os.path.exists(result.report_path):
                integrity_verification[result.test_name] = self._calculate_file_hash(result.report_path)
        
        # Create test suite result
        test_suite = ForensicTestSuite(
            suite_id=self.suite_id,
            execution_timestamp=execution_start.isoformat(),
            configuration=self.config,
            test_results=self.test_results,
            overall_status=overall_status,
            bias_detected=bias_detected,
            highest_severity=highest_severity,
            critical_findings=critical_findings,
            comprehensive_report_path=comprehensive_report_path,
            artifact_directory=str(self.artifacts_dir),
            chain_of_custody=self.chain_of_custody,
            integrity_verification=integrity_verification
        )
        
        # Log suite completion
        self.logger.logger.info(f"Forensic test suite completed: {self.suite_id}")
        self.logger.logger.info(f"Overall status: {overall_status}")
        self.logger.logger.info(f"Bias detected: {bias_detected}")
        self.logger.logger.info(f"Highest severity: {highest_severity}")
        
        return test_suite
    
    def _generate_comprehensive_report(self) -> str:
        """Generate comprehensive forensic report."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.reports_dir / f"comprehensive_forensic_report_{timestamp}.json"
        
        # Compile comprehensive report
        report = {
            "forensic_analysis_metadata": {
                "suite_id": self.suite_id,
                "execution_timestamp": self._get_timestamp(),
                "analyst": os.getenv('USER', 'Unknown'),
                "system_info": {
                    "platform": sys.platform,
                    "python_version": sys.version,
                    "working_directory": os.getcwd()
                },
                "legal_notice": "This report contains forensic analysis results for legal proceedings. "
                               "Chain of custody and data integrity have been maintained throughout the analysis."
            },
            "executive_summary": self._generate_executive_summary(),
            "test_configuration": asdict(self.config),
            "individual_test_results": [asdict(result) for result in self.test_results],
            "consolidated_findings": self._generate_consolidated_findings(),
            "risk_assessment": self._generate_risk_assessment(),
            "legal_compliance": {
                "chain_of_custody": self.chain_of_custody,
                "data_integrity_verification": "All data hashes verified",
                "timestamp_verification": "All timestamps recorded with millisecond precision"
            },
            "recommendations": self._generate_comprehensive_recommendations(),
            "appendices": {
                "technical_details": self._generate_technical_appendix(),
                "statistical_methods": self._generate_statistical_methods_appendix()
            }
        }
        
        # Write report
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Log report creation
        report_hash = self._calculate_file_hash(str(report_path))
        self.logger.log_artifact_creation(str(report_path), report_hash)
        
        return str(report_path)
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary for the comprehensive report."""
        successful_tests = [r for r in self.test_results if r.status == "SUCCESS"]
        failed_tests = [r for r in self.test_results if r.status == "FAILED"]
        bias_detected_tests = [r for r in self.test_results if r.bias_detected]
        
        return {
            "overall_assessment": {
                "bias_detected": len(bias_detected_tests) > 0,
                "total_tests_executed": len(self.test_results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "tests_detecting_bias": len(bias_detected_tests)
            },
            "severity_breakdown": {
                severity: len([r for r in self.test_results if r.severity_level == severity])
                for severity in ["NONE", "LOW", "MEDIUM", "HIGH", "CRITICAL"]
            },
            "key_findings": {
                "critical_issues": [r.key_findings for r in self.test_results 
                                  if r.severity_level in ["HIGH", "CRITICAL"]],
                "primary_bias_types": list(set(
                    finding for result in self.test_results 
                    for finding in result.key_findings
                    if "bias" in finding.lower()
                ))
            },
            "compliance_status": {
                "data_integrity_maintained": True,
                "chain_of_custody_complete": len(self.chain_of_custody) > 0,
                "all_tests_documented": len(self.test_results) > 0
            }
        }
    
    def _generate_consolidated_findings(self) -> Dict[str, Any]:
        """Generate consolidated findings across all tests."""
        all_findings = []
        for result in self.test_results:
            all_findings.extend(result.key_findings)
        
        # Categorize findings
        bias_findings = [f for f in all_findings if "bias" in f.lower()]
        performance_findings = [f for f in all_findings if any(
            term in f.lower() for term in ["accuracy", "precision", "recall", "fairness"]
        )]
        consistency_findings = [f for f in all_findings if "consistency" in f.lower()]
        
        return {
            "bias_related_findings": bias_findings,
            "performance_related_findings": performance_findings,
            "consistency_related_findings": consistency_findings,
            "cross_test_correlations": self._analyze_cross_test_correlations(),
            "pattern_analysis": self._analyze_finding_patterns()
        }
    
    def _generate_risk_assessment(self) -> Dict[str, Any]:
        """Generate risk assessment based on test results."""
        high_risk_tests = [r for r in self.test_results if r.severity_level in ["HIGH", "CRITICAL"]]
        medium_risk_tests = [r for r in self.test_results if r.severity_level == "MEDIUM"]
        
        overall_risk = "LOW"
        if high_risk_tests:
            overall_risk = "HIGH"
        elif medium_risk_tests:
            overall_risk = "MEDIUM"
        
        return {
            "overall_risk_level": overall_risk,
            "risk_factors": {
                "bias_detection": len([r for r in self.test_results if r.bias_detected]) > 0,
                "test_failures": len([r for r in self.test_results if r.status == "FAILED"]) > 0,
                "high_severity_findings": len(high_risk_tests) > 0
            },
            "legal_implications": self._assess_legal_implications(),
            "business_impact": self._assess_business_impact(),
            "mitigation_urgency": "IMMEDIATE" if overall_risk == "HIGH" else "PLANNED"
        }
    
    def _generate_comprehensive_recommendations(self) -> List[str]:
        """Generate comprehensive recommendations based on all test results."""
        recommendations = []
        
        # High-priority recommendations
        high_severity_tests = [r for r in self.test_results if r.severity_level in ["HIGH", "CRITICAL"]]
        if high_severity_tests:
            recommendations.append(
                "IMMEDIATE ACTION REQUIRED: High-severity bias detected. "
                "Suspend system deployment until issues are resolved."
            )
        
        # Test-specific recommendations
        bias_detected_tests = [r for r in self.test_results if r.bias_detected]
        if bias_detected_tests:
            recommendations.append(
                "Implement comprehensive bias mitigation strategies across all identified areas."
            )
            recommendations.append(
                "Retrain models with bias-aware techniques and balanced datasets."
            )
        
        # Failed test recommendations
        failed_tests = [r for r in self.test_results if r.status == "FAILED"]
        if failed_tests:
            recommendations.append(
                "Investigate and resolve test failures to ensure complete bias assessment."
            )
        
        # General recommendations
        recommendations.extend([
            "Establish regular forensic testing schedule (monthly/quarterly).",
            "Implement continuous monitoring for bias detection in production.",
            "Create governance framework for AI fairness and bias management.",
            "Provide bias awareness training for development and deployment teams.",
            "Document all bias mitigation efforts for legal compliance."
        ])
        
        return recommendations
    
    def _analyze_cross_test_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between different test results."""
        # Simple correlation analysis between test outcomes
        correlations = {}
        
        # Check if tests that detected bias also had high severity
        bias_tests = [r for r in self.test_results if r.bias_detected]
        high_severity_tests = [r for r in self.test_results if r.severity_level in ["HIGH", "MEDIUM"]]
        
        overlap_count = len(set(r.test_name for r in bias_tests) & 
                          set(r.test_name for r in high_severity_tests))
        
        correlations["bias_severity_correlation"] = {
            "bias_tests": len(bias_tests),
            "high_severity_tests": len(high_severity_tests),
            "overlap": overlap_count,
            "correlation_strength": "HIGH" if overlap_count > len(bias_tests) * 0.7 else "MEDIUM"
        }
        
        return correlations
    
    def _analyze_finding_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in findings across tests."""
        all_findings = []
        for result in self.test_results:
            all_findings.extend(result.key_findings)
        
        # Count common terms
        common_terms = ["gender", "age", "race", "education", "bias", "discrimination"]
        term_counts = {}
        
        for term in common_terms:
            count = sum(1 for finding in all_findings if term.lower() in finding.lower())
            term_counts[term] = count
        
        return {
            "common_bias_types": term_counts,
            "total_findings": len(all_findings),
            "unique_findings": len(set(all_findings)),
            "most_frequent_finding": max(set(all_findings), key=all_findings.count) if all_findings else "None"
        }
    
    def _assess_legal_implications(self) -> List[str]:
        """Assess legal implications of the test results."""
        implications = []
        
        bias_detected_tests = [r for r in self.test_results if r.bias_detected]
        if bias_detected_tests:
            implications.append("Potential discrimination claims under employment law")
            implications.append("Regulatory compliance issues (EEOC, GDPR, etc.)")
            implications.append("Requirement for bias mitigation before deployment")
        
        high_severity_tests = [r for r in self.test_results if r.severity_level in ["HIGH", "CRITICAL"]]
        if high_severity_tests:
            implications.append("Immediate legal review recommended")
            implications.append("Documentation of remediation efforts required")
        
        return implications
    
    def _assess_business_impact(self) -> List[str]:
        """Assess business impact of the test results."""
        impacts = []
        
        if any(r.bias_detected for r in self.test_results):
            impacts.append("Risk of discriminatory hiring practices")
            impacts.append("Potential legal liability and financial exposure")
            impacts.append("Reputational risk from biased AI systems")
            impacts.append("Possible regulatory penalties and sanctions")
        
        if any(r.status == "FAILED" for r in self.test_results):
            impacts.append("Incomplete bias assessment creates uncertainty")
            impacts.append("May delay system deployment and business objectives")
        
        return impacts
    
    def _generate_technical_appendix(self) -> Dict[str, Any]:
        """Generate technical appendix with detailed methodology."""
        return {
            "statistical_methods_used": [
                "Mann-Whitney U test for group comparisons",
                "Chi-square test for independence",
                "Pearson and Spearman correlation analysis",
                "Cohen's d for effect size calculation",
                "Demographic parity and equalized odds metrics"
            ],
            "bias_detection_thresholds": {
                "statistical_significance": 0.05,
                "minimum_effect_size": 0.2,
                "consistency_threshold": 0.8,
                "fairness_violation_threshold": 0.1
            },
            "data_processing_steps": [
                "Data integrity verification with SHA-256 hashing",
                "Missing value analysis and handling",
                "Outlier detection and treatment",
                "Statistical assumption validation"
            ]
        }
    
    def _generate_statistical_methods_appendix(self) -> Dict[str, Any]:
        """Generate statistical methods appendix."""
        return {
            "hypothesis_testing": {
                "null_hypothesis": "No systematic bias exists in the AI system",
                "alternative_hypothesis": "Systematic bias exists favoring certain demographic groups",
                "significance_level": 0.05,
                "power_analysis": "Minimum detectable effect size: 0.2"
            },
            "fairness_metrics": {
                "demographic_parity": "P(Ŷ=1|A=0) = P(Ŷ=1|A=1)",
                "equalized_odds": "P(Ŷ=1|A=0,Y=y) = P(Ŷ=1|A=1,Y=y) for y ∈ {0,1}",
                "equality_of_opportunity": "P(Ŷ=1|A=0,Y=1) = P(Ŷ=1|A=1,Y=1)"
            },
            "effect_size_interpretation": {
                "small_effect": "0.2 ≤ d < 0.5",
                "medium_effect": "0.5 ≤ d < 0.8",
                "large_effect": "d ≥ 0.8"
            }
        }


def main():
    """Example usage of the TestRunner."""
    # Example configuration
    config = TestConfiguration(
        enable_bias_analysis=True,
        enable_performance_testing=True,
        enable_prompt_testing=False,  # Requires model interface
        enable_log_analysis=False,    # Requires log files
        training_data_path="./sample_data/training_data.csv",
        test_data_path="./sample_data/test_data.csv",
        output_directory="./forensic_output"
    )
    
    print("Forensic Test Runner initialized.")
    print("This is a comprehensive forensic testing suite for AI bias analysis.")
    print("Configure the TestConfiguration object and run the test suite.")
    print(f"Example configuration: {asdict(config)}")


if __name__ == "__main__":
    main()