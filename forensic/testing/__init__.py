"""
Forensic Testing Suite for Resume Screening LLM
==============================================

A comprehensive forensic testing suite for bias detection and analysis in 
resume screening AI systems. This package provides legal-grade documentation
and analysis capabilities suitable for forensic examination.

Components:
- BiasAnalyzer: Statistical bias detection and analysis
- PerformanceTester: Performance and fairness metrics evaluation  
- AutomatedPromptTester: Automated prompt testing for bias detection
- LogAnalyzer: System log analysis for bias patterns
- ShapAnalyzer: SHAP-based explainability and interpretability analysis
- TestRunner: Comprehensive test orchestration and reporting

Author: Forensic AI Testing Suite
Created: 2025-07-01
Purpose: Legal forensic analysis of AI bias in hiring systems
"""

__version__ = "1.0.0"
__author__ = "Forensic AI Testing Suite"
__email__ = "contact@forensic-ai-testing.com"
__license__ = "Proprietary - For Legal Forensic Use"

# Core components
from .bias_analyzer import BiasAnalyzer, BiasTestResult, ForensicLogger
from .performance_tester import PerformanceTester, PerformanceMetrics, FairnessMetrics
from .automated_prompt_tester import AutomatedPromptTester, PromptTestResult
from .log_analyzer import LogAnalyzer
from .shap_analyzer import ShapAnalyzer, ShapExplanation, ShapAnalysisResult, TFIDFShapExplainer
from .test_runner import TestRunner, TestConfiguration, TestResult, ForensicTestSuite

# Evidently AI integration (optional, requires evidently library)
try:
    # Temporarily disabled due to import issues
    # from .evidently_analyzer import (
    #     EvidentlyAnalyzer, 
    #     EvidentlyAnalysisResult, 
    #     BiasMonitoringAlert,
    #     create_evidently_test_suite,
    #     run_evidently_forensic_analysis
    # )
    EVIDENTLY_AVAILABLE = False
except ImportError:
    EVIDENTLY_AVAILABLE = False

__all__ = [
    # Main classes
    'BiasAnalyzer',
    'PerformanceTester', 
    'AutomatedPromptTester',
    'LogAnalyzer',
    'ShapAnalyzer',
    'TestRunner',
    
    # Configuration and result classes
    'TestConfiguration',
    'TestResult',
    'ForensicTestSuite',
    'BiasTestResult',
    'PerformanceMetrics',
    'FairnessMetrics',
    'PromptTestResult',
    'BiasPromptResult',
    'ShapExplanation',
    'ShapAnalysisResult',
    'TFIDFShapExplainer',
    
    # Utility classes
    'ForensicLogger',
    
    # Evidently integration flags
    'EVIDENTLY_AVAILABLE',
]

# Add Evidently classes to __all__ if available
if EVIDENTLY_AVAILABLE:
    __all__.extend([
        'EvidentlyAnalyzer',
        'EvidentlyAnalysisResult',
        'BiasMonitoringAlert',
        'create_evidently_test_suite',
        'run_evidently_forensic_analysis'
    ])

# Package metadata
FORENSIC_TESTING_SUITE_INFO = {
    "name": "Forensic Testing Suite for Resume Screening LLM",
    "version": __version__,
    "description": "Comprehensive bias detection and forensic analysis for AI hiring systems",
    "components": {
        "bias_analyzer": "Statistical bias detection across demographic groups",
        "performance_tester": "Performance and fairness metrics evaluation",
        "automated_prompt_tester": "Automated bias detection through prompt testing",
        "log_analyzer": "System log analysis for bias patterns and anomalies",
        "shap_analyzer": "SHAP-based explainability and interpretability analysis",
        "evidently_analyzer": "Advanced bias detection and monitoring using Evidently library",
        "test_runner": "Comprehensive test orchestration and reporting"
    },
    "features": [
        "Legal-grade documentation and chain of custody",
        "Comprehensive statistical analysis with significance testing",
        "Forensic integrity with cryptographic verification",
        "Multiple bias detection methodologies",
        "Advanced data drift detection and monitoring",
        "Real-time bias detection with automated alerts",
        "Interactive HTML reports and dashboards",
        "Model performance degradation monitoring",
        "Data quality assessment and anomaly detection",
        "Automated report generation for legal proceedings",
        "Configurable thresholds and parameters",
        "Parallel test execution capabilities",
        "Integration with existing AI/ML pipelines"
    ],
    "compliance": [
        "EEOC guidelines for employment testing",
        "GDPR data protection requirements", 
        "IEEE standards for AI system testing",
        "Legal forensic evidence standards",
        "Chain of custody maintenance",
        "Data integrity verification"
    ]
}

def get_package_info():
    """Return comprehensive package information."""
    return FORENSIC_TESTING_SUITE_INFO

def version_info():
    """Return version information."""
    return {
        "version": __version__,
        "author": __author__,
        "license": __license__
    }

# Validation function for imports
def validate_dependencies():
    """Validate that all required dependencies are available."""
    import importlib
    required_packages = [
        'numpy',
        'pandas', 
        'scipy',
        'sklearn',
        'matplotlib',
        'seaborn'
    ]
    
    optional_packages = {
        'evidently': 'Advanced bias detection and monitoring capabilities'
    }
    
    missing_packages = []
    for package in required_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        raise ImportError(
            f"Missing required packages: {', '.join(missing_packages)}. "
            f"Please install using: pip install {' '.join(missing_packages)}"
        )
    
    # Check optional packages and warn if missing
    missing_optional = []
    for package, description in optional_packages.items():
        try:
            importlib.import_module(package)
        except ImportError:
            missing_optional.append((package, description))
    
    if missing_optional:
        import warnings
        for package, description in missing_optional:
            warnings.warn(
                f"Optional package '{package}' not available. "
                f"Install with 'pip install {package}' for {description}.",
                ImportWarning
            )
    
    return True

# Automatically validate dependencies on import
try:
    validate_dependencies()
except ImportError as e:
    import warnings
    warnings.warn(f"Dependency validation failed: {e}", ImportWarning)

# Package initialization message
import logging
logger = logging.getLogger(__name__)
logger.info(f"Forensic Testing Suite v{__version__} initialized successfully")
logger.info("Ready for bias detection and forensic analysis")