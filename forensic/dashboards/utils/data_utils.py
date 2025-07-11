"""
Data utilities for loading and validating evidence data.
"""

import pandas as pd
import numpy as np
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
import pickle
import sqlite3
import logging

from .error_handling import DataIntegrityError, ValidationError


def calculate_file_hash(file_path: Path, algorithm: str = 'sha256') -> str:
    """
    Calculate hash of a file for integrity verification.
    
    Args:
        file_path: Path to the file
        algorithm: Hash algorithm to use
    
    Returns:
        Hexadecimal hash string
    """
    hash_obj = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_obj.update(chunk)
    
    return hash_obj.hexdigest()


def verify_data_integrity(data: Any, expected_hash: str, 
                         algorithm: str = 'sha256') -> bool:
    """
    Verify data integrity using hash comparison.
    
    Args:
        data: Data to verify
        expected_hash: Expected hash value
        algorithm: Hash algorithm used
    
    Returns:
        True if integrity check passes
    
    Raises:
        DataIntegrityError: If integrity check fails
    """
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data, sort_keys=True)
    elif isinstance(data, pd.DataFrame):
        data_str = data.to_json(orient='records', sort_keys=True)
    else:
        data_str = str(data)
    
    hash_obj = hashlib.new(algorithm)
    hash_obj.update(data_str.encode())
    calculated_hash = hash_obj.hexdigest()
    
    if calculated_hash != expected_hash:
        raise DataIntegrityError(
            f"Data integrity check failed. Expected: {expected_hash}, "
            f"Calculated: {calculated_hash}"
        )
    
    return True


def load_json_with_validation(file_path: Path, schema: Optional[Dict] = None) -> Dict:
    """
    Load JSON file with optional schema validation.
    
    Args:
        file_path: Path to JSON file
        schema: Optional JSON schema for validation
    
    Returns:
        Loaded JSON data
    
    Raises:
        ValidationError: If schema validation fails
        DataIntegrityError: If file cannot be loaded
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Basic schema validation if provided
        if schema:
            validate_json_schema(data, schema)
        
        return data
        
    except json.JSONDecodeError as e:
        raise DataIntegrityError(f"Invalid JSON in {file_path}: {str(e)}")
    except FileNotFoundError:
        raise DataIntegrityError(f"File not found: {file_path}")


def validate_json_schema(data: Dict, schema: Dict) -> bool:
    """
    Basic JSON schema validation.
    
    Args:
        data: Data to validate
        schema: Schema definition
    
    Returns:
        True if validation passes
    
    Raises:
        ValidationError: If validation fails
    """
    # Check required fields
    required_fields = schema.get('required', [])
    for field in required_fields:
        if field not in data:
            raise ValidationError(f"Required field '{field}' missing")
    
    # Check field types
    properties = schema.get('properties', {})
    for field, field_schema in properties.items():
        if field in data:
            expected_type = field_schema.get('type')
            if expected_type:
                value = data[field]
                if not validate_json_type(value, expected_type):
                    raise ValidationError(
                        f"Field '{field}' should be of type '{expected_type}'"
                    )
    
    return True


def validate_json_type(value: Any, expected_type: str) -> bool:
    """Validate JSON value type."""
    type_mapping = {
        'string': str,
        'number': (int, float),
        'integer': int,
        'boolean': bool,
        'array': list,
        'object': dict,
        'null': type(None)
    }
    
    expected_python_type = type_mapping.get(expected_type)
    if expected_python_type is None:
        return True  # Unknown type, skip validation
    
    return isinstance(value, expected_python_type)


def load_evidence_data(evidence_dir: Path) -> Dict[str, Any]:
    """
    Load all evidence data from the forensic evidence directory.
    
    Args:
        evidence_dir: Path to evidence directory
    
    Returns:
        Dictionary containing all loaded evidence data
    """
    logger = logging.getLogger(__name__)
    evidence_data = {}
    
    if not evidence_dir.exists():
        logger.warning(f"Evidence directory does not exist: {evidence_dir}")
        return evidence_data
    
    # Load bias analysis results
    bias_results_file = evidence_dir / "bias_analysis_results.json"
    if bias_results_file.exists():
        try:
            evidence_data['bias_analysis'] = load_json_with_validation(bias_results_file)
            logger.info("Loaded bias analysis results")
        except Exception as e:
            logger.error(f"Failed to load bias analysis results: {e}")
    
    # Load SHAP analysis results
    shap_results_file = evidence_dir / "shap_analysis_results.json"
    if shap_results_file.exists():
        try:
            evidence_data['shap_analysis'] = load_json_with_validation(shap_results_file)
            logger.info("Loaded SHAP analysis results")
        except Exception as e:
            logger.error(f"Failed to load SHAP analysis results: {e}")
    
    # Load Evidently reports
    evidently_dir = evidence_dir / "evidently_reports"
    if evidently_dir.exists():
        evidence_data['evidently_reports'] = load_evidently_reports(evidently_dir)
        logger.info("Loaded Evidently reports")
    
    # Load performance metrics
    performance_file = evidence_dir / "performance_metrics.json"
    if performance_file.exists():
        try:
            evidence_data['performance_metrics'] = load_json_with_validation(performance_file)
            logger.info("Loaded performance metrics")
        except Exception as e:
            logger.error(f"Failed to load performance metrics: {e}")
    
    # Load audit logs
    audit_logs_file = evidence_dir / "audit_logs.json"
    if audit_logs_file.exists():
        try:
            evidence_data['audit_logs'] = load_json_with_validation(audit_logs_file)
            logger.info("Loaded audit logs")
        except Exception as e:
            logger.error(f"Failed to load audit logs: {e}")
    
    return evidence_data


def load_evidently_reports(evidently_dir: Path) -> Dict[str, Any]:
    """
    Load Evidently AI reports from directory.
    
    Args:
        evidently_dir: Path to Evidently reports directory
    
    Returns:
        Dictionary containing Evidently reports
    """
    reports = {}
    
    # Look for JSON report files
    for report_file in evidently_dir.glob("*.json"):
        try:
            with open(report_file, 'r') as f:
                report_data = json.load(f)
            reports[report_file.stem] = report_data
        except Exception as e:
            logging.getLogger(__name__).error(
                f"Failed to load Evidently report {report_file}: {e}"
            )
    
    return reports


def load_synthetic_data(data_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load synthetic data for testing and analysis.
    
    Args:
        data_dir: Path to data directory
    
    Returns:
        Dictionary containing synthetic datasets
    """
    synthetic_data = {}
    synthetic_dir = data_dir / "synthetic"
    
    if not synthetic_dir.exists():
        return synthetic_data
    
    # Load synthetic resumes
    resumes_file = synthetic_dir / "synthetic_resumes.json"
    if resumes_file.exists():
        try:
            with open(resumes_file, 'r') as f:
                resumes_data = json.load(f)
            synthetic_data['resumes'] = pd.DataFrame(resumes_data)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to load synthetic resumes: {e}")
    
    # Load synthetic job postings
    jobs_file = synthetic_dir / "synthetic_job_postings.json"
    if jobs_file.exists():
        try:
            with open(jobs_file, 'r') as f:
                jobs_data = json.load(f)
            synthetic_data['job_postings'] = pd.DataFrame(jobs_data)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to load synthetic job postings: {e}")
    
    # Load matched pairs
    pairs_file = synthetic_dir / "synthetic_matched_pairs.json"
    if pairs_file.exists():
        try:
            with open(pairs_file, 'r') as f:
                pairs_data = json.load(f)
            synthetic_data['matched_pairs'] = pd.DataFrame(pairs_data)
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to load matched pairs: {e}")
    
    return synthetic_data


def validate_data_integrity(data_dir: Path, evidence_dir: Path) -> Dict[str, bool]:
    """
    Comprehensive data integrity validation.
    
    Args:
        data_dir: Path to data directory
        evidence_dir: Path to evidence directory
    
    Returns:
        Dictionary with validation results for each component
    """
    results = {}
    logger = logging.getLogger(__name__)
    
    # Check synthetic data integrity
    try:
        synthetic_data = load_synthetic_data(data_dir)
        results['synthetic_data'] = validate_synthetic_data(synthetic_data)
    except Exception as e:
        logger.error(f"Synthetic data validation failed: {e}")
        results['synthetic_data'] = False
    
    # Check evidence data integrity
    try:
        evidence_data = load_evidence_data(evidence_dir)
        results['evidence_data'] = validate_evidence_data(evidence_data)
    except Exception as e:
        logger.error(f"Evidence data validation failed: {e}")
        results['evidence_data'] = False
    
    # Check for required files
    required_files = [
        data_dir / "synthetic" / "synthetic_resumes.json",
        data_dir / "synthetic" / "synthetic_job_postings.json",
    ]
    
    results['required_files'] = all(f.exists() for f in required_files)
    
    return results


def validate_synthetic_data(synthetic_data: Dict[str, pd.DataFrame]) -> bool:
    """
    Validate synthetic data structure and content.
    
    Args:
        synthetic_data: Dictionary of synthetic datasets
    
    Returns:
        True if validation passes
    """
    required_datasets = ['resumes', 'job_postings']
    
    for dataset_name in required_datasets:
        if dataset_name not in synthetic_data:
            logging.getLogger(__name__).error(f"Missing dataset: {dataset_name}")
            return False
        
        df = synthetic_data[dataset_name]
        if df.empty:
            logging.getLogger(__name__).error(f"Empty dataset: {dataset_name}")
            return False
    
    # Validate resume data structure
    if 'resumes' in synthetic_data:
        resumes_df = synthetic_data['resumes']
        required_resume_columns = ['id', 'content', 'skills', 'experience']
        
        for col in required_resume_columns:
            if col not in resumes_df.columns:
                logging.getLogger(__name__).error(f"Missing column in resumes: {col}")
                return False
    
    # Validate job postings structure
    if 'job_postings' in synthetic_data:
        jobs_df = synthetic_data['job_postings']
        required_job_columns = ['id', 'title', 'description', 'requirements']
        
        for col in required_job_columns:
            if col not in jobs_df.columns:
                logging.getLogger(__name__).error(f"Missing column in job_postings: {col}")
                return False
    
    return True


def validate_evidence_data(evidence_data: Dict[str, Any]) -> bool:
    """
    Validate evidence data structure and content.
    
    Args:
        evidence_data: Dictionary of evidence data
    
    Returns:
        True if validation passes
    """
    # Check if we have at least some evidence data
    if not evidence_data:
        logging.getLogger(__name__).warning("No evidence data found")
        return False
    
    # Validate bias analysis results
    if 'bias_analysis' in evidence_data:
        bias_data = evidence_data['bias_analysis']
        if not isinstance(bias_data, dict):
            logging.getLogger(__name__).error("Invalid bias analysis data structure")
            return False
    
    # Validate SHAP analysis results
    if 'shap_analysis' in evidence_data:
        shap_data = evidence_data['shap_analysis']
        if not isinstance(shap_data, dict):
            logging.getLogger(__name__).error("Invalid SHAP analysis data structure")
            return False
    
    return True


def create_data_summary(data: Union[pd.DataFrame, Dict, List]) -> Dict[str, Any]:
    """
    Create a summary of data for display purposes.
    
    Args:
        data: Data to summarize
    
    Returns:
        Dictionary containing data summary
    """
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_type': type(data).__name__
    }
    
    if isinstance(data, pd.DataFrame):
        summary.update({
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'null_counts': data.isnull().sum().to_dict(),
            'memory_usage': data.memory_usage(deep=True).sum()
        })
    elif isinstance(data, dict):
        summary.update({
            'keys': list(data.keys()),
            'total_items': len(data),
            'nested_structure': {k: type(v).__name__ for k, v in data.items()}
        })
    elif isinstance(data, list):
        summary.update({
            'length': len(data),
            'item_types': list(set(type(item).__name__ for item in data))
        })
    
    return summary