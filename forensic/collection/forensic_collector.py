#!/usr/bin/env python3
"""
Forensic Collection Framework - Main Collector
===============================================

A comprehensive forensic evidence collection tool that maintains
chain of custody and preserves digital evidence integrity.

Author: Forensic Collection Framework
Version: 1.0
"""

import os
import sys
import json
import hashlib
import shutil
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import stat
import uuid
import numpy as np
import pandas as pd
from collections import Counter
import re
import socket
import platform

from chain_of_custody import ChainOfCustody
from metadata_extractor import MetadataExtractor
from evidence_validator import EvidenceValidator


class ForensicCollector:
    """Main forensic evidence collection class."""
    
    def __init__(self, case_id: str = None, investigator: str = None):
        """Initialize the forensic collector.
        
        Args:
            case_id: Unique case identifier
            investigator: Name of the investigating officer/analyst
        """
        self.case_id = case_id or f"CASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.investigator = investigator or "Unknown"
        self.collection_id = str(uuid.uuid4())
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize helper classes
        self.chain_of_custody = ChainOfCustody(self.case_id, self.investigator)
        self.metadata_extractor = MetadataExtractor()
        self.evidence_validator = EvidenceValidator()
        
        # Capture system information for forensic record
        self.system_info = self._capture_system_info()
        
        self.logger.info(f"Forensic collector initialized - Case: {self.case_id}, Collection: {self.collection_id}")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.FileHandler(f'forensic_collection_{self.case_id}.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger('ForensicCollector')
    
    def _capture_system_info(self) -> Dict[str, str]:
        """Capture system information for forensic documentation."""
        try:
            system_info = {
                'hostname': socket.gethostname(),
                'platform': platform.platform(),
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'collection_location': os.getcwd(),
                'collection_time': datetime.now().isoformat(),
                'timezone': datetime.now().astimezone().tzinfo.tzname(None)
            }
            return system_info
        except Exception as e:
            self.logger.warning(f"Could not capture all system info: {str(e)}")
            return {'error': str(e)}
    
    def calculate_hashes(self, file_path: str) -> Dict[str, str]:
        """Calculate MD5, SHA1, and SHA256 hashes for a file.
        
        Args:
            file_path: Path to the file to hash
            
        Returns:
            Dictionary containing hash values
        """
        hashes = {'md5': None, 'sha1': None, 'sha256': None}
        
        try:
            with open(file_path, 'rb') as f:
                # Read file in chunks to handle large files
                md5_hash = hashlib.md5()
                sha1_hash = hashlib.sha1()
                sha256_hash = hashlib.sha256()
                
                for chunk in iter(lambda: f.read(8192), b""):
                    md5_hash.update(chunk)
                    sha1_hash.update(chunk)
                    sha256_hash.update(chunk)
                
                hashes['md5'] = md5_hash.hexdigest()
                hashes['sha1'] = sha1_hash.hexdigest()
                hashes['sha256'] = sha256_hash.hexdigest()
                
            self.logger.debug(f"Calculated hashes for {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error calculating hashes for {file_path}: {str(e)}")
            
        return hashes
    
    def extract_mac_timestamps(self, file_path: str) -> Dict[str, str]:
        """Extract MAC (Modified, Accessed, Created) timestamps.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing timestamp information
        """
        timestamps = {}
        
        try:
            stat_info = os.stat(file_path)
            
            timestamps['modified'] = datetime.fromtimestamp(stat_info.st_mtime).isoformat()
            timestamps['accessed'] = datetime.fromtimestamp(stat_info.st_atime).isoformat()
            
            # Try to get creation time (Windows) or birth time (macOS/BSD)
            if hasattr(stat_info, 'st_birthtime'):
                timestamps['created'] = datetime.fromtimestamp(stat_info.st_birthtime).isoformat()
            elif hasattr(stat_info, 'st_ctime'):
                timestamps['created'] = datetime.fromtimestamp(stat_info.st_ctime).isoformat()
            else:
                timestamps['created'] = None
                
            # Additional metadata
            timestamps['size'] = stat_info.st_size
            timestamps['permissions'] = oct(stat.S_IMODE(stat_info.st_mode))
            timestamps['uid'] = stat_info.st_uid
            timestamps['gid'] = stat_info.st_gid
            
            self.logger.debug(f"Extracted timestamps for {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error extracting timestamps for {file_path}: {str(e)}")
            
        return timestamps
    
    def detect_synthetic_data(self, data_path: str, data_type: str = 'json') -> Dict[str, Any]:
        """Detect if data is synthetic based on statistical patterns.
        
        Args:
            data_path: Path to data file
            data_type: Type of data (json, csv, etc.)
            
        Returns:
            Dictionary containing synthetic data detection results
        """
        synthetic_indicators = {
            'is_synthetic': False,
            'confidence': 0.0,
            'indicators': [],
            'statistical_anomalies': [],
            'pattern_analysis': {}
        }
        
        try:
            # Load data based on type
            if data_type == 'json':
                with open(data_path, 'r') as f:
                    data = json.load(f)
            elif data_type == 'csv':
                data = pd.read_csv(data_path).to_dict('records')
            else:
                self.logger.warning(f"Unsupported data type for synthetic detection: {data_type}")
                return synthetic_indicators
            
            # Analyze patterns that indicate synthetic data
            if isinstance(data, list) and len(data) > 0:
                # Check for repetitive patterns
                pattern_scores = []
                
                # 1. Name pattern analysis (for resume data)
                if 'personal_info' in data[0] or 'name' in data[0]:
                    names = [d.get('personal_info', {}).get('name', '') or d.get('name', '') for d in data]
                    name_patterns = self._analyze_name_patterns(names)
                    pattern_scores.extend(name_patterns['scores'])
                    if name_patterns['is_synthetic']:
                        synthetic_indicators['indicators'].append('Repetitive name patterns detected')
                
                # 2. Email pattern analysis
                emails = []
                for d in data:
                    if 'email' in d:
                        emails.append(d['email'])
                    elif 'personal_info' in d and 'email' in d['personal_info']:
                        emails.append(d['personal_info']['email'])
                
                if emails:
                    email_patterns = self._analyze_email_patterns(emails)
                    pattern_scores.extend(email_patterns['scores'])
                    if email_patterns['is_synthetic']:
                        synthetic_indicators['indicators'].append('Uniform email patterns detected')
                
                # 3. Text diversity analysis
                text_fields = []
                for d in data:
                    if 'summary' in d:
                        text_fields.append(d['summary'])
                    if 'description' in d:
                        text_fields.append(d['description'])
                    if 'experience' in d:
                        for exp in d['experience']:
                            if 'description' in exp:
                                text_fields.append(exp['description'])
                
                if text_fields:
                    text_diversity = self._analyze_text_diversity(text_fields)
                    pattern_scores.append(text_diversity['score'])
                    if text_diversity['is_synthetic']:
                        synthetic_indicators['indicators'].append('Low text diversity detected')
                
                # 4. Temporal pattern analysis
                dates = self._extract_dates(data)
                if dates:
                    temporal_patterns = self._analyze_temporal_patterns(dates)
                    pattern_scores.append(temporal_patterns['score'])
                    if temporal_patterns['is_synthetic']:
                        synthetic_indicators['indicators'].append('Uniform temporal patterns detected')
                
                # Calculate overall synthetic confidence
                if pattern_scores:
                    avg_score = np.mean(pattern_scores)
                    synthetic_indicators['confidence'] = avg_score
                    synthetic_indicators['is_synthetic'] = avg_score > 0.7
                
                # Statistical analysis
                synthetic_indicators['statistical_anomalies'] = self._detect_statistical_anomalies(data)
                
            self.logger.info(f"Synthetic data detection completed: {synthetic_indicators['is_synthetic']} (confidence: {synthetic_indicators['confidence']:.2f})")
            
        except Exception as e:
            self.logger.error(f"Error in synthetic data detection: {str(e)}")
            synthetic_indicators['error'] = str(e)
        
        return synthetic_indicators
    
    def _analyze_name_patterns(self, names: List[str]) -> Dict[str, Any]:
        """Analyze name patterns for synthetic data indicators."""
        result = {'is_synthetic': False, 'scores': []}
        
        if not names:
            return result
        
        # Check for sequential patterns (e.g., John Doe 1, John Doe 2)
        sequential_pattern = re.compile(r'(.+?)\s*(\d+)$')
        sequential_count = sum(1 for name in names if sequential_pattern.match(name))
        sequential_ratio = sequential_count / len(names)
        result['scores'].append(sequential_ratio)
        
        # Check for limited name diversity
        unique_first_names = set()
        unique_last_names = set()
        for name in names:
            parts = name.split()
            if parts:
                unique_first_names.add(parts[0])
                if len(parts) > 1:
                    unique_last_names.add(parts[-1])
        
        diversity_score = 1 - (len(unique_first_names) / len(names))
        result['scores'].append(diversity_score)
        
        # Check for common synthetic patterns
        synthetic_patterns = ['test', 'sample', 'example', 'demo', 'synthetic']
        pattern_count = sum(1 for name in names if any(p in name.lower() for p in synthetic_patterns))
        pattern_ratio = pattern_count / len(names)
        result['scores'].append(pattern_ratio)
        
        result['is_synthetic'] = any(score > 0.5 for score in result['scores'])
        return result
    
    def _analyze_email_patterns(self, emails: List[str]) -> Dict[str, Any]:
        """Analyze email patterns for synthetic data indicators."""
        result = {'is_synthetic': False, 'scores': []}
        
        if not emails:
            return result
        
        # Check domain diversity
        domains = [email.split('@')[1] if '@' in email else '' for email in emails]
        unique_domains = set(domains)
        domain_diversity = 1 - (len(unique_domains) / len(emails))
        result['scores'].append(domain_diversity)
        
        # Check for sequential patterns in email addresses
        sequential_emails = sum(1 for email in emails if re.search(r'\d+@', email))
        sequential_ratio = sequential_emails / len(emails)
        result['scores'].append(sequential_ratio)
        
        # Check for test domains
        test_domains = ['example.com', 'test.com', 'demo.com', 'sample.com']
        test_domain_count = sum(1 for domain in domains if domain in test_domains)
        test_ratio = test_domain_count / len(emails)
        result['scores'].append(test_ratio)
        
        result['is_synthetic'] = any(score > 0.5 for score in result['scores'])
        return result
    
    def _analyze_text_diversity(self, texts: List[str]) -> Dict[str, Any]:
        """Analyze text diversity to detect synthetic patterns."""
        result = {'is_synthetic': False, 'score': 0.0}
        
        if not texts:
            return result
        
        # Calculate vocabulary diversity
        all_words = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
        
        if all_words:
            unique_words = len(set(all_words))
            total_words = len(all_words)
            diversity_ratio = unique_words / total_words
            
            # Synthetic data often has lower diversity
            result['score'] = 1 - diversity_ratio
            result['is_synthetic'] = result['score'] > 0.8
        
        return result
    
    def _extract_dates(self, data: List[Dict]) -> List[datetime]:
        """Extract dates from data for temporal analysis."""
        dates = []
        
        for item in data:
            # Look for common date fields
            date_fields = ['created_at', 'modified_at', 'date', 'timestamp', 'updated_at']
            for field in date_fields:
                if field in item:
                    try:
                        date_str = item[field]
                        if isinstance(date_str, str):
                            # Try to parse ISO format
                            date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            dates.append(date_obj)
                    except:
                        pass
        
        return dates
    
    def _analyze_temporal_patterns(self, dates: List[datetime]) -> Dict[str, Any]:
        """Analyze temporal patterns in dates."""
        result = {'is_synthetic': False, 'score': 0.0}
        
        if len(dates) < 2:
            return result
        
        # Sort dates
        sorted_dates = sorted(dates)
        
        # Calculate time differences
        time_diffs = [(sorted_dates[i+1] - sorted_dates[i]).total_seconds() 
                      for i in range(len(sorted_dates)-1)]
        
        if time_diffs:
            # Check for uniform spacing (synthetic indicator)
            std_dev = np.std(time_diffs)
            mean_diff = np.mean(time_diffs)
            
            if mean_diff > 0:
                coefficient_of_variation = std_dev / mean_diff
                # Lower CV indicates more uniform spacing
                result['score'] = 1 - coefficient_of_variation
                result['is_synthetic'] = result['score'] > 0.8
        
        return result
    
    def _detect_statistical_anomalies(self, data: List[Dict]) -> List[str]:
        """Detect statistical anomalies that might indicate synthetic data."""
        anomalies = []
        
        # Check for perfect distributions
        if len(data) > 10:
            # Analyze numeric fields
            numeric_fields = {}
            for item in data:
                for key, value in item.items():
                    if isinstance(value, (int, float)):
                        if key not in numeric_fields:
                            numeric_fields[key] = []
                        numeric_fields[key].append(value)
            
            for field, values in numeric_fields.items():
                if len(values) > 5:
                    # Check for uniform distribution
                    unique_values = set(values)
                    if len(unique_values) < len(values) / 2:
                        value_counts = Counter(values)
                        # Check if values are too evenly distributed
                        count_values = list(value_counts.values())
                        if len(set(count_values)) == 1:
                            anomalies.append(f"Perfectly uniform distribution in field '{field}'")
        
        return anomalies
    
    def extract_model_specifications(self, model_path: str) -> Dict[str, Any]:
        """Extract technical specifications from a saved model.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Dictionary containing model specifications
        """
        model_specs = {
            'file_info': {},
            'model_architecture': {},
            'training_details': {},
            'parameters': {},
            'metadata': {}
        }
        
        try:
            # Get file information
            if os.path.exists(model_path):
                stat_info = os.stat(model_path)
                model_specs['file_info'] = {
                    'path': model_path,
                    'size_bytes': stat_info.st_size,
                    'size_mb': stat_info.st_size / (1024 * 1024),
                    'created': datetime.fromtimestamp(stat_info.st_ctime).isoformat(),
                    'modified': datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                    'hash': self.calculate_hashes(model_path)
                }
                
                # Try to load model data
                if model_path.endswith('.json'):
                    with open(model_path, 'r') as f:
                        model_data = json.load(f)
                    
                    # Extract model specifications
                    if 'model_type' in model_data:
                        model_specs['model_architecture']['type'] = model_data['model_type']
                    
                    if 'vocabulary' in model_data:
                        model_specs['parameters']['vocabulary_size'] = len(model_data['vocabulary'])
                    
                    if 'idf_values' in model_data:
                        model_specs['parameters']['num_features'] = len(model_data['idf_values'])
                    
                    if 'training_config' in model_data:
                        model_specs['training_details'] = model_data['training_config']
                    
                    if 'metadata' in model_data:
                        model_specs['metadata'] = model_data['metadata']
                    
                    # Count total parameters (simplified for TF-IDF model)
                    total_params = 0
                    if 'vocabulary' in model_data:
                        total_params += len(model_data['vocabulary'])
                    if 'idf_values' in model_data:
                        total_params += len(model_data['idf_values'])
                    
                    model_specs['parameters']['total_parameters'] = total_params
                    model_specs['parameters']['trainable_parameters'] = total_params
                    
            self.logger.info(f"Model specifications extracted from {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error extracting model specifications: {str(e)}")
            model_specs['error'] = str(e)
        
        return model_specs
    
    def analyze_bias_distributions(self, data_path: str, protected_features: List[str] = None) -> Dict[str, Any]:
        """Analyze distributions of protected features for bias detection.
        
        Args:
            data_path: Path to data file
            protected_features: List of features to analyze (default: gender, age, race)
            
        Returns:
            Dictionary containing bias distribution analysis
        """
        if protected_features is None:
            protected_features = ['gender', 'age', 'race', 'ethnicity']
        
        bias_analysis = {
            'distributions': {},
            'imbalances': [],
            'statistical_tests': {}
        }
        
        try:
            # Load data
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list) and len(data) > 0:
                # Extract features
                for feature in protected_features:
                    feature_values = []
                    
                    for item in data:
                        # Look for feature in various locations
                        value = None
                        if feature in item:
                            value = item[feature]
                        elif 'personal_info' in item and feature in item['personal_info']:
                            value = item['personal_info'][feature]
                        elif 'demographics' in item and feature in item['demographics']:
                            value = item['demographics'][feature]
                        
                        if value is not None:
                            feature_values.append(value)
                    
                    if feature_values:
                        # Calculate distribution
                        value_counts = Counter(feature_values)
                        total_count = len(feature_values)
                        
                        distribution = {
                            'counts': dict(value_counts),
                            'percentages': {k: (v/total_count)*100 for k, v in value_counts.items()},
                            'total_samples': total_count,
                            'unique_values': len(value_counts)
                        }
                        
                        bias_analysis['distributions'][feature] = distribution
                        
                        # Check for imbalances
                        if len(value_counts) >= 2:
                            max_percent = max(distribution['percentages'].values())
                            min_percent = min(distribution['percentages'].values())
                            
                            if max_percent / min_percent > 4:  # 4:1 ratio threshold
                                bias_analysis['imbalances'].append({
                                    'feature': feature,
                                    'severity': 'high',
                                    'max_group': max(value_counts.items(), key=lambda x: x[1])[0],
                                    'min_group': min(value_counts.items(), key=lambda x: x[1])[0],
                                    'ratio': max_percent / min_percent
                                })
                
                # Gender-specific analysis
                if 'gender' in bias_analysis['distributions']:
                    gender_dist = bias_analysis['distributions']['gender']
                    if 'male' in gender_dist['counts'] and 'female' in gender_dist['counts']:
                        male_count = gender_dist['counts']['male']
                        female_count = gender_dist['counts']['female']
                        
                        # Chi-square test for gender balance
                        from scipy.stats import chisquare
                        observed = [male_count, female_count]
                        expected = [sum(observed)/2, sum(observed)/2]
                        chi2, p_value = chisquare(observed, expected)
                        
                        bias_analysis['statistical_tests']['gender_balance'] = {
                            'test': 'chi-square',
                            'statistic': chi2,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        }
            
            self.logger.info(f"Bias distribution analysis completed for {len(bias_analysis['distributions'])} features")
            
        except Exception as e:
            self.logger.error(f"Error in bias distribution analysis: {str(e)}")
            bias_analysis['error'] = str(e)
        
        return bias_analysis
    
    def collect_evidence(self, source_path: str, destination_path: str, 
                        description: str = "", preserve_structure: bool = True) -> Dict:
        """Collect forensic evidence from source to destination.
        
        Args:
            source_path: Path to source file or directory
            destination_path: Path to evidence storage location
            description: Description of the evidence
            preserve_structure: Whether to preserve directory structure
            
        Returns:
            Dictionary containing collection results
        """
        collection_start = datetime.now()
        evidence_items = []
        errors = []
        
        self.logger.info(f"Starting evidence collection from {source_path}")
        
        try:
            source_path_obj = Path(source_path)
            destination_path_obj = Path(destination_path)
            
            # Create destination directory if it doesn't exist
            destination_path_obj.mkdir(parents=True, exist_ok=True)
            
            if source_path_obj.is_file():
                # Single file collection
                evidence_item = self._collect_single_file(
                    source_path_obj, destination_path_obj, description
                )
                evidence_items.append(evidence_item)
                
            elif source_path_obj.is_dir():
                # Directory collection
                for root, dirs, files in os.walk(source_path):
                    for file_name in files:
                        source_file = Path(root) / file_name
                        
                        if preserve_structure:
                            # Preserve directory structure
                            rel_path = source_file.relative_to(source_path_obj)
                            dest_file = destination_path_obj / rel_path
                            dest_file.parent.mkdir(parents=True, exist_ok=True)
                        else:
                            # Flatten structure
                            dest_file = destination_path_obj / file_name
                        
                        try:
                            evidence_item = self._collect_single_file(
                                source_file, dest_file.parent, 
                                f"{description} - {file_name}"
                            )
                            evidence_items.append(evidence_item)
                        except Exception as e:
                            error_msg = f"Failed to collect {source_file}: {str(e)}"
                            self.logger.error(error_msg)
                            errors.append(error_msg)
            
            collection_end = datetime.now()
            
            # Create collection summary
            collection_summary = {
                'collection_id': self.collection_id,
                'case_id': self.case_id,
                'investigator': self.investigator,
                'collection_start': collection_start.isoformat(),
                'collection_end': collection_end.isoformat(),
                'duration_seconds': (collection_end - collection_start).total_seconds(),
                'source_path': str(source_path),
                'destination_path': str(destination_path),
                'description': description,
                'evidence_items': evidence_items,
                'total_items': len(evidence_items),
                'errors': errors,
                'error_count': len(errors),
                'system_info': self.system_info,
                'collection_location': {
                    'hostname': self.system_info.get('hostname', 'Unknown'),
                    'ip_address': socket.gethostbyname(socket.gethostname()),
                    'working_directory': os.getcwd(),
                    'timezone': self.system_info.get('timezone', 'Unknown')
                }
            }
            
            # Save collection report
            report_path = destination_path_obj / f"collection_report_{self.collection_id}.json"
            with open(report_path, 'w') as f:
                json.dump(collection_summary, f, indent=2)
            
            self.logger.info(f"Collection completed. {len(evidence_items)} items collected, {len(errors)} errors")
            
            return collection_summary
            
        except Exception as e:
            error_msg = f"Critical error during evidence collection: {str(e)}"
            self.logger.error(error_msg)
            raise Exception(error_msg)
    
    def _collect_single_file(self, source_file: Path, destination_dir: Path, 
                           description: str) -> Dict:
        """Collect a single file with full forensic documentation.
        
        Args:
            source_file: Source file path
            destination_dir: Destination directory
            description: Evidence description
            
        Returns:
            Dictionary containing evidence metadata
        """
        evidence_id = str(uuid.uuid4())
        collection_time = datetime.now()
        
        # Calculate original hashes
        original_hashes = self.calculate_hashes(str(source_file))
        
        # Extract timestamps and metadata
        timestamps = self.extract_mac_timestamps(str(source_file))
        
        # Extract comprehensive metadata
        metadata = self.metadata_extractor.extract_metadata(str(source_file))
        
        # Copy file to evidence storage
        destination_file = destination_dir / source_file.name
        shutil.copy2(str(source_file), str(destination_file))
        
        # Verify copy integrity
        copy_hashes = self.calculate_hashes(str(destination_file))
        integrity_verified = (original_hashes['sha256'] == copy_hashes['sha256'])
        
        if not integrity_verified:
            raise Exception(f"Integrity verification failed for {source_file}")
        
        # Create evidence record
        evidence_record = {
            'evidence_id': evidence_id,
            'collection_time': collection_time.isoformat(),
            'original_path': str(source_file),
            'evidence_path': str(destination_file),
            'description': description,
            'original_hashes': original_hashes,
            'copy_hashes': copy_hashes,
            'integrity_verified': integrity_verified,
            'timestamps': timestamps,
            'metadata': metadata,
            'file_size': source_file.stat().st_size,
            'collector': self.investigator,
            'collection_location': {
                'hostname': self.system_info.get('hostname', 'Unknown'),
                'platform': self.system_info.get('platform', 'Unknown'),
                'working_directory': os.getcwd()
            }
        }
        
        # Add to chain of custody
        self.chain_of_custody.add_evidence(evidence_record)
        
        # Register with validator for future integrity checks
        self.evidence_validator.register_evidence(evidence_record)
        
        self.logger.info(f"Successfully collected evidence: {source_file} -> {destination_file}")
        
        return evidence_record


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Forensic Evidence Collector')
    parser.add_argument('source', help='Source file or directory path')
    parser.add_argument('--destination', '-d', required=True, 
                       help='Destination evidence storage path')
    parser.add_argument('--case-id', '-c', help='Case identifier')
    parser.add_argument('--investigator', '-i', help='Investigator name')
    parser.add_argument('--description', help='Evidence description')
    parser.add_argument('--preserve-structure', action='store_true',
                       help='Preserve directory structure')
    
    args = parser.parse_args()
    
    # Interactive prompts for missing required information
    if not args.case_id:
        args.case_id = input("Enter case ID: ").strip()
    
    if not args.investigator:
        args.investigator = input("Enter investigator name: ").strip()
    
    if not args.description:
        args.description = input("Enter evidence description: ").strip()
    
    # Confirm destination path
    print(f"\nEvidence will be collected to: {args.destination}")
    confirm = input("Proceed with collection? (y/N): ").strip().lower()
    
    if confirm != 'y':
        print("Collection cancelled.")
        sys.exit(0)
    
    try:
        # Initialize collector
        collector = ForensicCollector(args.case_id, args.investigator)
        
        # Perform collection
        result = collector.collect_evidence(
            args.source,
            args.destination,
            args.description,
            args.preserve_structure
        )
        
        print(f"\nCollection completed successfully!")
        print(f"Collection ID: {result['collection_id']}")
        print(f"Items collected: {result['total_items']}")
        print(f"Errors: {result['error_count']}")
        print(f"Report saved to: {args.destination}/collection_report_{result['collection_id']}.json")
        
    except Exception as e:
        print(f"Collection failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()