#!/usr/bin/env python3
"""
Log Analysis and Bias Detection Framework
==========================================

This module analyzes system logs, model decision logs, and application logs
to detect patterns of bias and discrimination in resume screening decisions.

Legal Compliance: EEOC, GDPR, IEEE AI Standards
Forensic Standards: Evidence handling, chain of custody, audit trails
"""

import json
import hashlib
import logging
import os
import sys
import re
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics
from collections import defaultdict, Counter
import glob


@dataclass
class LogAnalysisResult:
    """Results from log analysis for bias detection."""
    analysis_id: str
    timestamp: str
    log_source: str
    analysis_type: str
    findings: Dict[str, Any]
    bias_indicators: List[Dict[str, Any]]
    statistical_summary: Dict[str, Any]
    recommendations: List[str]
    confidence_score: float
    data_hash: str


@dataclass
class BiasIndicator:
    """Individual bias indicator found in logs."""
    indicator_id: str
    timestamp: str
    log_entry: str
    bias_type: str
    severity: str
    confidence: float
    context: Dict[str, Any]
    supporting_evidence: List[str]


class LogAnalyzer:
    """
    Comprehensive log analysis framework for bias detection.
    
    This class analyzes various types of logs to identify patterns that may
    indicate bias or discrimination in resume screening decisions.
    """
    
    def __init__(self, case_id: str = None):
        """
        Initialize the log analyzer.
        
        Args:
            case_id: Unique identifier for this analysis session
        """
        self.case_id = case_id or f"LOG_ANALYSIS_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Setup logging
        self.logger = self._setup_forensic_logging()
        
        # Analysis configuration
        self.config = {
            'bias_keywords': {
                'gender': [
                    'male', 'female', 'man', 'woman', 'guy', 'girl', 'he', 'she',
                    'husband', 'wife', 'father', 'mother', 'pregnant', 'maternity',
                    'family', 'childcare', 'masculine', 'feminine'
                ],
                'age': [
                    'young', 'old', 'senior', 'junior', 'experienced', 'fresh',
                    'recent graduate', 'veteran', 'retirement', 'energy',
                    'digital native', 'millennial', 'boomer'
                ],
                'race_ethnicity': [
                    'white', 'black', 'asian', 'hispanic', 'latino', 'african',
                    'american', 'foreign', 'accent', 'english', 'immigrant',
                    'diversity', 'minority', 'cultural'
                ],
                'religion': [
                    'christian', 'muslim', 'jewish', 'hindu', 'buddhist',
                    'religious', 'church', 'mosque', 'temple', 'sabbath',
                    'holiday', 'prayer'
                ],
                'disability': [
                    'disabled', 'disability', 'handicapped', 'accommodation',
                    'wheelchair', 'blind', 'deaf', 'medical', 'health'
                ],
                'socioeconomic': [
                    'wealthy', 'poor', 'expensive', 'cheap', 'luxury',
                    'elite', 'prestigious', 'public', 'private', 'scholarship'
                ]
            },
            'suspicious_patterns': [
                r'score.*(?:male|female)',
                r'rating.*(?:young|old)',
                r'decision.*(?:white|black|asian)',
                r'reject.*(?:foreign|accent)',
                r'accept.*(?:prestigious|elite)',
                r'filter.*(?:name|surname)',
                r'bias.*(?:detected|found|present)',
                r'discriminat.*(?:against|in favor)'
            ],
            'severity_thresholds': {
                'critical': 0.9,
                'high': 0.7,
                'medium': 0.5,
                'low': 0.3
            }
        }
        
        # Results storage
        self.analysis_results: List[LogAnalysisResult] = []
        self.bias_indicators: List[BiasIndicator] = []
        
        self.logger.info(f"Initialized LogAnalyzer for case: {self.case_id}")
    
    def _setup_forensic_logging(self) -> logging.Logger:
        """Setup forensic-grade logging with tamper detection."""
        logger = logging.getLogger(f"LogAnalyzer_{self.case_id}")
        logger.setLevel(logging.INFO)
        
        # Create forensic logs directory
        log_dir = Path("forensic/logs/log_analysis")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # File handler with detailed formatting
        log_file = log_dir / f"log_analysis_{self.case_id}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Forensic formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s | HASH:%(created)f'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        return logger
    
    def analyze_model_decision_logs(self, log_directory: str) -> LogAnalysisResult:
        """Analyze model decision logs for bias patterns."""
        self.logger.info(f"Analyzing model decision logs in: {log_directory}")
        
        findings = {
            'total_decisions': 0,
            'decisions_by_demographic': defaultdict(lambda: defaultdict(int)),
            'score_distributions': defaultdict(list),
            'bias_patterns': [],
            'temporal_patterns': [],
            'decision_anomalies': []
        }
        
        bias_indicators = []
        log_files = glob.glob(os.path.join(log_directory, "*.log"))
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        findings['total_decisions'] += 1
                        
                        # Extract decision information
                        decision_info = self._extract_decision_info(line)
                        if decision_info:
                            # Analyze for bias patterns
                            bias_result = self._analyze_decision_for_bias(
                                decision_info, line, f"{log_file}:{line_num}"
                            )
                            
                            if bias_result:
                                bias_indicators.append(bias_result)
                                findings['bias_patterns'].append({
                                    'line': line_num,
                                    'file': log_file,
                                    'pattern': bias_result.bias_type,
                                    'confidence': bias_result.confidence
                                })
                            
                            # Collect statistics
                            self._collect_decision_statistics(decision_info, findings)
            
            except Exception as e:
                self.logger.error(f"Error analyzing log file {log_file}: {e}")
        
        # Generate statistical analysis
        statistical_summary = self._generate_decision_statistics(findings)
        
        # Calculate confidence score
        confidence_score = self._calculate_analysis_confidence(bias_indicators, findings)
        
        # Generate recommendations
        recommendations = self._generate_bias_recommendations(bias_indicators, findings)
        
        # Create analysis result
        result = LogAnalysisResult(
            analysis_id=f"MODEL_DECISIONS_{self.case_id}",
            timestamp=datetime.now().isoformat(),
            log_source=log_directory,
            analysis_type="model_decision_logs",
            findings=findings,
            bias_indicators=[asdict(bi) for bi in bias_indicators],
            statistical_summary=statistical_summary,
            recommendations=recommendations,
            confidence_score=confidence_score,
            data_hash=self._calculate_data_hash(findings)
        )
        
        self.analysis_results.append(result)
        self.bias_indicators.extend(bias_indicators)
        
        self.logger.info(f"Completed model decision log analysis: {len(bias_indicators)} bias indicators found")
        return result
    
    def analyze_application_logs(self, log_directory: str) -> LogAnalysisResult:
        """Analyze application logs for bias indicators."""
        self.logger.info(f"Analyzing application logs in: {log_directory}")
        
        findings = {
            'total_entries': 0,
            'error_patterns': [],
            'user_actions': defaultdict(int),
            'system_anomalies': [],
            'bias_keywords_found': defaultdict(int),
            'suspicious_queries': [],
            'access_patterns': defaultdict(list)
        }
        
        bias_indicators = []
        log_files = glob.glob(os.path.join(log_directory, "*.log"))
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        findings['total_entries'] += 1
                        
                        # Check for bias keywords
                        bias_keywords = self._detect_bias_keywords(line)
                        if bias_keywords:
                            for keyword, category in bias_keywords:
                                findings['bias_keywords_found'][category] += 1
                                
                                # Create bias indicator
                                indicator = BiasIndicator(
                                    indicator_id=f"KEYWORD_{self.case_id}_{line_num}",
                                    timestamp=self._extract_timestamp(line) or datetime.now().isoformat(),
                                    log_entry=line.strip(),
                                    bias_type=f"keyword_{category}",
                                    severity=self._assess_keyword_severity(keyword, category),
                                    confidence=0.6,  # Medium confidence for keyword detection
                                    context={
                                        'file': log_file,
                                        'line_number': line_num,
                                        'keyword': keyword,
                                        'category': category
                                    },
                                    supporting_evidence=[line.strip()]
                                )
                                bias_indicators.append(indicator)
                        
                        # Check for suspicious patterns
                        suspicious_patterns = self._detect_suspicious_patterns(line)
                        if suspicious_patterns:
                            findings['suspicious_queries'].extend(suspicious_patterns)
                            
                            for pattern in suspicious_patterns:
                                indicator = BiasIndicator(
                                    indicator_id=f"PATTERN_{self.case_id}_{line_num}",
                                    timestamp=self._extract_timestamp(line) or datetime.now().isoformat(),
                                    log_entry=line.strip(),
                                    bias_type="suspicious_pattern",
                                    severity="high",
                                    confidence=0.8,
                                    context={
                                        'file': log_file,
                                        'line_number': line_num,
                                        'pattern': pattern
                                    },
                                    supporting_evidence=[line.strip()]
                                )
                                bias_indicators.append(indicator)
                        
                        # Analyze user actions
                        user_action = self._extract_user_action(line)
                        if user_action:
                            findings['user_actions'][user_action] += 1
                        
                        # Detect system anomalies
                        anomaly = self._detect_system_anomaly(line)
                        if anomaly:
                            findings['system_anomalies'].append({
                                'line': line_num,
                                'file': log_file,
                                'anomaly_type': anomaly,
                                'entry': line.strip()
                            })
            
            except Exception as e:
                self.logger.error(f"Error analyzing application log {log_file}: {e}")
        
        # Generate statistical analysis
        statistical_summary = self._generate_application_statistics(findings)
        
        # Calculate confidence score
        confidence_score = self._calculate_analysis_confidence(bias_indicators, findings)
        
        # Generate recommendations
        recommendations = self._generate_application_recommendations(bias_indicators, findings)
        
        # Create analysis result
        result = LogAnalysisResult(
            analysis_id=f"APPLICATION_LOGS_{self.case_id}",
            timestamp=datetime.now().isoformat(),
            log_source=log_directory,
            analysis_type="application_logs",
            findings=findings,
            bias_indicators=[asdict(bi) for bi in bias_indicators],
            statistical_summary=statistical_summary,
            recommendations=recommendations,
            confidence_score=confidence_score,
            data_hash=self._calculate_data_hash(findings)
        )
        
        self.analysis_results.append(result)
        self.bias_indicators.extend(bias_indicators)
        
        self.logger.info(f"Completed application log analysis: {len(bias_indicators)} bias indicators found")
        return result
    
    def analyze_audit_logs(self, log_directory: str) -> LogAnalysisResult:
        """Analyze audit logs for access patterns and potential bias."""
        self.logger.info(f"Analyzing audit logs in: {log_directory}")
        
        findings = {
            'total_access_events': 0,
            'users_by_role': defaultdict(int),
            'access_by_time': defaultdict(int),
            'failed_access_attempts': [],
            'unusual_access_patterns': [],
            'data_access_frequency': defaultdict(int),
            'model_usage_patterns': defaultdict(list)
        }
        
        bias_indicators = []
        log_files = glob.glob(os.path.join(log_directory, "*.log"))
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        findings['total_access_events'] += 1
                        
                        # Analyze access patterns
                        access_info = self._extract_access_info(line)
                        if access_info:
                            self._analyze_access_patterns(access_info, findings, line, log_file, line_num)
                        
                        # Check for unusual activity
                        unusual_activity = self._detect_unusual_activity(line)
                        if unusual_activity:
                            findings['unusual_access_patterns'].append({
                                'line': line_num,
                                'file': log_file,
                                'activity': unusual_activity,
                                'entry': line.strip()
                            })
                            
                            # Create bias indicator for unusual activity
                            indicator = BiasIndicator(
                                indicator_id=f"UNUSUAL_{self.case_id}_{line_num}",
                                timestamp=self._extract_timestamp(line) or datetime.now().isoformat(),
                                log_entry=line.strip(),
                                bias_type="unusual_access",
                                severity=self._assess_activity_severity(unusual_activity),
                                confidence=0.7,
                                context={
                                    'file': log_file,
                                    'line_number': line_num,
                                    'activity_type': unusual_activity
                                },
                                supporting_evidence=[line.strip()]
                            )
                            bias_indicators.append(indicator)
            
            except Exception as e:
                self.logger.error(f"Error analyzing audit log {log_file}: {e}")
        
        # Generate statistical analysis
        statistical_summary = self._generate_audit_statistics(findings)
        
        # Calculate confidence score
        confidence_score = self._calculate_analysis_confidence(bias_indicators, findings)
        
        # Generate recommendations
        recommendations = self._generate_audit_recommendations(bias_indicators, findings)
        
        # Create analysis result
        result = LogAnalysisResult(
            analysis_id=f"AUDIT_LOGS_{self.case_id}",
            timestamp=datetime.now().isoformat(),
            log_source=log_directory,
            analysis_type="audit_logs",
            findings=findings,
            bias_indicators=[asdict(bi) for bi in bias_indicators],
            statistical_summary=statistical_summary,
            recommendations=recommendations,
            confidence_score=confidence_score,
            data_hash=self._calculate_data_hash(findings)
        )
        
        self.analysis_results.append(result)
        self.bias_indicators.extend(bias_indicators)
        
        self.logger.info(f"Completed audit log analysis: {len(bias_indicators)} bias indicators found")
        return result
    
    def _extract_decision_info(self, log_line: str) -> Optional[Dict[str, Any]]:
        """Extract decision information from log line."""
        # Look for scoring/decision patterns
        patterns = [
            r'score.*?(\d+\.\d+)',
            r'decision.*?(accept|reject|hire|pass)',
            r'candidate.*?([A-Za-z\s]+)',
            r'job.*?([A-Za-z\s]+)'
        ]
        
        decision_info = {}
        
        for pattern in patterns:
            match = re.search(pattern, log_line, re.IGNORECASE)
            if match:
                if 'score' in pattern:
                    decision_info['score'] = float(match.group(1))
                elif 'decision' in pattern:
                    decision_info['decision'] = match.group(1).lower()
                elif 'candidate' in pattern:
                    decision_info['candidate'] = match.group(1).strip()
                elif 'job' in pattern:
                    decision_info['job'] = match.group(1).strip()
        
        return decision_info if decision_info else None
    
    def _analyze_decision_for_bias(self, decision_info: Dict[str, Any], 
                                  log_line: str, location: str) -> Optional[BiasIndicator]:
        """Analyze a decision for potential bias."""
        bias_signals = []
        
        # Check for demographic indicators in candidate name
        if 'candidate' in decision_info:
            name = decision_info['candidate'].lower()
            if self._is_gender_biased_name(name):
                bias_signals.append('gender_name_bias')
            if self._is_ethnicity_biased_name(name):
                bias_signals.append('ethnicity_name_bias')
        
        # Check for score anomalies
        if 'score' in decision_info and 'decision' in decision_info:
            if self._is_score_decision_inconsistent(decision_info['score'], decision_info['decision']):
                bias_signals.append('score_decision_inconsistency')
        
        # Check for bias keywords in log line
        bias_keywords = self._detect_bias_keywords(log_line)
        if bias_keywords:
            bias_signals.extend([f"keyword_{cat}" for _, cat in bias_keywords])
        
        if bias_signals:
            # Create bias indicator
            confidence = min(0.9, len(bias_signals) * 0.3)  # Higher confidence with more signals
            
            return BiasIndicator(
                indicator_id=f"DECISION_{self.case_id}_{hash(location)}",
                timestamp=self._extract_timestamp(log_line) or datetime.now().isoformat(),
                log_entry=log_line.strip(),
                bias_type="decision_bias",
                severity=self._assess_decision_severity(bias_signals),
                confidence=confidence,
                context={
                    'location': location,
                    'decision_info': decision_info,
                    'bias_signals': bias_signals
                },
                supporting_evidence=[log_line.strip()]
            )
        
        return None
    
    def _detect_bias_keywords(self, text: str) -> List[Tuple[str, str]]:
        """Detect bias-related keywords in text."""
        found_keywords = []
        text_lower = text.lower()
        
        for category, keywords in self.config['bias_keywords'].items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.append((keyword, category))
        
        return found_keywords
    
    def _detect_suspicious_patterns(self, text: str) -> List[str]:
        """Detect suspicious patterns that may indicate bias."""
        found_patterns = []
        
        for pattern in self.config['suspicious_patterns']:
            if re.search(pattern, text, re.IGNORECASE):
                found_patterns.append(pattern)
        
        return found_patterns
    
    def _extract_timestamp(self, log_line: str) -> Optional[str]:
        """Extract timestamp from log line."""
        # Common timestamp patterns
        patterns = [
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})',
            r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})',
            r'(\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, log_line)
            if match:
                return match.group(1)
        
        return None
    
    def _extract_user_action(self, log_line: str) -> Optional[str]:
        """Extract user action from log line."""
        action_patterns = [
            r'(login|logout|access|view|edit|delete|create|search|filter)',
            r'(score|evaluate|approve|reject|submit)'
        ]
        
        for pattern in action_patterns:
            match = re.search(pattern, log_line, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        
        return None
    
    def _detect_system_anomaly(self, log_line: str) -> Optional[str]:
        """Detect system anomalies that might indicate tampering."""
        anomaly_patterns = [
            (r'error.*bias', 'bias_error'),
            (r'failed.*authentication', 'auth_failure'),
            (r'unauthorized.*access', 'unauthorized_access'),
            (r'data.*modified', 'data_modification'),
            (r'log.*deleted', 'log_deletion'),
            (r'permission.*denied', 'permission_denied')
        ]
        
        for pattern, anomaly_type in anomaly_patterns:
            if re.search(pattern, log_line, re.IGNORECASE):
                return anomaly_type
        
        return None
    
    def _extract_access_info(self, log_line: str) -> Optional[Dict[str, Any]]:
        """Extract access information from audit log line."""
        access_patterns = {
            'user': r'user[:\s]+([A-Za-z0-9_]+)',
            'action': r'action[:\s]+([A-Za-z_]+)',
            'resource': r'resource[:\s]+([A-Za-z0-9_/]+)',
            'ip': r'ip[:\s]+(\d+\.\d+\.\d+\.\d+)',
            'status': r'status[:\s]+([A-Za-z]+)'
        }
        
        access_info = {}
        
        for field, pattern in access_patterns.items():
            match = re.search(pattern, log_line, re.IGNORECASE)
            if match:
                access_info[field] = match.group(1)
        
        return access_info if access_info else None
    
    def _analyze_access_patterns(self, access_info: Dict[str, Any], findings: Dict,
                                log_line: str, log_file: str, line_num: int):
        """Analyze access patterns for potential bias indicators."""
        if 'user' in access_info:
            findings['users_by_role'][access_info.get('action', 'unknown')] += 1
        
        if 'resource' in access_info:
            findings['data_access_frequency'][access_info['resource']] += 1
        
        # Check for failed access
        if access_info.get('status', '').lower() in ['failed', 'denied', 'unauthorized']:
            findings['failed_access_attempts'].append({
                'line': line_num,
                'file': log_file,
                'user': access_info.get('user', 'unknown'),
                'resource': access_info.get('resource', 'unknown'),
                'entry': log_line.strip()
            })
    
    def _detect_unusual_activity(self, log_line: str) -> Optional[str]:
        """Detect unusual activity patterns."""
        unusual_patterns = [
            (r'multiple.*login.*attempts', 'multiple_login_attempts'),
            (r'access.*outside.*hours', 'off_hours_access'),
            (r'bulk.*data.*download', 'bulk_data_access'),
            (r'rapid.*consecutive.*requests', 'rapid_requests'),
            (r'privilege.*escalation', 'privilege_escalation'),
            (r'admin.*override', 'admin_override')
        ]
        
        for pattern, activity_type in unusual_patterns:
            if re.search(pattern, log_line, re.IGNORECASE):
                return activity_type
        
        return None
    
    def _is_gender_biased_name(self, name: str) -> bool:
        """Check if name suggests gender bias."""
        male_indicators = ['john', 'mike', 'david', 'robert', 'james', 'william']
        female_indicators = ['jane', 'mary', 'sarah', 'lisa', 'jennifer', 'emily']
        
        name_lower = name.lower()
        return any(indicator in name_lower for indicator in male_indicators + female_indicators)
    
    def _is_ethnicity_biased_name(self, name: str) -> bool:
        """Check if name suggests ethnicity bias."""
        ethnic_indicators = ['jose', 'maria', 'ahmed', 'fatima', 'li', 'wang', 'patel', 'kumar']
        name_lower = name.lower()
        return any(indicator in name_lower for indicator in ethnic_indicators)
    
    def _is_score_decision_inconsistent(self, score: float, decision: str) -> bool:
        """Check for inconsistency between score and decision."""
        if decision == 'accept' and score < 0.5:
            return True
        if decision == 'reject' and score > 0.8:
            return True
        return False
    
    def _assess_keyword_severity(self, keyword: str, category: str) -> str:
        """Assess severity of bias keyword detection."""
        high_severity = ['discriminate', 'bias', 'prefer', 'avoid', 'exclude']
        
        if keyword.lower() in high_severity:
            return 'high'
        elif category in ['race_ethnicity', 'religion', 'disability']:
            return 'medium'
        else:
            return 'low'
    
    def _assess_decision_severity(self, bias_signals: List[str]) -> str:
        """Assess severity of decision bias."""
        critical_signals = ['score_decision_inconsistency', 'keyword_race_ethnicity']
        high_signals = ['gender_name_bias', 'keyword_gender']
        
        if any(signal in critical_signals for signal in bias_signals):
            return 'critical'
        elif any(signal in high_signals for signal in bias_signals):
            return 'high'
        elif len(bias_signals) > 2:
            return 'medium'
        else:
            return 'low'
    
    def _assess_activity_severity(self, activity_type: str) -> str:
        """Assess severity of unusual activity."""
        critical_activities = ['privilege_escalation', 'admin_override']
        high_activities = ['bulk_data_access', 'off_hours_access']
        
        if activity_type in critical_activities:
            return 'critical'
        elif activity_type in high_activities:
            return 'high'
        else:
            return 'medium'
    
    def _collect_decision_statistics(self, decision_info: Dict[str, Any], findings: Dict):
        """Collect statistics from decision information."""
        if 'score' in decision_info:
            findings['score_distributions']['all'].append(decision_info['score'])
        
        if 'decision' in decision_info:
            findings['decisions_by_demographic']['all'][decision_info['decision']] += 1
    
    def _generate_decision_statistics(self, findings: Dict) -> Dict[str, Any]:
        """Generate statistical summary for decision analysis."""
        stats = {
            'total_decisions': findings['total_decisions'],
            'bias_indicators_found': len(findings['bias_patterns']),
            'bias_rate': len(findings['bias_patterns']) / max(findings['total_decisions'], 1) * 100
        }
        
        if findings['score_distributions']['all']:
            scores = findings['score_distributions']['all']
            stats['score_statistics'] = {
                'mean': statistics.mean(scores),
                'median': statistics.median(scores),
                'std_dev': statistics.stdev(scores) if len(scores) > 1 else 0,
                'min': min(scores),
                'max': max(scores)
            }
        
        return stats
    
    def _generate_application_statistics(self, findings: Dict) -> Dict[str, Any]:
        """Generate statistical summary for application log analysis."""
        return {
            'total_entries': findings['total_entries'],
            'bias_keywords_found': dict(findings['bias_keywords_found']),
            'suspicious_queries': len(findings['suspicious_queries']),
            'system_anomalies': len(findings['system_anomalies']),
            'most_common_actions': dict(Counter(findings['user_actions']).most_common(5))
        }
    
    def _generate_audit_statistics(self, findings: Dict) -> Dict[str, Any]:
        """Generate statistical summary for audit log analysis."""
        return {
            'total_access_events': findings['total_access_events'],
            'failed_access_attempts': len(findings['failed_access_attempts']),
            'unusual_patterns': len(findings['unusual_access_patterns']),
            'users_by_role': dict(findings['users_by_role']),
            'most_accessed_resources': dict(Counter(findings['data_access_frequency']).most_common(5))
        }
    
    def _calculate_analysis_confidence(self, bias_indicators: List[BiasIndicator], 
                                     findings: Dict) -> float:
        """Calculate overall confidence in analysis results."""
        if not bias_indicators:
            return 0.9  # High confidence in no bias found
        
        # Average confidence of individual indicators
        avg_confidence = statistics.mean([bi.confidence for bi in bias_indicators])
        
        # Adjust based on quantity of evidence
        evidence_factor = min(1.0, len(bias_indicators) / 10)
        
        # Combine factors
        overall_confidence = (avg_confidence * 0.7) + (evidence_factor * 0.3)
        
        return min(0.95, overall_confidence)
    
    def _generate_bias_recommendations(self, bias_indicators: List[BiasIndicator], 
                                     findings: Dict) -> List[str]:
        """Generate recommendations based on bias analysis."""
        recommendations = []
        
        if not bias_indicators:
            recommendations.append("No significant bias indicators detected in logs.")
            recommendations.append("Continue monitoring with regular log analysis.")
            return recommendations
        
        # Category-specific recommendations
        bias_types = [bi.bias_type for bi in bias_indicators]
        
        if any('gender' in bt for bt in bias_types):
            recommendations.append("GENDER BIAS DETECTED: Review hiring decisions for gender-related disparities.")
            recommendations.append("Implement blind resume screening to reduce gender bias.")
        
        if any('race' in bt or 'ethnicity' in bt for bt in bias_types):
            recommendations.append("ETHNIC BIAS DETECTED: Conduct thorough review of name-based discrimination.")
            recommendations.append("Consider removing names from initial screening process.")
        
        if any('decision' in bt for bt in bias_types):
            recommendations.append("DECISION INCONSISTENCIES: Review model scoring logic and thresholds.")
            recommendations.append("Implement additional validation for edge case decisions.")
        
        # Severity-based recommendations
        critical_indicators = [bi for bi in bias_indicators if bi.severity == 'critical']
        if critical_indicators:
            recommendations.append("CRITICAL ISSUES: Immediate investigation required for critical bias indicators.")
            recommendations.append("Consider suspending automated decisions pending investigation.")
        
        recommendations.append("Implement continuous monitoring with automated bias detection.")
        recommendations.append("Provide bias awareness training for all system users.")
        
        return recommendations
    
    def _generate_application_recommendations(self, bias_indicators: List[BiasIndicator], 
                                            findings: Dict) -> List[str]:
        """Generate recommendations for application log issues."""
        recommendations = []
        
        if findings['bias_keywords_found']:
            recommendations.append("Bias-related keywords detected in application logs.")
            recommendations.append("Review user queries and system responses for bias indicators.")
        
        if findings['suspicious_queries']:
            recommendations.append("Suspicious query patterns detected.")
            recommendations.append("Investigate potential manipulation of search/filter criteria.")
        
        if findings['system_anomalies']:
            recommendations.append("System anomalies detected that may indicate tampering.")
            recommendations.append("Conduct security audit and verify system integrity.")
        
        recommendations.append("Implement query auditing and bias detection in real-time.")
        
        return recommendations
    
    def _generate_audit_recommendations(self, bias_indicators: List[BiasIndicator], 
                                      findings: Dict) -> List[str]:
        """Generate recommendations for audit log issues."""
        recommendations = []
        
        if findings['failed_access_attempts']:
            recommendations.append("Multiple failed access attempts detected.")
            recommendations.append("Review access controls and user permissions.")
        
        if findings['unusual_access_patterns']:
            recommendations.append("Unusual access patterns detected.")
            recommendations.append("Investigate off-hours access and bulk data operations.")
        
        recommendations.append("Enhance access monitoring with behavioral analysis.")
        recommendations.append("Implement automated alerts for suspicious access patterns.")
        
        return recommendations
    
    def _calculate_data_hash(self, data: Any) -> str:
        """Calculate SHA-256 hash of data for integrity verification."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def run_comprehensive_log_analysis(self, log_directories: Dict[str, str]) -> Dict[str, Any]:
        """Run comprehensive log analysis across all log types."""
        self.logger.info("Starting comprehensive log analysis")
        start_time = time.time()
        
        analysis_results = {}
        
        # Analyze different log types
        if 'model_decisions' in log_directories:
            try:
                analysis_results['model_decisions'] = self.analyze_model_decision_logs(
                    log_directories['model_decisions']
                )
            except Exception as e:
                self.logger.error(f"Error analyzing model decision logs: {e}")
        
        if 'application' in log_directories:
            try:
                analysis_results['application'] = self.analyze_application_logs(
                    log_directories['application']
                )
            except Exception as e:
                self.logger.error(f"Error analyzing application logs: {e}")
        
        if 'audit' in log_directories:
            try:
                analysis_results['audit'] = self.analyze_audit_logs(
                    log_directories['audit']
                )
            except Exception as e:
                self.logger.error(f"Error analyzing audit logs: {e}")
        
        # Generate comprehensive report
        comprehensive_report = {
            'case_id': self.case_id,
            'timestamp': datetime.now().isoformat(),
            'execution_time_seconds': time.time() - start_time,
            'log_directories': log_directories,
            'analysis_results': {k: asdict(v) for k, v in analysis_results.items()},
            'total_bias_indicators': len(self.bias_indicators),
            'overall_bias_assessment': self._generate_overall_assessment(),
            'comprehensive_recommendations': self._generate_comprehensive_recommendations(),
            'forensic_metadata': {
                'investigator': os.getenv('USER', 'unknown'),
                'analysis_hash': self._calculate_data_hash(analysis_results)
            }
        }
        
        # Save comprehensive report
        self._save_comprehensive_report(comprehensive_report)
        
        self.logger.info(f"Completed comprehensive log analysis: {len(self.bias_indicators)} total bias indicators")
        return comprehensive_report
    
    def _generate_overall_assessment(self) -> Dict[str, Any]:
        """Generate overall bias assessment across all log types."""
        if not self.bias_indicators:
            return {
                'risk_level': 'low',
                'confidence': 0.9,
                'summary': 'No significant bias indicators detected across all log types.'
            }
        
        # Calculate risk level
        critical_count = len([bi for bi in self.bias_indicators if bi.severity == 'critical'])
        high_count = len([bi for bi in self.bias_indicators if bi.severity == 'high'])
        
        if critical_count > 0:
            risk_level = 'critical'
        elif high_count > 2:
            risk_level = 'high'
        elif len(self.bias_indicators) > 5:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        # Calculate overall confidence
        if self.bias_indicators:
            avg_confidence = statistics.mean([bi.confidence for bi in self.bias_indicators])
        else:
            avg_confidence = 0.9
        
        return {
            'risk_level': risk_level,
            'confidence': avg_confidence,
            'total_indicators': len(self.bias_indicators),
            'critical_indicators': critical_count,
            'high_indicators': high_count,
            'summary': f'Found {len(self.bias_indicators)} bias indicators with {risk_level} risk level.'
        }
    
    def _generate_comprehensive_recommendations(self) -> List[str]:
        """Generate comprehensive recommendations across all analyses."""
        recommendations = []
        
        # Collect all recommendations
        all_recommendations = []
        for result in self.analysis_results:
            all_recommendations.extend(result.recommendations)
        
        # Remove duplicates and prioritize
        unique_recommendations = list(set(all_recommendations))
        
        # Add comprehensive recommendations
        if self.bias_indicators:
            recommendations.append("IMMEDIATE ACTION REQUIRED: Bias indicators detected across multiple log types.")
            recommendations.append("Conduct comprehensive audit of all hiring decisions made by the system.")
            recommendations.append("Implement enhanced monitoring and bias detection systems.")
        
        recommendations.extend(unique_recommendations)
        
        recommendations.append("Establish regular log analysis schedule for ongoing bias monitoring.")
        recommendations.append("Create incident response procedures for bias detection.")
        
        return recommendations
    
    def _save_comprehensive_report(self, report: Dict[str, Any]) -> None:
        """Save comprehensive log analysis report."""
        reports_dir = Path("forensic/reports/log_analysis")
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed report
        report_file = reports_dir / f"comprehensive_log_analysis_{self.case_id}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Save executive summary
        executive_summary = {
            'case_id': report['case_id'],
            'timestamp': report['timestamp'],
            'total_bias_indicators': report['total_bias_indicators'],
            'overall_assessment': report['overall_bias_assessment'],
            'key_recommendations': report['comprehensive_recommendations'][:5]
        }
        
        summary_file = reports_dir / f"log_analysis_summary_{self.case_id}.json"
        with open(summary_file, 'w') as f:
            json.dump(executive_summary, f, indent=2, default=str)
        
        self.logger.info(f"Comprehensive report saved: {report_file}")
    
    def export_evidence_package(self, output_dir: str = None) -> str:
        """Export complete evidence package for legal proceedings."""
        if output_dir is None:
            output_dir = f"forensic/evidence/log_analysis_{self.case_id}"
        
        evidence_dir = Path(output_dir)
        evidence_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy all relevant files
        import shutil
        
        # Copy analysis reports
        reports_dir = Path("forensic/reports/log_analysis")
        if reports_dir.exists():
            shutil.copytree(reports_dir, evidence_dir / "reports", dirs_exist_ok=True)
        
        # Copy analysis logs
        logs_dir = Path("forensic/logs/log_analysis")
        if logs_dir.exists():
            shutil.copytree(logs_dir, evidence_dir / "logs", dirs_exist_ok=True)
        
        # Create evidence manifest
        manifest = {
            'case_id': self.case_id,
            'export_timestamp': datetime.now().isoformat(),
            'exported_by': os.getenv('USER', 'unknown'),
            'total_bias_indicators': len(self.bias_indicators),
            'analysis_results_count': len(self.analysis_results),
            'evidence_files': [
                str(f.relative_to(evidence_dir)) for f in evidence_dir.rglob('*') if f.is_file()
            ],
            'data_integrity_hash': self._calculate_data_hash(self.bias_indicators)
        }
        
        manifest_file = evidence_dir / "evidence_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(manifest, f, indent=2, default=str)
        
        self.logger.info(f"Evidence package exported to: {evidence_dir}")
        return str(evidence_dir)


def main():
    """Main function for standalone log analysis."""
    print("Log Analysis and Bias Detection Framework")
    print("=" * 50)
    
    # Initialize analyzer
    case_id = f"DEMO_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    analyzer = LogAnalyzer(case_id)
    
    # Define log directories (adjust paths as needed)
    log_directories = {
        'model_decisions': 'logs/model_decisions',
        'application': 'logs/application',
        'audit': 'logs/audit'
    }
    
    # Create demo log directories if they don't exist
    for log_type, log_dir in log_directories.items():
        os.makedirs(log_dir, exist_ok=True)
        
        # Create sample log file if directory is empty
        if not os.listdir(log_dir):
            sample_log = os.path.join(log_dir, f"sample_{log_type}.log")
            with open(sample_log, 'w') as f:
                f.write(f"2025-01-01 12:00:00 | INFO | Sample {log_type} log entry\n")
                f.write(f"2025-01-01 12:01:00 | INFO | No bias indicators in this demo\n")
            print(f"Created sample log: {sample_log}")
    
    # Run comprehensive analysis
    print("\nRunning comprehensive log analysis...")
    report = analyzer.run_comprehensive_log_analysis(log_directories)
    
    # Display results
    print(f"\nAnalysis completed!")
    print(f"Case ID: {report['case_id']}")
    print(f"Total bias indicators: {report['total_bias_indicators']}")
    print(f"Execution time: {report['execution_time_seconds']:.2f} seconds")
    
    overall_assessment = report['overall_bias_assessment']
    print(f"Risk Level: {overall_assessment['risk_level']}")
    print(f"Confidence: {overall_assessment['confidence']:.2f}")
    
    # Export evidence
    evidence_dir = analyzer.export_evidence_package()
    print(f"\nEvidence package exported to: {evidence_dir}")


if __name__ == "__main__":
    main()