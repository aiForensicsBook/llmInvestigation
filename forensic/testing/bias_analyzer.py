#!/usr/bin/env python3
"""
Bias Analyzer for Resume Screening LLM Forensic Testing
========================================================

This module provides comprehensive bias analysis capabilities for forensic examination
of resume screening AI systems. It analyzes training data and model performance
across different demographic groups to detect various forms of bias.

Author: Forensic AI Testing Suite
Created: 2025-07-01
Purpose: Legal forensic analysis of AI bias in hiring systems
"""

import os
import json
import logging
import hashlib
import datetime
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class BiasTestResult:
    """Data class for storing bias test results with forensic integrity."""
    test_name: str
    test_type: str
    timestamp: str
    data_hash: str
    demographic_group: str
    protected_characteristic: str
    statistical_measure: str
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    bias_detected: bool
    sample_size: int
    test_parameters: Dict[str, Any]
    raw_data_summary: Dict[str, Any]


class ForensicLogger:
    """Forensic-grade logging with tamper-evident features."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup forensic logging
        self.logger = logging.getLogger('forensic_bias_analyzer')
        self.logger.setLevel(logging.DEBUG)
        
        # Create forensic log handler with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"bias_analysis_forensic_{timestamp}.log"
        
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s.%(msecs)03d|%(levelname)s|%(funcName)s:%(lineno)d|%(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
        # Log system information for forensic purposes
        self._log_system_info()
    
    def _log_system_info(self):
        """Log system information for forensic chain of custody."""
        import platform
        import sys
        
        self.logger.info("=== FORENSIC BIAS ANALYSIS SESSION STARTED ===")
        self.logger.info(f"System: {platform.system()} {platform.release()}")
        self.logger.info(f"Python: {sys.version}")
        self.logger.info(f"Working Directory: {os.getcwd()}")
        self.logger.info(f"User: {os.getenv('USER', 'Unknown')}")
    
    def log_test_start(self, test_name: str, parameters: Dict[str, Any]):
        """Log the start of a bias test with parameters."""
        param_hash = hashlib.sha256(str(parameters).encode()).hexdigest()[:16]
        self.logger.info(f"BIAS_TEST_START|{test_name}|PARAM_HASH:{param_hash}|{parameters}")
    
    def log_test_result(self, result: BiasTestResult):
        """Log bias test results with forensic integrity."""
        self.logger.info(f"BIAS_TEST_RESULT|{result.test_name}|{result.demographic_group}|"
                        f"BIAS_DETECTED:{result.bias_detected}|P_VALUE:{result.p_value:.6f}|"
                        f"EFFECT_SIZE:{result.effect_size:.6f}|SAMPLE_SIZE:{result.sample_size}")
    
    def log_data_integrity(self, data_description: str, data_hash: str, record_count: int):
        """Log data integrity information."""
        self.logger.info(f"DATA_INTEGRITY|{data_description}|HASH:{data_hash}|RECORDS:{record_count}")


class BiasAnalyzer:
    """
    Comprehensive bias analyzer for resume screening AI systems.
    
    This class provides methods to detect various forms of bias including:
    - Gender bias
    - Age bias  
    - Racial/ethnic bias
    - Educational institution bias
    - Geographic bias
    - Socioeconomic bias
    """
    
    def __init__(self, output_dir: str = "./forensic_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize forensic logging
        self.logger = ForensicLogger(self.output_dir / "logs")
        
        # Initialize results storage
        self.test_results: List[BiasTestResult] = []
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Statistical significance threshold
        self.alpha = 0.05
        
        # Bias detection thresholds
        self.bias_thresholds = {
            'small_effect': 0.2,
            'medium_effect': 0.5,
            'large_effect': 0.8
        }
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of data for forensic integrity."""
        data_string = data.to_string()
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def _get_timestamp(self) -> str:
        """Get ISO format timestamp for forensic records."""
        return datetime.datetime.now().isoformat()
    
    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def analyze_gender_bias(self, data: pd.DataFrame, score_column: str, 
                           gender_column: str = 'gender', 
                           bias_metadata_column: Optional[str] = None) -> List[BiasTestResult]:
        """
        Analyze gender bias in resume scoring.
        
        Args:
            data: DataFrame containing resume data and scores
            score_column: Column name containing AI scores/decisions
            gender_column: Column name containing gender information
            bias_metadata_column: Optional column containing bias metadata/multipliers
        
        Returns:
            List of BiasTestResult objects
        """
        self.logger.log_test_start("gender_bias_analysis", {
            "score_column": score_column,
            "gender_column": gender_column,
            "data_shape": data.shape
        })
        
        results = []
        data_hash = self._calculate_data_hash(data)
        timestamp = self._get_timestamp()
        
        # Log data integrity
        self.logger.log_data_integrity("gender_bias_dataset", data_hash, len(data))
        
        # Clean and prepare data
        clean_data = data.dropna(subset=[score_column, gender_column])
        
        # Get unique gender groups
        gender_groups = clean_data[gender_column].unique()
        
        # Perform pairwise comparisons between gender groups
        for i, gender1 in enumerate(gender_groups):
            for gender2 in gender_groups[i+1:]:
                group1_scores = clean_data[clean_data[gender_column] == gender1][score_column]
                group2_scores = clean_data[clean_data[gender_column] == gender2][score_column]
                
                # Mann-Whitney U test (non-parametric)
                statistic, p_value = stats.mannwhitneyu(
                    group1_scores, group2_scores, alternative='two-sided'
                )
                
                # Calculate effect size
                effect_size = self._calculate_cohens_d(group1_scores.values, group2_scores.values)
                
                # Calculate confidence interval for effect size
                n1, n2 = len(group1_scores), len(group2_scores)
                se = np.sqrt((n1 + n2) / (n1 * n2) + effect_size**2 / (2 * (n1 + n2)))
                ci_lower = effect_size - 1.96 * se
                ci_upper = effect_size + 1.96 * se
                
                # Determine if bias is detected
                is_significant = p_value < self.alpha
                bias_detected = is_significant and abs(effect_size) >= self.bias_thresholds['small_effect']
                
                # Capture bias multipliers if available
                bias_multipliers = {}
                if bias_metadata_column and bias_metadata_column in clean_data.columns:
                    group1_metadata = clean_data[clean_data[gender_column] == gender1][bias_metadata_column]
                    group2_metadata = clean_data[clean_data[gender_column] == gender2][bias_metadata_column]
                    
                    # Extract multipliers from metadata
                    bias_multipliers = {
                        f"{gender1}_avg_multiplier": self._extract_avg_multiplier(group1_metadata),
                        f"{gender2}_avg_multiplier": self._extract_avg_multiplier(group2_metadata),
                        f"{gender1}_multiplier_range": self._extract_multiplier_range(group1_metadata),
                        f"{gender2}_multiplier_range": self._extract_multiplier_range(group2_metadata)
                    }
                
                result = BiasTestResult(
                    test_name="gender_bias_mannwhitney",
                    test_type="statistical_comparison",
                    timestamp=timestamp,
                    data_hash=data_hash,
                    demographic_group=f"{gender1}_vs_{gender2}",
                    protected_characteristic="gender",
                    statistical_measure="mann_whitney_u",
                    p_value=p_value,
                    effect_size=effect_size,
                    confidence_interval=(ci_lower, ci_upper),
                    is_significant=is_significant,
                    bias_detected=bias_detected,
                    sample_size=n1 + n2,
                    test_parameters={
                        "gender1": gender1,
                        "gender2": gender2,
                        "group1_size": n1,
                        "group2_size": n2,
                        "test_statistic": float(statistic),
                        "bias_multipliers": bias_multipliers
                    },
                    raw_data_summary={
                        "group1_mean": float(group1_scores.mean()),
                        "group1_std": float(group1_scores.std()),
                        "group2_mean": float(group2_scores.mean()),
                        "group2_std": float(group2_scores.std()),
                        "score_ratio": float(group1_scores.mean() / group2_scores.mean()) if group2_scores.mean() > 0 else float('inf')
                    }
                )
                
                results.append(result)
                self.logger.log_test_result(result)
        
        self.test_results.extend(results)
        return results
    
    def analyze_age_bias(self, data: pd.DataFrame, score_column: str, 
                        age_column: str = 'age') -> List[BiasTestResult]:
        """
        Analyze age bias in resume scoring.
        
        Args:
            data: DataFrame containing resume data and scores
            score_column: Column name containing AI scores/decisions
            age_column: Column name containing age information
        
        Returns:
            List of BiasTestResult objects
        """
        self.logger.log_test_start("age_bias_analysis", {
            "score_column": score_column,
            "age_column": age_column,
            "data_shape": data.shape
        })
        
        results = []
        data_hash = self._calculate_data_hash(data)
        timestamp = self._get_timestamp()
        
        # Clean data
        clean_data = data.dropna(subset=[score_column, age_column])
        
        # Create age groups for analysis
        age_bins = [0, 25, 35, 45, 55, 100]
        age_labels = ['Under_25', '25-34', '35-44', '45-54', '55_Plus']
        clean_data['age_group'] = pd.cut(clean_data[age_column], bins=age_bins, 
                                        labels=age_labels, right=False)
        
        # Correlation analysis
        correlation, p_value_corr = stats.pearsonr(clean_data[age_column], clean_data[score_column])
        
        # Spearman correlation (non-parametric)
        spearman_corr, p_value_spearman = stats.spearmanr(clean_data[age_column], clean_data[score_column])
        
        # Age correlation result
        result_corr = BiasTestResult(
            test_name="age_bias_correlation",
            test_type="correlation_analysis",
            timestamp=timestamp,
            data_hash=data_hash,
            demographic_group="all_ages",
            protected_characteristic="age",
            statistical_measure="pearson_correlation",
            p_value=p_value_corr,
            effect_size=correlation,
            confidence_interval=self._correlation_confidence_interval(correlation, len(clean_data)),
            is_significant=p_value_corr < self.alpha,
            bias_detected=p_value_corr < self.alpha and abs(correlation) >= 0.1,
            sample_size=len(clean_data),
            test_parameters={
                "correlation_type": "pearson",
                "spearman_correlation": spearman_corr,
                "spearman_p_value": p_value_spearman
            },
            raw_data_summary={
                "age_mean": float(clean_data[age_column].mean()),
                "age_std": float(clean_data[age_column].std()),
                "score_mean": float(clean_data[score_column].mean()),
                "score_std": float(clean_data[score_column].std())
            }
        )
        
        results.append(result_corr)
        self.logger.log_test_result(result_corr)
        
        # ANOVA across age groups
        age_groups = [group[score_column].values for name, group in clean_data.groupby('age_group') 
                     if not group.empty]
        
        if len(age_groups) > 2:
            f_stat, p_value_anova = stats.f_oneway(*age_groups)
            
            # Calculate eta-squared (effect size for ANOVA)
            ss_between = sum(len(group) * (np.mean(group) - np.mean(clean_data[score_column]))**2 
                           for group in age_groups)
            ss_total = np.var(clean_data[score_column]) * (len(clean_data) - 1)
            eta_squared = ss_between / ss_total if ss_total > 0 else 0
            
            result_anova = BiasTestResult(
                test_name="age_bias_anova",
                test_type="group_comparison",
                timestamp=timestamp,
                data_hash=data_hash,
                demographic_group="age_groups",
                protected_characteristic="age",
                statistical_measure="one_way_anova",
                p_value=p_value_anova,
                effect_size=eta_squared,
                confidence_interval=(0, 1),  # Eta-squared is bounded [0,1]
                is_significant=p_value_anova < self.alpha,
                bias_detected=p_value_anova < self.alpha and eta_squared >= 0.01,
                sample_size=len(clean_data),
                test_parameters={
                    "f_statistic": float(f_stat),
                    "num_groups": len(age_groups),
                    "group_sizes": [len(group) for group in age_groups]
                },
                raw_data_summary={
                    "group_means": [float(np.mean(group)) for group in age_groups],
                    "group_stds": [float(np.std(group)) for group in age_groups]
                }
            )
            
            results.append(result_anova)
            self.logger.log_test_result(result_anova)
        
        self.test_results.extend(results)
        return results
    
    def _correlation_confidence_interval(self, r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
        """Calculate confidence interval for Pearson correlation coefficient."""
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1 - alpha/2)
        
        z_lower = z - z_crit * se
        z_upper = z + z_crit * se
        
        return (np.tanh(z_lower), np.tanh(z_upper))
    
    def analyze_racial_bias(self, data: pd.DataFrame, score_column: str, 
                           race_column: str = 'race') -> List[BiasTestResult]:
        """
        Analyze racial/ethnic bias in resume scoring.
        
        Args:
            data: DataFrame containing resume data and scores
            score_column: Column name containing AI scores/decisions
            race_column: Column name containing race/ethnicity information
        
        Returns:
            List of BiasTestResult objects
        """
        self.logger.log_test_start("racial_bias_analysis", {
            "score_column": score_column,
            "race_column": race_column,
            "data_shape": data.shape
        })
        
        results = []
        data_hash = self._calculate_data_hash(data)
        timestamp = self._get_timestamp()
        
        # Clean data
        clean_data = data.dropna(subset=[score_column, race_column])
        
        # Get racial groups
        racial_groups = clean_data[race_column].unique()
        
        # Perform pairwise comparisons
        for i, race1 in enumerate(racial_groups):
            for race2 in racial_groups[i+1:]:
                group1_scores = clean_data[clean_data[race_column] == race1][score_column]
                group2_scores = clean_data[clean_data[race_column] == race2][score_column]
                
                if len(group1_scores) < 5 or len(group2_scores) < 5:
                    continue  # Skip comparisons with too few samples
                
                # Kolmogorov-Smirnov test
                ks_stat, p_value_ks = stats.ks_2samp(group1_scores, group2_scores)
                
                # Mann-Whitney U test
                mw_stat, p_value_mw = stats.mannwhitneyu(
                    group1_scores, group2_scores, alternative='two-sided'
                )
                
                # Effect size
                effect_size = self._calculate_cohens_d(group1_scores.values, group2_scores.values)
                
                # Confidence interval for effect size
                n1, n2 = len(group1_scores), len(group2_scores)
                se = np.sqrt((n1 + n2) / (n1 * n2) + effect_size**2 / (2 * (n1 + n2)))
                ci_lower = effect_size - 1.96 * se
                ci_upper = effect_size + 1.96 * se
                
                # Bias detection
                is_significant = min(p_value_ks, p_value_mw) < self.alpha
                bias_detected = is_significant and abs(effect_size) >= self.bias_thresholds['small_effect']
                
                result = BiasTestResult(
                    test_name="racial_bias_comparison",
                    test_type="statistical_comparison",
                    timestamp=timestamp,
                    data_hash=data_hash,
                    demographic_group=f"{race1}_vs_{race2}",
                    protected_characteristic="race",
                    statistical_measure="ks_test_and_mannwhitney",
                    p_value=min(p_value_ks, p_value_mw),
                    effect_size=effect_size,
                    confidence_interval=(ci_lower, ci_upper),
                    is_significant=is_significant,
                    bias_detected=bias_detected,
                    sample_size=n1 + n2,
                    test_parameters={
                        "race1": race1,
                        "race2": race2,
                        "ks_statistic": float(ks_stat),
                        "ks_p_value": p_value_ks,
                        "mw_statistic": float(mw_stat),
                        "mw_p_value": p_value_mw,
                        "group1_size": n1,
                        "group2_size": n2
                    },
                    raw_data_summary={
                        "group1_mean": float(group1_scores.mean()),
                        "group1_median": float(group1_scores.median()),
                        "group1_std": float(group1_scores.std()),
                        "group2_mean": float(group2_scores.mean()),
                        "group2_median": float(group2_scores.median()),
                        "group2_std": float(group2_scores.std())
                    }
                )
                
                results.append(result)
                self.logger.log_test_result(result)
        
        self.test_results.extend(results)
        return results
    
    def analyze_education_bias(self, data: pd.DataFrame, score_column: str, 
                              education_column: str = 'education') -> List[BiasTestResult]:
        """
        Analyze educational institution bias in resume scoring.
        
        Args:
            data: DataFrame containing resume data and scores
            score_column: Column name containing AI scores/decisions
            education_column: Column name containing education information
        
        Returns:
            List of BiasTestResult objects
        """
        self.logger.log_test_start("education_bias_analysis", {
            "score_column": score_column,
            "education_column": education_column,
            "data_shape": data.shape
        })
        
        results = []
        data_hash = self._calculate_data_hash(data)
        timestamp = self._get_timestamp()
        
        # Clean data
        clean_data = data.dropna(subset=[score_column, education_column])
        
        # Categorize institutions (this would need domain-specific logic)
        # For now, we'll analyze by exact institution
        institution_groups = clean_data.groupby(education_column)[score_column]
        
        # Only analyze institutions with sufficient sample size
        valid_institutions = [name for name, group in institution_groups if len(group) >= 10]
        
        if len(valid_institutions) < 2:
            self.logger.logger.warning("Insufficient data for education bias analysis")
            return results
        
        # Perform Kruskal-Wallis test (non-parametric ANOVA)
        groups_data = [institution_groups.get_group(inst).values for inst in valid_institutions]
        
        h_stat, p_value = stats.kruskal(*groups_data)
        
        # Calculate effect size (epsilon-squared for Kruskal-Wallis)
        n = len(clean_data)
        epsilon_squared = (h_stat - len(valid_institutions) + 1) / (n - len(valid_institutions))
        epsilon_squared = max(0, epsilon_squared)  # Ensure non-negative
        
        result = BiasTestResult(
            test_name="education_bias_kruskal_wallis",
            test_type="group_comparison",
            timestamp=timestamp,
            data_hash=data_hash,
            demographic_group="education_institutions",
            protected_characteristic="education",
            statistical_measure="kruskal_wallis",
            p_value=p_value,
            effect_size=epsilon_squared,
            confidence_interval=(0, 1),
            is_significant=p_value < self.alpha,
            bias_detected=p_value < self.alpha and epsilon_squared >= 0.01,
            sample_size=n,
            test_parameters={
                "h_statistic": float(h_stat),
                "num_institutions": len(valid_institutions),
                "institutions": valid_institutions,
                "group_sizes": [len(group) for group in groups_data]
            },
            raw_data_summary={
                "institution_means": {inst: float(institution_groups.get_group(inst).mean()) 
                                    for inst in valid_institutions},
                "institution_medians": {inst: float(institution_groups.get_group(inst).median()) 
                                      for inst in valid_institutions}
            }
        )
        
        results.append(result)
        self.logger.log_test_result(result)
        self.test_results.extend(results)
        return results
    
    def generate_bias_report(self, output_file: str = None) -> str:
        """
        Generate comprehensive bias analysis report.
        
        Args:
            output_file: Optional output file path
        
        Returns:
            Path to generated report
        """
        if output_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.reports_dir / f"bias_analysis_report_{timestamp}.json"
        
        # Compile comprehensive report
        report = {
            "metadata": {
                "report_generated": self._get_timestamp(),
                "total_tests_conducted": len(self.test_results),
                "tests_with_bias_detected": sum(1 for r in self.test_results if r.bias_detected),
                "analysis_parameters": {
                    "significance_level": self.alpha,
                    "bias_thresholds": self.bias_thresholds
                }
            },
            "executive_summary": self._generate_executive_summary(),
            "detailed_results": [asdict(result) for result in self.test_results],
            "statistical_summary": self._generate_statistical_summary(),
            "recommendations": self._generate_recommendations()
        }
        
        # Write report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Log report generation
        self.logger.logger.info(f"BIAS_REPORT_GENERATED|{output_file}|TESTS:{len(self.test_results)}")
        
        return str(output_file)
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of bias analysis."""
        bias_found = [r for r in self.test_results if r.bias_detected]
        
        summary = {
            "overall_bias_detected": len(bias_found) > 0,
            "bias_types_found": list(set(r.protected_characteristic for r in bias_found)),
            "most_significant_bias": None,
            "severity_assessment": "NONE"
        }
        
        if bias_found:
            # Find most significant bias
            most_significant = min(bias_found, key=lambda x: x.p_value)
            summary["most_significant_bias"] = {
                "characteristic": most_significant.protected_characteristic,
                "group": most_significant.demographic_group,
                "p_value": most_significant.p_value,
                "effect_size": most_significant.effect_size
            }
            
            # Assess severity
            max_effect = max(abs(r.effect_size) for r in bias_found)
            if max_effect >= self.bias_thresholds['large_effect']:
                summary["severity_assessment"] = "HIGH"
            elif max_effect >= self.bias_thresholds['medium_effect']:
                summary["severity_assessment"] = "MEDIUM"
            else:
                summary["severity_assessment"] = "LOW"
        
        return summary
    
    def _generate_statistical_summary(self) -> Dict[str, Any]:
        """Generate statistical summary of all tests."""
        if not self.test_results:
            return {}
        
        p_values = [r.p_value for r in self.test_results]
        effect_sizes = [abs(r.effect_size) for r in self.test_results]
        
        return {
            "p_value_distribution": {
                "min": min(p_values),
                "max": max(p_values),
                "mean": np.mean(p_values),
                "median": np.median(p_values)
            },
            "effect_size_distribution": {
                "min": min(effect_sizes),
                "max": max(effect_sizes),
                "mean": np.mean(effect_sizes),
                "median": np.median(effect_sizes)
            },
            "tests_by_characteristic": {
                char: len([r for r in self.test_results if r.protected_characteristic == char])
                for char in set(r.protected_characteristic for r in self.test_results)
            }
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on bias analysis."""
        recommendations = []
        bias_found = [r for r in self.test_results if r.bias_detected]
        
        if not bias_found:
            recommendations.append("No significant bias detected in current analysis.")
            recommendations.append("Continue regular monitoring and testing.")
        else:
            characteristics_with_bias = set(r.protected_characteristic for r in bias_found)
            
            for char in characteristics_with_bias:
                char_results = [r for r in bias_found if r.protected_characteristic == char]
                max_effect = max(abs(r.effect_size) for r in char_results)
                
                if max_effect >= self.bias_thresholds['large_effect']:
                    recommendations.append(
                        f"URGENT: Large effect size bias detected for {char}. "
                        f"Immediate model retraining and bias mitigation required."
                    )
                elif max_effect >= self.bias_thresholds['medium_effect']:
                    recommendations.append(
                        f"WARNING: Medium effect size bias detected for {char}. "
                        f"Bias mitigation strategies should be implemented."
                    )
                else:
                    recommendations.append(
                        f"NOTICE: Small but significant bias detected for {char}. "
                        f"Monitor closely and consider preventive measures."
                    )
            
            recommendations.append("Conduct regular bias audits with expanded test coverage.")
            recommendations.append("Implement bias mitigation techniques in model training.")
            recommendations.append("Review training data for representational imbalances.")
        
        return recommendations
    
    def _extract_avg_multiplier(self, metadata_series) -> float:
        """Extract average bias multiplier from metadata."""
        multipliers = []
        for meta in metadata_series:
            if isinstance(meta, dict) and 'bias_multiplier' in meta:
                multipliers.append(meta['bias_multiplier'])
            elif isinstance(meta, str):
                try:
                    meta_dict = json.loads(meta)
                    if 'bias_multiplier' in meta_dict:
                        multipliers.append(meta_dict['bias_multiplier'])
                except:
                    pass
        return np.mean(multipliers) if multipliers else 1.0
    
    def _extract_multiplier_range(self, metadata_series) -> Tuple[float, float]:
        """Extract range of bias multipliers from metadata."""
        multipliers = []
        for meta in metadata_series:
            if isinstance(meta, dict) and 'bias_multiplier' in meta:
                multipliers.append(meta['bias_multiplier'])
            elif isinstance(meta, str):
                try:
                    meta_dict = json.loads(meta)
                    if 'bias_multiplier' in meta_dict:
                        multipliers.append(meta_dict['bias_multiplier'])
                except:
                    pass
        if multipliers:
            return (min(multipliers), max(multipliers))
        return (1.0, 1.0)
    
    def analyze_hidden_bias_mechanisms(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze hidden bias mechanisms by comparing identical resumes with different genders.
        
        Args:
            test_data: DataFrame with columns including 'base_score', 'final_score', 
                      'gender', 'bias_multiplier', etc.
        
        Returns:
            Dictionary containing analysis of hidden bias mechanisms
        """
        analysis = {
            "timestamp": self._get_timestamp(),
            "hidden_mechanisms_detected": [],
            "gender_multipliers": {},
            "score_adjustments": {},
            "bias_patterns": []
        }
        
        # Analyze score adjustments by gender
        for gender in test_data['gender'].unique():
            gender_data = test_data[test_data['gender'] == gender]
            
            if 'base_score' in test_data.columns and 'final_score' in test_data.columns:
                base_scores = gender_data['base_score'].values
                final_scores = gender_data['final_score'].values
                
                # Calculate average adjustment factor
                adjustment_factors = final_scores / (base_scores + 1e-10)
                avg_adjustment = np.mean(adjustment_factors)
                
                analysis["gender_multipliers"][gender] = {
                    "average_multiplier": float(avg_adjustment),
                    "multiplier_std": float(np.std(adjustment_factors)),
                    "multiplier_range": (float(np.min(adjustment_factors)), 
                                        float(np.max(adjustment_factors)))
                }
                
                # Check if this represents a hidden bias mechanism
                if abs(avg_adjustment - 1.0) > 0.1:
                    analysis["hidden_mechanisms_detected"].append({
                        "type": "gender_score_adjustment",
                        "gender": gender,
                        "adjustment_factor": float(avg_adjustment),
                        "severity": "high" if abs(avg_adjustment - 1.0) > 0.3 else "medium"
                    })
        
        # Analyze bias multiplier patterns
        if 'bias_multiplier' in test_data.columns:
            for gender in test_data['gender'].unique():
                gender_data = test_data[test_data['gender'] == gender]
                multipliers = gender_data['bias_multiplier'].values
                
                analysis["score_adjustments"][gender] = {
                    "avg_bias_multiplier": float(np.mean(multipliers)),
                    "bias_multiplier_std": float(np.std(multipliers)),
                    "bias_multiplier_range": (float(np.min(multipliers)), 
                                             float(np.max(multipliers)))
                }
        
        # Identify specific bias patterns
        if 'bias_reasons' in test_data.columns:
            reason_counts = {}
            for reasons in test_data['bias_reasons']:
                if isinstance(reasons, list):
                    for reason in reasons:
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
            
            analysis["bias_patterns"] = [
                {"pattern": reason, "frequency": count}
                for reason, count in sorted(reason_counts.items(), 
                                          key=lambda x: x[1], reverse=True)
            ]
        
        # Calculate gender score ratio
        gender_groups = test_data.groupby('gender')['final_score'].agg(['mean', 'std', 'count'])
        if len(gender_groups) >= 2:
            genders = list(gender_groups.index)
            if 'male' in genders and 'female' in genders:
                male_avg = gender_groups.loc['male', 'mean']
                female_avg = gender_groups.loc['female', 'mean']
                
                analysis["gender_score_ratio"] = {
                    "male_to_female_ratio": float(male_avg / female_avg) if female_avg > 0 else float('inf'),
                    "male_advantage_percentage": float((male_avg - female_avg) / female_avg * 100) if female_avg > 0 else float('inf')
                }
        
        return analysis


def main():
    """Example usage of the BiasAnalyzer."""
    # This would typically be called with real data
    print("Bias Analyzer initialized.")
    print("This is a forensic-grade bias analysis tool for AI systems.")
    print("Use the BiasAnalyzer class methods to analyze your data.")


if __name__ == "__main__":
    main()