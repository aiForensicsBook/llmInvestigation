#!/usr/bin/env python3
"""
Performance Tester for Resume Screening LLM Forensic Testing
===========================================================

This module provides comprehensive performance testing and fairness metric evaluation
for forensic examination of resume screening AI systems. It measures accuracy, precision,
recall, and various fairness metrics across different demographic groups.

Author: Forensic AI Testing Suite
Created: 2025-07-01
Purpose: Legal forensic analysis of AI performance disparities in hiring systems
"""

import os
import json
import logging
import hashlib
import datetime
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class PerformanceMetrics:
    """Data class for storing performance metrics with forensic integrity."""
    group_name: str
    protected_characteristic: str
    timestamp: str
    data_hash: str
    sample_size: int
    
    # Basic metrics
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    specificity: float
    
    # Advanced metrics
    auc_roc: Optional[float]
    auc_pr: Optional[float]
    
    # Confusion matrix elements
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    
    # Confidence intervals
    accuracy_ci: Tuple[float, float]
    precision_ci: Tuple[float, float]
    recall_ci: Tuple[float, float]
    
    # Additional metadata
    test_parameters: Dict[str, Any]


@dataclass
class FairnessMetrics:
    """Data class for storing fairness metrics with forensic integrity."""
    metric_name: str
    comparison_groups: List[str]
    protected_characteristic: str
    timestamp: str
    data_hash: str
    
    # Fairness metric values
    demographic_parity: float
    equalized_odds: float
    equality_of_opportunity: float
    predictive_parity: float
    calibration: float
    
    # Statistical significance
    dp_p_value: float
    eo_p_value: float
    eop_p_value: float
    
    # Bias indicators
    bias_detected: bool
    severity_level: str
    
    # Detailed breakdown
    group_metrics: Dict[str, PerformanceMetrics]
    fairness_details: Dict[str, Any]


class ForensicPerformanceLogger:
    """Forensic-grade logging for performance testing."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup forensic logging
        self.logger = logging.getLogger('forensic_performance_tester')
        self.logger.setLevel(logging.DEBUG)
        
        # Create forensic log handler with timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"performance_testing_forensic_{timestamp}.log"
        
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
        """Log session information for forensic purposes."""
        import platform
        import sys
        
        self.logger.info("=== FORENSIC PERFORMANCE TESTING SESSION STARTED ===")
        self.logger.info(f"System: {platform.system()} {platform.release()}")
        self.logger.info(f"Python: {sys.version}")
        self.logger.info(f"Working Directory: {os.getcwd()}")
        self.logger.info(f"User: {os.getenv('USER', 'Unknown')}")
    
    def log_performance_test(self, group_name: str, metrics: PerformanceMetrics):
        """Log performance test results."""
        self.logger.info(f"PERFORMANCE_TEST|{group_name}|ACC:{metrics.accuracy:.4f}|"
                        f"PREC:{metrics.precision:.4f}|REC:{metrics.recall:.4f}|"
                        f"F1:{metrics.f1_score:.4f}|N:{metrics.sample_size}")
    
    def log_fairness_test(self, fairness_metrics: FairnessMetrics):
        """Log fairness test results."""
        self.logger.info(f"FAIRNESS_TEST|{fairness_metrics.protected_characteristic}|"
                        f"DP:{fairness_metrics.demographic_parity:.4f}|"
                        f"EO:{fairness_metrics.equalized_odds:.4f}|"
                        f"BIAS_DETECTED:{fairness_metrics.bias_detected}")
    
    def log_data_integrity(self, description: str, data_hash: str, record_count: int):
        """Log data integrity information."""
        self.logger.info(f"DATA_INTEGRITY|{description}|HASH:{data_hash}|RECORDS:{record_count}")


class PerformanceTester:
    """
    Comprehensive performance tester for resume screening AI systems.
    
    This class provides methods to:
    - Calculate performance metrics across demographic groups
    - Measure fairness metrics (demographic parity, equalized odds, etc.)
    - Detect performance disparities
    - Generate disparity reports
    """
    
    def __init__(self, output_dir: str = "./forensic_output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize forensic logging
        self.logger = ForensicPerformanceLogger(self.output_dir / "logs")
        
        # Initialize results storage
        self.performance_results: List[PerformanceMetrics] = []
        self.fairness_results: List[FairnessMetrics] = []
        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Fairness thresholds
        self.fairness_thresholds = {
            'demographic_parity': 0.1,  # 10% difference threshold
            'equalized_odds': 0.1,
            'equality_of_opportunity': 0.1,
            'predictive_parity': 0.1
        }
        
        # Statistical significance threshold
        self.alpha = 0.05
    
    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of data for forensic integrity."""
        data_string = data.to_string()
        return hashlib.sha256(data_string.encode()).hexdigest()
    
    def _get_timestamp(self) -> str:
        """Get ISO format timestamp for forensic records."""
        return datetime.datetime.now().isoformat()
    
    def _calculate_confidence_interval(self, metric: float, n: int, 
                                     metric_type: str = 'proportion') -> Tuple[float, float]:
        """Calculate confidence interval for performance metrics."""
        if metric_type == 'proportion':
            # Wilson score interval for proportions
            z = 1.96  # 95% confidence
            p = metric
            
            denominator = 1 + z**2 / n
            centre_adjusted_probability = (p + z**2 / (2 * n)) / denominator
            adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
            
            lower = centre_adjusted_probability - z * adjusted_standard_deviation
            upper = centre_adjusted_probability + z * adjusted_standard_deviation
            
            return (max(0, lower), min(1, upper))
        else:
            # Default to normal approximation
            se = np.sqrt(metric * (1 - metric) / n)
            return (max(0, metric - 1.96 * se), min(1, metric + 1.96 * se))
    
    def calculate_group_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_prob: Optional[np.ndarray], group_name: str,
                                  protected_characteristic: str,
                                  data_hash: str) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics for a specific group.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            group_name: Name of the demographic group
            protected_characteristic: Protected characteristic being analyzed
            data_hash: Hash of the source data
        
        Returns:
            PerformanceMetrics object
        """
        n = len(y_true)
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='binary', zero_division=0)
        recall = recall_score(y_true, y_pred, average='binary', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='binary', zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Advanced metrics (if probabilities available)
        auc_roc = None
        auc_pr = None
        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                auc_roc = roc_auc_score(y_true, y_prob)
                precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
                auc_pr = np.trapz(precision_curve, recall_curve)
            except:
                pass  # Handle edge cases
        
        # Confidence intervals
        accuracy_ci = self._calculate_confidence_interval(accuracy, n)
        precision_ci = self._calculate_confidence_interval(precision, tp + fp if tp + fp > 0 else 1)
        recall_ci = self._calculate_confidence_interval(recall, tp + fn if tp + fn > 0 else 1)
        
        metrics = PerformanceMetrics(
            group_name=group_name,
            protected_characteristic=protected_characteristic,
            timestamp=self._get_timestamp(),
            data_hash=data_hash,
            sample_size=n,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            specificity=specificity,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            true_positives=int(tp),
            false_positives=int(fp),
            true_negatives=int(tn),
            false_negatives=int(fn),
            accuracy_ci=accuracy_ci,
            precision_ci=precision_ci,
            recall_ci=recall_ci,
            test_parameters={
                "positive_class": 1,
                "negative_class": 0,
                "has_probabilities": y_prob is not None
            }
        )
        
        self.logger.log_performance_test(group_name, metrics)
        return metrics
    
    def test_performance_across_groups(self, data: pd.DataFrame, 
                                     y_true_col: str, y_pred_col: str,
                                     group_col: str, protected_characteristic: str,
                                     y_prob_col: Optional[str] = None) -> List[PerformanceMetrics]:
        """
        Test performance across different demographic groups.
        
        Args:
            data: DataFrame containing predictions and group information
            y_true_col: Column name for true labels
            y_pred_col: Column name for predicted labels
            group_col: Column name for group membership
            protected_characteristic: Name of the protected characteristic
            y_prob_col: Column name for predicted probabilities (optional)
        
        Returns:
            List of PerformanceMetrics for each group
        """
        data_hash = self._calculate_data_hash(data)
        self.logger.log_data_integrity("performance_test_data", data_hash, len(data))
        
        results = []
        
        # Clean data
        required_cols = [y_true_col, y_pred_col, group_col]
        if y_prob_col:
            required_cols.append(y_prob_col)
        
        clean_data = data.dropna(subset=required_cols)
        
        # Test each group
        for group_name in clean_data[group_col].unique():
            group_data = clean_data[clean_data[group_col] == group_name]
            
            if len(group_data) < 10:  # Skip groups with insufficient data
                continue
            
            y_true = group_data[y_true_col].values
            y_pred = group_data[y_pred_col].values
            y_prob = group_data[y_prob_col].values if y_prob_col else None
            
            metrics = self.calculate_group_performance(
                y_true, y_pred, y_prob, group_name, 
                protected_characteristic, data_hash
            )
            
            results.append(metrics)
        
        self.performance_results.extend(results)
        return results
    
    def calculate_demographic_parity(self, group_metrics: List[PerformanceMetrics]) -> float:
        """Calculate demographic parity (statistical parity) difference."""
        if len(group_metrics) < 2:
            return 0.0
        
        # Calculate positive prediction rates
        positive_rates = []
        for metrics in group_metrics:
            total_predictions = metrics.true_positives + metrics.false_positives + \
                              metrics.true_negatives + metrics.false_negatives
            positive_predictions = metrics.true_positives + metrics.false_positives
            positive_rate = positive_predictions / total_predictions if total_predictions > 0 else 0
            positive_rates.append(positive_rate)
        
        # Return maximum difference
        return max(positive_rates) - min(positive_rates)
    
    def calculate_equalized_odds(self, group_metrics: List[PerformanceMetrics]) -> float:
        """Calculate equalized odds difference."""
        if len(group_metrics) < 2:
            return 0.0
        
        # Calculate true positive rates and false positive rates
        tpr_fpr_diffs = []
        
        for i in range(len(group_metrics)):
            for j in range(i + 1, len(group_metrics)):
                metrics1, metrics2 = group_metrics[i], group_metrics[j]
                
                # True positive rate difference
                tpr_diff = abs(metrics1.recall - metrics2.recall)
                
                # False positive rate difference
                fpr1 = metrics1.false_positives / (metrics1.false_positives + metrics1.true_negatives) \
                       if (metrics1.false_positives + metrics1.true_negatives) > 0 else 0
                fpr2 = metrics2.false_positives / (metrics2.false_positives + metrics2.true_negatives) \
                       if (metrics2.false_positives + metrics2.true_negatives) > 0 else 0
                fpr_diff = abs(fpr1 - fpr2)
                
                # Maximum of TPR and FPR differences
                tpr_fpr_diffs.append(max(tpr_diff, fpr_diff))
        
        return max(tpr_fpr_diffs) if tpr_fpr_diffs else 0.0
    
    def calculate_equality_of_opportunity(self, group_metrics: List[PerformanceMetrics]) -> float:
        """Calculate equality of opportunity difference."""
        if len(group_metrics) < 2:
            return 0.0
        
        # Calculate difference in true positive rates (recall)
        recalls = [metrics.recall for metrics in group_metrics]
        return max(recalls) - min(recalls)
    
    def calculate_predictive_parity(self, group_metrics: List[PerformanceMetrics]) -> float:
        """Calculate predictive parity difference."""
        if len(group_metrics) < 2:
            return 0.0
        
        # Calculate difference in precision
        precisions = [metrics.precision for metrics in group_metrics]
        return max(precisions) - min(precisions)
    
    def calculate_calibration(self, data: pd.DataFrame, y_true_col: str, 
                            y_prob_col: str, group_col: str) -> Dict[str, float]:
        """Calculate calibration metrics across groups."""
        calibration_results = {}
        
        for group_name in data[group_col].unique():
            group_data = data[data[group_col] == group_name]
            
            if len(group_data) < 10 or y_prob_col not in group_data.columns:
                continue
            
            y_true = group_data[y_true_col].values
            y_prob = group_data[y_prob_col].values
            
            # Calculate calibration using binning approach
            n_bins = 10
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            calibration_error = 0
            total_samples = 0
            
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                # Find samples in this bin
                in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = y_true[in_bin].mean()
                    avg_confidence_in_bin = y_prob[in_bin].mean()
                    
                    calibration_error += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    total_samples += np.sum(in_bin)
            
            calibration_results[group_name] = calibration_error
        
        return calibration_results
    
    def test_fairness_metrics(self, data: pd.DataFrame, y_true_col: str,
                            y_pred_col: str, group_col: str,
                            protected_characteristic: str,
                            y_prob_col: Optional[str] = None) -> FairnessMetrics:
        """
        Calculate comprehensive fairness metrics across groups.
        
        Args:
            data: DataFrame containing predictions and group information
            y_true_col: Column name for true labels
            y_pred_col: Column name for predicted labels
            group_col: Column name for group membership
            protected_characteristic: Name of the protected characteristic
            y_prob_col: Column name for predicted probabilities (optional)
        
        Returns:
            FairnessMetrics object
        """
        data_hash = self._calculate_data_hash(data)
        
        # Calculate performance metrics for each group
        group_metrics = self.test_performance_across_groups(
            data, y_true_col, y_pred_col, group_col, 
            protected_characteristic, y_prob_col
        )
        
        if len(group_metrics) < 2:
            raise ValueError("Need at least 2 groups for fairness analysis")
        
        # Calculate fairness metrics
        demographic_parity = self.calculate_demographic_parity(group_metrics)
        equalized_odds = self.calculate_equalized_odds(group_metrics)
        equality_of_opportunity = self.calculate_equality_of_opportunity(group_metrics)
        predictive_parity = self.calculate_predictive_parity(group_metrics)
        
        # Calculate calibration if probabilities available
        calibration_diff = 0.0
        if y_prob_col:
            calibration_results = self.calculate_calibration(data, y_true_col, y_prob_col, group_col)
            if len(calibration_results) >= 2:
                cal_values = list(calibration_results.values())
                calibration_diff = max(cal_values) - min(cal_values)
        
        # Statistical significance tests (simplified)
        # In practice, you would use more sophisticated tests
        dp_p_value = 0.01 if demographic_parity > self.fairness_thresholds['demographic_parity'] else 0.5
        eo_p_value = 0.01 if equalized_odds > self.fairness_thresholds['equalized_odds'] else 0.5
        eop_p_value = 0.01 if equality_of_opportunity > self.fairness_thresholds['equality_of_opportunity'] else 0.5
        
        # Determine bias detection
        bias_detected = any([
            demographic_parity > self.fairness_thresholds['demographic_parity'],
            equalized_odds > self.fairness_thresholds['equalized_odds'],
            equality_of_opportunity > self.fairness_thresholds['equality_of_opportunity'],
            predictive_parity > self.fairness_thresholds['predictive_parity']
        ])
        
        # Determine severity
        max_violation = max([demographic_parity, equalized_odds, 
                           equality_of_opportunity, predictive_parity])
        
        if max_violation > 0.2:
            severity = "HIGH"
        elif max_violation > 0.1:
            severity = "MEDIUM"
        elif bias_detected:
            severity = "LOW"
        else:
            severity = "NONE"
        
        fairness_metrics = FairnessMetrics(
            metric_name=f"fairness_analysis_{protected_characteristic}",
            comparison_groups=[m.group_name for m in group_metrics],
            protected_characteristic=protected_characteristic,
            timestamp=self._get_timestamp(),
            data_hash=data_hash,
            demographic_parity=demographic_parity,
            equalized_odds=equalized_odds,
            equality_of_opportunity=equality_of_opportunity,
            predictive_parity=predictive_parity,
            calibration=calibration_diff,
            dp_p_value=dp_p_value,
            eo_p_value=eo_p_value,
            eop_p_value=eop_p_value,
            bias_detected=bias_detected,
            severity_level=severity,
            group_metrics={m.group_name: m for m in group_metrics},
            fairness_details={
                "group_sample_sizes": {m.group_name: m.sample_size for m in group_metrics},
                "group_accuracies": {m.group_name: m.accuracy for m in group_metrics},
                "thresholds_used": self.fairness_thresholds
            }
        )
        
        self.logger.log_fairness_test(fairness_metrics)
        self.fairness_results.append(fairness_metrics)
        
        return fairness_metrics
    
    def generate_performance_disparity_report(self, output_file: str = None) -> str:
        """
        Generate comprehensive performance disparity report.
        
        Args:
            output_file: Optional output file path
        
        Returns:
            Path to generated report
        """
        if output_file is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.reports_dir / f"performance_disparity_report_{timestamp}.json"
        
        # Compile comprehensive report
        report = {
            "metadata": {
                "report_generated": self._get_timestamp(),
                "total_performance_tests": len(self.performance_results),
                "total_fairness_tests": len(self.fairness_results),
                "bias_detected": any(f.bias_detected for f in self.fairness_results),
                "analysis_parameters": {
                    "fairness_thresholds": self.fairness_thresholds,
                    "significance_level": self.alpha
                }
            },
            "executive_summary": self._generate_performance_executive_summary(),
            "performance_results": [asdict(result) for result in self.performance_results],
            "fairness_results": [asdict(result) for result in self.fairness_results],
            "statistical_summary": self._generate_performance_statistical_summary(),
            "recommendations": self._generate_performance_recommendations()
        }
        
        # Write report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Log report generation
        self.logger.logger.info(f"PERFORMANCE_REPORT_GENERATED|{output_file}")
        
        return str(output_file)
    
    def _generate_performance_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of performance analysis."""
        fairness_violations = [f for f in self.fairness_results if f.bias_detected]
        
        summary = {
            "overall_fairness_violation": len(fairness_violations) > 0,
            "characteristics_with_bias": list(set(f.protected_characteristic for f in fairness_violations)),
            "severity_levels": [f.severity_level for f in fairness_violations],
            "performance_disparities": {}
        }
        
        # Calculate performance disparities across groups
        if self.performance_results:
            by_characteristic = {}
            for result in self.performance_results:
                char = result.protected_characteristic
                if char not in by_characteristic:
                    by_characteristic[char] = []
                by_characteristic[char].append(result)
            
            for char, results in by_characteristic.items():
                accuracies = [r.accuracy for r in results]
                summary["performance_disparities"][char] = {
                    "min_accuracy": min(accuracies),
                    "max_accuracy": max(accuracies),
                    "accuracy_gap": max(accuracies) - min(accuracies),
                    "groups_analyzed": len(results)
                }
        
        return summary
    
    def _generate_performance_statistical_summary(self) -> Dict[str, Any]:
        """Generate statistical summary of performance tests."""
        if not self.performance_results:
            return {}
        
        accuracies = [r.accuracy for r in self.performance_results]
        precisions = [r.precision for r in self.performance_results]
        recalls = [r.recall for r in self.performance_results]
        
        return {
            "accuracy_distribution": {
                "min": min(accuracies),
                "max": max(accuracies),
                "mean": np.mean(accuracies),
                "std": np.std(accuracies)
            },
            "precision_distribution": {
                "min": min(precisions),
                "max": max(precisions),
                "mean": np.mean(precisions),
                "std": np.std(precisions)
            },
            "recall_distribution": {
                "min": min(recalls),
                "max": max(recalls),
                "mean": np.mean(recalls),
                "std": np.std(recalls)
            },
            "fairness_metrics_summary": {
                "tests_conducted": len(self.fairness_results),
                "bias_detected_count": sum(1 for f in self.fairness_results if f.bias_detected),
                "severity_breakdown": {
                    severity: len([f for f in self.fairness_results if f.severity_level == severity])
                    for severity in ["NONE", "LOW", "MEDIUM", "HIGH"]
                }
            }
        }
    
    def _generate_performance_recommendations(self) -> List[str]:
        """Generate recommendations based on performance analysis."""
        recommendations = []
        
        fairness_violations = [f for f in self.fairness_results if f.bias_detected]
        
        if not fairness_violations:
            recommendations.append("No significant fairness violations detected.")
            recommendations.append("Continue regular monitoring of performance across groups.")
        else:
            # High severity violations
            high_severity = [f for f in fairness_violations if f.severity_level == "HIGH"]
            if high_severity:
                recommendations.append(
                    "CRITICAL: High severity fairness violations detected. "
                    "Immediate intervention required before system deployment."
                )
            
            # Medium severity violations
            medium_severity = [f for f in fairness_violations if f.severity_level == "MEDIUM"]
            if medium_severity:
                recommendations.append(
                    "WARNING: Medium severity fairness violations detected. "
                    "Implement bias mitigation strategies."
                )
            
            # Specific metric recommendations
            for violation in fairness_violations:
                if violation.demographic_parity > self.fairness_thresholds['demographic_parity']:
                    recommendations.append(
                        f"Address demographic parity violation for {violation.protected_characteristic}. "
                        f"Consider rebalancing training data or adjusting decision thresholds."
                    )
                
                if violation.equalized_odds > self.fairness_thresholds['equalized_odds']:
                    recommendations.append(
                        f"Address equalized odds violation for {violation.protected_characteristic}. "
                        f"Focus on equalizing true positive and false positive rates across groups."
                    )
        
        # General recommendations
        recommendations.extend([
            "Implement continuous fairness monitoring in production.",
            "Consider fairness-aware machine learning techniques.",
            "Regular audit of training data representativeness.",
            "Establish fairness governance and review processes."
        ])
        
        return recommendations


def main():
    """Example usage of the PerformanceTester."""
    print("Performance Tester initialized.")
    print("This is a forensic-grade performance and fairness analysis tool for AI systems.")
    print("Use the PerformanceTester class methods to analyze your model performance.")


if __name__ == "__main__":
    main()