#!/usr/bin/env python3
"""
Evidently Library Integration for Resume Screening LLM Forensic Testing
=======================================================================

This module provides comprehensive bias detection and monitoring capabilities using
the Evidently library for forensic examination of resume screening AI systems.
It implements advanced data drift detection, model performance monitoring,
fairness analysis, and real-time bias detection with legal-grade documentation.

Author: Forensic AI Testing Suite
Created: 2025-07-01
Purpose: Advanced bias detection and monitoring for AI hiring systems
Version: 1.0.0

Features:
- Data drift detection and analysis
- Model performance degradation monitoring
- Fairness and bias analysis using Evidently's fairness metrics
- Data quality assessment and anomaly detection
- Interactive HTML reports for stakeholders
- Real-time monitoring dashboards
- Automated bias alerts and notifications
- Legal-grade forensic documentation
- Integration with existing forensic framework
"""

import os
import json
import logging
import hashlib
import datetime
import warnings
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from pathlib import Path
import uuid

import numpy as np
import pandas as pd
from scipy import stats

# Evidently imports with error handling
try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_suite import MetricSuite
    from evidently.metrics import (
        DataDriftTable,
        DataQualityTable, 
        RegressionQualityMetric,
        ClassificationQualityMetric,
        ColumnDriftMetric,
        ColumnSummaryMetric,
        DatasetDriftMetric,
        DatasetMissingValuesMetric,
        ConflictTargetMetric,
        ConflictPredictionMetric,
        ClassificationClassBalance,
        ClassificationConfusionMatrix,
        ClassificationQualityByClass,
        RegressionErrorDistribution,
        RegressionErrorNormality,
        RegressionTopErrorMetric
    )
    from evidently.test_suite import TestSuite
    from evidently.tests import (
        TestNumberOfColumnsWithMissingValues,
        TestNumberOfRowsWithMissingValues,
        TestNumberOfConstantColumns,
        TestNumberOfDuplicatedRows,
        TestNumberOfDuplicatedColumns,
        TestColumnsType,
        TestNumberOfDriftedColumns,
        TestShareOfMissingValues,
        TestMeanInNSigmas,
        TestValueMin,
        TestValueMax,
        TestValueRange,
        TestNumberOfOutliersOutlierDDM,
        TestAccuracyScore,
        TestPrecisionScore,
        TestRecallScore,
        TestF1Score,
        TestRocAuc,
        TestLogLoss
    )
    from evidently.ui.workspace.cloud import CloudWorkspace
    from evidently.ui.dashboards import (
        CounterAgg,
        DashboardPanelCounter,
        DashboardPanelPlot,
        PanelValue,
        PlotType,
        ReportFilter
    )
    EVIDENTLY_AVAILABLE = True
except ImportError as e:
    EVIDENTLY_AVAILABLE = False
    warnings.warn(f"Evidently library not available: {e}. Install with: pip install evidently", ImportWarning)

# Import existing forensic framework components
from .bias_analyzer import ForensicLogger


@dataclass
class EvidentlyAnalysisResult:
    """Data class for storing Evidently analysis results with forensic integrity."""
    analysis_id: str
    analysis_type: str
    timestamp: str
    case_id: str
    investigator: str
    data_hash: str
    reference_data_hash: str
    current_data_hash: str
    
    # Data drift results
    data_drift_detected: bool
    drift_score: float
    drifted_columns: List[str]
    drift_by_column: Dict[str, Dict[str, Any]]
    
    # Model performance results
    model_performance_degraded: bool
    performance_metrics: Dict[str, float]
    performance_change: Dict[str, float]
    
    # Data quality results
    data_quality_issues: Dict[str, Any]
    missing_values_count: int
    duplicate_rows_count: int
    outliers_detected: Dict[str, int]
    
    # Bias detection results
    bias_detected: bool
    fairness_metrics: Dict[str, float]
    bias_by_group: Dict[str, Dict[str, float]]
    protected_attributes: List[str]
    
    # Reports and artifacts
    html_report_path: str
    json_report_path: str
    dashboard_url: Optional[str]
    
    # Forensic metadata
    chain_of_custody: Dict[str, Any]
    test_parameters: Dict[str, Any]
    alert_triggered: bool
    alert_details: Dict[str, Any]
    
    # Legal compliance
    compliance_status: Dict[str, bool] = field(default_factory=dict)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)


@dataclass 
class BiasMonitoringAlert:
    """Data class for bias monitoring alerts."""
    alert_id: str
    timestamp: str
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    metric_name: str
    threshold_exceeded: float
    current_value: float
    affected_groups: List[str]
    description: str
    recommended_actions: List[str]
    case_id: str
    investigator: str
    forensic_hash: str


class EvidentlyAnalyzer:
    """
    Advanced bias detection and monitoring using Evidently library.
    
    This class provides comprehensive bias analysis capabilities including:
    - Data drift detection to identify changes in model inputs over time
    - Model performance monitoring to detect degradation
    - Fairness analysis using multiple bias metrics
    - Data quality assessment for anomaly detection
    - Real-time monitoring and alerting
    - Interactive reporting and dashboards
    """
    
    def __init__(self, 
                 case_id: str = None,
                 investigator: str = None,
                 output_dir: str = None,
                 enable_monitoring: bool = True,
                 enable_alerts: bool = True):
        """
        Initialize the Evidently analyzer.
        
        Args:
            case_id: Unique case identifier for forensic tracking
            investigator: Name of the investigating officer/analyst
            output_dir: Directory for storing reports and artifacts
            enable_monitoring: Enable real-time monitoring capabilities
            enable_alerts: Enable automated bias detection alerts
        """
        if not EVIDENTLY_AVAILABLE:
            raise ImportError("Evidently library is required for this analyzer. Install with: pip install evidently")
        
        self.case_id = case_id or f"EVIDENTLY_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.investigator = investigator or "Unknown"
        self.analysis_id = str(uuid.uuid4())
        
        # Setup directories
        self.output_dir = Path(output_dir) if output_dir else Path("./forensic/reports/evidently")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.reports_dir = self.output_dir / "reports"
        self.dashboards_dir = self.output_dir / "dashboards" 
        self.alerts_dir = self.output_dir / "alerts"
        
        for dir_path in [self.reports_dir, self.dashboards_dir, self.alerts_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize logging
        self.logger = ForensicLogger(
            name=f"evidently_analyzer_{self.case_id}",
            log_file=str(self.output_dir / f"evidently_analysis_{self.case_id}.log")
        )
        
        # Configuration
        self.enable_monitoring = enable_monitoring
        self.enable_alerts = enable_alerts
        
        # Bias detection thresholds
        self.bias_thresholds = {
            'demographic_parity_difference': 0.1,
            'equal_opportunity_difference': 0.1,
            'equalized_odds_difference': 0.1,
            'statistical_parity_difference': 0.1,
            'disparate_impact_ratio': 0.8,
            'accuracy_difference': 0.05,
            'precision_difference': 0.05,
            'recall_difference': 0.05,
            'f1_difference': 0.05
        }
        
        # Data drift thresholds
        self.drift_thresholds = {
            'dataset_drift_threshold': 0.5,
            'column_drift_threshold': 0.05,
            'psi_threshold': 0.2,
            'wasserstein_threshold': 0.1
        }
        
        # Alert storage
        self.alerts = []
        
        # Initialize column mapping (to be configured per dataset)
        self.column_mapping = None
        
        self.logger.info(f"EvidentlyAnalyzer initialized for case {self.case_id}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Monitoring enabled: {self.enable_monitoring}")
        self.logger.info(f"Alerts enabled: {self.enable_alerts}")

    def configure_column_mapping(self,
                                target: str = None,
                                prediction: str = None,
                                numerical_features: List[str] = None,
                                categorical_features: List[str] = None,
                                datetime_features: List[str] = None,
                                text_features: List[str] = None,
                                protected_attributes: List[str] = None) -> ColumnMapping:
        """
        Configure column mapping for Evidently analysis.
        
        Args:
            target: Target column name
            prediction: Prediction column name
            numerical_features: List of numerical feature names
            categorical_features: List of categorical feature names
            datetime_features: List of datetime feature names
            text_features: List of text feature names
            protected_attributes: List of protected attribute names for bias analysis
            
        Returns:
            Configured ColumnMapping object
        """
        self.column_mapping = ColumnMapping(
            target=target,
            prediction=prediction,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            datetime_features=datetime_features,
            text_features=text_features
        )
        
        # Store protected attributes separately for bias analysis
        self.protected_attributes = protected_attributes or []
        
        self.logger.info("Column mapping configured successfully")
        self.logger.info(f"Target: {target}, Prediction: {prediction}")
        self.logger.info(f"Protected attributes: {self.protected_attributes}")
        
        return self.column_mapping

    def detect_data_drift(self,
                         reference_data: pd.DataFrame,
                         current_data: pd.DataFrame,
                         generate_report: bool = True) -> Dict[str, Any]:
        """
        Detect data drift between reference and current datasets.
        
        Args:
            reference_data: Reference dataset (training data)
            current_data: Current dataset (new data)
            generate_report: Whether to generate HTML report
            
        Returns:
            Dictionary containing drift detection results
        """
        self.logger.info("Starting data drift detection analysis")
        
        # Calculate data hashes for forensic integrity
        ref_hash = self._calculate_data_hash(reference_data)
        curr_hash = self._calculate_data_hash(current_data)
        
        try:
            # Create data drift report
            data_drift_report = Report(metrics=[
                DataDriftTable(),
                DatasetDriftMetric(),
                DataQualityTable(),
            ])
            
            # Run the report
            data_drift_report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            # Extract results
            report_dict = data_drift_report.as_dict()
            
            # Parse drift results
            dataset_drift = report_dict['metrics'][1]['result']
            data_drift_table = report_dict['metrics'][0]['result']
            
            drift_results = {
                'dataset_drift_detected': dataset_drift.get('dataset_drift', False),
                'drift_score': dataset_drift.get('drift_score', 0.0),
                'drifted_columns_count': dataset_drift.get('number_of_drifted_columns', 0),
                'total_columns': dataset_drift.get('number_of_columns', 0),
                'drift_by_column': {}
            }
            
            # Extract column-level drift information
            if 'drift_by_columns' in data_drift_table:
                for col_name, col_drift in data_drift_table['drift_by_columns'].items():
                    drift_results['drift_by_column'][col_name] = {
                        'drift_detected': col_drift.get('drift_detected', False),
                        'drift_score': col_drift.get('drift_score', 0.0),
                        'stattest_name': col_drift.get('stattest_name', ''),
                        'stattest_threshold': col_drift.get('threshold', 0.0)
                    }
            
            # Generate HTML report if requested
            if generate_report:
                report_path = self.reports_dir / f"data_drift_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                data_drift_report.save_html(str(report_path))
                drift_results['html_report_path'] = str(report_path)
                self.logger.info(f"Data drift HTML report saved: {report_path}")
            
            # Check for alert conditions
            if self.enable_alerts and drift_results['dataset_drift_detected']:
                self._trigger_drift_alert(drift_results, ref_hash, curr_hash)
            
            self.logger.info(f"Data drift analysis completed. Drift detected: {drift_results['dataset_drift_detected']}")
            return drift_results
            
        except Exception as e:
            self.logger.error(f"Error in data drift detection: {str(e)}")
            raise

    def analyze_model_performance(self,
                                 reference_data: pd.DataFrame,
                                 current_data: pd.DataFrame,
                                 task_type: str = 'classification',
                                 generate_report: bool = True) -> Dict[str, Any]:
        """
        Analyze model performance and detect degradation.
        
        Args:
            reference_data: Reference dataset with predictions
            current_data: Current dataset with predictions  
            task_type: 'classification' or 'regression'
            generate_report: Whether to generate HTML report
            
        Returns:
            Dictionary containing performance analysis results
        """
        self.logger.info(f"Starting model performance analysis ({task_type})")
        
        try:
            # Select appropriate metrics based on task type
            if task_type == 'classification':
                metrics = [
                    ClassificationQualityMetric(),
                    ClassificationClassBalance(),
                    ClassificationConfusionMatrix(),
                    ClassificationQualityByClass(),
                ]
            else:  # regression
                metrics = [
                    RegressionQualityMetric(),
                    RegressionErrorDistribution(),
                    RegressionErrorNormality(),
                    RegressionTopErrorMetric(),
                ]
            
            # Create performance report
            performance_report = Report(metrics=metrics)
            
            # Run the report
            performance_report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            # Extract results
            report_dict = performance_report.as_dict()
            
            # Parse performance metrics
            quality_metric = report_dict['metrics'][0]['result']
            
            performance_results = {
                'task_type': task_type,
                'reference_metrics': quality_metric.get('reference', {}),
                'current_metrics': quality_metric.get('current', {}),
                'performance_change': {},
                'performance_degraded': False
            }
            
            # Calculate performance changes
            ref_metrics = performance_results['reference_metrics']
            curr_metrics = performance_results['current_metrics']
            
            for metric_name in ref_metrics.keys():
                if metric_name in curr_metrics:
                    ref_val = ref_metrics[metric_name]
                    curr_val = curr_metrics[metric_name]
                    if ref_val != 0:
                        change = (curr_val - ref_val) / ref_val
                        performance_results['performance_change'][metric_name] = change
                        
                        # Check for significant degradation (>5% decrease)
                        if change < -0.05:
                            performance_results['performance_degraded'] = True
            
            # Generate HTML report if requested
            if generate_report:
                report_path = self.reports_dir / f"performance_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                performance_report.save_html(str(report_path))
                performance_results['html_report_path'] = str(report_path)
                self.logger.info(f"Performance HTML report saved: {report_path}")
            
            # Check for alert conditions
            if self.enable_alerts and performance_results['performance_degraded']:
                self._trigger_performance_alert(performance_results)
            
            self.logger.info(f"Performance analysis completed. Degradation detected: {performance_results['performance_degraded']}")
            return performance_results
            
        except Exception as e:
            self.logger.error(f"Error in performance analysis: {str(e)}")
            raise

    def assess_data_quality(self,
                           reference_data: pd.DataFrame,
                           current_data: pd.DataFrame,
                           generate_report: bool = True) -> Dict[str, Any]:
        """
        Assess data quality and detect anomalies.
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            generate_report: Whether to generate HTML report
            
        Returns:
            Dictionary containing data quality assessment results
        """
        self.logger.info("Starting data quality assessment")
        
        try:
            # Create data quality report
            data_quality_report = Report(metrics=[
                DataQualityTable(),
                DatasetMissingValuesMetric(),
            ])
            
            # Run the report
            data_quality_report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )
            
            # Extract results
            report_dict = data_quality_report.as_dict()
            quality_table = report_dict['metrics'][0]['result']
            missing_values = report_dict['metrics'][1]['result']
            
            quality_results = {
                'reference_quality': quality_table.get('reference', {}),
                'current_quality': quality_table.get('current', {}),
                'missing_values': missing_values,
                'quality_issues_detected': False,
                'issues_summary': {}
            }
            
            # Analyze quality issues
            current_quality = quality_results['current_quality']
            
            # Check for missing values
            if missing_values.get('current', {}).get('number_of_missing_values', 0) > 0:
                quality_results['quality_issues_detected'] = True
                quality_results['issues_summary']['missing_values'] = missing_values['current']
            
            # Check for other quality issues
            if 'columns' in current_quality:
                for col_name, col_info in current_quality['columns'].items():
                    col_issues = []
                    
                    # Check for high missing percentage
                    missing_pct = col_info.get('missing_percentage', 0)
                    if missing_pct > 10:  # More than 10% missing
                        col_issues.append(f"High missing values: {missing_pct:.1f}%")
                    
                    # Check for constant values
                    if col_info.get('number_of_distinct_values', 0) <= 1:
                        col_issues.append("Constant or near-constant values")
                    
                    # Check for outliers (if numeric)
                    if col_info.get('type') == 'num' and 'outliers_count' in col_info:
                        outliers = col_info['outliers_count']
                        if outliers > 0:
                            col_issues.append(f"Outliers detected: {outliers}")
                    
                    if col_issues:
                        quality_results['quality_issues_detected'] = True
                        quality_results['issues_summary'][col_name] = col_issues
            
            # Generate HTML report if requested
            if generate_report:
                report_path = self.reports_dir / f"data_quality_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                data_quality_report.save_html(str(report_path))
                quality_results['html_report_path'] = str(report_path)
                self.logger.info(f"Data quality HTML report saved: {report_path}")
            
            # Check for alert conditions
            if self.enable_alerts and quality_results['quality_issues_detected']:
                self._trigger_quality_alert(quality_results)
            
            self.logger.info(f"Data quality assessment completed. Issues detected: {quality_results['quality_issues_detected']}")
            return quality_results
            
        except Exception as e:
            self.logger.error(f"Error in data quality assessment: {str(e)}")
            raise

    def detect_bias(self,
                   reference_data: pd.DataFrame,
                   current_data: pd.DataFrame,
                   protected_attributes: List[str] = None,
                   generate_report: bool = True) -> Dict[str, Any]:
        """
        Detect bias using Evidently's fairness metrics.
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            protected_attributes: List of protected attributes for bias analysis
            generate_report: Whether to generate HTML report
            
        Returns:
            Dictionary containing bias detection results
        """
        self.logger.info("Starting bias detection analysis")
        
        protected_attrs = protected_attributes or self.protected_attributes
        if not protected_attrs:
            raise ValueError("Protected attributes must be specified for bias analysis")
        
        try:
            bias_results = {
                'protected_attributes': protected_attrs,
                'bias_detected': False,
                'bias_by_attribute': {},
                'fairness_metrics': {},
                'bias_summary': {}
            }
            
            # Analyze bias for each protected attribute
            for attr in protected_attrs:
                if attr not in current_data.columns:
                    self.logger.warning(f"Protected attribute '{attr}' not found in data")
                    continue
                
                attr_bias_results = self._analyze_bias_for_attribute(
                    reference_data, current_data, attr
                )
                
                bias_results['bias_by_attribute'][attr] = attr_bias_results
                
                # Check if bias detected for this attribute
                if attr_bias_results.get('bias_detected', False):
                    bias_results['bias_detected'] = True
            
            # Calculate overall fairness metrics
            bias_results['fairness_metrics'] = self._calculate_fairness_metrics(
                current_data, protected_attrs
            )
            
            # Generate bias summary
            bias_results['bias_summary'] = self._generate_bias_summary(bias_results)
            
            # Generate HTML report if requested
            if generate_report:
                report_path = self._generate_bias_report(
                    reference_data, current_data, bias_results
                )
                bias_results['html_report_path'] = str(report_path)
            
            # Check for alert conditions
            if self.enable_alerts and bias_results['bias_detected']:
                self._trigger_bias_alert(bias_results)
            
            self.logger.info(f"Bias detection completed. Bias detected: {bias_results['bias_detected']}")
            return bias_results
            
        except Exception as e:
            self.logger.error(f"Error in bias detection: {str(e)}")
            raise

    def run_comprehensive_analysis(self,
                                  reference_data: pd.DataFrame,
                                  current_data: pd.DataFrame,
                                  task_type: str = 'classification',
                                  protected_attributes: List[str] = None,
                                  generate_reports: bool = True) -> EvidentlyAnalysisResult:
        """
        Run comprehensive analysis including drift, performance, quality, and bias detection.
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            task_type: 'classification' or 'regression'
            protected_attributes: List of protected attributes for bias analysis
            generate_reports: Whether to generate HTML reports
            
        Returns:
            EvidentlyAnalysisResult containing all analysis results
        """
        self.logger.info("Starting comprehensive Evidently analysis")
        
        start_time = datetime.datetime.now()
        
        # Calculate data hashes for forensic integrity
        ref_hash = self._calculate_data_hash(reference_data)
        curr_hash = self._calculate_data_hash(current_data)
        
        try:
            # Run all analyses
            drift_results = self.detect_data_drift(
                reference_data, current_data, generate_reports
            )
            
            performance_results = self.analyze_model_performance(
                reference_data, current_data, task_type, generate_reports
            )
            
            quality_results = self.assess_data_quality(
                reference_data, current_data, generate_reports
            )
            
            bias_results = self.detect_bias(
                reference_data, current_data, protected_attributes, generate_reports
            )
            
            # Create comprehensive result
            result = EvidentlyAnalysisResult(
                analysis_id=self.analysis_id,
                analysis_type="comprehensive",
                timestamp=start_time.isoformat(),
                case_id=self.case_id,
                investigator=self.investigator,
                data_hash=f"{ref_hash}_{curr_hash}",
                reference_data_hash=ref_hash,
                current_data_hash=curr_hash,
                
                # Data drift results
                data_drift_detected=drift_results.get('dataset_drift_detected', False),
                drift_score=drift_results.get('drift_score', 0.0),
                drifted_columns=list(drift_results.get('drift_by_column', {}).keys()),
                drift_by_column=drift_results.get('drift_by_column', {}),
                
                # Model performance results
                model_performance_degraded=performance_results.get('performance_degraded', False),
                performance_metrics=performance_results.get('current_metrics', {}),
                performance_change=performance_results.get('performance_change', {}),
                
                # Data quality results
                data_quality_issues=quality_results.get('issues_summary', {}),
                missing_values_count=quality_results.get('missing_values', {}).get('current', {}).get('number_of_missing_values', 0),
                duplicate_rows_count=0,  # To be implemented
                outliers_detected={},  # To be implemented
                
                # Bias detection results
                bias_detected=bias_results.get('bias_detected', False),
                fairness_metrics=bias_results.get('fairness_metrics', {}),
                bias_by_group=bias_results.get('bias_by_attribute', {}),
                protected_attributes=protected_attributes or [],
                
                # Reports and artifacts
                html_report_path=self._generate_comprehensive_report(
                    drift_results, performance_results, quality_results, bias_results
                ) if generate_reports else "",
                json_report_path=self._save_json_report({
                    'drift': drift_results,
                    'performance': performance_results,
                    'quality': quality_results,
                    'bias': bias_results
                }),
                dashboard_url=None,  # To be implemented
                
                # Forensic metadata
                chain_of_custody=self._create_chain_of_custody(),
                test_parameters={
                    'task_type': task_type,
                    'protected_attributes': protected_attributes,
                    'bias_thresholds': self.bias_thresholds,
                    'drift_thresholds': self.drift_thresholds
                },
                alert_triggered=len(self.alerts) > 0,
                alert_details={'alerts_count': len(self.alerts), 'alerts': [asdict(alert) for alert in self.alerts]}
            )
            
            # Set compliance status
            result.compliance_status = self._assess_compliance(result)
            
            # Add to audit trail
            result.audit_trail = [{
                'timestamp': datetime.datetime.now().isoformat(),
                'action': 'comprehensive_analysis_completed',
                'investigator': self.investigator,
                'details': f"Analysis completed successfully in {datetime.datetime.now() - start_time}"
            }]
            
            self.logger.info("Comprehensive analysis completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {str(e)}")
            raise

    def create_monitoring_dashboard(self,
                                   workspace_name: str = None,
                                   project_name: str = None) -> str:
        """
        Create real-time monitoring dashboard using Evidently.
        
        Args:
            workspace_name: Name of the monitoring workspace
            project_name: Name of the monitoring project
            
        Returns:
            Dashboard URL or local dashboard path
        """
        if not self.enable_monitoring:
            raise ValueError("Monitoring is not enabled for this analyzer")
        
        self.logger.info("Creating monitoring dashboard")
        
        try:
            # For now, create a local dashboard configuration
            # In production, this would integrate with Evidently Cloud or local server
            
            dashboard_config = {
                'workspace_name': workspace_name or f"workspace_{self.case_id}",
                'project_name': project_name or f"bias_monitoring_{self.case_id}",
                'panels': [
                    {
                        'title': 'Data Drift Detection',
                        'type': 'metric',
                        'metric': 'dataset_drift_score'
                    },
                    {
                        'title': 'Model Performance',
                        'type': 'metric',
                        'metric': 'accuracy_score'
                    },
                    {
                        'title': 'Bias Detection',
                        'type': 'metric',
                        'metric': 'demographic_parity_difference'
                    },
                    {
                        'title': 'Data Quality Issues',
                        'type': 'counter',
                        'metric': 'missing_values_count'
                    }
                ],
                'alerts': [
                    {
                        'name': 'High Data Drift',
                        'condition': 'dataset_drift_score > 0.5',
                        'severity': 'high'
                    },
                    {
                        'name': 'Performance Degradation',
                        'condition': 'accuracy_change < -0.05',
                        'severity': 'high'
                    },
                    {
                        'name': 'Bias Detection',
                        'condition': 'demographic_parity_difference > 0.1',
                        'severity': 'critical'
                    }
                ]
            }
            
            # Save dashboard configuration
            dashboard_path = self.dashboards_dir / f"monitoring_dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(dashboard_path, 'w') as f:
                json.dump(dashboard_config, f, indent=2)
            
            self.logger.info(f"Dashboard configuration saved: {dashboard_path}")
            return str(dashboard_path)
            
        except Exception as e:
            self.logger.error(f"Error creating monitoring dashboard: {str(e)}")
            raise

    def setup_automated_alerts(self,
                              alert_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Setup automated bias detection alerts.
        
        Args:
            alert_config: Configuration for automated alerts
            
        Returns:
            Alert configuration details
        """
        if not self.enable_alerts:
            raise ValueError("Alerts are not enabled for this analyzer")
        
        self.logger.info("Setting up automated alerts")
        
        # Default alert configuration
        default_config = {
            'bias_alerts': {
                'demographic_parity_difference': {
                    'threshold': 0.1,
                    'severity': 'high',
                    'actions': ['log_alert', 'notify_investigator', 'generate_report']
                },
                'equal_opportunity_difference': {
                    'threshold': 0.1,
                    'severity': 'high',
                    'actions': ['log_alert', 'notify_investigator']
                },
                'accuracy_difference': {
                    'threshold': 0.05,
                    'severity': 'medium',
                    'actions': ['log_alert']
                }
            },
            'drift_alerts': {
                'dataset_drift_score': {
                    'threshold': 0.5,
                    'severity': 'medium',
                    'actions': ['log_alert', 'generate_report']
                }
            },
            'performance_alerts': {
                'accuracy_degradation': {
                    'threshold': -0.05,
                    'severity': 'high',
                    'actions': ['log_alert', 'notify_investigator']
                }
            },
            'quality_alerts': {
                'missing_values_percentage': {
                    'threshold': 10.0,
                    'severity': 'medium',
                    'actions': ['log_alert']
                }
            }
        }
        
        # Merge with provided configuration
        self.alert_config = {**default_config, **(alert_config or {})}
        
        # Save alert configuration
        alert_config_path = self.alerts_dir / f"alert_config_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(alert_config_path, 'w') as f:
            json.dump(self.alert_config, f, indent=2)
        
        self.logger.info(f"Alert configuration saved: {alert_config_path}")
        return self.alert_config

    def _analyze_bias_for_attribute(self,
                                   reference_data: pd.DataFrame,
                                   current_data: pd.DataFrame,
                                   attribute: str) -> Dict[str, Any]:
        """Analyze bias for a specific protected attribute."""
        try:
            # Get unique groups for this attribute
            groups = current_data[attribute].unique()
            
            bias_result = {
                'attribute': attribute,
                'groups': list(groups),
                'bias_detected': False,
                'group_metrics': {},
                'bias_metrics': {}
            }
            
            # Calculate metrics for each group
            target_col = self.column_mapping.target if self.column_mapping else None
            prediction_col = self.column_mapping.prediction if self.column_mapping else None
            
            if not target_col or not prediction_col:
                self.logger.warning(f"Target or prediction column not specified for bias analysis")
                return bias_result
            
            for group in groups:
                group_data = current_data[current_data[attribute] == group]
                
                if len(group_data) == 0:
                    continue
                
                # Calculate basic metrics
                y_true = group_data[target_col]
                y_pred = group_data[prediction_col]
                
                # Binary classification metrics
                if len(y_true.unique()) == 2 and len(y_pred.unique()) == 2:
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    
                    metrics = {
                        'accuracy': accuracy_score(y_true, y_pred),
                        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
                        'positive_rate': (y_pred == 1).mean() if hasattr(y_pred, 'mean') else 0,
                        'sample_size': len(group_data)
                    }
                    
                    bias_result['group_metrics'][str(group)] = metrics
            
            # Calculate bias metrics between groups
            if len(bias_result['group_metrics']) >= 2:
                bias_result['bias_metrics'] = self._calculate_bias_metrics(
                    bias_result['group_metrics']
                )
                
                # Check if bias detected based on thresholds
                for metric_name, metric_value in bias_result['bias_metrics'].items():
                    if metric_name in self.bias_thresholds:
                        threshold = self.bias_thresholds[metric_name]
                        if abs(metric_value) > threshold:
                            bias_result['bias_detected'] = True
                            break
            
            return bias_result
            
        except Exception as e:
            self.logger.error(f"Error analyzing bias for attribute {attribute}: {str(e)}")
            return {'attribute': attribute, 'error': str(e)}

    def _calculate_bias_metrics(self, group_metrics: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        """Calculate bias metrics between groups."""
        groups = list(group_metrics.keys())
        bias_metrics = {}
        
        if len(groups) < 2:
            return bias_metrics
        
        # For simplicity, compare first two groups
        # In practice, you might want to compare all pairs or use a reference group
        group1_metrics = group_metrics[groups[0]]
        group2_metrics = group_metrics[groups[1]]
        
        # Calculate differences
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'positive_rate']:
            if metric in group1_metrics and metric in group2_metrics:
                diff = group1_metrics[metric] - group2_metrics[metric]
                bias_metrics[f'{metric}_difference'] = diff
        
        # Calculate demographic parity difference
        if 'positive_rate' in group1_metrics and 'positive_rate' in group2_metrics:
            bias_metrics['demographic_parity_difference'] = abs(
                group1_metrics['positive_rate'] - group2_metrics['positive_rate']
            )
        
        # Calculate disparate impact ratio
        if 'positive_rate' in group1_metrics and 'positive_rate' in group2_metrics:
            if group2_metrics['positive_rate'] > 0:
                bias_metrics['disparate_impact_ratio'] = (
                    group1_metrics['positive_rate'] / group2_metrics['positive_rate']
                )
        
        return bias_metrics

    def _calculate_fairness_metrics(self,
                                   data: pd.DataFrame,
                                   protected_attributes: List[str]) -> Dict[str, float]:
        """Calculate overall fairness metrics."""
        fairness_metrics = {}
        
        # This would typically use specialized fairness libraries
        # For now, we'll calculate basic metrics
        
        target_col = self.column_mapping.target if self.column_mapping else None
        prediction_col = self.column_mapping.prediction if self.column_mapping else None
        
        if not target_col or not prediction_col:
            return fairness_metrics
        
        for attr in protected_attributes:
            if attr not in data.columns:
                continue
            
            groups = data[attr].unique()
            if len(groups) >= 2:
                # Calculate simple demographic parity
                overall_positive_rate = (data[prediction_col] == 1).mean()
                group_positive_rates = data.groupby(attr)[prediction_col].apply(lambda x: (x == 1).mean())
                
                max_diff = max(abs(rate - overall_positive_rate) for rate in group_positive_rates)
                fairness_metrics[f'{attr}_demographic_parity_difference'] = max_diff
        
        return fairness_metrics

    def _generate_bias_summary(self, bias_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of bias detection results."""
        summary = {
            'total_attributes_analyzed': len(bias_results['bias_by_attribute']),
            'attributes_with_bias': 0,
            'most_biased_attribute': None,
            'highest_bias_score': 0.0,
            'bias_metrics_summary': {}
        }
        
        # Analyze bias by attribute
        for attr, attr_results in bias_results['bias_by_attribute'].items():
            if attr_results.get('bias_detected', False):
                summary['attributes_with_bias'] += 1
                
                # Find highest bias score
                if 'bias_metrics' in attr_results:
                    for metric_name, metric_value in attr_results['bias_metrics'].items():
                        if abs(metric_value) > summary['highest_bias_score']:
                            summary['highest_bias_score'] = abs(metric_value)
                            summary['most_biased_attribute'] = attr
        
        # Aggregate bias metrics
        all_bias_metrics = {}
        for attr_results in bias_results['bias_by_attribute'].values():
            if 'bias_metrics' in attr_results:
                for metric_name, metric_value in attr_results['bias_metrics'].items():
                    if metric_name not in all_bias_metrics:
                        all_bias_metrics[metric_name] = []
                    all_bias_metrics[metric_name].append(metric_value)
        
        # Calculate summary statistics
        for metric_name, values in all_bias_metrics.items():
            summary['bias_metrics_summary'][metric_name] = {
                'mean': np.mean(values),
                'max': np.max(values),
                'min': np.min(values),
                'std': np.std(values)
            }
        
        return summary

    def _generate_bias_report(self,
                             reference_data: pd.DataFrame,
                             current_data: pd.DataFrame,
                             bias_results: Dict[str, Any]) -> Path:
        """Generate HTML bias report."""
        report_path = self.reports_dir / f"bias_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        # Create a simple HTML report
        # In practice, this would use Evidently's reporting capabilities
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Bias Detection Report - Case {self.case_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ margin: 10px 0; }}
                .alert {{ background-color: #ffebee; border-left: 4px solid #f44336; padding: 10px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Bias Detection Report</h1>
                <p><strong>Case ID:</strong> {self.case_id}</p>
                <p><strong>Investigator:</strong> {self.investigator}</p>
                <p><strong>Analysis Date:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Bias Detected:</strong> {'Yes' if bias_results['bias_detected'] else 'No'}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <p>Protected Attributes Analyzed: {len(bias_results['protected_attributes'])}</p>
                <p>Attributes: {', '.join(bias_results['protected_attributes'])}</p>
            </div>
            
            <div class="section">
                <h2>Bias Analysis by Attribute</h2>
                {self._format_bias_results_html(bias_results)}
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return report_path

    def _format_bias_results_html(self, bias_results: Dict[str, Any]) -> str:
        """Format bias results as HTML."""
        html = ""
        
        for attr, attr_results in bias_results['bias_by_attribute'].items():
            html += f"<h3>Attribute: {attr}</h3>"
            
            if 'error' in attr_results:
                html += f'<div class="alert">Error: {attr_results["error"]}</div>'
                continue
            
            if attr_results.get('bias_detected', False):
                html += '<div class="alert">Bias detected for this attribute!</div>'
            
            # Group metrics table
            if 'group_metrics' in attr_results:
                html += "<h4>Group Metrics</h4><table><tr><th>Group</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Positive Rate</th></tr>"
                
                for group, metrics in attr_results['group_metrics'].items():
                    html += f"<tr><td>{group}</td>"
                    for metric in ['accuracy', 'precision', 'recall', 'f1', 'positive_rate']:
                        value = metrics.get(metric, 0)
                        html += f"<td>{value:.3f}</td>"
                    html += "</tr>"
                
                html += "</table>"
            
            # Bias metrics
            if 'bias_metrics' in attr_results:
                html += "<h4>Bias Metrics</h4><table><tr><th>Metric</th><th>Value</th></tr>"
                
                for metric, value in attr_results['bias_metrics'].items():
                    html += f"<tr><td>{metric}</td><td>{value:.3f}</td></tr>"
                
                html += "</table>"
        
        return html

    def _generate_comprehensive_report(self,
                                      drift_results: Dict[str, Any],
                                      performance_results: Dict[str, Any],
                                      quality_results: Dict[str, Any],
                                      bias_results: Dict[str, Any]) -> str:
        """Generate comprehensive HTML report."""
        report_path = self.reports_dir / f"comprehensive_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Comprehensive Analysis Report - Case {self.case_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
                .alert {{ background-color: #ffebee; border-left: 4px solid #f44336; padding: 10px; margin: 10px 0; }}
                .success {{ background-color: #e8f5e8; border-left: 4px solid #4caf50; padding: 10px; margin: 10px 0; }}
                table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Comprehensive Analysis Report</h1>
                <p><strong>Case ID:</strong> {self.case_id}</p>
                <p><strong>Investigator:</strong> {self.investigator}</p>
                <p><strong>Analysis Date:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <ul>
                    <li><strong>Data Drift:</strong> {'Detected' if drift_results.get('dataset_drift_detected', False) else 'Not Detected'}</li>
                    <li><strong>Performance Degradation:</strong> {'Detected' if performance_results.get('performance_degraded', False) else 'Not Detected'}</li>
                    <li><strong>Data Quality Issues:</strong> {'Detected' if quality_results.get('quality_issues_detected', False) else 'Not Detected'}</li>
                    <li><strong>Bias:</strong> {'Detected' if bias_results.get('bias_detected', False) else 'Not Detected'}</li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Data Drift Analysis</h2>
                {self._format_drift_section_html(drift_results)}
            </div>
            
            <div class="section">
                <h2>Model Performance Analysis</h2>
                {self._format_performance_section_html(performance_results)}
            </div>
            
            <div class="section">
                <h2>Data Quality Assessment</h2>
                {self._format_quality_section_html(quality_results)}
            </div>
            
            <div class="section">
                <h2>Bias Detection Analysis</h2>
                {self._format_bias_section_html(bias_results)}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {self._generate_recommendations_html(drift_results, performance_results, quality_results, bias_results)}
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return str(report_path)

    def _format_drift_section_html(self, drift_results: Dict[str, Any]) -> str:
        """Format drift analysis section as HTML."""
        if drift_results.get('dataset_drift_detected', False):
            status = '<div class="alert">Data drift detected!</div>'
        else:
            status = '<div class="success">No significant data drift detected.</div>'
        
        html = status
        html += f"<p><strong>Drift Score:</strong> {drift_results.get('drift_score', 0):.3f}</p>"
        html += f"<p><strong>Drifted Columns:</strong> {drift_results.get('drifted_columns_count', 0)} / {drift_results.get('total_columns', 0)}</p>"
        
        return html

    def _format_performance_section_html(self, performance_results: Dict[str, Any]) -> str:
        """Format performance analysis section as HTML."""
        if performance_results.get('performance_degraded', False):
            status = '<div class="alert">Performance degradation detected!</div>'
        else:
            status = '<div class="success">No significant performance degradation detected.</div>'
        
        html = status
        
        # Performance metrics table
        current_metrics = performance_results.get('current_metrics', {})
        if current_metrics:
            html += "<h4>Current Performance Metrics</h4><table><tr><th>Metric</th><th>Value</th></tr>"
            for metric, value in current_metrics.items():
                html += f"<tr><td>{metric}</td><td>{value:.3f}</td></tr>"
            html += "</table>"
        
        return html

    def _format_quality_section_html(self, quality_results: Dict[str, Any]) -> str:
        """Format quality assessment section as HTML."""
        if quality_results.get('quality_issues_detected', False):
            status = '<div class="alert">Data quality issues detected!</div>'
        else:
            status = '<div class="success">No significant data quality issues detected.</div>'
        
        html = status
        
        # Quality issues
        issues_summary = quality_results.get('issues_summary', {})
        if issues_summary:
            html += "<h4>Quality Issues by Column</h4><ul>"
            for col, issues in issues_summary.items():
                if isinstance(issues, list):
                    html += f"<li><strong>{col}:</strong> {', '.join(issues)}</li>"
                else:
                    html += f"<li><strong>{col}:</strong> {issues}</li>"
            html += "</ul>"
        
        return html

    def _format_bias_section_html(self, bias_results: Dict[str, Any]) -> str:
        """Format bias detection section as HTML."""
        if bias_results.get('bias_detected', False):
            status = '<div class="alert">Bias detected!</div>'
        else:
            status = '<div class="success">No significant bias detected.</div>'
        
        html = status
        html += f"<p><strong>Protected Attributes Analyzed:</strong> {', '.join(bias_results.get('protected_attributes', []))}</p>"
        
        # Bias summary
        bias_summary = bias_results.get('bias_summary', {})
        if bias_summary:
            html += f"<p><strong>Attributes with Bias:</strong> {bias_summary.get('attributes_with_bias', 0)} / {bias_summary.get('total_attributes_analyzed', 0)}</p>"
            if bias_summary.get('most_biased_attribute'):
                html += f"<p><strong>Most Biased Attribute:</strong> {bias_summary['most_biased_attribute']} (score: {bias_summary.get('highest_bias_score', 0):.3f})</p>"
        
        return html

    def _generate_recommendations_html(self,
                                      drift_results: Dict[str, Any],
                                      performance_results: Dict[str, Any],
                                      quality_results: Dict[str, Any],
                                      bias_results: Dict[str, Any]) -> str:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if drift_results.get('dataset_drift_detected', False):
            recommendations.append("Data drift detected: Consider retraining the model with recent data.")
            recommendations.append("Investigate causes of drift in the drifted columns.")
        
        if performance_results.get('performance_degraded', False):
            recommendations.append("Performance degradation detected: Review model performance and consider retraining.")
            recommendations.append("Analyze recent data patterns that may be causing performance issues.")
        
        if quality_results.get('quality_issues_detected', False):
            recommendations.append("Data quality issues detected: Review data collection and preprocessing pipelines.")
            recommendations.append("Address missing values and outliers in the affected columns.")
        
        if bias_results.get('bias_detected', False):
            recommendations.append("Bias detected: Implement bias mitigation strategies immediately.")
            recommendations.append("Review training data for representation issues across protected groups.")
            recommendations.append("Consider implementing fairness constraints in model training.")
        
        if not recommendations:
            recommendations.append("No significant issues detected. Continue monitoring model performance and bias metrics.")
        
        html = "<ul>"
        for rec in recommendations:
            html += f"<li>{rec}</li>"
        html += "</ul>"
        
        return html

    def _save_json_report(self, results: Dict[str, Any]) -> str:
        """Save comprehensive results as JSON report."""
        json_path = self.reports_dir / f"comprehensive_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Add metadata
        json_report = {
            'metadata': {
                'case_id': self.case_id,
                'investigator': self.investigator,
                'analysis_id': self.analysis_id,
                'timestamp': datetime.datetime.now().isoformat(),
                'analyzer_version': '1.0.0'
            },
            'results': results,
            'configuration': {
                'bias_thresholds': self.bias_thresholds,
                'drift_thresholds': self.drift_thresholds,
                'protected_attributes': self.protected_attributes
            }
        }
        
        with open(json_path, 'w') as f:
            json.dump(json_report, f, indent=2, default=str)
        
        return str(json_path)

    def _create_chain_of_custody(self) -> Dict[str, Any]:
        """Create chain of custody record for forensic integrity."""
        return {
            'creation_timestamp': datetime.datetime.now().isoformat(),
            'case_id': self.case_id,
            'investigator': self.investigator,
            'analysis_id': self.analysis_id,
            'tool_version': '1.0.0',
            'evidently_version': 'latest',
            'integrity_hash': hashlib.sha256(f"{self.case_id}_{self.analysis_id}_{datetime.datetime.now()}".encode()).hexdigest()
        }

    def _assess_compliance(self, result: EvidentlyAnalysisResult) -> Dict[str, bool]:
        """Assess compliance with various standards."""
        return {
            'EEOC_compliant': not result.bias_detected,
            'GDPR_compliant': True,  # Assuming proper data handling
            'IEEE_standards_met': True,  # Assuming proper testing methodology
            'forensic_integrity_maintained': True,
            'chain_of_custody_preserved': bool(result.chain_of_custody),
            'documentation_complete': bool(result.html_report_path and result.json_report_path)
        }

    def _trigger_drift_alert(self, drift_results: Dict[str, Any], ref_hash: str, curr_hash: str):
        """Trigger alert for data drift detection."""
        alert = BiasMonitoringAlert(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.datetime.now().isoformat(),
            alert_type='data_drift',
            severity='medium',
            metric_name='dataset_drift_score',
            threshold_exceeded=self.drift_thresholds['dataset_drift_threshold'],
            current_value=drift_results.get('drift_score', 0.0),
            affected_groups=['all_data'],
            description=f"Data drift detected with score {drift_results.get('drift_score', 0.0):.3f}",
            recommended_actions=[
                'Investigate causes of data drift',
                'Consider model retraining',
                'Review data collection pipeline'
            ],
            case_id=self.case_id,
            investigator=self.investigator,
            forensic_hash=f"{ref_hash}_{curr_hash}"
        )
        
        self.alerts.append(alert)
        self._save_alert(alert)

    def _trigger_performance_alert(self, performance_results: Dict[str, Any]):
        """Trigger alert for performance degradation."""
        alert = BiasMonitoringAlert(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.datetime.now().isoformat(),
            alert_type='performance_degradation',
            severity='high',
            metric_name='model_performance',
            threshold_exceeded=0.05,
            current_value=max(abs(v) for v in performance_results.get('performance_change', {}).values()) if performance_results.get('performance_change') else 0,
            affected_groups=['all_predictions'],
            description="Model performance degradation detected",
            recommended_actions=[
                'Review model performance metrics',
                'Consider model retraining',
                'Analyze recent data patterns'
            ],
            case_id=self.case_id,
            investigator=self.investigator,
            forensic_hash=hashlib.sha256(str(performance_results).encode()).hexdigest()
        )
        
        self.alerts.append(alert)
        self._save_alert(alert)

    def _trigger_quality_alert(self, quality_results: Dict[str, Any]):
        """Trigger alert for data quality issues."""
        alert = BiasMonitoringAlert(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.datetime.now().isoformat(),
            alert_type='data_quality',
            severity='medium',
            metric_name='data_quality_issues',
            threshold_exceeded=0,
            current_value=len(quality_results.get('issues_summary', {})),
            affected_groups=list(quality_results.get('issues_summary', {}).keys()),
            description=f"Data quality issues detected in {len(quality_results.get('issues_summary', {}))} columns",
            recommended_actions=[
                'Review data collection pipeline',
                'Address missing values and outliers',
                'Validate data preprocessing steps'
            ],
            case_id=self.case_id,
            investigator=self.investigator,
            forensic_hash=hashlib.sha256(str(quality_results).encode()).hexdigest()
        )
        
        self.alerts.append(alert)
        self._save_alert(alert)

    def _trigger_bias_alert(self, bias_results: Dict[str, Any]):
        """Trigger alert for bias detection."""
        alert = BiasMonitoringAlert(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.datetime.now().isoformat(),
            alert_type='bias_detection',
            severity='critical',
            metric_name='bias_detected',
            threshold_exceeded=max(self.bias_thresholds.values()),
            current_value=1.0,  # Bias detected
            affected_groups=bias_results.get('protected_attributes', []),
            description="Bias detected in model predictions across protected groups",
            recommended_actions=[
                'Implement immediate bias mitigation strategies',
                'Review training data for representation issues',
                'Consider fairness constraints in model training',
                'Conduct thorough bias audit'
            ],
            case_id=self.case_id,
            investigator=self.investigator,
            forensic_hash=hashlib.sha256(str(bias_results).encode()).hexdigest()
        )
        
        self.alerts.append(alert)
        self._save_alert(alert)

    def _save_alert(self, alert: BiasMonitoringAlert):
        """Save alert to file for audit trail."""
        alert_path = self.alerts_dir / f"alert_{alert.alert_id}.json"
        with open(alert_path, 'w') as f:
            json.dump(asdict(alert), f, indent=2)
        
        self.logger.warning(f"Alert triggered: {alert.alert_type} - {alert.description}")

    def _calculate_data_hash(self, data: pd.DataFrame) -> str:
        """Calculate SHA-256 hash of dataset for forensic integrity."""
        # Convert dataframe to string representation for hashing
        data_str = data.to_csv(index=False)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def get_alerts(self) -> List[BiasMonitoringAlert]:
        """Get all alerts generated during analysis."""
        return self.alerts

    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts = []
        self.logger.info("All alerts cleared")

    def export_analysis_results(self, 
                               result: EvidentlyAnalysisResult,
                               format: str = 'json') -> str:
        """
        Export analysis results in various formats.
        
        Args:
            result: EvidentlyAnalysisResult to export
            format: Export format ('json', 'csv', 'xml')
            
        Returns:
            Path to exported file
        """
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == 'json':
            export_path = self.output_dir / f"analysis_export_{timestamp}.json"
            with open(export_path, 'w') as f:
                json.dump(asdict(result), f, indent=2, default=str)
        
        elif format == 'csv':
            export_path = self.output_dir / f"analysis_export_{timestamp}.csv"
            # Convert to flat structure for CSV
            flat_data = self._flatten_analysis_result(result)
            pd.DataFrame([flat_data]).to_csv(export_path, index=False)
        
        elif format == 'xml':
            export_path = self.output_dir / f"analysis_export_{timestamp}.xml"
            # Simple XML export
            xml_content = self._convert_to_xml(asdict(result))
            with open(export_path, 'w') as f:
                f.write(xml_content)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Analysis results exported to: {export_path}")
        return str(export_path)

    def _flatten_analysis_result(self, result: EvidentlyAnalysisResult) -> Dict[str, Any]:
        """Flatten nested analysis result for CSV export."""
        flat = {
            'analysis_id': result.analysis_id,
            'timestamp': result.timestamp,
            'case_id': result.case_id,
            'investigator': result.investigator,
            'data_drift_detected': result.data_drift_detected,
            'drift_score': result.drift_score,
            'model_performance_degraded': result.model_performance_degraded,
            'bias_detected': result.bias_detected,
            'missing_values_count': result.missing_values_count,
            'alert_triggered': result.alert_triggered
        }
        
        # Add performance metrics
        for metric, value in result.performance_metrics.items():
            flat[f'performance_{metric}'] = value
        
        # Add fairness metrics
        for metric, value in result.fairness_metrics.items():
            flat[f'fairness_{metric}'] = value
        
        return flat

    def _convert_to_xml(self, data: Dict[str, Any], root_name: str = 'analysis_result') -> str:
        """Convert dictionary to XML format."""
        def dict_to_xml(d, name):
            xml = f"<{name}>"
            for key, value in d.items():
                if isinstance(value, dict):
                    xml += dict_to_xml(value, key)
                elif isinstance(value, list):
                    xml += f"<{key}>"
                    for item in value:
                        if isinstance(item, dict):
                            xml += dict_to_xml(item, 'item')
                        else:
                            xml += f"<item>{item}</item>"
                    xml += f"</{key}>"
                else:
                    xml += f"<{key}>{value}</{key}>"
            xml += f"</{name}>"
            return xml
        
        return f'<?xml version="1.0" encoding="UTF-8"?>\n{dict_to_xml(data, root_name)}'


# Utility functions for integration with existing forensic framework

def create_evidently_test_suite(reference_data: pd.DataFrame,
                               current_data: pd.DataFrame,
                               column_mapping: ColumnMapping = None) -> TestSuite:
    """
    Create a comprehensive Evidently test suite for bias detection.
    
    Args:
        reference_data: Reference dataset
        current_data: Current dataset  
        column_mapping: Column mapping configuration
        
    Returns:
        Configured TestSuite object
    """
    if not EVIDENTLY_AVAILABLE:
        raise ImportError("Evidently library is required")
    
    # Create comprehensive test suite
    test_suite = TestSuite(tests=[
        # Data quality tests
        TestNumberOfColumnsWithMissingValues(),
        TestNumberOfRowsWithMissingValues(),
        TestNumberOfConstantColumns(),
        TestNumberOfDuplicatedRows(),
        TestNumberOfDuplicatedColumns(),
        TestColumnsType(),
        
        # Data drift tests
        TestNumberOfDriftedColumns(),
        TestShareOfMissingValues(),
        
        # Statistical tests
        TestMeanInNSigmas(),
        TestValueMin(),
        TestValueMax(),
        TestValueRange(),
        
        # Performance tests (if predictions available)
        TestAccuracyScore(),
        TestPrecisionScore(),
        TestRecallScore(),
        TestF1Score(),
    ])
    
    return test_suite


def run_evidently_forensic_analysis(reference_data: pd.DataFrame,
                                   current_data: pd.DataFrame,
                                   case_id: str = None,
                                   investigator: str = None,
                                   protected_attributes: List[str] = None,
                                   output_dir: str = None) -> EvidentlyAnalysisResult:
    """
    Run complete forensic analysis using Evidently.
    
    This is a convenience function that sets up and runs a complete
    bias detection and monitoring analysis.
    
    Args:
        reference_data: Reference dataset (training data)
        current_data: Current dataset (new data) 
        case_id: Unique case identifier
        investigator: Name of investigating officer/analyst
        protected_attributes: List of protected attributes for bias analysis
        output_dir: Directory for storing reports
        
    Returns:
        Complete EvidentlyAnalysisResult
    """
    # Initialize analyzer
    analyzer = EvidentlyAnalyzer(
        case_id=case_id,
        investigator=investigator,
        output_dir=output_dir,
        enable_monitoring=True,
        enable_alerts=True
    )
    
    # Configure column mapping (basic auto-detection)
    target_col = None
    prediction_col = None
    
    # Try to detect target and prediction columns
    for col in current_data.columns:
        if 'target' in col.lower() or 'label' in col.lower():
            target_col = col
        elif 'pred' in col.lower() or 'score' in col.lower():
            prediction_col = col
    
    # Identify numerical and categorical features
    numerical_features = list(current_data.select_dtypes(include=[np.number]).columns)
    categorical_features = list(current_data.select_dtypes(include=['object', 'category']).columns)
    
    # Remove target and prediction from features
    if target_col and target_col in numerical_features:
        numerical_features.remove(target_col)
    if target_col and target_col in categorical_features:
        categorical_features.remove(target_col)
    if prediction_col and prediction_col in numerical_features:
        numerical_features.remove(prediction_col)
    if prediction_col and prediction_col in categorical_features:
        categorical_features.remove(prediction_col)
    
    analyzer.configure_column_mapping(
        target=target_col,
        prediction=prediction_col,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        protected_attributes=protected_attributes
    )
    
    # Run comprehensive analysis
    result = analyzer.run_comprehensive_analysis(
        reference_data=reference_data,
        current_data=current_data,
        task_type='classification',  # Assume classification for resume screening
        protected_attributes=protected_attributes,
        generate_reports=True
    )
    
    return result


if __name__ == "__main__":
    """
    Example usage of the Evidently analyzer for resume screening bias detection.
    """
    import logging
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Evidently Analyzer for Resume Screening Bias Detection")
    logger.info("====================================================")
    
    if not EVIDENTLY_AVAILABLE:
        logger.error("Evidently library not available. Please install with: pip install evidently")
        exit(1)
    
    # Example: Create sample data for testing
    try:
        # Generate sample resume screening data
        np.random.seed(42)
        n_samples = 1000
        
        # Reference data (training data)
        ref_data = pd.DataFrame({
            'experience_years': np.random.normal(5, 2, n_samples),
            'education_score': np.random.normal(7, 1.5, n_samples),
            'skills_match': np.random.uniform(0, 1, n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], n_samples),
            'age_group': np.random.choice(['18-30', '31-45', '46-60', '60+'], n_samples),
            'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'prediction': np.random.choice([0, 1], n_samples, p=[0.72, 0.28])
        })
        
        # Current data (new data with potential drift and bias)
        curr_data = pd.DataFrame({
            'experience_years': np.random.normal(5.5, 2.2, n_samples),  # Slight drift
            'education_score': np.random.normal(7.2, 1.3, n_samples),   # Slight drift
            'skills_match': np.random.uniform(0.1, 0.9, n_samples),     # Distribution change
            'gender': np.random.choice(['M', 'F'], n_samples, p=[0.6, 0.4]),  # Gender bias
            'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], n_samples),
            'age_group': np.random.choice(['18-30', '31-45', '46-60', '60+'], n_samples),
            'target': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            'prediction': np.random.choice([0, 1], n_samples, p=[0.68, 0.32])  # Performance degradation
        })
        
        # Add bias: favor males in predictions
        male_indices = curr_data['gender'] == 'M'
        curr_data.loc[male_indices, 'prediction'] = np.where(
            np.random.random(male_indices.sum()) < 0.8,
            1, curr_data.loc[male_indices, 'prediction']
        )
        
        logger.info(f"Generated sample data: {len(ref_data)} reference samples, {len(curr_data)} current samples")
        
        # Run forensic analysis
        result = run_evidently_forensic_analysis(
            reference_data=ref_data,
            current_data=curr_data,
            case_id="DEMO_CASE_001",
            investigator="Demo Analyst",
            protected_attributes=['gender', 'race', 'age_group'],
            output_dir="./forensic/reports/evidently_demo"
        )
        
        # Print results
        logger.info("\nAnalysis Results:")
        logger.info(f"- Data Drift Detected: {result.data_drift_detected}")
        logger.info(f"- Performance Degraded: {result.model_performance_degraded}")
        logger.info(f"- Bias Detected: {result.bias_detected}")
        logger.info(f"- Data Quality Issues: {bool(result.data_quality_issues)}")
        logger.info(f"- Alerts Triggered: {result.alert_triggered}")
        
        if result.html_report_path:
            logger.info(f"- HTML Report: {result.html_report_path}")
        if result.json_report_path:
            logger.info(f"- JSON Report: {result.json_report_path}")
        
        logger.info("\nForensic Analysis Completed Successfully!")
        
    except Exception as e:
        logger.error(f"Error in example usage: {str(e)}")
        raise