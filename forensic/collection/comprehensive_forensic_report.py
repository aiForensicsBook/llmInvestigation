#!/usr/bin/env python3
"""
Comprehensive Forensic Report Generator
=======================================

Generates detailed forensic reports including synthetic data detection,
bias analysis, model specifications, and chain of custody information.

Author: Forensic Analysis Framework
Version: 1.0
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

from forensic_collector import ForensicCollector
from chain_of_custody import ChainOfCustody


class ComprehensiveForensicReport:
    """Generate comprehensive forensic analysis reports."""
    
    def __init__(self, case_id: str, investigator: str):
        """Initialize the report generator.
        
        Args:
            case_id: Unique case identifier
            investigator: Name of the investigating officer/analyst
        """
        self.case_id = case_id
        self.investigator = investigator
        self.collector = ForensicCollector(case_id, investigator)
        self.report_data = {
            'case_id': case_id,
            'investigator': investigator,
            'report_generated': datetime.now().isoformat(),
            'sections': {}
        }
    
    def analyze_training_data(self, data_path: str) -> Dict[str, Any]:
        """Analyze training data for synthetic patterns and bias.
        
        Args:
            data_path: Path to training data
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {
            'synthetic_data_detection': {},
            'bias_distributions': {},
            'data_statistics': {}
        }
        
        # Detect synthetic data
        synthetic_results = self.collector.detect_synthetic_data(data_path, 'json')
        analysis['synthetic_data_detection'] = synthetic_results
        
        # Analyze bias distributions
        bias_results = self.collector.analyze_bias_distributions(data_path)
        analysis['bias_distributions'] = bias_results
        
        # Calculate basic statistics
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                analysis['data_statistics'] = {
                    'total_records': len(data),
                    'file_size_mb': os.path.getsize(data_path) / (1024 * 1024),
                    'data_hash': self.collector.calculate_hashes(data_path)
                }
        except Exception as e:
            analysis['data_statistics']['error'] = str(e)
        
        self.report_data['sections']['training_data_analysis'] = analysis
        return analysis
    
    def analyze_model(self, model_path: str) -> Dict[str, Any]:
        """Analyze model specifications and characteristics.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Dictionary containing model analysis
        """
        model_analysis = self.collector.extract_model_specifications(model_path)
        self.report_data['sections']['model_analysis'] = model_analysis
        return model_analysis
    
    def generate_chain_of_custody_report(self, evidence_path: str) -> Dict[str, Any]:
        """Generate chain of custody report for evidence.
        
        Args:
            evidence_path: Path to evidence directory
            
        Returns:
            Dictionary containing chain of custody information
        """
        coc = ChainOfCustody(self.case_id, self.investigator)
        
        # Load chain of custody records
        coc_file = Path(evidence_path) / f"chain_of_custody_{self.case_id}.json"
        if coc_file.exists():
            coc.load_from_file(str(coc_file))
        
        coc_report = {
            'case_id': self.case_id,
            'current_custodian': coc.current_custodian,
            'evidence_count': len(coc.evidence_items),
            'custody_transfers': coc.custody_log,
            'evidence_items': []
        }
        
        # Add detailed evidence information
        for evidence in coc.evidence_items:
            item_info = {
                'evidence_id': evidence.get('evidence_id'),
                'collection_time': evidence.get('collection_time'),
                'original_path': evidence.get('original_path'),
                'hashes': evidence.get('original_hashes'),
                'collector': evidence.get('collector'),
                'collection_location': evidence.get('collection_location', {}),
                'integrity_verified': evidence.get('integrity_verified')
            }
            coc_report['evidence_items'].append(item_info)
        
        self.report_data['sections']['chain_of_custody'] = coc_report
        return coc_report
    
    def create_visualizations(self, output_dir: str) -> List[str]:
        """Create visualization plots for the report.
        
        Args:
            output_dir: Directory to save visualizations
            
        Returns:
            List of generated plot file paths
        """
        plots = []
        viz_dir = Path(output_dir) / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Bias distribution plots
        if 'bias_distributions' in self.report_data['sections'].get('training_data_analysis', {}):
            bias_data = self.report_data['sections']['training_data_analysis']['bias_distributions']
            
            if 'distributions' in bias_data:
                for feature, distribution in bias_data['distributions'].items():
                    plt.figure(figsize=(10, 6))
                    
                    # Create bar plot
                    values = list(distribution['counts'].keys())
                    counts = list(distribution['counts'].values())
                    percentages = list(distribution['percentages'].values())
                    
                    bars = plt.bar(values, counts, color='skyblue', alpha=0.8)
                    
                    # Add percentage labels
                    for bar, pct in zip(bars, percentages):
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width()/2., height,
                                f'{pct:.1f}%', ha='center', va='bottom')
                    
                    plt.xlabel(feature.capitalize())
                    plt.ylabel('Count')
                    plt.title(f'Distribution of {feature.capitalize()} in Training Data')
                    plt.grid(True, alpha=0.3)
                    
                    plot_path = viz_dir / f"{feature}_distribution.png"
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    plots.append(str(plot_path))
        
        # 2. Synthetic data detection visualization
        if 'synthetic_data_detection' in self.report_data['sections'].get('training_data_analysis', {}):
            synthetic_data = self.report_data['sections']['training_data_analysis']['synthetic_data_detection']
            
            if 'indicators' in synthetic_data and synthetic_data['indicators']:
                plt.figure(figsize=(10, 6))
                
                # Create indicator plot
                indicators = synthetic_data['indicators']
                y_pos = range(len(indicators))
                
                plt.barh(y_pos, [1] * len(indicators), color='red', alpha=0.7)
                plt.yticks(y_pos, indicators)
                plt.xlabel('Detection')
                plt.title('Synthetic Data Indicators Detected')
                plt.xlim(0, 1.5)
                
                # Add confidence score
                plt.text(0.5, len(indicators), 
                        f"Overall Confidence: {synthetic_data['confidence']:.2%}",
                        ha='center', fontsize=12, weight='bold')
                
                plot_path = viz_dir / "synthetic_data_indicators.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                plots.append(str(plot_path))
        
        return plots
    
    def generate_html_report(self, output_path: str) -> str:
        """Generate comprehensive HTML forensic report.
        
        Args:
            output_path: Path to save the HTML report
            
        Returns:
            Path to generated report
        """
        # Create visualizations
        viz_dir = os.path.dirname(output_path)
        plots = self.create_visualizations(viz_dir)
        
        # HTML template
        html_template = Template("""
<!DOCTYPE html>
<html>
<head>
    <title>Forensic Analysis Report - Case {{ case_id }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }
        .header {
            background-color: #2c3e50;
            color: white;
            padding: 30px;
            border-radius: 5px;
            margin-bottom: 30px;
        }
        .section {
            background-color: white;
            margin: 20px 0;
            padding: 25px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .subsection {
            margin: 15px 0;
            padding: 15px;
            background-color: #f8f9fa;
            border-left: 4px solid #3498db;
        }
        .alert {
            background-color: #fee;
            border-left: 4px solid #e74c3c;
            padding: 15px;
            margin: 10px 0;
        }
        .success {
            background-color: #efe;
            border-left: 4px solid #27ae60;
            padding: 15px;
            margin: 10px 0;
        }
        .metric {
            display: inline-block;
            margin: 10px 20px 10px 0;
            padding: 10px;
            background-color: #ecf0f1;
            border-radius: 3px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #34495e;
            color: white;
        }
        tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .visualization {
            margin: 20px 0;
            text-align: center;
        }
        .visualization img {
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        pre {
            background-color: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        .timestamp {
            color: #7f8c8d;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>Forensic Analysis Report</h1>
        <p><strong>Case ID:</strong> {{ case_id }}</p>
        <p><strong>Investigator:</strong> {{ investigator }}</p>
        <p><strong>Report Generated:</strong> {{ report_generated }}</p>
    </div>
    
    <!-- Executive Summary -->
    <div class="section">
        <h2>Executive Summary</h2>
        {% if sections.training_data_analysis.synthetic_data_detection.is_synthetic %}
        <div class="alert">
            <strong>⚠️ Synthetic Data Detected</strong><br>
            Confidence: {{ "%.1f"|format(sections.training_data_analysis.synthetic_data_detection.confidence * 100) }}%<br>
            Indicators: {{ sections.training_data_analysis.synthetic_data_detection.indicators|join(', ') }}
        </div>
        {% else %}
        <div class="success">
            <strong>✓ No Synthetic Data Detected</strong><br>
            The training data appears to be authentic based on statistical analysis.
        </div>
        {% endif %}
        
        {% if sections.training_data_analysis.bias_distributions.imbalances %}
        <div class="alert">
            <strong>⚠️ Data Imbalances Detected</strong><br>
            {% for imbalance in sections.training_data_analysis.bias_distributions.imbalances %}
            {{ imbalance.feature }}: {{ imbalance.max_group }} vs {{ imbalance.min_group }} 
            (ratio: {{ "%.1f"|format(imbalance.ratio) }}:1)<br>
            {% endfor %}
        </div>
        {% endif %}
    </div>
    
    <!-- Training Data Analysis -->
    <div class="section">
        <h2>Training Data Analysis</h2>
        
        <div class="subsection">
            <h3>Data Statistics</h3>
            <div class="metric">
                <strong>Total Records:</strong> {{ sections.training_data_analysis.data_statistics.total_records }}
            </div>
            <div class="metric">
                <strong>File Size:</strong> {{ "%.2f"|format(sections.training_data_analysis.data_statistics.file_size_mb) }} MB
            </div>
            <div class="metric">
                <strong>SHA-256 Hash:</strong> {{ sections.training_data_analysis.data_statistics.data_hash.sha256[:16] }}...
            </div>
        </div>
        
        <div class="subsection">
            <h3>Synthetic Data Detection</h3>
            <p><strong>Overall Assessment:</strong> 
            {% if sections.training_data_analysis.synthetic_data_detection.is_synthetic %}
                <span style="color: red;">Synthetic data likely present</span>
            {% else %}
                <span style="color: green;">No synthetic patterns detected</span>
            {% endif %}
            </p>
            
            {% if sections.training_data_analysis.synthetic_data_detection.indicators %}
            <p><strong>Detected Indicators:</strong></p>
            <ul>
            {% for indicator in sections.training_data_analysis.synthetic_data_detection.indicators %}
                <li>{{ indicator }}</li>
            {% endfor %}
            </ul>
            {% endif %}
            
            {% if sections.training_data_analysis.synthetic_data_detection.statistical_anomalies %}
            <p><strong>Statistical Anomalies:</strong></p>
            <ul>
            {% for anomaly in sections.training_data_analysis.synthetic_data_detection.statistical_anomalies %}
                <li>{{ anomaly }}</li>
            {% endfor %}
            </ul>
            {% endif %}
        </div>
        
        <div class="subsection">
            <h3>Bias Distribution Analysis</h3>
            {% for feature, distribution in sections.training_data_analysis.bias_distributions.distributions.items() %}
            <h4>{{ feature|capitalize }} Distribution</h4>
            <table>
                <tr>
                    <th>Value</th>
                    <th>Count</th>
                    <th>Percentage</th>
                </tr>
                {% for value, count in distribution.counts.items() %}
                <tr>
                    <td>{{ value }}</td>
                    <td>{{ count }}</td>
                    <td>{{ "%.1f"|format(distribution.percentages[value]) }}%</td>
                </tr>
                {% endfor %}
            </table>
            <div class="visualization">
                <img src="visualizations/{{ feature }}_distribution.png" alt="{{ feature }} distribution">
            </div>
            {% endfor %}
        </div>
    </div>
    
    <!-- Model Specifications -->
    <div class="section">
        <h2>Model Technical Specifications</h2>
        
        <div class="subsection">
            <h3>File Information</h3>
            <table>
                <tr>
                    <th>Property</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td>File Path</td>
                    <td>{{ sections.model_analysis.file_info.path }}</td>
                </tr>
                <tr>
                    <td>File Size</td>
                    <td>{{ "%.2f"|format(sections.model_analysis.file_info.size_mb) }} MB</td>
                </tr>
                <tr>
                    <td>Created</td>
                    <td>{{ sections.model_analysis.file_info.created }}</td>
                </tr>
                <tr>
                    <td>Last Modified</td>
                    <td>{{ sections.model_analysis.file_info.modified }}</td>
                </tr>
                <tr>
                    <td>SHA-256 Hash</td>
                    <td style="font-family: monospace;">{{ sections.model_analysis.file_info.hash.sha256 }}</td>
                </tr>
            </table>
        </div>
        
        <div class="subsection">
            <h3>Model Architecture</h3>
            <div class="metric">
                <strong>Model Type:</strong> {{ sections.model_analysis.model_architecture.type|default('TF-IDF + Cosine Similarity') }}
            </div>
            <div class="metric">
                <strong>Total Parameters:</strong> {{ sections.model_analysis.parameters.total_parameters|default('N/A') }}
            </div>
            <div class="metric">
                <strong>Vocabulary Size:</strong> {{ sections.model_analysis.parameters.vocabulary_size|default('N/A') }}
            </div>
            <div class="metric">
                <strong>Number of Features:</strong> {{ sections.model_analysis.parameters.num_features|default('N/A') }}
            </div>
        </div>
        
        {% if sections.model_analysis.training_details %}
        <div class="subsection">
            <h3>Training Details</h3>
            <pre>{{ sections.model_analysis.training_details|tojson(indent=2) }}</pre>
        </div>
        {% endif %}
    </div>
    
    <!-- Chain of Custody -->
    <div class="section">
        <h2>Chain of Custody</h2>
        
        <div class="subsection">
            <h3>Custody Information</h3>
            <div class="metric">
                <strong>Case ID:</strong> {{ sections.chain_of_custody.case_id }}
            </div>
            <div class="metric">
                <strong>Current Custodian:</strong> {{ sections.chain_of_custody.current_custodian }}
            </div>
            <div class="metric">
                <strong>Evidence Items:</strong> {{ sections.chain_of_custody.evidence_count }}
            </div>
        </div>
        
        {% if sections.chain_of_custody.evidence_items %}
        <div class="subsection">
            <h3>Evidence Items</h3>
            <table>
                <tr>
                    <th>Evidence ID</th>
                    <th>Collection Time</th>
                    <th>Original Path</th>
                    <th>Collector</th>
                    <th>Location</th>
                    <th>Integrity</th>
                </tr>
                {% for item in sections.chain_of_custody.evidence_items %}
                <tr>
                    <td style="font-family: monospace;">{{ item.evidence_id[:8] }}...</td>
                    <td>{{ item.collection_time }}</td>
                    <td>{{ item.original_path }}</td>
                    <td>{{ item.collector }}</td>
                    <td>{{ item.collection_location.hostname|default('Unknown') }}</td>
                    <td>
                        {% if item.integrity_verified %}
                            <span style="color: green;">✓ Verified</span>
                        {% else %}
                            <span style="color: red;">✗ Failed</span>
                        {% endif %}
                    </td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}
        
        {% if sections.chain_of_custody.custody_transfers %}
        <div class="subsection">
            <h3>Custody Transfer Log</h3>
            <table>
                <tr>
                    <th>Timestamp</th>
                    <th>From</th>
                    <th>To</th>
                    <th>Action</th>
                    <th>Evidence ID</th>
                </tr>
                {% for transfer in sections.chain_of_custody.custody_transfers %}
                <tr>
                    <td>{{ transfer.timestamp }}</td>
                    <td>{{ transfer.from_custodian }}</td>
                    <td>{{ transfer.to_custodian }}</td>
                    <td>{{ transfer.action }}</td>
                    <td style="font-family: monospace;">{{ transfer.evidence_id[:8] }}...</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}
    </div>
    
    <!-- System Information -->
    <div class="section">
        <h2>Collection System Information</h2>
        <table>
            <tr>
                <th>Property</th>
                <th>Value</th>
            </tr>
            {% for key, value in sections.model_analysis.file_info.items() %}
            {% if key not in ['path', 'size_mb', 'created', 'modified', 'hash'] %}
            <tr>
                <td>{{ key|replace('_', ' ')|title }}</td>
                <td>{{ value }}</td>
            </tr>
            {% endif %}
            {% endfor %}
        </table>
    </div>
    
    <div class="section">
        <p class="timestamp">Report generated on {{ report_generated }} by {{ investigator }}</p>
        <p class="timestamp">This is an official forensic analysis report for Case ID: {{ case_id }}</p>
    </div>
</body>
</html>
        """)
        
        # Render HTML
        html_content = html_template.render(**self.report_data)
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Forensic report generated: {output_path}")
        return output_path
    
    def generate_json_report(self, output_path: str) -> str:
        """Generate JSON forensic report.
        
        Args:
            output_path: Path to save the JSON report
            
        Returns:
            Path to generated report
        """
        with open(output_path, 'w') as f:
            json.dump(self.report_data, f, indent=2, default=str)
        
        print(f"JSON report generated: {output_path}")
        return output_path


def main():
    """Example usage of the comprehensive forensic report generator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Comprehensive Forensic Report')
    parser.add_argument('--case-id', required=True, help='Case identifier')
    parser.add_argument('--investigator', required=True, help='Investigator name')
    parser.add_argument('--training-data', required=True, help='Path to training data')
    parser.add_argument('--model', required=True, help='Path to saved model')
    parser.add_argument('--evidence-dir', required=True, help='Path to evidence directory')
    parser.add_argument('--output', required=True, help='Output report path')
    parser.add_argument('--format', choices=['html', 'json'], default='html', help='Report format')
    
    args = parser.parse_args()
    
    # Generate report
    report_gen = ComprehensiveForensicReport(args.case_id, args.investigator)
    
    # Analyze training data
    print("Analyzing training data...")
    report_gen.analyze_training_data(args.training_data)
    
    # Analyze model
    print("Analyzing model specifications...")
    report_gen.analyze_model(args.model)
    
    # Generate chain of custody report
    print("Generating chain of custody report...")
    report_gen.generate_chain_of_custody_report(args.evidence_dir)
    
    # Generate final report
    if args.format == 'html':
        report_gen.generate_html_report(args.output)
    else:
        report_gen.generate_json_report(args.output)
    
    print(f"Report generation complete!")


if __name__ == "__main__":
    main()