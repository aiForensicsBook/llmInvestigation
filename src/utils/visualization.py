#!/usr/bin/env python3
"""
Visualization utilities for Resume Screening LLM forensic analysis.
Provides plotting functions for bias detection, feature importance, and model analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
from collections import defaultdict
import json
import os
from datetime import datetime
import hashlib

# Set default style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_score_distributions(scores_data: Dict[str, List[float]], 
                           title: str = "Score Distribution Analysis",
                           save_path: Optional[str] = None) -> None:
    """
    Plot score distributions across different groups.
    
    Args:
        scores_data: Dictionary mapping group names to lists of scores
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Create subplots for better visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Overlapping distributions
    for group, scores in scores_data.items():
        axes[0].hist(scores, alpha=0.7, bins=30, label=group, density=True)
    
    axes[0].set_xlabel('Score')
    axes[0].set_ylabel('Density')
    axes[0].set_title(f'{title} - Overlapping Distributions')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Box plots for comparison
    groups = list(scores_data.keys())
    scores_lists = [scores_data[group] for group in groups]
    
    axes[1].boxplot(scores_lists, labels=groups)
    axes[1].set_ylabel('Score')
    axes[1].set_title(f'{title} - Box Plot Comparison')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_feature_importance(idf_values: Dict[str, float],
                          title: str = "Feature Importance Analysis",
                          top_n: int = 20,
                          save_path: Optional[str] = None) -> None:
    """
    Plot feature importance based on IDF values.
    
    Args:
        idf_values: Dictionary mapping words to their IDF values
        title: Title for the plot
        top_n: Number of top features to display
        save_path: Optional path to save the plot
    """
    # Sort features by IDF value
    sorted_features = sorted(idf_values.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    words = [item[0] for item in top_features]
    values = [item[1] for item in top_features]
    
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar plot
    y_pos = np.arange(len(words))
    bars = plt.barh(y_pos, values, color='skyblue', alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', va='center', ha='left')
    
    plt.yticks(y_pos, words)
    plt.xlabel('IDF Value')
    plt.title(title)
    plt.grid(True, alpha=0.3, axis='x')
    
    # Invert y-axis to show highest values at top
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_bias_analysis(demographic_scores: Dict[str, Dict[str, float]],
                      title: str = "Bias Analysis - Scoring Patterns",
                      save_path: Optional[str] = None) -> None:
    """
    Plot bias analysis showing scoring differences across demographic groups.
    
    Args:
        demographic_scores: Dictionary mapping groups to their score statistics
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    groups = list(demographic_scores.keys())
    means = [demographic_scores[group]['mean'] for group in groups]
    stds = [demographic_scores[group]['std'] for group in groups]
    counts = [demographic_scores[group]['count'] for group in groups]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Mean scores by group
    bars1 = axes[0, 0].bar(groups, means, color='lightcoral', alpha=0.8)
    axes[0, 0].set_ylabel('Mean Score')
    axes[0, 0].set_title('Mean Scores by Group')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, mean in zip(bars1, means):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{mean:.3f}', ha='center', va='bottom')
    
    # Plot 2: Standard deviation by group
    bars2 = axes[0, 1].bar(groups, stds, color='lightgreen', alpha=0.8)
    axes[0, 1].set_ylabel('Standard Deviation')
    axes[0, 1].set_title('Score Variability by Group')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, std in zip(bars2, stds):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                       f'{std:.3f}', ha='center', va='bottom')
    
    # Plot 3: Sample sizes
    bars3 = axes[1, 0].bar(groups, counts, color='lightskyblue', alpha=0.8)
    axes[1, 0].set_ylabel('Sample Count')
    axes[1, 0].set_title('Sample Sizes by Group')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars3, counts):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{count}', ha='center', va='bottom')
    
    # Plot 4: Statistical significance indicator
    if len(groups) >= 2:
        # Calculate effect sizes (Cohen's d approximation)
        effect_sizes = []
        comparisons = []
        
        for i in range(len(groups)):
            for j in range(i+1, len(groups)):
                group1, group2 = groups[i], groups[j]
                mean1, mean2 = means[i], means[j]
                std1, std2 = stds[i], stds[j]
                
                # Pooled standard deviation
                pooled_std = np.sqrt((std1**2 + std2**2) / 2)
                effect_size = abs(mean1 - mean2) / pooled_std if pooled_std > 0 else 0
                
                effect_sizes.append(effect_size)
                comparisons.append(f"{group1} vs {group2}")
        
        bars4 = axes[1, 1].bar(range(len(comparisons)), effect_sizes, 
                              color='gold', alpha=0.8)
        axes[1, 1].set_ylabel('Effect Size (Cohen\'s d)')
        axes[1, 1].set_title('Effect Sizes Between Groups')
        axes[1, 1].set_xticks(range(len(comparisons)))
        axes[1, 1].set_xticklabels(comparisons, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add significance threshold line
        axes[1, 1].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
                          label='Medium Effect (0.5)')
        axes[1, 1].legend()
        
        # Add value labels
        for bar, effect in zip(bars4, effect_sizes):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{effect:.3f}', ha='center', va='bottom')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_training_history(training_history: List[Dict[str, Any]],
                         title: str = "Model Training Progress",
                         save_path: Optional[str] = None) -> None:
    """
    Plot model training history and performance metrics.
    
    Args:
        training_history: List of training history records
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    if not training_history:
        print("No training history available to plot.")
        return
    
    # Extract metrics from training history
    epochs = list(range(1, len(training_history) + 1))
    vocab_sizes = [record.get('vocab_size', 0) for record in training_history]
    processing_times = [record.get('processing_time', 0) for record in training_history]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Vocabulary size over training
    axes[0, 0].plot(epochs, vocab_sizes, marker='o', color='blue', alpha=0.7)
    axes[0, 0].set_xlabel('Training Iteration')
    axes[0, 0].set_ylabel('Vocabulary Size')
    axes[0, 0].set_title('Vocabulary Size Over Training')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Processing time
    axes[0, 1].plot(epochs, processing_times, marker='s', color='green', alpha=0.7)
    axes[0, 1].set_xlabel('Training Iteration')
    axes[0, 1].set_ylabel('Processing Time (seconds)')
    axes[0, 1].set_title('Processing Time Per Training Iteration')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Training parameters summary
    latest_record = training_history[-1]
    params = latest_record.get('parameters', {})
    
    if params:
        param_names = list(params.keys())[:5]  # Show top 5 parameters
        param_values = [params[name] for name in param_names]
        
        axes[1, 0].bar(param_names, param_values, color='orange', alpha=0.7)
        axes[1, 0].set_ylabel('Parameter Value')
        axes[1, 0].set_title('Training Parameters')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Model metadata summary
    axes[1, 1].text(0.1, 0.9, f"Total Training Iterations: {len(training_history)}", 
                   transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.8, f"Final Vocabulary Size: {vocab_sizes[-1]}", 
                   transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.7, f"Total Processing Time: {sum(processing_times):.2f}s", 
                   transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].text(0.1, 0.6, f"Average Time per Iteration: {np.mean(processing_times):.2f}s", 
                   transform=axes[1, 1].transAxes, fontsize=12)
    axes[1, 1].set_title('Training Summary')
    axes[1, 1].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def export_analysis_report(model, analysis_data: Dict[str, Any],
                          output_dir: str = "forensic_reports",
                          format: str = "html") -> str:
    """
    Export comprehensive analysis report with all visualizations.
    
    Args:
        model: The trained model instance
        analysis_data: Dictionary containing analysis results
        output_dir: Directory to save the report
        format: Export format ('html', 'pdf', 'json')
    
    Returns:
        Path to the generated report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp for unique filename
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if format == "html":
        report_path = os.path.join(output_dir, f"forensic_analysis_report_{timestamp}.html")
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Forensic Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                .alert {{ background-color: #ffebee; padding: 10px; border-left: 4px solid #f44336; }}
                .success {{ background-color: #e8f5e8; padding: 10px; border-left: 4px solid #4caf50; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Resume Screening LLM - Forensic Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>This report provides a comprehensive forensic analysis of the Resume Screening LLM system, 
                   including bias detection, transparency assessment, and performance evaluation.</p>
            </div>
            
            <div class="section">
                <h2>Model Information</h2>
                <div class="metric">Algorithm: TF-IDF + Cosine Similarity</div>
                <div class="metric">Vocabulary Size: {len(model.vocabulary) if hasattr(model, 'vocabulary') else 'N/A'}</div>
                <div class="metric">Training Data: {analysis_data.get('training_data_size', 'N/A')} samples</div>
            </div>
            
            <div class="section">
                <h2>Bias Analysis Results</h2>
                {_generate_bias_analysis_html(analysis_data.get('bias_analysis', {}))}
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                {_generate_recommendations_html(analysis_data.get('recommendations', []))}
            </div>
            
            <div class="section">
                <h2>Technical Details</h2>
                <pre>{json.dumps(analysis_data.get('technical_details', {}), indent=2)}</pre>
            </div>
        </body>
        </html>
        """
        
        with open(report_path, 'w') as f:
            f.write(html_content)
    
    elif format == "json":
        report_path = os.path.join(output_dir, f"forensic_analysis_report_{timestamp}.json")
        
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "algorithm": "TF-IDF + Cosine Similarity",
                "vocabulary_size": len(model.vocabulary) if hasattr(model, 'vocabulary') else None
            },
            "analysis_results": analysis_data
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
    
    print(f"Analysis report exported to: {report_path}")
    return report_path


def _generate_bias_analysis_html(bias_data: Dict[str, Any]) -> str:
    """Generate HTML for bias analysis section."""
    if not bias_data:
        return "<p>No bias analysis data available.</p>"
    
    html = ""
    if 'vocabulary_bias' in bias_data:
        vocab_bias = bias_data['vocabulary_bias']
        if vocab_bias:
            html += '<div class="alert"><h3>Vocabulary Bias Detected</h3><ul>'
            for category, words in vocab_bias.items():
                html += f'<li><strong>{category.title()}:</strong> {", ".join(words)}</li>'
            html += '</ul></div>'
        else:
            html += '<div class="success"><h3>No Vocabulary Bias Detected</h3></div>'
    
    if 'scoring_patterns' in bias_data:
        html += '<h3>Scoring Pattern Analysis</h3>'
        scoring_data = bias_data['scoring_patterns']
        if 'demographic' in scoring_data:
            html += '<table border="1"><tr><th>Group</th><th>Mean Score</th><th>Std Dev</th><th>Count</th></tr>'
            for group, stats in scoring_data['demographic'].items():
                html += f'<tr><td>{group}</td><td>{stats["mean"]:.3f}</td><td>{stats["std"]:.3f}</td><td>{stats["count"]}</td></tr>'
            html += '</table>'
    
    return html


def _generate_recommendations_html(recommendations: List[str]) -> str:
    """Generate HTML for recommendations section."""
    if not recommendations:
        return "<p>No specific recommendations generated.</p>"
    
    html = "<ul>"
    for rec in recommendations:
        css_class = "alert" if "ALERT" in rec else "success"
        html += f'<li class="{css_class}">{rec}</li>'
    html += "</ul>"
    
    return html


def create_comprehensive_visualization_dashboard(model, resumes: List[Dict], 
                                               job_postings: List[Dict],
                                               output_dir: str = "forensic_reports",
                                               synthetic_results: Optional[Dict] = None,
                                               model_specs: Optional[Dict] = None,
                                               custody_data: Optional[Dict] = None) -> None:
    """
    Create a comprehensive visualization dashboard with all analysis plots.
    
    Args:
        model: The trained model instance
        resumes: List of resume data
        job_postings: List of job posting data
        output_dir: Directory to save visualizations
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating comprehensive visualization dashboard...")
    
    # 1. Score distribution analysis
    print("1. Generating score distribution plots...")
    all_scores = []
    demographic_scores = defaultdict(list)
    
    for resume in resumes[:50]:  # Limit for performance
        for job in job_postings[:10]:
            result = model.score_resume(resume, job)
            score = result['overall_score']
            all_scores.append(score)
            
            # Mock demographic analysis based on name patterns
            name = resume.get('personal_info', {}).get('name', '').lower()
            if any(n in name for n in ['john', 'mike', 'david', 'robert', 'james']):
                demographic_scores['male_pattern'].append(score)
            elif any(n in name for n in ['jane', 'mary', 'sarah', 'lisa', 'jennifer']):
                demographic_scores['female_pattern'].append(score)
    
    # Add overall scores
    scores_data = {'Overall': all_scores}
    scores_data.update(demographic_scores)
    
    plot_score_distributions(
        scores_data, 
        title="Resume Scoring Distribution Analysis",
        save_path=os.path.join(output_dir, "score_distributions.png")
    )
    
    # 2. Feature importance analysis
    print("2. Generating feature importance plots...")
    if hasattr(model, 'idf_values') and model.idf_values:
        plot_feature_importance(
            model.idf_values,
            title="Top Discriminative Features (IDF Values)",
            save_path=os.path.join(output_dir, "feature_importance.png")
        )
    
    # 3. Bias analysis
    print("3. Generating bias analysis plots...")
    if demographic_scores:
        demo_stats = {}
        for group, scores in demographic_scores.items():
            demo_stats[group] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'count': len(scores)
            }
        
        plot_bias_analysis(
            demo_stats,
            title="Bias Analysis - Demographic Scoring Patterns",
            save_path=os.path.join(output_dir, "bias_analysis.png")
        )
    
    # 4. Training history
    print("4. Generating training history plots...")
    if hasattr(model, 'training_history') and model.training_history:
        plot_training_history(
            model.training_history,
            title="Model Training Progress",
            save_path=os.path.join(output_dir, "training_history.png")
        )
    
    # 5. Synthetic data detection visualization
    if synthetic_results:
        print("5. Generating synthetic data detection plots...")
        plot_synthetic_data_detection(
            synthetic_results,
            title="Synthetic Data Detection Analysis",
            save_path=os.path.join(output_dir, "synthetic_data_detection.png")
        )
    
    # 6. Model specifications visualization
    if model_specs:
        print("6. Generating model specifications plots...")
        plot_model_specifications(
            model_specs,
            title="Model Technical Specifications",
            save_path=os.path.join(output_dir, "model_specifications.png")
        )
    
    # 7. Chain of custody visualization
    if custody_data:
        print("7. Generating chain of custody plots...")
        plot_chain_of_custody(
            custody_data,
            title="Evidence Chain of Custody",
            save_path=os.path.join(output_dir, "chain_of_custody.png")
        )
    
    print(f"Visualization dashboard created in: {output_dir}")
    print("Generated files:")
    print("- score_distributions.png")
    print("- feature_importance.png")
    print("- bias_analysis.png")
    print("- training_history.png")
    if synthetic_results:
        print("- synthetic_data_detection.png")
    if model_specs:
        print("- model_specifications.png")
    if custody_data:
        print("- chain_of_custody.png")


def plot_synthetic_data_detection(synthetic_results: Dict[str, Any],
                                title: str = "Synthetic Data Detection Results",
                                save_path: Optional[str] = None) -> None:
    """
    Plot synthetic data detection results.
    
    Args:
        synthetic_results: Dictionary containing synthetic detection results
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Overall confidence gauge
    ax1 = axes[0, 0]
    confidence = synthetic_results.get('confidence', 0)
    is_synthetic = synthetic_results.get('is_synthetic', False)
    
    # Create a gauge-like visualization
    wedges, texts = ax1.pie([confidence, 1-confidence], 
                           colors=['red' if is_synthetic else 'green', 'lightgray'],
                           startangle=90, counterclock=False)
    
    # Add center text
    ax1.text(0, 0, f'{confidence:.1%}\nConfidence', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    ax1.set_title('Synthetic Data Detection Confidence')
    
    # Plot 2: Indicators detected
    ax2 = axes[0, 1]
    indicators = synthetic_results.get('indicators', [])
    
    if indicators:
        y_pos = np.arange(len(indicators))
        ax2.barh(y_pos, [1] * len(indicators), color='orange', alpha=0.7)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(indicators)
        ax2.set_xlabel('Detected')
        ax2.set_title('Synthetic Data Indicators')
        ax2.set_xlim(0, 1.2)
        
        for i, indicator in enumerate(indicators):
            ax2.text(1.05, i, '✓', fontsize=16, color='red', va='center')
    else:
        ax2.text(0.5, 0.5, 'No indicators detected', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('Synthetic Data Indicators')
        ax2.axis('off')
    
    # Plot 3: Statistical anomalies
    ax3 = axes[1, 0]
    anomalies = synthetic_results.get('statistical_anomalies', [])
    
    if anomalies:
        ax3.text(0.1, 0.9, 'Statistical Anomalies Detected:', 
                transform=ax3.transAxes, fontsize=12, weight='bold')
        
        for i, anomaly in enumerate(anomalies[:5]):  # Show max 5
            ax3.text(0.1, 0.8 - i*0.15, f'• {anomaly}', 
                    transform=ax3.transAxes, fontsize=10, wrap=True)
    else:
        ax3.text(0.5, 0.5, 'No statistical anomalies detected', 
                ha='center', va='center', transform=ax3.transAxes)
    
    ax3.set_title('Statistical Analysis')
    ax3.axis('off')
    
    # Plot 4: Pattern analysis summary
    ax4 = axes[1, 1]
    pattern_analysis = synthetic_results.get('pattern_analysis', {})
    
    summary_text = f"""Overall Assessment:
    
Synthetic Data: {'YES' if is_synthetic else 'NO'}
Confidence Level: {confidence:.1%}
Indicators Found: {len(indicators)}
Anomalies Found: {len(anomalies)}

Recommendation:
{'⚠️ Data appears to be synthetic' if is_synthetic else '✓ Data appears to be authentic'}
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes, 
            fontsize=11, verticalalignment='center')
    ax4.set_title('Summary')
    ax4.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_model_specifications(model_specs: Dict[str, Any],
                            title: str = "Model Technical Specifications",
                            save_path: Optional[str] = None) -> None:
    """
    Plot model technical specifications.
    
    Args:
        model_specs: Dictionary containing model specifications
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Model parameters
    ax1 = axes[0, 0]
    params = model_specs.get('parameters', {})
    
    if params:
        param_names = list(params.keys())
        param_values = list(params.values())
        
        bars = ax1.bar(range(len(param_names)), param_values, color='lightblue', alpha=0.8)
        ax1.set_xticks(range(len(param_names)))
        ax1.set_xticklabels(param_names, rotation=45, ha='right')
        ax1.set_ylabel('Value')
        ax1.set_title('Model Parameters')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, param_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:,}', ha='center', va='bottom')
    
    # Plot 2: File information
    ax2 = axes[0, 1]
    file_info = model_specs.get('file_info', {})
    
    info_text = f"""File Information:
    
Size: {file_info.get('size_mb', 0):.2f} MB
Created: {file_info.get('created', 'Unknown')}
Modified: {file_info.get('modified', 'Unknown')}

Integrity:
SHA-256: {file_info.get('hash', {}).get('sha256', 'N/A')[:16]}...
    """
    
    ax2.text(0.1, 0.5, info_text, transform=ax2.transAxes, 
            fontsize=10, verticalalignment='center')
    ax2.set_title('File Metadata')
    ax2.axis('off')
    
    # Plot 3: Architecture details
    ax3 = axes[1, 0]
    arch = model_specs.get('model_architecture', {})
    
    arch_text = f"""Model Architecture:
    
Type: {arch.get('type', 'Unknown')}
Framework: TF-IDF + Cosine Similarity
Feature Extraction: Bag of Words
Similarity Metric: Cosine
    """
    
    ax3.text(0.1, 0.5, arch_text, transform=ax3.transAxes, 
            fontsize=11, verticalalignment='center')
    ax3.set_title('Architecture Details')
    ax3.axis('off')
    
    # Plot 4: Training configuration
    ax4 = axes[1, 1]
    training_details = model_specs.get('training_details', {})
    
    if training_details:
        config_text = "Training Configuration:\n\n"
        for key, value in list(training_details.items())[:5]:
            config_text += f"{key}: {value}\n"
    else:
        config_text = "No training configuration available"
    
    ax4.text(0.1, 0.5, config_text, transform=ax4.transAxes, 
            fontsize=10, verticalalignment='center')
    ax4.set_title('Training Configuration')
    ax4.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_chain_of_custody(custody_data: Dict[str, Any],
                         title: str = "Chain of Custody Timeline",
                         save_path: Optional[str] = None) -> None:
    """
    Plot chain of custody timeline and information.
    
    Args:
        custody_data: Dictionary containing chain of custody information
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Timeline visualization
    ax1 = axes[0]
    transfers = custody_data.get('custody_transfers', [])
    
    if transfers:
        # Extract timestamps and convert to datetime objects
        timestamps = []
        actions = []
        custodians = []
        
        for transfer in transfers:
            try:
                ts = datetime.fromisoformat(transfer['timestamp'].replace('Z', '+00:00'))
                timestamps.append(ts)
                actions.append(transfer.get('action', 'Unknown'))
                custodians.append(transfer.get('to_custodian', 'Unknown'))
            except:
                pass
        
        if timestamps:
            # Create timeline
            y_positions = range(len(timestamps))
            
            # Plot timeline
            for i, (ts, action, custodian) in enumerate(zip(timestamps, actions, custodians)):
                ax1.plot([0, 1], [i, i], 'b-', alpha=0.3)
                ax1.plot(0, i, 'bo', markersize=10)
                
                # Add timestamp
                ax1.text(-0.1, i, ts.strftime('%Y-%m-%d %H:%M'), 
                        ha='right', va='center', fontsize=9)
                
                # Add action and custodian
                ax1.text(0.05, i, f"{action} → {custodian}", 
                        ha='left', va='center', fontsize=10)
            
            ax1.set_ylim(-0.5, len(timestamps)-0.5)
            ax1.set_xlim(-0.5, 1.5)
            ax1.set_yticks([])
            ax1.set_xticks([])
            ax1.set_title('Custody Transfer Timeline')
            ax1.spines['top'].set_visible(False)
            ax1.spines['right'].set_visible(False)
            ax1.spines['bottom'].set_visible(False)
    else:
        ax1.text(0.5, 0.5, 'No custody transfers recorded', 
                ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('Custody Transfer Timeline')
        ax1.axis('off')
    
    # Plot 2: Evidence summary
    ax2 = axes[1]
    evidence_items = custody_data.get('evidence_items', [])
    
    summary_text = f"""Chain of Custody Summary:
    
Case ID: {custody_data.get('case_id', 'Unknown')}
Current Custodian: {custody_data.get('current_custodian', 'Unknown')}
Total Evidence Items: {custody_data.get('evidence_count', 0)}
Total Custody Transfers: {len(transfers)}

Evidence Integrity:
"""
    
    # Add integrity status
    if evidence_items:
        verified_count = sum(1 for item in evidence_items if item.get('integrity_verified', False))
        summary_text += f"✓ Verified: {verified_count}\n"
        summary_text += f"✗ Failed: {len(evidence_items) - verified_count}\n"
    
    ax2.text(0.1, 0.5, summary_text, transform=ax2.transAxes, 
            fontsize=12, verticalalignment='center')
    ax2.set_title('Custody Summary')
    ax2.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_hidden_bias_mechanisms(bias_data: Dict[str, Any],
                              title: str = "Hidden Bias Mechanisms Analysis",
                              save_path: Optional[str] = None) -> None:
    """
    Plot analysis of hidden bias mechanisms including multipliers and adjustments.
    
    Args:
        bias_data: Dictionary containing hidden bias analysis data
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Gender multipliers comparison
    ax1 = axes[0, 0]
    gender_multipliers = bias_data.get('gender_multipliers', {})
    
    if gender_multipliers:
        genders = list(gender_multipliers.keys())
        avg_multipliers = [gender_multipliers[g]['average_multiplier'] for g in genders]
        std_multipliers = [gender_multipliers[g]['multiplier_std'] for g in genders]
        
        x_pos = np.arange(len(genders))
        bars = ax1.bar(x_pos, avg_multipliers, yerr=std_multipliers, 
                       color=['lightblue' if g.lower() == 'male' else 'lightpink' for g in genders],
                       capsize=5, alpha=0.8)
        
        # Add value labels
        for bar, mult in zip(bars, avg_multipliers):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{mult:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Add reference line at 1.0 (no bias)
        ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No Bias (1.0)')
        
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(genders)
        ax1.set_ylabel('Average Bias Multiplier')
        ax1.set_title('Hidden Gender Bias Multipliers')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add percentage labels
        for i, (bar, mult) in enumerate(zip(bars, avg_multipliers)):
            pct_diff = (mult - 1.0) * 100
            ax1.text(bar.get_x() + bar.get_width()/2, 0.5,
                    f'{pct_diff:+.1f}%', ha='center', va='center',
                    fontsize=12, fontweight='bold',
                    color='darkgreen' if pct_diff > 0 else 'darkred')
    
    # Plot 2: Score adjustment visualization
    ax2 = axes[0, 1]
    score_adjustments = bias_data.get('score_adjustments', {})
    
    if score_adjustments:
        # Create before/after comparison
        genders = list(score_adjustments.keys())
        base_score = 0.5  # Assume normalized base score
        
        x = np.arange(len(genders))
        width = 0.35
        
        # Base scores (all same)
        bars1 = ax2.bar(x - width/2, [base_score] * len(genders), width, 
                        label='Base Score', color='gray', alpha=0.6)
        
        # Adjusted scores
        adjusted_scores = [base_score * score_adjustments[g]['avg_bias_multiplier'] 
                          for g in genders]
        bars2 = ax2.bar(x + width/2, adjusted_scores, width, 
                        label='After Bias Applied',
                        color=['lightblue' if g.lower() == 'male' else 'lightpink' for g in genders],
                        alpha=0.8)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        ax2.set_xlabel('Gender')
        ax2.set_ylabel('Score')
        ax2.set_title('Score Before and After Hidden Bias Application')
        ax2.set_xticks(x)
        ax2.set_xticklabels(genders)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Bias patterns frequency
    ax3 = axes[1, 0]
    bias_patterns = bias_data.get('bias_patterns', [])
    
    if bias_patterns:
        # Extract top patterns
        patterns = [p['pattern'] for p in bias_patterns[:10]]
        frequencies = [p['frequency'] for p in bias_patterns[:10]]
        
        # Create horizontal bar chart
        y_pos = np.arange(len(patterns))
        bars = ax3.barh(y_pos, frequencies, color='coral', alpha=0.8)
        
        # Add value labels
        for bar, freq in zip(bars, frequencies):
            ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{freq}', ha='left', va='center')
        
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([p.split(':')[0] for p in patterns])  # Shorten labels
        ax3.set_xlabel('Frequency')
        ax3.set_title('Most Common Bias Patterns Detected')
        ax3.grid(True, alpha=0.3, axis='x')
        ax3.invert_yaxis()  # Show highest frequency at top
    
    # Plot 4: Gender score ratio visualization
    ax4 = axes[1, 1]
    gender_ratio = bias_data.get('gender_score_ratio', {})
    
    if gender_ratio:
        ratio = gender_ratio.get('male_to_female_ratio', 1.0)
        advantage_pct = gender_ratio.get('male_advantage_percentage', 0)
        
        # Create a visual representation of the ratio
        if ratio != float('inf'):
            # Create pie chart showing relative scoring
            sizes = [ratio, 1.0]  # Male vs Female relative scores
            labels = ['Male', 'Female']
            colors = ['lightblue', 'lightpink']
            explode = (0.1, 0)  # Explode the larger slice
            
            wedges, texts, autotexts = ax4.pie(sizes, labels=labels, colors=colors,
                                               autopct='%1.1f%%', explode=explode,
                                               startangle=90, shadow=True)
            
            # Add center text showing advantage
            ax4.text(0, 0, f'Male\nAdvantage:\n{advantage_pct:.1f}%',
                    ha='center', va='center', fontsize=12, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'Extreme bias detected\n(Female scores ≈ 0)',
                    ha='center', va='center', transform=ax4.transAxes,
                    fontsize=14, fontweight='bold', color='red')
        
        ax4.set_title('Gender Score Ratio Analysis')
    
    # Add overall title and adjust layout
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def plot_identical_resume_comparison(comparison_data: List[Dict[str, Any]],
                                   title: str = "Identical Resume Gender Bias Test",
                                   save_path: Optional[str] = None) -> None:
    """
    Plot comparison of identical resumes with different genders.
    
    Args:
        comparison_data: List of dictionaries containing resume comparisons
        title: Title for the plot
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Extract data for plotting
    male_scores = []
    female_scores = []
    job_titles = []
    
    for comp in comparison_data:
        if 'male_score' in comp and 'female_score' in comp:
            male_scores.append(comp['male_score'])
            female_scores.append(comp['female_score'])
            job_titles.append(comp.get('job_title', f'Job {len(job_titles)+1}'))
    
    if not male_scores:
        print("No comparison data available")
        return
    
    # Plot 1: Side-by-side score comparison
    ax1 = axes[0, 0]
    x = np.arange(len(job_titles))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, male_scores, width, label='Male Candidate',
                    color='lightblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, female_scores, width, label='Female Candidate',
                    color='lightpink', alpha=0.8)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Job Position')
    ax1.set_ylabel('Score')
    ax1.set_title('Identical Resume Scores by Gender')
    ax1.set_xticks(x)
    ax1.set_xticklabels(job_titles, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Score difference analysis
    ax2 = axes[0, 1]
    score_diffs = [m - f for m, f in zip(male_scores, female_scores)]
    pct_diffs = [(m - f) / f * 100 if f > 0 else 0 for m, f in zip(male_scores, female_scores)]
    
    bars = ax2.bar(range(len(score_diffs)), score_diffs, 
                   color=['darkgreen' if d > 0 else 'darkred' for d in score_diffs],
                   alpha=0.7)
    
    # Add percentage labels
    for i, (bar, pct) in enumerate(zip(bars, pct_diffs)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{pct:+.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Job Position')
    ax2.set_ylabel('Score Difference (Male - Female)')
    ax2.set_title('Gender Bias in Scoring (Positive = Male Advantage)')
    ax2.set_xticks(range(len(job_titles)))
    ax2.set_xticklabels(job_titles, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Scatter plot showing correlation
    ax3 = axes[1, 0]
    ax3.scatter(female_scores, male_scores, s=100, alpha=0.6, color='purple')
    
    # Add diagonal line (equal scores)
    min_score = min(min(female_scores), min(male_scores))
    max_score = max(max(female_scores), max(male_scores))
    ax3.plot([min_score, max_score], [min_score, max_score], 'r--', 
             label='Equal Scores', alpha=0.7)
    
    # Add trend line
    z = np.polyfit(female_scores, male_scores, 1)
    p = np.poly1d(z)
    ax3.plot(female_scores, p(female_scores), 'b-', label='Trend Line', alpha=0.7)
    
    ax3.set_xlabel('Female Candidate Score')
    ax3.set_ylabel('Male Candidate Score')
    ax3.set_title('Score Correlation Analysis')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    
    # Calculate statistics
    avg_male = np.mean(male_scores)
    avg_female = np.mean(female_scores)
    avg_diff = avg_male - avg_female
    avg_pct_diff = (avg_male - avg_female) / avg_female * 100 if avg_female > 0 else 0
    
    # Create summary text
    summary_text = f"""Summary Statistics:
    
Average Scores:
  Male: {avg_male:.4f}
  Female: {avg_female:.4f}
  
Bias Metrics:
  Absolute Difference: {avg_diff:.4f}
  Percentage Difference: {avg_pct_diff:+.1f}%
  
Consistency:
  Male always scored higher: {all(m > f for m, f in zip(male_scores, female_scores))}
  Tests conducted: {len(male_scores)}
  
Conclusion:
  {'🚨 SIGNIFICANT GENDER BIAS DETECTED' if avg_pct_diff > 10 else '⚠️ Gender bias present' if avg_pct_diff > 0 else '✓ No bias detected'}
    """
    
    ax4.text(0.1, 0.5, summary_text, transform=ax4.transAxes,
            fontsize=12, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title('Statistical Summary')
    ax4.axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()