"""
Export utilities for generating reports and exporting data.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
import base64
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import logging

from .error_handling import ExportError


def generate_pdf_report(data: Dict[str, Any], output_path: Path, 
                       report_title: str = "Forensic Analysis Report",
                       include_charts: bool = True) -> Path:
    """
    Generate a comprehensive PDF report from analysis data.
    
    Args:
        data: Analysis data to include in report
        output_path: Path for output PDF file
        report_title: Title for the report
        include_charts: Whether to include charts in the PDF
    
    Returns:
        Path to generated PDF file
    
    Raises:
        ExportError: If PDF generation fails
    """
    try:
        # Create PDF document
        doc = SimpleDocTemplate(str(output_path), pagesize=A4)
        story = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.darkblue,
            alignment=1  # Center
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=12,
            textColor=colors.darkblue
        )
        
        # Title page
        story.append(Paragraph(report_title, title_style))
        story.append(Spacer(1, 20))
        story.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
        story.append(Spacer(1, 30))
        
        # Executive Summary
        if 'executive_summary' in data:
            story.append(Paragraph("Executive Summary", heading_style))
            summary = data['executive_summary']
            
            if isinstance(summary, dict):
                for key, value in summary.items():
                    story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b> {value}", styles['Normal']))
                    story.append(Spacer(1, 6))
            else:
                story.append(Paragraph(str(summary), styles['Normal']))
            
            story.append(Spacer(1, 20))
        
        # Bias Analysis Section
        if 'bias_analysis' in data:
            story.append(Paragraph("Bias Analysis Results", heading_style))
            bias_data = data['bias_analysis']
            
            if isinstance(bias_data, dict):
                # Create summary table
                bias_table_data = [['Metric', 'Value', 'Status']]
                
                for metric, value in bias_data.items():
                    if isinstance(value, (int, float)):
                        status = "Pass" if value < 0.1 else "Fail"
                        bias_table_data.append([metric.replace('_', ' ').title(), f"{value:.4f}", status])
                
                if len(bias_table_data) > 1:
                    bias_table = Table(bias_table_data)
                    bias_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 14),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(bias_table)
            
            story.append(Spacer(1, 20))
        
        # SHAP Analysis Section
        if 'shap_analysis' in data:
            story.append(Paragraph("SHAP Analysis Results", heading_style))
            shap_data = data['shap_analysis']
            
            if isinstance(shap_data, dict):
                for key, value in shap_data.items():
                    if key != 'plots':  # Skip plot data for now
                        story.append(Paragraph(f"<b>{key.replace('_', ' ').title()}:</b> {value}", styles['Normal']))
                        story.append(Spacer(1, 6))
            
            story.append(Spacer(1, 20))
        
        # Performance Metrics Section
        if 'performance_metrics' in data:
            story.append(Paragraph("Performance Metrics", heading_style))
            perf_data = data['performance_metrics']
            
            if isinstance(perf_data, dict):
                perf_table_data = [['Metric', 'Value']]
                
                for metric, value in perf_data.items():
                    if isinstance(value, (int, float)):
                        perf_table_data.append([metric.replace('_', ' ').title(), f"{value:.4f}"])
                    else:
                        perf_table_data.append([metric.replace('_', ' ').title(), str(value)])
                
                if len(perf_table_data) > 1:
                    perf_table = Table(perf_table_data)
                    perf_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    story.append(perf_table)
            
            story.append(Spacer(1, 20))
        
        # Conclusions and Recommendations
        story.append(PageBreak())
        story.append(Paragraph("Conclusions and Recommendations", heading_style))
        
        if 'conclusions' in data:
            conclusions = data['conclusions']
            if isinstance(conclusions, list):
                for conclusion in conclusions:
                    story.append(Paragraph(f"• {conclusion}", styles['Normal']))
                    story.append(Spacer(1, 6))
            else:
                story.append(Paragraph(str(conclusions), styles['Normal']))
        else:
            story.append(Paragraph("Based on the analysis performed, the following key findings were identified:", styles['Normal']))
            story.append(Spacer(1, 12))
            story.append(Paragraph("• Comprehensive bias analysis has been conducted", styles['Normal']))
            story.append(Paragraph("• SHAP values provide feature importance insights", styles['Normal']))
            story.append(Paragraph("• Performance metrics indicate model behavior", styles['Normal']))
            story.append(Paragraph("• Further investigation may be warranted based on findings", styles['Normal']))
        
        # Build PDF
        doc.build(story)
        
        logging.getLogger(__name__).info(f"PDF report generated: {output_path}")
        return output_path
        
    except Exception as e:
        raise ExportError(f"Failed to generate PDF report: {str(e)}")


def export_to_excel(data: Dict[str, Any], output_path: Path, 
                   sheet_names: Optional[Dict[str, str]] = None) -> Path:
    """
    Export data to Excel file with multiple sheets.
    
    Args:
        data: Data to export
        output_path: Path for output Excel file
        sheet_names: Custom sheet names mapping
    
    Returns:
        Path to generated Excel file
    
    Raises:
        ExportError: If Excel export fails
    """
    try:
        with pd.ExcelWriter(str(output_path), engine='openpyxl') as writer:
            
            # Export bias analysis
            if 'bias_analysis' in data:
                bias_df = _dict_to_dataframe(data['bias_analysis'], 'Metric', 'Value')
                sheet_name = sheet_names.get('bias_analysis', 'Bias Analysis') if sheet_names else 'Bias Analysis'
                bias_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Export SHAP analysis
            if 'shap_analysis' in data:
                shap_df = _dict_to_dataframe(data['shap_analysis'], 'Feature', 'Importance')
                sheet_name = sheet_names.get('shap_analysis', 'SHAP Analysis') if sheet_names else 'SHAP Analysis'
                shap_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Export performance metrics
            if 'performance_metrics' in data:
                perf_df = _dict_to_dataframe(data['performance_metrics'], 'Metric', 'Value')
                sheet_name = sheet_names.get('performance_metrics', 'Performance') if sheet_names else 'Performance'
                perf_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Export audit logs if available
            if 'audit_logs' in data:
                audit_data = data['audit_logs']
                if isinstance(audit_data, list):
                    audit_df = pd.DataFrame(audit_data)
                    sheet_name = sheet_names.get('audit_logs', 'Audit Logs') if sheet_names else 'Audit Logs'
                    audit_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Export raw data if available
            if 'raw_data' in data and isinstance(data['raw_data'], pd.DataFrame):
                sheet_name = sheet_names.get('raw_data', 'Raw Data') if sheet_names else 'Raw Data'
                data['raw_data'].to_excel(writer, sheet_name=sheet_name, index=False)
        
        logging.getLogger(__name__).info(f"Excel file exported: {output_path}")
        return output_path
        
    except Exception as e:
        raise ExportError(f"Failed to export to Excel: {str(e)}")


def export_to_csv(data: Union[pd.DataFrame, Dict[str, Any]], output_path: Path) -> Path:
    """
    Export data to CSV file.
    
    Args:
        data: Data to export
        output_path: Path for output CSV file
    
    Returns:
        Path to generated CSV file
    
    Raises:
        ExportError: If CSV export fails
    """
    try:
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path, index=False)
        elif isinstance(data, dict):
            # Convert dict to DataFrame
            df = _dict_to_dataframe(data)
            df.to_csv(output_path, index=False)
        else:
            raise ExportError(f"Unsupported data type for CSV export: {type(data)}")
        
        logging.getLogger(__name__).info(f"CSV file exported: {output_path}")
        return output_path
        
    except Exception as e:
        raise ExportError(f"Failed to export to CSV: {str(e)}")


def export_charts_to_html(charts: List[go.Figure], output_path: Path, 
                         title: str = "Forensic Analysis Charts") -> Path:
    """
    Export Plotly charts to HTML file.
    
    Args:
        charts: List of Plotly figures
        output_path: Path for output HTML file
        title: Title for the HTML page
    
    Returns:
        Path to generated HTML file
    
    Raises:
        ExportError: If HTML export fails
    """
    try:
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .chart-container {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; }}
                .chart-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
                .header {{ text-align: center; margin-bottom: 40px; }}
                .timestamp {{ color: #666; font-size: 14px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p class="timestamp">Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """
        
        for i, chart in enumerate(charts):
            chart_html = chart.to_html(include_plotlyjs=False, div_id=f"chart_{i}")
            html_content += f"""
            <div class="chart-container">
                <div class="chart-title">Chart {i + 1}</div>
                {chart_html}
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logging.getLogger(__name__).info(f"HTML charts exported: {output_path}")
        return output_path
        
    except Exception as e:
        raise ExportError(f"Failed to export charts to HTML: {str(e)}")


def create_chart_from_data(data: Dict[str, Any], chart_type: str = "bar", 
                          title: str = "Analysis Results") -> go.Figure:
    """
    Create a Plotly chart from data.
    
    Args:
        data: Data to visualize
        chart_type: Type of chart (bar, line, scatter, pie)
        title: Chart title
    
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    if chart_type == "bar":
        if isinstance(data, dict):
            x_values = list(data.keys())
            y_values = list(data.values())
            fig.add_trace(go.Bar(x=x_values, y=y_values, name=title))
        
    elif chart_type == "line":
        if isinstance(data, dict):
            x_values = list(data.keys())
            y_values = list(data.values())
            fig.add_trace(go.Scatter(x=x_values, y=y_values, mode='lines+markers', name=title))
    
    elif chart_type == "pie":
        if isinstance(data, dict):
            labels = list(data.keys())
            values = list(data.values())
            fig.add_trace(go.Pie(labels=labels, values=values, name=title))
    
    fig.update_layout(title=title, height=500)
    return fig


def generate_executive_summary_chart(bias_results: Dict[str, float]) -> go.Figure:
    """
    Generate executive summary chart for bias analysis.
    
    Args:
        bias_results: Bias analysis results
    
    Returns:
        Plotly figure
    """
    # Create bias metrics chart
    metrics = []
    values = []
    colors_list = []
    
    for metric, value in bias_results.items():
        if isinstance(value, (int, float)):
            metrics.append(metric.replace('_', ' ').title())
            values.append(value)
            # Color coding based on threshold
            if value < 0.05:
                colors_list.append('green')
            elif value < 0.1:
                colors_list.append('yellow')
            else:
                colors_list.append('red')
    
    fig = go.Figure(data=[
        go.Bar(
            x=metrics,
            y=values,
            marker_color=colors_list,
            text=[f"{v:.4f}" for v in values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Bias Analysis Overview",
        xaxis_title="Bias Metrics",
        yaxis_title="Bias Score",
        height=500,
        template="plotly_white"
    )
    
    # Add threshold lines
    fig.add_hline(y=0.05, line_dash="dash", line_color="green", 
                  annotation_text="Low Risk Threshold")
    fig.add_hline(y=0.1, line_dash="dash", line_color="orange", 
                  annotation_text="High Risk Threshold")
    
    return fig


def _dict_to_dataframe(data: Dict[str, Any], key_name: str = "Key", 
                      value_name: str = "Value") -> pd.DataFrame:
    """
    Convert dictionary to DataFrame for export.
    
    Args:
        data: Dictionary to convert
        key_name: Name for key column
        value_name: Name for value column
    
    Returns:
        DataFrame representation
    """
    if not data:
        return pd.DataFrame(columns=[key_name, value_name])
    
    rows = []
    for key, value in data.items():
        if isinstance(value, dict):
            # Flatten nested dictionary
            for sub_key, sub_value in value.items():
                rows.append({key_name: f"{key}.{sub_key}", value_name: sub_value})
        elif isinstance(value, list):
            # Convert list to string
            rows.append({key_name: key, value_name: ", ".join(map(str, value))})
        else:
            rows.append({key_name: key, value_name: value})
    
    return pd.DataFrame(rows)


def create_forensic_evidence_package(data: Dict[str, Any], output_dir: Path, 
                                    case_id: str) -> Path:
    """
    Create a complete forensic evidence package with all exports.
    
    Args:
        data: All analysis data
        output_dir: Directory for output files
        case_id: Case identifier
    
    Returns:
        Path to evidence package directory
    
    Raises:
        ExportError: If package creation fails
    """
    try:
        # Create case directory
        case_dir = output_dir / f"case_{case_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        case_dir.mkdir(exist_ok=True)
        
        # Generate PDF report
        pdf_path = case_dir / f"forensic_report_{case_id}.pdf"
        generate_pdf_report(data, pdf_path, f"Forensic Analysis Report - Case {case_id}")
        
        # Export to Excel
        excel_path = case_dir / f"analysis_data_{case_id}.xlsx"
        export_to_excel(data, excel_path)
        
        # Export individual CSV files
        csv_dir = case_dir / "csv_exports"
        csv_dir.mkdir(exist_ok=True)
        
        for data_type, data_content in data.items():
            if data_type in ['bias_analysis', 'shap_analysis', 'performance_metrics']:
                csv_path = csv_dir / f"{data_type}_{case_id}.csv"
                export_to_csv(data_content, csv_path)
        
        # Create charts HTML
        if 'charts' in data:
            charts_path = case_dir / f"charts_{case_id}.html"
            export_charts_to_html(data['charts'], charts_path)
        
        # Create metadata file
        metadata = {
            'case_id': case_id,
            'generated_at': datetime.now().isoformat(),
            'files': {
                'pdf_report': pdf_path.name,
                'excel_data': excel_path.name,
                'csv_directory': csv_dir.name,
                'charts_html': f"charts_{case_id}.html" if 'charts' in data else None
            },
            'data_summary': {
                'total_data_types': len(data),
                'data_types': list(data.keys())
            }
        }
        
        metadata_path = case_dir / "package_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logging.getLogger(__name__).info(f"Forensic evidence package created: {case_dir}")
        return case_dir
        
    except Exception as e:
        raise ExportError(f"Failed to create forensic evidence package: {str(e)}")