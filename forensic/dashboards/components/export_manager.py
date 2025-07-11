"""
Export Manager component for generating and downloading reports.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import json

from ..config import DashboardConfig
from ..auth import AuthenticationManager
from ..utils.data_utils import load_evidence_data
from ..utils.export_utils import generate_pdf_report, export_to_excel, create_forensic_evidence_package
from ..utils.error_handling import error_handler
from .login_page import require_permission, audit_user_action


def render(config: DashboardConfig, auth_manager: AuthenticationManager):
    """Render export manager page."""
    if not require_permission("export", auth_manager):
        st.error("You need export permissions to access this page")
        return
    
    audit_user_action("view", "export_manager")
    
    st.header("📤 Export Manager")
    
    with error_handler("Failed to load export manager"):
        evidence_data = load_evidence_data(config.evidence_dir)
        
        # Export options
        st.subheader("📋 Export Options")
        
        export_tabs = st.tabs(["📄 PDF Report", "📊 Excel Export", "📦 Evidence Package", "🗂️ Raw Data"])
        
        with export_tabs[0]:
            render_pdf_export(evidence_data, config)
        
        with export_tabs[1]:
            render_excel_export(evidence_data, config)
        
        with export_tabs[2]:
            render_evidence_package(evidence_data, config)
        
        with export_tabs[3]:
            render_raw_data_export(evidence_data, config)


def render_pdf_export(evidence_data, config):
    """Render PDF export options."""
    st.subheader("PDF Report Generation")
    
    # Report configuration
    report_title = st.text_input("Report Title", "Forensic Analysis Report")
    case_id = st.text_input("Case ID", f"CASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    include_charts = st.checkbox("Include Charts and Visualizations", True)
    include_raw_data = st.checkbox("Include Raw Data Tables", False)
    
    # Report sections
    st.markdown("**Report Sections to Include:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        executive_summary = st.checkbox("Executive Summary", True)
        bias_analysis = st.checkbox("Bias Analysis", True)
        shap_analysis = st.checkbox("SHAP Analysis", True)
    
    with col2:
        performance_metrics = st.checkbox("Performance Metrics", True)
        recommendations = st.checkbox("Recommendations", True)
        appendices = st.checkbox("Technical Appendices", False)
    
    if st.button("Generate PDF Report", type="primary"):
        with st.spinner("Generating PDF report..."):
            try:
                # Prepare report data
                report_data = {
                    'executive_summary': evidence_data.get('bias_analysis', {}),
                    'bias_analysis': evidence_data.get('bias_analysis', {}),
                    'shap_analysis': evidence_data.get('shap_analysis', {}),
                    'performance_metrics': evidence_data.get('performance_metrics', {}),
                    'conclusions': [
                        "Comprehensive bias analysis completed",
                        "SHAP analysis provides feature importance insights",
                        "Recommendations provided for bias mitigation"
                    ]
                }
                
                output_path = config.exports_dir / f"{case_id}_report.pdf"
                pdf_path = generate_pdf_report(report_data, output_path, report_title)
                
                st.success(f"PDF report generated successfully!")
                st.info(f"Report saved to: {pdf_path}")
                
                # Provide download button
                with open(pdf_path, "rb") as pdf_file:
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=pdf_file.read(),
                        file_name=f"{case_id}_report.pdf",
                        mime="application/pdf"
                    )
                
            except Exception as e:
                st.error(f"Failed to generate PDF report: {str(e)}")


def render_excel_export(evidence_data, config):
    """Render Excel export options."""
    st.subheader("Excel Data Export")
    
    # Data selection
    st.markdown("**Select data to export:**")
    
    export_bias = st.checkbox("Bias Analysis Results", True)
    export_shap = st.checkbox("SHAP Analysis Results", True)
    export_performance = st.checkbox("Performance Metrics", True)
    export_audit = st.checkbox("Audit Logs", False)
    
    # Sheet naming
    with st.expander("📝 Customize Sheet Names"):
        bias_sheet = st.text_input("Bias Analysis Sheet", "Bias_Analysis")
        shap_sheet = st.text_input("SHAP Analysis Sheet", "SHAP_Analysis")
        perf_sheet = st.text_input("Performance Sheet", "Performance_Metrics")
        audit_sheet = st.text_input("Audit Sheet", "Audit_Logs")
    
    if st.button("Generate Excel Export", type="primary"):
        with st.spinner("Generating Excel file..."):
            try:
                # Prepare export data
                export_data = {}
                sheet_names = {}
                
                if export_bias and 'bias_analysis' in evidence_data:
                    export_data['bias_analysis'] = evidence_data['bias_analysis']
                    sheet_names['bias_analysis'] = bias_sheet
                
                if export_shap and 'shap_analysis' in evidence_data:
                    export_data['shap_analysis'] = evidence_data['shap_analysis']
                    sheet_names['shap_analysis'] = shap_sheet
                
                if export_performance and 'performance_metrics' in evidence_data:
                    export_data['performance_metrics'] = evidence_data['performance_metrics']
                    sheet_names['performance_metrics'] = perf_sheet
                
                case_id = f"EXPORT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                output_path = config.exports_dir / f"{case_id}.xlsx"
                
                excel_path = export_to_excel(export_data, output_path, sheet_names)
                
                st.success("Excel file generated successfully!")
                st.info(f"File saved to: {excel_path}")
                
                # Provide download button
                with open(excel_path, "rb") as excel_file:
                    st.download_button(
                        label="📥 Download Excel File",
                        data=excel_file.read(),
                        file_name=f"{case_id}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
            except Exception as e:
                st.error(f"Failed to generate Excel export: {str(e)}")


def render_evidence_package(evidence_data, config):
    """Render evidence package creation."""
    st.subheader("Forensic Evidence Package")
    
    st.markdown("""
    **Forensic Evidence Package** includes:
    - PDF comprehensive report
    - Excel data export
    - Individual CSV files
    - Charts and visualizations
    - Metadata and chain of custody
    """)
    
    case_id = st.text_input("Case ID", f"CASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    investigator = st.text_input("Lead Investigator", st.session_state.user.username if st.session_state.get('user') else "")
    
    package_notes = st.text_area("Package Notes", "Complete forensic analysis package for legal investigation")
    
    if st.button("Create Evidence Package", type="primary"):
        with st.spinner("Creating forensic evidence package..."):
            try:
                # Create comprehensive evidence package
                package_dir = create_forensic_evidence_package(
                    evidence_data,
                    config.exports_dir,
                    case_id
                )
                
                st.success("Forensic evidence package created successfully!")
                st.info(f"Package location: {package_dir}")
                
                # Display package contents
                st.markdown("**Package Contents:**")
                package_files = list(package_dir.glob("*"))
                for file_path in package_files:
                    if file_path.is_file():
                        st.write(f"📄 {file_path.name}")
                    elif file_path.is_dir():
                        st.write(f"📁 {file_path.name}/")
                
            except Exception as e:
                st.error(f"Failed to create evidence package: {str(e)}")


def render_raw_data_export(evidence_data, config):
    """Render raw data export options."""
    st.subheader("Raw Data Export")
    
    if not evidence_data:
        st.warning("No evidence data available for export")
        return
    
    # Data preview
    st.markdown("**Available Data:**")
    
    for data_type, data_content in evidence_data.items():
        with st.expander(f"📊 {data_type.replace('_', ' ').title()}"):
            if isinstance(data_content, dict):
                st.json(data_content)
            elif isinstance(data_content, list):
                if data_content:
                    st.write(f"List with {len(data_content)} items")
                    st.json(data_content[:3])  # Show first 3 items
                else:
                    st.write("Empty list")
            else:
                st.write(str(data_content))
    
    # Export format selection
    export_format = st.selectbox("Export Format", ["JSON", "CSV"])
    
    if st.button("Export Raw Data", type="primary"):
        with st.spinner("Preparing raw data export..."):
            try:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                if export_format == "JSON":
                    export_content = json.dumps(evidence_data, indent=2, default=str)
                    filename = f"raw_data_{timestamp}.json"
                    mime_type = "application/json"
                    
                    st.download_button(
                        label="📥 Download JSON",
                        data=export_content,
                        file_name=filename,
                        mime=mime_type
                    )
                
                else:  # CSV
                    # Convert to flat structure for CSV
                    flat_data = []
                    for data_type, data_content in evidence_data.items():
                        if isinstance(data_content, dict):
                            for key, value in data_content.items():
                                flat_data.append({
                                    'Data_Type': data_type,
                                    'Key': key,
                                    'Value': str(value)
                                })
                    
                    if flat_data:
                        df = pd.DataFrame(flat_data)
                        csv_content = df.to_csv(index=False)
                        filename = f"raw_data_{timestamp}.csv"
                        
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_content,
                            file_name=filename,
                            mime="text/csv"
                        )
                    else:
                        st.warning("No data available for CSV export")
                
                st.success("Raw data export prepared!")
                
            except Exception as e:
                st.error(f"Failed to export raw data: {str(e)}")