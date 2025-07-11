# Forensic Dashboard

A comprehensive forensic analysis dashboard for LLM bias detection and investigation, specifically designed for legal investigations of AI systems.

## Overview

This dashboard provides interactive visualizations, real-time monitoring, and audit capabilities for forensic investigations of machine learning bias in resume screening systems. It's designed to meet legal standards for evidence collection and analysis.

## Features

### 🔐 Security & Authentication
- Multi-user authentication system
- Role-based access control (Admin, Investigator, Analyst)
- Session management with timeout
- Comprehensive audit logging
- Secure data handling and encryption

### 📊 Dashboard Components

#### Executive Summary
- High-level investigation overview
- Key performance indicators
- Risk assessment and alerts
- Data integrity status
- Investigation progress tracking

#### Technical Analysis
- Detailed model investigation
- Data quality assessment
- Statistical analysis tools
- Hypothesis testing framework
- Anomaly detection

#### Bias Analysis
- Comprehensive fairness testing
- Demographic group comparisons
- Intersectional bias detection
- Temporal bias tracking
- Statistical significance testing

#### SHAP Analysis
- Feature importance analysis
- Individual prediction explanations
- Model behavior understanding
- Interaction effect detection
- Dependency plot analysis

#### Evidently Monitoring
- Data drift detection
- Model performance tracking
- Target distribution monitoring
- Automated report generation

#### Real-time Monitoring
- Live system status
- Performance metrics streaming
- Active alert management
- Threshold monitoring

#### Audit Trail
- Complete action logging
- User access tracking
- Change documentation
- Forensic timeline

#### Export Manager
- PDF report generation
- Excel data exports
- Evidence package creation
- Raw data downloads

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- 10GB+ disk space

### Quick Start

1. **Clone or navigate to the dashboard directory:**
   ```bash
   cd /path/to/resume-screening-llm/forensic/dashboards/
   ```

2. **Run the setup script:**
   ```bash
   python3 setup.py
   ```

3. **Start the dashboard:**
   ```bash
   ./start_dashboard.sh
   ```

4. **Access the dashboard:**
   Open your browser and go to: http://localhost:8501

### Manual Installation

1. **Install requirements:**
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Create directories:**
   ```bash
   mkdir -p data logs exports
   ```

3. **Initialize database:**
   ```bash
   python3 -c "from auth import init_default_users, AuthenticationManager; from config import get_config; config = get_config(); auth = AuthenticationManager(config.security.secret_key, str(config.data_dir / 'users.db')); init_default_users(auth)"
   ```

4. **Start dashboard:**
   ```bash
   streamlit run main.py
   ```

## Configuration

### Environment Variables

- `ENVIRONMENT`: Set to "production" for production deployment
- `DEBUG`: Set to "true" for debug mode
- `SECRET_KEY`: Override default secret key (required for production)
- `DB_TYPE`: Database type (sqlite, postgresql)
- `DB_HOST`: Database host (for PostgreSQL)
- `DB_PORT`: Database port
- `DB_NAME`: Database name
- `DB_USERNAME`: Database username
- `DB_PASSWORD`: Database password

### Configuration File

Create `config/dashboard_config.json` to override default settings:

```json
{
  "database": {
    "type": "sqlite",
    "database": "forensic_dashboard.db"
  },
  "security": {
    "session_timeout": 3600,
    "max_login_attempts": 5,
    "require_2fa": false
  },
  "monitoring": {
    "refresh_interval": 30,
    "alert_threshold_bias": 0.1,
    "alert_threshold_drift": 0.15
  },
  "export": {
    "max_export_rows": 100000,
    "retention_days": 30
  }
}
```

## Default User Accounts

**⚠️ IMPORTANT: Change these passwords in production!**

- **Admin**
  - Username: `admin`
  - Password: `admin123!`
  - Permissions: Full system access

- **Investigator**
  - Username: `investigator`
  - Password: `invest123!`
  - Permissions: Read and export access

- **Analyst**
  - Username: `analyst`
  - Password: `analyst123!`
  - Permissions: Read and write analysis access

## Usage Guide

### 1. Login
- Access the dashboard at http://localhost:8501
- Use one of the default accounts or create new users via admin panel
- Select "Remember me" for extended sessions

### 2. Executive Summary
- Start here for high-level overview
- Review key metrics and alerts
- Check investigation progress

### 3. Detailed Analysis
- Use Technical Analysis for model investigation
- Use Bias Analysis for fairness testing
- Use SHAP Analysis for explainability

### 4. Evidence Collection
- Export reports via Export Manager
- Create forensic evidence packages
- Maintain chain of custody documentation

### 5. Monitoring
- Use Real-time Monitoring for live tracking
- Set up alerts for bias thresholds
- Review Audit Trail for all activities

## Data Sources

The dashboard expects evidence data in the following structure:

```
forensic/
├── evidence/
│   ├── bias_analysis_results.json
│   ├── shap_analysis_results.json
│   ├── performance_metrics.json
│   ├── audit_logs.json
│   └── evidently_reports/
├── data/
│   └── synthetic/
│       ├── synthetic_resumes.json
│       ├── synthetic_job_postings.json
│       └── synthetic_matched_pairs.json
└── dashboards/
    └── [dashboard files]
```

## Security Considerations

### Production Deployment

1. **Change Default Passwords**
   - Update all default user passwords immediately
   - Use strong, unique passwords for each account

2. **Secure Configuration**
   - Set `ENVIRONMENT=production`
   - Use strong `SECRET_KEY`
   - Enable HTTPS
   - Configure firewall rules

3. **Database Security**
   - Use PostgreSQL for production (not SQLite)
   - Enable database encryption
   - Configure backup procedures
   - Implement access controls

4. **Network Security**
   - Deploy behind reverse proxy
   - Use VPN for remote access
   - Implement IP whitelisting
   - Enable security headers

5. **Audit and Monitoring**
   - Review audit logs regularly
   - Set up log aggregation
   - Implement intrusion detection
   - Monitor for unusual activity

## Troubleshooting

### Common Issues

1. **Dashboard won't start**
   - Check Python version (3.8+)
   - Verify all requirements installed
   - Check port 8501 availability

2. **Login fails**
   - Verify username/password
   - Check account isn't locked
   - Review security logs

3. **No data visible**
   - Check evidence directory exists
   - Verify file permissions
   - Run data collection scripts

4. **Performance issues**
   - Increase system memory
   - Reduce auto-refresh frequency
   - Optimize database queries

### Log Files

Check these files for detailed error information:
- `logs/dashboard.log`: General application logs
- `logs/errors.log`: Error-specific logs
- `logs/audit.log`: User action logs
- `logs/security.log`: Security events
- `logs/performance.log`: Performance metrics

## API Documentation

The dashboard includes a REST API for programmatic access:

- Base URL: `http://localhost:8501/api/`
- Authentication: Bearer token
- Endpoints: `/api/docs` for full documentation

## Development

### Project Structure

```
dashboards/
├── auth/                    # Authentication system
├── components/             # Dashboard pages
├── config/                 # Configuration management
├── utils/                  # Utility functions
├── static/                 # Static assets
├── templates/              # HTML templates
├── main.py                # Main application
├── requirements.txt       # Dependencies
└── README.md             # This file
```

### Adding New Features

1. Create component in `components/`
2. Add route in `main.py`
3. Update navigation menu
4. Add permissions checking
5. Update documentation

### Testing

Run tests with:
```bash
python3 -m pytest tests/
```

## Legal Considerations

This dashboard is designed for forensic investigations that may be used in legal proceedings:

1. **Evidence Integrity**
   - All data modifications are logged
   - Hash verification for file integrity
   - Chain of custody documentation

2. **Audit Trail**
   - Complete user action logging
   - Tamper-evident log storage
   - Timestamped entries

3. **Data Privacy**
   - Secure handling of sensitive data
   - Access control and authorization
   - Data retention policies

4. **Compliance**
   - Support for legal discovery
   - Exportable evidence packages
   - Professional report generation

## Support

For support and questions:

- **Technical Issues**: Check troubleshooting section
- **Bug Reports**: Document with steps to reproduce
- **Feature Requests**: Submit with business justification
- **Security Issues**: Report immediately to administrators

## License

This software is designed for forensic investigation purposes. Use in accordance with applicable laws and regulations.

## Version History

- **v1.0.0** (July 2024): Initial release
  - Complete dashboard functionality
  - Authentication and security
  - Bias analysis and SHAP integration
  - Export and reporting capabilities
  - Production-ready deployment

---

**⚠️ Important Notice**: This dashboard handles sensitive data and is designed for legal investigations. Ensure proper security measures, access controls, and audit procedures are in place before deployment.