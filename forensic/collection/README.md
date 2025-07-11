# Forensic Collection Framework

A comprehensive digital forensics collection and validation framework designed for maintaining chain of custody and evidence integrity in legal investigations.

## Overview

This framework provides production-ready tools for:
- **Forensic Evidence Collection**: Secure collection with hash verification and metadata preservation
- **Chain of Custody Management**: Complete tracking of evidence handling and transfers
- **Metadata Extraction**: Comprehensive file analysis and forensic timeline creation
- **Evidence Validation**: Continuous integrity monitoring and tampering detection

## Features

### 🔍 Forensic Collector (`forensic_collector.py`)
- **Multi-hash verification**: MD5, SHA1, SHA256
- **MAC timestamp extraction**: Modified, Accessed, Created times
- **Interactive collection workflow**: User prompts for storage location and case details
- **Comprehensive logging**: All collection activities and errors
- **Metadata preservation**: Original file attributes and permissions
- **Batch processing**: Single files or entire directory trees

### 🔗 Chain of Custody (`chain_of_custody.py`)
- **Evidence tracking**: Complete lifecycle management
- **Transfer logging**: Custody changes with reasons and timestamps
- **Access control**: Detailed access logs for examinations
- **Integrity verification**: Cryptographic signatures for all entries
- **Report generation**: Comprehensive custody documentation
- **Legal compliance**: Audit trails suitable for court proceedings

### 📊 Metadata Extractor (`metadata_extractor.py`)
- **File system analysis**: Permissions, timestamps, ownership
- **Content analysis**: File type detection and structure analysis
- **Image metadata**: EXIF data extraction from photos
- **Archive analysis**: Contents of ZIP, TAR, and other archives
- **Security analysis**: Entropy calculation and string extraction
- **Relationship mapping**: File dependencies and correlations

### ✅ Evidence Validator (`evidence_validator.py`)
- **Integrity monitoring**: Continuous hash verification
- **Tampering detection**: Sophisticated analysis of file changes
- **Automated alerts**: Real-time notifications of integrity issues
- **Historical tracking**: Complete validation history
- **Batch validation**: Verify all evidence at once
- **Reporting**: Detailed integrity reports with recommendations

## Installation

### Prerequisites
```bash
# Python 3.8 or higher required
python --version

# Install required system packages (Ubuntu/Debian)
sudo apt-get install python3-magic libmagic1

# Install required system packages (CentOS/RHEL)
sudo yum install python3-magic file-devel

# Install required system packages (macOS)
brew install libmagic
```

### Python Dependencies
```bash
# Install from requirements file
pip install -r requirements.txt

# Or install individual packages
pip install python-magic Pillow PyMuPDF
```

### Optional Dependencies
For enhanced functionality, install additional packages:
```bash
pip install pyexiv2 pefile pyelftools ssdeep yara-python
```

## Quick Start

### Basic Evidence Collection
```python
from forensic_collector import ForensicCollector

# Initialize collector
collector = ForensicCollector(
    case_id="CASE_2024_001",
    investigator="Detective Smith"
)

# Collect evidence
result = collector.collect_evidence(
    source_path="/path/to/evidence",
    destination_path="/secure/evidence/storage",
    description="Suspect's computer files"
)

print(f"Collected {result['total_items']} items")
```

### Command Line Usage
```bash
# Collect evidence interactively
python forensic_collector.py /path/to/evidence --destination /evidence/storage

# Specify case details
python forensic_collector.py /path/to/evidence \
    --destination /evidence/storage \
    --case-id "CASE_2024_001" \
    --investigator "Detective Smith" \
    --description "Laptop hard drive image"
```

### Chain of Custody Management
```python
from chain_of_custody import ChainOfCustody

# Initialize chain of custody
custody = ChainOfCustody("CASE_2024_001", "Detective Smith")

# Log evidence access
custody.log_access(
    evidence_id="evidence_uuid",
    accessor="Forensic Analyst Jones",
    purpose="Malware analysis"
)

# Transfer custody
custody.transfer_custody(
    evidence_id="evidence_uuid",
    new_custodian="Senior Investigator Brown",
    reason="Court preparation"
)

# Generate custody report
report_path = custody.generate_custody_report()
```

### Evidence Validation
```python
from evidence_validator import EvidenceValidator

# Initialize validator
validator = EvidenceValidator()

# Validate specific evidence
result = validator.validate_evidence("evidence_uuid")
print(f"Integrity: {result['integrity_status']}")

# Start continuous monitoring
validator.start_monitoring(interval_seconds=3600)  # Check every hour
```

## Advanced Usage

### Comprehensive Workflow
```python
from forensic_collector import ForensicCollector
from chain_of_custody import ChainOfCustody
from evidence_validator import EvidenceValidator

# 1. Collect evidence
collector = ForensicCollector("CASE_2024_001", "Detective Smith")
collection_result = collector.collect_evidence(
    source_path="/suspect/laptop",
    destination_path="/secure/evidence"
)

# 2. Set up chain of custody (automatically initialized during collection)
custody = collector.chain_of_custody

# 3. Start integrity monitoring
validator = collector.evidence_validator
validator.start_monitoring(interval_seconds=1800)  # 30 minutes

# 4. Generate reports
custody_report = custody.generate_custody_report()
integrity_report = validator.get_integrity_report()
```

### Metadata Analysis
```python
from metadata_extractor import MetadataExtractor

extractor = MetadataExtractor()
metadata = extractor.extract_metadata("/path/to/file")

# Access different metadata categories
print("File System:", metadata['file_system'])
print("File Type:", metadata['file_type'])
print("Security Attributes:", metadata['security_attributes'])
print("Relationships:", metadata['relationships'])
```

## Configuration

### Logging Configuration
```python
import logging

# Configure logging level
logging.basicConfig(level=logging.DEBUG)

# Custom log format
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=log_format)
```

### Database Configuration
```python
# Custom database location for evidence validator
validator = EvidenceValidator(database_path="/secure/validation.db")
```

## Security Considerations

### Evidence Storage
- Store evidence on write-protected media when possible
- Use encrypted storage for sensitive evidence
- Implement access controls on evidence directories
- Regular backup of evidence and metadata

### Chain of Custody
- All custody transfers must be documented
- Access logs should be immutable
- Regular integrity verification is essential
- Maintain offline backups of custody records

### Validation Database
- Protect validation database from unauthorized access
- Regular backups of integrity records
- Monitor for database tampering
- Use separate database for different cases

## File Formats Supported

### Analysis Capabilities
- **Images**: JPEG, PNG, TIFF, BMP (with EXIF data)
- **Documents**: PDF (metadata and structure)
- **Archives**: ZIP, TAR, JAR, WAR
- **Executables**: PE, ELF, Mach-O (basic analysis)
- **Text**: Various encodings and formats
- **Binary**: Hex dumps and string extraction

### Hash Algorithms
- **MD5**: Legacy support (not recommended for new investigations)
- **SHA1**: Transitional support
- **SHA256**: Recommended for new investigations
- **SHA512**: Available through custom implementation

## Reporting

### Chain of Custody Reports
- Complete evidence lifecycle
- All custody transfers and access logs
- Integrity verification results
- Timestamps and digital signatures
- Legal compliance formatting

### Integrity Reports
- Evidence status summary
- Validation history
- Alert notifications
- Tampering analysis
- Recommendations for action

### Metadata Reports
- Comprehensive file analysis
- Forensic timeline creation
- File relationship mapping
- Security assessment
- JSON and HTML formats

## API Reference

### ForensicCollector
```python
class ForensicCollector:
    def __init__(self, case_id: str, investigator: str)
    def collect_evidence(self, source_path: str, destination_path: str, 
                        description: str = "", preserve_structure: bool = True) -> Dict
    def calculate_hashes(self, file_path: str) -> Dict[str, str]
    def extract_mac_timestamps(self, file_path: str) -> Dict[str, str]
```

### ChainOfCustody
```python
class ChainOfCustody:
    def __init__(self, case_id: str, primary_investigator: str)
    def add_evidence(self, evidence_record: Dict)
    def transfer_custody(self, evidence_id: str, new_custodian: str, 
                        reason: str, location: str = None) -> str
    def log_access(self, evidence_id: str, accessor: str, purpose: str, 
                   access_type: str = 'EXAMINATION')
    def generate_custody_report(self, output_path: str = None) -> str
    def verify_chain_integrity(self) -> Dict[str, Any]
```

### EvidenceValidator
```python
class EvidenceValidator:
    def __init__(self, database_path: str = None)
    def register_evidence(self, evidence_record: Dict[str, Any]) -> str
    def validate_evidence(self, evidence_id: str, validator: str = "System") -> Dict[str, Any]
    def validate_all_evidence(self, validator: str = "System") -> Dict[str, Any]
    def start_monitoring(self, interval_seconds: int = 300)
    def get_integrity_report(self, evidence_id: str = None) -> Dict[str, Any]
```

### MetadataExtractor
```python
class MetadataExtractor:
    def __init__(self)
    def extract_metadata(self, file_path: str) -> Dict[str, Any]
```

## Error Handling

The framework includes comprehensive error handling:

```python
try:
    collector = ForensicCollector("CASE_001", "Detective")
    result = collector.collect_evidence("/source", "/destination")
except Exception as e:
    print(f"Collection failed: {e}")
    # Error details are logged automatically
```

All errors are logged with full context and timestamps for forensic review.

## Testing

### Run Example Demonstration
```bash
python example_usage.py --demo
```

### Unit Tests
```bash
pytest tests/
```

### Coverage Report
```bash
pytest --cov=forensic_collection tests/
```

## Legal Compliance

This framework is designed to meet forensic investigation requirements:

- **Admissibility**: Evidence collection follows established digital forensics standards
- **Integrity**: Cryptographic hashing ensures evidence hasn't been altered
- **Chain of Custody**: Complete documentation of evidence handling
- **Auditability**: All actions are logged with timestamps and digital signatures
- **Reproducibility**: Collection and validation processes can be independently verified

## Contributing

### Development Setup
```bash
git clone <repository>
cd forensic-collection
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Code Style
- Follow PEP 8 guidelines
- Use type hints for all functions
- Include comprehensive docstrings
- Write unit tests for new features

### Submitting Changes
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For technical support or questions:
- Open an issue on the project repository
- Contact the development team
- Review the documentation and examples

## Version History

- **v1.0.0**: Initial release with core functionality
  - Evidence collection and validation
  - Chain of custody management
  - Metadata extraction
  - Integrity monitoring

## Acknowledgments

This framework incorporates best practices from:
- Digital Forensics Research Workshop (DFRWS)
- National Institute of Standards and Technology (NIST)
- International Association of Computer Investigative Specialists (IACIS)
- SANS Digital Forensics and Incident Response community