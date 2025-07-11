#!/usr/bin/env python3
"""
Forensic Collection Framework - Example Usage
=============================================

Demonstrates how to use the forensic collection framework for
evidence collection, validation, and chain of custody management.

Author: Forensic Collection Framework
Version: 1.0
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime

# Add the collection directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forensic_collector import ForensicCollector
from chain_of_custody import ChainOfCustody
from metadata_extractor import MetadataExtractor
from evidence_validator import EvidenceValidator


def create_sample_files():
    """Create sample files for demonstration purposes."""
    temp_dir = Path(tempfile.mkdtemp(prefix="forensic_demo_"))
    
    # Create various types of sample files
    sample_files = []
    
    # Text file
    text_file = temp_dir / "evidence.txt"
    with open(text_file, 'w') as f:
        f.write("This is sample evidence text.\nCreated for forensic demonstration.\n")
    sample_files.append(text_file)
    
    # Binary file
    binary_file = temp_dir / "data.bin"
    with open(binary_file, 'wb') as f:
        f.write(b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F')
    sample_files.append(binary_file)
    
    # JSON file
    json_file = temp_dir / "config.json"
    with open(json_file, 'w') as f:
        json.dump({
            "application": "forensic_demo",
            "version": "1.0",
            "created": datetime.now().isoformat()
        }, f, indent=2)
    sample_files.append(json_file)
    
    # Create a subdirectory with files
    sub_dir = temp_dir / "subdirectory"
    sub_dir.mkdir()
    
    sub_file = sub_dir / "hidden_data.dat"
    with open(sub_file, 'wb') as f:
        f.write(b'Secret data for forensic analysis')
    sample_files.append(sub_file)
    
    print(f"Sample files created in: {temp_dir}")
    for file_path in sample_files:
        print(f"  - {file_path}")
    
    return temp_dir, sample_files


def demonstrate_metadata_extraction(sample_files):
    """Demonstrate metadata extraction capabilities."""
    print("\n" + "="*60)
    print("METADATA EXTRACTION DEMONSTRATION")
    print("="*60)
    
    extractor = MetadataExtractor()
    
    for file_path in sample_files[:2]:  # Analyze first 2 files
        print(f"\nAnalyzing: {file_path}")
        print("-" * 40)
        
        metadata = extractor.extract_metadata(str(file_path))
        
        # Display key metadata
        basic_info = metadata.get('basic_info', {})
        print(f"File: {basic_info.get('filename', 'Unknown')}")
        print(f"Size: {basic_info.get('size_human', 'Unknown')}")
        print(f"Extension: {basic_info.get('extension', 'None')}")
        
        file_type = metadata.get('file_type', {})
        print(f"MIME Type: {file_type.get('mime_type', 'Unknown')}")
        print(f"Type Confidence: {file_type.get('confidence', 'Unknown')}")
        
        filesystem = metadata.get('file_system', {})
        timestamps = filesystem.get('timestamps', {})
        if timestamps:
            print(f"Modified: {timestamps.get('modified', 'Unknown')}")
            print(f"Accessed: {timestamps.get('accessed', 'Unknown')}")


def demonstrate_evidence_collection(source_dir, evidence_storage):
    """Demonstrate evidence collection process."""
    print("\n" + "="*60)
    print("EVIDENCE COLLECTION DEMONSTRATION")
    print("="*60)
    
    # Initialize forensic collector
    collector = ForensicCollector(
        case_id="DEMO_CASE_001",
        investigator="Detective Demo"
    )
    
    print(f"Collecting evidence from: {source_dir}")
    print(f"Storage location: {evidence_storage}")
    
    # Collect evidence
    collection_result = collector.collect_evidence(
        source_path=str(source_dir),
        destination_path=str(evidence_storage),
        description="Sample evidence for framework demonstration",
        preserve_structure=True
    )
    
    print(f"\nCollection Summary:")
    print(f"  Collection ID: {collection_result['collection_id']}")
    print(f"  Items collected: {collection_result['total_items']}")
    print(f"  Errors: {collection_result['error_count']}")
    print(f"  Duration: {collection_result['duration_seconds']:.2f} seconds")
    
    return collection_result


def demonstrate_chain_of_custody(collection_result):
    """Demonstrate chain of custody management."""
    print("\n" + "="*60)
    print("CHAIN OF CUSTODY DEMONSTRATION")
    print("="*60)
    
    # Initialize chain of custody
    custody = ChainOfCustody(
        case_id=collection_result['case_id'],
        primary_investigator=collection_result['investigator']
    )
    
    # The evidence should already be added during collection
    # Let's demonstrate additional custody operations
    
    evidence_items = collection_result['evidence_items']
    if evidence_items:
        first_evidence = evidence_items[0]
        evidence_id = first_evidence['evidence_id']
        
        print(f"Demonstrating custody operations for evidence: {evidence_id}")
        
        # Log access to evidence
        custody.log_access(
            evidence_id=evidence_id,
            accessor="Forensic Analyst Smith",
            purpose="Initial examination",
            access_type="EXAMINATION"
        )
        
        # Transfer custody
        transfer_id = custody.transfer_custody(
            evidence_id=evidence_id,
            new_custodian="Senior Investigator Jones",
            reason="Advanced analysis required",
            location="Forensic Lab B"
        )
        
        print(f"Evidence transferred (Transfer ID: {transfer_id})")
        
        # Get evidence status
        status = custody.get_evidence_status(evidence_id)
        print(f"Current custodian: {status['evidence_info']['current_custodian']}")
        print(f"Access count: {status['access_count']}")
        
        # Verify chain integrity
        verification = custody.verify_chain_integrity()
        print(f"Chain integrity: {verification['overall_status']}")
        
        return custody, evidence_id
    
    return custody, None


def demonstrate_evidence_validation(collection_result, evidence_id):
    """Demonstrate evidence validation and integrity checking."""
    print("\n" + "="*60)
    print("EVIDENCE VALIDATION DEMONSTRATION")
    print("="*60)
    
    # Initialize evidence validator
    validator = EvidenceValidator("demo_integrity.db")
    
    print(f"Validating evidence: {evidence_id}")
    
    # Register evidence (should already be done during collection)
    # Validate evidence integrity
    validation_result = validator.validate_evidence(evidence_id, "Demo Validator")
    
    print(f"Validation Results:")
    print(f"  Evidence ID: {validation_result['evidence_id']}")
    print(f"  Integrity Status: {validation_result['integrity_status']}")
    print(f"  File Exists: {validation_result['file_exists']}")
    print(f"  File Accessible: {validation_result['file_accessible']}")
    
    if validation_result['integrity_status'] == 'VERIFIED':
        print("  ✓ Evidence integrity verified")
    else:
        print("  ✗ Evidence integrity compromised!")
    
    # Generate integrity report
    report = validator.get_integrity_report(evidence_id)
    print(f"\nIntegrity Report Summary:")
    print(f"  Total Evidence: {report['evidence_summary']['total_evidence']}")
    print(f"  Validation History: {len(report['validation_history'])} entries")
    print(f"  Alerts: {len(report['alerts'])}")
    print(f"  Recommendations: {len(report['recommendations'])}")
    
    return validator


def demonstrate_complete_workflow():
    """Demonstrate a complete forensic workflow."""
    print("FORENSIC COLLECTION FRAMEWORK - COMPLETE DEMONSTRATION")
    print("=" * 60)
    
    try:
        # Step 1: Create sample files
        source_dir, sample_files = create_sample_files()
        
        # Step 2: Create evidence storage directory
        evidence_storage = Path(tempfile.mkdtemp(prefix="evidence_storage_"))
        
        # Step 3: Demonstrate metadata extraction
        demonstrate_metadata_extraction(sample_files)
        
        # Step 4: Demonstrate evidence collection
        collection_result = demonstrate_evidence_collection(source_dir, evidence_storage)
        
        # Step 5: Demonstrate chain of custody
        custody, evidence_id = demonstrate_chain_of_custody(collection_result)
        
        # Step 6: Demonstrate evidence validation
        if evidence_id:
            validator = demonstrate_evidence_validation(collection_result, evidence_id)
            
            # Generate final reports
            print("\n" + "="*60)
            print("GENERATING FINAL REPORTS")
            print("="*60)
            
            # Chain of custody report
            custody_report = custody.generate_custody_report(
                str(evidence_storage / "final_custody_report.json")
            )
            print(f"Chain of custody report: {custody_report}")
            
            # Integrity report
            integrity_report = validator.export_validation_data(
                str(evidence_storage / "integrity_report.json"),
                evidence_id
            )
            print(f"Integrity report: {integrity_report}")
            
            print(f"\nAll reports and evidence stored in: {evidence_storage}")
        
        print("\n" + "="*60)
        print("DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\nDemonstration failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function for running the demonstration."""
    print("Forensic Collection Framework - Example Usage")
    print("=" * 50)
    
    # Check if this is being run as a demonstration
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        success = demonstrate_complete_workflow()
        sys.exit(0 if success else 1)
    
    # Interactive usage
    print("\nThis script demonstrates the forensic collection framework.")
    print("Available actions:")
    print("  1. Run complete demonstration (--demo)")
    print("  2. Individual component testing")
    
    choice = input("\nEnter your choice (1-2): ").strip()
    
    if choice == '1':
        success = demonstrate_complete_workflow()
        sys.exit(0 if success else 1)
    elif choice == '2':
        print("Individual component testing not implemented in this example.")
        print("Please examine the source code for component usage examples.")
    else:
        print("Invalid choice. Exiting.")


if __name__ == "__main__":
    main()