#!/usr/bin/env python3
"""
Basic Test Suite for Forensic Collection Framework
=================================================

Simple tests to validate the framework functionality.
Run with: python test_framework.py

Author: Forensic Collection Framework
Version: 1.0
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from forensic_collector import ForensicCollector
    from chain_of_custody import ChainOfCustody
    from metadata_extractor import MetadataExtractor
    from evidence_validator import EvidenceValidator
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required dependencies are installed.")
    sys.exit(1)


class TestResults:
    """Simple test result tracking."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def test_pass(self, test_name):
        print(f"✓ {test_name}")
        self.passed += 1
    
    def test_fail(self, test_name, error):
        print(f"✗ {test_name}: {error}")
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\nTest Results: {self.passed}/{total} passed")
        if self.failed > 0:
            print("Failures:")
            for error in self.errors:
                print(f"  - {error}")
        return self.failed == 0


def create_test_files():
    """Create temporary test files."""
    temp_dir = Path(tempfile.mkdtemp(prefix="forensic_test_"))
    
    # Create test files
    test_files = []
    
    # Text file
    text_file = temp_dir / "test.txt"
    with open(text_file, 'w') as f:
        f.write("Test evidence file\nLine 2\nLine 3\n")
    test_files.append(text_file)
    
    # Binary file
    binary_file = temp_dir / "test.bin"
    with open(binary_file, 'wb') as f:
        f.write(bytes(range(256)))
    test_files.append(binary_file)
    
    # JSON file
    json_file = temp_dir / "test.json"
    with open(json_file, 'w') as f:
        json.dump({"test": True, "data": [1, 2, 3]}, f)
    test_files.append(json_file)
    
    return temp_dir, test_files


def test_metadata_extractor(results, test_files):
    """Test metadata extractor functionality."""
    try:
        extractor = MetadataExtractor()
        
        for test_file in test_files:
            metadata = extractor.extract_metadata(str(test_file))
            
            # Check required fields
            required_fields = ['basic_info', 'file_system', 'file_type']
            for field in required_fields:
                if field not in metadata:
                    results.test_fail(f"MetadataExtractor - {field} missing", f"Field {field} not found")
                    return
            
            # Check basic info
            basic_info = metadata['basic_info']
            if 'filename' not in basic_info or 'size_bytes' not in basic_info:
                results.test_fail("MetadataExtractor - basic_info incomplete", "Missing required basic info fields")
                return
        
        results.test_pass("MetadataExtractor - Basic functionality")
        
    except Exception as e:
        results.test_fail("MetadataExtractor", str(e))


def test_forensic_collector(results, source_dir):
    """Test forensic collector functionality."""
    try:
        # Create evidence storage directory
        evidence_dir = Path(tempfile.mkdtemp(prefix="evidence_test_"))
        
        # Initialize collector
        collector = ForensicCollector(
            case_id="TEST_CASE_001",
            investigator="Test Investigator"
        )
        
        # Test hash calculation
        test_file = next(source_dir.glob("*.txt"))
        hashes = collector.calculate_hashes(str(test_file))
        
        required_hashes = ['md5', 'sha1', 'sha256']
        for hash_type in required_hashes:
            if hash_type not in hashes or not hashes[hash_type]:
                results.test_fail("ForensicCollector - Hash calculation", f"Missing {hash_type} hash")
                return
        
        results.test_pass("ForensicCollector - Hash calculation")
        
        # Test timestamp extraction
        timestamps = collector.extract_mac_timestamps(str(test_file))
        required_timestamps = ['modified', 'accessed', 'size']
        for ts in required_timestamps:
            if ts not in timestamps:
                results.test_fail("ForensicCollector - Timestamps", f"Missing {ts} timestamp")
                return
        
        results.test_pass("ForensicCollector - Timestamp extraction")
        
        # Test evidence collection
        collection_result = collector.collect_evidence(
            source_path=str(source_dir),
            destination_path=str(evidence_dir),
            description="Test evidence collection"
        )
        
        if collection_result['total_items'] == 0:
            results.test_fail("ForensicCollector - Collection", "No items collected")
            return
        
        if collection_result['error_count'] > 0:
            results.test_fail("ForensicCollector - Collection", f"{collection_result['error_count']} errors during collection")
            return
        
        results.test_pass("ForensicCollector - Evidence collection")
        
        return collector, collection_result
        
    except Exception as e:
        results.test_fail("ForensicCollector", str(e))
        return None, None


def test_chain_of_custody(results, collector, collection_result):
    """Test chain of custody functionality."""
    if not collector or not collection_result:
        results.test_fail("ChainOfCustody", "Collector not available")
        return None
    
    try:
        custody = collector.chain_of_custody
        
        # Test basic initialization
        if custody.case_id != collection_result['case_id']:
            results.test_fail("ChainOfCustody - Initialization", "Case ID mismatch")
            return None
        
        results.test_pass("ChainOfCustody - Initialization")
        
        # Test evidence registry
        if len(custody.evidence_registry) == 0:
            results.test_fail("ChainOfCustody - Evidence registry", "No evidence registered")
            return None
        
        results.test_pass("ChainOfCustody - Evidence registry")
        
        # Test access logging
        evidence_items = collection_result['evidence_items']
        if evidence_items:
            evidence_id = evidence_items[0]['evidence_id']
            
            custody.log_access(
                evidence_id=evidence_id,
                accessor="Test Analyst",
                purpose="Testing access logging"
            )
            
            evidence_status = custody.get_evidence_status(evidence_id)
            if evidence_status['access_count'] == 0:
                results.test_fail("ChainOfCustody - Access logging", "Access not logged")
                return None
            
            results.test_pass("ChainOfCustody - Access logging")
        
        # Test integrity verification
        verification = custody.verify_chain_integrity()
        if verification['overall_status'] not in ['VERIFIED', 'ISSUES_FOUND']:
            results.test_fail("ChainOfCustody - Integrity verification", "Invalid verification status")
            return None
        
        results.test_pass("ChainOfCustody - Integrity verification")
        
        return custody, evidence_id if evidence_items else None
        
    except Exception as e:
        results.test_fail("ChainOfCustody", str(e))
        return None, None


def test_evidence_validator(results, collector, evidence_id):
    """Test evidence validator functionality."""
    if not collector or not evidence_id:
        results.test_fail("EvidenceValidator", "Prerequisites not available")
        return
    
    try:
        validator = collector.evidence_validator
        
        # Test evidence validation
        validation_result = validator.validate_evidence(evidence_id, "Test Validator")
        
        if 'integrity_status' not in validation_result:
            results.test_fail("EvidenceValidator - Validation", "Missing integrity status")
            return
        
        if validation_result['integrity_status'] not in ['VERIFIED', 'COMPROMISED', 'MISSING']:
            results.test_fail("EvidenceValidator - Validation", "Invalid integrity status")
            return
        
        results.test_pass("EvidenceValidator - Evidence validation")
        
        # Test integrity report
        report = validator.get_integrity_report(evidence_id)
        
        required_report_fields = ['evidence_summary', 'validation_history', 'alerts']
        for field in required_report_fields:
            if field not in report:
                results.test_fail("EvidenceValidator - Report", f"Missing {field} in report")
                return
        
        results.test_pass("EvidenceValidator - Integrity report")
        
    except Exception as e:
        results.test_fail("EvidenceValidator", str(e))


def run_tests():
    """Run all tests."""
    print("Forensic Collection Framework - Test Suite")
    print("=" * 50)
    
    results = TestResults()
    
    try:
        # Create test files
        print("Creating test files...")
        source_dir, test_files = create_test_files()
        print(f"Test files created in: {source_dir}")
        
        # Test metadata extractor
        print("\nTesting MetadataExtractor...")
        test_metadata_extractor(results, test_files)
        
        # Test forensic collector
        print("\nTesting ForensicCollector...")
        collector, collection_result = test_forensic_collector(results, source_dir)
        
        # Test chain of custody
        print("\nTesting ChainOfCustody...")
        custody, evidence_id = test_chain_of_custody(results, collector, collection_result)
        
        # Test evidence validator
        print("\nTesting EvidenceValidator...")
        test_evidence_validator(results, collector, evidence_id)
        
        # Cleanup
        print("\nCleaning up test files...")
        shutil.rmtree(source_dir)
        
        # Final results
        success = results.summary()
        return success
        
    except Exception as e:
        print(f"Test setup failed: {e}")
        return False


def main():
    """Main test function."""
    success = run_tests()
    
    if success:
        print("\n🎉 All tests passed! The framework is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        sys.exit(1)


if __name__ == "__main__":
    main()