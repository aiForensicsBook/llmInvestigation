#!/usr/bin/env python3
"""
Chain of Custody Management
===========================

Manages the chain of custody for digital evidence, tracking all
handling, transfers, and access to maintain legal admissibility.

Author: Forensic Collection Framework
Version: 1.0
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid
import hashlib


class ChainOfCustody:
    """Manages chain of custody for forensic evidence."""
    
    def __init__(self, case_id: str, primary_investigator: str):
        """Initialize chain of custody tracking.
        
        Args:
            case_id: Unique case identifier
            primary_investigator: Primary investigating officer/analyst
        """
        self.case_id = case_id
        self.primary_investigator = primary_investigator
        self.custody_log = []
        self.evidence_registry = {}
        self.transfer_log = []
        
        # Setup logging
        self.logger = logging.getLogger('ChainOfCustody')
        
        # Initialize custody chain
        self._initialize_custody_chain()
    
    def _initialize_custody_chain(self):
        """Initialize the custody chain with case opening."""
        initial_entry = {
            'entry_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'action': 'CASE_OPENED',
            'officer': self.primary_investigator,
            'details': f'Case {self.case_id} opened by {self.primary_investigator}',
            'evidence_affected': [],
            'location': 'Investigation Unit',
            'signature_hash': None
        }
        
        initial_entry['signature_hash'] = self._generate_entry_signature(initial_entry)
        self.custody_log.append(initial_entry)
        
        self.logger.info(f"Chain of custody initialized for case {self.case_id}")
    
    def _generate_entry_signature(self, entry: Dict) -> str:
        """Generate a cryptographic signature for a custody entry.
        
        Args:
            entry: Custody log entry
            
        Returns:
            SHA256 hash signature
        """
        # Create signature from key fields
        signature_data = {
            'timestamp': entry['timestamp'],
            'action': entry['action'],
            'officer': entry['officer'],
            'details': entry['details'],
            'evidence_affected': sorted(entry.get('evidence_affected', []))
        }
        
        signature_string = json.dumps(signature_data, sort_keys=True)
        return hashlib.sha256(signature_string.encode()).hexdigest()
    
    def add_evidence(self, evidence_record: Dict):
        """Add new evidence to the custody chain.
        
        Args:
            evidence_record: Evidence metadata record
        """
        evidence_id = evidence_record['evidence_id']
        
        # Register evidence in registry
        self.evidence_registry[evidence_id] = {
            'evidence_id': evidence_id,
            'original_path': evidence_record['original_path'],
            'evidence_path': evidence_record['evidence_path'],
            'description': evidence_record['description'],
            'collection_time': evidence_record['collection_time'],
            'collector': evidence_record['collector'],
            'original_hash': evidence_record['original_hashes']['sha256'],
            'current_custodian': self.primary_investigator,
            'custody_status': 'COLLECTED',
            'access_log': []
        }
        
        # Add custody log entry
        custody_entry = {
            'entry_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'action': 'EVIDENCE_COLLECTED',
            'officer': evidence_record['collector'],
            'details': f'Evidence collected: {evidence_record["description"]}',
            'evidence_affected': [evidence_id],
            'location': 'Evidence Storage',
            'original_path': evidence_record['original_path'],
            'evidence_path': evidence_record['evidence_path'],
            'hash_verification': evidence_record['integrity_verified']
        }
        
        custody_entry['signature_hash'] = self._generate_entry_signature(custody_entry)
        self.custody_log.append(custody_entry)
        
        self.logger.info(f"Evidence {evidence_id} added to chain of custody")
    
    def transfer_custody(self, evidence_id: str, new_custodian: str, 
                        reason: str, location: str = None) -> str:
        """Transfer custody of evidence to a new custodian.
        
        Args:
            evidence_id: ID of evidence being transferred
            new_custodian: Name of new custodian
            reason: Reason for transfer
            location: New location (optional)
            
        Returns:
            Transfer ID
        """
        if evidence_id not in self.evidence_registry:
            raise ValueError(f"Evidence {evidence_id} not found in registry")
        
        transfer_id = str(uuid.uuid4())
        current_custodian = self.evidence_registry[evidence_id]['current_custodian']
        
        # Create transfer record
        transfer_record = {
            'transfer_id': transfer_id,
            'timestamp': datetime.now().isoformat(),
            'evidence_id': evidence_id,
            'from_custodian': current_custodian,
            'to_custodian': new_custodian,
            'reason': reason,
            'location': location or 'Not specified',
            'status': 'COMPLETED'
        }
        
        self.transfer_log.append(transfer_record)
        
        # Update evidence registry
        self.evidence_registry[evidence_id]['current_custodian'] = new_custodian
        if location:
            self.evidence_registry[evidence_id]['current_location'] = location
        
        # Add custody log entry
        custody_entry = {
            'entry_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'action': 'CUSTODY_TRANSFER',
            'officer': new_custodian,
            'details': f'Custody transferred from {current_custodian} to {new_custodian}. Reason: {reason}',
            'evidence_affected': [evidence_id],
            'location': location or 'Not specified',
            'transfer_id': transfer_id
        }
        
        custody_entry['signature_hash'] = self._generate_entry_signature(custody_entry)
        self.custody_log.append(custody_entry)
        
        self.logger.info(f"Custody of evidence {evidence_id} transferred to {new_custodian}")
        
        return transfer_id
    
    def log_access(self, evidence_id: str, accessor: str, purpose: str, 
                   access_type: str = 'EXAMINATION'):
        """Log access to evidence.
        
        Args:
            evidence_id: ID of evidence being accessed
            accessor: Name of person accessing evidence
            purpose: Purpose of access
            access_type: Type of access (EXAMINATION, ANALYSIS, etc.)
        """
        if evidence_id not in self.evidence_registry:
            raise ValueError(f"Evidence {evidence_id} not found in registry")
        
        access_record = {
            'access_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'accessor': accessor,
            'purpose': purpose,
            'access_type': access_type
        }
        
        # Add to evidence access log
        self.evidence_registry[evidence_id]['access_log'].append(access_record)
        
        # Add custody log entry
        custody_entry = {
            'entry_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'action': 'EVIDENCE_ACCESS',
            'officer': accessor,
            'details': f'Evidence accessed for {purpose} ({access_type})',
            'evidence_affected': [evidence_id],
            'location': 'Evidence Storage',
            'access_id': access_record['access_id']
        }
        
        custody_entry['signature_hash'] = self._generate_entry_signature(custody_entry)
        self.custody_log.append(custody_entry)
        
        self.logger.info(f"Access logged for evidence {evidence_id} by {accessor}")
    
    def verify_chain_integrity(self) -> Dict[str, Any]:
        """Verify the integrity of the chain of custody.
        
        Returns:
            Dictionary containing verification results
        """
        verification_results = {
            'timestamp': datetime.now().isoformat(),
            'case_id': self.case_id,
            'total_entries': len(self.custody_log),
            'signature_verification': {'passed': 0, 'failed': 0, 'details': []},
            'continuity_check': {'passed': True, 'gaps': []},
            'evidence_tracking': {'total_evidence': len(self.evidence_registry), 'issues': []},
            'overall_status': 'VERIFIED'
        }
        
        # Verify signatures
        for entry in self.custody_log:
            expected_signature = self._generate_entry_signature(entry)
            if entry.get('signature_hash') == expected_signature:
                verification_results['signature_verification']['passed'] += 1
            else:
                verification_results['signature_verification']['failed'] += 1
                verification_results['signature_verification']['details'].append({
                    'entry_id': entry.get('entry_id'),
                    'timestamp': entry.get('timestamp'),
                    'issue': 'Signature mismatch'
                })
        
        # Check for timeline continuity
        sorted_entries = sorted(self.custody_log, key=lambda x: x['timestamp'])
        for i in range(1, len(sorted_entries)):
            prev_time = datetime.fromisoformat(sorted_entries[i-1]['timestamp'])
            curr_time = datetime.fromisoformat(sorted_entries[i]['timestamp'])
            
            # Check for unreasonable time gaps (more than 24 hours)
            if (curr_time - prev_time).total_seconds() > 86400:
                verification_results['continuity_check']['gaps'].append({
                    'start': sorted_entries[i-1]['timestamp'],
                    'end': sorted_entries[i]['timestamp'],
                    'duration_hours': (curr_time - prev_time).total_seconds() / 3600
                })
        
        if verification_results['continuity_check']['gaps']:
            verification_results['continuity_check']['passed'] = False
        
        # Determine overall status
        if (verification_results['signature_verification']['failed'] > 0 or 
            not verification_results['continuity_check']['passed']):
            verification_results['overall_status'] = 'ISSUES_FOUND'
        
        self.logger.info(f"Chain integrity verification completed: {verification_results['overall_status']}")
        
        return verification_results
    
    def generate_custody_report(self, output_path: str = None) -> str:
        """Generate a comprehensive chain of custody report.
        
        Args:
            output_path: Optional path to save report
            
        Returns:
            Path to generated report
        """
        report_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'case_id': self.case_id,
                'primary_investigator': self.primary_investigator,
                'report_type': 'CHAIN_OF_CUSTODY'
            },
            'case_summary': {
                'total_custody_entries': len(self.custody_log),
                'total_evidence_items': len(self.evidence_registry),
                'total_transfers': len(self.transfer_log),
                'case_duration': self._calculate_case_duration()
            },
            'evidence_registry': self.evidence_registry,
            'custody_log': self.custody_log,
            'transfer_log': self.transfer_log,
            'integrity_verification': self.verify_chain_integrity()
        }
        
        # Generate report filename if not provided
        if not output_path:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"custody_report_{self.case_id}_{timestamp}.json"
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"Custody report generated: {output_path}")
        
        return output_path
    
    def _calculate_case_duration(self) -> str:
        """Calculate the duration of the case so far."""
        if not self.custody_log:
            return "0 days"
        
        start_time = datetime.fromisoformat(self.custody_log[0]['timestamp'])
        current_time = datetime.now()
        duration = current_time - start_time
        
        return f"{duration.days} days, {duration.seconds // 3600} hours"
    
    def get_evidence_status(self, evidence_id: str) -> Dict[str, Any]:
        """Get current status of specific evidence.
        
        Args:
            evidence_id: Evidence identifier
            
        Returns:
            Dictionary containing evidence status
        """
        if evidence_id not in self.evidence_registry:
            raise ValueError(f"Evidence {evidence_id} not found")
        
        evidence = self.evidence_registry[evidence_id]
        
        # Get related custody entries
        related_entries = [
            entry for entry in self.custody_log 
            if evidence_id in entry.get('evidence_affected', [])
        ]
        
        return {
            'evidence_info': evidence,
            'custody_entries': related_entries,
            'transfer_history': [
                transfer for transfer in self.transfer_log 
                if transfer['evidence_id'] == evidence_id
            ],
            'access_count': len(evidence.get('access_log', [])),
            'last_access': evidence.get('access_log', [{}])[-1] if evidence.get('access_log') else None
        }
    
    def close_case(self, closing_officer: str, reason: str = "Investigation complete"):
        """Close the case and finalize chain of custody.
        
        Args:
            closing_officer: Officer closing the case
            reason: Reason for case closure
        """
        closing_entry = {
            'entry_id': str(uuid.uuid4()),
            'timestamp': datetime.now().isoformat(),
            'action': 'CASE_CLOSED',
            'officer': closing_officer,
            'details': f'Case {self.case_id} closed by {closing_officer}. Reason: {reason}',
            'evidence_affected': list(self.evidence_registry.keys()),
            'location': 'Investigation Unit',
            'final_evidence_count': len(self.evidence_registry)
        }
        
        closing_entry['signature_hash'] = self._generate_entry_signature(closing_entry)
        self.custody_log.append(closing_entry)
        
        # Generate final report
        final_report_path = self.generate_custody_report(
            f"final_custody_report_{self.case_id}.json"
        )
        
        self.logger.info(f"Case {self.case_id} closed. Final report: {final_report_path}")
        
        return final_report_path


def main():
    """Example usage of ChainOfCustody."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Chain of Custody Management')
    parser.add_argument('--case-id', required=True, help='Case ID')
    parser.add_argument('--investigator', required=True, help='Primary investigator')
    parser.add_argument('--action', choices=['status', 'report', 'verify'], 
                       default='status', help='Action to perform')
    
    args = parser.parse_args()
    
    # Initialize chain of custody
    custody = ChainOfCustody(args.case_id, args.investigator)
    
    if args.action == 'status':
        print(f"Case ID: {custody.case_id}")
        print(f"Primary Investigator: {custody.primary_investigator}")
        print(f"Total Custody Entries: {len(custody.custody_log)}")
        print(f"Total Evidence Items: {len(custody.evidence_registry)}")
        
    elif args.action == 'report':
        report_path = custody.generate_custody_report()
        print(f"Custody report generated: {report_path}")
        
    elif args.action == 'verify':
        verification = custody.verify_chain_integrity()
        print(f"Chain integrity status: {verification['overall_status']}")
        print(f"Signature verification: {verification['signature_verification']['passed']} passed, "
              f"{verification['signature_verification']['failed']} failed")


if __name__ == "__main__":
    main()