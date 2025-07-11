#!/usr/bin/env python3
"""
Evidence Validator for Forensic Integrity
==========================================

Validates the integrity of forensic evidence over time, detects tampering,
and maintains verification logs for legal admissibility.

Author: Forensic Collection Framework
Version: 1.0
"""

import os
import json
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import sqlite3
import threading
import time
import uuid


class EvidenceValidator:
    """Validates and monitors forensic evidence integrity."""
    
    def __init__(self, database_path: str = None):
        """Initialize the evidence validator.
        
        Args:
            database_path: Path to SQLite database for validation logs
        """
        self.database_path = database_path or "evidence_integrity.db"
        self.logger = logging.getLogger('EvidenceValidator')
        
        # Initialize database
        self._init_database()
        
        # Monitoring thread control
        self._monitoring_active = False
        self._monitoring_thread = None
        self._monitoring_interval = 300  # 5 minutes default
        
        # Validation results cache
        self._validation_cache = {}
        self._cache_lock = threading.Lock()
    
    def _init_database(self):
        """Initialize the SQLite database for integrity tracking."""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Create evidence registry table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS evidence_registry (
                    evidence_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    original_hash TEXT NOT NULL,
                    registration_time TIMESTAMP NOT NULL,
                    last_verified TIMESTAMP,
                    verification_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'ACTIVE',
                    metadata TEXT
                )
            ''')
            
            # Create validation log table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS validation_log (
                    validation_id TEXT PRIMARY KEY,
                    evidence_id TEXT NOT NULL,
                    validation_time TIMESTAMP NOT NULL,
                    current_hash TEXT NOT NULL,
                    integrity_status TEXT NOT NULL,
                    details TEXT,
                    validator TEXT,
                    FOREIGN KEY (evidence_id) REFERENCES evidence_registry (evidence_id)
                )
            ''')
            
            # Create alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS integrity_alerts (
                    alert_id TEXT PRIMARY KEY,
                    evidence_id TEXT NOT NULL,
                    alert_time TIMESTAMP NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    description TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (evidence_id) REFERENCES evidence_registry (evidence_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Database initialized: {self.database_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    def register_evidence(self, evidence_record: Dict[str, Any]) -> str:
        """Register evidence for integrity monitoring.
        
        Args:
            evidence_record: Evidence metadata record
            
        Returns:
            Registration ID
        """
        try:
            evidence_id = evidence_record['evidence_id']
            file_path = evidence_record['evidence_path']
            original_hash = evidence_record['original_hashes']['sha256']
            
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO evidence_registry 
                (evidence_id, file_path, original_hash, registration_time, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                evidence_id,
                file_path,
                original_hash,
                datetime.now().isoformat(),
                json.dumps(evidence_record)
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Evidence registered for monitoring: {evidence_id}")
            
            return evidence_id
            
        except Exception as e:
            self.logger.error(f"Failed to register evidence: {str(e)}")
            raise
    
    def validate_evidence(self, evidence_id: str, validator: str = "System") -> Dict[str, Any]:
        """Validate the integrity of a specific piece of evidence.
        
        Args:
            evidence_id: Evidence identifier
            validator: Name of person/system performing validation
            
        Returns:
            Validation results
        """
        validation_id = str(uuid.uuid4())
        validation_time = datetime.now()
        
        try:
            # Get evidence information from database
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT file_path, original_hash, metadata 
                FROM evidence_registry 
                WHERE evidence_id = ? AND status = 'ACTIVE'
            ''', (evidence_id,))
            
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"Evidence {evidence_id} not found or inactive")
            
            file_path, original_hash, metadata_json = result
            
            # Calculate current hash
            current_hash = self._calculate_file_hash(file_path)
            
            # Determine integrity status
            integrity_status = "VERIFIED" if current_hash == original_hash else "COMPROMISED"
            
            # Prepare validation results
            validation_results = {
                'validation_id': validation_id,
                'evidence_id': evidence_id,
                'validation_time': validation_time.isoformat(),
                'file_path': file_path,
                'original_hash': original_hash,
                'current_hash': current_hash,
                'integrity_status': integrity_status,
                'validator': validator,
                'file_exists': os.path.exists(file_path),
                'file_accessible': os.access(file_path, os.R_OK) if os.path.exists(file_path) else False,
                'details': {}
            }
            
            # Add detailed analysis
            if os.path.exists(file_path):
                file_stat = os.stat(file_path)
                validation_results['details'] = {
                    'file_size': file_stat.st_size,
                    'last_modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                    'permissions': oct(file_stat.st_mode)[-3:]
                }
                
                # Check for suspicious modifications
                if integrity_status == "COMPROMISED":
                    validation_results['details']['tampering_indicators'] = self._analyze_tampering(
                        file_path, json.loads(metadata_json)
                    )
            else:
                validation_results['details']['error'] = "File not found"
                integrity_status = "MISSING"
                validation_results['integrity_status'] = integrity_status
            
            # Log validation
            cursor.execute('''
                INSERT INTO validation_log 
                (validation_id, evidence_id, validation_time, current_hash, 
                 integrity_status, details, validator)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                validation_id,
                evidence_id,
                validation_time.isoformat(),
                current_hash or "N/A",
                integrity_status,
                json.dumps(validation_results['details']),
                validator
            ))
            
            # Update evidence registry
            cursor.execute('''
                UPDATE evidence_registry 
                SET last_verified = ?, verification_count = verification_count + 1
                WHERE evidence_id = ?
            ''', (validation_time.isoformat(), evidence_id))
            
            conn.commit()
            
            # Generate alert if integrity is compromised
            if integrity_status in ["COMPROMISED", "MISSING"]:
                self._generate_integrity_alert(evidence_id, integrity_status, validation_results)
            
            conn.close()
            
            # Update cache
            with self._cache_lock:
                self._validation_cache[evidence_id] = validation_results
            
            self.logger.info(f"Evidence validation completed: {evidence_id} - {integrity_status}")
            
            return validation_results
            
        except Exception as e:
            error_msg = f"Validation failed for evidence {evidence_id}: {str(e)}"
            self.logger.error(error_msg)
            
            # Log failed validation
            try:
                conn = sqlite3.connect(self.database_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO validation_log 
                    (validation_id, evidence_id, validation_time, current_hash, 
                     integrity_status, details, validator)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    validation_id,
                    evidence_id,
                    validation_time.isoformat(),
                    "ERROR",
                    "VALIDATION_FAILED",
                    json.dumps({'error': error_msg}),
                    validator
                ))
                conn.commit()
                conn.close()
            except:
                pass
            
            raise Exception(error_msg)
    
    def validate_all_evidence(self, validator: str = "System") -> Dict[str, Any]:
        """Validate all registered evidence.
        
        Args:
            validator: Name of person/system performing validation
            
        Returns:
            Summary of all validation results
        """
        validation_summary = {
            'validation_time': datetime.now().isoformat(),
            'validator': validator,
            'total_evidence': 0,
            'verified': 0,
            'compromised': 0,
            'missing': 0,
            'errors': 0,
            'results': []
        }
        
        try:
            # Get all active evidence
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT evidence_id FROM evidence_registry 
                WHERE status = 'ACTIVE'
            ''')
            
            evidence_ids = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            validation_summary['total_evidence'] = len(evidence_ids)
            
            # Validate each piece of evidence
            for evidence_id in evidence_ids:
                try:
                    result = self.validate_evidence(evidence_id, validator)
                    validation_summary['results'].append(result)
                    
                    # Update counters
                    status = result['integrity_status']
                    if status == "VERIFIED":
                        validation_summary['verified'] += 1
                    elif status == "COMPROMISED":
                        validation_summary['compromised'] += 1
                    elif status == "MISSING":
                        validation_summary['missing'] += 1
                    else:
                        validation_summary['errors'] += 1
                        
                except Exception as e:
                    validation_summary['errors'] += 1
                    validation_summary['results'].append({
                        'evidence_id': evidence_id,
                        'error': str(e)
                    })
            
            self.logger.info(f"Bulk validation completed: {validation_summary['verified']} verified, "
                           f"{validation_summary['compromised']} compromised, "
                           f"{validation_summary['missing']} missing, "
                           f"{validation_summary['errors']} errors")
            
            return validation_summary
            
        except Exception as e:
            self.logger.error(f"Bulk validation failed: {str(e)}")
            raise
    
    def start_monitoring(self, interval_seconds: int = 300):
        """Start continuous monitoring of evidence integrity.
        
        Args:
            interval_seconds: Monitoring interval in seconds
        """
        if self._monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self._monitoring_interval = interval_seconds
        self._monitoring_active = True
        
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        
        self.logger.info(f"Evidence monitoring started (interval: {interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self._monitoring_active = False
        
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=10)
        
        self.logger.info("Evidence monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                self.validate_all_evidence("Automated Monitor")
                
                # Sleep for the specified interval
                for _ in range(self._monitoring_interval):
                    if not self._monitoring_active:
                        break
                    time.sleep(1)
                    
            except Exception as e:
                self.logger.error(f"Monitoring error: {str(e)}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def _calculate_file_hash(self, file_path: str) -> Optional[str]:
        """Calculate SHA256 hash of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            SHA256 hash or None if file doesn't exist
        """
        try:
            if not os.path.exists(file_path):
                return None
            
            sha256_hash = hashlib.sha256()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)
            
            return sha256_hash.hexdigest()
            
        except Exception as e:
            self.logger.error(f"Error calculating hash for {file_path}: {str(e)}")
            return None
    
    def _analyze_tampering(self, file_path: str, original_metadata: Dict[str, Any]) -> List[str]:
        """Analyze potential tampering indicators.
        
        Args:
            file_path: Path to file
            original_metadata: Original metadata from collection
            
        Returns:
            List of tampering indicators
        """
        indicators = []
        
        try:
            current_stat = os.stat(file_path)
            original_timestamps = original_metadata.get('timestamps', {})
            
            # Check for timestamp anomalies
            if 'modified' in original_timestamps:
                original_mtime = datetime.fromisoformat(original_timestamps['modified'])
                current_mtime = datetime.fromtimestamp(current_stat.st_mtime)
                
                if current_mtime > original_mtime:
                    indicators.append("File modification time changed")
            
            # Check for size changes
            original_size = original_metadata.get('file_size', 0)
            if current_stat.st_size != original_size:
                indicators.append(f"File size changed from {original_size} to {current_stat.st_size}")
            
            # Check for permission changes
            original_permissions = original_metadata.get('timestamps', {}).get('permissions')
            current_permissions = oct(current_stat.st_mode)[-3:]
            
            if original_permissions and original_permissions != current_permissions:
                indicators.append(f"File permissions changed from {original_permissions} to {current_permissions}")
            
        except Exception as e:
            indicators.append(f"Error analyzing tampering: {str(e)}")
        
        return indicators
    
    def _generate_integrity_alert(self, evidence_id: str, alert_type: str, validation_results: Dict[str, Any]):
        """Generate an integrity alert.
        
        Args:
            evidence_id: Evidence identifier
            alert_type: Type of alert (COMPROMISED, MISSING, etc.)
            validation_results: Validation results
        """
        try:
            alert_id = str(uuid.uuid4())
            alert_time = datetime.now()
            
            severity_map = {
                'COMPROMISED': 'CRITICAL',
                'MISSING': 'HIGH',
                'VALIDATION_FAILED': 'MEDIUM'
            }
            
            severity = severity_map.get(alert_type, 'LOW')
            
            description = f"Evidence integrity alert: {alert_type}"
            if alert_type == "COMPROMISED":
                indicators = validation_results.get('details', {}).get('tampering_indicators', [])
                if indicators:
                    description += f" - Indicators: {', '.join(indicators)}"
            
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO integrity_alerts 
                (alert_id, evidence_id, alert_time, alert_type, severity, description)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                alert_id,
                evidence_id,
                alert_time.isoformat(),
                alert_type,
                severity,
                description
            ))
            
            conn.commit()
            conn.close()
            
            self.logger.warning(f"Integrity alert generated: {alert_id} - {description}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate alert: {str(e)}")
    
    def get_integrity_report(self, evidence_id: str = None) -> Dict[str, Any]:
        """Generate integrity report for specific evidence or all evidence.
        
        Args:
            evidence_id: Optional evidence ID for specific report
            
        Returns:
            Integrity report
        """
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            report = {
                'report_time': datetime.now().isoformat(),
                'report_type': 'INTEGRITY_REPORT',
                'evidence_summary': {},
                'validation_history': [],
                'alerts': [],
                'recommendations': []
            }
            
            # Get evidence summary
            if evidence_id:
                cursor.execute('''
                    SELECT * FROM evidence_registry WHERE evidence_id = ?
                ''', (evidence_id,))
                evidence_filter = "WHERE evidence_id = ?"
                filter_params = (evidence_id,)
            else:
                cursor.execute('''
                    SELECT * FROM evidence_registry WHERE status = 'ACTIVE'
                ''')
                evidence_filter = "WHERE er.status = 'ACTIVE'"
                filter_params = ()
            
            evidence_records = cursor.fetchall()
            
            report['evidence_summary'] = {
                'total_evidence': len(evidence_records),
                'evidence_details': []
            }
            
            for record in evidence_records:
                evidence_detail = {
                    'evidence_id': record[0],
                    'file_path': record[1],
                    'registration_time': record[3],
                    'last_verified': record[4],
                    'verification_count': record[5],
                    'status': record[6]
                }
                report['evidence_summary']['evidence_details'].append(evidence_detail)
            
            # Get validation history
            if evidence_id:
                cursor.execute('''
                    SELECT * FROM validation_log 
                    WHERE evidence_id = ? 
                    ORDER BY validation_time DESC 
                    LIMIT 50
                ''', (evidence_id,))
            else:
                cursor.execute('''
                    SELECT * FROM validation_log 
                    ORDER BY validation_time DESC 
                    LIMIT 100
                ''')
            
            validation_records = cursor.fetchall()
            
            for record in validation_records:
                validation_detail = {
                    'validation_id': record[0],
                    'evidence_id': record[1],
                    'validation_time': record[2],
                    'current_hash': record[3],
                    'integrity_status': record[4],
                    'validator': record[6]
                }
                report['validation_history'].append(validation_detail)
            
            # Get alerts
            if evidence_id:
                cursor.execute('''
                    SELECT * FROM integrity_alerts 
                    WHERE evidence_id = ? 
                    ORDER BY alert_time DESC
                ''', (evidence_id,))
            else:
                cursor.execute('''
                    SELECT * FROM integrity_alerts 
                    ORDER BY alert_time DESC 
                    LIMIT 50
                ''')
            
            alert_records = cursor.fetchall()
            
            for record in alert_records:
                alert_detail = {
                    'alert_id': record[0],
                    'evidence_id': record[1],
                    'alert_time': record[2],
                    'alert_type': record[3],
                    'severity': record[4],
                    'description': record[5],
                    'resolved': bool(record[6])
                }
                report['alerts'].append(alert_detail)
            
            # Generate recommendations
            report['recommendations'] = self._generate_recommendations(report)
            
            conn.close()
            
            return report
            
        except Exception as e:
            self.logger.error(f"Failed to generate integrity report: {str(e)}")
            raise
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on integrity report.
        
        Args:
            report: Integrity report data
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Check for unresolved alerts
        unresolved_alerts = [alert for alert in report['alerts'] if not alert['resolved']]
        if unresolved_alerts:
            critical_alerts = [alert for alert in unresolved_alerts if alert['severity'] == 'CRITICAL']
            if critical_alerts:
                recommendations.append("URGENT: Address critical integrity alerts immediately")
            
            recommendations.append(f"Investigate and resolve {len(unresolved_alerts)} unresolved alerts")
        
        # Check for evidence not verified recently
        cutoff_time = datetime.now() - timedelta(days=7)
        evidence_details = report['evidence_summary']['evidence_details']
        
        stale_evidence = [
            ev for ev in evidence_details 
            if not ev['last_verified'] or 
            datetime.fromisoformat(ev['last_verified']) < cutoff_time
        ]
        
        if stale_evidence:
            recommendations.append(f"Verify {len(stale_evidence)} evidence items not checked in past 7 days")
        
        # Check validation frequency
        if not report['validation_history']:
            recommendations.append("Establish regular validation schedule")
        
        return recommendations
    
    def export_validation_data(self, output_path: str, evidence_id: str = None) -> str:
        """Export validation data to JSON file.
        
        Args:
            output_path: Path to output file
            evidence_id: Optional evidence ID for specific export
            
        Returns:
            Path to exported file
        """
        try:
            report = self.get_integrity_report(evidence_id)
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Validation data exported to {output_path}")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to export validation data: {str(e)}")
            raise


def main():
    """Example usage of EvidenceValidator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evidence Integrity Validator')
    parser.add_argument('--action', choices=['validate', 'monitor', 'report'], 
                       default='validate', help='Action to perform')
    parser.add_argument('--evidence-id', help='Evidence ID for specific validation')
    parser.add_argument('--database', help='Database path')
    parser.add_argument('--output', help='Output file for reports')
    parser.add_argument('--interval', type=int, default=300, 
                       help='Monitoring interval in seconds')
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = EvidenceValidator(args.database)
    
    if args.action == 'validate':
        if args.evidence_id:
            result = validator.validate_evidence(args.evidence_id)
            print(f"Validation result: {result['integrity_status']}")
        else:
            summary = validator.validate_all_evidence()
            print(f"Validation summary: {summary['verified']} verified, "
                  f"{summary['compromised']} compromised, "
                  f"{summary['missing']} missing")
    
    elif args.action == 'monitor':
        print(f"Starting monitoring (interval: {args.interval}s)")
        print("Press Ctrl+C to stop...")
        
        validator.start_monitoring(args.interval)
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            validator.stop_monitoring()
            print("Monitoring stopped")
    
    elif args.action == 'report':
        report = validator.get_integrity_report(args.evidence_id)
        
        if args.output:
            validator.export_validation_data(args.output, args.evidence_id)
            print(f"Report saved to {args.output}")
        else:
            print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()