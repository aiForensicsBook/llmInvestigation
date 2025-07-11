"""
Forensic Collection Framework
============================

A comprehensive digital forensics collection and validation framework
for maintaining chain of custody and evidence integrity.

Author: Forensic Collection Framework
Version: 1.0
"""

from .forensic_collector import ForensicCollector
from .chain_of_custody import ChainOfCustody
from .metadata_extractor import MetadataExtractor
from .evidence_validator import EvidenceValidator

__version__ = "1.0.0"
__author__ = "Forensic Collection Framework"

__all__ = [
    'ForensicCollector',
    'ChainOfCustody', 
    'MetadataExtractor',
    'EvidenceValidator'
]