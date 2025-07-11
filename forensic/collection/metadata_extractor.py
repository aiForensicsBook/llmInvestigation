#!/usr/bin/env python3
"""
Metadata Extractor for Forensic Analysis
========================================

Comprehensive file metadata extraction and analysis for forensic investigations.
Extracts file system metadata, analyzes file headers, and creates forensic timelines.

Author: Forensic Collection Framework
Version: 1.0
"""

import os
import json
import logging
import magic
import mimetypes
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import hashlib
import struct
import zipfile
import tarfile
try:
    from PIL import Image
    from PIL.ExifTags import TAGS
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import pymupdf as fitz  # PyMuPDF for PDF analysis
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


class MetadataExtractor:
    """Comprehensive metadata extraction for forensic analysis."""
    
    def __init__(self):
        """Initialize the metadata extractor."""
        self.logger = logging.getLogger('MetadataExtractor')
        
        # Initialize magic for file type detection
        try:
            self.magic_mime = magic.Magic(mime=True)
            self.magic_desc = magic.Magic()
        except Exception as e:
            self.logger.warning(f"Python-magic not available: {str(e)}")
            self.magic_mime = None
            self.magic_desc = None
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract comprehensive metadata from a file.
        
        Args:
            file_path: Path to the file to analyze
            
        Returns:
            Dictionary containing all extracted metadata
        """
        metadata = {
            'file_path': file_path,
            'extraction_timestamp': datetime.now().isoformat(),
            'basic_info': {},
            'file_system': {},
            'file_type': {},
            'header_analysis': {},
            'content_analysis': {},
            'relationships': {},
            'security_attributes': {},
            'errors': []
        }
        
        try:
            file_path_obj = Path(file_path)
            
            if not file_path_obj.exists():
                metadata['errors'].append(f"File does not exist: {file_path}")
                return metadata
            
            # Extract basic file information
            metadata['basic_info'] = self._extract_basic_info(file_path_obj)
            
            # Extract file system metadata
            metadata['file_system'] = self._extract_filesystem_metadata(file_path_obj)
            
            # Determine file type and MIME type
            metadata['file_type'] = self._determine_file_type(file_path_obj)
            
            # Analyze file headers
            metadata['header_analysis'] = self._analyze_file_headers(file_path_obj)
            
            # Extract content-specific metadata
            metadata['content_analysis'] = self._extract_content_metadata(file_path_obj)
            
            # Analyze file relationships
            metadata['relationships'] = self._analyze_file_relationships(file_path_obj)
            
            # Extract security attributes
            metadata['security_attributes'] = self._extract_security_attributes(file_path_obj)
            
            self.logger.debug(f"Metadata extraction completed for {file_path}")
            
        except Exception as e:
            error_msg = f"Error extracting metadata from {file_path}: {str(e)}"
            self.logger.error(error_msg)
            metadata['errors'].append(error_msg)
        
        return metadata
    
    def _extract_basic_info(self, file_path: Path) -> Dict[str, Any]:
        """Extract basic file information."""
        try:
            stat_info = file_path.stat()
            
            return {
                'filename': file_path.name,
                'basename': file_path.stem,
                'extension': file_path.suffix.lower(),
                'full_path': str(file_path.resolve()),
                'parent_directory': str(file_path.parent),
                'size_bytes': stat_info.st_size,
                'size_human': self._format_file_size(stat_info.st_size),
                'is_hidden': file_path.name.startswith('.'),
                'is_executable': os.access(file_path, os.X_OK),
                'is_readable': os.access(file_path, os.R_OK),
                'is_writable': os.access(file_path, os.W_OK)
            }
        except Exception as e:
            self.logger.error(f"Error extracting basic info: {str(e)}")
            return {'error': str(e)}
    
    def _extract_filesystem_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract file system level metadata."""
        try:
            stat_info = file_path.stat()
            
            filesystem_data = {
                'inode': stat_info.st_ino,
                'device': stat_info.st_dev,
                'hard_link_count': stat_info.st_nlink,
                'uid': stat_info.st_uid,
                'gid': stat_info.st_gid,
                'mode': oct(stat_info.st_mode),
                'permissions': {
                    'owner': {
                        'read': bool(stat_info.st_mode & 0o400),
                        'write': bool(stat_info.st_mode & 0o200),
                        'execute': bool(stat_info.st_mode & 0o100)
                    },
                    'group': {
                        'read': bool(stat_info.st_mode & 0o040),
                        'write': bool(stat_info.st_mode & 0o020),
                        'execute': bool(stat_info.st_mode & 0o010)
                    },
                    'other': {
                        'read': bool(stat_info.st_mode & 0o004),
                        'write': bool(stat_info.st_mode & 0o002),
                        'execute': bool(stat_info.st_mode & 0o001)
                    }
                },
                'timestamps': {
                    'modified': datetime.fromtimestamp(stat_info.st_mtime).isoformat(),
                    'accessed': datetime.fromtimestamp(stat_info.st_atime).isoformat(),
                    'changed': datetime.fromtimestamp(stat_info.st_ctime).isoformat()
                }
            }
            
            # Try to get creation time if available
            if hasattr(stat_info, 'st_birthtime'):
                filesystem_data['timestamps']['created'] = datetime.fromtimestamp(stat_info.st_birthtime).isoformat()
            
            return filesystem_data
            
        except Exception as e:
            self.logger.error(f"Error extracting filesystem metadata: {str(e)}")
            return {'error': str(e)}
    
    def _determine_file_type(self, file_path: Path) -> Dict[str, Any]:
        """Determine file type using multiple methods."""
        file_type_data = {
            'extension_based': None,
            'mime_type': None,
            'magic_description': None,
            'confidence': 'unknown'
        }
        
        try:
            # Extension-based detection
            mime_type, _ = mimetypes.guess_type(str(file_path))
            file_type_data['extension_based'] = mime_type
            
            # Magic number detection
            if self.magic_mime and self.magic_desc:
                try:
                    file_type_data['mime_type'] = self.magic_mime.from_file(str(file_path))
                    file_type_data['magic_description'] = self.magic_desc.from_file(str(file_path))
                except Exception as e:
                    self.logger.warning(f"Magic detection failed: {str(e)}")
            
            # Determine confidence level
            if (file_type_data['extension_based'] and 
                file_type_data['mime_type'] and 
                file_type_data['extension_based'] == file_type_data['mime_type']):
                file_type_data['confidence'] = 'high'
            elif file_type_data['mime_type']:
                file_type_data['confidence'] = 'medium'
            elif file_type_data['extension_based']:
                file_type_data['confidence'] = 'low'
            
        except Exception as e:
            self.logger.error(f"Error determining file type: {str(e)}")
            file_type_data['error'] = str(e)
        
        return file_type_data
    
    def _analyze_file_headers(self, file_path: Path) -> Dict[str, Any]:
        """Analyze file headers for forensic signatures."""
        header_data = {
            'first_bytes': None,
            'file_signature': None,
            'header_analysis': {},
            'anomalies': []
        }
        
        try:
            with open(file_path, 'rb') as f:
                # Read first 512 bytes for header analysis
                header_bytes = f.read(512)
                
                if header_bytes:
                    header_data['first_bytes'] = header_bytes[:32].hex()
                    header_data['file_signature'] = self._identify_file_signature(header_bytes)
                    
                    # Analyze specific file types
                    if header_bytes.startswith(b'\x50\x4B'):  # ZIP/Office files
                        header_data['header_analysis'] = self._analyze_zip_header(file_path)
                    elif header_bytes.startswith(b'%PDF'):  # PDF files
                        header_data['header_analysis'] = self._analyze_pdf_header(file_path)
                    elif header_bytes.startswith((b'\xFF\xD8\xFF', b'\x89PNG')):  # Image files
                        header_data['header_analysis'] = self._analyze_image_header(file_path)
                    elif header_bytes.startswith(b'MZ'):  # PE executables
                        header_data['header_analysis'] = self._analyze_pe_header(header_bytes)
                    
        except Exception as e:
            self.logger.error(f"Error analyzing file headers: {str(e)}")
            header_data['error'] = str(e)
        
        return header_data
    
    def _extract_content_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract content-specific metadata based on file type."""
        content_data = {}
        
        try:
            file_ext = file_path.suffix.lower()
            
            # Image files
            if file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp'] and PIL_AVAILABLE:
                content_data['image_metadata'] = self._extract_image_metadata(file_path)
            
            # PDF files
            elif file_ext == '.pdf' and PYMUPDF_AVAILABLE:
                content_data['pdf_metadata'] = self._extract_pdf_metadata(file_path)
            
            # Archive files
            elif file_ext in ['.zip', '.jar', '.war']:
                content_data['archive_metadata'] = self._extract_zip_metadata(file_path)
            
            elif file_ext in ['.tar', '.tar.gz', '.tgz']:
                content_data['archive_metadata'] = self._extract_tar_metadata(file_path)
            
            # Text files
            elif file_ext in ['.txt', '.log', '.conf', '.cfg', '.ini']:
                content_data['text_metadata'] = self._extract_text_metadata(file_path)
            
            # Executable files
            elif file_ext in ['.exe', '.dll', '.so']:
                content_data['executable_metadata'] = self._extract_executable_metadata(file_path)
            
        except Exception as e:
            self.logger.error(f"Error extracting content metadata: {str(e)}")
            content_data['error'] = str(e)
        
        return content_data
    
    def _analyze_file_relationships(self, file_path: Path) -> Dict[str, Any]:
        """Analyze file relationships and dependencies."""
        relationships = {
            'parent_directory': str(file_path.parent),
            'sibling_files': [],
            'potential_dependencies': [],
            'related_extensions': [],
            'timestamp_correlation': {}
        }
        
        try:
            # Get sibling files
            if file_path.parent.exists():
                siblings = [f.name for f in file_path.parent.iterdir() 
                          if f.is_file() and f != file_path]
                relationships['sibling_files'] = siblings[:20]  # Limit to first 20
                
                # Find files with similar names
                base_name = file_path.stem
                related = [f for f in siblings if base_name in f]
                relationships['potential_dependencies'] = related
                
                # Find files with related extensions
                related_exts = [f for f in siblings 
                              if f.split('.')[-1] in ['log', 'tmp', 'bak', 'old']]
                relationships['related_extensions'] = related_exts
            
            # Analyze timestamp correlations with nearby files
            relationships['timestamp_correlation'] = self._analyze_timestamp_correlation(file_path)
            
        except Exception as e:
            self.logger.error(f"Error analyzing file relationships: {str(e)}")
            relationships['error'] = str(e)
        
        return relationships
    
    def _extract_security_attributes(self, file_path: Path) -> Dict[str, Any]:
        """Extract security-related attributes."""
        security_data = {
            'potential_risks': [],
            'suspicious_indicators': [],
            'file_entropy': None,
            'string_analysis': {}
        }
        
        try:
            # Calculate file entropy (indicator of encryption/compression)
            security_data['file_entropy'] = self._calculate_file_entropy(file_path)
            
            # Check for suspicious file characteristics
            if file_path.stat().st_size == 0:
                security_data['suspicious_indicators'].append('Zero-byte file')
            
            if file_path.name.startswith('.'):
                security_data['suspicious_indicators'].append('Hidden file')
            
            # Check for double extensions
            if file_path.name.count('.') > 1:
                security_data['suspicious_indicators'].append('Multiple extensions')
            
            # Basic string analysis for executables
            if file_path.suffix.lower() in ['.exe', '.dll', '.so', '.bin']:
                security_data['string_analysis'] = self._extract_strings(file_path)
            
        except Exception as e:
            self.logger.error(f"Error extracting security attributes: {str(e)}")
            security_data['error'] = str(e)
        
        return security_data
    
    def _identify_file_signature(self, header_bytes: bytes) -> str:
        """Identify file type by signature."""
        signatures = {
            b'\x50\x4B\x03\x04': 'ZIP',
            b'\x50\x4B\x05\x06': 'ZIP (empty)',
            b'\x50\x4B\x07\x08': 'ZIP (spanned)',
            b'%PDF': 'PDF',
            b'\xFF\xD8\xFF': 'JPEG',
            b'\x89PNG\r\n\x1A\n': 'PNG',
            b'GIF87a': 'GIF87a',
            b'GIF89a': 'GIF89a',
            b'BM': 'BMP',
            b'MZ': 'PE Executable',
            b'\x7FELF': 'ELF',
            b'\xCA\xFE\xBA\xBE': 'Java Class',
            b'PK': 'ZIP/JAR/APK'
        }
        
        for sig, file_type in signatures.items():
            if header_bytes.startswith(sig):
                return file_type
        
        return 'Unknown'
    
    def _extract_image_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract EXIF and other metadata from images."""
        if not PIL_AVAILABLE:
            return {'error': 'PIL not available'}
        
        try:
            with Image.open(file_path) as img:
                metadata = {
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'has_transparency': 'transparency' in img.info,
                    'exif': {}
                }
                
                # Extract EXIF data
                exif_data = img.getexif()
                if exif_data:
                    for tag_id, value in exif_data.items():
                        tag = TAGS.get(tag_id, tag_id)
                        metadata['exif'][tag] = str(value)
                
                return metadata
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_pdf_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF files."""
        if not PYMUPDF_AVAILABLE:
            return {'error': 'PyMuPDF not available'}
        
        try:
            doc = fitz.open(file_path)
            metadata = {
                'page_count': doc.page_count,
                'metadata': doc.metadata,
                'is_encrypted': doc.is_encrypted,
                'needs_pass': doc.needs_pass,
                'permissions': doc.permissions
            }
            doc.close()
            return metadata
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_zip_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from ZIP archives."""
        try:
            with zipfile.ZipFile(file_path, 'r') as zf:
                metadata = {
                    'file_count': len(zf.filelist),
                    'files': [],
                    'compression_info': {},
                    'total_compressed_size': 0,
                    'total_uncompressed_size': 0
                }
                
                for info in zf.filelist[:50]:  # Limit to first 50 files
                    file_info = {
                        'filename': info.filename,
                        'file_size': info.file_size,
                        'compress_size': info.compress_size,
                        'date_time': info.date_time,
                        'compress_type': info.compress_type
                    }
                    metadata['files'].append(file_info)
                    metadata['total_compressed_size'] += info.compress_size
                    metadata['total_uncompressed_size'] += info.file_size
                
                return metadata
        except Exception as e:
            return {'error': str(e)}
    
    def _calculate_file_entropy(self, file_path: Path) -> float:
        """Calculate file entropy (measure of randomness)."""
        try:
            with open(file_path, 'rb') as f:
                data = f.read(min(8192, file_path.stat().st_size))  # Sample first 8KB
                
            if not data:
                return 0.0
            
            # Calculate byte frequency
            byte_counts = [0] * 256
            for byte in data:
                byte_counts[byte] += 1
            
            # Calculate entropy
            entropy = 0.0
            data_len = len(data)
            
            for count in byte_counts:
                if count > 0:
                    freq = count / data_len
                    entropy -= freq * (freq.bit_length() - 1)
            
            return entropy
            
        except Exception as e:
            self.logger.error(f"Error calculating entropy: {str(e)}")
            return 0.0
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def _analyze_timestamp_correlation(self, file_path: Path) -> Dict[str, Any]:
        """Analyze timestamp correlations with nearby files."""
        correlations = {
            'similar_timestamps': [],
            'suspicious_patterns': []
        }
        
        try:
            file_stat = file_path.stat()
            file_mtime = file_stat.st_mtime
            
            # Check sibling files for similar timestamps
            for sibling in file_path.parent.iterdir():
                if sibling.is_file() and sibling != file_path:
                    sibling_mtime = sibling.stat().st_mtime
                    time_diff = abs(file_mtime - sibling_mtime)
                    
                    # Files modified within 1 minute of each other
                    if time_diff < 60:
                        correlations['similar_timestamps'].append({
                            'file': sibling.name,
                            'time_difference_seconds': time_diff
                        })
            
            # Check for suspicious patterns (e.g., round timestamps)
            if file_mtime % 3600 == 0:  # Exactly on the hour
                correlations['suspicious_patterns'].append('Timestamp exactly on hour boundary')
            
        except Exception as e:
            correlations['error'] = str(e)
        
        return correlations
    
    def _extract_strings(self, file_path: Path, min_length: int = 4) -> Dict[str, Any]:
        """Extract printable strings from binary files."""
        strings_data = {
            'total_strings': 0,
            'suspicious_strings': [],
            'interesting_patterns': []
        }
        
        try:
            with open(file_path, 'rb') as f:
                data = f.read(min(32768, file_path.stat().st_size))  # First 32KB
            
            current_string = b''
            strings = []
            
            for byte in data:
                if 32 <= byte <= 126:  # Printable ASCII
                    current_string += bytes([byte])
                else:
                    if len(current_string) >= min_length:
                        strings.append(current_string.decode('ascii', errors='ignore'))
                    current_string = b''
            
            # Add final string if valid
            if len(current_string) >= min_length:
                strings.append(current_string.decode('ascii', errors='ignore'))
            
            strings_data['total_strings'] = len(strings)
            
            # Look for suspicious patterns
            suspicious_keywords = ['password', 'admin', 'root', 'key', 'secret', 'token']
            for string in strings:
                string_lower = string.lower()
                for keyword in suspicious_keywords:
                    if keyword in string_lower:
                        strings_data['suspicious_strings'].append(string)
            
        except Exception as e:
            strings_data['error'] = str(e)
        
        return strings_data
    
    def _analyze_zip_header(self, file_path: Path) -> Dict[str, Any]:
        """Analyze ZIP file header structure."""
        return {'analysis': 'ZIP header analysis', 'note': 'Basic ZIP detection'}
    
    def _analyze_pdf_header(self, file_path: Path) -> Dict[str, Any]:
        """Analyze PDF file header structure."""
        return {'analysis': 'PDF header analysis', 'note': 'Basic PDF detection'}
    
    def _analyze_image_header(self, file_path: Path) -> Dict[str, Any]:
        """Analyze image file header structure."""
        return {'analysis': 'Image header analysis', 'note': 'Basic image detection'}
    
    def _analyze_pe_header(self, header_bytes: bytes) -> Dict[str, Any]:
        """Analyze PE executable header structure."""
        return {'analysis': 'PE header analysis', 'note': 'Basic PE detection'}
    
    def _extract_tar_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from TAR archives."""
        try:
            with tarfile.open(file_path, 'r') as tf:
                metadata = {
                    'file_count': len(tf.getmembers()),
                    'files': [],
                    'total_size': 0
                }
                
                for member in tf.getmembers()[:50]:  # Limit to first 50
                    file_info = {
                        'name': member.name,
                        'size': member.size,
                        'type': 'file' if member.isfile() else 'directory',
                        'mode': oct(member.mode),
                        'mtime': member.mtime
                    }
                    metadata['files'].append(file_info)
                    metadata['total_size'] += member.size
                
                return metadata
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_text_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from text files."""
        try:
            with open(file_path, 'rb') as f:
                sample = f.read(8192)  # Sample first 8KB
            
            # Try to decode with different encodings
            encoding = 'unknown'
            content = None
            
            for enc in ['utf-8', 'ascii', 'latin-1', 'cp1252']:
                try:
                    content = sample.decode(enc)
                    encoding = enc
                    break
                except UnicodeDecodeError:
                    continue
            
            if content:
                lines = content.split('\n')
                metadata = {
                    'encoding': encoding,
                    'line_count_sample': len(lines),
                    'avg_line_length': sum(len(line) for line in lines) / len(lines) if lines else 0,
                    'contains_binary': any(ord(c) < 32 and c not in '\t\n\r' for c in content[:1000])
                }
            else:
                metadata = {'encoding': 'binary', 'readable': False}
            
            return metadata
            
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_executable_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from executable files."""
        try:
            with open(file_path, 'rb') as f:
                header = f.read(64)
            
            metadata = {'file_type': 'executable'}
            
            if header.startswith(b'MZ'):
                metadata['format'] = 'PE'
                metadata['architecture'] = 'Windows'
            elif header.startswith(b'\x7FELF'):
                metadata['format'] = 'ELF'
                metadata['architecture'] = 'Unix/Linux'
            elif header.startswith(b'\xCA\xFE\xBA\xBE'):
                metadata['format'] = 'Mach-O'
                metadata['architecture'] = 'macOS'
            
            return metadata
            
        except Exception as e:
            return {'error': str(e)}


def main():
    """Example usage of MetadataExtractor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Forensic Metadata Extractor')
    parser.add_argument('file_path', help='Path to file for metadata extraction')
    parser.add_argument('--output', '-o', help='Output file for JSON results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    # Extract metadata
    extractor = MetadataExtractor()
    metadata = extractor.extract_metadata(args.file_path)
    
    # Output results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {args.output}")
    else:
        print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()